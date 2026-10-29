"""Source-role supported rigid initialization, followed by unchanged bounded fit.

This candidate performs no search over target IoU to choose the initial yaw.
It writes no models, inputs, approval manifests or production fit records.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import shape


ORIGINAL_POLICY = {
    "minimum_xy_scale": .80, "maximum_xy_scale": 1.25,
    "maximum_incremental_anisotropy": 1.15, "maximum_total_anisotropy": 1.55,
    "maximum_yaw_change_degrees": 10.0, "maximum_center_offset_m": 3.0,
    "maximum_center_offset_extent_fraction": .05,
    "minimum_z_scale_ratio": .80, "maximum_z_scale_ratio": 1.25,
    "minimum_footprint_iou": .70, "minimum_target_coverage": .80,
    "minimum_asset_coverage": .80, "desired_raster_cell_m": .10,
    "maximum_raster_dimension": 2048, "fitting_simplification_m": .15,
    "maximum_optimizer_evaluations": 650,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def proof(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def verify(item, root):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"}, "Exact file proof required")
    path = Path(item["path"])
    require(path.is_absolute() and not path.is_symlink(), "Evidence must be an explicit nonsymlink file")
    resolved = path.resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(), "Evidence escaped the explicitly selected root")
    require(proof(resolved) == item, "Evidence bytes changed: " + str(path))
    return resolved


def affine(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            and abs(float(np.linalg.det(matrix[:3, :3]))) > 1e-12,
            "Invalid finite invertible affine " + name)
    return matrix


def matrix_identity(matrix):
    matrix = affine(matrix, "original prior")
    encoded = json.dumps(matrix.tolist(), separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_supported_yaw_initialization(original_prior, building, cue_path, *,
                                        approved_cue_sha256, evidence_root):
    """Return a new prior from approved directed role/landmark correspondences.

    Source anchors use current observed/predicted WORLD XY, not screen axes.
    Model anchors use actual DEFAULT-SCENE glTF XYZ after node transforms.
    Two distinct, identified roles resolve the 180-degree axis ambiguity.
    With more anchors, a proper 2D Procrustes rotation is used, with no scale,
    translation, footprint objective, deformation or reflection optimization.
    The final yaw rotates the original prior about the current target centroid.

    A separate reviewed cue file and its explicit SHA are required. Coarse,
    pending or merely numeric-search cues cannot enter this initialization.
    File proofs bind source observation/region evidence and actual model bytes;
    the approval is an explicit human/agent correspondence judgement, not a
    claim that hashes can establish semantic identity automatically.
    """
    root = Path(evidence_root).resolve()
    cue_path = Path(cue_path).resolve()
    require(isinstance(approved_cue_sha256, str) and len(approved_cue_sha256) == 64
            and sha256_file(cue_path) == approved_cue_sha256, "Explicit approved cue SHA is required")
    cue_proof = proof(cue_path)
    verify(cue_proof, root)
    cue = json.loads(cue_path.read_text())
    require(cue.get("schema") == "citycode.source_role_orientation_approval.v1"
            and cue.get("status") == "approved_for_initialization_trial"
            and cue.get("building_id") == building.get("building_id"), "Cue is pending, unapproved or bound to another building")
    require(cue.get("source_basis") == "current_prediction_observation"
            and cue.get("selection_method") == "directed_source_role_correspondence"
            and cue.get("ground_truth_or_oracle_used") is False
            and cue.get("target_iou_used_to_choose_orientation") is False,
            "Source-only directed cues required; GT/oracle or target-IoU orientation selection is forbidden")
    require(cue.get("reviewer") and cue.get("coordinate_systems") == {
        "source": "UE world XY metres Z-up",
        "model": "default-scene glTF XYZ Y-up including node transforms"}, "Reviewer and explicit coordinate systems are required")
    building_path = verify(cue["current_building"], root)
    require(json.loads(building_path.read_text()) == building, "Cue uses different current structure")
    verify(cue["actual_model"], root)
    original = affine(original_prior, "original prior")
    require(cue.get("original_prior_sha256") == matrix_identity(original), "Cue belongs to another raw initial prior")
    require(np.allclose(original[:2, 1], 0, atol=1e-12, rtol=0)
            and np.allclose(original[2, [0, 2]], 0, atol=1e-12, rtol=0)
            and original[2, 1] > 0, "Prior must map glTF Y-up to world Z-up without tilt")
    target = shape(building["footprint_world"])
    require(target.geom_type in {"Polygon", "MultiPolygon"} and target.is_valid
            and math.isfinite(target.area) and target.area > 0, "Current footprint is invalid")
    center = np.array(target.centroid.coords[0], dtype=float)
    require(np.isfinite(center).all(), "Current building center is not finite")
    anchors = cue.get("anchors")
    require(isinstance(anchors, list) and 2 <= len(anchors) <= 32,
            "At least two reliable distinct role/landmark anchors are required")
    ids = [row.get("role_id") for row in anchors]
    require(all(isinstance(value, str) and value.strip() for value in ids)
            and len(set(ids)) == len(ids), "Role identities must be unique and directed")
    checked = [cue_proof, cue["current_building"], cue["actual_model"]]
    source_points, model_points = [], []
    for row in anchors:
        require(row.get("status") == "approved_correspondence"
                and row.get("source_evidence_kind") in {"qualified_predicted_region", "observed_region_world_mapping"}
                and row.get("model_evidence_kind") in {"actual_mesh_region", "actual_triangle_point"},
                "Unapproved, unsupported or non-source anchor")
        for key in ("source_evidence", "model_evidence"):
            require(isinstance(row.get(key), list) and row[key], "Anchor lacks actual " + key)
            for item in row[key]:
                verify(item, root)
                checked.append(item)
        source = np.asarray(row["source_anchor_world_xy_m"], dtype=float)
        model = np.asarray(row["model_anchor_asset_gltf_xyz"], dtype=float)
        require(source.shape == (2,) and model.shape == (3,)
                and np.isfinite(source).all() and np.isfinite(model).all(), "Nonfinite or invalid anchor coordinates")
        source_points.append(source)
        model_points.append((original @ np.r_[model, 1.])[:2])
    source_points, model_points = np.asarray(source_points), np.asarray(model_points)
    # Duplicate/zero XY baselines carry no directed yaw information. Vertical
    # differences alone are intentionally insufficient.
    for i, j in itertools.combinations(range(len(anchors)), 2):
        require(np.linalg.norm(source_points[i] - source_points[j]) > 1e-8
                and np.linalg.norm(model_points[i] - model_points[j]) > 1e-8,
                "Degenerate source/model XY role baseline")
    # A reflected noncollinear role arrangement cannot be repaired by yaw.
    for i, j, k in itertools.combinations(range(len(anchors)), 3):
        a, b = source_points[j] - source_points[i], source_points[k] - source_points[i]
        c, d = model_points[j] - model_points[i], model_points[k] - model_points[i]
        source_cross, model_cross = a[0] * b[1] - a[1] * b[0], c[0] * d[1] - c[1] * d[0]
        tolerance = 1e-10 * max(1., np.linalg.norm(a) * np.linalg.norm(b), np.linalg.norm(c) * np.linalg.norm(d))
        if abs(source_cross) > tolerance and abs(model_cross) > tolerance:
            require(source_cross * model_cross > 0, "Reflected role correspondence cannot be corrected by rigid yaw")
    a = model_points - model_points.mean(axis=0)
    b = source_points - source_points.mean(axis=0)
    cosine = float(np.sum(a * b))
    sine = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))
    require(math.hypot(cosine, sine) > 1e-12 * max(1., float(np.linalg.norm(a) * np.linalg.norm(b))),
            "Degenerate or cancelling orientation evidence")
    radians = math.atan2(sine, cosine)
    rotation = np.array([[math.cos(radians), -math.sin(radians)],
                         [math.sin(radians), math.cos(radians)]])
    delta = np.eye(4)
    delta[:2, :2] = rotation
    delta[:2, 3] = center - rotation @ center
    corrected = affine(delta @ original, "source-supported corrected prior")
    old_singular = np.linalg.svd(original[:3, :3], compute_uv=False)
    new_singular = np.linalg.svd(corrected[:3, :3], compute_uv=False)
    require(np.allclose(old_singular, new_singular, rtol=1e-12, atol=1e-10)
            and np.array_equal(corrected[2], original[2])
            and np.sign(np.linalg.det(corrected[:3, :3])) == np.sign(np.linalg.det(original[:3, :3])),
            "Rigid initialization changed scale, vertical mapping or handedness")
    world_from_building = affine(building["world_from_local_4x4"], "world_from_building_local")
    building_from_world = affine(building["local_from_world_4x4"], "building_local_from_world")
    require(np.allclose(world_from_building @ building_from_world, np.eye(4), rtol=1e-10, atol=1e-8),
            "Current building local/world matrices do not close")
    local = building_from_world @ corrected
    residuals = np.linalg.norm(a @ rotation.T - b, axis=1)
    for item in checked:
        verify(item, root)
    return {
        "schema": "citycode.source_supported_initial_orientation.v1",
        "status": "initialized_original_bounded_fit_and_semantic_review_pending",
        "building_id": building["building_id"], "initialization_version": "directed_source_roles_rigid_yaw_v1",
        "cue_approval": cue_proof, "source_evidence": checked,
        "original_prior_world_from_asset_4x4": original.tolist(),
        "source_supported_prior_world_from_asset_4x4": corrected.tolist(),
        "asset_from_world_4x4": np.linalg.inv(corrected).tolist(),
        "building_local_from_asset_4x4": local.tolist(),
        "world_from_building_local_4x4": world_from_building.tolist(),
        "normal_asset_to_world_3x3": np.linalg.inv(corrected[:3, :3]).T.tolist(),
        "rigid_world_correction_4x4": delta.tolist(),
        "initial_yaw_correction_degrees": math.degrees(radians),
        "rotation_center_world_xy_m": center.tolist(),
        "singular_values_before": old_singular.tolist(), "singular_values_after": new_singular.tolist(),
        "vertical_mapping_unchanged": True, "handedness_unchanged": True,
        "local_world_closure_error": float(np.max(np.abs(world_from_building @ local - corrected))),
        "role_ids": ids, "anchor_centered_residual_m": residuals.tolist(),
        "anchor_residual_is_release_gate": False,
        "interpretation": "Coarse role centers establish a new initialization; their residuals are descriptive, not exact physical landmark or release thresholds.",
        "target_iou_used_to_choose_orientation": False, "ground_truth_or_oracle_used": False,
        "original_prior_yaw_bound_satisfied_claimed": False,
        "model_geometry_uv_materials_written": False,
    }


def load_registration(path, expected_sha256):
    path = Path(path).resolve()
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and sha256_file(path) == expected_sha256, "Explicit original registration source SHA required")
    name = "_orientation_original_registration_" + expected_sha256
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(asdict(module.FitPolicy()) == ORIGINAL_POLICY, "The original FitPolicy must not be relaxed")
    return module


def fit_with_source_orientation(source_glb, building, original_prior, cue_path, *,
                                 approved_cue_sha256, evidence_root,
                                 registration_path, expected_registration_sha256):
    """Invoke the original fitter with +/-10 degrees around the NEW prior.

    The returned bounded_fit is preserved verbatim, including failure reasons.
    Neither numeric acceptance nor this wrapper grants semantic/public release.
    """
    initialization = source_supported_yaw_initialization(original_prior, building, cue_path,
        approved_cue_sha256=approved_cue_sha256, evidence_root=evidence_root)
    cue = json.loads(Path(cue_path).read_text())
    require(Path(source_glb).resolve() == Path(cue["actual_model"]["path"]).resolve(),
            "Fitted model is not the approved cue's actual model")
    registration = load_registration(registration_path, expected_registration_sha256)
    fitted = registration.fit_predicted_asset(source_glb, building,
        initialization["source_supported_prior_world_from_asset_4x4"],
        expected_source_sha256=cue["actual_model"]["sha256"], policy=registration.FitPolicy())
    require(fitted["policy"] == ORIGINAL_POLICY
            and np.array_equal(np.asarray(fitted["prior_world_from_asset_4x4"]),
                               np.asarray(initialization["source_supported_prior_world_from_asset_4x4"])),
            "Original fitter did not use the source-supported prior and unchanged policy")
    # No editing of status/reasons: an original optimizer/gate failure remains
    # a failure. Check that the original module and all inputs stayed immutable.
    require(sha256_file(registration_path) == expected_registration_sha256, "Registration source changed during fit")
    for item in initialization["source_evidence"]:
        verify(item, Path(evidence_root).resolve())
    return {"schema": "citycode.source_orientation_bounded_fit_trial.v1",
        "building_id": building["building_id"], "numeric_status": fitted["status"],
        "initialization": initialization, "bounded_fit": fitted,
        "registration_source": proof(registration_path), "mechanism_source": proof(__file__),
        "local_fit_yaw_bound_reference": "source_supported_prior_world_from_asset_4x4",
        "original_raw_prior_yaw_bound_satisfied_claimed": False,
        "both_lods_same_M_and_source_semantic_QA_required_before_release": True,
        "release_approved": False, "production_records_written": False}
