"""Prediction-derived placement of newly generated glTF buildings."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from .appearance import sha256_file
from .generation import normalize_geometry


NATIVE_TO_GLTF = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                           [0., -1., 0., 0.], [0., 0., 0., 1.]])


def _affine(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12):
        raise ValueError(f"Invalid finite invertible affine {name}")
    return matrix


def initial_generated_placement(building: dict, height_audit: dict) -> dict:
    """Invert the recorded control-to-model similarity without any old asset.

    XY comes from the exact two normalizations and fitted model-frame pose.
    The Z prior uses the requested occupied voxel height, including rounding
    and clipping, rather than the generated mesh's measured height. Subsequent
    bounded registration independently checks the actual model.
    """
    footprint = shape(building["footprint_world"])
    if footprint.geom_type not in ("Polygon", "MultiPolygon") or not footprint.is_valid or footprint.area <= 0:
        raise ValueError("Generated placement needs a valid current predicted footprint")
    base, top, height = (float(building[key]) for key in ("base_z_m", "top_z_m", "height_m"))
    if not all(math.isfinite(v) for v in (base, top, height)) or height <= 0 or not math.isclose(top - base, height, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("Prediction base, top and height are inconsistent")
    if (not math.isclose(float(height_audit["height_m"]), height, rel_tol=1e-8, abs_tol=1e-6)
            or not math.isclose(float(height_audit["footprint_area_m2"]), footprint.area, rel_tol=1e-8, abs_tol=1e-6)):
        raise ValueError("Height audit belongs to different predicted geometry")
    _, center, metric_scale = normalize_geometry(footprint)
    center = np.asarray(center, dtype=float)
    alignment = height_audit["model_frame_alignment"]
    source = np.asarray(alignment["prediction_normalization"], dtype=float)
    target = np.asarray(alignment["target_normalization"], dtype=float)
    translation = np.asarray(alignment["translation"], dtype=float)
    scale, angle = float(alignment["scale_after_area_normalization"]), float(alignment["rotation_deg"])
    if (source.shape != (3,) or target.shape != (3,) or translation.shape != (2,)
            or not all(np.isfinite(v).all() for v in (source, target, translation))
            or source[2] <= 0 or target[2] <= 0 or not math.isfinite(scale + angle) or scale <= 0):
        raise ValueError("Model-frame alignment is not a finite positive similarity")
    radians = math.radians(angle)
    rotation = np.array([[math.cos(radians), -math.sin(radians)],
                         [math.sin(radians), math.cos(radians)]])
    # world -> bbox-normalized control -> source normalized -> fitted target
    # normalized -> original native model XY. These origins are distinct.
    linear = (target[2] * scale * metric_scale / source[2]) * rotation
    offset = target[:2] + target[2] * translation - (target[2] * scale / source[2]) * rotation @ (metric_scale * center + source[:2])
    native_from_world = np.eye(3)
    native_from_world[:2, :2], native_from_world[:2, 2] = linear, offset
    world_from_native = np.linalg.inv(native_from_world)
    count = height_audit["requested_height_voxels"]
    low, high = height_audit["z_min"], height_audit["z_max"]
    if (any(isinstance(v, bool) or int(v) != v for v in (count, low, high))
            or not 1 <= count <= 32 or not 0 <= low <= high <= 31 or high - low + 1 != count):
        raise ValueError("Height audit does not describe a valid occupied voxel span")
    vertical_scale = height / (count / 32)
    world = np.eye(4)
    # o_voxel exports (native_x, native_z, -native_y) as glTF Y-up.
    world[:2, 0] = world_from_native[:2, 0]
    world[:2, 1] = 0
    world[:2, 2] = -world_from_native[:2, 1]
    world[:2, 3] = world_from_native[:2, 2]
    world[2] = [0, vertical_scale, 0, base - vertical_scale * (low / 32 - .5)]
    world = _affine(world, "world_from_asset")
    building_world = _affine(building["world_from_local_4x4"], "world_from_building_local")
    building_inverse = _affine(building["local_from_world_4x4"], "building_local_from_world")
    if not np.allclose(building_world @ building_inverse, np.eye(4), atol=1e-8, rtol=1e-9):
        raise ValueError("Building local/world transforms do not close")
    local = building_inverse @ world
    return {"schema": "citycode.generated_asset_initial_placement.v1", "building_id": building["building_id"],
            "status": "initialized_geometry_acceptance_pending", "local_coordinate_system": "glTF Y-up",
            "world_coordinate_system": "UE XYZ metres Z-up",
            "world_from_asset_4x4": world.tolist(), "asset_from_world_4x4": np.linalg.inv(world).tolist(),
            "building_local_from_asset_4x4": local.tolist(),
            "world_from_building_local_4x4": building_world.tolist(),
            "normal_asset_to_world_3x3": np.linalg.inv(world[:3, :3]).T.tolist(),
            "native_model_xy_from_world_3x3": native_from_world.tolist(),
            "native_xyz_to_gltf_y_up_4x4": NATIVE_TO_GLTF.tolist(),
            "height_control": {"requested_height_voxels": int(count), "z_min": int(low), "z_max": int(high),
                               "prior_metric_vertical_scale": vertical_scale,
                               "native_voxel_bounds": [low / 32 - .5, (high + 1) / 32 - .5]},
            "topology_fallback": alignment.get("topology_fallback", {"used": False}),
            "old_asset_or_reference_geometry_read": False, "actual_mesh_acceptance_performed": False}


def fit_generated_asset(source_glb: str | Path, building: dict, height_audit: dict, *,
                        expected_source_sha256=None, policy=None) -> dict:
    """Initialize from current prediction, then apply the standard bounded fit."""
    from .registration import fit_predicted_asset

    placement = initial_generated_placement(building, height_audit)
    fitted = fit_predicted_asset(source_glb, building, placement["world_from_asset_4x4"],
                                expected_source_sha256=expected_source_sha256, policy=policy)
    fitted["initial_generated_placement"] = placement
    return fitted


def load_placement_inputs(building_path: str | Path, generation_output: str | Path) -> tuple[dict, dict]:
    """Require a successful raw generation record that hashes the pose evidence."""
    building_path, output = Path(building_path), Path(generation_output).resolve()
    raw = json.loads((output / "raw_manifest.json").read_text())
    if raw.get("status") != "complete" or raw.get("contract", {}).get("building_sha256") != sha256_file(building_path):
        raise ValueError("Generation cache is not bound to this current building")
    raw_path = output / "raw_mesh.pt"
    if (not raw_path.is_file() or raw_path.stat().st_size != raw.get("bytes")
            or sha256_file(raw_path) != raw.get("sha256")):
        raise ValueError("Raw decoded generation bytes changed")
    inputs = raw.get("placement_inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"height_correction", "image_only_structure", "height_corrected_structure"}:
        raise ValueError("Generation record lacks hash-bound placement inputs")
    paths = {}
    for name, record in inputs.items():
        value = record["path"]
        path = (output / value).resolve()
        if Path(value).is_absolute() or ".." in Path(value).parts or not path.is_relative_to(output):
            raise ValueError("Placement evidence must remain inside its generation output")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Generation placement evidence changed: {name}")
        paths[name] = path
    return json.loads(building_path.read_text()), json.loads(paths["height_correction"].read_text())
