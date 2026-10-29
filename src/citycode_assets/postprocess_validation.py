"""Validate source-bound structural replacements and observed roof fragments.

These checks keep assembly readiness separate from detailed-model acceptance.
They never infer an unobserved complete building from a retained roof fragment.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np


_CONTRACTS = {
    "accepted_structure": ("source_structure", "structure"),
    "resolved_fragment": ("observed_fragment_surface", "observed_surface"),
    "accepted_repaired": ("local_mesh_repair", "repaired_detailed"),
}
_LODS = ("0.25M", "0.1M")
_UE_TO_GLTF = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                        [0., -1., 0., 0.], [0., 0., 0., 1.]])


def _proof(asset_root, value, label):
    """Read a declared source once, allowing the sibling structure package."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} needs a file proof")
    name = value.get("path")
    if value.get("path_base", "manifest_directory") != "manifest_directory":
        raise ValueError(f"{label} proof must be rebased to the asset manifest directory")
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()):
        raise ValueError(f"{label} needs a relative POSIX source path")
    root = Path(asset_root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root.parent) or not path.is_file():
        raise ValueError(f"{label} source escapes the data directory or is missing")
    digest, count = value.get("sha256"), value.get("bytes")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} needs a complete SHA256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{label} needs its actual byte count")
    payload = path.read_bytes()
    if len(payload) != count or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{label} source SHA256 or byte count changed")
    return path, payload


def _matrix(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite affine matrix") from exc
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or abs(float(np.linalg.det(result[:3, :3]))) <= 1e-12):
        raise ValueError(f"{label} must be a finite invertible affine matrix")
    return result


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _same_number(value, expected, label, tolerance=1e-8):
    if abs(_number(value, label) - expected) > tolerance:
        raise ValueError(f"{label} differs from the current source")


def _scene_triangles(scene):
    return np.asarray(scene["vertices"], dtype=float)[scene["faces"]]


def _upward_triangles(scene):
    triangles = _scene_triangles(scene)
    vectors = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(vectors, axis=1)
    # Supported heightfields may be steep. A vertical extrusion wall has zero
    # Y normal, whereas every source-grid top triangle has strictly positive Y.
    selected = (lengths > 1e-12) & (vectors[:, 1] > 1e-12)
    if not np.any(selected):
        raise ValueError("Source or output mesh has no upward roof triangles")
    return triangles[selected]


def _canonical_triangles(triangles):
    """Ignore vertex/face ordering, retaining every triangle occurrence."""
    triangles = np.asarray(triangles, dtype=float)
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1).reshape(-1, 9)
    return triangles[np.lexsort(tuple(triangles[:, i] for i in range(8, -1, -1)))]


def _same_triangles(actual, expected, label, position_tolerance_m=None):
    first, second = _canonical_triangles(actual), _canonical_triangles(expected)
    # GLB positions are float32. Permit only their rounding at the source scale.
    scale = max(1., float(np.abs(second).max()))
    tolerance = (float(max(1e-6, 2 * np.finfo(np.float32).eps * scale))
                 if position_tolerance_m is None else float(position_tolerance_m))
    if first.shape != second.shape or not np.allclose(first, second, atol=tolerance, rtol=0):
        raise ValueError(f"{label} triangle positions differ from the source proxy")
    return {"triangles": len(second), "position_tolerance_m": tolerance}


def _output_scene(asset_root, lod, label):
    from .registration import read_glb_scene_geometry

    if not isinstance(lod, dict):
        raise ValueError(f"{label} needs an actual LOD file proof")
    path, _ = _proof(asset_root, lod, label)
    # Output assets, unlike source evidence, must belong to this asset package.
    if not path.is_relative_to(Path(asset_root).resolve()):
        raise ValueError(f"{label} output escapes the asset package")
    return read_glb_scene_geometry(path)


def validate_postprocess_record(asset_root, record, geometry=False):
    """Validate one new postprocess state; raise ValueError on stale evidence.

    The caller retains its existing detailed GLB/LOD checks. In geometry mode,
    structural replacements also prove unchanged source triangle positions;
    fragments prove the source roof plus a visualization-only thin thickness.
    """
    from .registration import read_glb_scene_geometry

    status = record.get("status")
    if status not in _CONTRACTS:
        raise ValueError("Postprocess validation requires an explicit postprocess status")
    post = record.get("postprocess")
    if not isinstance(post, dict) or post.get("schema") != "citycode.asset_postprocess_record.v1":
        raise ValueError("Postprocess record schema is required")
    method, representation = _CONTRACTS[status]
    if (post.get("method"), post.get("representation")) != (method, representation):
        raise ValueError("Postprocess status, method and representation disagree")
    source = post.get("source")
    if not isinstance(source, dict):
        raise ValueError("Postprocess source evidence is required")
    building_path, payload = _proof(asset_root, source.get("building"), "building")
    proxy_path, _ = _proof(asset_root, source.get("proxy"), "proxy")
    evidence = source.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Postprocess needs observed source evidence")
    for index, proof in enumerate(evidence):
        _proof(asset_root, proof, f"source evidence {index}")
    building = json.loads(payload)
    sid = record.get("building_id")
    if building.get("building_id") != sid or not isinstance(sid, str) or not sid:
        raise ValueError("Postprocess building identity differs from the current source")
    footprint = building.get("footprint_world")
    if not isinstance(footprint, dict):
        raise ValueError("Current source footprint is required")
    if post.get("source_target_geometry") != footprint:
        raise ValueError("Postprocess must retain the exact current source footprint")
    if status != "resolved_fragment" and record.get("target_geometry") != footprint:
        raise ValueError("Postprocess must retain the exact current source footprint")
    base = _number(building.get("base_z_m"), "source base")
    top = _number(building.get("top_z_m"), "source top")
    if top <= base:
        raise ValueError("Current source building height must be positive")
    _same_number(post.get("source_base_z_m"), base, "postprocess source base")
    _same_number(post.get("source_top_z_m"), top, "postprocess source top")
    proxy_name = building.get("paths", {}).get("proxy_local_gltf_y_up")
    if (not isinstance(proxy_name, str) or not proxy_name
            or (building_path.parent / proxy_name).resolve() != proxy_path):
        raise ValueError("Postprocess proxy is not the current building's glTF source proxy")
    lods = record.get("lods")
    if not isinstance(lods, dict) or set(lods) != set(_LODS):
        raise ValueError("Postprocess requires both actual LOD variants")
    result = {"status": "passed", "method": method, "representation": representation,
              "source_building_sha256": source["building"]["sha256"],
              "source_proxy_sha256": source["proxy"]["sha256"],
              "actual_source_geometry_checked": bool(geometry),
              "detailed_semantic_acceptance_inferred": False}

    if status == "accepted_repaired":
        repair = post.get("repair")
        if not isinstance(repair, dict) or set(repair.get("original_lods", {})) != set(_LODS):
            raise ValueError("Repaired assets need both original LOD file proofs")
        for label in _LODS:
            _proof(asset_root, repair["original_lods"][label], f"original {label}")
        _, report_payload = _proof(asset_root, repair.get("report"), "repair report")
        report = json.loads(report_payload)
        if report.get("status") != "passed" or report.get("building_id") != sid:
            raise ValueError("Repair report must pass for the same building")
        expected = {label: lods[label].get("sha256") for label in _LODS}
        if report.get("output_lod_hashes") != expected:
            raise ValueError("Repair report does not bind the actual replacement LODs")
        _same_number(record.get("base_z_m"), base, "repaired base")
        _same_number(record.get("top_z_m"), top, "repaired top")
        result["repair_report_sha256"] = repair["report"]["sha256"]
        result["actual_source_geometry_checked"] = False
        return result

    translation = _matrix(building.get("world_from_local_4x4"), "source building local transform")
    if not np.allclose(translation[:3, :3], np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("Current source proxy requires its original UE translation frame")
    _same_number(translation[2, 3], base, "source local frame base")
    expected_world = translation @ np.linalg.inv(_UE_TO_GLTF)
    world = _matrix(record.get("world_from_local_4x4"), "postprocess world transform")
    if not np.allclose(world, expected_world, atol=1e-8, rtol=0):
        raise ValueError("Postprocess world transform must preserve the source glTF Y-up frame")
    if status == "accepted_structure":
        if post.get("detailed_asset_accepted") is not False:
            raise ValueError("A structural replacement cannot claim detailed asset acceptance")
        _same_number(record.get("base_z_m"), base, "structural base")
        _same_number(record.get("top_z_m"), top, "structural top")
        layered = source.get("layered_mesh")
        frame = source.get("geometry_source_frame")
        if layered is not None or frame is not None:
            if frame != "world_gltf_y_up":
                raise ValueError("Layered structural source must declare world_gltf_y_up")
            layered_path, _ = _proof(asset_root, layered, "layered structural mesh")
            if (building_path.name != "building.json" or building_path.parent.name != sid
                    or building_path.parents[1].name != "buildings"
                    or building_path.parents[2].name != "building_generation_inputs"):
                raise ValueError("Layered source requires the canonical structure package layout")
            expected_path = building_path.parents[3] / "release" / "canonical_building_glb" / (sid + ".glb")
            if layered_path != expected_path.resolve():
                raise ValueError("Layered structural mesh is not this building's canonical release file")
            from .lod import _read_glb
            _, document, _ = _read_glb(layered_path)
            if not document.get("meshes") or any(
                    mesh.get("extras", {}).get("predicted_global_building_id") != sid
                    for mesh in document["meshes"]):
                raise ValueError("Layered structural mesh contains a different source building identity")
            result.update(source_layered_mesh_sha256=layered["sha256"],
                          source_geometry_frame="world_gltf_y_up")
            if geometry:
                original = _scene_triangles(read_glb_scene_geometry(layered_path))
                # Release coordinates already contain the world translation.
                # Only undo their axis rotation; adding T again is incorrect.
                source_world = original @ np.linalg.inv(_UE_TO_GLTF)[:3, :3].T
                inverse = np.linalg.inv(world)
                recentered = source_world @ inverse[:3, :3].T + inverse[:3, 3]
                local_scale = max(1., float(np.abs(recentered).max()))
                tolerance = float(max(2e-6, 2 * np.finfo(np.float32).eps * local_scale))
                result["lods"] = {}
                for label in _LODS:
                    actual = _scene_triangles(_output_scene(asset_root, lods[label], label))
                    actual_world = actual @ world[:3, :3].T + world[:3, 3]
                    comparison = _same_triangles(actual_world, source_world, label, tolerance)
                    comparison["position_tolerance_basis"] = "float32 rounding after source recentering in local metres"
                    result["lods"][label] = comparison
        elif geometry:
            source_triangles = _scene_triangles(read_glb_scene_geometry(proxy_path))
            result["lods"] = {label: _same_triangles(
                _scene_triangles(_output_scene(asset_root, lods[label], label)),
                source_triangles, label) for label in _LODS}
        return result

    surface = post.get("surface")
    if (not isinstance(surface, dict) or post.get("complete_building") is not False
            or surface.get("thickness_is_visualization_only") is not True
            or post.get("detailed_asset_accepted") is not False):
        raise ValueError("A retained fragment must declare an incomplete visualization-only surface")
    thickness = _number(surface.get("thickness_m"), "surface thickness")
    if abs(thickness - .12) > 1e-10:
        raise ValueError("Observed roof surface thickness must be 0.12 metres")
    support = _validate_surface_support(asset_root, record, building, surface)
    source_roof = support["triangles_world"]
    low, high = float(source_roof[:, :, 2].min()), float(source_roof[:, :, 2].max())
    _same_number(record.get("base_z_m"), low - thickness, "fragment surface base", 2e-5)
    _same_number(record.get("top_z_m"), high, "fragment surface top", 2e-5)
    result.update(complete_building=False, surface_thickness_m=thickness,
                  thickness_is_visualization_only=True,
                  assembly_enabled=_assembly_enabled(surface),
                  support=support["validation"])
    if geometry:
        results = {}
        for label in _LODS:
            scene = _output_scene(asset_root, lods[label], label)
            local_triangles = _upward_triangles(scene)
            world_triangles = local_triangles @ world[:3, :3].T + world[:3, 3]
            comparison = _same_triangles(world_triangles, source_roof, label)
            all_triangles = _scene_triangles(scene)
            vectors = np.cross(all_triangles[:, 1] - all_triangles[:, 0],
                               all_triangles[:, 2] - all_triangles[:, 0])
            bottom = all_triangles[vectors[:, 1] < -1e-12]
            _same_triangles(bottom @ world[:3, :3].T + world[:3, 3],
                            source_roof - [0, 0, thickness], f"{label} bottom")
            vertices = scene["vertices"] @ world[:3, :3].T + world[:3, 3]
            tolerance = comparison["position_tolerance_m"]
            if (abs(float(vertices[:, 2].min()) - (low - thickness)) > tolerance
                    or abs(float(vertices[:, 2].max()) - high) > tolerance):
                raise ValueError(f"{label} fragment is not the thin source roof surface")
            # Every output position is either an original roof position or its
            # downward thickness copy; this excludes added unrelated geometry.
            allowed = np.concatenate([source_roof.reshape(-1, 3),
                                      source_roof.reshape(-1, 3) - [0, 0, thickness]])
            from scipy.spatial import cKDTree
            distances, _ = cKDTree(allowed).query(vertices, k=1)
            if np.any(distances > tolerance):
                raise ValueError(f"{label} fragment adds positions beyond its observed roof")
            results[label] = comparison
        result["lods"] = results
    return result


def _validate_surface_support(asset_root, record, building, surface):
    """Replay every supported vertex against its original qualified sample."""
    from shapely import covers, polygons, union_all
    from shapely.geometry import shape

    metadata_path, metadata_bytes = _proof(asset_root, surface.get("metadata"), "surface metadata")
    metadata = json.loads(metadata_bytes)
    if (metadata.get("schema") not in {"citycode.observed_fragment_surface.v1", "citycode.observed_fragment_surface.v2"}
            or metadata.get("building_id") != record["building_id"]
            or metadata.get("independent_building") is not False
            or metadata.get("status") != "supported_surface_candidate"):
        raise ValueError("Observed fragment needs the supported source surface metadata")
    support_path, support_bytes = _proof(asset_root, surface.get("support"), "surface support")
    expected_support_path = (Path(asset_root) / metadata.get("surface", {}).get("path", "")).resolve()
    if (metadata.get("surface", {}).get("path_base", "manifest_directory") != "manifest_directory"
            or expected_support_path != support_path
            or metadata.get("surface", {}).get("sha256") != surface["support"]["sha256"]
            or metadata.get("surface", {}).get("bytes") != surface["support"]["bytes"]):
        raise ValueError("Surface metadata does not bind the actual support NPZ")
    source_proof = record["postprocess"]["source"]["building"]
    declared_building = metadata.get("source_building", {})
    if (declared_building.get("path_base", "manifest_directory") != "manifest_directory"
            or (Path(asset_root) / declared_building.get("path", "")).resolve()
            != (Path(asset_root) / source_proof["path"]).resolve()
            or declared_building.get("sha256") != source_proof["sha256"]
            or declared_building.get("bytes") != source_proof["bytes"]):
        raise ValueError("Surface metadata refers to a different current building")
    fields = {"vertices_world_xyz_m", "faces", "vertex_rgb", "vertex_source_flat_index",
              "vertex_original_grid_uv", "vertex_source_instance_id", "vertex_source_provenance",
              "vertex_observation_index", "triangle_observation_index", "triangle_roof_layer_index"}
    if metadata["schema"] == "citycode.observed_fragment_surface.v2":
        fields.add("triangle_component_index")
    with np.load(io.BytesIO(support_bytes), allow_pickle=False) as loaded:
        if set(loaded.files) != fields:
            raise ValueError("Unsupported observed fragment support NPZ fields")
        arrays = {name: loaded[name] for name in fields}
    xyz, faces = arrays["vertices_world_xyz_m"], arrays["faces"]
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or len(faces) < (8 if metadata["schema"].endswith(".v2") else 16)
            or faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError("Invalid supported source surface vertices or triangles")
    count = len(xyz)
    if np.any(xyz[:, 2] < building["base_z_m"] + .12 - 1e-8):
        raise ValueError("Observed surface must retain only complete triangles above the source ground")
    sizes = {"vertex_rgb": (count, 3), "vertex_source_flat_index": (count,),
             "vertex_original_grid_uv": (count, 2), "vertex_source_instance_id": (count,),
             "vertex_source_provenance": (count,), "vertex_observation_index": (count,),
             "triangle_observation_index": (len(faces),), "triangle_roof_layer_index": (len(faces),)}
    if metadata["schema"] == "citycode.observed_fragment_surface.v2":
        sizes["triangle_component_index"] = (len(faces),)
    if any(arrays[name].shape != dimensions for name, dimensions in sizes.items()):
        raise ValueError("Source surface support arrays have inconsistent dimensions")
    if any(arrays[name].dtype.kind not in "iu" for name in sizes):
        raise ValueError("Source surface support indices and colors must be integers")
    if (metadata.get("vertex_count") != count or metadata.get("triangle_count") != len(faces)
            or len(np.unique(faces)) != count):
        raise ValueError("Surface metadata counts or used-vertex membership disagree")
    observations = metadata.get("source_observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Source surface observation proofs are required")
    policy = metadata.get("policy", {})
    stride = policy.get("grid_stride", 1)
    if isinstance(stride, bool) or stride not in (1, 2, 3):
        raise ValueError("Observed surface grid stride must be one, two or three")
    if metadata["schema"] == "citycode.observed_fragment_surface.v2" and stride not in (2, 3):
        raise ValueError("Coarse source surface v2 requires explicit grid stride two or three")
    if metadata["schema"] == "citycode.observed_fragment_surface.v1" and stride != 1:
        raise ValueError("Original source surface v1 cannot claim an unbound coarse grid")
    vobs, fobs = arrays["vertex_observation_index"], arrays["triangle_observation_index"]
    if vobs.min() < 0 or vobs.max() >= len(observations) or fobs.min() < 0 or fobs.max() >= len(observations):
        raise ValueError("Source surface observation indices are invalid")
    if not np.all(vobs[faces] == fobs[:, None]):
        raise ValueError("Source surface triangle mixes source observations")
    original_footprint = shape(building["footprint_world"])
    if not original_footprint.is_valid or original_footprint.area <= 0:
        raise ValueError("Source fragment needs a valid original footprint")
    building_path = (Path(asset_root) / source_proof["path"]).resolve()
    current_observations = {row["observation_id"]: row for row in building.get("observations", [])}
    triangle_geometry = polygons(xyz[faces, :2])
    if not np.asarray(covers(original_footprint, triangle_geometry)).all():
        raise ValueError("Supported source triangles leave the original footprint")
    maximum_grid_error = 0.
    for index, item in enumerate(observations):
        if not isinstance(item, dict):
            raise ValueError("Invalid source surface observation metadata")
        oid = item.get("observation_id")
        current = current_observations.get(oid)
        if current is None:
            raise ValueError("Surface observation is absent from the current building")
        observation_path, observation_bytes = _proof(asset_root, item.get("observation"), "surface observation")
        if observation_path != (building_path.parent / current["path"] / "observation.json").resolve():
            raise ValueError("Surface observation proof differs from the current building input")
        observation = json.loads(observation_bytes)
        if (observation.get("observation_id") != oid
                or observation.get("building_id") != building["building_id"]
                or observation.get("sample_id") != item.get("sample_id")
                or observation.get("source_instance_id") != item.get("source_instance_id")
                or observation.get("bev_instance_id") != item.get("bev_instance_id")
                or item.get("source_instance_id") != item.get("bev_instance_id")):
            raise ValueError("Surface observation sample or instance identity changed")
        layer_index = item.get("roof_layer_index")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError("Surface observation needs its original roof-layer index")
        if layer_index == -1:
            if oid != building.get("canonical_observation_id"):
                raise ValueError("An unlayered fragment must use its canonical observation")
            clip = original_footprint
        else:
            layers = building.get("roof_layers", [])
            if not 0 <= layer_index < len(layers) or oid not in layers[layer_index].get("observation_ids", []):
                raise ValueError("Surface alternate is not supported by an existing roof layer")
            clip = shape(layers[layer_index]["geometry"]).intersection(original_footprint)
        chosen_vertices, chosen_faces = np.flatnonzero(vobs == index), np.flatnonzero(fobs == index)
        if np.any(arrays["triangle_roof_layer_index"][chosen_faces] != layer_index):
            raise ValueError("Surface triangle changed its original roof-layer membership")
        if len(chosen_faces) and not np.asarray(covers(clip, triangle_geometry[chosen_faces])).all():
            raise ValueError("Surface triangles leave their declared observed roof layer")
        point_path, point_bytes = _proof(asset_root, item.get("source_points"), "surface source points")
        if point_path.name != item["sample_id"] + "__height_cleaned_building_points.npz":
            raise ValueError("Source point cache does not identify the observation sample")
        with np.load(io.BytesIO(point_bytes), allow_pickle=False) as loaded:
            points = {name: loaded[name] for name in ("xyz_m", "rgb", "instance_id", "provenance")}
        flat = arrays["vertex_source_flat_index"][chosen_vertices]
        if (len(flat) and (flat.min() < 0 or flat.max() >= len(points["xyz_m"]))
                or len(np.unique(flat)) != len(flat)):
            raise ValueError("Source surface sample indices are invalid or duplicated")
        expected_xyz = points["xyz_m"][flat].astype(np.float64)
        if (not np.array_equal(xyz[chosen_vertices], expected_xyz)
                or not np.array_equal(arrays["vertex_rgb"][chosen_vertices], points["rgb"][flat])
                or not np.all(points["instance_id"][flat] == item["bev_instance_id"])
                or not np.all(points["provenance"][flat] == 1)
                or not np.array_equal(arrays["vertex_source_instance_id"][chosen_vertices], points["instance_id"][flat])
                or not np.array_equal(arrays["vertex_source_provenance"][chosen_vertices], points["provenance"][flat])):
            raise ValueError("Surface vertex is not its original qualified source sample")
        matrix = np.asarray(observation.get("bev_crop_pixel_center_to_world_xy_3x3"), dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or abs(np.linalg.det(matrix)) <= 1e-12
                or not np.allclose(matrix, item.get("grid_pixel_center_to_world_xy_3x3"), atol=1e-12, rtol=0)):
            raise ValueError("Surface source grid transform changed")
        replay = np.column_stack([expected_xyz[:, :2], np.ones(len(expected_xyz))]) @ np.linalg.inv(matrix).T
        grid_error = np.abs(replay[:, :2] - arrays["vertex_original_grid_uv"][chosen_vertices])
        if len(grid_error) and float(grid_error.max()) > .025:
            raise ValueError("Surface samples do not replay to their original adjacent BEV grid")
        if len(grid_error):
            maximum_grid_error = max(maximum_grid_error, float(grid_error.max()))
        if stride > 1 and len(chosen_faces):
            # Coarsening may join only a complete original observed grid block.
            # Checking just its corner samples would bridge unsupported holes.
            from shapely import contains_xy
            direct = ((points["instance_id"] == item["bev_instance_id"])
                      & (points["provenance"] == 1) & np.isfinite(points["xyz_m"]).all(axis=1))
            direct_xyz = points["xyz_m"][direct].astype(np.float64)
            inside = contains_xy(clip, direct_xyz[:, 0], direct_xyz[:, 1])
            direct_xyz = direct_xyz[inside]
            grid_xy = (np.column_stack([direct_xyz[:, :2], np.ones(len(direct_xyz))])
                       @ np.linalg.inv(matrix).T)[:, :2]
            direct_uv = np.rint(grid_xy).astype(np.int64)
            valid = np.max(np.abs(grid_xy - direct_uv), axis=1) <= .025
            direct_uv, direct_xyz = direct_uv[valid], direct_xyz[valid]
            if not len(direct_uv):
                raise ValueError("Coarse observed surface has no qualified source neighborhood")
            origin = direct_uv.min(axis=0)
            span = int(direct_uv[:, 0].max() - origin[0] + 3)
            source_keys = (direct_uv[:, 1] - origin[1]) * span + direct_uv[:, 0] - origin[0]
            order = np.argsort(source_keys)
            sorted_keys = source_keys[order]
            if np.any(np.diff(sorted_keys) == 0):
                raise ValueError("Coarse source grid has duplicate qualified sample locations")
            starts = arrays["vertex_original_grid_uv"][faces[chosen_faces]].min(axis=1)
            offsets = np.array([[x, y] for y in range(stride + 1) for x in range(stride + 1)])
            neighborhoods = starts[:, None, :] + offsets
            query = ((neighborhoods[:, :, 1] - origin[1]) * span
                     + neighborhoods[:, :, 0] - origin[0])
            positions = np.searchsorted(sorted_keys, query)
            safe = np.minimum(positions, len(sorted_keys) - 1)
            found = ((positions < len(sorted_keys)) & (sorted_keys[safe] == query)
                     & (neighborhoods[:, :, 0] >= origin[0])
                     & (neighborhoods[:, :, 0] <= direct_uv[:, 0].max())
                     & (neighborhoods[:, :, 1] >= origin[1])
                     & (neighborhoods[:, :, 1] <= direct_uv[:, 1].max()))
            if not found.all():
                raise ValueError("Coarse surface bridges an unsupported original grid neighborhood")
            neighborhood_xyz = direct_xyz[order[safe]]
            if (np.any(np.ptp(neighborhood_xyz[:, :, 2], axis=1) > 1.5 + 1e-8)
                    or np.any(neighborhood_xyz[:, :, 2] < building["base_z_m"] + .12 - 1e-8)):
                raise ValueError("Coarse surface source neighborhood crosses a height or ground discontinuity")
    uv = arrays["vertex_original_grid_uv"][faces]
    lower = uv.min(axis=1)
    normalized = uv - lower[:, None, :]
    keys = np.sort(normalized[:, :, 0] + 2 * normalized[:, :, 1], axis=1)
    if (not np.all((normalized == 0) | (normalized == stride))
            or not np.all(np.all(keys == [0, stride, 3 * stride], axis=1)
                          | np.all(keys == [0, 2 * stride, 3 * stride], axis=1))):
        raise ValueError("Surface triangles must connect three original adjacent grid samples")
    triangles = xyz[faces]
    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    edges = np.stack([triangles[:, 1, :2] - triangles[:, 0, :2],
                      triangles[:, 2, :2] - triangles[:, 1, :2],
                      triangles[:, 0, :2] - triangles[:, 2, :2]], axis=1)
    if (np.any(vectors[:, 2] <= 0)
            or np.any(np.linalg.norm(edges, axis=2) > .22 * stride + 1e-8)
            or np.any(np.ptp(triangles[:, :, 2], axis=1) > 1.5 + 1e-8)):
        raise ValueError("Supported source triangle orientation, edge or height range is invalid")
    projected = union_all(triangle_geometry)
    if metadata["schema"] == "citycode.observed_fragment_surface.v2":
        _validate_surface_components(faces, triangle_geometry, arrays["triangle_component_index"])
    target = shape(record.get("target_geometry"))
    tolerance = max(1e-8, original_footprint.area * 1e-10)
    if not target.is_valid or target.area <= 0 or target.symmetric_difference(projected).area > tolerance:
        raise ValueError("Fragment target must equal the actual supported surface projection")
    coverage = projected.area / original_footprint.area
    declared_coverage = surface.get("visible_source_coverage", record["postprocess"].get("visible_source_coverage"))
    _same_number(declared_coverage, coverage, "visible source coverage", 1e-9)
    _same_number(metadata.get("source_footprint_planar_coverage"), coverage, "surface metadata coverage", 1e-9)
    return {"triangles_world": triangles,
            "validation": {"schema": "citycode.observed_fragment_support_validation.v1",
                           "status": "passed", "source_vertices_replayed": count,
                           "source_triangles_checked": len(faces), "allowed_provenance": [1],
                           "grid_stride": stride,
                           "maximum_grid_replay_error_pixels": maximum_grid_error,
                           "visible_source_coverage": coverage,
                           "metadata_sha256": surface["metadata"]["sha256"],
                           "support_sha256": surface["support"]["sha256"]}}


def _validate_surface_components(faces, triangle_geometry, declared):
    """The coarse surface retains supported patches, not tiny isolated shards."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from shapely import union_all

    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    ordered = edges[order]
    repeated = np.all(ordered[1:] == ordered[:-1], axis=1)
    if np.any(repeated[1:] & repeated[:-1]):
        raise ValueError("Supported surface has a nonmanifold or duplicate original grid edge")
    owner = owners[order]
    left, right = owner[:-1][repeated], owner[1:][repeated]
    graph = coo_matrix((np.ones(2 * len(left)),
                        (np.concatenate([left, right]), np.concatenate([right, left]))),
                       shape=(len(faces), len(faces)))
    count, labels = connected_components(graph, directed=False)
    if np.any(declared < 0) or len(np.unique(declared)) != count:
        raise ValueError("Surface component metadata does not match its actual shared-edge connectivity")
    for label in range(count):
        mask = labels == label
        if len(np.unique(declared[mask])) != 1:
            raise ValueError("Surface component metadata splits one connected patch")
        if mask.sum() < 8 or union_all(triangle_geometry[mask]).area < .1 - 1e-9:
            raise ValueError("Coarse observed surface contains an unsupported tiny component")


def validate_evidence_only_record(asset_root, record):
    """Verify a retained source fragment without approving a renderable asset."""
    if record.get("status") not in {"excluded_incomplete", "excluded_noise"} or record.get("lods"):
        raise ValueError("Evidence-only postprocessing cannot carry accepted model LODs")
    post = record.get("postprocess")
    if (not isinstance(post, dict) or post.get("schema") != "citycode.asset_postprocess_record.v1"
            or post.get("method") != "retain_fragment_evidence"
            or post.get("representation") != "evidence_only"
            or post.get("detailed_asset_accepted") is not False):
        raise ValueError("Evidence-only record requires its explicit source-retention method")
    source = post.get("source")
    if not isinstance(source, dict):
        raise ValueError("Evidence-only record needs current source proofs")
    building_path, payload = _proof(asset_root, source.get("building"), "evidence-only building")
    proxy_path, _ = _proof(asset_root, source.get("proxy"), "evidence-only proxy")
    evidence = source.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Evidence-only record needs observed source evidence")
    for index, proof in enumerate(evidence):
        _proof(asset_root, proof, f"retained evidence {index}")
    building = json.loads(payload)
    if building.get("building_id") != record.get("building_id") or not record.get("building_id"):
        raise ValueError("Evidence-only identity differs from the current source")
    footprint = building.get("footprint_world")
    if (not isinstance(footprint, dict) or post.get("source_target_geometry") != footprint
            or record.get("target_geometry") != footprint):
        raise ValueError("Evidence-only record must preserve the exact current source footprint")
    base, top = _number(building.get("base_z_m"), "source base"), _number(building.get("top_z_m"), "source top")
    if top <= base:
        raise ValueError("Evidence-only source height must be positive")
    _same_number(post.get("source_base_z_m"), base, "retained source base")
    _same_number(post.get("source_top_z_m"), top, "retained source top")
    proxy_name = building.get("paths", {}).get("proxy_local_gltf_y_up")
    if (not isinstance(proxy_name, str) or not proxy_name
            or (building_path.parent / proxy_name).resolve() != proxy_path):
        raise ValueError("Retained proxy differs from the current building input")
    return {"status": "passed", "method": "retain_fragment_evidence", "representation": "evidence_only",
            "source_building_sha256": source["building"]["sha256"],
            "source_proxy_sha256": source["proxy"]["sha256"],
            "source_file_proofs_checked": len(evidence) + 2,
            "renderable_asset_approved": False, "complete_building_approved": False}


def _assembly_enabled(value):
    enabled = value.get("assembly_enabled", True)
    if type(enabled) is not bool:
        raise ValueError("Surface assembly_enabled must be a boolean")
    return enabled


def validate_surface_instances(manifest, rows):
    """Require one independent surface instance per retained source fragment."""
    surfaces = manifest.get("surface_instances", [])
    if not isinstance(surfaces, list):
        raise ValueError("Surface instances must be a list")
    by_id = {row.get("building_id"): row for row in rows}
    fragments = {sid: row for sid, row in by_id.items() if row.get("status") == "resolved_fragment"}
    expected, owners = {}, {}
    for sid, row in fragments.items():
        identity = row.get("surface_instance_id")
        if not isinstance(identity, str) or not identity or identity in expected:
            raise ValueError("Each fragment needs its own unique surface_instance_id")
        expected[identity] = row
        _assembly_enabled(row.get("postprocess", {}).get("surface", {}))
        parent = row.get("postprocess", {}).get("surface", {}).get("parent_building_id")
        if parent is not None and (parent not in by_id or parent == sid):
            raise ValueError("Fragment parent must be another existing source building")
    seen = set()
    for surface in surfaces:
        identity = surface.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Surface instance IDs must be nonempty and unique")
        seen.add(identity)
        row = expected.get(identity)
        if row is None or surface.get("status") != "resolved_fragment":
            raise ValueError("Surface instance has no matching resolved source fragment")
        sid = row["building_id"]
        sources = surface.get("source_building_ids")
        if sources != [sid] or sid in owners:
            raise ValueError("Surface sources must contain its own fragment exactly once")
        owners[sid] = identity
        world = _matrix(surface.get("world_from_local_4x4"), "surface world transform")
        if (surface.get("lods") != row.get("lods")
                or not np.array_equal(world, np.asarray(row.get("world_from_local_4x4")))):
            raise ValueError("Surface instance LODs or placement differ from its source row")
        if surface.get("parent_building_id") != row.get("postprocess", {}).get("surface", {}).get("parent_building_id"):
            raise ValueError("Surface parent association differs from its source row")
        if _assembly_enabled(surface) != _assembly_enabled(row.get("postprocess", {}).get("surface", {})):
            raise ValueError("Surface assembly_enabled differs from its source row")
    if seen != set(expected):
        raise ValueError("Surface table must contain every resolved fragment exactly once")
    return len(surfaces)
