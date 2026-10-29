"""CPU registration of immutable glTF assets to predicted metric structures.

This module reads no ground truth. Scene-node transforms are evaluated before
measuring the actual triangle projection. Assets are never baked or rewritten:
the returned matrices compose with the original glTF scene, preserving normals,
UVs and materials. Raster footprint approximation and all bounds are reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from shapely import affinity, make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union


@dataclass(frozen=True)
class FitPolicy:
    """Bounds apply to changes from the prior metric asset registration."""

    minimum_xy_scale: float = 0.80
    maximum_xy_scale: float = 1.25
    maximum_incremental_anisotropy: float = 1.15
    maximum_total_anisotropy: float = 1.55
    maximum_yaw_change_degrees: float = 10.0
    maximum_center_offset_m: float = 3.0
    maximum_center_offset_extent_fraction: float = 0.05
    minimum_z_scale_ratio: float = 0.80
    maximum_z_scale_ratio: float = 1.25
    minimum_footprint_iou: float = 0.70
    minimum_target_coverage: float = 0.80
    minimum_asset_coverage: float = 0.80
    desired_raster_cell_m: float = 0.10
    maximum_raster_dimension: int = 2048
    fitting_simplification_m: float = 0.15
    maximum_optimizer_evaluations: int = 650


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _polygonal(geometry):
    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    polygons = []
    for child in getattr(geometry, "geoms", []):
        child = _polygonal(child)
        if not child.is_empty:
            polygons.extend(child.geoms if isinstance(child, MultiPolygon) else [child])
    return unary_union(polygons) if polygons else Polygon()


def _parts(geometry):
    return list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]


def geometry_summary(geometry) -> dict[str, Any]:
    parts = _parts(geometry)
    return {
        "area_m2": float(geometry.area),
        "component_count": len(parts),
        "hole_count": sum(len(p.interiors) for p in parts),
        "bounds": list(geometry.bounds),
    }


def _affine2(geometry, matrix, translation=(0.0, 0.0)):
    return affinity.affine_transform(geometry, [
        float(matrix[0, 0]), float(matrix[0, 1]),
        float(matrix[1, 0]), float(matrix[1, 1]),
        float(translation[0]), float(translation[1]),
    ])


def overlap_metrics(asset, target) -> dict[str, float]:
    intersection = float(asset.intersection(target).area)
    union = float(asset.area + target.area - intersection)
    return {
        "iou": intersection / union if union else 0.0,
        "target_coverage": intersection / target.area if target.area else 0.0,
        "asset_coverage": intersection / asset.area if asset.area else 0.0,
        "intersection_m2": intersection,
        "outside_target_m2": float(asset.area - intersection),
        "uncovered_target_m2": float(target.area - intersection),
    }


def _node_matrix(node):
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            raise ValueError("glTF node declares both matrix and TRS")
        matrix = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    else:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(node.get("rotation", [0, 0, 0, 1])).as_matrix()
        matrix[:3, :3] @= np.diag(node.get("scale", [1, 1, 1]))
        matrix[:3, 3] = node.get("translation", [0, 0, 0])
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("nonfinite or non-affine glTF node transform")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("singular glTF node transform")
    return matrix


def read_glb_scene_geometry(path: str | Path) -> dict[str, Any]:
    """Read triangles and original NORMAL attributes without texture decoding.

    Evaluates the selected glTF scene including instanced meshes and nested TRS.
    Unsupported deformations/compression fail explicitly instead of silently
    fitting untransformed accessors. Unreferenced meshes are not included.
    """
    payload = Path(path).read_bytes()
    if len(payload) < 20 or struct.unpack_from("<4sII", payload) != (b"glTF", 2, len(payload)):
        raise ValueError("invalid GLB v2 header or declared length")
    document = None
    binary = None
    offset = 12
    while offset < len(payload):
        length, kind = struct.unpack_from("<II", payload, offset)
        offset += 8
        if offset + length > len(payload):
            raise ValueError("GLB chunk exceeds declared length")
        if kind == 0x4E4F534A:
            document = json.loads(payload[offset:offset + length])
        elif kind == 0x004E4942:
            binary = memoryview(payload)[offset:offset + length]
        offset += length
    if document is None or binary is None:
        raise ValueError("GLB requires JSON and embedded binary chunks")
    if document.get("animations") or document.get("skins"):
        raise ValueError("animated/skinned assets require an explicit static pose")
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or buffers[0].get("uri"):
        raise ValueError("only self-contained single-buffer GLB is supported")
    # Texture/material extensions do not change geometry; the immutable asset
    # retains them for its renderer. Geometry compression stays unsupported.
    unsupported = set(document.get("extensionsRequired", [])) - {
        "KHR_materials_unlit", "EXT_texture_webp", "KHR_texture_basisu",
        "KHR_texture_transform", "KHR_materials_pbrSpecularGlossiness",
    }
    if unsupported:
        raise ValueError(f"unsupported required glTF extensions: {sorted(unsupported)}")
    component_types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def accessor(index):
        spec = document["accessors"][index]
        if spec.get("sparse"):
            raise ValueError("sparse geometry accessors are not supported")
        if spec.get("normalized"):
            raise ValueError("normalized geometry accessors are not supported")
        view = document["bufferViews"][spec["bufferView"]]
        if view.get("buffer", 0) != 0 or view.get("extensions"):
            raise ValueError("external/compressed geometry buffer view")
        dtype = np.dtype(component_types[spec["componentType"]])
        width, count = widths[spec["type"]], int(spec["count"])
        stride = int(view.get("byteStride", width * dtype.itemsize))
        relative = int(spec.get("byteOffset", 0))
        end = relative + max(0, count - 1) * stride + width * dtype.itemsize
        if count <= 0 or stride < width * dtype.itemsize or end > view["byteLength"]:
            raise ValueError("invalid accessor bounds/stride")
        start = int(view.get("byteOffset", 0)) + relative
        if start + (count - 1) * stride + width * dtype.itemsize > len(binary):
            raise ValueError("geometry accessor exceeds GLB binary")
        return np.ndarray((count, width), dtype=dtype, buffer=binary,
                          offset=start, strides=(stride, dtype.itemsize)).copy()

    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not scenes:
        raise ValueError("GLB has no explicit scene")
    roots = scenes[int(document.get("scene", 0))].get("nodes", [])
    vertices, faces, normals, instances = [], [], [], []
    vertex_offset = 0

    def visit(index, parent, ancestors):
        nonlocal vertex_offset
        if index in ancestors:
            raise ValueError("cycle in glTF node graph")
        node = nodes[index]
        if "skin" in node or node.get("weights"):
            raise ValueError("deformed node requires explicit static pose")
        matrix = parent @ _node_matrix(node)
        if "mesh" in node:
            mesh = document["meshes"][node["mesh"]]
            for primitive_index, primitive in enumerate(mesh["primitives"]):
                if primitive.get("mode", 4) != 4 or primitive.get("targets") or primitive.get("extensions"):
                    raise ValueError("only uncompressed static triangle primitives are supported")
                attrs = primitive["attributes"]
                positions = accessor(attrs["POSITION"]).astype(np.float64)
                if positions.shape[1] != 3 or not np.isfinite(positions).all():
                    raise ValueError("invalid POSITION values")
                indices = (accessor(primitive["indices"]).reshape(-1).astype(np.int64)
                           if "indices" in primitive else np.arange(len(positions)))
                if len(indices) % 3 or indices.min() < 0 or indices.max() >= len(positions):
                    raise ValueError("invalid triangle indices")
                used = np.unique(indices)
                if len(used) != len(positions):
                    remap = np.full(len(positions), -1, dtype=np.int64)
                    remap[used] = np.arange(len(used))
                    indices = remap[indices]
                positions = positions[used]
                positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
                transformed_normals = None
                if "NORMAL" in attrs:
                    normal_values = accessor(attrs["NORMAL"]).astype(np.float64)[used]
                    if normal_values.shape != positions.shape or not np.isfinite(normal_values).all():
                        raise ValueError("invalid NORMAL values")
                    transformed_normals = normal_values @ np.linalg.inv(matrix[:3, :3])
                    lengths = np.linalg.norm(transformed_normals, axis=1)
                    if np.any(lengths < 1e-10):
                        raise ValueError("zero-length NORMAL attribute")
                    transformed_normals /= lengths[:, None]
                vertices.append(positions)
                faces.append(indices.reshape(-1, 3) + vertex_offset)
                normals.append(transformed_normals)
                instances.append({
                    "node_index": index, "mesh_index": node["mesh"], "primitive_index": primitive_index,
                    "asset_from_node_4x4": matrix.tolist(),
                    "normal_attribute_present": transformed_normals is not None,
                    "texcoord_0_present": "TEXCOORD_0" in attrs,
                    "material_index": primitive.get("material"),
                    "vertex_count": len(positions), "triangle_count": len(indices) // 3,
                    "node_determinant": float(np.linalg.det(matrix[:3, :3])),
                })
                vertex_offset += len(positions)
        for child in node.get("children", []):
            visit(child, matrix, ancestors | {index})

    for index in roots:
        visit(index, np.eye(4), set())
    if not vertices:
        raise ValueError("selected GLB scene has no triangle geometry")
    return {
        "vertices": np.concatenate(vertices), "faces": np.concatenate(faces),
        "normals": normals, "instances": instances,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "all_primitives_have_normals": all(n is not None for n in normals),
    }


def raster_triangle_footprint(vertices, faces, matrix, *, cell_m=0.10, max_dimension=2048):
    """Project every actual mesh triangle, retaining holes and all components.

    Raster union is explicit and deterministic. Triangle filling is performed
    separately (batched fillPoly would use even/odd overlap and create holes).
    No convex hull, largest-component selection or hole filling is used.
    """
    xy = np.asarray(vertices) @ np.asarray(matrix)[:2, :3].T + np.asarray(matrix)[:2, 3]
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    extent = upper - lower
    if extent.min() <= 1e-8:
        raise ValueError("mesh has degenerate horizontal extent")
    cell = max(float(cell_m), float(extent.max()) / (max_dimension - 8))
    origin = lower - 3.0 * cell
    dimensions = np.ceil(extent / cell).astype(int) + 7
    mask = np.zeros((int(dimensions[1]), int(dimensions[0])), dtype=np.uint8)
    # Fixed-point coordinates retain 1/16-pixel positions before rasterization.
    pixels = np.rint((xy - origin) / cell * 16).astype(np.int32)
    triangles = pixels[np.asarray(faces)]
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 1, lineType=cv2.LINE_8, shift=4)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for index, contour in enumerate(contours):
            if hierarchy[index, 3] >= 0 or len(contour) < 3:
                continue
            outer = contour[:, 0].astype(float) * cell + origin
            holes, child = [], int(hierarchy[index, 2])
            while child >= 0:
                if len(contours[child]) >= 3:
                    holes.append(contours[child][:, 0].astype(float) * cell + origin)
                child = int(hierarchy[child, 0])
            # A contour can touch itself at a raster corner. Repair each
            # contour before union so GEOS never unions an invalid polygon.
            polygons.append(_polygonal(Polygon(outer, holes)))
    footprint = _polygonal(unary_union(polygons))
    if footprint.is_empty or footprint.area <= 1e-8:
        raise ValueError("triangle projection is empty")
    return footprint, {
        "method": "all_actual_triangles_raster_union_contour",
        "cell_size_m": cell, "maximum_boundary_quantization_m": math.sqrt(2) * cell,
        "raster_width": int(dimensions[0]), "raster_height": int(dimensions[1]),
        "triangles_projected": int(len(faces)),
        "no_hull_or_component_removal": True,
    }


def _delta_matrix(parameters):
    angle, sx, sy, dx, dy = (float(v) for v in parameters)
    radians = math.radians(angle)
    rotation = np.array([[math.cos(radians), -math.sin(radians)],
                         [math.sin(radians), math.cos(radians)]])
    return rotation @ np.diag([sx, sy]), np.array([dx, dy])


def _parameter_violation(parameters, policy, horizontal_prior, component_priors, offset_limit):
    """Check every control bound for optimizer and finite fallback candidates."""
    values = np.asarray(parameters, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all() or min(values[1:3]) <= 0:
        return math.inf
    angle, sx, sy, dx, dy = values
    delta, offset = _delta_matrix(values)
    violation = max(0.0, abs(angle) - policy.maximum_yaw_change_degrees)
    for scale in (sx, sy):
        violation += max(0.0, policy.minimum_xy_scale - scale, scale - policy.maximum_xy_scale)
    violation += max(0.0, max(sx / sy, sy / sx) - policy.maximum_incremental_anisotropy)
    for horizontal in [horizontal_prior, *component_priors]:
        singular = np.linalg.svd(delta @ horizontal, compute_uv=False)
        if singular[-1] <= 0:
            return math.inf
        violation += max(0.0, singular[0] / singular[-1] - policy.maximum_total_anisotropy)
    violation += max(0.0, float(np.linalg.norm(offset)) - offset_limit)
    return float(violation)


def _overlap_passes(metrics, policy):
    return all(metrics[name] + 1e-9 >= threshold for name, threshold in (
        ("iou", policy.minimum_footprint_iou),
        ("target_coverage", policy.minimum_target_coverage),
        ("asset_coverage", policy.minimum_asset_coverage)))


def _registration_score(parameters, metrics, policy):
    regularization = .01 * sum(abs(math.log(float(scale))) for scale in parameters[1:3])
    regularization += .002 * (parameters[0] / max(policy.maximum_yaw_change_degrees, 1e-6)) ** 2
    return float(1.0 - metrics["iou"] + regularization)


def _rank_feasible_candidates(source, target, best, evaluated, policy,
                              horizontal_prior, component_priors, offset_limit):
    """Cheap, finite feasibility-first search over an already projected mesh.

    Keep the optimizer's evaluated controls and add a fixed local grid around
    its result. No triangle, component or hole is removed. Full-policy feasible
    polygons are ranked with the same IoU/regularization score as the optimizer;
    the caller must still reproject actual triangles before accepting a result.
    """
    best = np.asarray(best, dtype=float)
    proposed = list(evaluated)
    original_delta, original_offset = _delta_matrix(best)
    # 13 fixed scale levels; uniform and either source axis, with at most nine
    # small XY offsets. This is a bounded fallback, not an optimizer restart.
    factors = [(.90 + .02 * index) for index in range(13)]
    scale_pairs = sorted({(factor, factor) for factor in factors}
                         | {(factor, 1.0) for factor in factors}
                         | {(1.0, factor) for factor in factors})
    step = .1 * offset_limit
    for factor_x, factor_y in scale_pairs:
        candidate = best.copy()
        candidate[1:3] *= [factor_x, factor_y]
        delta, _ = _delta_matrix(candidate)
        # Scale about the target centroid, including the existing offset.
        scaled_offset = delta @ np.linalg.solve(original_delta, original_offset)
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                value = candidate.copy(); value[3:5] = scaled_offset + [dx, dy]
                proposed.append(value)
    unique = {}; control_rejected = 0; ranked = []
    for parameters in proposed:
        value = np.asarray(parameters, dtype=float)
        key = tuple(np.round(value, 12))
        if key in unique:
            continue
        unique[key] = True
        if _parameter_violation(value, policy, horizontal_prior, component_priors, offset_limit) > 0:
            control_rejected += 1
            continue
        delta, offset = _delta_matrix(value)
        metrics = overlap_metrics(_affine2(source, delta, offset), target)
        if _overlap_passes(metrics, policy):
            ranked.append({"parameters": value.tolist(), "projected_polygon_overlap": metrics,
                           "score": _registration_score(value, metrics, policy)})
    ranked.sort(key=lambda row: (row["score"], row["parameters"]))
    return ranked, {"unique_candidates": len(unique), "control_bound_rejected": control_rejected,
                    "full_polygon_feasible_candidates": len(ranked),
                    "fixed_scale_levels": factors, "translation_step_m": step}


def fit_predicted_asset(
    source_glb: str | Path,
    building: dict[str, Any],
    prior_world_from_asset,
    *,
    expected_source_sha256: str | None = None,
    policy: FitPolicy | None = None,
    component_prior_world_from_asset: list[Any] | None = None,
) -> dict[str, Any]:
    """Fit a static source asset to one predicted building, with bounded edits.

    The prior supplies only asset metric scale/orientation. No old target polygon
    enters the objective. Acceptance is recomputed from actual mesh triangles at
    the final transform against the complete new predicted polygon. Optional
    component transforms constrain a composite's shared XY change so every
    constituent also remains within the same total-anisotropy bound.
    """
    policy = policy or FitPolicy()
    target = _polygonal(shape(building["footprint_world"]))
    if target.is_empty or target.area <= 1e-6:
        raise ValueError("empty predicted target footprint")
    base, top = float(building["base_z_m"]), float(building["top_z_m"])
    if not math.isfinite(base + top) or top <= base:
        raise ValueError("invalid predicted height")
    prior = np.asarray(prior_world_from_asset, dtype=np.float64)
    if prior.shape != (4, 4) or not np.isfinite(prior).all() or not np.allclose(prior[3], [0, 0, 0, 1]):
        raise ValueError("invalid prior affine matrix")
    if not np.allclose(prior[:2, 1], 0) or not np.allclose(prior[2, [0, 2]], 0) or prior[2, 1] <= 0:
        raise ValueError("prior must map glTF Y-up to metric world Z-up without tilt")
    scene = read_glb_scene_geometry(source_glb)
    if expected_source_sha256 and scene["source_sha256"] != expected_source_sha256:
        raise ValueError("source GLB SHA256 does not match selected asset")
    vertices, faces = scene["vertices"], scene["faces"]
    heights = vertices[:, 1]
    low, high = float(heights.min()), float(heights.max())
    if high - low < 1e-8:
        raise ValueError("degenerate actual mesh height")
    z_scale = (top - base) / (high - low)
    z_ratio = z_scale / prior[2, 1]
    # Work near the origin for stable polygon calculations; prior translation
    # is not imposed on the new target and is retained only for provenance.
    prior_centered = prior.copy()
    prior_centered[:2, 3] = 0
    source, raster_profile = raster_triangle_footprint(
        vertices, faces, prior_centered,
        cell_m=policy.desired_raster_cell_m,
        max_dimension=policy.maximum_raster_dimension,
    )
    source_center = np.array(source.centroid.coords[0])
    target_center = np.array(target.centroid.coords[0])
    centered_source = affinity.translate(source, xoff=-source_center[0], yoff=-source_center[1])
    centered_target = affinity.translate(target, xoff=-target_center[0], yoff=-target_center[1])
    source_fit = centered_source.simplify(policy.fitting_simplification_m, preserve_topology=True)
    target_fit = centered_target.simplify(policy.fitting_simplification_m, preserve_topology=True)
    target_extent = max(target.bounds[2] - target.bounds[0], target.bounds[3] - target.bounds[1])
    offset_limit = min(policy.maximum_center_offset_m,
                       policy.maximum_center_offset_extent_fraction * target_extent)
    horizontal_prior = prior[:2][:, [0, 2]] @ np.diag([1.0, -1.0])
    component_horizontal_priors = []
    for component in component_prior_world_from_asset or []:
        component = np.asarray(component, dtype=np.float64)
        if (component.shape != (4, 4) or not np.isfinite(component).all()
                or not np.allclose(component[3], [0, 0, 0, 1])
                or not np.allclose(component[:2, 1], 0)
                or not np.allclose(component[2, [0, 2]], 0)
                or component[2, 1] <= 0):
            raise ValueError("component prior must be an affine glTF Y-up to metric Z-up transform without tilt")
        horizontal = component[:2][:, [0, 2]] @ np.diag([1.0, -1.0])
        if abs(np.linalg.det(horizontal)) < 1e-12:
            raise ValueError("component prior has singular horizontal scale")
        component_horizontal_priors.append(horizontal)
    bounds = [(-policy.maximum_yaw_change_degrees, policy.maximum_yaw_change_degrees),
              (policy.minimum_xy_scale, policy.maximum_xy_scale),
              (policy.minimum_xy_scale, policy.maximum_xy_scale),
              (-offset_limit, offset_limit), (-offset_limit, offset_limit)]
    evaluations = 0
    evaluated_parameters = []

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        evaluated_parameters.append(np.asarray(parameters, dtype=float).copy())
        delta, offset = _delta_matrix(parameters)
        violation = _parameter_violation(parameters, policy, horizontal_prior,
                                         component_horizontal_priors, offset_limit)
        if violation > 0:
            return 2.0 + violation
        transformed = _affine2(source_fit, delta, offset)
        return _registration_score(parameters, overlap_metrics(transformed, target_fit), policy)

    area_scale = float(np.clip(math.sqrt(target.area / source.area),
                               policy.minimum_xy_scale, policy.maximum_xy_scale))
    starts = [[angle, scale, scale, 0, 0]
              for angle in (-policy.maximum_yaw_change_degrees, -5.0, 0.0, 5.0, policy.maximum_yaw_change_degrees)
              if abs(angle) <= policy.maximum_yaw_change_degrees
              for scale in (1.0, area_scale)]
    best = min(starts, key=objective)
    result = minimize(objective, best, method="Powell", bounds=bounds,
                      options={"maxfev": policy.maximum_optimizer_evaluations, "maxiter": 35,
                               "xtol": 0.001, "ftol": 0.0001})
    if objective(result.x) < objective(best):
        best = result.x
    def actual_candidate(parameters):
        delta, offset = _delta_matrix(parameters)
        world = prior.copy()
        world[:2, :3] = delta @ prior[:2, :3]
        world[:2, 3] = target_center + offset - delta @ source_center
        world[2] = [0.0, z_scale, 0.0, base - z_scale * low]
        footprint, raster = raster_triangle_footprint(
            vertices, faces, world, cell_m=policy.desired_raster_cell_m,
            max_dimension=policy.maximum_raster_dimension)
        return world, footprint, raster, overlap_metrics(footprint, target)

    world, actual, final_raster, metrics = actual_candidate(best)
    selection = {"algorithm": "feasible_first_bounded_candidates_v1", "source_sha256": sha256_file(__file__),
                 "original_parameters": [float(value) for value in best],
                 "original_actual_mesh_overlap": metrics.copy(), "fallback_used": False,
                 "maximum_actual_fallback_candidates": 12, "actual_fallback_candidates": []}
    original_controls_pass = _parameter_violation(
        best, policy, horizontal_prior, component_horizontal_priors, offset_limit) <= 1e-9
    selection["original_control_bounds_passed"] = original_controls_pass
    fixed_gates_pass = (scene["all_primitives_have_normals"]
                        and policy.minimum_z_scale_ratio - 1e-9 <= z_ratio <= policy.maximum_z_scale_ratio + 1e-9)
    if (not _overlap_passes(metrics, policy) or not original_controls_pass) and fixed_gates_pass:
        ranked, search = _rank_feasible_candidates(
            centered_source, centered_target, best, evaluated_parameters, policy,
            horizontal_prior, component_horizontal_priors, offset_limit)
        selection.update(fallback_used=True, search=search)
        for candidate in ranked[:selection["maximum_actual_fallback_candidates"]]:
            parameters = candidate["parameters"]
            projected_world, footprint, raster, overlap = actual_candidate(parameters)
            passed = _overlap_passes(overlap, policy)
            audit = {**candidate, "actual_mesh_overlap": overlap, "passed": passed}
            # Finer projection prevents an approximate polygon or coarse raster
            # from approving a candidate that only appears feasible near a gate.
            margin = min(overlap[name] - threshold for name, threshold in (
                ("iou", policy.minimum_footprint_iou),
                ("target_coverage", policy.minimum_target_coverage),
                ("asset_coverage", policy.minimum_asset_coverage)))
            if passed and margin < .025:
                fine, profile = raster_triangle_footprint(
                    vertices, faces, projected_world, cell_m=policy.desired_raster_cell_m / 2,
                    max_dimension=policy.maximum_raster_dimension * 2)
                fine_overlap = overlap_metrics(fine, target)
                passed = _overlap_passes(fine_overlap, policy)
                audit["fine_projection_check"] = {"same_transform_no_retuning": True,
                    "metrics": fine_overlap, "projection": profile, "passed": passed}
                audit["passed"] = passed
            selection["actual_fallback_candidates"].append(audit)
            if passed:
                best = np.asarray(parameters)
                world, actual, final_raster, metrics = projected_world, footprint, raster, overlap
                selection["selected_parameters"] = candidate["parameters"]
                break
    delta, offset = _delta_matrix(best)
    singular = np.linalg.svd(delta @ horizontal_prior, compute_uv=False)
    incremental_anisotropy = max(best[1] / best[2], best[2] / best[1])
    total_anisotropy = float(singular[0] / singular[-1])
    reasons = []
    if not scene["all_primitives_have_normals"]:
        reasons.append("missing_original_normal_attribute")
    for name, threshold in (("iou", policy.minimum_footprint_iou),
                            ("target_coverage", policy.minimum_target_coverage),
                            ("asset_coverage", policy.minimum_asset_coverage)):
        if metrics[name] + 1e-9 < threshold:
            reasons.append(f"actual_mesh_{name}_below_threshold")
    if not policy.minimum_z_scale_ratio - 1e-9 <= z_ratio <= policy.maximum_z_scale_ratio + 1e-9:
        reasons.append("height_scale_change_exceeds_bound")
    if any(value < policy.minimum_xy_scale - 1e-9 or value > policy.maximum_xy_scale + 1e-9 for value in best[1:3]):
        reasons.append("xy_scale_change_exceeds_bound")
    if abs(best[0]) > policy.maximum_yaw_change_degrees + 1e-9:
        reasons.append("yaw_change_exceeds_bound")
    if incremental_anisotropy > policy.maximum_incremental_anisotropy + 1e-8:
        reasons.append("incremental_xy_anisotropy_exceeds_bound")
    if total_anisotropy > policy.maximum_total_anisotropy + 1e-8:
        reasons.append("total_xy_anisotropy_exceeds_bound")
    component_anisotropies = []
    for index, horizontal in enumerate(component_horizontal_priors):
        component_singular = np.linalg.svd(delta @ horizontal, compute_uv=False)
        anisotropy = float(component_singular[0] / component_singular[-1])
        component_anisotropies.append(anisotropy)
        if anisotropy > policy.maximum_total_anisotropy + 1e-8:
            reasons.append(f"component_total_xy_anisotropy_exceeds_bound:{index}")
    if np.linalg.norm(offset) > offset_limit + 1e-8:
        reasons.append("centroid_offset_radius_exceeds_bound")
    local_from_world = np.asarray(building["local_from_world_4x4"], dtype=float)
    world_from_local = np.asarray(building["world_from_local_4x4"], dtype=float)
    if not np.allclose(world_from_local @ local_from_world, np.eye(4), atol=1e-8):
        raise ValueError("building local/world matrices are not inverses")
    local = local_from_world @ world
    world_normal = np.linalg.inv(world[:3, :3]).T
    local_normal = np.linalg.inv(local[:3, :3]).T
    transformed_vertices = vertices @ world[:3, :3].T + world[:3, 3]
    for instance in scene["instances"]:
        node = np.array(instance["asset_from_node_4x4"])
        instance["world_from_node_4x4"] = (world @ node).tolist()
        instance["normal_node_to_world_3x3"] = np.linalg.inv((world @ node)[:3, :3]).T.tolist()
    record = {
        "schema": "citycode.asset_registration.v1", "building_id": building["building_id"],
        "status": "accepted" if not reasons else "review_required", "reasons": reasons,
        "acceptance_scope": "bounded individual CPU geometry fit; appearance and city collision checks separate",
        "source_glb": str(Path(source_glb).resolve()), "source_sha256": scene["source_sha256"],
        "source_bytes": scene["source_bytes"], "source_unchanged": True,
        "source_coordinate_system": "glTF scene Y-up including original node transforms",
        "target_coordinate_system": "UE XYZ metres Z-up",
        "policy": asdict(policy), "world_from_asset_4x4": world.tolist(),
        "candidate_selection": selection,
        "building_local_from_asset_4x4": local.tolist(),
        "world_from_building_local_4x4": world_from_local.tolist(),
        "normal_asset_to_world_3x3": world_normal.tolist(),
        "normal_asset_to_building_local_3x3": local_normal.tolist(),
        "normals_contract": "keep source NORMAL; inverse-transpose of composed node/asset matrix, then normalize",
        "source_node_instances": scene["instances"],
        "prior_world_from_asset_4x4": prior.tolist(),
        "fit": {
            "yaw_change_degrees": float(best[0]), "relative_xy_scales": [float(best[1]), float(best[2])],
            "relative_z_scale": float(z_ratio), "incremental_anisotropy": float(incremental_anisotropy),
            "total_xy_anisotropy": total_anisotropy,
            "offset_from_target_centroid_m": offset.tolist(), "offset_bound_m": offset_limit,
            "offset_bound_metric": "Euclidean radius in world XY metres",
            "actual_mesh_overlap": metrics, "optimizer_evaluations": evaluations,
            "optimizer_terminated_normally": bool(result.success),
            "prior_metric_centroid_world": (source_center + prior[:2, 3]).tolist(),
            "new_target_centroid_world": target_center.tolist(),
            "metric_relocation_m": float(np.linalg.norm(target_center - source_center - prior[:2, 3])),
        },
        "mesh": {"vertex_count": len(vertices), "triangle_count": len(faces),
                 "raw_scene_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                 "registered_bounds_world": [transformed_vertices.min(axis=0).tolist(), transformed_vertices.max(axis=0).tolist()],
                 "all_primitives_have_original_normals": scene["all_primitives_have_normals"]},
        "source_projection": raster_profile, "validation_projection": final_raster,
        "target_geometry": mapping(target), "actual_registered_footprint": mapping(actual),
        "target_summary": geometry_summary(target), "actual_footprint_summary": geometry_summary(actual),
        "predicted_base_z_m": base, "predicted_top_z_m": top,
        "actual_registered_base_z_m": float(transformed_vertices[:, 2].min()),
        "actual_registered_top_z_m": float(transformed_vertices[:, 2].max()),
    }
    if component_horizontal_priors:
        record["fit"]["component_total_xy_anisotropies"] = component_anisotropies
        record["component_prior_world_from_asset_4x4"] = component_prior_world_from_asset
        record["component_constraints_in_optimizer"] = True
    return record
