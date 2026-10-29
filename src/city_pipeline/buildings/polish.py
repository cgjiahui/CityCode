"""Optional prediction-only footprint cleanup and contextual facade filtering.

No operation is active with an empty config. Explicit configs use ``operations``
(closing/open_close/components) and ``remove_tiny_facades`` or ``fragment``.
Facade evidence is recomputed from directly observed predicted XYZ, never from
an evaluation table or a list of building IDs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from city_pipeline.runtime import read_json, sha256, write_json


NORMAL_RECIPE = {
    "random_seed": 2026, "max_direct_point_subsample": 20000,
    "max_queries": 512, "knn": 16, "planarity_gt": .15,
    "horizontal_abs_nz_gt": .85, "vertical_abs_nz_lt": .3,
    "direct_provenance_equals": 1, "finite_only": True,
}
FORBIDDEN_PATH_PARTS = (
    "evaluation", "ground_truth", "global_vectors", "per_gt_building",
    "gt_", "oracle", "diagnostic", "feature_cache",
)


def _parts(g):
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, (MultiPolygon, GeometryCollection)):
        return [part for child in g.geoms for part in _parts(child)]
    return []


def _clean(g):
    return unary_union(_parts(make_valid(g)))


def _support(p):
    return len(p.get("consensus_observation_ids", p.get("member_observation_ids", [])))


def _observation_lookup(observations):
    if isinstance(observations, dict) and "features" in observations:
        observations = observations["features"]
    if isinstance(observations, dict):
        return {str(key): value.get("properties", value) for key, value in observations.items()}
    result = {}
    for item in observations:
        p = item.get("properties", item)
        oid = str(p["observation_id"])
        if oid in result:
            raise ValueError("Duplicate predicted observation: " + oid)
        result[oid] = p
    return result


def _config(config):
    cfg = copy.deepcopy(config or {})
    unexpected = set(cfg) - {"name", "operations", "fragment", "remove_tiny_facades"}
    if unexpected:
        raise ValueError("Unknown polish parameters: " + ", ".join(sorted(unexpected)))
    operations = cfg.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("Each geometry operation must be an object")
        kind = op.get("kind")
        if kind not in {"closing", "bounded_closing", "open_close", "components", "control"}:
            raise ValueError("Unsupported geometry operation: " + str(kind))
        permitted = {"kind", "name", "height_min", "support_max", "solidity_max"}
        permitted |= {"minimum_area", "maximum_removed_ratio"} if kind == "components" else {"radius"}
        if kind == "bounded_closing":
            permitted |= {"conditions", "max_growth", "max_loss", "max_drift", "max_hausdorff_m"}
        if set(op) - permitted:
            raise ValueError("Unknown geometry operation parameters")
        for key, value in op.items():
            if key in {"kind", "name"}:
                continue
            if key == "conditions":
                allowed = {field + "_" + bound for field in ("area", "height", "support", "solidity", "rectangle_fill", "components") for bound in ("min", "max")}
                if not isinstance(value, dict) or set(value) - allowed:
                    raise ValueError("Only global numeric geometry conditions are accepted")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in value.values()):
                    raise ValueError("Geometry conditions must be finite and nonnegative")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Geometry parameter must be finite and nonnegative: " + key)
        if kind in {"closing", "bounded_closing", "open_close"} and "radius" not in op:
            raise ValueError("Geometry operation needs radius")
        if kind == "components" and "minimum_area" not in op:
            raise ValueError("Component operation needs minimum_area")
        if kind == "components" and op.get("maximum_removed_ratio", .01) > .01:
            raise ValueError("Tiny component removal is capped at one percent per operation")
        if kind == "bounded_closing" and not {"max_growth", "max_loss", "max_drift", "max_hausdorff_m"}.issubset(op):
            raise ValueError("Bounded closing needs explicit area, drift and distance limits")
    enabled = cfg.get("remove_tiny_facades", cfg.get("fragment") is not None)
    if not isinstance(enabled, bool):
        raise ValueError("remove_tiny_facades must be boolean")
    fragment = cfg.get("fragment") or {}
    if not isinstance(fragment, dict) or set(fragment) - {"area_max", "horizontal_max"}:
        raise ValueError("fragment accepts only global area_max and horizontal_max")
    for key, default in [("area_max", 100.), ("horizontal_max", .15)]:
        value = fragment.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid global fragment parameter: " + key)
        fragment[key] = value
    if fragment["horizontal_max"] > 1:
        raise ValueError("horizontal_max must be a fraction")
    cfg.update(operations=operations, remove_tiny_facades=enabled, fragment=fragment)
    return cfg


def _prediction_path(value, data_root, point_boundary_root=None):
    root = Path(data_root).resolve()
    p = Path(value)
    absolute = p.is_absolute()
    p = (p if absolute else root / p).resolve()
    boundary = Path(point_boundary_root).resolve() if point_boundary_root else root
    if not p.is_relative_to(boundary) or (not absolute and not p.is_relative_to(root)):
        raise PermissionError("Predicted point cloud is outside data_root: " + str(p))
    # Annotation-prefixed and evaluation paths are reserved for offline assessment.
    rel = p.relative_to(boundary).as_posix().lower()
    if any(token in rel for token in FORBIDDEN_PATH_PARTS) or p.suffix != ".npz":
        raise PermissionError("Expected raw predicted point cloud: " + str(p))
    return p


def normal_statistics(points):
    """Deterministic local PCA of finite direct points; no reference inputs."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected N by 3 predicted XYZ")
    points = points[np.isfinite(points).all(axis=1)]
    stats = {"direct_point_count": len(points), "planar_sample_count": 0,
             "horizontal_fraction": None, "vertical_fraction": None}
    if len(points) < 32:
        return stats
    rng = np.random.default_rng(NORMAL_RECIPE["random_seed"])
    sub = points[rng.choice(len(points), min(NORMAL_RECIPE["max_direct_point_subsample"], len(points)), replace=False)]
    query = sub[rng.choice(len(sub), min(NORMAL_RECIPE["max_queries"], len(sub)), replace=False)]
    _, indices = cKDTree(sub).query(query, k=NORMAL_RECIPE["knn"])
    local = sub[indices]
    local -= local.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local) / NORMAL_RECIPE["knn"]
    values, vectors = np.linalg.eigh(cov)
    planar = (values[:, 1] - values[:, 0]) / np.maximum(values[:, 2], 1e-9)
    good = planar > NORMAL_RECIPE["planarity_gt"]
    nz = np.abs(vectors[:, 2, 0])
    stats["planar_sample_count"] = int(good.sum())
    if good.any():
        stats.update(horizontal_fraction=float(np.mean(nz[good] > NORMAL_RECIPE["horizontal_abs_nz_gt"])),
                     vertical_fraction=float(np.mean(nz[good] < NORMAL_RECIPE["vertical_abs_nz_lt"])))
    return stats


def _facade_candidates(features, observations, data_root, fragment, point_boundary_root=None):
    geometries = [shape(f["geometry"]) for f in features]
    qualities = []
    for f in features:
        ids = f["properties"].get("consensus_observation_ids", [])
        qualities.append(max((float(observations.get(oid, {}).get("preliminary_quality", 0)) for oid in ids), default=0.))
    strong = [i for i, (f, g) in enumerate(zip(features, geometries))
              if g.area >= 250 and _support(f["properties"]) >= 2 and qualities[i] >= 85]
    strong_tree = STRtree([geometries[i] for i in strong])
    grouped = defaultdict(list)
    checks = []
    for i, (f, g) in enumerate(zip(features, geometries)):
        p = f["properties"]
        rectangle_area = g.minimum_rotated_rectangle.area
        fill = g.area / max(rectangle_area, 1e-9)
        if _support(p) != 1 or g.area >= fragment["area_max"] or fill >= .45:
            continue
        nearby = [(g.distance(geometries[strong[int(j)]]), strong[int(j)])
                  for j in strong_tree.query(g.buffer(1.), predicate="intersects") if strong[int(j)] != i]
        # Choose the nearest strong entity before testing relative area, exactly
        # as the exploratory context rule did. Do not search for a favorable peer.
        nearby.sort(key=lambda pair: (pair[0], pair[1]))
        if not nearby or nearby[0][0] >= 1:
            continue
        distance, neighbor = nearby[0]
        ratio = g.area / geometries[neighbor].area
        if ratio >= .25:
            continue
        check = {"id": p["predicted_global_building_id"], "index": i,
                 "area_m2": g.area, "rectangle_fill": fill,
                 "neighbor_id": features[neighbor]["properties"]["predicted_global_building_id"],
                 "neighbor_distance_m": distance, "area_ratio_self_over_neighbor": ratio,
                 "decision": "retain", "reason": "missing_predicted_points"}
        checks.append(check)
        oid = p.get("canonical_observation_id")
        observation = observations.get(oid, {})
        value = p.get("canonical_building_point_cloud_npz_path") or observation.get("building_point_cloud_npz_path")
        instance_id = p.get("canonical_instance_id", observation.get("rgb_instance_id"))
        if not value or instance_id is None:
            continue
        path = _prediction_path(value, data_root, point_boundary_root)
        grouped[path].append((check, int(instance_id)))
    hashes = {}
    reads = []
    for path, items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                xyz, ids, provenance = data["xyz_m"], data["instance_id"], data["provenance"]
                if xyz.ndim != 2 or xyz.shape[1] != 3 or ids.shape != (len(xyz),) or provenance.shape != (len(xyz),):
                    raise ValueError("Inconsistent predicted point cloud arrays")
            hashes[str(path)] = sha256(path)
            reads.append(str(path))
            for check, instance_id in items:
                points = xyz[(ids == instance_id) & (provenance == 1)]
                points = points[np.isfinite(points).all(axis=1)]
                stats = normal_statistics(points)
                check["normal_statistics"] = stats
                hf, vf = stats["horizontal_fraction"], stats["vertical_fraction"]
                if (stats["direct_point_count"] >= 500 and stats["planar_sample_count"] >= 200
                        and hf is not None and vf is not None
                        and hf < fragment["horizontal_max"] and vf > .4):
                    check.update(decision="delete", reason="tiny_contextual_vertical_fragment")
                else:
                    check["reason"] = "insufficient_or_nonvertical_normal_evidence"
        except PermissionError:
            raise
        except (OSError, ValueError, KeyError):
            for check, _ in items:
                check["reason"] = "invalid_predicted_point_cloud"
    return checks, hashes, reads


def _transform(feature, operation):
    g = shape(feature["geometry"])
    p = feature["properties"]
    kind = operation["kind"]
    if kind == "control":
        return g
    if p.get("height_m", 0.) < operation.get("height_min", 0.) or _support(p) > operation.get("support_max", float("inf")):
        return g
    if g.area / max(g.convex_hull.area, 1e-9) > operation.get("solidity_max", 1.):
        return g
    if kind == "bounded_closing":
        values = {"area": g.area, "height": p.get("height_m", 0.), "support": _support(p),
                  "solidity": g.area / max(g.convex_hull.area, 1e-9),
                  "rectangle_fill": g.area / max(g.minimum_rotated_rectangle.area, 1e-9),
                  "components": len(_parts(g))}
        for condition, threshold in operation.get("conditions", {}).items():
            field, bound = condition.rsplit("_", 1)
            if (bound == "min" and values[field] < threshold) or (bound == "max" and values[field] > threshold):
                return g
    if kind in {"closing", "bounded_closing"}:
        r = operation["radius"]
        result = g.buffer(r, join_style=2).buffer(-r, join_style=2)
    elif kind == "open_close":
        r = operation["radius"]
        result = g.buffer(-r, join_style=2).buffer(r, join_style=2)
        result = result.buffer(r, join_style=2).buffer(-r, join_style=2)
    else:
        components = _parts(g)
        if len(components) < 2:
            return g
        largest = max(components, key=lambda part: part.area)
        result = unary_union([part for part in components if part == largest or part.area >= operation["minimum_area"]])
        if 1 - result.area / g.area > operation.get("maximum_removed_ratio", .01):
            return g
    result = _clean(result)
    if result.is_empty or not result.is_valid or result.area < .5:
        return g
    if kind == "bounded_closing":
        ratio = result.area / g.area
        if ratio > 1 + operation["max_growth"] + 1e-10 or ratio < 1 - operation["max_loss"] - 1e-10:
            return g
        if result.symmetric_difference(g).area / g.area > operation["max_drift"] + 1e-10:
            return g
        if result.hausdorff_distance(g) > operation["max_hausdorff_m"] + 1e-10:
            return g
        holes = [Polygon(ring) for part in _parts(g) for ring in part.interiors if Polygon(ring).area >= 1.]
        if any(result.intersection(hole).area > max(.1, .05 * hole.area) for hole in holes):
            return g
    if not .95 <= result.area / g.area <= 1.15 or result.symmetric_difference(g).area / g.area > .16:
        return g
    return result


def _apply_geometry(features, operation):
    result = copy.deepcopy(features)
    original = [shape(f["geometry"]) for f in features]
    geometries = [_transform(f, operation) for f in features]
    reverted = set()
    tree = STRtree(geometries)
    for i, g in enumerate(geometries):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            area = g.intersection(geometries[j]).area
            if area >= 20 and area / min(g.area, geometries[j].area) >= .5:
                reverted.update([i, j])
            elif operation["kind"] == "bounded_closing":
                before = original[i].intersection(original[j]).area
                if area - before > max(5., .02 * min(g.area, geometries[j].area)):
                    reverted.update([i, j])
    changes = []
    for i, f in enumerate(result):
        g = original[i] if i in reverted else geometries[i]
        difference = g.symmetric_difference(original[i]).area
        if difference < 1e-8:
            continue
        p = f["properties"]
        f["geometry"] = mapping(g)
        p.update(footprint_area_m2=g.area, centroid_x_m=g.centroid.x,
                 centroid_y_m=g.centroid.y, geometry_polish=copy.deepcopy(operation))
        layers = []
        for layer in p.get("roof_layers", []):
            layer = copy.deepcopy(layer)
            clipped = _clean(shape(layer["geometry"]).intersection(g))
            if not clipped.is_empty and clipped.area > .5:
                layer["geometry"] = mapping(clipped)
                layers.append(layer)
        p["roof_layers"] = layers
        changes.append({"id": p["predicted_global_building_id"], "area_before": original[i].area,
                        "area_after": g.area, "symmetric_difference_m2": difference})
    return result, {"operation": copy.deepcopy(operation), "changes": changes,
                    "overlap_reverted_indices": sorted(reverted)}


def apply_polish(features, observations, data_root, config, point_boundary_root=None):
    """Return independent features and an audit report, keeping all heights fixed."""
    cfg = _config(config)
    result = copy.deepcopy(features)
    lookup = _observation_lookup(observations)
    checks, hashes, reads = ([], {}, [])
    if cfg["remove_tiny_facades"]:
        checks, hashes, reads = _facade_candidates(result, lookup, data_root, cfg["fragment"], point_boundary_root)
    removed = {item["id"] for item in checks if item["decision"] == "delete"}
    result = [f for f in result if f["properties"]["predicted_global_building_id"] not in removed]
    operation_reports = []
    for operation in cfg["operations"]:
        result, report = _apply_geometry(result, operation)
        operation_reports.append(report)
    return result, {"parameters": cfg, "input_entity_count": len(features),
                    "output_entity_count": len(result), "removed_by_global_rule": sorted(removed),
                    "fragment_checks": checks, "operations": operation_reports,
                    "source_sha256": hashes, "runtime_data_reads": sorted(reads),
                    "normal_recipe": dict(NORMAL_RECIPE), "uses_ground_truth": False,
                    "height_policy": "Retained body/top heights unchanged; upper roof regions clipped to retained footprint."}


def _install_read_guard(paths, data_root, output_root):
    allowed = {Path(path).resolve() for path in paths}
    root = Path(data_root).resolve()
    output = Path(output_root).resolve()
    reads = set()
    def guard(event, values):
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        mode, flags = values[1:3]
        if isinstance(mode, str) and any(letter in mode for letter in "wax"):
            return
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.suffix in {".py", ".pyc", ".so"}:
            return
        if path in allowed or path.is_relative_to(output):
            reads.add(str(path))
        elif path.is_relative_to(root):
            raise PermissionError("Non-prediction input in polish stage: " + str(path))
    sys.addaudithook(guard)
    return reads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--point-boundary-root", type=Path,
                        help="Explicit boundary for absolute prediction paths; relative paths stay rooted at data-root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON global config; empty object leaves predictions unchanged.")
    args = parser.parse_args(argv)
    source = (args.input_root / "global_building_footprints.geojson").resolve()
    observation_path, config_path = args.observations.resolve(), args.config.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in [source, observation_path, config_path]:
        if any(token in path.as_posix().lower() for token in FORBIDDEN_PATH_PARTS):
            raise PermissionError("Reference/diagnostic metadata is not a polish input: " + str(path))
    # Metadata is loaded explicitly; no directory search or prior trial is read.
    document, observations, config = read_json(source), read_json(observation_path), _config(read_json(config_path))
    allow = [source, observation_path, config_path]
    if config["remove_tiny_facades"]:
        for f in document["features"]:
            value = f["properties"].get("canonical_building_point_cloud_npz_path")
            if value:
                allow.append(_prediction_path(value, args.data_root, args.point_boundary_root))
        for p in _observation_lookup(observations).values():
            if p.get("building_point_cloud_npz_path"):
                allow.append(_prediction_path(p["building_point_cloud_npz_path"], args.data_root, args.point_boundary_root))
    reads = _install_read_guard(allow, args.point_boundary_root or args.data_root, output)
    features, report = apply_polish(document["features"], observations, args.data_root, config, args.point_boundary_root)
    report["source_sha256"].update({str(path): sha256(path) for path in [source, observation_path, config_path]})
    output.mkdir(parents=True)
    prediction = output / "global_building_footprints.geojson"
    write_json(prediction, {**document, "features": features})
    changes = [{"id": item, "action": "delete_tiny_contextual_facade"} for item in report["removed_by_global_rule"]]
    for stage in report["operations"]:
        changes.extend({**item, "action": "geometry_polish", "operation": stage["operation"]} for item in stage["changes"])
    write_json(output / "changes.json", changes)
    write_json(output / "manifest.json", {"status": "complete", "schema": "citycode.structure_cleanup.v1",
               **report, "runtime_data_reads": sorted(reads), "building_count": len(features),
               "prediction_sha256": sha256(prediction),
               "prediction_contract": "Global rules on predicted geometry, observations and raw direct XYZ only. No GT/evaluation lookup or configured per-building decisions. No meshes produced."})
    print("Polished", len(features), "buildings; removed", len(report["removed_by_global_rule"]), flush=True)
    return output


if __name__ == "__main__":
    main()
