"""Export partial observed surfaces without completing an independent building.

Vertices retain their original predicted XYZ and RGB. Coarser faces are allowed
only when every original grid sample in the spanned neighborhood is directly
observed, belongs to the same instance, and passes the unchanged footprint and
height checks. These patches may contain roofs, steps, and visible facades.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from shapely import contains_xy, covers, polygons, union_all
from shapely.geometry import mapping, shape


SCHEMA = "citycode.observed_fragment_surface.v2"
POLICY = {
    "allowed_provenance_ids": [1],
    "allowed_grid_strides": [2, 3],
    "grid_replay_tolerance_pixels": 0.025,
    "maximum_triangle_edge_xy_m_per_stride": 0.22,
    "maximum_neighborhood_z_range_m": 1.5,
    "minimum_z_above_source_base_m": 0.12,
    "minimum_component_triangle_count": 8,
    "minimum_component_projected_area_m2": 0.1,
    "maximum_closed_triangle_count": 100000,
    "full_neighborhood_direct": True,
    "triangle_rule": "two fixed-diagonal triangles from actual grid corner samples",
    "component_connectivity": "shared original edge",
    "world_xyz_policy": "original source float32 XYZ copied losslessly; no interpolation or clamping",
    "footprint_rule": "all neighborhood samples inside current footprint and full triangles covered",
    "unknown_completion": False,
    "independent_building": False,
    "ground_extrusion": False,
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _proof(path: Path, base: Path, path_base: str) -> dict:
    return {
        "path": path.resolve().relative_to(base.resolve()).as_posix(),
        "path_base": path_base,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _lookup_grid(uv: np.ndarray):
    if not len(uv):
        return lambda query: np.full(np.asarray(query).shape, -1, dtype=np.int64)
    lo = uv.min(axis=0)
    span = int(uv[:, 0].max() - lo[0] + 5)
    keys = (uv[:, 1] - lo[1]).astype(np.int64) * span + uv[:, 0] - lo[0]
    order = np.argsort(keys)
    sorted_keys = keys[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise ValueError("Multiple source samples occupy an original BEV pixel")

    def lookup(query):
        query = np.asarray(query)
        key = (query[..., 1] - lo[1]).astype(np.int64) * span + query[..., 0] - lo[0]
        pos = np.searchsorted(sorted_keys, key)
        ix = np.minimum(pos, len(sorted_keys) - 1)
        ok = (pos < len(sorted_keys)) & (sorted_keys[ix] == key)
        # Exact UV replay also guards key wrapping outside the indexed range.
        ok &= np.all(uv[order[ix]] == query, axis=-1)
        return np.where(ok, order[ix], -1)

    return lookup


def grid_triangles(xyz: np.ndarray, uv: np.ndarray, footprint, stride: int) -> np.ndarray:
    """Use real corners only when the full original neighborhood is supported."""
    if stride not in POLICY["allowed_grid_strides"]:
        raise ValueError("Unsupported original-grid stride")
    if not len(uv):
        return np.empty((0, 3), dtype=np.int64)
    lookup = _lookup_grid(uv)
    anchors = uv[(uv[:, 0] % stride == 0) & (uv[:, 1] % stride == 0)]
    offsets = np.asarray([(u, v) for v in range(stride + 1) for u in range(stride + 1)])
    neighborhood = lookup(anchors[:, None, :] + offsets[None, :, :])
    neighborhood = neighborhood[np.all(neighborhood >= 0, axis=1)]
    if not len(neighborhood):
        return np.empty((0, 3), dtype=np.int64)
    neighborhood = neighborhood[
        np.ptp(xyz[neighborhood, 2], axis=1) <= POLICY["maximum_neighborhood_z_range_m"]
    ]
    right, diagonal, down = stride, (stride + 1) ** 2 - 1, stride * (stride + 1)
    faces = np.concatenate([
        neighborhood[:, [0, right, diagonal]], neighborhood[:, [0, diagonal, down]]
    ], axis=0)
    if not len(faces):
        return faces
    q = xyz[faces]
    edges = np.stack([q[:, 1, :2] - q[:, 0, :2], q[:, 2, :2] - q[:, 1, :2], q[:, 0, :2] - q[:, 2, :2]], axis=1)
    good = np.linalg.norm(edges, axis=2).max(axis=1) <= POLICY["maximum_triangle_edge_xy_m_per_stride"] * stride
    faces = faces[good]
    if len(faces):
        faces = faces[np.asarray(covers(footprint, polygons(xyz[faces, :2])), dtype=bool)]
        q = xyz[faces, :2]
        signed = ((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                  - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0]))
        flip = signed < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def _edge_groups(faces: np.ndarray):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(edges, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(edges)])
    return edges, owner, starts, counts


def filter_components(xyz: np.ndarray, faces: np.ndarray):
    """Remove isolated tiny pieces, retaining an explicit record of each removal."""
    if not len(faces):
        return faces, np.empty(0, dtype=np.int64), []
    edges, owner, starts, counts = _edge_groups(faces)
    if np.any(counts > 2):
        raise ValueError("Observed surface contains a nonmanifold shared edge")
    parent = np.arange(len(faces), dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for start in starts[counts == 2]:
        a, b = find(int(owner[start])), find(int(owner[start + 1]))
        if a != b:
            parent[b] = a
    roots = np.asarray([find(i) for i in range(len(faces))])
    _, component = np.unique(roots, return_inverse=True)
    q = xyz[faces, :2]
    area = np.abs((q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                  - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])) / 2
    size = np.bincount(component)
    areas = np.bincount(component, weights=area)
    good = (size >= POLICY["minimum_component_triangle_count"]) & (areas >= POLICY["minimum_component_projected_area_m2"])
    removed = [
        {"component_index": int(k), "triangle_count": int(size[k]), "projected_area_m2": float(areas[k])}
        for k in np.flatnonzero(~good)
    ]
    keep = good[component]
    return faces[keep], component[keep], removed


def closed_mesh_budget(faces: np.ndarray) -> dict:
    if not len(faces):
        return {"top_triangle_count": 0, "boundary_edge_count": 0, "closed_triangle_count": 0}
    _, _, _, counts = _edge_groups(faces)
    if np.any(counts > 2):
        raise ValueError("A nonmanifold patch cannot be closed as a thin surface")
    boundary = int(np.count_nonzero(counts == 1))
    return {"top_triangle_count": len(faces), "boundary_edge_count": boundary,
            "closed_triangle_count": 2 * len(faces) + 2 * boundary}


def _source_patch(building_dir: Path, cache_root: Path, item: dict, footprint, layer_index: int, minimum_z: float):
    op = building_dir / item["path"] / "observation.json"
    ob = json.loads(op.read_text())
    source = cache_root / "data/points" / (ob["sample_id"] + "__height_cleaned_building_points.npz")
    with np.load(source, allow_pickle=False) as a:
        sid = int(ob["source_instance_id"])
        if int(ob["bev_instance_id"]) != sid:
            raise ValueError("Source and BEV instance identities require reconciliation")
        eligible = ((a["instance_id"] == sid) & (a["provenance"] == 1)
                    & np.isfinite(a["xyz_m"]).all(axis=1))
        idx = np.flatnonzero(eligible)
        xyz = a["xyz_m"][idx].astype(np.float64)
        inside = contains_xy(footprint, xyz[:, 0], xyz[:, 1])
        above = xyz[:, 2] >= minimum_z
        rejected_below = int(np.count_nonzero(inside & ~above))
        idx, xyz = idx[inside & above], xyz[inside & above]
        matrix = np.asarray(ob["bev_crop_pixel_center_to_world_xy_3x3"], dtype=np.float64)
        grid = (np.column_stack([xyz[:, :2], np.ones(len(xyz))]) @ np.linalg.inv(matrix).T)[:, :2]
        uv = np.rint(grid).astype(np.int64)
        residual = np.max(np.abs(grid - uv), axis=1)
        replay = residual <= POLICY["grid_replay_tolerance_pixels"]
        idx, xyz, uv = idx[replay], xyz[replay], uv[replay]
        arrays = {"xyz": xyz, "uv": uv, "flat_index": idx,
                  "rgb": a["rgb"][idx], "instance_id": a["instance_id"][idx],
                  "provenance": a["provenance"][idx]}
    record = {
        "observation_id": ob["observation_id"], "sample_id": ob["sample_id"],
        "source_instance_id": sid, "bev_instance_id": sid,
        "source_points": _proof(source, cache_root, "cache_root"),
        "observation": _proof(op, building_dir, "building_directory"),
        "grid_pixel_center_to_world_xy_3x3": matrix.tolist(),
        "roof_layer_index": layer_index,
        "inside_direct_sample_count": int(inside.sum()),
        "rejected_below_source_base_plus_thickness_sample_count": rejected_below,
        "grid_replay_valid_sample_count": len(xyz),
        "grid_replay_maximum_error_px": float(residual.max()) if len(residual) else None,
        "rejected_non_grid_samples": int((~replay).sum()),
        "clip_geometry_world": mapping(footprint),
    }
    return {**arrays, "footprint": footprint, "record": record}


def build_observed_surface(building_path: Path, cache_root: Path, output_dir: Path,
                           observation_ids: Sequence[str] | None = None) -> dict:
    """Write one portable, directly supported partial-surface candidate.

    cache_root contains the published prediction cache; building_path points at
    one published structure input. Alternate observations are permitted only
    for explicitly recorded roof-layer regions, preventing duplicate same-roof
    views from being silently overlaid.
    """
    building_path, cache_root, output_dir = Path(building_path), Path(cache_root), Path(output_dir)
    b = json.loads(building_path.read_text())
    fp = shape(b["footprint_world"])
    available = {o["observation_id"]: o for o in b["observations"]}
    canonical = b["canonical_observation_id"]
    selected = list(observation_ids) if observation_ids is not None else [canonical]
    if observation_ids is None:
        selected.extend(oi for layer in b.get("roof_layers", []) for oi in layer.get("observation_ids", []) if oi != canonical)
    selected = list(dict.fromkeys(selected))
    if canonical not in selected:
        raise ValueError("The canonical observation must remain the primary source")
    layers = {}
    for li, layer in enumerate(b.get("roof_layers", [])):
        for oi in layer.get("observation_ids", []):
            if oi != canonical:
                if oi in layers:
                    raise ValueError("One alternate view supports multiple layer regions; explicit split needed")
                layers[oi] = (li, shape(layer["geometry"]).intersection(fp))
    minimum_z = float(b["base_z_m"]) + POLICY["minimum_z_above_source_base_m"]
    parts = []
    for oi in selected:
        if oi not in available or (oi != canonical and oi not in layers):
            raise ValueError("Selected alternate observation lacks an explicit current roof layer")
        li, clip = (-1, fp) if oi == canonical else layers[oi]
        parts.append(_source_patch(building_path.parent, cache_root, available[oi], clip, li, minimum_z))
    attempts = []
    chosen = None
    for stride in POLICY["allowed_grid_strides"]:
        offset, component_offset = 0, 0
        arrays = {k: [] for k in ["xyz", "rgb", "flat_index", "uv", "instance_id", "provenance", "faces", "vobs", "fobs", "flayer", "component"]}
        stats = []
        for oi, part in enumerate(parts):
            original_faces = grid_triangles(part["xyz"], part["uv"], part["footprint"], stride)
            faces, component, removed = filter_components(part["xyz"], original_faces)
            used = np.unique(faces.reshape(-1)) if len(faces) else np.empty(0, dtype=np.int64)
            lookup = np.full(len(part["xyz"]), -1, dtype=np.int64)
            lookup[used] = np.arange(len(used))
            for k in ["xyz", "rgb", "flat_index", "uv", "instance_id", "provenance"]:
                arrays[k].append(part[k][used])
            arrays["faces"].append(lookup[faces] + offset)
            arrays["vobs"].append(np.full(len(used), oi, dtype=np.int16))
            arrays["fobs"].append(np.full(len(faces), oi, dtype=np.int16))
            arrays["flayer"].append(np.full(len(faces), part["record"]["roof_layer_index"], dtype=np.int16))
            arrays["component"].append(component + component_offset)
            component_offset += int(component.max()) + 1 if len(component) else 0
            offset += len(used)
            stats.append({"observation_index": oi, "used_vertex_count": len(used), "triangle_count": len(faces),
                          "triangles_before_component_filter": len(original_faces), "removed_small_components": removed})
        result = {k: np.concatenate(v) for k, v in arrays.items()}
        budget = closed_mesh_budget(result["faces"])
        attempts.append({"grid_stride": stride, **budget})
        if budget["closed_triangle_count"] <= POLICY["maximum_closed_triangle_count"]:
            chosen = (stride, result, stats, budget)
            break
    if chosen is None:
        raise ValueError("Supported surface remains above closed-mesh face budget; no geometry was extrapolated")
    stride, a, stats, budget = chosen
    xyz, faces = a["xyz"], a["faces"]
    projected = union_all(polygons(xyz[faces, :2])) if len(faces) else fp.difference(fp)
    stratum = []
    for oi in range(1, len(parts)):
        if not np.any(a["fobs"] == oi) or not np.any(a["fobs"] == 0):
            continue
        canonical_union = union_all(polygons(xyz[faces[a["fobs"] == 0], :2]))
        alternate_union = union_all(polygons(xyz[faces[a["fobs"] == oi], :2]))
        overlap = float(canonical_union.intersection(alternate_union).area)
        dz = abs(float(np.median(xyz[a["vobs"] == 0, 2])) - float(np.median(xyz[a["vobs"] == oi, 2])))
        if overlap and dz <= POLICY["maximum_neighborhood_z_range_m"]:
            raise ValueError("Alternate source may duplicate the canonical stratum at the same height")
        stratum.append({"observation_index": oi, "planar_overlap_m2": overlap,
                        "median_stratum_height_difference_m": dz})
    output_dir.mkdir(parents=True, exist_ok=True)
    npz = output_dir / "supported_surface.npz"
    np.savez_compressed(npz, vertices_world_xyz_m=xyz, faces=faces,
        vertex_rgb=a["rgb"], vertex_source_flat_index=a["flat_index"],
        vertex_original_grid_uv=a["uv"], vertex_source_instance_id=a["instance_id"],
        vertex_source_provenance=a["provenance"], vertex_observation_index=a["vobs"],
        triangle_observation_index=a["fobs"], triangle_roof_layer_index=a["flayer"],
        triangle_component_index=a["component"])
    observations = [{**part["record"], **stat} for part, stat in zip(parts, stats)]
    rec = {
        "schema": SCHEMA, "building_id": b["building_id"],
        "status": "supported_surface_candidate" if len(faces) else "evidence_only_insufficient_supported_triangles",
        "representation": "observed_surface_fragment", "independent_building": False,
        "release_approved": False,
        "source_building": _proof(building_path, building_path.parent, "building_directory"),
        "source_cache_root_resolution": "caller supplied published prediction cache directory",
        "algorithm": {"module": "citycode_assets.observed_surface", "sha256": file_sha256(Path(__file__))},
        "policy": {**POLICY, "grid_stride": stride, "minimum_source_z_m": minimum_z,
                   "maximum_triangle_edge_xy_m": POLICY["maximum_triangle_edge_xy_m_per_stride"] * stride},
        "stride_budget_attempts": attempts, "closed_mesh_budget": budget,
        "source_observations": observations,
        "excluded_noncanonical_observations": [oi for oi in available if oi not in selected],
        "world_coordinate_system": "UE XYZ metres Z-up", "vertex_count": len(xyz), "triangle_count": len(faces),
        "source_footprint_area_m2": float(fp.area), "supported_planar_union_area_m2": float(projected.area),
        "source_footprint_planar_coverage": float(projected.area / fp.area),
        "actual_observed_surface_z_range_m": [float(xyz[:, 2].min()), float(xyz[:, 2].max())] if len(xyz) else None,
        "actual_observed_surface_z_quantiles_m": np.quantile(xyz[:, 2], [.05, .5, .95]).tolist() if len(xyz) else None,
        "source_global_top_z_metadata_m": b["top_z_m"], "multiple_source_strata": stratum,
        "surface": _proof(npz, output_dir, "surface_output_directory"),
        "parent_source_building_id": None, "parent_asset_instance_id": None,
        "assembly_action": "root_review_required",
        "qualifications": [
            "Partial predicted surface includes roof, steps, and visible facade fragments; it is not an independent complete building.",
            "All retained XYZ/RGB are direct source predictions. Neither a global flat top nor a ground extrusion is used.",
            "Every spanned original grid neighborhood is directly supported; unknown regions and discontinuities remain absent.",
            "Sub-base samples and tiny isolated components are removed with counts; no coordinates are clamped or interpolated.",
            "Surface coverage reports supported area only and is not a completeness or building accuracy score.",
        ],
    }
    (output_dir / "SURFACE_MANIFEST.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observation-id", action="append")
    args = parser.parse_args()
    result = build_observed_surface(args.building, args.cache, args.output, args.observation_id)
    print(json.dumps({key: result[key] for key in ["building_id", "status", "triangle_count", "source_footprint_planar_coverage", "closed_mesh_budget"]}))


if __name__ == "__main__":
    main()
