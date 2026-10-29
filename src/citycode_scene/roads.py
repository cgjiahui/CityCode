"""Height-aware navigation clearance against placed mesh columns.

Lane centreline XYZ and lane width define a navigation envelope. A projected
intersection alone is not a collision: upper decks, canopies and separate mesh
components are tested at their own height. Small edge contacts remain advisory.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, mapping, shape
from shapely.strtree import STRtree


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lanes(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"vehicle": [], "pedestrian": []}
    skipped: list[dict[str, Any]] = []
    inferred = 0
    raw = json.loads(path.read_text())
    for i, feature in enumerate(raw.get("features", [])):
        props = feature.get("properties") or {}
        tags = props.get("tags", props.get("lane_type", props.get("type", [])))
        tags = [tags] if isinstance(tags, str) else (tags or [])
        normal = {str(t).lower() for t in tags}
        kinds = [k for k, aliases in [("vehicle", {"vehicle", "driving", "car"}), ("pedestrian", {"pedestrian", "walking", "walk"})] if normal & aliases]
        if not kinds:
            continue
        geometry = feature.get("geometry") or {}
        parts = geometry.get("coordinates", [])
        if geometry.get("type") == "LineString":
            parts = [parts]
        elif geometry.get("type") != "MultiLineString":
            skipped.append({"feature": i, "reason": "expected LineString or MultiLineString"})
            continue
        for part_no, coordinates in enumerate(parts):
            xyz = np.asarray(coordinates, dtype=float)
            if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] < 2 or not np.isfinite(xyz).all():
                skipped.append({"feature": i, "part": part_no, "reason": "invalid line coordinates"})
                continue
            assumed_z = xyz.shape[1] < 3
            if assumed_z:
                # A fallback makes a 2D dataset usable, while the report keeps
                # this limitation explicit instead of inventing bridge levels.
                z = float(props.get("z_m", props.get("elevation_m", 0.0)))
                xyz = np.column_stack([xyz[:, :2], np.full(len(xyz), z)])
                inferred += 1
            else:
                xyz = xyz[:, :3]
            if np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum() <= 1e-9:
                skipped.append({"feature": i, "part": part_no, "reason": "zero horizontal length"})
                continue
            for kind in kinds:
                width = float(props.get("width_m", props.get("width", 4.0 if kind == "vehicle" else 1.0)))
                if not math.isfinite(width) or width <= 0:
                    skipped.append({"feature": i, "part": part_no, "reason": "non-positive lane width"})
                    continue
                line = LineString(xyz[:, :2])
                corridor = line.buffer(width / 2, cap_style=2, join_style=2)
                lane_id = str(props.get("lane_id", feature.get("id", i)))
                if len(parts) > 1:
                    lane_id += f"/part{part_no}"
                groups[kind].append({"lane_id": lane_id, "road_section_id": props.get("road_section_id"), "road_class": props.get("road_class"), "width_m": width, "coordinates_xyz_m": xyz, "corridor": corridor, "elevation_assumed": assumed_z})
    return groups, {"input_features": len(raw.get("features", [])), "inferred_elevation_line_parts": inferred, "skipped": skipped}


def road_height(xy: np.ndarray, lane: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate Z on the nearest horizontal segment of one original lane."""
    xyz = np.asarray(lane["coordinates_xyz_m"], dtype=float)
    a = xyz[:-1]
    delta = xyz[1:] - a
    squared = np.sum(delta[:, :2] ** 2, axis=1)
    distance2 = np.full(len(xy), np.inf)
    z = np.zeros(len(xy))
    for p, d, denominator in zip(a, delta, squared):
        if denominator <= 1e-12:
            continue
        t = np.clip(np.sum((xy - p[:2]) * d[:2], axis=1) / denominator, 0, 1)
        q = p[:2] + t[:, None] * d[:2]
        value = np.sum((xy - q) ** 2, axis=1)
        use = value < distance2
        z[use] = (p[2] + t * d[2])[use]
        distance2[use] = value[use]
    return z, np.sqrt(distance2)


def column_occupancy(ids: np.ndarray, road_z: np.ndarray, columns: dict[str, Any], clearance_m: float, deck_tolerance_m: float = .15, minimum_solid_overlap_m: float = .1) -> dict[str, np.ndarray]:
    """Pair closed column crossings; keep open/odd columns explicitly uncertain."""
    offsets, hits = columns["offsets"], columns["hits"]
    starts = offsets[ids]
    counts = offsets[ids + 1] - starts
    even = (counts >= 2) & (counts % 2 == 0)
    # The occupied navigation volume starts above the road surface. A closed
    # ground/base slab ending at road Z is deck contact, not blocked headroom.
    low, high = road_z + .25, road_z + clearance_m
    volume = np.zeros(len(ids))
    surface = np.zeros(len(ids), dtype=bool)
    deck = surface.copy()
    for k in range(int(counts.max(initial=0))):
        use = counts > k
        ix = np.flatnonzero(use)
        zs = hits[starts[use] + k]
        surface[ix] |= (zs >= road_z[use] + .25) & (zs <= high[use])
        deck[ix] |= (zs >= road_z[use] - deck_tolerance_m) & (zs <= road_z[use] + deck_tolerance_m)
        if k % 2 == 0:
            pair = even & (counts > k + 1)
            ix = np.flatnonzero(pair)
            if len(ix):
                z0, z1 = hits[starts[pair] + k], hits[starts[pair] + k + 1]
                volume[ix] = np.maximum(volume[ix], np.maximum(0, np.minimum(z1, high[pair]) - np.maximum(z0, low[pair])))
    solid = volume >= minimum_solid_overlap_m
    occupied = solid | surface
    return {"occupied": occupied, "solid": solid, "surface": surface, "deck": deck, "uncertain": (counts > 0) & ~even & ~occupied, "vertical_overlap": volume}


def union_occupancy(xy: np.ndarray, road_z: np.ndarray, components: list[dict[str, Any]], clearance_m: float, deck_tolerance_m: float = .15) -> dict[str, np.ndarray]:
    """Union physical components; do not apply aggregate parity XOR to them."""
    out = {key: np.zeros(len(xy), dtype=bool) for key in ["occupied", "solid", "surface", "deck", "uncertain"]}
    out["vertical_overlap"] = np.zeros(len(xy))
    for c in components:
        ij = np.floor((xy - c["origin"]) / c["cell"]).astype(int)
        h, w = c["zmin"].shape
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        ix = np.flatnonzero(ok)
        ids = ij[ok, 1] * w + ij[ok, 0]
        if not len(ids):
            continue
        active = np.isfinite(c["zmin"].ravel()[ids])
        ix, ids = ix[active], ids[active]
        if not len(ids):
            continue
        value = column_occupancy(ids, road_z[ix], c, clearance_m, deck_tolerance_m)
        for key in ["occupied", "solid", "surface", "deck", "uncertain"]:
            out[key][ix] |= value[key]
        out["vertical_overlap"][ix] = np.maximum(out["vertical_overlap"][ix], value["vertical_overlap"])
    out["uncertain"] &= ~out["occupied"]
    return out


def _line_blockage(lane: dict[str, Any], components: list[dict[str, Any]], clearance: float, cell: float, deck: float) -> tuple[float, float]:
    xyz = np.asarray(lane["coordinates_xyz_m"], dtype=float)
    delta = np.diff(xyz, axis=0)
    lengths = np.linalg.norm(delta[:, :2], axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    total = cumulative[-1]
    if total <= 0:
        return 0., 0.
    n = max(1, int(math.ceil(total / cell)))
    step = total / n
    s = (np.arange(n) + .5) * step
    i = np.clip(np.searchsorted(cumulative, s, side="right") - 1, 0, len(lengths) - 1)
    t = (s - cumulative[i]) / np.maximum(lengths[i], 1e-12)
    samples = xyz[i] + delta[i] * t[:, None]
    value = union_occupancy(samples[:, :2], samples[:, 2], components, clearance, deck)
    return float(value["occupied"].sum() * step), float(value["uncertain"].sum() * step)


def _load_columns(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        return {key: data[key] for key in ["origin", "cell", "zmin", "offsets", "hits"]}


def _one(cache: Path, meta: dict[str, Any], lanes: list[dict[str, Any]], tree: STRtree | None, clearance: float, kind: str, edge_tolerance: float, centerline_threshold: float, deck: float, masks_dir: Path | None, columns_override: dict[str, Any] | None = None, component_columns_override: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    c = columns_override if columns_override is not None else _load_columns(cache / meta["columns"])
    descriptors = meta.get("components") or [meta]
    if columns_override is not None and meta.get("components") and component_columns_override is None:
        raise ValueError("Transformed multi-component columns require transformed component_columns; original component caches cannot be reused")
    if component_columns_override is not None:
        components = component_columns_override or [c]
    else:
        components = [c] if not meta.get("components") else [_load_columns(cache / x["columns"]) for x in descriptors]
    origin, cell = c["origin"], float(c["cell"])
    h, w = c["zmin"].shape
    geom = shape(meta["footprint_world"])
    indices = tree.query(geom.buffer(cell), predicate="intersects") if tree is not None else []
    masks = {key: np.zeros(h * w, dtype=bool) for key in ["projected", "occupied", "solid", "surface", "deck", "uncertain", "deep_edge", "deep05"]}
    rows = []
    max_depth = max_vertical = total_blocked = total_unknown = 0.
    for j in indices:
        lane = lanes[int(j)]
        poly = lane["corridor"]
        b = poly.bounds
        x0, x1 = max(0, math.floor((b[0] - origin[0]) / cell)), min(w, math.ceil((b[2] - origin[0]) / cell))
        y0, y1 = max(0, math.floor((b[1] - origin[1]) / cell)), min(h, math.ceil((b[3] - origin[1]) / cell))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ids = (yy.ravel() * w + xx.ravel())
        xy = np.column_stack([origin[0] + (xx.ravel() + .5) * cell, origin[1] + (yy.ravel() + .5) * cell])
        inside = contains_xy(poly, xy[:, 0], xy[:, 1]) & np.isfinite(c["zmin"].ravel()[ids])
        xy, ids = xy[inside], ids[inside]
        if not len(ids):
            continue
        road_z, distance = road_height(xy, lane)
        value = union_occupancy(xy, road_z, components, clearance, deck)
        depth = np.maximum(0, lane["width_m"] / 2 - distance)
        occ = value["occupied"]
        masks["projected"][ids] = True
        for key in ["occupied", "solid", "surface", "deck", "uncertain"]:
            masks[key][ids[value[key]]] = True
        masks["deep_edge"][ids[occ & (depth > edge_tolerance)]] = True
        masks["deep05"][ids[occ & (depth > .5)]] = True
        if occ.any():
            max_depth = max(max_depth, float(depth[occ].max()))
            max_vertical = max(max_vertical, float(value["vertical_overlap"][occ].max()))
        blocked, unknown = _line_blockage(lane, components, clearance, cell, deck)
        total_blocked += blocked
        total_unknown += unknown
        rows.append({"lane_id": lane["lane_id"], "road_section_id": lane["road_section_id"], "road_class": lane["road_class"], "road_z_min_m": float(road_z.min()), "road_z_max_m": float(road_z.max()), "projected_area_m2": float(len(ids) * cell ** 2), "occupied_area_m2": float(occ.sum() * cell ** 2), "uncertain_area_m2": float(value["uncertain"].sum() * cell ** 2), "max_margin_intrusion_m": float(depth[occ].max()) if occ.any() else 0., "centerline_blocked_length_m": blocked, "centerline_unknown_length_m": unknown})
    areas = {key: float(value.sum() * cell ** 2) for key, value in masks.items()}
    maximum = max([x["centerline_blocked_length_m"] for x in rows], default=0.)
    if maximum >= centerline_threshold:
        status = f"{kind}_centerline_obstruction"
    elif areas["occupied"] >= 5 and areas["deep05"] >= 1:
        status = "substantial_edge_encroachment"
    elif areas["occupied"] >= 1 and max_depth > edge_tolerance:
        status = "edge_review"
    elif areas["occupied"] > 0:
        status = "minor_edge_contact"
    elif areas["uncertain"] > 0:
        status = "uncertain_open_mesh_clearance"
    elif areas["projected"] > 0:
        status = "vertically_clear_projection"
    else:
        status = "no_corridor_projection"
    result = {"id": meta["id"], "source_ids": meta.get("source_ids", []), "kind": meta.get("kind"), "entity_id": meta.get("entity_id"), "parent_source_id": meta.get("parent_source_id"), "status": status, "world_bounds": meta.get("world_bounds"), "clearance_m": clearance, "areas_m2": areas, "max_margin_intrusion_m": max_depth, "max_solid_vertical_overlap_m": max_vertical, "max_single_lane_obstructed_length_m": maximum, "summed_lane_obstructed_length_m": total_blocked, "summed_lane_unknown_length_m": total_unknown, "component_interval_union_applied": bool(meta.get("components")), "columns_sha256": _sha(cache / meta["columns"]) if columns_override is None else None, "lanes": sorted(rows, key=lambda x: (-x["centerline_blocked_length_m"], -x["occupied_area_m2"]))}
    if masks_dir is not None and (areas["occupied"] or areas["uncertain"]):
        masks_dir.mkdir(parents=True, exist_ok=True)
        path = masks_dir / (str(meta["id"]) + ".npz")
        np.savez_compressed(path, origin=origin, cell=cell, shape=np.asarray([h, w]), **{key: value.reshape(h, w) for key, value in masks.items()})
        result["mask_file"] = str(path.resolve())
    return result


def road_audit(cache_dir: str | Path, lane_geojson: str | Path, output_dir: str | Path, *, vehicle_clearance_m: float = 4.5, car_clearance_m: float = 2.0, pedestrian_clearance_m: float = 2.2, edge_tolerance_m: float = .25, centerline_block_threshold_m: float = 1.0, deck_tolerance_m: float = .15, save_masks: bool = True) -> dict[str, Any]:
    """Audit all cached placements against Vehicle and Pedestrian lane geometry.

    Returns three source-backed reports and writes their full per-instance
    results. Review candidates are not automatic move/delete instructions;
    pedestrian doorway, arcade and stair contacts need scene context.
    """
    cache, lane_path, output = Path(cache_dir), Path(lane_geojson), Path(output_dir)
    index_path = cache / "INDEX.json"
    index = json.loads(index_path.read_text())
    if not index.get("complete", True):
        raise ValueError("A completed geometry cache is required")
    records = index.get("records", [])
    if not records:
        raise ValueError("Geometry cache contains no instance records")
    for value in [vehicle_clearance_m, car_clearance_m, pedestrian_clearance_m, centerline_block_threshold_m]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Clearance heights and centerline threshold must be positive")
    groups, source_info = _load_lanes(lane_path)
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for name, kind, clearance in [("vehicle_4_5m", "vehicle", vehicle_clearance_m), ("car_2m", "vehicle", car_clearance_m), ("pedestrian_2_2m", "pedestrian", pedestrian_clearance_m)]:
        lanes = groups[kind]
        tree = STRtree([x["corridor"] for x in lanes]) if lanes else None
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        results = [_one(cache, meta, lanes, tree, clearance, kind, edge_tolerance_m, centerline_block_threshold_m, deck_tolerance_m, folder / "masks" if save_masks else None) for meta in records]
        results.sort(key=lambda x: (-x["max_single_lane_obstructed_length_m"], -x["areas_m2"]["occupied"]))
        counts = dict(Counter(x["status"] for x in results))
        affected = {lane["lane_id"] for r in results for lane in r["lanes"] if lane["occupied_area_m2"] > 0}
        elevated = {lane["lane_id"] for r in results for lane in r["lanes"] if lane["occupied_area_m2"] > 0 and lane["road_z_max_m"] > 3}
        summary = {"schema": "citycode.scene_road_clearance.v1", "placement_count": len(results), "lane_kind": kind, "lane_count": len(lanes), "clearance_m": clearance, "status_counts": counts, "counts_by_placement_kind": {k: dict(Counter(r["status"] for r in results if r.get("kind") == k)) for k in sorted({str(r.get("kind")) for r in results})}, "affected_unique_lanes": len(affected), "affected_unique_lanes_above_3m": len(elevated), "thresholds": {"edge_tolerance_m": edge_tolerance_m, "one_lane_centerline_block_m": centerline_block_threshold_m, "deck_tolerance_m": deck_tolerance_m, "navigation_volume_base_tolerance_m": .25}, "source_cache_index_sha256": _sha(index_path), "lane_geojson_sha256": _sha(lane_path), "source_lane_info": source_info, "method": "Actual mesh vertical crossings against each original lane XYZ and width; physical components combined by union. Odd/open columns remain uncertain unless an actual surface enters headroom.", "limitations": ["Raster sampling can change near-edge results by one cell; small edge contact is advisory.", "Road corridor occupation is a navigation clearance measure, not a rigid-body physics simulation or a direct road-deck intersection.", "A single original route must exceed the centreline threshold; lengths across overlapping turn paths are not summed for classification.", "Pedestrian stairs, doorways and arcades require context review; a flagged route does not automatically justify deleting a building.", "Any line with assumed Z cannot establish grade-separated clearance reliably."], "no_scene_changes": True}
        _write(folder / "SUMMARY.json", summary)
        _write(folder / "DETAILS.json", {"schema": summary["schema"], "instances": results})
        reports[name] = {"summary": summary, "summary_path": str((folder / "SUMMARY.json").resolve()), "details_path": str((folder / "DETAILS.json").resolve())}
    _write(output / "ROAD_AUDIT.json", {"schema": "citycode.scene_road_audit.v1", "reports": reports, "no_scene_changes": True})
    return reports


class RoadAudit:
    """Reusable no-write evaluator for proposed per-instance transforms.

    Supplied columns must already represent the candidate world mesh, and
    ``meta["footprint_world"]`` must describe that same candidate.
    """

    def __init__(self, cache_dir: str | Path, lane_geojson: str | Path, *, kind: str = "vehicle", clearance_m: float = 4.5, edge_tolerance_m: float = .25, centerline_block_threshold_m: float = 1., deck_tolerance_m: float = .15):
        if kind not in {"vehicle", "pedestrian"}:
            raise ValueError("kind must be vehicle or pedestrian")
        self.cache = Path(cache_dir)
        groups, self.source_info = _load_lanes(Path(lane_geojson))
        self.lanes = groups[kind]
        self.tree = STRtree([r["corridor"] for r in self.lanes]) if self.lanes else None
        self.kind = kind
        self.clearance_m = clearance_m
        self.edge_tolerance_m = edge_tolerance_m
        self.centerline_block_threshold_m = centerline_block_threshold_m
        self.deck_tolerance_m = deck_tolerance_m

    def analyze(self, meta: dict[str, Any], columns_dict: dict[str, Any] | None = None, component_columns: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return _one(self.cache, meta, self.lanes, self.tree, self.clearance_m, self.kind, self.edge_tolerance_m, self.centerline_block_threshold_m, self.deck_tolerance_m, None, columns_dict, component_columns)


# Descriptive and compact names are both available to callers adapting the
# original mesh-column audit functions.
road_z = road_height
occupancy = column_occupancy
