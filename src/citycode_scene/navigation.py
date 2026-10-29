"""Bounded navigation-clearance proposals with real-mesh confirmation.

The inexpensive search inverse-samples existing height columns. Every returned
proposal is then measured on newly transformed triangles and checked against
neighbours. Nothing is applied to the search state or written to the scene.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import box

from . import pairs
from .optimizer import PlacementSearch, contact_penalty
from .replay import delta_matrix
from .roads import RoadAudit, road_height, union_occupancy


def _context(search: PlacementSearch, ids: list[str], road_model: RoadAudit, max_shift: float) -> dict[str, Any]:
    bounds = np.asarray([search.current[x]["world_bounds"] for x in ids])
    lo, hi = bounds[:, 0, :2].min(0), bounds[:, 1, :2].max(0)
    cell = search.cell
    padding = max_shift + 2 * cell
    origin = np.floor((lo - padding) / cell) * cell
    dims = np.ceil((hi + padding - origin) / cell).astype(int) + 1
    w, h = map(int, dims)
    region = box(*lo - padding, *hi + padding)
    indices = road_model.tree.query(region, predicate="intersects") if road_model.tree is not None else []
    components = []
    for aid in ids:
        row = search.current[aid]
        components.extend(pairs.columns_for(c) for c in row.get("components", [row]))
    samples = []
    for j in indices:
        lane = road_model.lanes[int(j)]
        poly = lane["corridor"]
        b = poly.bounds
        x0, x1 = max(0, math.floor((b[0]-origin[0])/cell)), min(w, math.ceil((b[2]-origin[0])/cell))
        y0, y1 = max(0, math.floor((b[1]-origin[1])/cell)), min(h, math.ceil((b[3]-origin[1])/cell))
        if x0 >= x1 or y0 >= y1:
            continue
        from shapely import contains_xy
        yy, xx = np.mgrid[y0:y1, x0:x1]
        xy = np.column_stack([origin[0]+(xx.ravel()+.5)*cell, origin[1]+(yy.ravel()+.5)*cell])
        flat = yy.ravel()*w+xx.ravel()
        keep = contains_xy(poly, xy[:, 0], xy[:, 1])
        xy, flat = xy[keep], flat[keep]
        road_z, distance = road_height(xy, lane)
        xyz = np.asarray(lane["coordinates_xyz_m"], dtype=float)
        delta = np.diff(xyz, axis=0)
        lengths = np.linalg.norm(delta[:, :2], axis=1)
        cumulative = np.r_[0, lengths.cumsum()]
        total = cumulative[-1]
        n = max(1, int(math.ceil(total/cell)))
        step = total/n
        s = (np.arange(n)+.5)*step
        i = np.clip(np.searchsorted(cumulative, s, side="right")-1, 0, len(lengths)-1)
        t = (s-cumulative[i])/np.maximum(lengths[i], 1e-12)
        line = xyz[i]+delta[i]*t[:, None]
        line = line[(line[:, 0] >= origin[0]) & (line[:, 0] < origin[0]+w*cell) & (line[:, 1] >= origin[1]) & (line[:, 1] < origin[1]+h*cell)]
        samples.append({"xy": xy, "z": road_z, "flat": flat, "depth": np.maximum(0, lane["width_m"]/2-distance), "line": line, "step": step})
    return {"pivot": (lo+hi)/2, "components": components, "samples": samples, "shape": (h, w), "cell": cell}


def _proxy(ctx: dict[str, Any], xy_delta: np.ndarray, scale: float, road_model: RoadAudit) -> dict[str, float]:
    occupied = np.zeros(np.prod(ctx["shape"]), dtype=bool)
    deep = occupied.copy()
    maximum = 0.
    for sample in ctx["samples"]:
        inverse = ctx["pivot"]+(sample["xy"]-ctx["pivot"]-xy_delta)/scale
        v = union_occupancy(inverse, sample["z"], ctx["components"], road_model.clearance_m, road_model.deck_tolerance_m)
        occupied[sample["flat"][v["occupied"]]] = True
        deep[sample["flat"][v["occupied"] & (sample["depth"] > .5)]] = True
        line = sample["line"]
        inverse_line = ctx["pivot"]+(line[:, :2]-ctx["pivot"]-xy_delta)/scale
        v = union_occupancy(inverse_line, line[:, 2], ctx["components"], road_model.clearance_m, road_model.deck_tolerance_m)
        maximum = max(maximum, float(v["occupied"].sum()*sample["step"]))
    return {"max_single_lane_blocked_m": maximum, "occupied_area_m2": float(occupied.sum()*ctx["cell"]**2), "deep05_area_m2": float(deep.sum()*ctx["cell"]**2)}


def propose_navigation(search: PlacementSearch, ids: list[str], road_model: RoadAudit, max_shift_m: float = 2., min_scale: float = .95, *, finalist_count: int = 12, allowed_pair_area_m2: float = 5.) -> list[dict[str, Any]]:
    """Return reviewed-candidate data for a group, without applying proposals.

    ``ids`` is a rigid placement group. Translation is bounded relative to the
    original placement and scaling is equal on world X and Y; height is retained.
    The caller can lower min_scale to .925/.9 only after reviewing why the usual
    five-percent correction cannot restore clearance without worsening neighbours.
    """
    if not ids or any(aid not in search.current for aid in ids):
        raise ValueError("ids must identify existing placed assets")
    if not 0 < min_scale <= 1 or max_shift_m < 0 or finalist_count < 1:
        raise ValueError("Invalid navigation proposal limits")
    ctx = _context(search, ids, road_model, max_shift_m)
    before_rows = {aid: road_model.analyze(search.current[aid], pairs.columns_for(search.current[aid]), [pairs.columns_for(c) for c in search.current[aid].get("components", [])] or None) for aid in ids}
    before_pairs = search.evaluate({aid: search.current[aid] for aid in ids})
    old_pairs = {frozenset([r["asset_a"], r["asset_b"]]): r for r in before_pairs}
    shifts = [(0., 0.)]
    # A regular grid includes diagonal retreats that can be missed by searching
    # only cardinal directions or a bounding-box separating axis.
    step = max(.25, search.cell)
    for dx in np.arange(-max_shift_m, max_shift_m+step/2, step):
        for dy in np.arange(-max_shift_m, max_shift_m+step/2, step):
            if 1e-9 < math.hypot(dx, dy) <= max_shift_m+1e-9:
                shifts.append((float(dx), float(dy)))
    scales = sorted(set([1., (1.+min_scale)/2, min_scale]), reverse=True)
    coarse = []
    for scale in scales:
        for dx, dy in shifts:
            if scale == 1 and dx == dy == 0:
                continue
            delta = delta_matrix((dx, dy, 0), scale, ctx["pivot"])
            permitted = True
            for aid in ids:
                total = delta @ np.asarray(search.current[aid].get("_world_delta", np.eye(4)))
                centre = np.asarray(search.original[aid]["world_bounds"])[:, :2].mean(0)
                displacement = total[:2, :2]@centre+total[:2, 3]-centre
                if np.linalg.norm(displacement) > max_shift_m+1e-8 or min(total[0, 0], total[1, 1]) < min_scale-1e-8:
                    permitted = False
                    break
            if not permitted:
                continue
            proxy = _proxy(ctx, np.array([dx, dy]), scale, road_model)
            cost = math.hypot(dx, dy)+30*(1-scale)
            score = (proxy["max_single_lane_blocked_m"] >= road_model.centerline_block_threshold_m,
                     proxy["max_single_lane_blocked_m"]*30+proxy["deep05_area_m2"]*2+proxy["occupied_area_m2"]*.2+cost)
            coarse.append({"proxy": proxy, "score": score, "cost": cost, "translation_xy_m": [dx, dy], "scale_xy": scale, "delta": delta})
    coarse.sort(key=lambda r: r["score"])
    finalists = coarse[:finalist_count]
    # Include a few inexpensive edge-tolerance candidates: forcing zero contact
    # may cost a large translation while a small correction already restores a lane.
    for candidate in sorted(coarse, key=lambda r: r["cost"])[:max(2, finalist_count//4)]:
        if not any(candidate is row for row in finalists):
            finalists.append(candidate)
    results = []
    for candidate in finalists:
        changed = {aid: search.transform(aid, candidate["delta"]) for aid in ids}
        road_after = [road_model.analyze(row, pairs.columns_for(row), [pairs.columns_for(c) for c in row.get("components", [])] or None) for row in changed.values()]
        measured = search.evaluate(changed)
        extra = sum(max(0., contact_penalty(row, allowed_pair_area_m2)-contact_penalty(old_pairs.get(frozenset([row["asset_a"], row["asset_b"]])), allowed_pair_area_m2)) for row in measured)
        new_clear = [row for row in measured if row["classification"] == "clear_overlap" and old_pairs.get(frozenset([row["asset_a"], row["asset_b"]]), {}).get("classification") != "clear_overlap"]
        maximum = max([row["max_single_lane_obstructed_length_m"] for row in road_after], default=0.)
        results.append({"ids": list(ids), "pivot_xy_m": ctx["pivot"].tolist(), "translation_xy_m": candidate["translation_xy_m"], "scale_xy": candidate["scale_xy"], "world_delta_4x4": candidate["delta"].tolist(), "cost": candidate["cost"], "approximate_search": candidate["proxy"], "road_before": list(before_rows.values()), "road_after": road_after, "max_single_lane_blocked_after_m": maximum, "occupied_area_after_m2": sum(row["areas_m2"]["occupied"] for row in road_after), "other_contact_excess_increase_m2": extra, "new_clear_pairs": new_clear, "measured_pairs_after": measured, "verification": "actual_transformed_triangles_recast_with_physical_component_union", "no_scene_changes": True})
    results.sort(key=lambda r: (bool(r["new_clear_pairs"]), r["other_contact_excess_increase_m2"] > .5, r["max_single_lane_blocked_after_m"] >= road_model.centerline_block_threshold_m,
                                r["max_single_lane_blocked_after_m"]*30+r["occupied_area_after_m2"]*.2+r["cost"]+r["other_contact_excess_increase_m2"]*5))
    return results
