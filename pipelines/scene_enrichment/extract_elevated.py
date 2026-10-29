#!/usr/bin/env python3
"""Recover elevated road ribbons from the portable predicted E09 cache.

Run with the repository's CPU background environment::

    environment/.venv-background/bin/python pipelines/scene_enrichment/extract_elevated.py \
      --cache data/smallcity_p384_recovery_cache \
      --predictions data/smallcity_p384_predictions/masks \
      --output data/smallcity_p384_scene_enrichment

Only V7B class 4, observed E09 heights, and their numerical world transforms
define bridge geometry. No completed background, reference bridge, object
transform, or evaluation annotation is read. Camera ground references are
declared scale metadata, not recovered lower surfaces. This is a numerical
replay of historical predictions, not a new neural depth inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Config:
    grid_m: float = 0.75
    source_stride_px: int = 4
    closing_radius_m: float = 1.5
    max_hole_area_m2: float = 9.0
    min_component_area_m2: float = 45.0
    min_segment_length_m: float = 24.0
    spur_width_multiple: float = 1.5
    max_spur_length_m: float = 100.0
    profile_spacing_m: float = 4.0
    max_xy_adjustment_m: float = 1.5
    max_grade: float = 0.10
    max_height_adjustment_m: float = 2.0
    min_clearance_m: float = 1.5
    min_width_m: float = 3.0
    max_width_m: float = 45.0
    max_height_mad_m: float = 1.5
    height_layer_gap_m: float = 2.5
    semantic_height_tolerance_m: float = 2.5
    min_support_views: int = 2
    max_semantic_conflict_fraction: float = 0.5
    max_ground_search_m: float = 40.0
    module_length_m: float = 10.0
    module_width_m: float = 10.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Prediction inputs require relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input escapes prediction cache: " + relative)
    if not path.is_file() or (root / relative).is_symlink():
        raise ValueError("Missing or linked prediction input: " + relative)
    return path


def validate_config(c: Config) -> None:
    for k, v in asdict(c).items():
        if not math.isfinite(float(v)) or v <= 0:
            raise ValueError("Configuration must be finite and positive: " + k)
    if c.min_width_m >= c.max_width_m:
        raise ValueError("Width bounds are reversed")
    if c.max_semantic_conflict_fraction > 1:
        raise ValueError("Semantic conflict is a fraction")


def audit_sources(cache: Path, predictions: Path | None, c: Config) -> tuple[dict, list, dict]:
    """Check inventories and every consumed geometry file against frozen hashes."""
    inventory = read_json(cache / "INPUTS.json")
    if (inventory.get("schema") != "city_pipeline.cached_input_bundle.v2"
            or inventory.get("uses_ground_truth") is not False
            or inventory.get("status") != "complete"):
        raise ValueError("A complete prediction-only v2 cache is required")
    files = {row["path"]: row for row in inventory["files"]}
    consumed: dict[str, dict] = {}

    def verified(relative: str) -> Path:
        if relative in consumed:
            return cache / relative
        if relative not in files:
            raise ValueError("Unlisted input: " + relative)
        p = contained(cache, relative)
        row = files[relative]
        if p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
            raise ValueError("Input checksum changed: " + relative)
        consumed[relative] = {"path": relative, "sha256": row["sha256"], "bytes": row["bytes"]}
        return p

    e09 = read_json(verified("source/e09_full_pipeline/manifest.json"))
    if e09.get("schema") != "city_pipeline.cached_e09_surfaces.v1":
        raise ValueError("Unsupported E09 prediction surface schema")
    rows = sorted(e09["samples"], key=lambda s: s["id"])
    if len(rows) != inventory["sample_count"] or len({s["id"] for s in rows}) != len(rows):
        raise ValueError("E09 sample inventory is incomplete or duplicated")
    visibility = read_json(verified("visibility_inputs.json"))
    if visibility.get("uses_ground_truth") is not False:
        raise ValueError("Undeclared depth lineage")
    source_info = []
    for row in rows:
        outputs = row["outputs"]
        for role in ("surface_z", "background_semantic_ids", "final_ids", "provenance"):
            verified(outputs[role])
        raster = row["rasterization"]
        m = np.asarray(raster["pixel_center_index_to_ue_world_xy_3x3"], dtype=float)
        if m.shape != (3, 3) or not np.isfinite(m).all() or not np.allclose(m[2], [0, 0, 1]):
            raise ValueError("Invalid E09 world transform")
        if abs(np.linalg.det(m[:2, :2])) < 1e-8:
            raise ValueError("Singular E09 world transform")
        pixel_step = float(np.linalg.norm(m[:2, :2], axis=0).max())
        if c.source_stride_px * pixel_step > c.grid_m:
            raise ValueError("Source sampling is sparser than the output grid")
        camera_rel = visibility["samples"][row["id"]]["camera"]
        camera = read_json(verified(camera_rel))
        source_info.append({"sample_id": row["id"], "ground_reference_z_m": camera.get("ground_reference_z_m"),
                            "depth_stage": visibility.get("depth_stage"),
                            "surface_z": outputs["surface_z"],
                            "semantic_ids": outputs["background_semantic_ids"]})
    segmentation = {"provided": False}
    if predictions:
        package = read_json(predictions / "prediction_manifest.json")
        if package.get("gt_labels_included") is not False:
            raise ValueError("Segmentation package does not exclude reference labels")
        v7b_path = predictions / "structure_v7b/manifest.json"
        v7b = read_json(v7b_path)
        if v7b.get("provenance", {}).get("gt_labels_included") is not False:
            raise ValueError("V7B provenance does not exclude reference labels")
        if {r["id"] for r in v7b["samples"]} != {r["id"] for r in rows}:
            raise ValueError("V7B and E09 sample IDs differ")
        # These establish the semantic predictor provenance. E09 is already the
        # frozen downstream rasterization; raw masks are not substituted for it.
        for r in v7b["samples"]:
            p = contained(v7b_path.parent, r["result"])
            if digest(p) != r["result_sha256"]:
                raise ValueError("Changed historical V7B prediction: " + r["id"])
        segmentation = {"provided": True, "manifest_sha256": digest(v7b_path),
                        "prediction_count": len(v7b["samples"]),
                        "model": v7b.get("generation_model"),
                        "provenance": v7b.get("provenance"),
                        "raw_mask_checksums_verified": True,
                        "relation_to_e09": "same sample IDs and published V7B stage; E09 intermediate mask hashes were not retained"}
    audit = {"schema": "citycode.elevated_source_audit.v1", "uses_ground_truth": False,
             "geometry_source": "cached E09 predicted observed BEV surface and V7B class 4",
             "coordinate_system": {"world": "UE XYZ", "units": "metres", "handedness": "left", "up": "Z"},
             "cache_inventory_sha256": digest(cache / "INPUTS.json"),
             "source_inventory_scope": inventory.get("scope"),
             "source_content": inventory.get("source_content"),
             "sample_count": len(rows), "consumed_files": list(consumed.values()),
             "segmentation": segmentation,
             "limitations": ["Historical raw neural depth files are absent; this run replays calibrated prediction surfaces.",
                              "E09 is a top-visible 2.5D surface; overlapping decks cannot be separated reliably.",
                              "Prediction calibration uses explicit road/water/hardscape plane priors; it is not metric depth recovered without priors.",
                              "Class 4 is excluded from calibration anchors; semantics can still contain false positives.",
                              "Local lower-ground height is estimated from nearby predicted ground semantics and may be occluded.",
                              "Completed background height layers and GT bridge transforms are excluded."],
             "source_samples": source_info}
    return audit, rows, visibility


def grouped_median(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized median per integer key, including even group sizes."""
    if not len(keys):
        return np.empty(0, np.int64), np.empty(0, np.float32)
    order = np.lexsort((values, keys))
    sorted_keys, sorted_values = keys[order], values[order]
    unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    a = starts + (counts - 1) // 2
    b = starts + counts // 2
    return unique, ((sorted_values[a] + sorted_values[b]) * 0.5).astype(np.float32)


def cluster_height_layers(keys: np.ndarray, z: np.ndarray, views: np.ndarray,
                          gap_m: float) -> dict:
    """Cluster each XY cell by observed Z gaps, never median across deck layers.

    Every input is already one median per view/cell. Primary-layer choice is
    maximum independent view support, breaking ties by greater Z. Alternatives
    and sample membership are retained explicitly for downstream 3D reasoning.
    """
    order = np.lexsort((z, keys))
    k, values, v = keys[order], z[order], views[order]
    starts_new = np.r_[True, (k[1:] != k[:-1]) | (np.diff(values) > gap_m)]
    group = np.cumsum(starts_new)-1
    starts = np.flatnonzero(starts_new)
    counts = np.diff(np.r_[starts, len(k)])
    middle_a = starts + (counts-1)//2
    middle_b = starts + counts//2
    medians = ((values[middle_a]+values[middle_b])*.5).astype(np.float32)
    _, mad = grouped_median(group, np.abs(values-medians[group]))
    cell_keys = k[starts]
    # First sort by cell, then decreasing support and decreasing height.
    ranking = np.lexsort((-medians, -counts, cell_keys))
    ranked_keys = cell_keys[ranking]
    primary = ranking[np.r_[True, ranked_keys[1:] != ranked_keys[:-1]]]
    primary_group = np.zeros(len(starts), bool)
    primary_group[primary] = True
    supported_layer_count = np.bincount(np.searchsorted(np.unique(cell_keys), cell_keys),
                                       weights=(counts >= 2).astype(int)).astype(np.uint16)
    return {"layer_cell_index": cell_keys, "layer_height_m": medians,
            "layer_height_mad_m": mad, "layer_view_count": counts.astype(np.uint16),
            "layer_member_start": starts, "layer_member_count": counts,
            "layer_member_sample_index": v, "layer_is_primary": primary_group,
            "layer_primary_indices": primary, "layer_sorted_evidence_index": order,
            "primary_member_mask": primary_group[group],
            "supported_layer_count_per_observed_cell": supported_layer_count,
            "observed_cell_index": np.unique(cell_keys)}


def compatible_conflict_counts(shape: tuple, height: np.ndarray, selected_keys: np.ndarray,
                               selected_views: np.ndarray, other_keys: np.ndarray,
                               other_z: np.ndarray, other_views: np.ndarray,
                               sample_count: int, tolerance_m: float) -> np.ndarray:
    compatible = np.isfinite(height.ravel()[other_keys]) & (np.abs(other_z-height.ravel()[other_keys]) <= tolerance_m)
    same_view = np.isin(other_keys * sample_count + other_views,
                        selected_keys * sample_count + selected_views)
    conflicts = np.zeros(shape, np.uint16)
    np.add.at(conflicts.ravel(), other_keys[compatible & ~same_view], 1)
    return conflicts


def fuse_prediction_grid(cache: Path, rows: list, c: Config) -> tuple[dict, list]:
    footprints = np.concatenate([s["rasterization"]["raster_footprint_world_xy_m"] for s in rows])
    origin = np.floor(footprints.min(axis=0) / c.grid_m) * c.grid_m - 2 * c.grid_m
    upper = np.ceil(footprints.max(axis=0) / c.grid_m) * c.grid_m + 2 * c.grid_m
    nx, ny = map(int, np.ceil((upper - origin) / c.grid_m))
    shape = (ny, nx)
    view_counts = np.zeros(shape, np.uint16)
    known_counts = np.zeros(shape, np.uint16)
    elevated_keys, elevated_z, elevated_view = [], [], []
    ground_keys, ground_z = [], []
    other_keys, other_z, other_view = [], [], []
    stats = []
    stride = c.source_stride_px
    for view_index, row in enumerate(rows):
        outputs = row["outputs"]
        labels_full = np.asarray(Image.open(cache / outputs["background_semantic_ids"]))
        labels = labels_full[::stride, ::stride]
        z_full = np.load(cache / outputs["surface_z"], mmap_mode="r")
        if labels_full.shape != z_full.shape:
            raise ValueError("E09 semantic/height dimensions differ")
        final_full = np.asarray(Image.open(cache / outputs["final_ids"]))
        provenance_full = np.asarray(Image.open(cache / outputs["provenance"]))
        # A finite class-4 height with no building ID is observed E09 geometry.
        full_mask = labels_full == 4
        overlap = int(np.count_nonzero(full_mask & (final_full > 0)))
        filled = int(np.count_nonzero(full_mask & (provenance_full != 0)))
        if overlap or filled:
            raise ValueError("Class-4 input contains building labels or filled geometry")
        z = np.asarray(z_full[::stride, ::stride])
        iy, ix = np.nonzero(np.isfinite(z) & (labels > 0))
        m = np.asarray(row["rasterization"]["pixel_center_index_to_ue_world_xy_3x3"])
        px, py = ix * stride, iy * stride
        x = m[0, 0] * px + m[0, 1] * py + m[0, 2]
        y = m[1, 0] * px + m[1, 1] * py + m[1, 2]
        gx = np.floor((x - origin[0]) / c.grid_m).astype(np.int64)
        gy = np.floor((y - origin[1]) / c.grid_m).astype(np.int64)
        keys = gy * nx + gx
        semantic = labels[iy, ix]
        values = z[iy, ix]
        known_counts.ravel()[np.unique(keys)] += 1
        elevated = semantic == 4
        ek, ez = grouped_median(keys[elevated], values[elevated])
        elevated_keys.append(ek)
        elevated_z.append(ez)
        elevated_view.append(np.full(len(ek), view_index, np.uint16))
        view_counts.ravel()[ek] += 1
        ground = np.isin(semantic, (1, 3))
        gk, gz = grouped_median(keys[ground], values[ground])
        ground_keys.append(gk)
        ground_z.append(gz)
        ok, oz = grouped_median(keys[~elevated], values[~elevated])
        other_keys.append(ok)
        other_z.append(oz)
        other_view.append(np.full(len(ok), view_index, np.uint16))
        finite_class4 = z_full[full_mask]
        finite_class4 = finite_class4[np.isfinite(finite_class4)]
        stats.append({"sample_id": row["id"], "class4_pixels": int(full_mask.sum()),
                      "finite_class4_pixels": int(len(finite_class4)),
                      "sampled_elevated_cells": len(ek), "class4_building_overlap": overlap,
                      "class4_filled_pixels": filled,
                      "class4_z_p05_p50_p95_m": np.quantile(finite_class4, [.05, .5, .95]).tolist() if len(finite_class4) else []})
    all_keys, all_z = np.concatenate(elevated_keys), np.concatenate(elevated_z)
    all_views = np.concatenate(elevated_view)
    layers = cluster_height_layers(all_keys, all_z, all_views, c.height_layer_gap_m)
    primary = layers["layer_primary_indices"]
    unique, medians = layers["layer_cell_index"][primary], layers["layer_height_m"][primary]
    height = np.full(shape, np.nan, np.float32)
    height.ravel()[unique] = medians
    mad = np.full(shape, np.nan, np.float32)
    mad.ravel()[unique] = layers["layer_height_mad_m"][primary]
    total_view_counts = view_counts.copy()
    view_counts[:] = 0
    view_counts.ravel()[unique] = layers["layer_view_count"][primary]
    layer_count = np.zeros(shape, np.uint16)
    layer_count.ravel()[layers["observed_cell_index"]] = layers["supported_layer_count_per_observed_cell"]
    order = layers["layer_sorted_evidence_index"]
    member_mask = layers["primary_member_mask"]
    selected_keys = all_keys[order][member_mask]
    selected_z = all_z[order][member_mask]
    selected_views = all_views[order][member_mask]
    ok, oz, ov = np.concatenate(other_keys), np.concatenate(other_z), np.concatenate(other_view)
    # Different class labels in the same coarse boundary cell of the same
    # view cannot create a second independent contradictory view.
    conflict_counts = compatible_conflict_counts(shape, height, selected_keys, selected_views,
                                                 ok, oz, ov, len(rows), c.semantic_height_tolerance_m)
    gk, gz = grouped_median(np.concatenate(ground_keys), np.concatenate(ground_z))
    ground = np.full(shape, np.nan, np.float32)
    ground.ravel()[gk] = gz
    grid = {"origin_xy_m": origin, "cell_size_m": c.grid_m, "shape": shape,
            "height_m": height, "height_mad_m": mad, "elevated_view_count": view_counts,
            "known_view_count": known_counts, "ground_m": ground,
            "compatible_conflict_view_count": conflict_counts,
            "supported_height_layer_count": layer_count,
            "all_layer_elevated_view_count": total_view_counts,
            "evidence_cell_index": selected_keys,
            "evidence_view_index": selected_views,
            "evidence_height_m": selected_z,
            **{k: v for k, v in layers.items() if k.startswith("layer_") and k not in ("layer_primary_indices", "layer_sorted_evidence_index")}}
    return grid, stats


def disk(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return x*x + y*y <= radius*radius


def clean_mask(mask: np.ndarray, c: Config) -> tuple[np.ndarray, dict]:
    closed = ndimage.binary_closing(mask, structure=disk(max(1, int(c.closing_radius_m / c.grid_m))))
    # Closing may alter only cells within the configured radius of evidence.
    closed |= mask
    holes = ndimage.binary_fill_holes(closed) & ~closed
    labels, _ = ndimage.label(holes)
    sizes = np.bincount(labels.ravel())
    small = sizes * c.grid_m**2 <= c.max_hole_area_m2
    small[0] = False
    closed |= small[labels]
    labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    keep = sizes * c.grid_m**2 >= c.min_component_area_m2
    keep[0] = False
    result = keep[labels]
    return result, {"observed_cells": int(mask.sum()), "retained_cells": int(result.sum()),
                    "bounded_added_cells": int(np.count_nonzero(result & ~mask)),
                    "discarded_cells": int(np.count_nonzero(mask & ~result)),
                    "connected_components_before_area_filter": n,
                    "retained_components": int(keep.sum()),
                    "closing_radius_m": c.closing_radius_m, "max_hole_area_m2": c.max_hole_area_m2}


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen topology-preserving thinning, deterministic simultaneous edits."""
    result = np.pad(mask.astype(bool), 1)
    while True:
        changed = False
        for second in (False, True):
            p2, p3, p4 = result[:-2, 1:-1], result[:-2, 2:], result[1:-1, 2:]
            p5, p6, p7 = result[2:, 2:], result[2:, 1:-1], result[2:, :-2]
            p8, p9 = result[1:-1, :-2], result[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            n = sum(a.astype(np.uint8) for a in neighbors)
            transitions = sum((~neighbors[i] & neighbors[(i + 1) % 8]).astype(np.uint8) for i in range(8))
            a = (p2 & p4 & p8) if second else (p2 & p4 & p6)
            b = (p2 & p6 & p8) if second else (p4 & p6 & p8)
            remove = result[1:-1, 1:-1] & (n >= 2) & (n <= 6) & (transitions == 1) & ~a & ~b
            if remove.any():
                result[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return result[1:-1, 1:-1]


def skeleton_chains(skeleton: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
    """Compress the pixel graph, preserving cycles and collapsing junction clumps."""
    points = [tuple(p) for p in np.argwhere(skeleton)]
    pointset = set(points)
    adjacency = {}
    for y, x in points:
        near = []
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = y + dy, x + dx
            if q in pointset:
                # Remove redundant diagonals when an orthogonal connection exists.
                if dy and dx and ((y + dy, x) in pointset or (y, x + dx) in pointset):
                    continue
                near.append(q)
        adjacency[(y, x)] = sorted(near)
    junction = {p for p in points if len(adjacency[p]) >= 3}
    node_map, node_points = {}, []
    remaining = set(junction)
    while remaining:
        first = min(remaining)
        stack, component = [first], []
        remaining.remove(first)
        while stack:
            p = stack.pop()
            component.append(p)
            for q in adjacency[p]:
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        index = len(node_points)
        node_points.append(component)
        node_map.update({p: index for p in component})
    for p in points:
        if len(adjacency[p]) != 2 and p not in node_map:
            node_map[p] = len(node_points)
            node_points.append([p])
    visited = set()
    chains = []
    def edge(a, b):
        return (a, b) if a < b else (b, a)
    for p in sorted(node_map):
        for q in adjacency[p]:
            if edge(p, q) in visited or node_map.get(q) == node_map[p]:
                continue
            chain = [p]
            previous, current = p, q
            visited.add(edge(p, q))
            while True:
                chain.append(current)
                if current in node_map:
                    break
                options = [k for k in adjacency[current] if k != previous]
                if not options:
                    break
                nxt = options[0]
                if edge(current, nxt) in visited:
                    break
                visited.add(edge(current, nxt))
                previous, current = current, nxt
            if len(chain) > 1:
                chains.append(np.asarray(chain, float))
    # Closed rings have no graph endpoint. Start them at their lexicographic minimum.
    for p in points:
        for q in adjacency[p]:
            if edge(p, q) in visited or (p in junction and q in junction):
                continue
            chain, previous, current = [p], p, q
            visited.add(edge(p, q))
            while True:
                chain.append(current)
                options = [k for k in adjacency[current] if k != previous and edge(current, k) not in visited]
                if not options:
                    break
                nxt = options[0]
                visited.add(edge(current, nxt))
                previous, current = current, nxt
                if current == p:
                    chain.append(p)
                    break
            if len(chain) > 2:
                chains.append(np.asarray(chain, float))
    nodes = [{"pixel_yx": np.mean(ps, axis=0).tolist(),
              "kind": "junction" if any(p in junction for p in ps) else "endpoint",
              "pixel_count": len(ps), "pixel_members_yx": [list(p) for p in ps]} for ps in node_points]
    return chains, nodes


def prune_short_leaves(skeleton: np.ndarray, c: Config, width: np.ndarray | None = None,
                       support: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Remove short dangling branches before they split a supported long road.

    This deletes only leaf chains already below the published minimum segment
    length. Interior connections and cycles are preserved; no new pixels or
    world coordinates are introduced.
    """
    result = skeleton.copy()
    removed = 0
    while True:
        chains, nodes = skeleton_chains(result)
        endpoints = {tuple(n["pixel_yx"]) for n in nodes if n["kind"] == "endpoint"}
        discard = []
        for chain in chains:
            start_leaf, end_leaf = tuple(chain[0]) in endpoints, tuple(chain[-1]) in endpoints
            if not (start_leaf or end_leaf):
                continue
            length = np.linalg.norm(np.diff(chain, axis=0), axis=1).sum() * c.grid_m
            threshold = c.min_segment_length_m
            if width is not None:
                attachment = chain[-1] if start_leaf else chain[0]
                root_width = float(width[tuple(attachment.astype(int))])
                pixels = chain.astype(int)
                median_width = float(np.median(width[pixels[:, 0], pixels[:, 1]]))
                repeated_support = float(np.mean(support[pixels[:, 0], pixels[:, 1]] >= c.min_support_views)) if support is not None else 0.
                # A distinct narrow branch with sustained multiview evidence
                # survives even near a much wider trunk. Broad tapering teeth
                # of the same road ribbon do not become junctions.
                distinct_branch = (repeated_support >= .8 and length >= 2*median_width
                                   and median_width < .45*root_width)
                if not distinct_branch:
                    threshold = max(threshold, min(c.max_spur_length_m, c.spur_width_multiple*root_width))
            if length >= threshold:
                continue
            points = chain if start_leaf and end_leaf else chain[:-1] if start_leaf else chain[1:]
            discard.extend(points.astype(int).tolist())
        if not discard:
            return result, removed
        pixels = np.asarray(discard, int)
        before = int(result.sum())
        result[pixels[:, 0], pixels[:, 1]] = False
        difference = before - int(result.sum())
        removed += difference
        if not difference:
            return result, removed


def prune_local_cycles(skeleton: np.ndarray, width: np.ndarray, support: np.ndarray,
                        c: Config) -> tuple[np.ndarray, int]:
    """Prefer supported wide center paths around small holes inside one ribbon.

    A short edge is removed only if a stronger already accepted path connects
    the same graph nodes within a width-dependent local distance. Long ramp
    loops remain intact. This is not a global spanning tree of the road net.
    """
    chains, nodes = skeleton_chains(skeleton)
    node_map = {tuple(p): i for i, n in enumerate(nodes) for p in n["pixel_members_yx"]}
    edges = []
    for chain in chains:
        a, b = node_map.get(tuple(chain[0])), node_map.get(tuple(chain[-1]))
        if a is None or b is None:
            continue
        pixels = chain.astype(int)
        w = float(np.median(width[pixels[:, 0], pixels[:, 1]]))
        count = float(np.mean(support[pixels[:, 0], pixels[:, 1]]))
        length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()*c.grid_m)
        score = w * (1+math.log1p(count))
        edges.append((score, length, a, b, chain, w))
    adjacency: dict[int, list] = {}
    discard = []
    for score, length, a, b, chain, w in sorted(edges, key=lambda e: (-e[0], -e[1], e[2], e[3])):
        limit = min(150., max(24., 3*w))
        shortcut = False
        if length <= min(c.max_spur_length_m, max(24., 2*w)):
            queue, distances = [(0., a)], {a: 0.}
            while queue:
                d, node = heapq.heappop(queue)
                if d > limit:
                    break
                if node == b:
                    shortcut = True
                    break
                if d != distances[node]:
                    continue
                for nxt, cost in adjacency.get(node, []):
                    value = d+cost
                    if value <= limit and value < distances.get(nxt, math.inf):
                        distances[nxt] = value
                        heapq.heappush(queue, (value, nxt))
        if shortcut:
            discard.extend(chain[1:-1].astype(int).tolist())
        else:
            adjacency.setdefault(a, []).append((b, length))
            adjacency.setdefault(b, []).append((a, length))
    result = skeleton.copy()
    if discard:
        points = np.asarray(discard, int)
        result[points[:, 0], points[:, 1]] = False
    return result, int(skeleton.sum()-result.sum())


def sample_field(field: np.ndarray, xy: np.ndarray, grid: dict, order: int = 0) -> np.ndarray:
    ij = (xy - grid["origin_xy_m"]) / grid["cell_size_m"] - 0.5
    return ndimage.map_coordinates(field, [ij[:, 1], ij[:, 0]], order=order, mode="nearest")


def resample_line(xy: np.ndarray, spacing: float) -> np.ndarray:
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-8]
    xy, lengths = xy[keep], lengths[keep]
    if len(xy) < 2:
        return xy
    targets = np.linspace(0, lengths[-1], max(2, int(math.ceil(lengths[-1] / spacing)) + 1))
    return np.column_stack([np.interp(targets, lengths, xy[:, k]) for k in (0, 1)])


def fit_bounded_height(z: np.ndarray, distance: np.ndarray, weights: np.ndarray,
                       max_grade: float) -> np.ndarray:
    """Smooth a profile then apply an exact Lipschitz envelope projection.

    The midpoint of lower/upper McShane envelopes is slope-bounded in one
    dimension. It is deterministic and linear time; excessive corrections
    are exported as uncertainty and prevent automatic module placement.
    """
    n = len(z)
    # Short incompatible spikes must not force a Lipschitz envelope to shift
    # an entire well-supported long road. Retain original observations in the
    # exported profile: their correction still triggers withholding locally.
    local_median = ndimage.median_filter(z, size=5, mode="nearest")
    robust_z = np.where(np.abs(z-local_median) > 4.0, local_median, z)
    w = np.maximum(weights, 0.05)
    if n >= 3:
        d2 = diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
        smooth = spsolve(diags(w) + 4.0 * (d2.T @ d2), w * robust_z)
    else:
        smooth = robust_z.copy()
    lower, upper = smooth.copy(), smooth.copy()
    for k in range(1, n):
        bound = max_grade * (distance[k] - distance[k-1])
        lower[k] = max(lower[k], lower[k-1] - bound)
        upper[k] = min(upper[k], upper[k-1] + bound)
    for k in range(n-2, -1, -1):
        bound = max_grade * (distance[k+1] - distance[k])
        lower[k] = max(lower[k], lower[k+1] - bound)
        upper[k] = min(upper[k], upper[k+1] + bound)
    return (lower + upper) * 0.5


def uncertainty_reasons(node: dict, c: Config) -> list[str]:
    reasons = []
    if node["support_views"] < c.min_support_views:
        reasons.append("single_view_support")
    if node["semantic_conflict_fraction"] > c.max_semantic_conflict_fraction:
        reasons.append("semantic_disagreement")
    if node["height_mad_m"] > c.max_height_mad_m:
        reasons.append("height_disagreement")
    if node["height_adjustment_m"] > c.max_height_adjustment_m:
        reasons.append("height_fit_exceeds_evidence_tolerance")
    if node["clearance_proxy_m"] < c.min_clearance_m:
        reasons.append("insufficient_predicted_elevation")
    if not c.min_width_m <= node["width_m"] <= c.max_width_m:
        reasons.append("width_outside_deck_range")
    if node["evidence_distance_m"] > c.closing_radius_m:
        reasons.append("height_support_gap")
    if node["ground_distance_m"] > c.max_ground_search_m:
        reasons.append("lower_ground_not_observed_nearby")
    if node.get("supported_height_layers", 0) > 1:
        reasons.append("multiple_supported_height_layers_need_3d_topology")
    return reasons


def recover_ribbons(grid: dict, c: Config) -> tuple[list, dict, list]:
    observed = grid["elevated_view_count"] > 0
    mask, cleanup = clean_mask(observed, c)
    grid["retained_mask"] = mask
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return [], cleanup, []
    sl = (slice(max(0, ys.min()-2), min(mask.shape[0], ys.max()+3)),
          slice(max(0, xs.min()-2), min(mask.shape[1], xs.max()+3)))
    widths = 2 * ndimage.distance_transform_edt(mask, sampling=c.grid_m)
    skeleton = thin(mask[sl])
    skeleton, removed_leaf_pixels = prune_short_leaves(skeleton, c, widths[sl], grid["elevated_view_count"][sl])
    skeleton, removed_cycle_pixels = prune_local_cycles(skeleton, widths[sl], grid["elevated_view_count"][sl], c)
    skeleton, final_leaf_pixels = prune_short_leaves(skeleton, c, widths[sl], grid["elevated_view_count"][sl])
    removed_leaf_pixels += final_leaf_pixels
    chains, nodes = skeleton_chains(skeleton)
    offset = np.array([sl[0].start, sl[1].start])
    chains = [a + offset for a in chains]
    for n in nodes:
        n.pop("pixel_members_yx", None)
        p = (np.asarray(n.pop("pixel_yx")) + offset)[::-1]
        n["position_xy_m"] = (grid["origin_xy_m"] + (p + 0.5) * c.grid_m).tolist()
    distance, nearest = ndimage.distance_transform_edt(~observed, sampling=c.grid_m, return_indices=True)
    heights = grid["height_m"][tuple(nearest)]
    mads = grid["height_mad_m"][tuple(nearest)]
    counts = grid["elevated_view_count"][tuple(nearest)]
    known = grid["known_view_count"][tuple(nearest)]
    conflicts = grid.get("compatible_conflict_view_count", np.zeros(mask.shape, np.uint16))[tuple(nearest)]
    layer_counts = grid.get("supported_height_layer_count", np.zeros(mask.shape, np.uint16))[tuple(nearest)]
    ground_valid = np.isfinite(grid["ground_m"])
    if not ground_valid.any():
        raise ValueError("No predicted ground/hardscape is available to check elevation")
    ground_distance, ground_nearest = ndimage.distance_transform_edt(~ground_valid, sampling=c.grid_m, return_indices=True)
    lower = grid["ground_m"][tuple(ground_nearest)]
    evidence_order = np.argsort(grid["evidence_cell_index"], kind="stable")
    sorted_evidence_keys = grid["evidence_cell_index"][evidence_order]
    sorted_evidence_views = grid["evidence_view_index"][evidence_order]
    junction_xy = np.asarray([n["position_xy_m"] for n in nodes if n["kind"] == "junction"])
    ribbons = []
    skipped_short = 0
    for chain in chains:
        xy = grid["origin_xy_m"] + (chain[:, ::-1] + 0.5) * c.grid_m
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if length < c.min_segment_length_m:
            skipped_short += 1
            continue
        xy = resample_line(xy, c.profile_spacing_m)
        smooth = ndimage.gaussian_filter1d(xy, 1.0, axis=0, mode="nearest")
        correction = smooth - xy
        correction *= np.minimum(1.0, c.max_xy_adjustment_m / np.maximum(1e-9, np.linalg.norm(correction, axis=1)))[:, None]
        correction[[0, -1]] = 0
        candidate = xy + correction
        inside = sample_field(mask.astype(np.uint8), candidate, grid) > 0
        xy[inside] = candidate[inside]
        along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        raw_z = sample_field(heights, xy, grid)
        raw_mad = sample_field(mads, xy, grid)
        view_count = sample_field(counts, xy, grid).astype(int)
        known_count = sample_field(known, xy, grid).astype(int)
        conflict_count = sample_field(conflicts, xy, grid).astype(int)
        height_layers = sample_field(layer_counts, xy, grid).astype(int)
        z = fit_bounded_height(raw_z, along, view_count / (1 + raw_mad), c.max_grade)
        width = sample_field(widths, xy, grid)
        # Distance to edge is a conservative inscribed ribbon width; preserve
        # raw width so an overly wide junction is refused instead of clamped.
        width = ndimage.median_filter(width, size=3, mode="nearest")
        support_distance = sample_field(distance, xy, grid)
        ground_z = sample_field(lower, xy, grid)
        ground_d = sample_field(ground_distance, xy, grid)
        pixel = np.floor((xy-grid["origin_xy_m"])/c.grid_m).astype(int)
        nearest_cell = nearest[:, pixel[:, 1], pixel[:, 0]]
        evidence_keys = nearest_cell[0] * mask.shape[1] + nearest_cell[1]
        profile = []
        for k in range(len(xy)):
            item = {"s_m": round(float(along[k]), 6),
                    "position_m": [round(float(xy[k, 0]), 6), round(float(xy[k, 1]), 6), round(float(z[k]), 6)],
                    "observed_z_m": round(float(raw_z[k]), 6),
                    "width_m": round(float(width[k]), 6), "support_views": int(view_count[k]),
                    "known_semantic_views": int(known_count[k]),
                    "compatible_conflicting_views": int(conflict_count[k]),
                    "semantic_conflict_fraction": round(float(conflict_count[k] / max(1, conflict_count[k]+view_count[k])), 6),
                    "supported_height_layers": int(height_layers[k]),
                    "height_mad_m": round(float(raw_mad[k]), 6),
                    "height_adjustment_m": round(float(abs(z[k]-raw_z[k])), 6),
                    "evidence_distance_m": round(float(support_distance[k]), 6),
                    "nearby_ground_z_m": round(float(ground_z[k]), 6),
                    "ground_distance_m": round(float(ground_d[k]), 6),
                    "clearance_proxy_m": round(float(z[k]-ground_z[k]), 6)}
            left = np.searchsorted(sorted_evidence_keys, evidence_keys[k], side="left")
            right = np.searchsorted(sorted_evidence_keys, evidence_keys[k], side="right")
            item["support_sample_indices"] = sorted_evidence_views[left:right].astype(int).tolist()
            item["uncertainty_reasons"] = uncertainty_reasons(item, c)
            if len(junction_xy) and np.linalg.norm(junction_xy-xy[k], axis=1).min() <= max(3.0, width[k]*0.5):
                item["uncertainty_reasons"].append("junction_requires_connection_solution")
            xy_failures = {"single_view_support", "semantic_disagreement", "width_outside_deck_range", "height_support_gap"}
            height_failures = {"height_disagreement", "height_fit_exceeds_evidence_tolerance", "insufficient_predicted_elevation", "lower_ground_not_observed_nearby"}
            item["evidence_state"] = ("supported_geometry" if not item["uncertainty_reasons"] else
                                      "uncertain_geometry" if xy_failures.intersection(item["uncertainty_reasons"]) else
                                      "xy_supported_height_pending" if height_failures.intersection(item["uncertainty_reasons"]) else
                                      "xy_supported_topology_pending")
            profile.append(item)
        positions = np.asarray([p["position_m"] for p in profile])
        grades = np.diff(positions[:, 2]) / np.maximum(1e-9, np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1))
        ribbons.append({"segment_id": f"PRED_ELEVATED_{len(ribbons)+1:05d}",
                        "length_m": round(float(along[-1]), 6),
                        "width_median_m": round(float(np.median(width)), 6),
                        "z_min_m": round(float(z.min()), 6), "z_max_m": round(float(z.max()), 6),
                        "max_abs_grade": round(float(np.abs(grades).max()), 8),
                        "eligible_profile_fraction": round(float(np.mean([not p["uncertainty_reasons"] for p in profile])), 6),
                        "xy_supported_height_pending_fraction": round(float(np.mean([p["evidence_state"] == "xy_supported_height_pending" for p in profile])), 6),
                        "observed_max_abs_grade": round(float(np.max(np.abs(np.diff(raw_z)/np.maximum(1e-9, np.diff(along))))), 6),
                        "maximum_height_fit_adjustment_m": round(float(np.max(np.abs(z-raw_z))), 6),
                        "profile": profile})
    cleanup.update({"skeleton_pixels": int(skeleton.sum()), "graph_chains": len(chains),
                    "short_leaf_pixels_pruned": removed_leaf_pixels,
                    "local_cycle_pixels_pruned": removed_cycle_pixels,
                    "short_graph_chains_withheld": skipped_short, "retained_segments": len(ribbons)})
    return ribbons, cleanup, nodes


def feature_collection(ribbons: list, grid: dict) -> dict:
    return {"type": "FeatureCollection", "name": "prediction_elevated_centerlines",
            "coordinate_system": {"world": "UE XYZ metres, left-handed, Z up", "geographic_crs": False},
            "features": [{"type": "Feature", "id": r["segment_id"],
                          "properties": {k: v for k, v in r.items() if k != "profile"},
                          "geometry": {"type": "LineString", "coordinates": [p["position_m"] for p in r["profile"]]}}
                         for r in ribbons]}


def deck_footprint(start: list, end: list, width: float) -> dict:
    a, b = np.asarray(start, float), np.asarray(end, float)
    d = b-a
    normal = np.array([-d[1], d[0], 0.]) / max(1e-9, np.linalg.norm(d[:2])) * width * .5
    corners = [a+normal, b+normal, b-normal, a-normal, a+normal]
    return {"type": "Polygon", "coordinates": [[p.round(6).tolist() for p in corners]]}


def placement_proposal(ribbons: list, c: Config) -> dict:
    placements, withheld = [], []
    for ribbon in ribbons:
        profile = ribbon["profile"]
        s = np.asarray([p["s_m"] for p in profile])
        xyz = np.asarray([p["position_m"] for p in profile])
        # Equal intervals avoid an arbitrarily tiny last module. This scale is
        # an abstract reference only; the UE binder uses actual mesh bounds.
        n = max(1, int(math.ceil(s[-1] / c.module_length_m)))
        edges = np.linspace(0, s[-1], n+1)
        for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            selected = np.flatnonzero((s >= a) & (s <= b))
            selected = np.unique(np.r_[selected, max(0, np.searchsorted(s, a)-1), min(len(s)-1, np.searchsorted(s, b))])
            local = [profile[int(k)] for k in selected]
            reasons = sorted({r for p in local for r in p["uncertainty_reasons"]})
            start, end = (np.asarray([np.interp(v, s, xyz[:, k]) for k in range(3)]) for v in (a, b))
            delta = end - start
            horizontal = float(np.linalg.norm(delta[:2]))
            # Curved or folded chords cannot be represented by one straight deck.
            if horizontal < 0.5 * (b-a):
                reasons.append("excessive_module_curvature")
            width = float(np.median([p["width_m"] for p in local]))
            item = {"placement_id": f"{ribbon['segment_id']}_DECK_{index+1:04d}",
                    "category": "bridge_deck", "asset_id": "bridge_deck_default",
                    "reference_asset_required": True,
                    "position_m": ((start + end) * 0.5).round(6).tolist(),
                    "rotation_deg": [round(math.degrees(math.atan2(delta[2], horizontal)), 6),
                                     round(math.degrees(math.atan2(delta[1], delta[0])), 6), 0.0],
                    "scale": [round(float(np.linalg.norm(delta))/c.module_length_m, 6),
                              round(width/c.module_width_m, 6), 1.0],
                    "start_m": start.round(6).tolist(), "end_m": end.round(6).tolist(),
                    "width_m": round(width, 6), "placement_source": "predicted_elevated_structure",
                    "source_segment_id": ribbon["segment_id"],
                    "support": {"minimum_views": min(p["support_views"] for p in local),
                                "maximum_height_mad_m": max(p["height_mad_m"] for p in local),
                                "maximum_height_adjustment_m": max(p["height_adjustment_m"] for p in local),
                                "minimum_clearance_proxy_m": min(p["clearance_proxy_m"] for p in local)},
                    "uncertainty_reasons": reasons,
                    "evidence_state": "supported_geometry" if not reasons else
                                      "uncertain_geometry" if any(p["evidence_state"] == "uncertain_geometry" for p in local) else
                                      "xy_supported_height_pending" if any(p["evidence_state"] == "xy_supported_height_pending" for p in local) else
                                      "xy_supported_topology_pending",
                    "status": "withheld_uncertain" if reasons else "requires_asset_binding"}
            (withheld if reasons else placements).append(item)
    return {"schema": "citycode.predicted_elevated_placement.v1",
            "coordinate_system": {"world": "UE", "units": "m", "up": "Z", "handedness": "left"},
            "source": "prediction_only_e09_cached", "uses_ground_truth": False,
            "abstract_asset": {"asset_id": "bridge_deck_default", "length_m": c.module_length_m,
                               "width_m": c.module_width_m, "deck_top_z_m": 0,
                               "local_forward_axis": "+X", "local_width_axis": "+Y", "local_up_axis": "+Z"},
            "transform_semantics": "position_m is the desired top-deck center; actual mesh local pivot and bounds must be resolved by the asset binder. scale is abstract, not a real UE asset transform.",
            "placements": placements, "withheld_placements": withheld,
            "placement_count": len(placements), "withheld_count": len(withheld)}



def regularized_proposal(ribbons: list, c: Config, ground_reference_z_m: float) -> tuple[dict, dict]:
    """Create explicit, limited design inference from repeated high-deck evidence.

    Unlike the strict observed proposal, nearest visible ground is not treated
    as the lower surface of an occluded bridge. The audited camera calibration
    plane supplies a declared clearance *prior*. Nothing is lifted from this
    plane when high-Z multiview anchors are absent. Competing supported layers
    and actual semantic contradictions remain hard breaks.
    """
    rules = {"minimum_anchor_height_above_calibration_plane_m": 3.5,
             "minimum_anchor_views": c.min_support_views,
             "maximum_anchor_height_mad_m": c.max_height_mad_m,
             "maximum_anchor_correction_m": 3.0,
             "maximum_anchor_gap_m": 24.0,
             "minimum_persistent_height_step_break_m": 3.0,
             "maximum_end_extrapolation_m": 8.0,
             "minimum_continuous_length_m": c.min_segment_length_m,
             "maximum_design_deck_width_m": 32.0,
             "maximum_semantic_ribbon_width_m": 80.0,
             "maximum_grade": c.max_grade,
             "ground_reference_z_m": float(ground_reference_z_m),
             "ground_reference_role": "calibration prior; bridge-underlying ground is not observed",
             "junction_policy": "Selected-layer corridors may meet; all competing supported height layers remain hard breaks. Not a solved interchange topology.",
             "no_anchor_policy": "Withhold: a class-4 mask alone cannot create a raised deck"}
    recovered, decisions = [], []
    minimum_height = ground_reference_z_m + rules["minimum_anchor_height_above_calibration_plane_m"]
    for ribbon in ribbons:
        profile = ribbon["profile"]
        s = np.asarray([p["s_m"] for p in profile])
        xyz = np.asarray([p["position_m"] for p in profile])
        raw = np.asarray([p["observed_z_m"] for p in profile])
        widths = np.asarray([p["width_m"] for p in profile])
        safe = np.asarray([p["support_views"] >= 1
                           and p["semantic_conflict_fraction"] <= c.max_semantic_conflict_fraction
                           and p["supported_height_layers"] <= 1
                           and p["evidence_distance_m"] <= c.closing_radius_m
                           and c.min_width_m <= p["width_m"] <= rules["maximum_semantic_ribbon_width_m"]
                           for p in profile])
        anchors = safe & np.asarray([p["support_views"] >= c.min_support_views
                                    and p["height_mad_m"] <= c.max_height_mad_m
                                    for p in profile]) & (raw >= minimum_height) & (raw <= ground_reference_z_m+40.)
        # Do not let isolated depth spikes become elevation anchors. Repeated
        # observations and a robust neighbourhood are both required.
        median = ndimage.median_filter(raw, size=5, mode="nearest")
        anchors &= np.abs(raw-median) <= 4.
        # A persistent abrupt Z step can be a different deck, not a noisy
        # point. Split its topology before fitting so one level cannot drag
        # a long valid road to an invented average between decks.
        height_step = np.r_[False, np.abs(np.diff(median)) > np.maximum(
            rules["minimum_persistent_height_step_break_m"], c.max_grade*np.diff(s)+1.5)]
        safe[height_step] = False
        anchors &= safe
        labels, component_count = ndimage.label(safe)
        accepted_length = 0.; accepted_runs = 0
        rejected = Counter()
        if not np.any(anchors):
            rejected["no_repeated_high_elevation_anchors"] += len(profile)
        for component in range(1, component_count+1):
            indices = np.flatnonzero(labels == component)
            local_anchor = indices[anchors[indices]]
            if len(local_anchor) < 3:
                rejected["fewer_than_three_anchors_in_continuous_semantic_span"] += len(indices)
                continue
            positions = s[local_anchor]
            insertion = np.searchsorted(positions, s[indices])
            previous = np.maximum(0, insertion-1)
            following = np.minimum(len(positions)-1, insertion)
            between = (insertion > 0) & (insertion < len(positions))
            near_left = s[indices] >= positions[0]-rules["maximum_end_extrapolation_m"]
            near_right = s[indices] <= positions[-1]+rules["maximum_end_extrapolation_m"]
            bounded_gap = (~between) | ((positions[following]-positions[previous]) <= rules["maximum_anchor_gap_m"])
            supported = near_left & near_right & bounded_gap
            # A sufficiently long unsupported span always splits the road;
            # local smoothing never fills a long missing observation region.
            supported_labels, n = ndimage.label(supported)
            for run_index in range(1, n+1):
                chosen = indices[supported_labels == run_index]
                if len(chosen) < 2 or s[chosen[-1]]-s[chosen[0]] < c.min_segment_length_m:
                    rejected["continuous_run_too_short"] += len(chosen)
                    continue
                actual_anchors = chosen[anchors[chosen]]
                if len(actual_anchors) < 3:
                    rejected["continuous_run_lacks_three_anchors"] += len(chosen)
                    continue
                fitted_input = np.interp(s[chosen], s[actual_anchors], raw[actual_anchors])
                weight = np.where(anchors[chosen], 2., .1)
                fitted = fit_bounded_height(fitted_input, s[chosen], weight, c.max_grade)
                anchor_errors = np.abs(fitted[anchors[chosen]]-raw[actual_anchors])
                if np.max(anchor_errors) > rules["maximum_anchor_correction_m"]:
                    rejected["grade_fit_cannot_preserve_local_anchors"] += len(chosen)
                    continue
                if np.min(fitted) < minimum_height-.5:
                    rejected["rule_clearance_prior_failed"] += len(chosen)
                    continue
                output_profile = []
                for k, source_index in enumerate(chosen):
                    source = profile[int(source_index)]
                    width = min(float(widths[source_index]),
                                float(ndimage.median_filter(widths[chosen], size=5, mode="nearest")[k]),
                                rules["maximum_design_deck_width_m"])
                    item = dict(source)
                    item.update({"s_m": round(float(s[source_index]-s[chosen[0]]), 6),
                                 "position_m": [*xyz[source_index, :2].tolist(), round(float(fitted[k]), 6)],
                                 "source_profile_index": int(source_index), "source_s_m": float(s[source_index]),
                                 "width_m": round(width, 6), "observed_ribbon_width_m": float(widths[source_index]),
                                 "height_adjustment_m": round(abs(float(fitted[k]-raw[source_index])), 6),
                                 "original_uncertainty_reasons": source["uncertainty_reasons"],
                                 "uncertainty_reasons": [], "evidence_state": "rule_regularized_geometry",
                                 "height_anchor": bool(anchors[source_index]),
                                 "height_source": "bounded_fit_to_multiview_predicted_high_Z_anchors",
                                 "calibration_clearance_prior_m": round(float(fitted[k]-ground_reference_z_m), 6)})
                    output_profile.append(item)
                identity = f"{ribbon['segment_id']}_REGULARIZED_{accepted_runs+1:02d}"
                length = float(s[chosen[-1]]-s[chosen[0]])
                recovered.append({"segment_id": identity, "source_segment_id": ribbon["segment_id"],
                                  "length_m": round(length, 6), "profile": output_profile,
                                  "anchor_count": len(actual_anchors), "inference": "explicit_rule_regularization"})
                accepted_length += length; accepted_runs += 1
        decisions.append({"source_segment_id": ribbon["segment_id"], "observed_length_m": ribbon["length_m"],
                          "repeated_high_Z_anchor_count": int(anchors.sum()), "regularized_runs": accepted_runs,
                          "regularized_length_m": round(accepted_length, 6), "withholding_sample_counts": dict(rejected)})
    proposal = placement_proposal(recovered, c)
    # All these transforms are a declared design inference, even when a point
    # coincides with an observation. Never relabel it as strict observed truth.
    for row in proposal["placements"]:
        row["inference"] = "explicit_rule_regularization"
        row["evidence_state"] = "rule_regularized_geometry"
        row["support"]["clearance_measurement_is_proxy"] = True
    proposal.update({"schema": "citycode.predicted_elevated_regularized_placement.v1",
                     "source": "prediction_only_e09_cached_with_declared_design_priors",
                     "replaces_not_appends": "placement_proposal.json",
                     "inference_rules": rules,
                     "continuous_segment_count": len(recovered),
                     "continuous_length_m": round(sum(r["length_m"] for r in recovered), 6),
                     "segment_decisions": decisions,
                     "continuous_segments": recovered,
                     "limitations": ["Prediction replay and rule inference, not surveyed bridge geometry.",
                                      "Multiple supported deck layers are withheld; interchange connectivity is incomplete.",
                                      "UE bridge mesh shape, joins and support pillars require asset and scene QA.",
                                      "North-side class-4 regions near ground level are not lifted without high-elevation anchors."]})
    features = {"type": "FeatureCollection", "name": "regularized_prediction_elevated_centerlines",
                "coordinate_system": proposal["coordinate_system"],
                "features": [{"type": "Feature", "id": r["segment_id"],
                              "properties": {key: value for key, value in r.items() if key != "profile"},
                              "geometry": {"type": "LineString", "coordinates": [p["position_m"] for p in r["profile"]]}}
                             for r in recovered]}
    return proposal, features



def background_filter_mask(grid: dict, output: Path, margin_m: float = 1.) -> dict:
    """Export predicted XY suppression only, never a replacement height field.

    The UE-only background copy can omit old elevated triangles here before
    new native decks are installed. This mask deliberately includes withheld
    high-road semantics, otherwise old/reference-supported geometry would show
    through where the new prediction is uncertain. Ground removal is forbidden.
    """
    source = grid["retained_mask"].astype(bool)
    mask = source.copy()
    if source.any():
        mask |= ndimage.distance_transform_edt(~source, sampling=grid["cell_size_m"]) <= margin_m
    filename = "prediction_elevated_filter_mask.npz"
    np.savez_compressed(output/filename, remove_high_surface_mask=mask,
                        source_retained_class4_mask=source,
                        origin_xy_m=grid["origin_xy_m"],
                        cell_size_m=np.asarray(grid["cell_size_m"], dtype=np.float64))
    descriptor = {"schema": "citycode.prediction_elevated_background_filter.v1",
                  "uses_ground_truth_for_mask": False,
                  "mask_file": filename, "mask_sha256": digest(output/filename),
                  "mask_field": "remove_high_surface_mask",
                  "source_field": "source_retained_class4_mask",
                  "coordinate_system": {"world": "UE", "units": "m", "up": "Z"},
                  "origin_xy_m": grid["origin_xy_m"].tolist(),
                  "cell_size_m": grid["cell_size_m"], "shape_yx": list(mask.shape),
                  "world_to_cell": "ix=floor((world_x-origin_x)/cell_size); iy=floor((world_y-origin_y)/cell_size); index mask[iy,ix]. Origin is lower-left corner, not cell center.",
                  "margin_m": margin_m, "high_surface_threshold_m": 2.0,
                  "retained_class4_cells": int(source.sum()), "suppression_mask_cells": int(mask.sum()),
                  "mask_policy": "All area-retained predicted class-4 corridors, including single-view/height-withheld portions, plus an EDT-bounded margin. This does not assert that every masked region is a valid raised deck.",
                  "triangle_policy": "Test XY triangle-footprint overlap against mask cells, including triangles whose corners and centroid miss the mask. Remove only portions/triangles confirmed above threshold; preserve lower ground. Record conservative straddling-triangle handling explicitly.",
                  "destination_policy": "UE interchange background copy only; published source background remains unchanged",
                  "no_height_substitution": True}
    write_json(output/"prediction_elevated_filter_mask.json", descriptor)
    return descriptor


def run(cache: Path, predictions: Path | None, output: Path, c: Config) -> dict:
    executing_code_sha256 = digest(Path(__file__))
    validate_config(c)
    for source in (cache, predictions):
        if source and (output.is_relative_to(source) or source.is_relative_to(output)):
            raise ValueError("Output must be separate from prediction input directories")
    output.mkdir(parents=True, exist_ok=True)
    audit, rows, visibility = audit_sources(cache, predictions, c)
    print(f"Verified prediction inputs: {len(rows)} E09 regions", flush=True)
    grid, stats = fuse_prediction_grid(cache, rows, c)
    print("Fused observed predicted elevated XYZ; extracting centerlines", flush=True)
    ribbons, cleanup, nodes = recover_ribbons(grid, c)
    audit["sample_evidence_statistics"] = stats
    write_json(output / "source_audit.json", audit)
    model = {"schema": "citycode.predicted_elevated_structure.v1", "uses_ground_truth": False,
             "method": "sample-level medians, per-cell height-gap clustering, supported-layer selection, bounded morphology, Zhang-Suen skeleton graph, inscribed widths, smooth Lipschitz-bounded height profiles",
             "config": asdict(c), "coordinate_system": audit["coordinate_system"],
             "grid": {"origin_xy_m": grid["origin_xy_m"].tolist(), "cell_size_m": c.grid_m, "shape_yx": list(grid["shape"])},
             "cleanup": cleanup, "junctions_and_endpoints": nodes, "segments": ribbons,
             "support_evidence": {"grid_file": "prediction_evidence_grid.npz", "sample_order": [r["id"] for r in rows],
                                  "view_ids": "evidence_view_index indexes sample_order; each evidence_cell_index is row-major in shape_yx",
                                  "height_mad": "median absolute difference between per-view cell medians; zero for a single view does not imply certainty",
                                  "height_layers": "Z gaps larger than height_layer_gap_m split layers. All layers and member sample indices are retained. Primary layer: most independent views, then higher Z; multiple supported layers require explicit 3D topology.",
                                  "semantic_conflict": "Only other-class observations within semantic_height_tolerance_m of the selected elevated layer count as contradictory views. Lower ground beneath a bridge is compatible multilevel evidence.",
                                  "width": "twice distance to the predicted mask boundary, a conservative inscribed ribbon estimate",
                                  "clearance": "difference from nearest visible predicted ground/hardscape; a proxy, not observed ground beneath the deck"}}
    model["inference_policy"] = {"xy": "Topology-preserving skeleton, short-leaf removal, and at most max_xy_adjustment_m local smoothing within the retained semantic mask.",
                                 "height": "Five-sample median identifies >4m short spikes, followed by weighted second-difference smoothing and max_grade Lipschitz envelopes. Original predicted heights remain in every profile sample; corrections over max_height_adjustment_m withhold modules.",
                                 "ground": "No forced clearance or reference bridge Z is inserted. Insufficient elevation remains explicitly pending.",
                                 "uncertain_geometry": "Retained for review, not silently deleted or placed."}
    write_json(output / "elevated_structure.json", model)
    write_json(output / "elevated_centerlines.geojson", feature_collection(ribbons, grid))
    proposal = placement_proposal(ribbons, c)
    write_json(output / "placement_proposal.json", proposal)
    ground_references = {s["ground_reference_z_m"] for s in audit["source_samples"]}
    if len(ground_references) != 1 or None in ground_references:
        raise ValueError("Regularized proposal requires one explicit audited calibration ground plane")
    regularized, regularized_lines = regularized_proposal(ribbons, c, float(next(iter(ground_references))))
    write_json(output / "regularized_placement_proposal.json", regularized)
    write_json(output / "regularized_centerlines.geojson", regularized_lines)
    uncertain_features = [{"type": "Feature", "id": p["placement_id"],
                           "geometry": deck_footprint(p["start_m"], p["end_m"], p["width_m"]),
                           "properties": {"segment_id": p["source_segment_id"], "status": p["status"],
                                          "uncertainty_reasons": p["uncertainty_reasons"], "support": p["support"]}}
                          for p in proposal["withheld_placements"]]
    write_json(output / "uncertain_regions.geojson", {"type": "FeatureCollection", "coordinate_system": audit["coordinate_system"],
                                                    "features": uncertain_features})
    np.savez_compressed(output / "prediction_evidence_grid.npz", **{k: v for k, v in grid.items() if isinstance(v, np.ndarray)})
    background_filter_mask(grid, output)
    if digest(Path(__file__)) != executing_code_sha256:
        raise RuntimeError("Extractor source changed during execution; rerun the frozen code")
    summary = {"schema": "citycode.predicted_elevated_run.v1", "status": "complete",
               "uses_ground_truth": False, "source_audit": "source_audit.json", "config": asdict(c),
               "code_sha256": executing_code_sha256, "segment_count": len(ribbons),
               "centerline_length_m": round(sum(r["length_m"] for r in ribbons), 3),
               "eligible_abstract_deck_count": proposal["placement_count"],
               "withheld_deck_count": proposal["withheld_count"],
               "withheld_reason_counts": dict(Counter(reason for p in proposal["withheld_placements"] for reason in p["uncertainty_reasons"])),
               "withheld_evidence_state_counts": dict(Counter(p["evidence_state"] for p in proposal["withheld_placements"])),
               "regularized_continuous_segment_count": regularized["continuous_segment_count"],
               "regularized_continuous_length_m": regularized["continuous_length_m"],
               "regularized_abstract_deck_count": regularized["placement_count"],
               "scene_modified": False,
               "outputs": {name: {"sha256": digest(output/name), "bytes": (output/name).stat().st_size}
                           for name in ["source_audit.json", "elevated_structure.json", "elevated_centerlines.geojson",
                                        "placement_proposal.json", "uncertain_regions.geojson", "prediction_evidence_grid.npz",
                                        "regularized_placement_proposal.json", "regularized_centerlines.geojson",
                                        "prediction_elevated_filter_mask.npz", "prediction_elevated_filter_mask.json"]}}
    write_json(output / "RUN.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON overrides for Config fields")
    args = parser.parse_args()
    c = Config(**read_json(args.config)) if args.config else Config()
    summary = run(args.cache.resolve(), args.predictions.resolve() if args.predictions else None, args.output.resolve(), c)
    print(json.dumps({k: v for k, v in summary.items() if k != "outputs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
