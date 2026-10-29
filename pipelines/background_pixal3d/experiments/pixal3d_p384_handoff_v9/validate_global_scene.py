#!/usr/bin/env python3
"""Validate all canonical tiles, neighbor overlaps and global owned coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-m", type=float, default=2.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_array(path: Path):
    try:
        return np.load(path, mmap_mode="r")
    except OSError as error:
        if error.errno != 19:
            raise
        return np.load(path)


def mesh_vertices(path: Path):
    scene = trimesh.load(path, force="scene", process=False)
    chunks = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        chunks.append(trimesh.transform_points(scene.geometry[geometry_name].vertices, transform))
    return np.concatenate(chunks, axis=0)


def top_grid(vertices, bounds, bin_m):
    x0, y0, x1, y1 = map(float, bounds)
    width = int(round((x1 - x0) / bin_m))
    height = int(round((y1 - y0) / bin_m))
    inside = (
        (vertices[:, 0] >= x0) & (vertices[:, 0] < x1) &
        (vertices[:, 2] >= y0) & (vertices[:, 2] < y1)
    )
    points = vertices[inside]
    ix = np.floor((points[:, 0] - x0) / bin_m).astype(np.int64)
    iy = np.floor((points[:, 2] - y0) / bin_m).astype(np.int64)
    flat = iy * width + ix
    top = np.full(height * width, -np.inf, np.float32)
    bottom = np.full(height * width, np.inf, np.float32)
    np.maximum.at(top, flat, points[:, 1])
    np.minimum.at(bottom, flat, points[:, 1])
    top = top.reshape(height, width)
    bottom = bottom.reshape(height, width)
    absent = ~np.isfinite(top)
    top[absent] = np.nan
    bottom[absent] = np.nan
    return top, top - bottom


def stats(values, ps=(50, 90, 95, 99, 100)):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return {str(p): float(v) for p, v in zip(ps, np.percentile(values, ps))} if values.size else {}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    items = inputs["patches"]
    available = [item for item in items if (args.runs / item["patch_id"] / "mesh_pbr_world_canonical_axis_v3.glb").is_file()]
    missing = sorted(set(item["patch_id"] for item in items) - set(item["patch_id"] for item in available))
    if missing and not args.allow_incomplete:
        raise ValueError(f"Missing {len(missing)} canonical meshes")

    teacher_manifest = json.loads((args.teacher / "manifest.json").read_text())
    teacher_base = load_array(args.teacher / teacher_manifest["outputs"]["base_height_m"])
    teacher_elevated = load_array(args.teacher / teacher_manifest["outputs"]["elevated_height_m"])
    teacher_bounds = np.asarray(teacher_manifest["world_bounds_xy_m"], np.float64)
    teacher_mpp = float(teacher_manifest["resolution_m"])
    x0, y0, x1, y1 = teacher_bounds
    global_width = int(round((x1 - x0) / args.bin_m))
    global_height = int(round((y1 - y0) / args.bin_m))
    global_top = np.full((global_height, global_width), np.nan, np.float32)
    global_span = np.full((global_height, global_width), np.nan, np.float32)
    centers = np.asarray([item["center_world_m"][:2] for item in available], np.float64)
    tree = cKDTree(centers)
    grids = {}
    patch_records = []

    for owner_index, item in enumerate(available):
        patch_id = item["patch_id"]
        path = args.runs / patch_id / "mesh_pbr_world_canonical_axis_v3.glb"
        vertices = mesh_vertices(path)
        patch_top, patch_span = top_grid(vertices, item["data_bounds_xy_world_m"], args.bin_m)
        px0, py0, px1, py1 = map(float, item["data_bounds_xy_world_m"])
        gx0 = int(round((px0 - x0) / args.bin_m))
        gy0 = int(round((py0 - y0) / args.bin_m))
        h, w = patch_top.shape
        xs = px0 + (np.arange(w) + 0.5) * args.bin_m
        ys = py0 + (np.arange(h) + 0.5) * args.bin_m
        wx, wy = np.meshgrid(xs, ys, indexing="xy")
        _, nearest = tree.query(np.column_stack((wx.ravel(), wy.ravel())), k=1)
        owned = (nearest.reshape(h, w) == owner_index) & np.isfinite(patch_top)
        target_top = global_top[gy0:gy0 + h, gx0:gx0 + w]
        target_span = global_span[gy0:gy0 + h, gx0:gx0 + w]
        target_top[owned] = patch_top[owned]
        target_span[owned] = patch_span[owned]
        grids[patch_id] = {"top": patch_top, "gx0": gx0, "gy0": gy0, "bounds": item["data_bounds_xy_world_m"]}
        canonical_metadata = json.loads((args.runs / patch_id / "canonical_metadata.json").read_text())
        patch_records.append({
            "patch_id": patch_id,
            "vertices": int(len(vertices)),
            "occupied_bins": int(np.isfinite(patch_top).sum()),
            "owned_bins": int(owned.sum()),
            "vertical_span_m": stats(patch_span),
            "canonical": canonical_metadata,
        })
        print(json.dumps({"index": owner_index + 1, **patch_records[-1]}), flush=True)

    # Sample the 0.15 m two-layer teacher at the centers of the validation bins.
    xs = x0 + (np.arange(global_width) + 0.5) * args.bin_m
    ys = y0 + (np.arange(global_height) + 0.5) * args.bin_m
    wx, wy = np.meshgrid(xs, ys, indexing="xy")
    tx = np.floor((wx - x0) / teacher_mpp).astype(np.int64)
    ty = np.floor((wy - y0) / teacher_mpp).astype(np.int64)
    teacher_base_grid = np.asarray(teacher_base[ty, tx])
    teacher_elevated_grid = np.asarray(teacher_elevated[ty, tx])
    teacher_grid = np.where(np.isfinite(teacher_elevated_grid), teacher_elevated_grid, teacher_base_grid)
    comparable = np.isfinite(global_top) & np.isfinite(teacher_grid)
    global_error = np.full(global_top.shape, np.nan, np.float32)
    global_error[comparable] = global_top[comparable] - teacher_grid[comparable]

    seam_records = []
    all_deltas = []
    for a in range(len(available)):
        for b in range(a + 1, len(available)):
            delta = centers[b] - centers[a]
            if not (np.isclose(abs(delta[0]) + abs(delta[1]), 160.0) and (np.isclose(delta[0], 0) or np.isclose(delta[1], 0))):
                continue
            item_a, item_b = available[a], available[b]
            ga, gb = grids[item_a["patch_id"]], grids[item_b["patch_id"]]
            ax0, ay0 = ga["gx0"], ga["gy0"]
            bx0, by0 = gb["gx0"], gb["gy0"]
            ah, aw = ga["top"].shape; bh, bw = gb["top"].shape
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
            top_a = ga["top"][iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
            top_b = gb["top"][iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
            shared = np.isfinite(top_a) & np.isfinite(top_b)
            seam_delta = np.abs(top_a[shared] - top_b[shared])
            all_deltas.append(seam_delta)
            seam_records.append({
                "patch_a": item_a["patch_id"], "patch_b": item_b["patch_id"],
                "orientation": "horizontal" if delta[0] else "vertical",
                "shared_bins": int(shared.sum()),
                "absolute_height_delta_m": stats(seam_delta),
            })

    all_deltas = np.concatenate(all_deltas) if all_deltas else np.empty(0)
    validation = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_global_validation.v1",
        "axis_mapping_version": "pbr_local_z_to_ue_y_positive_v3",
        "status": "complete" if not missing else "partial",
        "patch_count": len(available), "missing_patches": missing,
        "bin_m": args.bin_m,
        "global": {
            "occupied_owned_bins": int(np.isfinite(global_top).sum()),
            "teacher_valid_bins": int(np.isfinite(teacher_grid).sum()),
            "teacher_base_valid_bins": int(np.isfinite(teacher_base_grid).sum()),
            "teacher_elevated_valid_bins": int(np.isfinite(teacher_elevated_grid).sum()),
            "coverage_fraction": float(np.isfinite(global_top).sum() / max(np.isfinite(teacher_grid).sum(), 1)),
            "top_multilayer_teacher_abs_error_m": stats(np.abs(global_error[comparable])),
            "vertical_span_m": stats(global_span),
            "span_over_5m_fraction": float((global_span[np.isfinite(global_span)] > 5).mean()),
        },
        "seams": {
            "pair_count": len(seam_records),
            "shared_bins": int(sum(item["shared_bins"] for item in seam_records)),
            "absolute_height_delta_m": stats(all_deltas),
            "pairs": seam_records,
        },
        "patches": patch_records,
    }
    (args.output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    images = [
        axes[0].imshow(global_top, origin="lower", cmap="terrain", vmin=-5, vmax=50),
        axes[1].imshow(global_span, origin="lower", cmap="magma", vmin=0, vmax=5),
        axes[2].imshow(global_error, origin="lower", cmap="coolwarm", vmin=-2, vmax=2),
    ]
    titles = ["Single-owner generated top height", "Vertical layer span", "Top minus two-layer teacher"]
    for axis, image, title in zip(axes, images, titles):
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=.72, label="metres")
    fig.savefig(args.output / "global_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": validation["status"], "global": validation["global"], "seams": {k: v for k, v in validation["seams"].items() if k != "pairs"}}), flush=True)


if __name__ == "__main__":
    main()
