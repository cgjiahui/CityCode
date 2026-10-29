#!/usr/bin/env python3
"""Build a faithful small top-surface preview by rasterizing final triangles."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import trimesh
import nvdiffrast.torch as dr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-m", type=float, default=2.0)
    parser.add_argument("--max-height-step-m", type=float, default=6.0)
    return parser.parse_args()


def vertex_rgb(geometry):
    visual = geometry.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if uv is None or texture is None:
        colors = getattr(visual, "vertex_colors", None)
        if colors is not None and len(colors) == len(geometry.vertices):
            return np.asarray(colors, np.uint8)[:, :3]
        return np.full((len(geometry.vertices), 3), 160, np.uint8)
    image = np.asarray(texture.convert("RGB"))
    uv = np.asarray(uv, np.float64)
    x = np.clip(np.rint(uv[:, 0] * (image.shape[1] - 1)), 0, image.shape[1] - 1).astype(np.int64)
    y = np.clip(np.rint((1.0 - uv[:, 1]) * (image.shape[0] - 1)), 0, image.shape[0] - 1).astype(np.int64)
    colors = image[y, x].astype(np.float32)
    factor = np.asarray(getattr(material, "baseColorFactor", [1, 1, 1, 1]), np.float32)[:3]
    if factor.max(initial=0) > 1.0:
        factor /= 255.0
    return np.clip(colors * factor, 0, 255).astype(np.uint8)


def flatten_scene(path):
    scene = trimesh.load(path, force="scene", process=False)
    vertices = []
    faces = []
    colors = []
    offset = 0
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        v = np.asarray(geometry.vertices, np.float32)
        vertices.append(v)
        faces.append(np.asarray(geometry.faces, np.int64) + offset)
        colors.append(vertex_rgb(geometry))
        offset += len(v)
    return scene, np.vstack(vertices), np.vstack(faces), np.vstack(colors)


def render_tile(context, path, bounds, resolution_hw):
    scene, vertices, faces, colors = flatten_scene(path)
    x0, y0, x1, y1 = bounds
    v = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
    f = torch.as_tensor(faces, dtype=torch.int32, device="cuda")
    c = torch.as_tensor(colors.astype(np.float32) / 255.0, dtype=torch.float32, device="cuda")
    low = float(v[:, 1].min().item())
    high = float(v[:, 1].max().item())
    scale = max(high - low, 1e-3)
    # High world elevation maps to the near side (-1) of the depth interval.
    clip = torch.stack((
        2.0 * (v[:, 0] - x0) / (x1 - x0) - 1.0,
        2.0 * (v[:, 2] - y0) / (y1 - y0) - 1.0,
        1.0 - 2.0 * (v[:, 1] - low) / scale,
        torch.ones_like(v[:, 0]),
    ), dim=1)[None]
    raster, _ = dr.rasterize(context, clip, f, resolution=list(resolution_hw))
    color, _ = dr.interpolate(c[None], raster, f)
    elevation, _ = dr.interpolate(v[:, 1:2][None].contiguous(), raster, f)
    occupied = (raster[0, :, :, 3] > 0).cpu().numpy()
    rgb = np.clip(color[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    top = elevation[0, :, :, 0].cpu().numpy().astype(np.float32)
    del scene, vertices, faces, colors, v, f, c, clip, raster, color, elevation
    torch.cuda.empty_cache()
    return occupied, top, rgb


def resolve_tile(fusion, record):
    path = Path(record.get("tile") or record["tile_url"])
    if not path.is_absolute():
        path = path if path.is_file() else fusion / path
    return path


def main():
    args = parse_args()
    manifest = json.loads((args.fusion / "scene_manifest.json").read_text())
    gx0, gy0, gx1, gy1 = map(float, manifest["world_bounds_xy_m"])
    width = int(round((gx1 - gx0) / args.bin_m))
    height = int(round((gy1 - gy0) / args.bin_m))
    top = np.full((height, width), -np.inf, np.float32)
    rgb = np.zeros((height, width, 3), np.uint8)
    expected = np.zeros((height, width), bool)
    context = dr.RasterizeCudaContext()
    records = manifest["tiles"]
    for index, record in enumerate(records, 1):
        x0, y0, x1, y1 = map(float, record["data_bounds_xy_world_m"])
        tile_width = int(round((x1 - x0) / args.bin_m))
        tile_height = int(round((y1 - y0) / args.bin_m))
        column0 = int(round((x0 - gx0) / args.bin_m))
        row0 = int(round((y0 - gy0) / args.bin_m))
        column1 = column0 + tile_width
        row1 = row0 + tile_height
        expected[row0:row1, column0:column1] = True
        occupied, tile_top, tile_rgb = render_tile(
            context, resolve_tile(args.fusion, record), (x0, y0, x1, y1), (tile_height, tile_width),
        )
        target_top = top[row0:row1, column0:column1]
        improve = occupied & (tile_top > target_top)
        target_top[improve] = tile_top[improve]
        rgb[row0:row1, column0:column1][improve] = tile_rgb[improve]
        print(json.dumps({
            "index": index, "total": len(records), "patch_id": record["patch_id"],
            "rasterized_pixels": int(occupied.sum()), "updated_pixels": int(improve.sum()),
        }), flush=True)
        gc.collect()
    occupied = np.isfinite(top)
    missing = expected & ~occupied
    vertex_index = np.full((height, width), -1, np.int64)
    rows, columns = np.nonzero(occupied)
    vertex_index[rows, columns] = np.arange(len(rows), dtype=np.int64)
    vertices = np.column_stack((
        gx0 + (columns + 0.5) * args.bin_m,
        top[rows, columns],
        gy0 + (rows + 0.5) * args.bin_m,
    ))
    colors = np.column_stack((rgb[rows, columns], np.full(len(rows), 255, np.uint8)))
    a = vertex_index[:-1, :-1]
    b = vertex_index[:-1, 1:]
    c = vertex_index[1:, :-1]
    d = vertex_index[1:, 1:]
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    h00, h01, h10, h11 = top[:-1, :-1], top[:-1, 1:], top[1:, :-1], top[1:, 1:]
    with np.errstate(invalid="ignore"):
        span = np.where(
            quad,
            np.maximum.reduce([h00, h01, h10, h11]) - np.minimum.reduce([h00, h01, h10, h11]),
            np.inf,
        )
    quad &= span <= args.max_height_step_m
    qa, qb, qc, qd = a[quad], b[quad], c[quad], d[quad]
    faces = np.concatenate((np.column_stack((qa, qc, qb)), np.column_stack((qb, qc, qd))), axis=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    preview_rgb = rgb.copy()
    preview_rgb[~occupied] = 8
    preview_image_path = args.output.with_suffix(".png")
    Image.fromarray(preview_rgb).save(preview_image_path)
    expected_count = int(expected.sum())
    metadata = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_rasterized_top_preview.v1",
        "source_manifest": str((args.fusion / "scene_manifest.json").resolve()),
        "source_axis_mapping_version": manifest.get("axis_mapping_version"),
        "bin_m": args.bin_m,
        "max_height_step_m": args.max_height_step_m,
        "expected_domain_cells": expected_count,
        "occupied_domain_cells": int((occupied & expected).sum()),
        "missing_domain_cells": int(missing.sum()),
        "projected_triangle_coverage_fraction": float((occupied & expected).sum() / max(expected_count, 1)),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "size_bytes": args.output.stat().st_size,
        "preview_image": str(preview_image_path.resolve()),
        "bounds": mesh.bounds.tolist(),
        "representation": "GPU triangle-rasterized highest final fused surface; colors interpolated from final PBR base-color samples",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", **metadata}), flush=True)


if __name__ == "__main__":
    main()
