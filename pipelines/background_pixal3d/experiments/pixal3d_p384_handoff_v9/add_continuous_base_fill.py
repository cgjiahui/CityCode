#!/usr/bin/env python3
"""Add a dense height-field base only where fused Pixal3D tiles have no top coverage.

The source Pixal3D geometry and PBR materials are preserved.  A 1 m world grid
is raster-tested against the generated base mesh; owner cells with no generated
triangle at the cell centre receive two real triangles sampled from the filled
base_surface_z_m teacher and textured with the restored 2560 px RGB.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fusion", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-m", type=float, default=1.0)
    parser.add_argument("--patch-id", action="append")
    parser.add_argument("--full-underlay", action="store_true")
    parser.add_argument("--underlay-offset-m", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_tile(source_fusion: Path, record: dict):
    path = Path(record.get("tile") or record["tile_url"])
    if not path.is_absolute():
        path = path if path.is_file() else source_fusion / path
    return path


def world_geometries(scene: trimesh.Scene):
    output = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        output.append((geometry_name, geometry))
    return output


def rasterized_base_coverage(context, geometries, bounds, resolution_hw):
    base = [geometry for name, geometry in geometries if name.endswith("_0")]
    if not base:
        base = [geometries[0][1]]
    mesh = trimesh.util.concatenate(tuple(base))
    vertices = torch.as_tensor(
        np.asarray(mesh.vertices), dtype=torch.float32, device="cuda",
    )
    faces = torch.as_tensor(
        np.asarray(mesh.faces), dtype=torch.int32, device="cuda",
    )
    x0, y0, x1, y1 = bounds
    min_height = float(vertices[:, 1].min().item())
    max_height = float(vertices[:, 1].max().item())
    height_scale = max(max_height - min_height, 1e-3)
    clip = torch.stack((
        2.0 * (vertices[:, 0] - x0) / (x1 - x0) - 1.0,
        2.0 * (vertices[:, 2] - y0) / (y1 - y0) - 1.0,
        2.0 * (vertices[:, 1] - min_height) / height_scale - 1.0,
        torch.ones_like(vertices[:, 0]),
    ), dim=1)[None]
    raster, _ = dr.rasterize(context, clip, faces, resolution=list(resolution_hw))
    covered = (raster[0, :, :, 3] > 0).cpu().numpy()
    del raster, clip, faces, vertices, mesh
    torch.cuda.empty_cache()
    return covered


def sample_height_grid(height_path: Path, bounds, width, height):
    with np.load(height_path) as bundle:
        base = np.asarray(bundle["base_surface_z_m"], np.float32)
    x0, y0, x1, y1 = bounds
    mpp_x = (x1 - x0) / base.shape[1]
    mpp_y = (y1 - y0) / base.shape[0]
    world_x = np.linspace(x0, x1, width + 1, dtype=np.float32)
    world_y = np.linspace(y0, y1, height + 1, dtype=np.float32)
    map_x = np.broadcast_to(((world_x - x0) / mpp_x - 0.5)[None, :], (height + 1, width + 1))
    map_y = np.broadcast_to(((world_y - y0) / mpp_y - 0.5)[:, None], (height + 1, width + 1))
    sampled = cv2.remap(
        base, np.ascontiguousarray(map_x), np.ascontiguousarray(map_y),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    if not np.isfinite(sampled).all():
        raise ValueError(f"Non-finite filled base height: {height_path}")
    return sampled


def make_fill_mesh(item, fill_cells, grid_m, height_path, rgb_path, height_offset_m=0.0):
    x0, y0, x1, y1 = map(float, item["data_bounds_xy_world_m"])
    height, width = fill_cells.shape
    sampled = sample_height_grid(height_path, (x0, y0, x1, y1), width, height) - height_offset_m
    columns = np.arange(width + 1, dtype=np.float64)
    rows = np.arange(height + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x0 + columns * grid_m, y0 + rows * grid_m)
    vertices = np.column_stack((grid_x.ravel(), sampled.ravel(), grid_y.ravel()))
    uv = np.column_stack((
        ((grid_x - x0) / (x1 - x0)).ravel(),
        (1.0 - (grid_y - y0) / (y1 - y0)).ravel(),
    ))
    index = np.arange((height + 1) * (width + 1), dtype=np.int64).reshape(height + 1, width + 1)
    a = index[:-1, :-1][fill_cells]
    b = index[:-1, 1:][fill_cells]
    c = index[1:, :-1][fill_cells]
    d = index[1:, 1:][fill_cells]
    faces = np.concatenate((np.column_stack((a, c, b)), np.column_stack((b, c, d))), axis=0)
    image = Image.open(rgb_path).convert("RGB")
    material = trimesh.visual.material.PBRMaterial(
        name=f"{item['patch_id']}_restored_rgb_base_fill",
        baseColorTexture=image,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        alphaMode="OPAQUE",
    )
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def build_covered_nearest_owner(items, grid_m):
    bounds = np.asarray([item["data_bounds_xy_world_m"] for item in items], np.float64)
    gx0, gy0 = bounds[:, :2].min(axis=0)
    gx1, gy1 = bounds[:, 2:].max(axis=0)
    width = int(round((gx1 - gx0) / grid_m))
    height = int(round((gy1 - gy0) / grid_m))
    owner = np.full((height, width), -1, np.int32)
    distance = np.full((height, width), np.inf, np.float64)
    for index, item in enumerate(items):
        x0, y0, x1, y1 = map(float, item["data_bounds_xy_world_m"])
        column0 = int(round((x0 - gx0) / grid_m))
        row0 = int(round((y0 - gy0) / grid_m))
        column1 = int(round((x1 - gx0) / grid_m))
        row1 = int(round((y1 - gy0) / grid_m))
        x = gx0 + (np.arange(column0, column1, dtype=np.float64) + 0.5) * grid_m
        y = gy0 + (np.arange(row0, row1, dtype=np.float64) + 0.5) * grid_m
        xx, yy = np.meshgrid(x, y)
        center_x, center_y = map(float, item["center_world_m"][:2])
        candidate_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        current_distance = distance[row0:row1, column0:column1]
        current_owner = owner[row0:row1, column0:column1]
        improve = candidate_distance < current_distance
        current_distance[improve] = candidate_distance[improve]
        current_owner[improve] = index
    if not np.any(owner >= 0):
        raise ValueError("No world-grid cell is covered by an input patch")
    return owner, (float(gx0), float(gy0), float(gx1), float(gy1))


def main():
    args = parse_args()
    if args.grid_m <= 0:
        raise ValueError("grid-m must be positive")
    source_manifest = read_json(args.source_fusion / "scene_manifest.json")
    input_manifest = read_json(args.inputs / "manifest.json")
    input_by_patch = {item["patch_id"]: item for item in input_manifest["patches"]}
    source_by_patch = {item["patch_id"]: item for item in source_manifest["tiles"]}
    selected = set(args.patch_id or input_by_patch)
    unknown = selected - input_by_patch.keys()
    if unknown:
        raise ValueError(f"Unknown patches: {sorted(unknown)}")
    records_input = [item for item in input_manifest["patches"] if item["patch_id"] in selected]
    owner_grid, owner_bounds = build_covered_nearest_owner(input_manifest["patches"], args.grid_m)
    owner_index = {item["patch_id"]: i for i, item in enumerate(input_manifest["patches"])}
    args.output.mkdir(parents=True, exist_ok=True)
    tile_dir = args.output / "tiles"
    metadata_dir = args.output / "tile_metadata"
    tile_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)
    context = dr.RasterizeCudaContext()
    records = []
    for number, item in enumerate(records_input, 1):
        patch_id = item["patch_id"]
        destination = tile_dir / f"{patch_id}.glb"
        metadata_path = metadata_dir / f"{patch_id}.json"
        if destination.is_file() and metadata_path.is_file() and not args.overwrite:
            record = read_json(metadata_path)
            records.append(record)
            print(json.dumps({"index": number, "total": len(records_input), "patch_id": patch_id, "status": "cached"}), flush=True)
            continue
        source_record = source_by_patch[patch_id]
        source_path = resolved_tile(args.source_fusion, source_record)
        source_scene = trimesh.load(source_path, force="scene", process=False)
        geometries = world_geometries(source_scene)
        x0, y0, x1, y1 = map(float, item["data_bounds_xy_world_m"])
        width = int(round((x1 - x0) / args.grid_m))
        height = int(round((y1 - y0) / args.grid_m))
        if abs(width * args.grid_m - (x1 - x0)) > 1e-6 or abs(height * args.grid_m - (y1 - y0)) > 1e-6:
            raise ValueError(f"Grid does not divide patch bounds: {patch_id}")
        generated_covered = rasterized_base_coverage(
            context, geometries, (x0, y0, x1, y1), (height, width),
        )
        gx0, gy0, _, _ = owner_bounds
        column0 = int(round((x0 - gx0) / args.grid_m))
        row0 = int(round((y0 - gy0) / args.grid_m))
        nearest = owner_grid[row0:row0 + height, column0:column0 + width]
        owned = nearest == owner_index[patch_id]
        fill_cells = owned if args.full_underlay else owned & ~generated_covered
        height_path = Path(item["height_bundle"])
        rgb_path = Path(item["input"])
        fill_mesh = make_fill_mesh(
            item, fill_cells, args.grid_m, height_path, rgb_path,
            height_offset_m=args.underlay_offset_m if args.full_underlay else 0.0,
        )
        output_scene = trimesh.Scene()
        for geometry_name, geometry in geometries:
            output_scene.add_geometry(geometry, node_name=geometry_name, geom_name=geometry_name)
        fill_name = f"{patch_id}_continuous_base_gap_fill"
        output_scene.add_geometry(fill_mesh, node_name=fill_name, geom_name=fill_name)
        output_scene.export(destination, extension_webp=True)
        verified = trimesh.load(destination, force="scene", process=False)
        generated_owner = int((owned & generated_covered).sum())
        owner_cells = int(owned.sum())
        fill_count = int(fill_cells.sum())
        covered_after = int((owned & (generated_covered | fill_cells)).sum())
        record = {
            "patch_id": patch_id,
            "row": item["row"],
            "column": item["column"],
            "center_world_m": item["center_world_m"],
            "data_bounds_xy_world_m": item["data_bounds_xy_world_m"],
            "core_bounds_xy_world_m": item["core_bounds_xy_world_m"],
            "source": str(source_path.resolve()),
            "tile": str(destination.resolve()),
            "tile_url": f"tiles/{patch_id}.glb",
            "grid_m": args.grid_m,
            "owner_cells": owner_cells,
            "generated_base_covered_owner_cells": generated_owner,
            "generated_base_coverage_fraction": generated_owner / max(owner_cells, 1),
            "base_gap_fill_cells": fill_count,
            "base_gap_fill_faces": int(len(fill_mesh.faces)),
            "covered_owner_cells_after": covered_after,
            "coverage_after_fraction": covered_after / max(owner_cells, 1),
            "fill_mode": "full_base_underlay" if args.full_underlay else "generated_gap_only",
            "underlay_offset_m": args.underlay_offset_m if args.full_underlay else 0.0,
            "source_faces": int(sum(len(g.faces) for _, g in geometries)),
            "faces": int(sum(len(g.faces) for g in verified.geometry.values())),
            "geometries": int(len(verified.geometry)),
            "size_bytes": destination.stat().st_size,
            "height_source": str(height_path.resolve()),
            "texture_source": str(rgb_path.resolve()),
            "fill_policy": "real 1 m filled-base height-field underlay over every owner cell" if args.full_underlay else "real 1 m base_surface_z_m triangles only where generated base has no rasterized top coverage at owner-cell centre",
        }
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        records.append(record)
        print(json.dumps({"index": number, "total": len(records_input), **record}), flush=True)
        del verified, output_scene, fill_mesh, source_scene, geometries
        gc.collect()
    if len(selected) == len(input_by_patch):
        total_owner = sum(item["owner_cells"] for item in records)
        total_generated = sum(item["generated_base_covered_owner_cells"] for item in records)
        total_fill = sum(item["base_gap_fill_cells"] for item in records)
        total_covered_after = sum(item["covered_owner_cells_after"] for item in records)
        all_bounds = np.asarray([item["data_bounds_xy_world_m"] for item in input_manifest["patches"]], np.float64)
        manifest = {
            "schema": "air2vectorcity.pixal3d_p384_handoff_v9_continuous_scene.v3",
            "status": "complete",
            "axis_mapping_version": source_manifest.get("axis_mapping_version"),
            "source_scene_manifest": str((args.source_fusion / "scene_manifest.json").resolve()),
            "composition": "Pixal3D generated PBR geometry over a complete restored-RGB textured filled-base height-field underlay" if args.full_underlay else "Pixal3D generated PBR geometry plus restored-RGB textured filled-base height-field triangles in uncovered owner cells",
            "fill_mode": "full_base_underlay" if args.full_underlay else "generated_gap_only",
            "ownership": "nearest patch center among patches whose data bounds cover each regular world-grid cell centre",
            "grid_m": args.grid_m,
            "patch_count_expected": len(input_by_patch),
            "patch_count_available": len(records),
            "missing_patches": [],
            "world_bounds_xy_m": [
                float(all_bounds[:, 0].min()), float(all_bounds[:, 1].min()),
                float(all_bounds[:, 2].max()), float(all_bounds[:, 3].max()),
            ],
            "owner_cells": total_owner,
            "generated_base_covered_owner_cells": total_generated,
            "generated_base_coverage_fraction": total_generated / max(total_owner, 1),
            "base_gap_fill_cells": total_fill,
            "covered_owner_cells_after": total_covered_after,
            "coverage_after_fraction": total_covered_after / max(total_owner, 1),
            "total_faces_owned": sum(item["faces"] for item in records),
            "total_tile_bytes": sum(item["size_bytes"] for item in records),
            "monolithic": None,
            "tiles": records,
        }
        (args.output / "scene_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "complete", **{k: manifest[k] for k in (
            "patch_count_available", "generated_base_coverage_fraction",
            "coverage_after_fraction", "total_faces_owned", "total_tile_bytes",
        )}}), flush=True)


if __name__ == "__main__":
    main()
