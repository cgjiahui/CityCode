#!/usr/bin/env python3
"""Crop canonical patches to global Voronoi ownership and emit a tiled scene."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--monolithic", action="store_true")
    return parser.parse_args()


def load_world_geometries(path: Path):
    scene = trimesh.load(path, force="scene", process=False)
    geometries = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        geometries.append(geometry)
    return geometries


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = args.output / "tiles"
    tiles.mkdir(exist_ok=True)
    input_manifest = json.loads((args.inputs / "manifest.json").read_text())
    expected = input_manifest["patches"]
    available = [
        item for item in expected
        if (args.runs / item["patch_id"] / "mesh_pbr_world_canonical_axis_v3.glb").is_file()
    ]
    missing = [item["patch_id"] for item in expected if item not in available]
    if missing and not args.allow_incomplete:
        raise ValueError(f"Missing {len(missing)} canonical patches: {missing}")
    centers = np.asarray([item["center_world_m"][:2] for item in available], np.float64)
    tree = cKDTree(centers)
    combined = trimesh.Scene() if args.monolithic else None
    records = []
    for index, item in enumerate(available):
        patch_id = item["patch_id"]
        source = args.runs / patch_id / "mesh_pbr_world_canonical_axis_v3.glb"
        destination = tiles / f"{patch_id}.glb"
        if destination.is_file() and destination.stat().st_size:
            cached_scene = trimesh.load(destination, force="scene", process=False)
            cached_geometries = list(cached_scene.geometry.values())
            canonical_metadata = json.loads((args.runs / patch_id / "canonical_metadata.json").read_text())
            record = {
                "patch_id": patch_id,
                "row": item["row"],
                "column": item["column"],
                "center_world_m": item["center_world_m"],
                "data_bounds_xy_world_m": item["data_bounds_xy_world_m"],
                "core_bounds_xy_world_m": item["core_bounds_xy_world_m"],
                "source": str(source),
                "tile": str(destination.resolve()),
                "tile_url": f"tiles/{patch_id}.glb",
                "faces_before": int(canonical_metadata["faces_after"]),
                "faces_owned": int(sum(len(geometry.faces) for geometry in cached_geometries)),
                "vertices_owned": int(sum(len(geometry.vertices) for geometry in cached_geometries)),
                "owned_fraction": 0.0,
                "size_bytes": destination.stat().st_size,
                "bounds": cached_scene.bounds.tolist(),
                "cached": True,
            }
            record["owned_fraction"] = record["faces_owned"] / max(record["faces_before"], 1)
            records.append(record)
            print(json.dumps({"index": index + 1, **record}), flush=True)
            del cached_scene, cached_geometries
            gc.collect()
            continue
        geometries = load_world_geometries(source)
        owned_scene = trimesh.Scene()
        faces_before = faces_after = vertices_after = 0
        for geometry_number, geometry in enumerate(geometries):
            faces_before += len(geometry.faces)
            centroids = geometry.triangles_center
            queries = np.column_stack((centroids[:, 0], centroids[:, 2]))
            _, owner = tree.query(queries, k=1)
            geometry.update_faces(owner == index)
            geometry.remove_unreferenced_vertices()
            faces_after += len(geometry.faces)
            vertices_after += len(geometry.vertices)
            if len(geometry.faces):
                name = f"{patch_id}_{geometry_number}"
                owned_scene.add_geometry(geometry, node_name=name, geom_name=name)
                if combined is not None:
                    combined.add_geometry(geometry.copy(), node_name=name, geom_name=name)
        if len(owned_scene.geometry):
            owned_scene.export(destination, extension_webp=True)
        record = {
            "patch_id": patch_id,
            "row": item["row"],
            "column": item["column"],
            "center_world_m": item["center_world_m"],
            "data_bounds_xy_world_m": item["data_bounds_xy_world_m"],
            "core_bounds_xy_world_m": item["core_bounds_xy_world_m"],
            "source": str(source),
            "tile": str(destination.resolve()) if destination.is_file() else None,
            "tile_url": f"tiles/{patch_id}.glb" if destination.is_file() else None,
            "faces_before": int(faces_before),
            "faces_owned": int(faces_after),
            "vertices_owned": int(vertices_after),
            "owned_fraction": float(faces_after / max(faces_before, 1)),
            "size_bytes": destination.stat().st_size if destination.is_file() else 0,
            "bounds": owned_scene.bounds.tolist() if len(owned_scene.geometry) else None,
        }
        records.append(record)
        print(json.dumps({"index": index + 1, **record}), flush=True)
        del owned_scene, geometries
        gc.collect()

    monolithic_path = args.output / "full_scene_single_owner.glb"
    if combined is not None:
        combined.export(monolithic_path, extension_webp=True)
    all_bounds = np.asarray([item["data_bounds_xy_world_m"] for item in available], np.float64)
    manifest = {
        "schema": "air2vectorcity.pixal3d_p384_v5_tiled_scene.v2",
        "axis_mapping_version": "pbr_local_z_to_ue_y_positive_v3",
        "status": "complete" if not missing else "partial",
        "ownership": "nearest patch center Voronoi; exactly one source mesh per face",
        "patch_count_expected": len(expected),
        "patch_count_available": len(available),
        "missing_patches": missing,
        "world_bounds_xy_m": [
            float(all_bounds[:, 0].min()), float(all_bounds[:, 1].min()),
            float(all_bounds[:, 2].max()), float(all_bounds[:, 3].max()),
        ],
        "total_faces_before": sum(item["faces_before"] for item in records),
        "total_faces_owned": sum(item["faces_owned"] for item in records),
        "total_vertices_owned": sum(item["vertices_owned"] for item in records),
        "total_tile_bytes": sum(item["size_bytes"] for item in records),
        "monolithic": str(monolithic_path) if monolithic_path.is_file() else None,
        "tiles": records,
    }
    (args.output / "scene_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "patch_count": len(available),
        "total_faces_owned": manifest["total_faces_owned"],
        "total_tile_bytes": manifest["total_tile_bytes"],
    }), flush=True)


if __name__ == "__main__":
    main()
