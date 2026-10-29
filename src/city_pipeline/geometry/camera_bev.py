#!/usr/bin/env python3
"""Build the full camera-aligned, instance-aware 2.5D E03 experiment.

This experiment fixes the orthographic raster to a 256 m square whose axes
follow the source camera's image-right and image-down directions projected on
UE world XY.  It reuses the frozen E01 calibrated depth, cleaned source IDs and
all conservative BEV denoising thresholds.
"""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.roof_masks import two_row_comparison
from city_pipeline.geometry.camera_geometry import (
    height_preview,
    lift_camera_forward_depth,
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.bev_grid import (
    camera_aligned_axes,
    rasterize_local,
    world_frame_diagram,
)
from city_pipeline.geometry.depth_lifting import (
    build_overview,
    finite_distribution,
    instance_palette,
    instance_preview,
    mask_rgb,
    rgb_overlay,
    save_u16,
    write_binary_ply,
)
from city_pipeline.geometry.depth_denoising import (
    BEV_MAX_HOLE_AREA_M2,
    BEV_MIN_COMPONENT_AREA_M2,
    SOURCE_MIN_COMPONENT_AREA_1024PX,
    SOURCE_OPENING_SIZE,
    clean_bev_ids,
    provenance_preview,
)


DEFAULT_BASELINE_ROOT = Path("UNCONFIGURED_DEFAULT_BASELINE_ROOT")
DEFAULT_E01_ROOT = DEFAULT_BASELINE_ROOT / "denoising_v1"
DEFAULT_E02_ROOT = DEFAULT_BASELINE_ROOT / "orientation_extent_ablation_e02"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASELINE_ROOT / "camera_aligned_25d_e03"
CELL_SIZE_M = 0.125
HALF_EXTENT_M = 128.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--e01-root", type=Path, default=DEFAULT_E01_ROOT)
    parser.add_argument("--e02-root", type=Path, default=DEFAULT_E02_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def xy_for_mask(
    mask: np.ndarray,
    pixel_center_to_world: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, u = np.nonzero(mask)
    matrix = np.asarray(pixel_center_to_world, dtype=np.float64)
    x = matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2]
    y = matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2]
    return x.astype(np.float32), y.astype(np.float32), (v, u)


def building_point_cloud(
    surface_z: np.ndarray,
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    provenance: np.ndarray,
    pixel_center_to_world: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = (instance_ids > 0) & np.isfinite(surface_z)
    x, y, indices = xy_for_mask(selected, pixel_center_to_world)
    xyz = np.stack((x, y, surface_z[selected].astype(np.float32)), axis=1)
    return (
        xyz,
        rgb[selected].astype(np.uint8),
        instance_ids[selected].astype(np.uint16),
        provenance[selected].astype(np.uint8),
    )


def per_instance_statistics(
    source_ids: np.ndarray,
    bev_ids: np.ndarray,
    surface_z: np.ndarray,
    ground_reference_z_m: float,
    pixel_center_to_world: list[list[float]],
    cell_size_m: float = CELL_SIZE_M,
) -> list[dict[str, Any]]:
    source_counts = np.bincount(source_ids.reshape(-1))
    records: list[dict[str, Any]] = []
    for instance_id in np.unique(bev_ids):
        if instance_id == 0:
            continue
        mask = (bev_ids == instance_id) & np.isfinite(surface_z)
        x, y, _ = xy_for_mask(mask, pixel_center_to_world)
        z = surface_z[mask]
        records.append(
            {
                "instance_id": int(instance_id),
                "source_pixel_count": (
                    int(source_counts[instance_id])
                    if instance_id < source_counts.size
                    else 0
                ),
                "observed_bev_cell_count": int(z.size),
                "observed_bev_area_m2": float(z.size * cell_size_m * cell_size_m),
                "observed_world_z_m": finite_distribution(z),
                "observed_height_above_reference_ground_m": finite_distribution(
                    z - np.float32(ground_reference_z_m)
                ),
                "observed_xy_centroid_ue_world_m": [
                    float(np.mean(x)),
                    float(np.mean(y)),
                ],
                "observed_xy_bbox_ue_world_m": [
                    float(np.min(x)),
                    float(np.min(y)),
                    float(np.max(x)),
                    float(np.max(y)),
                ],
            }
        )
    return records


def save_float(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def main() -> int:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    e01_root = args.e01_root.resolve()
    e02_root = args.e02_root.resolve()
    output_root = args.output_root.resolve()
    baseline = read_json(baseline_root / "manifest.json")
    e01 = read_json(e01_root / "manifest.json")
    baseline_by_id = {sample["id"]: sample for sample in baseline["samples"]}
    e01_by_id = {sample["id"]: sample for sample in e01["samples"]}
    sample_ids = args.sample_id or [sample["id"] for sample in baseline["samples"]]

    directory_names = (
        "bev_rgb",
        "bev_validity",
        "bev_selected_source_depth_zc_m",
        "bev_selected_source_depth_previews",
        "bev_surface_z_m",
        "bev_surface_z_previews",
        "bev_height_above_reference_ground_m",
        "bev_height_above_reference_ground_previews",
        "bev_building_height_m",
        "bev_building_height_previews",
        "bev_instance_ids_raw",
        "bev_instance_ids_component_cleaned",
        "bev_instance_ids_final",
        "bev_instance_previews_raw",
        "bev_instance_previews_final",
        "bev_instance_overlays_final",
        "bev_cleaning_provenance",
        "bev_cleaning_provenance_previews",
        "pixel_to_ue_world_transforms",
        "building_point_cloud_npz",
        "building_point_cloud_ply_instance",
        "per_instance_statistics",
        "comparisons",
    )
    directories = {name: output_root / name for name in directory_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        base_sample = baseline_by_id[sample_id]
        e01_sample = e01_by_id[sample_id]
        source_rgb = load_rgb(project_path(base_sample["input"]["rgb"]))
        camera = read_json(project_path(base_sample["input"]["camera"]))
        calibrated_depth = np.load(project_path(e01_sample["outputs"]["depth"])).astype(
            np.float32
        )
        with Image.open(project_path(e01_sample["outputs"]["source_ids"])) as image:
            source_ids = np.asarray(image, dtype=np.uint16)
        calibration_mask = load_rgb(project_path(e01_sample["outputs"]["calibration"]))
        source_overlay = load_rgb(project_path(e01_sample["outputs"]["source_overlay"]))
        source_instance_preview = load_rgb(project_path(e01_sample["outputs"]["source_preview"]))

        world_xyz, lifting = lift_camera_forward_depth(calibrated_depth, camera)
        world_x, world_y, world_z = world_xyz
        e_u, e_v = camera_aligned_axes(camera)
        center_xy = np.asarray(camera["target_center_world_m"][:2], dtype=np.float64)
        recovered, valid, surface_z, rasterization = rasterize_local(
            {
                "rgb": source_rgb,
                "instance_id": source_ids,
                "source_depth_zc_m": calibrated_depth,
            },
            world_xyz,
            center_xy,
            e_u,
            e_v,
            (-HALF_EXTENT_M, -HALF_EXTENT_M, HALF_EXTENT_M, HALF_EXTENT_M),
        )
        raw_ids = recovered["instance_id"].astype(np.uint16)
        selected_depth = recovered["source_depth_zc_m"].astype(np.float32)
        selected_depth[~valid] = np.nan
        (
            component_ids,
            final_ids,
            clean_surface_z,
            clean_rgb,
            provenance,
            cleaning,
        ) = clean_bev_ids(
            raw_ids,
            valid,
            surface_z,
            recovered["rgb"],
            (CELL_SIZE_M, CELL_SIZE_M),
        )

        ground_reference_z = float(camera["ground_reference_z_m"])
        final_surface_valid = np.isfinite(clean_surface_z)
        surface_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
        surface_height[final_surface_valid] = (
            clean_surface_z[final_surface_valid] - np.float32(ground_reference_z)
        )
        building_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
        building_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
        building_height[building_valid] = (
            clean_surface_z[building_valid] - np.float32(ground_reference_z)
        )

        raw_preview = instance_preview(raw_ids)
        final_preview = instance_preview(final_ids)
        final_overlay = rgb_overlay(clean_rgb, final_preview, building_valid)
        depth_preview = height_preview(selected_depth, valid)
        depth_preview[~valid] = 0
        z_preview = height_preview(clean_surface_z, final_surface_valid)
        z_preview[~final_surface_valid] = 0
        surface_height_preview = height_preview(surface_height, final_surface_valid)
        surface_height_preview[~final_surface_valid] = 0
        building_height_preview = height_preview(building_height, building_valid)
        building_height_preview[~building_valid] = 0
        provenance_rgb = provenance_preview(provenance)

        transform_path = (
            directories["pixel_to_ue_world_transforms"]
            / f"{sample_id}__pixel_to_ue_world.json"
        )
        transform_path.write_text(
            json.dumps(rasterization, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths = {
            "bev_rgb": directories["bev_rgb"] / f"{sample_id}__bev_rgb.png",
            "validity": directories["bev_validity"] / f"{sample_id}__bev_validity.png",
            "selected_depth": directories["bev_selected_source_depth_zc_m"] / f"{sample_id}__bev_selected_source_depth_zc_m.npy",
            "selected_depth_preview": directories["bev_selected_source_depth_previews"] / f"{sample_id}__bev_selected_source_depth_preview.png",
            "surface_z": directories["bev_surface_z_m"] / f"{sample_id}__bev_surface_z_m.npy",
            "surface_z_preview": directories["bev_surface_z_previews"] / f"{sample_id}__bev_surface_z_preview.png",
            "surface_height": directories["bev_height_above_reference_ground_m"] / f"{sample_id}__bev_height_above_reference_ground_m.npy",
            "surface_height_preview": directories["bev_height_above_reference_ground_previews"] / f"{sample_id}__bev_height_above_reference_ground_preview.png",
            "building_height": directories["bev_building_height_m"] / f"{sample_id}__bev_building_height_m.npy",
            "building_height_preview": directories["bev_building_height_previews"] / f"{sample_id}__bev_building_height_preview.png",
            "raw_ids": directories["bev_instance_ids_raw"] / f"{sample_id}__bev_instance_ids_raw.png",
            "component_ids": directories["bev_instance_ids_component_cleaned"] / f"{sample_id}__bev_instance_ids_component_cleaned.png",
            "final_ids": directories["bev_instance_ids_final"] / f"{sample_id}__bev_instance_ids_final.png",
            "raw_preview": directories["bev_instance_previews_raw"] / f"{sample_id}__bev_instance_preview_raw.png",
            "final_preview": directories["bev_instance_previews_final"] / f"{sample_id}__bev_instance_preview_final.png",
            "overlay": directories["bev_instance_overlays_final"] / f"{sample_id}__bev_instance_overlay_final.png",
            "provenance": directories["bev_cleaning_provenance"] / f"{sample_id}__bev_cleaning_provenance.png",
            "provenance_preview": directories["bev_cleaning_provenance_previews"] / f"{sample_id}__bev_cleaning_provenance_preview.png",
            "transform": transform_path,
            "point_npz": directories["building_point_cloud_npz"] / f"{sample_id}__building_points_camera_aligned.npz",
            "point_ply": directories["building_point_cloud_ply_instance"] / f"{sample_id}__building_instance_camera_aligned.ply",
            "stats": directories["per_instance_statistics"] / f"{sample_id}__per_instance_statistics.json",
            "comparison": directories["comparisons"] / f"{sample_id}__camera_aligned_25d_e03.png",
        }
        save_rgb(paths["bev_rgb"], clean_rgb)
        save_u8(paths["validity"], valid.astype(np.uint8) * 255)
        save_float(paths["selected_depth"], selected_depth)
        save_rgb(paths["selected_depth_preview"], depth_preview)
        save_float(paths["surface_z"], clean_surface_z)
        save_rgb(paths["surface_z_preview"], z_preview)
        save_float(paths["surface_height"], surface_height)
        save_rgb(paths["surface_height_preview"], surface_height_preview)
        save_float(paths["building_height"], building_height)
        save_rgb(paths["building_height_preview"], building_height_preview)
        save_u16(paths["raw_ids"], raw_ids)
        save_u16(paths["component_ids"], component_ids)
        save_u16(paths["final_ids"], final_ids)
        save_rgb(paths["raw_preview"], raw_preview)
        save_rgb(paths["final_preview"], final_preview)
        save_rgb(paths["overlay"], final_overlay)
        save_u8(paths["provenance"], provenance)
        save_rgb(paths["provenance_preview"], provenance_rgb)

        xyz, point_rgb, point_ids, point_provenance = building_point_cloud(
            clean_surface_z,
            clean_rgb,
            final_ids,
            provenance,
            rasterization["pixel_center_index_to_ue_world_xy_3x3"],
        )
        paths["point_npz"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths["point_npz"],
            xyz_m=xyz,
            rgb=point_rgb,
            instance_id=point_ids,
            provenance=point_provenance,
        )
        point_instance_rgb = instance_palette(int(np.max(point_ids)))[point_ids]
        ply = write_binary_ply(
            paths["point_ply"], xyz, point_instance_rgb, point_ids, args.max_ply_points
        )
        statistics = per_instance_statistics(
            source_ids,
            final_ids,
            clean_surface_z,
            ground_reference_z,
            rasterization["pixel_center_index_to_ue_world_xy_3x3"],
        )
        paths["stats"].write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        finite_world = np.isfinite(world_x) & np.isfinite(world_y) & np.isfinite(world_z)
        accepted_xy = np.stack((world_x[finite_world], world_y[finite_world]), axis=1)
        frame = world_frame_diagram(
            [float(value) for value in camera["target_patch_bounds_xy_m"]],
            rasterization["raster_footprint_world_xy_m"],
            accepted_xy,
        )
        source_depth_preview = height_preview(
            calibrated_depth, np.isfinite(calibrated_depth) & (calibrated_depth > 0)
        )
        two_row_comparison(
            f"{sample_id} / E03 CAMERA-ALIGNED INSTANCE-AWARE 2.5D",
            [
                ("S1. Input RGB", "2048 source view", source_rgb, False),
                ("S2. Clean source IDs", f"E01 {SOURCE_OPENING_SIZE}x{SOURCE_OPENING_SIZE}; >={SOURCE_MIN_COMPONENT_AREA_1024PX}px", source_instance_preview, True),
                ("S3. Source overlay", "cleaned V9.2 buildings", source_overlay, False),
                ("S4. Calibrated depth", "E01 class-aware camera zc", source_depth_preview, False),
                ("S5. Scale anchors", "road full; water/hard cores", calibration_mask, True),
                ("S6. World footprint", "white canonical / orange E03", frame, False),
            ],
            [
                ("B1. Camera-aligned RGB", f"coverage={rasterization['raw_target_valid_fraction']:.1%}", clean_rgb, False),
                ("B2. Validity", "white=observed", mask_rgb(valid), True),
                ("B3. Selected source depth", "camera-forward zc; audit layer", depth_preview, False),
                ("B4. Raw BEV IDs", "before E01 BEV cleaning", raw_preview, True),
                ("B5. Final BEV IDs", f">={BEV_MIN_COMPONENT_AREA_M2:g}m2; holes<={BEV_MAX_HOLE_AREA_M2:g}m2", final_preview, True),
                ("B6. Building height", "world Z - reference Z", building_height_preview, False),
            ],
            paths["comparison"],
            tile=320,
        )
        comparisons.append(paths["comparison"])

        e02_validation: dict[str, Any] = {"available": False}
        e02_rgb_path = e02_root / "b_camera_fixed" / "bev_rgb" / f"{sample_id}__bev_rgb.png"
        if e02_rgb_path.exists():
            e02_surface_path = e02_root / "b_camera_fixed" / "surface_fields" / f"{sample_id}__surface_fields.npz"
            with Image.open(e02_rgb_path) as image:
                e02_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with Image.open(
                e02_root / "b_camera_fixed" / "bev_validity" / f"{sample_id}__bev_validity.png"
            ) as image:
                e02_valid = np.asarray(image) > 0
            with Image.open(
                e02_root / "b_camera_fixed" / "bev_instance_ids" / f"{sample_id}__bev_instance_ids.png"
            ) as image:
                e02_ids = np.asarray(image, dtype=np.uint16)
            with np.load(e02_surface_path) as archive:
                e02_surface = archive["surface_z_m"]
            e02_validation = {
                "available": True,
                "rgb_equal": bool(np.array_equal(clean_rgb, e02_rgb)),
                "validity_equal": bool(np.array_equal(valid, e02_valid)),
                "final_instance_ids_equal": bool(np.array_equal(final_ids, e02_ids)),
                "surface_z_equal_nan_aware": bool(
                    np.allclose(clean_surface_z, e02_surface, rtol=0.0, atol=0.0, equal_nan=True)
                ),
            }

        records.append(
            {
                "id": sample_id,
                "input": {
                    "rgb": base_sample["input"]["rgb"],
                    "camera": base_sample["input"]["camera"],
                    "e01_calibrated_depth": e01_sample["outputs"]["depth"],
                    "e01_cleaned_source_ids": e01_sample["outputs"]["source_ids"],
                    "e01_calibration_scale": e01_sample["calibration"]["scale"],
                },
                "lifting": lifting,
                "rasterization": rasterization,
                "cleaning": cleaning,
                "valid_surface_distribution_world_z_m": finite_distribution(
                    clean_surface_z[valid]
                ),
                "building_height_distribution_m": finite_distribution(
                    building_height[building_valid]
                ),
                "final_instance_count": int(np.unique(final_ids[final_ids > 0]).size),
                "building_point_count": int(xyz.shape[0]),
                "ply": ply,
                "e02_b_exact_validation": e02_validation,
                "outputs": {
                    **{
                        key: path.relative_to(PROJECT_ROOT).as_posix()
                        for key, path in paths.items()
                    },
                    # Stable semantic aliases consumed by the generic point
                    # cloud renderer and later pipeline stages.
                    "building_point_cloud_npz": paths["point_npz"]
                    .relative_to(PROJECT_ROOT)
                    .as_posix(),
                    "building_point_cloud_ply_instance": paths["point_ply"]
                    .relative_to(PROJECT_ROOT)
                    .as_posix(),
                },
            }
        )

    overview = output_root / "all_8_camera_aligned_25d_e03.png"
    build_overview(comparisons, overview)
    manifest = {
        "schema": "air2vectorcity.camera_aligned_instance_aware_25d_e03.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "completed_experiment_not_yet_promoted_to_baseline",
        "purpose": "full camera-aligned fixed-256m orthographic RGB, instances, heights and UE-world building points",
        "frozen_e01_root": e01_root.relative_to(PROJECT_ROOT).as_posix(),
        "policy": {
            "raster_frame": "camera right/image-down projected to UE world XY",
            "extent": "fixed 256m square centered at camera target center",
            "resolution_m_per_px": CELL_SIZE_M,
            "output_size_px": [2048, 2048],
            "source_instances": f"E01 {SOURCE_OPENING_SIZE}x{SOURCE_OPENING_SIZE} opening; >= {SOURCE_MIN_COMPONENT_AREA_1024PX}px residual components",
            "calibration": "frozen E01 full road; 3px water/hardscape core; asset-height priors",
            "bev_components_min_area_m2": BEV_MIN_COMPONENT_AREA_M2,
            "bev_max_filled_hole_area_m2": BEV_MAX_HOLE_AREA_M2,
            "height_smoothing": False,
            "world_coordinates": "all point outputs remain UE world XYZ",
        },
        "overview": overview.relative_to(PROJECT_ROOT).as_posix(),
        "samples": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
