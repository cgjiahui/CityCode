#!/usr/bin/env python3
"""Conservative denoising for the instance-aware observed 2.5D baseline.

The script never overwrites the raw baseline.  It performs three bounded edits:

1. per-instance 3x3 opening in the 1024 source label grid, followed by removal
   of residual components smaller than 16 pixels;
2. full (non-eroded) V7B road anchors, while water and hardscape retain a
   three-pixel internal core for class-aware depth calibration;
3. per-instance BEV connected-component filtering and filling only small holes
   fully enclosed by one instance.  Existing heights are never smoothed and
   the main BEV surfaces are not morphologically eroded.

Every removed or filled pixel is saved as an explicit mask/provenance code.
"""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT
from city_pipeline.geometry.priors import read_plane_priors

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.roof_masks import two_row_comparison
from city_pipeline.geometry.camera_geometry import (
    height_preview,
    lift_camera_forward_depth,
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
    zbuffer_world_xy,
)
from city_pipeline.geometry.depth_lifting import (
    CALIBRATION_CLASSES,
    EXCLUDED_CALIBRATION_CLASSES,
    V7B_CLASS_NAMES,
    V7B_PALETTE,
    bev_point_cloud,
    class_aware_scale,
    calibration_planes,
    class_core_preview,
    finite_distribution,
    instance_palette,
    instance_preview,
    mask_rgb,
    per_instance_statistics,
    relative,
    resize_nearest,
    rgb_overlay,
    save_u16,
    write_binary_ply,
)


DEFAULT_BASELINE_ROOT = Path("UNCONFIGURED_DEFAULT_BASELINE_ROOT")
DEFAULT_OUTPUT_ROOT = DEFAULT_BASELINE_ROOT / "denoising_v1"

SOURCE_OPENING_SIZE = 3
SOURCE_MIN_COMPONENT_AREA_1024PX = 16
BEV_MIN_COMPONENT_AREA_M2 = 2.0
BEV_MAX_HOLE_AREA_M2 = 2.0

PROVENANCE_COLORS = np.asarray(
    [
        [0, 0, 0],       # 0 nonbuilding / unknown
        [60, 200, 90],    # 1 raw observed building retained
        [255, 210, 40],   # 2 observed surface assigned into an enclosed ID hole
        [255, 80, 190],   # 3 geometry-filled enclosed hole
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservatively clean source and BEV instance labels."
    )
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-cached-smoke-recalibration", action="store_true")
    parser.add_argument("--priors-path", type=Path, help="Explicit fixed-plane priors JSON")
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument(
        "--calibration-plane-policy",
        choices=("fixed-smallcity", "camera-ground"),
        default="fixed-smallcity",
    )
    parser.add_argument(
        "--anchor-erosion-radius-px",
        type=int,
        default=3,
        help="Erosion radius for non-road calibration anchors at source resolution.",
    )
    return parser.parse_args()


def clean_source_ids(ids_native: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    native_size = (int(ids_native.shape[1]), int(ids_native.shape[0]))
    ids_1024 = resize_nearest(ids_native, (1024, 1024)).astype(np.uint16)
    clean = np.zeros_like(ids_1024)
    structure = np.ones((SOURCE_OPENING_SIZE, SOURCE_OPENING_SIZE), dtype=bool)
    per_instance: list[dict[str, Any]] = []
    for instance_id in np.unique(ids_1024):
        if instance_id == 0:
            continue
        raw = ids_1024 == instance_id
        opened = ndimage.binary_opening(raw, structure=structure)
        labels, count = ndimage.label(opened, structure=structure)
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= SOURCE_MIN_COMPONENT_AREA_1024PX
        if keep.size:
            keep[0] = False
        if not np.any(keep) and count:
            keep[int(np.argmax(sizes[1:])) + 1] = True
        retained = keep[labels]
        clean[retained] = instance_id
        per_instance.append(
            {
                "instance_id": int(instance_id),
                "raw_pixel_count_1024": int(np.count_nonzero(raw)),
                "retained_pixel_count_1024_before_restricted_closing": int(
                    np.count_nonzero(retained)
                ),
                "opened_component_count": int(count),
                "retained_component_count": int(np.count_nonzero(keep)),
            }
        )

    # Recover tiny indentations only inside the same original label.  This can
    # never expand an instance into a road, background or neighbouring ID.
    for instance_id in np.unique(clean):
        if instance_id == 0:
            continue
        closed = ndimage.binary_closing(clean == instance_id, structure=structure)
        recover = closed & (ids_1024 == instance_id) & (clean == 0)
        clean[recover] = instance_id

    removed_1024 = (ids_1024 > 0) & (clean == 0)
    clean_native = resize_nearest(clean, native_size).astype(np.uint16)
    removed_native = resize_nearest(
        removed_1024.astype(np.uint8), native_size
    ).astype(bool)
    return clean_native, removed_native, {
        "method": "per-instance 3x3 opening + >=16px residual components + same-label restricted closing",
        "raw_building_pixel_count_1024": int(np.count_nonzero(ids_1024)),
        "clean_building_pixel_count_1024": int(np.count_nonzero(clean)),
        "removed_pixel_count_1024": int(np.count_nonzero(removed_1024)),
        "removed_fraction_of_raw_building": float(
            np.count_nonzero(removed_1024) / max(1, np.count_nonzero(ids_1024))
        ),
        "raw_instance_count": int(np.unique(ids_1024[ids_1024 > 0]).size),
        "clean_instance_count": int(np.unique(clean[clean > 0]).size),
        "per_instance": per_instance,
    }


def road_full_other_core_masks(
    class_ids: np.ndarray,
    active_classes: tuple[int, ...],
    erosion_radius_px: int,
) -> dict[int, np.ndarray]:
    if erosion_radius_px < 0:
        raise ValueError("anchor erosion radius cannot be negative")
    structure = np.ones(
        (2 * erosion_radius_px + 1, 2 * erosion_radius_px + 1), dtype=bool
    )
    cores: dict[int, np.ndarray] = {}
    for class_id in active_classes:
        mask = class_ids == class_id
        cores[class_id] = (
            mask
            if class_id == 1 or erosion_radius_px == 0
            else ndimage.binary_erosion(mask, structure=structure, border_value=0)
        )
    return cores


def clean_bev_ids(
    raw_ids: np.ndarray,
    valid: np.ndarray,
    surface_z: np.ndarray,
    bev_rgb: np.ndarray,
    meters_per_pixel: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove thin/small instance noise and conservatively fill enclosed holes."""

    pixel_area_m2 = float(meters_per_pixel[0] * meters_per_pixel[1])
    min_component_pixels = max(1, int(math.ceil(BEV_MIN_COMPONENT_AREA_M2 / pixel_area_m2)))
    max_hole_pixels = max(1, int(math.floor(BEV_MAX_HOLE_AREA_M2 / pixel_area_m2)))
    component_clean = np.zeros_like(raw_ids, dtype=np.uint16)
    component_records: list[dict[str, Any]] = []

    for instance_id in np.unique(raw_ids):
        if instance_id == 0:
            continue
        raw = raw_ids == instance_id
        raw_y, raw_x = np.nonzero(raw)
        y0 = max(0, int(np.min(raw_y)) - 1)
        y1 = min(raw.shape[0], int(np.max(raw_y)) + 2)
        x0 = max(0, int(np.min(raw_x)) - 1)
        x1 = min(raw.shape[1], int(np.max(raw_x)) + 2)
        crop = raw[y0:y1, x0:x1]
        labels, count = ndimage.label(
            crop,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= min_component_pixels
        if keep.size:
            keep[0] = False
        # Preserve the largest observed part even when a building is only
        # partially visible and smaller than the physical threshold.
        if count and not np.any(keep):
            keep[int(np.argmax(sizes[1:])) + 1] = True
        retained = keep[labels]
        output_crop = component_clean[y0:y1, x0:x1]
        output_crop[retained] = instance_id
        component_records.append(
            {
                "instance_id": int(instance_id),
                "raw_cell_count": int(np.count_nonzero(raw)),
                "retained_cell_count": int(np.count_nonzero(retained)),
                "raw_component_count": int(count),
                "retained_component_count": int(np.count_nonzero(keep)),
            }
        )

    removed_noise = (raw_ids > 0) & (component_clean == 0)
    final_ids = component_clean.copy()
    observed_id_fill = np.zeros(raw_ids.shape, dtype=bool)
    geometry_hole_fill = np.zeros(raw_ids.shape, dtype=bool)

    # Only fill a hole when it is fully enclosed by one instance, small in
    # physical area, and had no competing raw instance ID.
    for instance_id in np.unique(component_clean):
        if instance_id == 0:
            continue
        mask = component_clean == instance_id
        mask_y, mask_x = np.nonzero(mask)
        y0 = max(0, int(np.min(mask_y)) - 1)
        y1 = min(mask.shape[0], int(np.max(mask_y)) + 2)
        x0 = max(0, int(np.min(mask_x)) - 1)
        x1 = min(mask.shape[1], int(np.max(mask_x)) + 2)
        mask_crop = mask[y0:y1, x0:x1]
        enclosed = ndimage.binary_fill_holes(mask_crop) & ~mask_crop
        hole_labels, hole_count = ndimage.label(
            enclosed,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        hole_sizes = np.bincount(hole_labels.reshape(-1))
        for hole_id in range(1, hole_count + 1):
            if hole_sizes[hole_id] > max_hole_pixels:
                continue
            hole_crop = hole_labels == hole_id
            raw_crop = raw_ids[y0:y1, x0:x1]
            final_crop = final_ids[y0:y1, x0:x1]
            if np.any(raw_crop[hole_crop] > 0) or np.any(final_crop[hole_crop] > 0):
                continue
            final_crop[hole_crop] = instance_id
            observed_crop = (
                hole_crop
                & valid[y0:y1, x0:x1]
                & np.isfinite(surface_z[y0:y1, x0:x1])
            )
            missing_crop = hole_crop & ~observed_crop
            observed_id_fill[y0:y1, x0:x1] |= observed_crop
            geometry_hole_fill[y0:y1, x0:x1] |= missing_crop

    clean_surface_z = surface_z.copy()
    clean_rgb = bev_rgb.copy()
    # Interpolate only genuinely missing geometry in tiny enclosed holes.
    # Observed surfaces retain their original unsmoothed height values.
    for instance_id in np.unique(final_ids[geometry_hole_fill]):
        if instance_id == 0:
            continue
        target = geometry_hole_fill & (final_ids == instance_id)
        combined = target | ((final_ids == instance_id) & np.isfinite(clean_surface_z))
        all_y, all_x = np.nonzero(combined)
        y0 = max(0, int(np.min(all_y)) - 1)
        y1 = min(target.shape[0], int(np.max(all_y)) + 2)
        x0 = max(0, int(np.min(all_x)) - 1)
        x1 = min(target.shape[1], int(np.max(all_x)) + 2)
        target_crop = target[y0:y1, x0:x1]
        source_crop = (
            (final_ids[y0:y1, x0:x1] == instance_id)
            & np.isfinite(clean_surface_z[y0:y1, x0:x1])
            & ~target_crop
        )
        if not np.any(source_crop):
            final_ids[y0:y1, x0:x1][target_crop] = 0
            geometry_hole_fill[y0:y1, x0:x1][target_crop] = False
            continue
        _, nearest = ndimage.distance_transform_edt(~source_crop, return_indices=True)
        z_crop = clean_surface_z[y0:y1, x0:x1]
        rgb_crop = clean_rgb[y0:y1, x0:x1]
        z_crop[target_crop] = z_crop[nearest[0][target_crop], nearest[1][target_crop]]
        rgb_crop[target_crop] = rgb_crop[nearest[0][target_crop], nearest[1][target_crop]]

    final_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
    provenance = np.zeros(raw_ids.shape, dtype=np.uint8)
    provenance[(final_ids > 0) & (raw_ids == final_ids) & valid] = 1
    provenance[observed_id_fill] = 2
    provenance[geometry_hole_fill] = 3
    return (
        component_clean,
        final_ids,
        clean_surface_z,
        clean_rgb,
        provenance,
        {
            "method": "per-instance physical component filtering without BEV morphology + small enclosed-hole fill",
            "meters_per_pixel": list(meters_per_pixel),
            "pixel_area_m2": pixel_area_m2,
            "minimum_component_area_m2": BEV_MIN_COMPONENT_AREA_M2,
            "minimum_component_pixels": min_component_pixels,
            "maximum_filled_hole_area_m2": BEV_MAX_HOLE_AREA_M2,
            "maximum_filled_hole_pixels": max_hole_pixels,
            "raw_instance_cell_count": int(np.count_nonzero(raw_ids)),
            "component_clean_cell_count": int(np.count_nonzero(component_clean)),
            "final_instance_cell_count": int(np.count_nonzero(final_ids)),
            "removed_noise_cell_count": int(np.count_nonzero(removed_noise)),
            "removed_noise_area_m2": float(np.count_nonzero(removed_noise) * pixel_area_m2),
            "observed_id_hole_fill_cell_count": int(np.count_nonzero(observed_id_fill)),
            "observed_id_hole_fill_area_m2": float(
                np.count_nonzero(observed_id_fill) * pixel_area_m2
            ),
            "geometry_hole_fill_cell_count": int(np.count_nonzero(geometry_hole_fill)),
            "geometry_hole_fill_area_m2": float(
                np.count_nonzero(geometry_hole_fill) * pixel_area_m2
            ),
            "final_valid_building_cell_count": int(np.count_nonzero(final_valid)),
            "height_smoothing_applied": False,
            "component_records": component_records,
        },
    )


def provenance_preview(provenance: np.ndarray) -> np.ndarray:
    return PROVENANCE_COLORS[provenance]


def build_overview(paths: list[Path], destination: Path) -> None:
    columns = 2
    tile_width = 1500
    tile_height = 650
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), (15, 17, 20))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width + (tile_width - preview.width) // 2
        y = (index // columns) * tile_height + (tile_height - preview.height) // 2
        canvas.paste(preview, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()
    baseline = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))

    names = (
        "source_instance_ids_cleaned",
        "source_instance_previews_cleaned",
        "source_removed_noise_masks",
        "source_instance_overlays_cleaned",
        "calibration_masks_road_full",
        "depth_road_full_class_aware_zc_m",
        "bev_instance_ids_raw_from_clean_source",
        "bev_instance_ids_component_cleaned",
        "bev_instance_ids_final",
        "bev_instance_previews_final",
        "bev_removed_noise_masks",
        "bev_observed_id_hole_fill_masks",
        "bev_geometry_hole_fill_masks",
        "bev_cleaning_provenance",
        "bev_building_height_cleaned_m",
        "bev_building_height_previews_cleaned",
        "bev_instance_overlays_cleaned",
        "point_cloud_npz_cleaned",
        "point_cloud_ply_instance_cleaned",
        "per_instance_statistics_cleaned",
        "comparisons",
    )
    directories = {name: output_root / name for name in names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    for index, sample in enumerate(baseline["samples"], start=1):
        if sample["input"].get("depth_input_stage") == "already_calibrated_prediction" and not args.allow_cached_smoke_recalibration:
            raise ValueError("Already-calibrated fixture replay must be explicitly requested")
        sample_id = sample["id"]
        print(f"[{index}/{len(baseline['samples'])}] {sample_id}", flush=True)
        source_rgb = load_rgb(PROJECT_ROOT / sample["input"]["rgb"])
        camera = read_json(PROJECT_ROOT / sample["input"]["camera"])
        raw_depth = np.load(
            PROJECT_ROOT / sample["input"]["aerialmetric_raw_depth_zc_m"]
        ).astype(np.float32, copy=False)
        raw_source_ids = np.asarray(
            Image.open(PROJECT_ROOT / sample["outputs"]["source_instance_ids"])
        ).astype(np.uint16)
        v7b_class_ids = np.asarray(
            Image.open(PROJECT_ROOT / sample["outputs"]["v7b_class_ids"])
        ).astype(np.uint8)
        v7b_normalized = load_rgb(PROJECT_ROOT / sample["outputs"]["v7b_normalized_mask"])

        cleaned_source_ids, source_removed, source_clean_metadata = clean_source_ids(
            raw_source_ids
        )
        source_preview_raw = instance_preview(raw_source_ids)
        source_preview_clean = instance_preview(cleaned_source_ids)
        source_overlay_clean = rgb_overlay(
            source_rgb, source_preview_clean, cleaned_source_ids > 0
        )

        priors_path = args.priors_path
        if args.calibration_plane_policy == "fixed-smallcity" and priors_path is None:
            value = baseline.get("construction", {}).get("priors_json")
            if not value:
                raise ValueError("Fixed calibration requires an explicit priors path")
            priors_path = PROJECT_ROOT / value
        configured_priors = read_plane_priors(priors_path) if priors_path else None
        sample_plane_z_m = calibration_planes(camera, args.calibration_plane_policy, configured_priors)
        cores = road_full_other_core_masks(
            v7b_class_ids,
            tuple(sorted(sample_plane_z_m)),
            args.anchor_erosion_radius_px,
        )
        calibration_preview = class_core_preview(cores)
        scale, scale_metadata = class_aware_scale(
            raw_depth, camera, cores, 200_000, sample_plane_z_m
        )
        depth = (raw_depth * np.float32(scale)).astype(np.float32)
        world_xyz, lift_metadata = lift_camera_forward_depth(depth, camera)
        recovered, valid, surface_z, rasterization = zbuffer_world_xy(
            {"rgb": source_rgb, "instance_id": cleaned_source_ids},
            world_xyz,
            camera["target_patch_bounds_xy_m"],
            tuple(int(value) for value in camera["image_size_px"]),
        )
        raw_bev_ids = recovered["instance_id"].astype(np.uint16)
        meters_per_pixel = tuple(
            float(value) for value in rasterization["meters_per_pixel"]
        )
        (
            component_clean_ids,
            final_ids,
            clean_surface_z,
            clean_bev_rgb,
            provenance,
            bev_clean_metadata,
        ) = clean_bev_ids(
            raw_bev_ids,
            valid,
            surface_z,
            recovered["rgb"],
            meters_per_pixel,
        )

        final_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
        ground_z = float(camera["ground_reference_z_m"])
        clean_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
        clean_height[final_valid] = clean_surface_z[final_valid] - np.float32(ground_z)
        final_preview = instance_preview(final_ids)
        final_overlay = rgb_overlay(clean_bev_rgb, final_preview, final_valid)
        clean_height_preview = height_preview(clean_height, final_valid)
        clean_height_preview[~final_valid] = 0
        raw_bev_preview = instance_preview(raw_bev_ids)
        component_preview = instance_preview(component_clean_ids)
        removed_bev = (raw_bev_ids > 0) & (component_clean_ids == 0)
        observed_fill = provenance == 2
        geometry_fill = provenance == 3

        paths = {
            "source_ids": directories["source_instance_ids_cleaned"] / f"{sample_id}__source_instance_ids_cleaned.png",
            "source_preview": directories["source_instance_previews_cleaned"] / f"{sample_id}__source_instance_preview_cleaned.png",
            "source_removed": directories["source_removed_noise_masks"] / f"{sample_id}__source_removed_noise.png",
            "source_overlay": directories["source_instance_overlays_cleaned"] / f"{sample_id}__source_instance_overlay_cleaned.png",
            "calibration": directories["calibration_masks_road_full"] / f"{sample_id}__road_full_other_cores.png",
            "depth": directories["depth_road_full_class_aware_zc_m"] / f"{sample_id}__depth_road_full_class_aware_zc_m.npy",
            "bev_raw_ids": directories["bev_instance_ids_raw_from_clean_source"] / f"{sample_id}__bev_instance_ids_raw_from_clean_source.png",
            "bev_component_ids": directories["bev_instance_ids_component_cleaned"] / f"{sample_id}__bev_instance_ids_component_cleaned.png",
            "bev_final_ids": directories["bev_instance_ids_final"] / f"{sample_id}__bev_instance_ids_final.png",
            "bev_final_preview": directories["bev_instance_previews_final"] / f"{sample_id}__bev_instance_preview_final.png",
            "bev_removed": directories["bev_removed_noise_masks"] / f"{sample_id}__bev_removed_noise.png",
            "bev_observed_fill": directories["bev_observed_id_hole_fill_masks"] / f"{sample_id}__bev_observed_id_hole_fill.png",
            "bev_geometry_fill": directories["bev_geometry_hole_fill_masks"] / f"{sample_id}__bev_geometry_hole_fill.png",
            "provenance": directories["bev_cleaning_provenance"] / f"{sample_id}__bev_cleaning_provenance.png",
            "height": directories["bev_building_height_cleaned_m"] / f"{sample_id}__bev_building_height_cleaned_m.npy",
            "height_preview": directories["bev_building_height_previews_cleaned"] / f"{sample_id}__bev_building_height_preview_cleaned.png",
            "overlay": directories["bev_instance_overlays_cleaned"] / f"{sample_id}__bev_instance_overlay_cleaned.png",
            "point_npz": directories["point_cloud_npz_cleaned"] / f"{sample_id}__building_points_cleaned.npz",
            "point_ply": directories["point_cloud_ply_instance_cleaned"] / f"{sample_id}__building_instance_colors_cleaned.ply",
            "stats": directories["per_instance_statistics_cleaned"] / f"{sample_id}__per_instance_statistics_cleaned.json",
            "comparison": directories["comparisons"] / f"{sample_id}__raw_vs_denoising_v1.png",
        }
        save_u16(paths["source_ids"], cleaned_source_ids)
        save_rgb(paths["source_preview"], source_preview_clean)
        save_u8(paths["source_removed"], source_removed.astype(np.uint8) * 255)
        save_rgb(paths["source_overlay"], source_overlay_clean)
        save_rgb(paths["calibration"], calibration_preview)
        np.save(paths["depth"], depth)
        save_u16(paths["bev_raw_ids"], raw_bev_ids)
        save_u16(paths["bev_component_ids"], component_clean_ids)
        save_u16(paths["bev_final_ids"], final_ids)
        save_rgb(paths["bev_final_preview"], final_preview)
        save_u8(paths["bev_removed"], removed_bev.astype(np.uint8) * 255)
        save_u8(paths["bev_observed_fill"], observed_fill.astype(np.uint8) * 255)
        save_u8(paths["bev_geometry_fill"], geometry_fill.astype(np.uint8) * 255)
        save_u8(paths["provenance"], provenance)
        np.save(paths["height"], clean_height)
        save_rgb(paths["height_preview"], clean_height_preview)
        save_rgb(paths["overlay"], final_overlay)

        xyz, point_rgb, point_ids = bev_point_cloud(
            clean_surface_z,
            clean_bev_rgb,
            final_ids,
            final_valid,
            camera["target_patch_bounds_xy_m"],
        )
        np.savez_compressed(
            paths["point_npz"],
            xyz_m=xyz,
            rgb=point_rgb,
            instance_id=point_ids,
            provenance=provenance[final_valid],
        )
        point_instance_rgb = instance_palette(int(np.max(point_ids)))[point_ids]
        ply_metadata = write_binary_ply(
            paths["point_ply"],
            xyz,
            point_instance_rgb,
            point_ids,
            args.max_ply_points,
        )
        stats = per_instance_statistics(
            cleaned_source_ids,
            final_ids,
            clean_surface_z,
            ground_z,
            camera["target_patch_bounds_xy_m"],
        )
        paths["stats"].write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        depth_audit = {"status": "not_in_release", "abs_rel": None}

        two_row_comparison(
            f"{sample_id} / CONSERVATIVE DENOISING V1",
            [
                ("S1. Raw source IDs", "baseline provisional labels", source_preview_raw, True),
                ("S2. Clean source IDs", "3x3 per-instance opening", source_preview_clean, True),
                ("S3. Removed strips", f"removed={source_clean_metadata['removed_fraction_of_raw_building']:.2%}", mask_rgb(source_removed), True),
                ("S4. Clean overlay", "no expansion into road/background", source_overlay_clean, False),
                (
                    "S5. V7B normalized",
                    "complete road mask preserved",
                    resize_nearest(
                        v7b_normalized,
                        (raw_source_ids.shape[1], raw_source_ids.shape[0]),
                    ),
                    True,
                ),
                ("S6. Scale anchors", "road full; water/hardscape cores", calibration_preview, True),
            ],
            [
                ("B1. Raw BEV IDs", "clean source; no BEV cleanup", raw_bev_preview, True),
                ("B2. Component clean", f">={BEV_MIN_COMPONENT_AREA_M2:g} m2; no BEV erosion", component_preview, True),
                ("B3. Final IDs", f"holes <={BEV_MAX_HOLE_AREA_M2:g} m2 only", final_preview, True),
                ("B4. Removed noise", f"{bev_clean_metadata['removed_noise_area_m2']:.1f} m2 removed", mask_rgb(removed_bev), True),
                ("B5. Fill provenance", "green raw / yellow ID / pink geometry", provenance_preview(provenance), True),
                ("B6. Clean height", "observed heights unsmoothed", clean_height_preview, False),
            ],
            paths["comparison"],
            tile=320,
        )
        comparison_paths.append(paths["comparison"])

        records.append(
            {
                "id": sample_id,
                "source_cleaning": source_clean_metadata,
                "calibration": {
                    "road_mask_eroded": False,
                    "nonroad_anchor_erosion_radius_px": args.anchor_erosion_radius_px,
                    "calibration_plane_policy": args.calibration_plane_policy,
                    "priors_json": str(Path(priors_path).resolve()) if priors_path else None,
                    "scale": scale,
                    "metadata": scale_metadata,
                    "gt_absrel_for_audit_only": depth_audit["abs_rel"],
                    "previous_all_classes_eroded_scale": sample["scale_calibration"]["class_aware"]["scale"],
                    "previous_all_classes_eroded_gt_absrel": None,
                },
                "lifting": lift_metadata,
                "rasterization": rasterization,
                "bev_cleaning": bev_clean_metadata,
                "final_observed_instance_count": int(np.unique(final_ids[final_ids > 0]).size),
                "final_point_count": int(xyz.shape[0]),
                "final_height_distribution_m": finite_distribution(clean_height[final_valid]),
                "ply": ply_metadata,
                "outputs": {key: relative(value) for key, value in paths.items()},
            }
        )

    overview = output_root / f"all_{len(records)}_raw_vs_denoising_v1.png"
    build_overview(comparison_paths, overview)
    new_absrel = [item["calibration"]["gt_absrel_for_audit_only"] for item in records]
    previous_absrel = [
        item["calibration"]["previous_all_classes_eroded_gt_absrel"]
        for item in records
    ]
    manifest = {
        "schema": "air2vectorcity.instance_aware_25d_denoising_v1.v1",
        "policy": {
            "raw_baseline_overwritten": False,
            "source_instances": "3x3 per-instance opening and tiny residual component removal",
            "calibration": (
                "full road mask; configurable non-road cores; robust class-balanced scale; "
                f"plane policy={args.calibration_plane_policy}"
            ),
            "bev_instances": ">=2m2 connected components; largest always retained; no BEV morphology",
            "holes": "only <=2m2 holes fully enclosed by one instance",
            "height_smoothing": False,
            "height_fill": "nearest same-instance height only for invalid geometry in approved tiny holes",
        },
        "summary": {
            "road_full_calibration_absrel_mean": None,
            "all_classes_eroded_calibration_absrel_mean": None,
            "source_removed_fraction_mean": float(
                np.mean(
                    [
                        item["source_cleaning"]["removed_fraction_of_raw_building"]
                        for item in records
                    ]
                )
            ),
            "bev_removed_noise_area_m2_total": float(
                np.sum(
                    [item["bev_cleaning"]["removed_noise_area_m2"] for item in records]
                )
            ),
            "observed_id_hole_fill_area_m2_total": float(
                np.sum(
                    [
                        item["bev_cleaning"]["observed_id_hole_fill_area_m2"]
                        for item in records
                    ]
                )
            ),
            "geometry_hole_fill_area_m2_total": float(
                np.sum(
                    [
                        item["bev_cleaning"]["geometry_hole_fill_area_m2"]
                        for item in records
                    ]
                )
            ),
        },
        "overview": relative(overview),
        "samples": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_root, flush=True)


if __name__ == "__main__":
    main()
