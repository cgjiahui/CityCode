#!/usr/bin/env python3
"""Run E09 with the latest source filtering and configured metric-scale Closing."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.roof_masks import two_row_comparison  # noqa: E402
from city_pipeline.geometry.camera_geometry import (  # noqa: E402
    height_preview,
    lift_camera_forward_depth,
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.bev_grid import (  # noqa: E402
    camera_aligned_axes,
    rasterize_local,
)
from city_pipeline.geometry.surface_denoising import (  # noqa: E402
    PROVENANCE_COLORS,
)
from city_pipeline.geometry.surface_fill import (  # noqa: E402
    clean_with_policy,
)
from city_pipeline.geometry.camera_bev import (  # noqa: E402
    building_point_cloud,
    per_instance_statistics,
    xy_for_mask,
)
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    V7B_PALETTE,
    build_overview,
    instance_palette,
    instance_preview,
    mask_rgb,
    resize_nearest,
    rgb_overlay,
    save_u16,
    write_binary_ply,
)


BASELINE_ROOT = Path("UNCONFIGURED_BASELINE_ROOT")
E01_ROOT = BASELINE_ROOT / "denoising_v1"
E05_ROOT = BASELINE_ROOT / "camera_aligned_25d_e05_bev_threshold_ablation"
E08_ROOT = BASELINE_ROOT / "camera_aligned_25d_e08_closing_area_grid"
DEFAULT_OUTPUT_ROOT = BASELINE_ROOT / "camera_aligned_25d_e09_full_pipeline_p5_close7"

E08_VARIANT = "closing_7x7__p5_12m2_48m2"
CELL_SIZE_M = 0.125
HALF_EXTENT_M = 128.0
MIN_COMPONENT_AREA_M2 = 12.0
CLOSING_SIZE = 7
MAX_HOLE_AREA_M2 = 48.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--e01-root", type=Path, default=E01_ROOT)
    parser.add_argument("--e05-root", type=Path, default=E05_ROOT)
    parser.add_argument("--e08-root", type=Path, default=E08_ROOT)
    parser.add_argument(
        "--skip-e08-reference",
        action="store_true",
        help="Skip the frozen E08 equality audit when running a new full dataset.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument("--closing-size", type=int, default=CLOSING_SIZE)
    parser.add_argument(
        "--minimum-total-instance-area-m2",
        type=float,
        default=0.0,
        help=(
            "Remove a complete final instance when its post-fill BEV area is below "
            "this threshold; 0 preserves the historical E09 behaviour."
        ),
    )
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def save_float(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def full_surface_point_cloud(
    surface_z: np.ndarray,
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    background_semantic_ids: np.ndarray,
    provenance: np.ndarray,
    pixel_center_to_world: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.isfinite(surface_z)
    x, y, indices = xy_for_mask(selected, pixel_center_to_world)
    xyz = np.stack((x, y, surface_z[selected].astype(np.float32)), axis=1)
    return (
        xyz,
        rgb[selected].astype(np.uint8),
        instance_ids[selected].astype(np.uint16),
        background_semantic_ids[selected].astype(np.uint8),
        provenance[selected].astype(np.uint8),
    )


def main() -> int:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    e01_root = args.e01_root.resolve()
    e05_root = args.e05_root.resolve()
    e08_root = args.e08_root.resolve()
    output_root = args.output_root.resolve()

    baseline = read_json(baseline_root / "manifest.json")
    e01 = read_json(e01_root / "manifest.json")
    e05 = read_json(e05_root / "manifest.json")
    e08 = None if args.skip_e08_reference else read_json(e08_root / "manifest.json")
    baseline_by_id = {item["id"]: item for item in baseline["samples"]}
    e01_by_id = {item["id"]: item for item in e01["samples"]}
    e05_by_id = {item["id"]: item for item in e05["samples"]}
    e08_by_id = (
        {} if e08 is None else {item["id"]: item for item in e08["samples"]}
    )
    sample_ids = args.sample_id or [item["id"] for item in e05["samples"]]

    directory_names = (
        "source_instance_ids_1024",
        "bev_rgb",
        "bev_validity",
        "bev_surface_z_m",
        "bev_background_semantic_ids",
        "bev_background_semantic_previews",
        "bev_instance_ids_components",
        "bev_instance_ids_final",
        "bev_instance_previews_final",
        "bev_component_removed",
        "bev_closing_fill",
        "bev_hole_fill",
        "bev_cleaning_provenance",
        "bev_cleaning_provenance_previews",
        "bev_building_height_m",
        "bev_building_height_previews",
        "bev_instance_overlays",
        "pixel_to_ue_world_transforms",
        "building_point_cloud_npz",
        "building_point_cloud_ply_instance",
        "full_surface_point_cloud_npz",
        "full_surface_point_cloud_ply_rgb",
        "per_instance_statistics",
        "comparisons",
    )
    directories = {name: output_root / name for name in directory_names}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        base_sample = baseline_by_id[sample_id]
        e01_sample = e01_by_id[sample_id]
        e05_sample = e05_by_id[sample_id]
        e08_sample = e08_by_id.get(sample_id)

        source_rgb = load_rgb(project_path(base_sample["input"]["rgb"]))
        camera = read_json(project_path(base_sample["input"]["camera"]))
        ground_reference_z_m = float(camera["ground_reference_z_m"])
        calibrated_depth = np.load(project_path(e01_sample["outputs"]["depth"])).astype(
            np.float32
        )
        source_ids_1024 = np.asarray(
            Image.open(project_path(e05_sample["shared_outputs"]["source_ids_1024"])),
            dtype=np.uint16,
        )
        if calibrated_depth.shape != source_rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth shape mismatch: {sample_id}: "
                f"{source_rgb.shape[:2]} versus {calibrated_depth.shape}"
            )
        aligned_size = (int(calibrated_depth.shape[1]), int(calibrated_depth.shape[0]))
        source_ids_2048 = resize_nearest(source_ids_1024, aligned_size).astype(np.uint16)
        v7b_ids_2048 = np.asarray(
            Image.open(project_path(base_sample["outputs"]["v7b_class_ids"])),
            dtype=np.uint8,
        )
        if v7b_ids_2048.shape != calibrated_depth.shape:
            raise ValueError(
                f"V7B/depth shape mismatch: {sample_id}: "
                f"{v7b_ids_2048.shape} versus {calibrated_depth.shape}"
            )

        world_xyz, lifting = lift_camera_forward_depth(calibrated_depth, camera)
        e_u, e_v = camera_aligned_axes(camera)
        center_xy = np.asarray(camera["target_center_world_m"][:2], dtype=np.float64)
        target_bounds = [float(value) for value in camera["target_patch_bounds_xy_m"]]
        extent_x = target_bounds[2] - target_bounds[0]
        extent_y = target_bounds[3] - target_bounds[1]
        if not np.isclose(extent_x, extent_y):
            raise ValueError(f"Expected square target patch for {sample_id}: {target_bounds}")
        half_extent_m = extent_x / 2.0
        cell_size_m = extent_x / source_rgb.shape[1]
        recovered, valid, surface_z, rasterization = rasterize_local(
            {
                "rgb": source_rgb,
                "instance_id": source_ids_2048,
                "background_semantic_id": v7b_ids_2048,
            },
            world_xyz,
            center_xy,
            e_u,
            e_v,
            (-half_extent_m, -half_extent_m, half_extent_m, half_extent_m),
            cell_size_m=cell_size_m,
        )
        raw_ids = recovered["instance_id"].astype(np.uint16)
        raw_rgb = recovered["rgb"].astype(np.uint8)
        background_semantic_ids = recovered["background_semantic_id"].astype(np.uint8)

        # Reprojection must be byte/geometry-identical to the frozen E05 raw inputs.
        frozen_raw_ids = np.asarray(
            Image.open(project_path(e05_sample["shared_outputs"]["bev_raw_ids"])),
            dtype=np.uint16,
        )
        frozen_raw_rgb = load_rgb(project_path(e05_sample["shared_outputs"]["bev_rgb_raw"]))
        frozen_surface_z = np.load(
            project_path(e05_sample["shared_outputs"]["bev_surface_z"])
        ).astype(np.float32)
        if not np.array_equal(raw_ids, frozen_raw_ids):
            raise AssertionError(f"E09 raw IDs differ from E05: {sample_id}")
        if not np.array_equal(raw_rgb, frozen_raw_rgb):
            raise AssertionError(f"E09 raw RGB differs from E05: {sample_id}")
        if not np.array_equal(np.isfinite(surface_z), np.isfinite(frozen_surface_z)):
            raise AssertionError(f"E09 surface validity differs from E05: {sample_id}")
        finite = np.isfinite(surface_z)
        surface_max_error = float(
            np.max(np.abs(surface_z[finite] - frozen_surface_z[finite]))
        )
        if surface_max_error > 1e-5:
            raise AssertionError(
                f"E09 surface Z differs from E05 by {surface_max_error}: {sample_id}"
            )

        (
            component_ids,
            final_ids,
            clean_surface_z,
            clean_rgb,
            provenance,
            opening_removed,
            component_removed,
            closing_fill,
            hole_fill,
            cleaning,
        ) = clean_with_policy(
            raw_ids,
            surface_z,
            raw_rgb,
            opening_size=1,
            max_opening_loss_fraction=0.0,
            minimum_component_area_m2=MIN_COMPONENT_AREA_M2,
            minimum_total_instance_area_m2=args.minimum_total_instance_area_m2,
            closing_size=args.closing_size,
            maximum_hole_area_m2=MAX_HOLE_AREA_M2,
            cell_size_m=cell_size_m,
        )
        if np.any(opening_removed):
            raise AssertionError("E09 must not apply opening")
        if e08_sample is not None and args.minimum_total_instance_area_m2 <= 0.0:
            e08_final = np.asarray(
                Image.open(
                    project_path(
                        e08_sample["variants"][E08_VARIANT]["outputs"]["final_ids"]
                    )
                ),
                dtype=np.uint16,
            )
            if not np.array_equal(final_ids, e08_final):
                raise AssertionError(
                    f"E09 final IDs differ from E08 P5/Closing7: {sample_id}"
                )

        # Background semantics are exclusive of final building cells.
        background_semantic_ids = background_semantic_ids.copy()
        background_semantic_ids[final_ids > 0] = 0
        semantic_preview = V7B_PALETTE[background_semantic_ids]
        building_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
        building_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
        building_height[building_valid] = (
            clean_surface_z[building_valid] - np.float32(ground_reference_z_m)
        )
        final_preview = instance_preview(final_ids)
        height_rgb = height_preview(building_height, building_valid)
        height_rgb[~building_valid] = 0
        overlay = rgb_overlay(clean_rgb, final_preview, building_valid)
        provenance_preview = PROVENANCE_COLORS[provenance]

        paths = {
            "source_ids": directories["source_instance_ids_1024"] / f"{sample_id}__source_ids_120_2_5.png",
            "bev_rgb": directories["bev_rgb"] / f"{sample_id}__bev_rgb.png",
            "bev_validity": directories["bev_validity"] / f"{sample_id}__bev_validity.png",
            "surface_z": directories["bev_surface_z_m"] / f"{sample_id}__bev_surface_z_m.npy",
            "background_semantic_ids": directories["bev_background_semantic_ids"] / f"{sample_id}__bev_background_semantic_ids.png",
            "background_semantic_preview": directories["bev_background_semantic_previews"] / f"{sample_id}__bev_background_semantic_preview.png",
            "component_ids": directories["bev_instance_ids_components"] / f"{sample_id}__bev_instance_ids_components.png",
            "final_ids": directories["bev_instance_ids_final"] / f"{sample_id}__bev_instance_ids_final.png",
            "final_preview": directories["bev_instance_previews_final"] / f"{sample_id}__bev_instance_preview_final.png",
            "component_removed": directories["bev_component_removed"] / f"{sample_id}__bev_component_removed.png",
            "closing_fill": directories["bev_closing_fill"] / f"{sample_id}__bev_closing_fill.png",
            "hole_fill": directories["bev_hole_fill"] / f"{sample_id}__bev_hole_fill.png",
            "provenance": directories["bev_cleaning_provenance"] / f"{sample_id}__bev_cleaning_provenance.png",
            "provenance_preview": directories["bev_cleaning_provenance_previews"] / f"{sample_id}__bev_cleaning_provenance_preview.png",
            "building_height": directories["bev_building_height_m"] / f"{sample_id}__bev_building_height_m.npy",
            "building_height_preview": directories["bev_building_height_previews"] / f"{sample_id}__bev_building_height_preview.png",
            "overlay": directories["bev_instance_overlays"] / f"{sample_id}__bev_instance_overlay.png",
            "transform": directories["pixel_to_ue_world_transforms"] / f"{sample_id}__pixel_to_ue_world.json",
            "building_npz": directories["building_point_cloud_npz"] / f"{sample_id}__building_points_e09.npz",
            "building_ply": directories["building_point_cloud_ply_instance"] / f"{sample_id}__building_instances_e09.ply",
            "full_npz": directories["full_surface_point_cloud_npz"] / f"{sample_id}__full_surface_e09.npz",
            "full_ply": directories["full_surface_point_cloud_ply_rgb"] / f"{sample_id}__full_surface_rgb_e09.ply",
            "stats": directories["per_instance_statistics"] / f"{sample_id}__per_instance_statistics.json",
            "comparison": directories["comparisons"] / f"{sample_id}__e09_full_pipeline.png",
        }
        save_u16(paths["source_ids"], source_ids_1024)
        save_rgb(paths["bev_rgb"], clean_rgb)
        save_u8(paths["bev_validity"], np.isfinite(clean_surface_z).astype(np.uint8) * 255)
        save_float(paths["surface_z"], clean_surface_z)
        save_u8(paths["background_semantic_ids"], background_semantic_ids)
        save_rgb(paths["background_semantic_preview"], semantic_preview)
        save_u16(paths["component_ids"], component_ids)
        save_u16(paths["final_ids"], final_ids)
        save_rgb(paths["final_preview"], final_preview)
        save_u8(paths["component_removed"], component_removed.astype(np.uint8) * 255)
        save_u8(paths["closing_fill"], closing_fill.astype(np.uint8) * 255)
        save_u8(paths["hole_fill"], hole_fill.astype(np.uint8) * 255)
        save_u8(paths["provenance"], provenance)
        save_rgb(paths["provenance_preview"], provenance_preview)
        save_float(paths["building_height"], building_height)
        save_rgb(paths["building_height_preview"], height_rgb)
        save_rgb(paths["overlay"], overlay)
        paths["transform"].write_text(
            json.dumps(rasterization, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        transform = rasterization["pixel_center_index_to_ue_world_xy_3x3"]
        xyz, point_rgb, point_ids, point_provenance = building_point_cloud(
            clean_surface_z, clean_rgb, final_ids, provenance, transform
        )
        np.savez_compressed(
            paths["building_npz"],
            xyz_m=xyz,
            rgb=point_rgb,
            instance_id=point_ids,
            provenance=point_provenance,
        )
        building_ply = write_binary_ply(
            paths["building_ply"],
            xyz,
            instance_palette(int(np.max(point_ids)))[point_ids],
            point_ids,
            args.max_ply_points,
        )

        full_xyz, full_rgb, full_ids, full_semantic, full_provenance = (
            full_surface_point_cloud(
                clean_surface_z,
                clean_rgb,
                final_ids,
                background_semantic_ids,
                provenance,
                transform,
            )
        )
        np.savez_compressed(
            paths["full_npz"],
            xyz_m=full_xyz,
            rgb=full_rgb,
            building_instance_id=full_ids,
            background_semantic_id=full_semantic,
            building_provenance=full_provenance,
        )
        full_ply = write_binary_ply(
            paths["full_ply"],
            full_xyz,
            full_rgb,
            full_ids,
            args.max_ply_points,
        )
        stats = per_instance_statistics(
            source_ids_1024,
            final_ids,
            clean_surface_z,
            ground_reference_z_m,
            transform,
            cell_size_m,
        )
        paths["stats"].write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        two_row_comparison(
            f"{sample_id} / E09 FULL INSTANCE-AWARE 2.5D",
            [
                ("Raw BEV RGB", "class-aware calibrated depth", raw_rgb, False),
                ("V7B BEV", "road / water / hard / elevated", semantic_preview, True),
                ("Raw BEV IDs", f"{np.unique(raw_ids[raw_ids > 0]).size} IDs", instance_preview(raw_ids), True),
                (
                    "Component / tiny instance removed",
                    (
                        f"components <12 m2; whole instance "
                        f"<{args.minimum_total_instance_area_m2:g} m2"
                    ),
                    mask_rgb(component_removed),
                    True,
                ),
                (
                    "Closing fill",
                    f"{args.closing_size}x{args.closing_size} unique-ID",
                    mask_rgb(closing_fill),
                    True,
                ),
                ("Hole fill", "<=48 m2 enclosed", mask_rgb(hole_fill), True),
            ],
            [
                ("Final IDs", f"{np.unique(final_ids[final_ids > 0]).size} IDs", final_preview, True),
                ("Provenance", "green raw / blue-orange filled", provenance_preview, True),
                (
                    "Building height",
                    f"world Z - local ground {ground_reference_z_m:.2f} m",
                    height_rgb,
                    False,
                ),
                ("Final overlay", f"building points={xyz.shape[0]:,}", overlay, False),
                ("Full 2.5D validity", f"surface points={full_xyz.shape[0]:,}", mask_rgb(np.isfinite(clean_surface_z)), True),
                ("UE world frame", "camera-aligned raster; exact XY transform", raw_rgb, False),
            ],
            paths["comparison"],
            tile=320,
        )
        comparisons.append(paths["comparison"])
        records.append(
            {
                "id": sample_id,
                "input": {
                    "rgb": base_sample["input"]["rgb"],
                    "camera": base_sample["input"]["camera"],
                    "v7b_class_ids": base_sample["outputs"]["v7b_class_ids"],
                    "v7b_ground_anchor_cores": base_sample["outputs"]["ground_anchor_cores"],
                    "class_aware_depth_zc_m": e01_sample["outputs"]["depth"],
                    "source_ids_120_2_5": e05_sample["shared_outputs"]["source_ids_1024"],
                    "e08_final_reference": (
                        None
                        if e08_sample is None
                        else e08_sample["variants"][E08_VARIANT]["outputs"]["final_ids"]
                    ),
                },
                "lifting": lifting,
                "rasterization": rasterization,
                "ground_reference_z_m": ground_reference_z_m,
                "raw_reprojection_surface_max_abs_error_vs_e05_m": surface_max_error,
                "cleaning": cleaning,
                "final_instance_count": int(np.unique(final_ids[final_ids > 0]).size),
                "building_point_count": int(xyz.shape[0]),
                "full_surface_point_count": int(full_xyz.shape[0]),
                "building_ply": building_ply,
                "full_surface_ply": full_ply,
                "transform": relative(paths["transform"]),
                "outputs": {
                    "final_ids": relative(paths["final_ids"]),
                    "final_preview": relative(paths["final_preview"]),
                    "surface_z": relative(paths["surface_z"]),
                    "provenance": relative(paths["provenance"]),
                    "bev_rgb": relative(paths["bev_rgb"]),
                    "background_semantic_ids": relative(paths["background_semantic_ids"]),
                    "background_semantic_preview": relative(paths["background_semantic_preview"]),
                    "building_height": relative(paths["building_height"]),
                    "building_height_preview": relative(paths["building_height_preview"]),
                    "overlay": relative(paths["overlay"]),
                    "component_removed": relative(paths["component_removed"]),
                    "closing_fill": relative(paths["closing_fill"]),
                    "hole_fill": relative(paths["hole_fill"]),
                    "building_point_cloud_npz": relative(paths["building_npz"]),
                    "building_point_cloud_ply_instance": relative(paths["building_ply"]),
                    "full_surface_point_cloud_npz": relative(paths["full_npz"]),
                    "full_surface_point_cloud_ply_rgb": relative(paths["full_ply"]),
                    "per_instance_statistics": relative(paths["stats"]),
                    "comparison": relative(paths["comparison"]),
                },
            }
        )

    overview = output_root / f"all_{len(records)}_e09_full_pipeline.png"
    build_overview(comparisons, overview)
    manifest = {
        "schema": "air2vectorcity.camera_aligned_25d_full_pipeline_e09.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete full pipeline candidate",
        "inputs": {
            "baseline": relative(baseline_root),
            "e01_depth_and_v7b_calibration": relative(e01_root),
            "e05_latest_source_and_raw_bev": relative(e05_root),
            "e08_final_id_reference": (
                None if args.skip_e08_reference else relative(e08_root)
            ),
            "e08_variant": None if args.skip_e08_reference else E08_VARIANT,
        },
        "policy": {
            "source_filter": "V7B-gated blue + 120/2/5 guarded near-black",
            "opening": False,
            "minimum_component_area_m2": MIN_COMPONENT_AREA_M2,
            "minimum_total_instance_area_m2": args.minimum_total_instance_area_m2,
            "closing_size": args.closing_size,
            "maximum_hole_area_m2": MAX_HOLE_AREA_M2,
            "height_smoothing": False,
            "building_height_reference": "per-sample camera ground_reference_z_m",
            "depth_scale": "frozen E01 V7B class-aware calibration",
            "calibration_planes": (
                "inherited per sample from E01; see calibration.metadata.class_plane_z_m"
            ),
        },
        "representation": {
            "bev": "camera-aligned released patch extent at source resolution",
            "meters_per_pixel": sorted(
                {sample["rasterization"]["meters_per_pixel"][0] for sample in records}
            ),
            "building_point_cloud": "UE world XYZ + RGB + instance ID + provenance",
            "full_surface_point_cloud": "UE world XYZ + RGB + building instance + V7B background semantic + provenance",
            "world_coordinates": "all point outputs remain UE world XYZ metres",
        },
        "overview": relative(overview),
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
