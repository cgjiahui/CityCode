#!/usr/bin/env python3
"""Compare three physical BEV cleanup thresholds using fixed 120/2/5 source IDs."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


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
    CLOSING_SIZE,
    MAX_OPENING_LOSS_FRACTION,
    OPENING_SIZE,
    PROVENANCE_COLORS,
)
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    build_overview,
    instance_preview,
    mask_rgb,
    resize_nearest,
    rgb_overlay,
    save_u16,
)
from city_pipeline.geometry.mask_geometry import (  # noqa: E402
    fit_panel,
    font,
)


BASELINE_ROOT = Path("UNCONFIGURED_BASELINE_ROOT")
E01_ROOT = BASELINE_ROOT / "denoising_v1"
SOURCE_PARAMETER_ROOT = (
    BASELINE_ROOT
    / "camera_aligned_25d_e04_aggressive"
    / "source_blue_dark_parameter_comparison"
)
SOURCE_VARIANT = "aggressive_120_d2_r5"
DEFAULT_OUTPUT_ROOT = BASELINE_ROOT / "camera_aligned_25d_e05_bev_threshold_ablation"

CELL_SIZE_M = 0.125
PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
HALF_EXTENT_M = 128.0
VARIANTS: dict[str, dict[str, float | str | tuple[int, int, int]]] = {
    "variant_a_2m2_4m2": {
        "label": "A / CONSERVATIVE / 2 m2 COMPONENTS / 4 m2 HOLES",
        "minimum_component_area_m2": 2.0,
        "maximum_hole_area_m2": 4.0,
        "color": (38, 92, 80),
    },
    "variant_b_3m2_6m2": {
        "label": "B / BALANCED / 3 m2 COMPONENTS / 6 m2 HOLES",
        "minimum_component_area_m2": 3.0,
        "maximum_hole_area_m2": 6.0,
        "color": (55, 75, 111),
    },
    "variant_c_5m2_12m2": {
        "label": "C / AGGRESSIVE / 5 m2 COMPONENTS / 12 m2 HOLES",
        "minimum_component_area_m2": 5.0,
        "maximum_hole_area_m2": 12.0,
        "color": (112, 55, 43),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--e01-root", type=Path, default=E01_ROOT)
    parser.add_argument("--source-parameter-root", type=Path, default=SOURCE_PARAMETER_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_float(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def parameterized_clean(
    input_ids: np.ndarray,
    input_surface_z: np.ndarray,
    input_rgb: np.ndarray,
    minimum_component_area_m2: float,
    maximum_hole_area_m2: float,
    cell_size_m: float = CELL_SIZE_M,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """E04 geometry policy with parameterized physical area thresholds."""

    structure_open = np.ones((OPENING_SIZE, OPENING_SIZE), dtype=bool)
    structure_close = np.ones((CLOSING_SIZE, CLOSING_SIZE), dtype=bool)
    pixel_area_m2 = cell_size_m * cell_size_m
    min_component_pixels = int(math.ceil(minimum_component_area_m2 / pixel_area_m2))
    max_hole_pixels = int(math.floor(maximum_hole_area_m2 / pixel_area_m2))
    core_ids = np.zeros_like(input_ids, dtype=np.uint16)
    instance_records: list[dict[str, Any]] = []

    for instance_id in np.unique(input_ids):
        if instance_id == 0:
            continue
        raw = input_ids == instance_id
        y, x = np.nonzero(raw)
        y0 = max(0, int(np.min(y)) - CLOSING_SIZE)
        y1 = min(raw.shape[0], int(np.max(y)) + CLOSING_SIZE + 1)
        x0 = max(0, int(np.min(x)) - CLOSING_SIZE)
        x1 = min(raw.shape[1], int(np.max(x)) + CLOSING_SIZE + 1)
        crop = raw[y0:y1, x0:x1]
        opened = ndimage.binary_opening(crop, structure=structure_open)
        raw_count = int(np.count_nonzero(crop))
        opened_count = int(np.count_nonzero(opened))
        opening_loss = 1.0 - opened_count / raw_count
        opening_applied = opened_count > 0 and opening_loss <= MAX_OPENING_LOSS_FRACTION
        candidate = opened if opening_applied else crop
        labels, component_count = ndimage.label(
            candidate, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= min_component_pixels
        if keep.size:
            keep[0] = False
        retained = keep[labels]
        core_ids[y0:y1, x0:x1][retained] = instance_id
        instance_records.append(
            {
                "instance_id": int(instance_id),
                "input_cell_count": raw_count,
                "input_area_m2": float(raw_count * pixel_area_m2),
                "opening_applied": bool(opening_applied),
                "opening_loss_fraction_candidate": float(opening_loss),
                "component_count": int(component_count),
                "retained_component_count": int(np.count_nonzero(keep)),
                "core_cell_count": int(np.count_nonzero(retained)),
                "removed_complete_instance": bool(np.count_nonzero(retained) == 0),
            }
        )

    claim_count = np.zeros(input_ids.shape, dtype=np.uint16)
    claim_id = np.zeros(input_ids.shape, dtype=np.uint16)
    for instance_id in np.unique(core_ids):
        if instance_id == 0:
            continue
        mask = core_ids == instance_id
        y, x = np.nonzero(mask)
        y0 = max(0, int(np.min(y)) - CLOSING_SIZE)
        y1 = min(mask.shape[0], int(np.max(y)) + CLOSING_SIZE + 1)
        x0 = max(0, int(np.min(x)) - CLOSING_SIZE)
        x1 = min(mask.shape[1], int(np.max(x)) + CLOSING_SIZE + 1)
        crop = mask[y0:y1, x0:x1]
        closed = ndimage.binary_closing(crop, structure=structure_close)
        additions = closed & ~crop & (input_ids[y0:y1, x0:x1] == 0)
        claim_count[y0:y1, x0:x1][additions] += 1
        claim_id[y0:y1, x0:x1][additions] = instance_id

    final_ids = core_ids.copy()
    closing_fill = (claim_count == 1) & (core_ids == 0) & (input_ids == 0)
    final_ids[closing_fill] = claim_id[closing_fill]

    hole_fill = np.zeros(input_ids.shape, dtype=bool)
    hole_records: list[dict[str, Any]] = []
    for instance_id in np.unique(final_ids):
        if instance_id == 0:
            continue
        mask = final_ids == instance_id
        y, x = np.nonzero(mask)
        y0 = max(0, int(np.min(y)) - 1)
        y1 = min(mask.shape[0], int(np.max(y)) + 2)
        x0 = max(0, int(np.min(x)) - 1)
        x1 = min(mask.shape[1], int(np.max(x)) + 2)
        crop = mask[y0:y1, x0:x1]
        enclosed = ndimage.binary_fill_holes(crop) & ~crop
        labels, hole_count = ndimage.label(
            enclosed, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.reshape(-1))
        for label_id in range(1, hole_count + 1):
            pixel_count = int(sizes[label_id])
            accepted = pixel_count <= max_hole_pixels
            candidate = labels == label_id
            if accepted:
                input_crop = input_ids[y0:y1, x0:x1]
                final_crop = final_ids[y0:y1, x0:x1]
                competing = (
                    (input_crop[candidate] > 0)
                    & (input_crop[candidate] != instance_id)
                )
                accepted = not np.any(competing) and not np.any(final_crop[candidate] > 0)
                if accepted:
                    final_crop[candidate] = instance_id
                    hole_fill[y0:y1, x0:x1] |= candidate
            hole_records.append(
                {
                    "instance_id": int(instance_id),
                    "pixel_count": pixel_count,
                    "area_m2": float(pixel_count * pixel_area_m2),
                    "filled": bool(accepted),
                }
            )

    added = closing_fill | hole_fill
    removed = (input_ids > 0) & (core_ids == 0)
    clean_surface_z = input_surface_z.copy()
    clean_rgb = input_rgb.copy()
    added_with_observed_height = added & np.isfinite(clean_surface_z)
    added_geometry_fill = added & ~np.isfinite(clean_surface_z)

    for instance_id in np.unique(final_ids[added_geometry_fill]):
        if instance_id == 0:
            continue
        target = added_geometry_fill & (final_ids == instance_id)
        support = (final_ids == instance_id) & np.isfinite(clean_surface_z) & ~target
        combined = target | support
        y, x = np.nonzero(combined)
        y0 = max(0, int(np.min(y)) - 1)
        y1 = min(target.shape[0], int(np.max(y)) + 2)
        x0 = max(0, int(np.min(x)) - 1)
        x1 = min(target.shape[1], int(np.max(x)) + 2)
        target_crop = target[y0:y1, x0:x1]
        support_crop = support[y0:y1, x0:x1]
        if not np.any(support_crop):
            final_ids[y0:y1, x0:x1][target_crop] = 0
            added_geometry_fill[y0:y1, x0:x1][target_crop] = False
            continue
        _, nearest = ndimage.distance_transform_edt(~support_crop, return_indices=True)
        z_crop = clean_surface_z[y0:y1, x0:x1]
        rgb_crop = clean_rgb[y0:y1, x0:x1]
        z_crop[target_crop] = z_crop[nearest[0][target_crop], nearest[1][target_crop]]
        rgb_crop[target_crop] = rgb_crop[
            nearest[0][target_crop], nearest[1][target_crop]
        ]

    provenance = np.zeros(input_ids.shape, dtype=np.uint8)
    retained = (final_ids == input_ids) & (final_ids > 0)
    provenance[retained] = 1
    provenance[added_with_observed_height & (final_ids > 0)] = 4
    provenance[added_geometry_fill & (final_ids > 0)] = 5

    metadata = {
        "method": "fixed E04 morphology with parameterized metric component/hole areas",
        "cell_size_m": cell_size_m,
        "pixel_area_m2": pixel_area_m2,
        "opening_size": OPENING_SIZE,
        "max_opening_loss_fraction": MAX_OPENING_LOSS_FRACTION,
        "minimum_component_area_m2": minimum_component_area_m2,
        "minimum_component_pixels": min_component_pixels,
        "closing_size": CLOSING_SIZE,
        "maximum_hole_area_m2": maximum_hole_area_m2,
        "maximum_hole_pixels": max_hole_pixels,
        "input_cell_count": int(np.count_nonzero(input_ids)),
        "core_cell_count": int(np.count_nonzero(core_ids)),
        "final_cell_count": int(np.count_nonzero(final_ids)),
        "input_instance_count": int(np.unique(input_ids[input_ids > 0]).size),
        "final_instance_count": int(np.unique(final_ids[final_ids > 0]).size),
        "removed_cell_count": int(np.count_nonzero(removed)),
        "removed_area_m2": float(np.count_nonzero(removed) * pixel_area_m2),
        "closing_fill_cell_count": int(np.count_nonzero(closing_fill)),
        "closing_fill_area_m2": float(np.count_nonzero(closing_fill) * pixel_area_m2),
        "hole_fill_cell_count": int(np.count_nonzero(hole_fill)),
        "hole_fill_area_m2": float(np.count_nonzero(hole_fill) * pixel_area_m2),
        "added_observed_height_cell_count": int(np.count_nonzero(added_with_observed_height)),
        "added_geometry_fill_cell_count": int(np.count_nonzero(added_geometry_fill)),
        "removed_complete_instance_ids": [
            item["instance_id"] for item in instance_records if item["removed_complete_instance"]
        ],
        "instance_records": instance_records,
        "hole_records": hole_records,
        "height_smoothing": False,
    }
    return (
        core_ids,
        final_ids,
        clean_surface_z,
        clean_rgb,
        provenance,
        closing_fill,
        hole_fill,
        metadata,
    )


def color_provenance(provenance: np.ndarray) -> np.ndarray:
    return PROVENANCE_COLORS[provenance]


def comparison_figure(
    sample_id: str,
    source_preview: np.ndarray,
    raw_ids: np.ndarray,
    raw_rgb: np.ndarray,
    raw_surface_z: np.ndarray,
    valid: np.ndarray,
    ground_reference_z_m: float,
    variant_panels: list[tuple[str, tuple[int, int, int], list[tuple[str, str, np.ndarray]]]],
    destination: Path,
) -> None:
    tile = 330
    gap = 7
    title_height = 52
    row_label_height = 30
    header = 66
    columns = 5
    rows = 1 + len(variant_panels)
    width = columns * tile + (columns - 1) * gap
    row_height = row_label_height + header + tile
    height = title_height + rows * row_height + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (17, 19, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        f"{sample_id} / FIXED 120-2-5 SOURCE / CAMERA-ALIGNED BEV THRESHOLDS",
        font=font(22),
        fill=(248, 248, 248),
    )
    raw_building = (raw_ids > 0) & np.isfinite(raw_surface_z)
    raw_height = raw_surface_z - np.float32(ground_reference_z_m)
    raw_height_preview = height_preview(raw_height, raw_building)
    raw_height_preview[~raw_building] = 0
    raw_overlay = rgb_overlay(raw_rgb, instance_preview(raw_ids), raw_ids > 0)
    all_rows = [
        (
            "SHARED INPUT / 120-2-5 SOURCE + SAME DEPTH + RAW BEV",
            (61, 65, 73),
            [
                ("Source 120 IDs", "1024 perspective", source_preview),
                ("Raw BEV IDs", f"{np.unique(raw_ids[raw_ids > 0]).size} IDs", instance_preview(raw_ids)),
                ("Raw BEV overlay", "before BEV cleanup", raw_overlay),
                ("Raw building height", "observed only", raw_height_preview),
                ("BEV validity", f"coverage={np.mean(valid):.1%}", mask_rgb(valid)),
            ],
        ),
        *variant_panels,
    ]
    for row_index, (row_label, row_color, panels) in enumerate(all_rows):
        y = title_height + row_index * (row_height + gap)
        draw.rectangle((0, y, width, y + row_label_height - 1), fill=row_color)
        draw.text((10, y + 4), row_label, font=font(14), fill=(248, 248, 248))
        header_y = y + row_label_height
        for column, (label, subtitle, array) in enumerate(panels):
            x = column * (tile + gap)
            draw.rectangle(
                (x, header_y, x + tile - 1, header_y + header - 1),
                fill=(35, 38, 44),
            )
            draw.text((x + 8, header_y + 5), label, font=font(17), fill=(255, 255, 255))
            draw.text((x + 8, header_y + 36), subtitle, font=font(12), fill=(190, 197, 207))
            canvas.paste(fit_panel(array, tile, nearest=True), (x, header_y + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def main() -> int:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    e01_root = args.e01_root.resolve()
    source_root = args.source_parameter_root.resolve()
    output_root = args.output_root.resolve()
    baseline = read_json(baseline_root / "manifest.json")
    e01 = read_json(e01_root / "manifest.json")
    source_manifest = read_json(source_root / "manifest.json")
    baseline_by_id = {item["id"]: item for item in baseline["samples"]}
    e01_by_id = {item["id"]: item for item in e01["samples"]}
    source_by_id = {item["id"]: item for item in source_manifest["samples"]}
    sample_ids = args.sample_id or [item["id"] for item in source_manifest["samples"]]

    shared_names = (
        "source_instance_ids_1024",
        "source_instance_previews_1024",
        "bev_rgb_raw",
        "bev_validity",
        "bev_surface_z_m_raw",
        "bev_instance_ids_raw",
        "bev_instance_previews_raw",
        "pixel_to_ue_world_transforms",
    )
    shared_dirs = {name: output_root / "shared" / name for name in shared_names}
    for path in shared_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    variant_dirs: dict[str, dict[str, Path]] = {}
    variant_names = (
        "bev_instance_ids_core",
        "bev_instance_ids_final",
        "bev_instance_previews_final",
        "bev_removed_components",
        "bev_closing_fill",
        "bev_hole_fill",
        "bev_provenance",
        "bev_provenance_previews",
        "bev_rgb_clean",
        "bev_instance_overlays",
        "bev_surface_z_m",
        "bev_building_height_m",
        "bev_building_height_previews",
    )
    for key in VARIANTS:
        variant_dirs[key] = {name: output_root / key / name for name in variant_names}
        for path in variant_dirs[key].values():
            path.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        base_sample = baseline_by_id[sample_id]
        e01_sample = e01_by_id[sample_id]
        source_sample = source_by_id[sample_id]
        source_ids_path = source_root / source_sample["variants"][SOURCE_VARIANT]["outputs"]["provisional_ids"]
        source_ids_1024 = np.asarray(Image.open(source_ids_path), dtype=np.uint16)
        source_preview_1024 = instance_preview(source_ids_1024)
        source_rgb = load_rgb(project_path(base_sample["input"]["rgb"]))
        camera = read_json(project_path(base_sample["input"]["camera"]))
        calibrated_depth = np.load(project_path(e01_sample["outputs"]["depth"])).astype(np.float32)
        if calibrated_depth.shape != source_rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth shape mismatch: {sample_id}: "
                f"{source_rgb.shape[:2]} versus {calibrated_depth.shape}"
            )
        aligned_size = (int(calibrated_depth.shape[1]), int(calibrated_depth.shape[0]))
        source_ids_2048 = resize_nearest(source_ids_1024, aligned_size).astype(np.uint16)
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
        shared_outputs = {
            "source_ids_1024": shared_dirs["source_instance_ids_1024"] / f"{sample_id}__source_ids_120_2_5.png",
            "source_preview_1024": shared_dirs["source_instance_previews_1024"] / f"{sample_id}__source_preview_120_2_5.png",
            "bev_rgb_raw": shared_dirs["bev_rgb_raw"] / f"{sample_id}__bev_rgb_raw.png",
            "bev_validity": shared_dirs["bev_validity"] / f"{sample_id}__bev_validity.png",
            "bev_surface_z": shared_dirs["bev_surface_z_m_raw"] / f"{sample_id}__bev_surface_z_m_raw.npy",
            "bev_raw_ids": shared_dirs["bev_instance_ids_raw"] / f"{sample_id}__bev_instance_ids_raw.png",
            "bev_raw_preview": shared_dirs["bev_instance_previews_raw"] / f"{sample_id}__bev_instance_preview_raw.png",
            "transform": shared_dirs["pixel_to_ue_world_transforms"] / f"{sample_id}__pixel_to_ue_world.json",
        }
        save_u16(shared_outputs["source_ids_1024"], source_ids_1024)
        save_rgb(shared_outputs["source_preview_1024"], source_preview_1024)
        save_rgb(shared_outputs["bev_rgb_raw"], raw_rgb)
        save_u8(shared_outputs["bev_validity"], valid.astype(np.uint8) * 255)
        save_float(shared_outputs["bev_surface_z"], surface_z)
        save_u16(shared_outputs["bev_raw_ids"], raw_ids)
        save_rgb(shared_outputs["bev_raw_preview"], instance_preview(raw_ids))
        shared_outputs["transform"].write_text(
            json.dumps(rasterization, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        sample_record: dict[str, Any] = {
            "id": sample_id,
            "source": {
                "variant": SOURCE_VARIANT,
                "ids_1024": relative(source_ids_path),
                "instance_count": int(np.unique(source_ids_1024[source_ids_1024 > 0]).size),
                "pixel_count": int(np.count_nonzero(source_ids_1024)),
                "aligned_resolution_px": list(aligned_size),
                "upsampling": (
                    "nearest-neighbor 1024 to native depth/camera resolution only "
                    "for geometric alignment"
                ),
            },
            "lifting": lifting,
            "rasterization": rasterization,
            "shared_outputs": {
                key: relative(path) for key, path in shared_outputs.items()
            },
            "variants": {},
        }
        panel_rows = []
        ground_reference_z = float(camera["ground_reference_z_m"])
        for key, parameters in VARIANTS.items():
            (
                core_ids,
                final_ids,
                clean_surface_z,
                clean_rgb,
                provenance,
                closing_fill,
                hole_fill,
                cleaning,
            ) = parameterized_clean(
                raw_ids,
                surface_z,
                raw_rgb,
                float(parameters["minimum_component_area_m2"]),
                float(parameters["maximum_hole_area_m2"]),
                cell_size_m,
            )
            removed = (raw_ids > 0) & (core_ids == 0)
            building_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
            building_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
            building_height[building_valid] = clean_surface_z[building_valid] - np.float32(ground_reference_z)
            height_rgb = height_preview(building_height, building_valid)
            height_rgb[~building_valid] = 0
            final_preview = instance_preview(final_ids)
            overlay = rgb_overlay(clean_rgb, final_preview, building_valid)
            directories = variant_dirs[key]
            paths = {
                "core_ids": directories["bev_instance_ids_core"] / f"{sample_id}__bev_instance_ids_core.png",
                "final_ids": directories["bev_instance_ids_final"] / f"{sample_id}__bev_instance_ids_final.png",
                "final_preview": directories["bev_instance_previews_final"] / f"{sample_id}__bev_instance_preview_final.png",
                "removed": directories["bev_removed_components"] / f"{sample_id}__bev_removed_components.png",
                "closing_fill": directories["bev_closing_fill"] / f"{sample_id}__bev_closing_fill.png",
                "hole_fill": directories["bev_hole_fill"] / f"{sample_id}__bev_hole_fill.png",
                "provenance": directories["bev_provenance"] / f"{sample_id}__bev_provenance.png",
                "provenance_preview": directories["bev_provenance_previews"] / f"{sample_id}__bev_provenance_preview.png",
                "bev_rgb": directories["bev_rgb_clean"] / f"{sample_id}__bev_rgb_clean.png",
                "overlay": directories["bev_instance_overlays"] / f"{sample_id}__bev_instance_overlay.png",
                "surface_z": directories["bev_surface_z_m"] / f"{sample_id}__bev_surface_z_m.npy",
                "building_height": directories["bev_building_height_m"] / f"{sample_id}__bev_building_height_m.npy",
                "building_height_preview": directories["bev_building_height_previews"] / f"{sample_id}__bev_building_height_preview.png",
            }
            save_u16(paths["core_ids"], core_ids)
            save_u16(paths["final_ids"], final_ids)
            save_rgb(paths["final_preview"], final_preview)
            save_rgb(paths["removed"], mask_rgb(removed))
            save_rgb(paths["closing_fill"], mask_rgb(closing_fill))
            save_rgb(paths["hole_fill"], mask_rgb(hole_fill))
            save_u8(paths["provenance"], provenance)
            save_rgb(paths["provenance_preview"], color_provenance(provenance))
            save_rgb(paths["bev_rgb"], clean_rgb)
            save_rgb(paths["overlay"], overlay)
            save_float(paths["surface_z"], clean_surface_z)
            save_float(paths["building_height"], building_height)
            save_rgb(paths["building_height_preview"], height_rgb)
            sample_record["variants"][key] = {
                "parameters": {
                    name: value for name, value in parameters.items() if name != "color"
                },
                "cleaning": cleaning,
                "outputs": {
                    name: relative(path) for name, path in paths.items()
                },
            }
            panel_rows.append(
                (
                    str(parameters["label"]),
                    tuple(parameters["color"]),
                    [
                        ("Final BEV IDs", f"{cleaning['final_instance_count']} IDs", final_preview),
                        ("Final overlay", f"cells={cleaning['final_cell_count']}", overlay),
                        ("Removed", f"{cleaning['removed_area_m2']:.1f} m2", mask_rgb(removed)),
                        ("Hole fill", f"{cleaning['hole_fill_area_m2']:.1f} m2", mask_rgb(hole_fill)),
                        ("Building height", "original z; fills tracked", height_rgb),
                    ],
                )
            )
        comparison = comparison_dir / f"{sample_id}__bev_thresholds_a_b_c.png"
        comparison_figure(
            sample_id,
            source_preview_1024,
            raw_ids,
            raw_rgb,
            surface_z,
            valid,
            ground_reference_z,
            panel_rows,
            comparison,
        )
        sample_record["comparison"] = relative(comparison)
        records.append(sample_record)
        comparisons.append(comparison)

    overview = output_root / f"all_{len(records)}_bev_thresholds_a_b_c.png"
    build_overview(comparisons, overview)
    aggregate: dict[str, Any] = {}
    for key in VARIANTS:
        cleanings = [sample["variants"][key]["cleaning"] for sample in records]
        aggregate[key] = {
            "parameters": {
                name: value for name, value in VARIANTS[key].items() if name != "color"
            },
            "input_instance_count_sum": int(sum(item["input_instance_count"] for item in cleanings)),
            "final_instance_count_sum": int(sum(item["final_instance_count"] for item in cleanings)),
            "removed_complete_instance_count": int(sum(len(item["removed_complete_instance_ids"]) for item in cleanings)),
            "removed_area_m2": float(sum(item["removed_area_m2"] for item in cleanings)),
            "closing_fill_area_m2": float(sum(item["closing_fill_area_m2"] for item in cleanings)),
            "hole_fill_area_m2": float(sum(item["hole_fill_area_m2"] for item in cleanings)),
            "final_building_area_m2": float(
                sum(item["final_cell_count"] * item["pixel_area_m2"] for item in cleanings)
            ),
        }
    manifest = {
        "schema": "air2vectorcity.camera_aligned_bev_threshold_ablation_e05.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete BEV threshold ablation; E00-E04 unchanged",
        "fixed_inputs": {
            "source_variant": SOURCE_VARIANT,
            "source_processing": "V7B-gated blue filtering + 120/2/5 guarded near-black filtering; no source 16px post-cleanup",
            "source_resolution": [1024, 1024],
            "depth_resolution": sorted(
                {tuple(sample["source"]["aligned_resolution_px"]) for sample in records}
            ),
            "source_id_alignment": "nearest-neighbor 1024 to native depth resolution",
            "bev_extent_m": sorted(
                {
                    (
                        sample["rasterization"]["bounds_local_m"][2]
                        - sample["rasterization"]["bounds_local_m"][0],
                        sample["rasterization"]["bounds_local_m"][3]
                        - sample["rasterization"]["bounds_local_m"][1],
                    )
                    for sample in records
                }
            ),
            "bev_resolution": sorted(
                {
                    tuple(sample["rasterization"]["output_size_px"])
                    for sample in records
                }
            ),
            "bev_cell_size_m": sorted(
                {sample["rasterization"]["meters_per_pixel"][0] for sample in records}
            ),
            "bev_cell_area_m2": sorted(
                {
                    sample["rasterization"]["meters_per_pixel"][0]
                    * sample["rasterization"]["meters_per_pixel"][1]
                    for sample in records
                }
            ),
            "depth": "frozen E01 class-aware calibrated camera z_c",
            "orientation": "camera-aligned E03 frame",
            "opening_size": OPENING_SIZE,
            "max_opening_loss_fraction": MAX_OPENING_LOSS_FRACTION,
            "closing_size": CLOSING_SIZE,
            "height_smoothing": False,
        },
        "variants": VARIANTS,
        "aggregate": aggregate,
        "overview": relative(overview),
        "samples": records,
        "metric_warning": "No direct saved camera-aligned BEV instance GT; this ablation reports physical edits and visual comparisons, not BEV IoU.",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(overview)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
