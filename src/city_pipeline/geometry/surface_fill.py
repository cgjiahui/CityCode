#!/usr/bin/env python3
"""Compare five BEV ID cleanup policies that progressively favor preservation and filling."""

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
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    instance_preview,
    mask_rgb,
    rgb_overlay,
    save_u16,
)
from city_pipeline.geometry.mask_geometry import (  # noqa: E402
    fit_panel,
    font,
)
from city_pipeline.geometry.surface_denoising import (  # noqa: E402
    PROVENANCE_COLORS,
)


BASELINE_ROOT = Path("UNCONFIGURED_BASELINE_ROOT")
E01_ROOT = BASELINE_ROOT / "denoising_v1"
E05_ROOT = BASELINE_ROOT / "camera_aligned_25d_e05_bev_threshold_ablation"
DEFAULT_OUTPUT_ROOT = BASELINE_ROOT / "camera_aligned_25d_e06_fill_preservation_ladder"

CELL_SIZE_M = 0.125
PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M

# C is the exact E05 aggressive baseline. D--G progressively separate the
# destructive opening/component decisions from the desired fill strength.
VARIANTS: dict[str, dict[str, Any]] = {
    "variant_c_e05_baseline": {
        "label": "C / E05 BASELINE / OPEN 3 / COMPONENT 5 m2 / CLOSE 5 / HOLE 12 m2",
        "opening_size": 3,
        "max_opening_loss_fraction": 0.08,
        "minimum_component_area_m2": 5.0,
        "closing_size": 5,
        "maximum_hole_area_m2": 12.0,
        "color": (112, 55, 43),
    },
    "variant_d_no_opening": {
        "label": "D / NO OPENING / COMPONENT 5 m2 / CLOSE 5 / HOLE 12 m2",
        "opening_size": 1,
        "max_opening_loss_fraction": 0.0,
        "minimum_component_area_m2": 5.0,
        "closing_size": 5,
        "maximum_hole_area_m2": 12.0,
        "color": (112, 79, 37),
    },
    "variant_e_preserve_2m2": {
        "label": "E / PRESERVE / NO OPEN / COMPONENT 2 m2 / CLOSE 5 / HOLE 12 m2",
        "opening_size": 1,
        "max_opening_loss_fraction": 0.0,
        "minimum_component_area_m2": 2.0,
        "closing_size": 5,
        "maximum_hole_area_m2": 12.0,
        "color": (41, 92, 78),
    },
    "variant_f_fill_24m2": {
        "label": "F / STRONG FILL / NO OPEN / COMPONENT 2 m2 / CLOSE 7 / HOLE 24 m2",
        "opening_size": 1,
        "max_opening_loss_fraction": 0.0,
        "minimum_component_area_m2": 2.0,
        "closing_size": 7,
        "maximum_hole_area_m2": 24.0,
        "color": (40, 70, 112),
    },
    "variant_g_fill_50m2": {
        "label": "G / UPPER BOUND / NO OPEN / COMPONENT 2 m2 / CLOSE 9 / HOLE 50 m2",
        "opening_size": 1,
        "max_opening_loss_fraction": 0.0,
        "minimum_component_area_m2": 2.0,
        "closing_size": 9,
        "maximum_hole_area_m2": 50.0,
        "color": (87, 48, 105),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e05-root", type=Path, default=E05_ROOT)
    parser.add_argument("--e01-root", type=Path, default=E01_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_float(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def clean_with_policy(
    input_ids: np.ndarray,
    input_surface_z: np.ndarray,
    input_rgb: np.ndarray,
    *,
    opening_size: int,
    max_opening_loss_fraction: float,
    minimum_component_area_m2: float,
    minimum_total_instance_area_m2: float = 0.0,
    closing_size: int,
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
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Clean integer IDs without interpolating or changing any surviving ID value."""

    structure_open = np.ones((opening_size, opening_size), dtype=bool)
    structure_close = np.ones((closing_size, closing_size), dtype=bool)
    pixel_area_m2 = cell_size_m * cell_size_m
    min_component_pixels = int(math.ceil(minimum_component_area_m2 / pixel_area_m2))
    max_hole_pixels = int(math.floor(maximum_hole_area_m2 / pixel_area_m2))
    core_ids = np.zeros_like(input_ids, dtype=np.uint16)
    opening_removed = np.zeros(input_ids.shape, dtype=bool)
    component_removed = np.zeros(input_ids.shape, dtype=bool)
    instance_records: list[dict[str, Any]] = []

    for instance_id in np.unique(input_ids):
        if instance_id == 0:
            continue
        raw = input_ids == instance_id
        y, x = np.nonzero(raw)
        margin = max(opening_size, closing_size)
        y0 = max(0, int(np.min(y)) - margin)
        y1 = min(raw.shape[0], int(np.max(y)) + margin + 1)
        x0 = max(0, int(np.min(x)) - margin)
        x1 = min(raw.shape[1], int(np.max(x)) + margin + 1)
        crop = raw[y0:y1, x0:x1]
        raw_count = int(np.count_nonzero(crop))

        if opening_size <= 1:
            candidate = crop
            opened_count = raw_count
            opening_loss = 0.0
            opening_applied = False
        else:
            opened = ndimage.binary_opening(crop, structure=structure_open)
            opened_count = int(np.count_nonzero(opened))
            opening_loss = 1.0 - opened_count / raw_count
            opening_applied = (
                opened_count > 0 and opening_loss <= max_opening_loss_fraction
            )
            candidate = opened if opening_applied else crop
            if opening_applied:
                opening_removed[y0:y1, x0:x1] |= crop & ~opened

        labels, component_count = ndimage.label(
            candidate, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= min_component_pixels
        if keep.size:
            keep[0] = False
        retained = keep[labels]
        component_removed[y0:y1, x0:x1] |= candidate & ~retained
        core_ids[y0:y1, x0:x1][retained] = instance_id
        instance_records.append(
            {
                "instance_id": int(instance_id),
                "input_cell_count": raw_count,
                "opening_enabled": bool(opening_size > 1),
                "opening_applied": bool(opening_applied),
                "opening_loss_fraction_candidate": float(opening_loss),
                "post_opening_cell_count": int(np.count_nonzero(candidate)),
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
        y0 = max(0, int(np.min(y)) - closing_size)
        y1 = min(mask.shape[0], int(np.max(y)) + closing_size + 1)
        x0 = max(0, int(np.min(x)) - closing_size)
        x1 = min(mask.shape[1], int(np.max(x)) + closing_size + 1)
        crop = mask[y0:y1, x0:x1]
        closed = ndimage.binary_closing(crop, structure=structure_close)
        additions = closed & ~crop & (input_ids[y0:y1, x0:x1] == 0)
        claim_count[y0:y1, x0:x1][additions] += 1
        claim_id[y0:y1, x0:x1][additions] = instance_id

    final_ids = core_ids.copy()
    closing_fill = (claim_count == 1) & (core_ids == 0) & (input_ids == 0)
    final_ids[closing_fill] = claim_id[closing_fill]

    hole_fill = np.zeros(input_ids.shape, dtype=bool)
    rejected_large_holes = np.zeros(input_ids.shape, dtype=bool)
    rejected_competing_holes = np.zeros(input_ids.shape, dtype=bool)
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
            candidate = labels == label_id
            accepted_by_area = pixel_count <= max_hole_pixels
            accepted = accepted_by_area
            competing = False
            if accepted:
                input_crop = input_ids[y0:y1, x0:x1]
                final_crop = final_ids[y0:y1, x0:x1]
                competing = bool(
                    np.any(
                        (input_crop[candidate] > 0)
                        & (input_crop[candidate] != instance_id)
                    )
                    or np.any(final_crop[candidate] > 0)
                )
                accepted = not competing
                if accepted:
                    final_crop[candidate] = instance_id
                    hole_fill[y0:y1, x0:x1] |= candidate
            if not accepted_by_area:
                rejected_large_holes[y0:y1, x0:x1] |= candidate
            elif competing:
                rejected_competing_holes[y0:y1, x0:x1] |= candidate
            hole_records.append(
                {
                    "instance_id": int(instance_id),
                    "pixel_count": pixel_count,
                    "area_m2": float(pixel_count * pixel_area_m2),
                    "accepted_by_area": bool(accepted_by_area),
                    "competing_id": bool(competing),
                    "filled": bool(accepted),
                }
            )

    # Component filtering alone can retain a complete but globally tiny instance
    # (for example a single 16 m2 boundary sliver when the component threshold is
    # 12 m2).  Keep this separate from the component threshold so callers can
    # reject only whole tiny instances without eroding valid multi-part buildings.
    whole_instance_removed = np.zeros(input_ids.shape, dtype=bool)
    whole_instance_removed_ids: list[int] = []
    whole_instance_removed_records: list[dict[str, Any]] = []
    if minimum_total_instance_area_m2 > 0.0:
        for instance_id in np.unique(final_ids):
            if instance_id == 0:
                continue
            mask = final_ids == instance_id
            cell_count = int(np.count_nonzero(mask))
            area_m2 = float(cell_count * pixel_area_m2)
            if area_m2 >= minimum_total_instance_area_m2:
                continue
            whole_instance_removed_ids.append(int(instance_id))
            whole_instance_removed_records.append(
                {
                    "instance_id": int(instance_id),
                    "final_cell_count_before_removal": cell_count,
                    "final_area_m2_before_removal": area_m2,
                }
            )
            whole_instance_removed |= mask
            final_ids[mask] = 0
            core_ids[core_ids == instance_id] = 0
            component_removed |= input_ids == instance_id

        closing_fill &= final_ids > 0
        hole_fill &= final_ids > 0

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
        "method": "integer-ID preservation/fill ladder; no color interpolation",
        "cell_size_m": cell_size_m,
        "pixel_area_m2": pixel_area_m2,
        "opening_size": opening_size,
        "max_opening_loss_fraction": max_opening_loss_fraction,
        "minimum_component_area_m2": minimum_component_area_m2,
        "minimum_component_pixels": min_component_pixels,
        "minimum_total_instance_area_m2": minimum_total_instance_area_m2,
        "closing_size": closing_size,
        "maximum_hole_area_m2": maximum_hole_area_m2,
        "maximum_hole_pixels": max_hole_pixels,
        "input_cell_count": int(np.count_nonzero(input_ids)),
        "core_cell_count": int(np.count_nonzero(core_ids)),
        "final_cell_count": int(np.count_nonzero(final_ids)),
        "input_instance_count": int(np.unique(input_ids[input_ids > 0]).size),
        "final_instance_count": int(np.unique(final_ids[final_ids > 0]).size),
        "opening_removed_cell_count": int(np.count_nonzero(opening_removed)),
        "opening_removed_area_m2": float(np.count_nonzero(opening_removed) * pixel_area_m2),
        "component_removed_cell_count": int(np.count_nonzero(component_removed)),
        "component_removed_area_m2": float(np.count_nonzero(component_removed) * pixel_area_m2),
        "whole_instance_removed_count": len(whole_instance_removed_ids),
        "whole_instance_removed_ids": whole_instance_removed_ids,
        "whole_instance_removed_cell_count": int(np.count_nonzero(whole_instance_removed)),
        "whole_instance_removed_area_m2": float(
            np.count_nonzero(whole_instance_removed) * pixel_area_m2
        ),
        "whole_instance_removed_records": whole_instance_removed_records,
        "removed_cell_count": int(np.count_nonzero(removed)),
        "removed_area_m2": float(np.count_nonzero(removed) * pixel_area_m2),
        "closing_fill_cell_count": int(np.count_nonzero(closing_fill)),
        "closing_fill_area_m2": float(np.count_nonzero(closing_fill) * pixel_area_m2),
        "hole_fill_cell_count": int(np.count_nonzero(hole_fill)),
        "hole_fill_area_m2": float(np.count_nonzero(hole_fill) * pixel_area_m2),
        "rejected_large_hole_cell_count": int(np.count_nonzero(rejected_large_holes)),
        "rejected_competing_hole_cell_count": int(np.count_nonzero(rejected_competing_holes)),
        "raw_building_lost_cell_count": int(np.count_nonzero((input_ids > 0) & (final_ids == 0))),
        "raw_background_added_cell_count": int(np.count_nonzero((input_ids == 0) & (final_ids > 0))),
        "removed_complete_instance_ids": [
            item["instance_id"] for item in instance_records if item["removed_complete_instance"]
        ],
        "instance_records": instance_records,
        "hole_records": hole_records,
        "height_smoothing": False,
        "id_color_policy": "uint16 IDs are copied or assigned exactly; previews are visualization only",
    }
    return (
        core_ids,
        final_ids,
        clean_surface_z,
        clean_rgb,
        provenance,
        opening_removed,
        component_removed,
        closing_fill,
        hole_fill,
        metadata,
    )


def comparison_figure(
    sample_id: str,
    raw_ids: np.ndarray,
    raw_rgb: np.ndarray,
    variant_panels: list[tuple[str, tuple[int, int, int], list[tuple[str, str, np.ndarray]]]],
    destination: Path,
) -> None:
    tile = 286
    gap = 6
    title_height = 52
    row_label_height = 30
    header = 64
    columns = 6
    rows = 1 + len(variant_panels)
    width = columns * tile + (columns - 1) * gap
    row_height = row_label_height + header + tile
    height = title_height + rows * row_height + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (17, 19, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        f"{sample_id} / WHY HOLES GROW + PRESERVATION/FILL LADDER",
        font=font(22),
        fill=(248, 248, 248),
    )
    raw_overlay = rgb_overlay(raw_rgb, instance_preview(raw_ids), raw_ids > 0)
    empty = np.zeros(raw_ids.shape, dtype=np.uint8)
    all_rows = [
        (
            "SHARED INPUT / RAW CAMERA-ALIGNED BEV INTEGER IDs",
            (61, 65, 73),
            [
                ("Raw BEV IDs", f"{np.unique(raw_ids[raw_ids > 0]).size} IDs", instance_preview(raw_ids)),
                ("Raw overlay", f"cells={np.count_nonzero(raw_ids)}", raw_overlay),
                ("Opening removed", "none before cleanup", mask_rgb(empty)),
                ("Component removed", "none before cleanup", mask_rgb(empty)),
                ("Closing fill", "none before cleanup", mask_rgb(empty)),
                ("Enclosed-hole fill", "none before cleanup", mask_rgb(empty)),
            ],
        ),
        *variant_panels,
    ]
    for row_index, (row_label, row_color, panels) in enumerate(all_rows):
        y = title_height + row_index * (row_height + gap)
        draw.rectangle((0, y, width, y + row_label_height - 1), fill=row_color)
        draw.text((10, y + 4), row_label, font=font(13), fill=(248, 248, 248))
        header_y = y + row_label_height
        for column, (label, subtitle, array) in enumerate(panels):
            x = column * (tile + gap)
            draw.rectangle((x, header_y, x + tile - 1, header_y + header - 1), fill=(35, 38, 44))
            draw.text((x + 8, header_y + 5), label, font=font(16), fill=(255, 255, 255))
            draw.text((x + 8, header_y + 35), subtitle, font=font(11), fill=(190, 197, 207))
            canvas.paste(fit_panel(array, tile, nearest=True), (x, header_y + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def build_tall_overview(paths: list[Path], destination: Path) -> None:
    """Build an overview sized for the unusually tall six-row audit figures."""

    columns = 2
    tile_width = 900
    tile_height = 1230
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


def main() -> int:
    args = parse_args()
    e05_root = args.e05_root.resolve()
    e01_root = args.e01_root.resolve()
    output_root = args.output_root.resolve()
    baseline = read_json(BASELINE_ROOT / "manifest.json")
    e05 = read_json(e05_root / "manifest.json")
    e01 = read_json(e01_root / "manifest.json")
    baseline_by_id = {item["id"]: item for item in baseline["samples"]}
    e01_by_id = {item["id"]: item for item in e01["samples"]}
    e05_by_id = {item["id"]: item for item in e05["samples"]}
    sample_ids = args.sample_id or [item["id"] for item in e05["samples"]]

    output_names = (
        "bev_instance_ids_core",
        "bev_instance_ids_final",
        "bev_instance_previews_final",
        "bev_opening_removed",
        "bev_component_removed",
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
    variant_dirs: dict[str, dict[str, Path]] = {}
    for key in VARIANTS:
        variant_dirs[key] = {name: output_root / key / name for name in output_names}
        for path in variant_dirs[key].values():
            path.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        e05_sample = e05_by_id[sample_id]
        _ = e01_by_id[sample_id]
        baseline_sample = baseline_by_id[sample_id]
        shared = e05_sample["shared_outputs"]
        raw_ids = np.asarray(Image.open(project_path(shared["bev_raw_ids"])), dtype=np.uint16)
        raw_rgb = load_rgb(project_path(shared["bev_rgb_raw"]))
        surface_z = np.load(project_path(shared["bev_surface_z"])).astype(np.float32)
        camera = read_json(project_path(baseline_sample["input"]["camera"]))
        ground_reference_z = float(camera["ground_reference_z_m"])

        sample_record: dict[str, Any] = {
            "id": sample_id,
            "shared_e05_inputs": shared,
            "variants": {},
        }
        panel_rows = []
        for key, parameters in VARIANTS.items():
            kwargs = {name: value for name, value in parameters.items() if name not in {"label", "color"}}
            (
                core_ids,
                final_ids,
                clean_surface_z,
                clean_rgb,
                provenance,
                opening_removed,
                component_removed,
                closing_fill,
                hole_fill,
                cleaning,
            ) = clean_with_policy(raw_ids, surface_z, raw_rgb, **kwargs)
            building_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
            building_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
            building_height[building_valid] = clean_surface_z[building_valid] - np.float32(ground_reference_z)
            height_rgb = height_preview(building_height, building_valid)
            height_rgb[~building_valid] = 0
            final_preview = instance_preview(final_ids)
            overlay = rgb_overlay(clean_rgb, final_preview, final_ids > 0)
            directories = variant_dirs[key]
            paths = {
                "core_ids": directories["bev_instance_ids_core"] / f"{sample_id}__bev_instance_ids_core.png",
                "final_ids": directories["bev_instance_ids_final"] / f"{sample_id}__bev_instance_ids_final.png",
                "final_preview": directories["bev_instance_previews_final"] / f"{sample_id}__bev_instance_preview_final.png",
                "opening_removed": directories["bev_opening_removed"] / f"{sample_id}__bev_opening_removed.png",
                "component_removed": directories["bev_component_removed"] / f"{sample_id}__bev_component_removed.png",
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
            save_rgb(paths["opening_removed"], mask_rgb(opening_removed))
            save_rgb(paths["component_removed"], mask_rgb(component_removed))
            save_rgb(paths["closing_fill"], mask_rgb(closing_fill))
            save_rgb(paths["hole_fill"], mask_rgb(hole_fill))
            save_u8(paths["provenance"], provenance)
            save_rgb(paths["provenance_preview"], PROVENANCE_COLORS[provenance])
            save_rgb(paths["bev_rgb"], clean_rgb)
            save_rgb(paths["overlay"], overlay)
            save_float(paths["surface_z"], clean_surface_z)
            save_float(paths["building_height"], building_height)
            save_rgb(paths["building_height_preview"], height_rgb)
            sample_record["variants"][key] = {
                "parameters": {name: value for name, value in parameters.items() if name != "color"},
                "cleaning": cleaning,
                "outputs": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in paths.items()},
            }
            panel_rows.append(
                (
                    str(parameters["label"]),
                    tuple(parameters["color"]),
                    [
                        ("Final BEV IDs", f"cells={cleaning['final_cell_count']}", final_preview),
                        ("Final overlay", f"lost raw={cleaning['raw_building_lost_cell_count']}", overlay),
                        ("Opening removed", f"{cleaning['opening_removed_area_m2']:.1f} m2", mask_rgb(opening_removed)),
                        ("Component removed", f"{cleaning['component_removed_area_m2']:.1f} m2", mask_rgb(component_removed)),
                        ("Closing fill", f"{cleaning['closing_fill_area_m2']:.1f} m2", mask_rgb(closing_fill)),
                        ("Enclosed-hole fill", f"{cleaning['hole_fill_area_m2']:.1f} m2", mask_rgb(hole_fill)),
                    ],
                )
            )

        comparison = comparison_dir / f"{sample_id}__preservation_fill_ladder_c_to_g.png"
        comparison_figure(sample_id, raw_ids, raw_rgb, panel_rows, comparison)
        sample_record["comparison"] = str(comparison.relative_to(PROJECT_ROOT))
        records.append(sample_record)
        comparisons.append(comparison)

    overview = output_root / "all_8_preservation_fill_ladder_c_to_g.png"
    build_tall_overview(comparisons, overview)
    aggregate: dict[str, Any] = {}
    for key in VARIANTS:
        cleanings = [sample["variants"][key]["cleaning"] for sample in records]
        aggregate[key] = {
            "parameters": {name: value for name, value in VARIANTS[key].items() if name != "color"},
            "input_instance_count_sum": int(sum(item["input_instance_count"] for item in cleanings)),
            "final_instance_count_sum": int(sum(item["final_instance_count"] for item in cleanings)),
            "removed_complete_instance_count": int(sum(len(item["removed_complete_instance_ids"]) for item in cleanings)),
            "opening_removed_area_m2": float(sum(item["opening_removed_area_m2"] for item in cleanings)),
            "component_removed_area_m2": float(sum(item["component_removed_area_m2"] for item in cleanings)),
            "removed_area_m2": float(sum(item["removed_area_m2"] for item in cleanings)),
            "closing_fill_area_m2": float(sum(item["closing_fill_area_m2"] for item in cleanings)),
            "hole_fill_area_m2": float(sum(item["hole_fill_area_m2"] for item in cleanings)),
            "raw_building_lost_area_m2": float(sum(item["raw_building_lost_cell_count"] * PIXEL_AREA_M2 for item in cleanings)),
            "raw_background_added_area_m2": float(sum(item["raw_background_added_cell_count"] * PIXEL_AREA_M2 for item in cleanings)),
            "final_building_area_m2": float(sum(item["final_cell_count"] * PIXEL_AREA_M2 for item in cleanings)),
        }

    manifest = {
        "schema": "air2vectorcity.camera_aligned_bev_fill_preservation_ladder_e06.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete five-policy preservation/fill ladder; E05 unchanged",
        "shared_input": {
            "source": str(e05_root.relative_to(PROJECT_ROOT)),
            "description": "exact E05 raw camera-aligned 2048 BEV uint16 IDs, RGB, and surface Z",
            "bev_resolution": [2048, 2048],
            "cell_size_m": CELL_SIZE_M,
            "pixel_area_m2": PIXEL_AREA_M2,
            "id_normalization": "already uint16 integer IDs before all C--G policies",
        },
        "variants": VARIANTS,
        "aggregate": aggregate,
        "overview": str(overview.relative_to(PROJECT_ROOT)),
        "samples": records,
        "interpretation_warning": "G is a visual upper bound and may fill real courtyards up to 50 m2; it is not automatically the recommended production policy.",
        "metric_warning": "No direct saved camera-aligned BEV instance GT; report physical edits and visual comparisons, not BEV IoU.",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(overview)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
