#!/usr/bin/env python3
"""Apply a more aggressive, safeguarded cleanup to E03 camera-aligned BEV."""

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
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.roof_masks import two_row_comparison
from city_pipeline.geometry.camera_geometry import (
    height_preview,
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.camera_bev import (
    building_point_cloud,
    per_instance_statistics,
)
from city_pipeline.geometry.depth_lifting import (
    build_overview,
    instance_palette,
    instance_preview,
    mask_rgb,
    rgb_overlay,
    save_u16,
    write_binary_ply,
)


DEFAULT_E03_ROOT = Path("UNCONFIGURED_DEFAULT_E03_ROOT")
DEFAULT_OUTPUT_ROOT = Path("UNCONFIGURED_DEFAULT_OUTPUT_ROOT")

CELL_SIZE_M = 0.125
PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
OPENING_SIZE = 3
MAX_OPENING_LOSS_FRACTION = 0.08
MIN_COMPONENT_AREA_M2 = 5.0
CLOSING_SIZE = 5
MAX_HOLE_AREA_M2 = 12.0

PROVENANCE_COLORS = np.asarray(
    [
        [0, 0, 0],
        [50, 210, 100],
        [255, 210, 40],
        [255, 80, 190],
        [40, 180, 255],
        [255, 120, 35],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e03-root", type=Path, default=DEFAULT_E03_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def color_provenance(provenance: np.ndarray) -> np.ndarray:
    return PROVENANCE_COLORS[provenance]


def aggressive_clean(
    input_ids: np.ndarray,
    input_surface_z: np.ndarray,
    input_rgb: np.ndarray,
    input_provenance: np.ndarray,
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
    """Remove thin exterior noise and fill moderate unambiguous holes."""

    structure_open = np.ones((OPENING_SIZE, OPENING_SIZE), dtype=bool)
    structure_close = np.ones((CLOSING_SIZE, CLOSING_SIZE), dtype=bool)
    min_component_pixels = int(math.ceil(MIN_COMPONENT_AREA_M2 / PIXEL_AREA_M2))
    max_hole_pixels = int(math.floor(MAX_HOLE_AREA_M2 / PIXEL_AREA_M2))
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

        labels, count = ndimage.label(
            candidate, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.reshape(-1))
        keep = sizes >= min_component_pixels
        if keep.size:
            keep[0] = False
        # Unlike the conservative E01 baseline, E04 has no largest-component
        # exception below 5 m2.  A provisional instance without any component
        # meeting the physical threshold is removed as exterior noise.
        retained = keep[labels]
        output = core_ids[y0:y1, x0:x1]
        output[retained] = instance_id
        instance_records.append(
            {
                "instance_id": int(instance_id),
                "input_cell_count": raw_count,
                "opening_applied": bool(opening_applied),
                "opening_loss_fraction_candidate": float(opening_loss),
                "component_count": int(count),
                "retained_component_count": int(np.count_nonzero(keep)),
                "core_cell_count": int(np.count_nonzero(retained)),
            }
        )

    # A 5x5 closing removes narrow notches and cracks.  Additions are accepted
    # only if exactly one instance claims the pixel and no original competing
    # instance occupied it.
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

    # Fill larger but still bounded holes than E01.  Courtyards larger than
    # 12 m2 and holes touching the exterior remain untouched.
    hole_fill = np.zeros(input_ids.shape, dtype=bool)
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
        labels, count = ndimage.label(
            enclosed, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.reshape(-1))
        for label_id in range(1, count + 1):
            if sizes[label_id] > max_hole_pixels:
                continue
            candidate = labels == label_id
            input_crop = input_ids[y0:y1, x0:x1]
            final_crop = final_ids[y0:y1, x0:x1]
            competing_input = (
                (input_crop[candidate] > 0)
                & (input_crop[candidate] != instance_id)
            )
            if np.any(competing_input) or np.any(final_crop[candidate] > 0):
                continue
            final_crop[candidate] = instance_id
            hole_fill[y0:y1, x0:x1] |= candidate

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
        support = (
            (final_ids == instance_id)
            & np.isfinite(clean_surface_z)
            & ~target
        )
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
        _, nearest = ndimage.distance_transform_edt(
            ~support_crop, return_indices=True
        )
        z_crop = clean_surface_z[y0:y1, x0:x1]
        rgb_crop = clean_rgb[y0:y1, x0:x1]
        z_crop[target_crop] = z_crop[nearest[0][target_crop], nearest[1][target_crop]]
        rgb_crop[target_crop] = rgb_crop[
            nearest[0][target_crop], nearest[1][target_crop]
        ]

    provenance = np.zeros(input_ids.shape, dtype=np.uint8)
    retained = final_ids == input_ids
    provenance[retained] = input_provenance[retained]
    # Some input building pixels may have provenance zero only because they
    # were valid non-filled E03 cells in an older manifest; keep them observed.
    provenance[retained & (final_ids > 0) & (provenance == 0)] = 1
    provenance[added_with_observed_height & (final_ids > 0)] = 4
    provenance[added_geometry_fill & (final_ids > 0)] = 5

    metadata = {
        "method": "safeguarded 3x3 opening + >=5m2 components + unique 5x5 closing + <=12m2 holes",
        "opening_size": OPENING_SIZE,
        "max_opening_loss_fraction": MAX_OPENING_LOSS_FRACTION,
        "minimum_component_area_m2": MIN_COMPONENT_AREA_M2,
        "minimum_component_pixels": min_component_pixels,
        "closing_size": CLOSING_SIZE,
        "maximum_hole_area_m2": MAX_HOLE_AREA_M2,
        "maximum_hole_pixels": max_hole_pixels,
        "input_cell_count": int(np.count_nonzero(input_ids)),
        "core_cell_count": int(np.count_nonzero(core_ids)),
        "final_cell_count": int(np.count_nonzero(final_ids)),
        "removed_cell_count": int(np.count_nonzero(removed)),
        "removed_area_m2": float(np.count_nonzero(removed) * PIXEL_AREA_M2),
        "removed_fraction": float(
            np.count_nonzero(removed) / max(1, np.count_nonzero(input_ids))
        ),
        "closing_fill_cell_count": int(np.count_nonzero(closing_fill)),
        "closing_fill_area_m2": float(
            np.count_nonzero(closing_fill) * PIXEL_AREA_M2
        ),
        "hole_fill_cell_count": int(np.count_nonzero(hole_fill)),
        "hole_fill_area_m2": float(np.count_nonzero(hole_fill) * PIXEL_AREA_M2),
        "added_observed_height_cell_count": int(
            np.count_nonzero(added_with_observed_height)
        ),
        "added_geometry_fill_cell_count": int(
            np.count_nonzero(added_geometry_fill)
        ),
        "instance_records": instance_records,
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


def save_float(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def main() -> int:
    args = parse_args()
    e03_root = args.e03_root.resolve()
    output_root = args.output_root.resolve()
    e03 = read_json(e03_root / "manifest.json")
    samples = e03["samples"]
    if args.sample_id:
        requested = set(args.sample_id)
        samples = [sample for sample in samples if sample["id"] in requested]

    names = (
        "bev_instance_ids_opened_component_cleaned",
        "bev_instance_ids_final",
        "bev_instance_previews_final",
        "bev_removed_external_noise",
        "bev_closing_fill_masks",
        "bev_hole_fill_masks",
        "bev_cleaning_provenance",
        "bev_cleaning_provenance_previews",
        "bev_surface_z_m",
        "bev_building_height_m",
        "bev_building_height_previews",
        "bev_rgb",
        "bev_instance_overlays",
        "building_point_cloud_npz",
        "building_point_cloud_ply_instance",
        "per_instance_statistics",
        "comparisons",
    )
    directories = {name: output_root / name for name in names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    for index, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        print(f"[{index}/{len(samples)}] {sample_id}", flush=True)
        input_ids = np.asarray(
            Image.open(project_path(sample["outputs"]["final_ids"])), dtype=np.uint16
        )
        input_rgb = load_rgb(project_path(sample["outputs"]["bev_rgb"]))
        input_surface_z = np.load(project_path(sample["outputs"]["surface_z"])).astype(
            np.float32
        )
        input_provenance = np.asarray(
            Image.open(project_path(sample["outputs"]["provenance"])), dtype=np.uint8
        )
        transform = read_json(project_path(sample["outputs"]["transform"]))

        (
            core_ids,
            final_ids,
            clean_surface_z,
            clean_rgb,
            provenance,
            closing_mask,
            hole_mask,
            cleaning,
        ) = aggressive_clean(
            input_ids, input_surface_z, input_rgb, input_provenance
        )
        removed = (input_ids > 0) & (core_ids == 0)
        # Includes new background fills and same-instance pixels temporarily
        # removed by opening but restored because they form an approved hole.
        added = (final_ids > 0) & (core_ids == 0)

        building_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
        ground_reference_z = 0.48
        building_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
        building_height[building_valid] = (
            clean_surface_z[building_valid] - np.float32(ground_reference_z)
        )
        input_preview = instance_preview(input_ids)
        core_preview = instance_preview(core_ids)
        final_preview = instance_preview(final_ids)
        height_rgb = height_preview(building_height, building_valid)
        height_rgb[~building_valid] = 0
        overlay = rgb_overlay(clean_rgb, final_preview, building_valid)
        provenance_rgb = color_provenance(provenance)

        paths = {
            "core_ids": directories["bev_instance_ids_opened_component_cleaned"] / f"{sample_id}__bev_instance_ids_opened_component_cleaned.png",
            "final_ids": directories["bev_instance_ids_final"] / f"{sample_id}__bev_instance_ids_final.png",
            "final_preview": directories["bev_instance_previews_final"] / f"{sample_id}__bev_instance_preview_final.png",
            "removed": directories["bev_removed_external_noise"] / f"{sample_id}__bev_removed_external_noise.png",
            "closing_fill": directories["bev_closing_fill_masks"] / f"{sample_id}__bev_closing_fill.png",
            "hole_fill": directories["bev_hole_fill_masks"] / f"{sample_id}__bev_hole_fill.png",
            "provenance": directories["bev_cleaning_provenance"] / f"{sample_id}__bev_cleaning_provenance.png",
            "provenance_preview": directories["bev_cleaning_provenance_previews"] / f"{sample_id}__bev_cleaning_provenance_preview.png",
            "surface_z": directories["bev_surface_z_m"] / f"{sample_id}__bev_surface_z_m.npy",
            "building_height": directories["bev_building_height_m"] / f"{sample_id}__bev_building_height_m.npy",
            "height_preview": directories["bev_building_height_previews"] / f"{sample_id}__bev_building_height_preview.png",
            "bev_rgb": directories["bev_rgb"] / f"{sample_id}__bev_rgb.png",
            "overlay": directories["bev_instance_overlays"] / f"{sample_id}__bev_instance_overlay.png",
            "point_npz": directories["building_point_cloud_npz"] / f"{sample_id}__building_points_aggressive.npz",
            "point_ply": directories["building_point_cloud_ply_instance"] / f"{sample_id}__building_instances_aggressive.ply",
            "stats": directories["per_instance_statistics"] / f"{sample_id}__per_instance_statistics.json",
            "comparison": directories["comparisons"] / f"{sample_id}__e03_vs_e04_aggressive.png",
        }
        save_u16(paths["core_ids"], core_ids)
        save_u16(paths["final_ids"], final_ids)
        save_rgb(paths["final_preview"], final_preview)
        save_u8(paths["removed"], removed.astype(np.uint8) * 255)
        save_u8(paths["closing_fill"], closing_mask.astype(np.uint8) * 255)
        save_u8(paths["hole_fill"], hole_mask.astype(np.uint8) * 255)
        save_u8(paths["provenance"], provenance)
        save_rgb(paths["provenance_preview"], provenance_rgb)
        save_float(paths["surface_z"], clean_surface_z)
        save_float(paths["building_height"], building_height)
        save_rgb(paths["height_preview"], height_rgb)
        save_rgb(paths["bev_rgb"], clean_rgb)
        save_rgb(paths["overlay"], overlay)

        xyz, point_rgb, point_ids, point_provenance = building_point_cloud(
            clean_surface_z,
            clean_rgb,
            final_ids,
            provenance,
            transform["pixel_center_index_to_ue_world_xy_3x3"],
        )
        np.savez_compressed(
            paths["point_npz"],
            xyz_m=xyz,
            rgb=point_rgb,
            instance_id=point_ids,
            provenance=point_provenance,
        )
        ply = write_binary_ply(
            paths["point_ply"],
            xyz,
            instance_palette(int(np.max(point_ids)))[point_ids],
            point_ids,
            args.max_ply_points,
        )
        source_ids = np.asarray(
            Image.open(project_path(sample["input"]["e01_cleaned_source_ids"])),
            dtype=np.uint16,
        )
        stats = per_instance_statistics(
            source_ids,
            final_ids,
            clean_surface_z,
            ground_reference_z,
            transform["pixel_center_index_to_ue_world_xy_3x3"],
        )
        paths["stats"].write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        two_row_comparison(
            f"{sample_id} / E04 AGGRESSIVE DENOISING",
            [
                ("E03 IDs", "conservative input", input_preview, True),
                ("E03 height", "unsmoothed", load_rgb(project_path(sample["outputs"]["building_height_preview"])), False),
                ("E03 overlay", "before E04", load_rgb(project_path(sample["outputs"]["overlay"])), False),
                ("E04 core", "3x3 guarded + >=5m2", core_preview, True),
                ("Removed exterior", f"{cleaning['removed_area_m2']:.1f}m2", mask_rgb(removed), True),
                ("Added pixels", f"{np.count_nonzero(added)*PIXEL_AREA_M2:.1f}m2", mask_rgb(added), True),
            ],
            [
                ("E04 final IDs", "5x5 close + <=12m2 holes", final_preview, True),
                ("Closing fill", "unique-instance claims", mask_rgb(closing_mask), True),
                ("Hole fill", "enclosed only", mask_rgb(hole_mask), True),
                ("Provenance", "green observed / blue-orange E04", provenance_rgb, True),
                ("E04 height", "original heights unsmoothed", height_rgb, False),
                ("E04 overlay", "final aggressive mask", overlay, False),
            ],
            paths["comparison"],
            tile=320,
        )
        comparisons.append(paths["comparison"])
        records.append(
            {
                "id": sample_id,
                "e03_sample": sample_id,
                "cleaning": cleaning,
                "final_instance_count": int(np.unique(final_ids[final_ids > 0]).size),
                "final_point_count": int(xyz.shape[0]),
                "ply": ply,
                "transform": sample["outputs"]["transform"],
                "outputs": {
                    **{
                        key: path.relative_to(PROJECT_ROOT).as_posix()
                        for key, path in paths.items()
                    },
                    "building_point_cloud_npz": paths["point_npz"].relative_to(PROJECT_ROOT).as_posix(),
                },
            }
        )

    overview = output_root / "all_8_e03_vs_e04_aggressive.png"
    build_overview(comparisons, overview)
    manifest = {
        "schema": "air2vectorcity.camera_aligned_25d_aggressive_e04.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "aggressive_candidate_not_yet_baseline",
        "input_e03": e03_root.relative_to(PROJECT_ROOT).as_posix(),
        "policy": {
            "opening_size": OPENING_SIZE,
            "max_opening_loss_fraction": MAX_OPENING_LOSS_FRACTION,
            "minimum_component_area_m2": MIN_COMPONENT_AREA_M2,
            "closing_size": CLOSING_SIZE,
            "maximum_hole_area_m2": MAX_HOLE_AREA_M2,
            "height_smoothing": False,
        },
        "overview": overview.relative_to(PROJECT_ROOT).as_posix(),
        "samples": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
