#!/usr/bin/env python3
"""Compare three guarded near-black parameter sets after Variant B blue filtering."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.camera_geometry import read_json  # noqa: E402
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    V7B_PALETTE,
    instance_preview,
    mask_rgb,
    save_u16,
)
from city_pipeline.geometry.mask_geometry import (  # noqa: E402
    BASELINE_ROOT,
    E04_ROOT,
    binary_metrics,
    build_overview,
    building_only_colors,
    fit_panel,
    font,
    load_rgb,
    project_path,
    retain_existing_ids,
    save_rgb,
    stable_seed_mask,
    strong_v7b_black_gate,
)


DEFAULT_OUTPUT_ROOT = E04_ROOT / "source_blue_dark_parameter_comparison"
DEFAULT_DATASET_ROOT = Path("UNCONFIGURED_DEFAULT_DATASET_ROOT")
CURRENT_BLACK_DISTANCE = 35.0
GROUP_CLOSING_SIZE = 3
PARAMETER_SETS: dict[str, dict[str, float | int | str]] = {
    "previous_80_d1_r3": {
        "label": "PREVIOUS C / 80, 1 PX, 3 PX",
        "black_upper": 80.0,
        "v7b_dilation_px": 1,
        "max_internal_radius_px": 3.0,
    },
    "mild_100_d2_r4": {
        "label": "MILD / 100, 2 PX, 4 PX",
        "black_upper": 100.0,
        "v7b_dilation_px": 2,
        "max_internal_radius_px": 4.0,
    },
    "aggressive_120_d2_r5": {
        "label": "AGGRESSIVE / 120, 2 PX, 5 PX",
        "black_upper": 120.0,
        "v7b_dilation_px": 2,
        "max_internal_radius_px": 5.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def guarded_near_black(
    rgb: np.ndarray,
    original_building: np.ndarray,
    blue_filtered_building: np.ndarray,
    v7b_surface: np.ndarray,
    black_upper: float,
    v7b_dilation_px: int,
    max_internal_radius_px: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    work = rgb.astype(np.int32)
    distance = np.sqrt(np.sum(work * work, axis=-1))
    candidate = (
        blue_filtered_building
        & (distance > CURRENT_BLACK_DISTANCE)
        & (distance <= float(black_upper))
    )
    local_support = ndimage.binary_dilation(
        v7b_surface,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(v7b_dilation_px),
    )
    supported = candidate & local_support
    grouped = ndimage.binary_closing(
        supported,
        structure=np.ones((GROUP_CLOSING_SIZE, GROUP_CLOSING_SIZE), dtype=bool),
    )
    labels, component_count = ndimage.label(
        grouped, structure=np.ones((3, 3), dtype=np.uint8)
    )
    stable = stable_seed_mask(rgb, original_building)
    remove = np.zeros(original_building.shape, dtype=bool)
    records: list[dict[str, Any]] = []
    for component_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        component = labels[slices] == component_id
        original = supported[slices] & component
        count = int(np.count_nonzero(original))
        if not count:
            continue
        radius = float(np.max(ndimage.distance_transform_edt(component)))
        stable_overlap = int(np.count_nonzero(original & stable[slices]))
        accepted = radius <= float(max_internal_radius_px) and stable_overlap == 0
        if accepted:
            target = remove[slices]
            target |= original
        records.append(
            {
                "component_id": component_id,
                "original_candidate_pixels": count,
                "maximum_internal_radius_px": radius,
                "stable_seed_overlap_pixels": stable_overlap,
                "removed": accepted,
            }
        )
    refined = blue_filtered_building & ~remove
    metadata = {
        "current_black_distance": CURRENT_BLACK_DISTANCE,
        "black_upper": float(black_upper),
        "v7b_surface_dilation_px": int(v7b_dilation_px),
        "max_internal_radius_px": float(max_internal_radius_px),
        "temporary_group_closing_size": GROUP_CLOSING_SIZE,
        "stable_seed_protection": "any overlap with a >=80px stable seed protects the group",
        "near_black_candidate_pixels": int(np.count_nonzero(candidate)),
        "v7b_supported_candidate_pixels": int(np.count_nonzero(supported)),
        "removed_near_black_pixels": int(np.count_nonzero(remove)),
        "retained_building_pixels": int(np.count_nonzero(refined)),
        "temporary_component_count": int(component_count),
        "removed_component_count": int(sum(item["removed"] for item in records)),
        "protected_component_count": int(sum(not item["removed"] for item in records)),
        "components": records,
    }
    return refined, metadata, {
        "candidate": candidate,
        "local_support": local_support,
        "supported": supported,
        "grouped": grouped,
        "remove": remove,
    }


def parameter_comparison_figure(
    sample_id: str,
    rows: list[tuple[str, tuple[int, int, int], list[tuple[str, str, np.ndarray]]]],
    destination: Path,
) -> None:
    tile = 330
    gap = 7
    title_height = 52
    row_label_height = 30
    header = 66
    columns = 5
    width = columns * tile + (columns - 1) * gap
    row_height = row_label_height + header + tile
    height = title_height + len(rows) * row_height + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), (17, 19, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        f"{sample_id} / GUARDED NEAR-BLACK PARAMETER COMPARISON",
        font=font(24),
        fill=(248, 248, 248),
    )
    for row_index, (row_label, row_color, panels) in enumerate(rows):
        y = title_height + row_index * (row_height + gap)
        draw.rectangle((0, y, width, y + row_label_height - 1), fill=row_color)
        draw.text((10, y + 4), row_label, font=font(16), fill=(248, 248, 248))
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


def aggregate_metrics(records: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    tp = sum(item["metrics"][key]["true_positive"] for item in records)
    fp = sum(item["metrics"][key]["false_positive"] for item in records)
    fn = sum(item["metrics"][key]["false_negative"] for item in records)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "iou": float(tp / max(1, tp + fp + fn)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }


def main() -> int:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    baseline = read_json(baseline_root / "manifest.json")
    by_id = {item["id"]: item for item in baseline["samples"]}
    sample_ids = args.sample_id or [item["id"] for item in baseline["samples"]]
    variant_roots = {key: output_root / key for key in PARAMETER_SETS}
    for root in variant_roots.values():
        for name in (
            "near_black_candidates",
            "removed_near_black_pixels",
            "total_removed_pixels",
            "building_only_colors",
            "building_binary",
            "provisional_ids",
            "provisional_previews",
        ):
            (root / name).mkdir(parents=True, exist_ok=True)
    comparison_root = output_root / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    row_colors = [(84, 67, 29), (44, 91, 76), (105, 53, 43)]
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        sample = by_id[sample_id]
        raw = load_rgb(project_path(sample["input"]["v9_2_raw_joint_mask"]))
        baseline_ids_2048 = np.asarray(
            Image.open(project_path(sample["outputs"]["source_instance_ids"])),
            dtype=np.uint16,
        )
        baseline_ids = np.asarray(
            Image.fromarray(baseline_ids_2048).resize(
                (1024, 1024), Image.Resampling.NEAREST
            ),
            dtype=np.uint16,
        )
        baseline_binary_2048 = load_rgb(
            project_path(sample["outputs"]["source_building_binary"])
        )[..., 0] > 0
        baseline_building = np.asarray(
            Image.fromarray(baseline_binary_2048.astype(np.uint8)).resize(
                (1024, 1024), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ).astype(bool)
        v7b = load_rgb(project_path(sample["outputs"]["v7b_normalized_mask"]))
        v7b_black = np.all(v7b == V7B_PALETTE[0], axis=-1)
        v7b_surface = ~v7b_black
        blue_filtered, blue_meta, _ = strong_v7b_black_gate(
            raw, baseline_building, v7b_black
        )
        gt = None
        sample_record: dict[str, Any] = {
            "id": sample_id,
            "strong_blue_filter_removed_pixels": int(blue_meta["removed_pixels"]),
            "variants": {},
            "metrics": (
                {
                    "current": binary_metrics(baseline_building, gt),
                    "strong_blue_filter": binary_metrics(blue_filtered, gt),
                }
                if gt is not None
                else {}
            ),
        }
        rows = []
        for (key, parameters), color in zip(PARAMETER_SETS.items(), row_colors):
            refined, metadata, arrays = guarded_near_black(
                raw,
                baseline_building,
                blue_filtered,
                v7b_surface,
                float(parameters["black_upper"]),
                int(parameters["v7b_dilation_px"]),
                float(parameters["max_internal_radius_px"]),
            )
            ids, assignment = retain_existing_ids(baseline_ids, refined)
            total_removed = baseline_building & ~refined
            root = variant_roots[key]
            paths = {
                "near_black_candidates": root / "near_black_candidates" / f"{sample_id}__near_black_candidates.png",
                "removed_near_black_pixels": root / "removed_near_black_pixels" / f"{sample_id}__removed_near_black_pixels.png",
                "total_removed_pixels": root / "total_removed_pixels" / f"{sample_id}__total_removed_pixels.png",
                "building_only_colors": root / "building_only_colors" / f"{sample_id}__building_only_colors.png",
                "building_binary": root / "building_binary" / f"{sample_id}__building_binary.png",
                "provisional_ids": root / "provisional_ids" / f"{sample_id}__provisional_ids.png",
                "provisional_previews": root / "provisional_previews" / f"{sample_id}__provisional_preview.png",
            }
            save_rgb(paths["near_black_candidates"], mask_rgb(arrays["candidate"]))
            save_rgb(paths["removed_near_black_pixels"], mask_rgb(arrays["remove"]))
            save_rgb(paths["total_removed_pixels"], mask_rgb(total_removed))
            save_rgb(paths["building_only_colors"], building_only_colors(raw, refined))
            save_rgb(paths["building_binary"], mask_rgb(refined))
            save_u16(paths["provisional_ids"], ids)
            save_rgb(paths["provisional_previews"], instance_preview(ids))
            metrics = binary_metrics(refined, gt) if gt is not None else None
            if metrics is not None:
                sample_record["metrics"][key] = metrics
            sample_record["variants"][key] = {
                "parameters": parameters,
                "filter": metadata,
                "assignment": assignment,
                "total_removed_pixels": int(np.count_nonzero(total_removed)),
                "outputs": {
                    name: str(path.relative_to(output_root))
                    for name, path in paths.items()
                },
            }
            rows.append(
                (
                    str(parameters["label"]),
                    color,
                    [
                        ("Near-black candidates", f"35 < RGB norm <= {parameters['black_upper']:.0f}", mask_rgb(arrays["candidate"])),
                        ("Removed near-black", f"{metadata['removed_near_black_pixels']} px", mask_rgb(arrays["remove"])),
                        ("Total removed", f"blue + dark = {np.count_nonzero(total_removed)} px", mask_rgb(total_removed)),
                        (
                            "Final building colors",
                            (
                                f"IoU {metrics['iou']:.4f}"
                                if metrics is not None
                                else "no building GT"
                            ),
                            building_only_colors(raw, refined),
                        ),
                        ("Final provisional IDs", f"{assignment['provisional_instance_count']} IDs", instance_preview(ids)),
                    ],
                )
            )
        comparison = comparison_root / f"{sample_id}__near_black_parameter_comparison.png"
        parameter_comparison_figure(sample_id, rows, comparison)
        sample_record["comparison"] = str(comparison.relative_to(output_root))
        comparisons.append(comparison)
        records.append(sample_record)

    overview = output_root / f"all_{len(records)}_near_black_parameter_comparisons.png"
    build_overview(comparisons, overview)
    aggregate = {}
    manifest = {
        "schema": "air2vectorcity.source_blue_dark_parameter_comparison.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "source-view parameter comparison; E00-E04 unchanged",
        "shared_preprocessing": "Variant B strong V7B black-gated blue filter",
        "shared_policy": {
            "temporary_closing": "3x3 on candidate grouping only",
            "stable_seed_protection": "any overlap with >=80px stable source seed",
            "id_assignment": "frozen E00 IDs; rejected pixels set to zero",
        },
        "parameter_sets": PARAMETER_SETS,
        "aggregate_metrics": aggregate,
        "overview": str(overview.relative_to(output_root)),
        "samples": records,
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
