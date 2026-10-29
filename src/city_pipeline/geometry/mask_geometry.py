#!/usr/bin/env python3
"""Shared source-mask filtering and comparison-image helpers."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.camera_geometry import read_json  # noqa: E402
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    V7B_PALETTE,
    instance_preview,
    local_rgb_gradient,
    mask_rgb,
    save_u16,
)


BASELINE_ROOT = Path("UNCONFIGURED_BASELINE_ROOT")
E04_ROOT = BASELINE_ROOT / "camera_aligned_25d_e04_aggressive"
DEFAULT_OUTPUT_ROOT = E04_ROOT / "source_blue_line_filter_ablation"

BLUE_MIN_B = 45
BLUE_MIN_DOMINANCE = 25
V7B_SURFACE_DILATION_PX = 1
GROUP_CLOSING_SIZE = 3
MAX_THIN_COMPONENT_RADIUS_PX = 3.0
STABLE_SEED_MIN_AREA_PX = 80
LOCAL_GRADIENT_THRESHOLD = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(
        path, format="PNG", optimize=True
    )


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def blue_like_candidate(rgb: np.ndarray, building: np.ndarray) -> np.ndarray:
    work = rgb.astype(np.int32)
    red, green, blue = work[..., 0], work[..., 1], work[..., 2]
    return (
        building
        & (blue >= BLUE_MIN_B)
        & ((blue - red) >= BLUE_MIN_DOMINANCE)
        & ((blue - green) >= BLUE_MIN_DOMINANCE)
    )


def stable_seed_mask(rgb: np.ndarray, building: np.ndarray) -> np.ndarray:
    flat = building & (local_rgb_gradient(rgb) < LOCAL_GRADIENT_THRESHOLD)
    labels, _ = ndimage.label(flat, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.reshape(-1))
    stable_labels = np.flatnonzero(sizes >= STABLE_SEED_MIN_AREA_PX)
    stable_labels = stable_labels[stable_labels > 0]
    return np.isin(labels, stable_labels)


def guarded_thin_blue_filter(
    rgb: np.ndarray,
    building: np.ndarray,
    v7b_surface: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Variant A: color + V7B neighborhood + thinness + stable-seed guard."""

    blue = blue_like_candidate(rgb, building)
    near_surface = ndimage.binary_dilation(
        v7b_surface,
        structure=np.ones((3, 3), dtype=bool),
        iterations=V7B_SURFACE_DILATION_PX,
    )
    supported = blue & near_surface
    grouped = ndimage.binary_closing(
        supported,
        structure=np.ones((GROUP_CLOSING_SIZE, GROUP_CLOSING_SIZE), dtype=bool),
    )
    labels, component_count = ndimage.label(
        grouped, structure=np.ones((3, 3), dtype=np.uint8)
    )
    stable = stable_seed_mask(rgb, building)
    remove = np.zeros(building.shape, dtype=bool)
    records: list[dict[str, Any]] = []
    for component_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        component = labels[slices] == component_id
        original = supported[slices] & component
        original_count = int(np.count_nonzero(original))
        if not original_count:
            continue
        radius = float(np.max(ndimage.distance_transform_edt(component)))
        stable_overlap = int(np.count_nonzero(original & stable[slices]))
        accepted = (
            radius <= MAX_THIN_COMPONENT_RADIUS_PX and stable_overlap == 0
        )
        if accepted:
            target = remove[slices]
            target |= original
        records.append(
            {
                "component_id": component_id,
                "original_candidate_pixels": original_count,
                "temporary_group_pixels": int(np.count_nonzero(component)),
                "maximum_internal_radius_px": radius,
                "stable_seed_overlap_pixels": stable_overlap,
                "removed": accepted,
            }
        )
    refined = building & ~remove
    metadata = {
        "method": "blue chroma + 1px V7B surface neighborhood + temporary 3x3 closing + <=3px radius + stable-seed guard",
        "blue_min_b": BLUE_MIN_B,
        "blue_min_channel_dominance": BLUE_MIN_DOMINANCE,
        "v7b_surface_dilation_px": V7B_SURFACE_DILATION_PX,
        "temporary_group_closing_size": GROUP_CLOSING_SIZE,
        "maximum_thin_component_internal_radius_px": MAX_THIN_COMPONENT_RADIUS_PX,
        "stable_seed_min_area_px": STABLE_SEED_MIN_AREA_PX,
        "blue_candidate_pixels": int(np.count_nonzero(blue)),
        "v7b_supported_candidate_pixels": int(np.count_nonzero(supported)),
        "removed_pixels": int(np.count_nonzero(remove)),
        "retained_building_pixels": int(np.count_nonzero(refined)),
        "temporary_component_count": int(component_count),
        "removed_component_count": int(sum(item["removed"] for item in records)),
        "protected_component_count": int(sum(not item["removed"] for item in records)),
        "components": records,
    }
    arrays = {
        "blue_candidate": blue,
        "v7b_support": near_surface,
        "supported_candidate": supported,
        "temporary_group": grouped,
        "stable_seed": stable,
        "remove": remove,
    }
    return refined, metadata, arrays


def strong_v7b_black_gate(
    rgb: np.ndarray,
    building: np.ndarray,
    v7b_black: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Variant B: any blue-like candidate outside V7B black is removed."""

    blue = blue_like_candidate(rgb, building)
    remove = blue & ~v7b_black
    refined = building & ~remove
    metadata = {
        "method": "remove every blue-like V9.2 building candidate outside exact V7B black",
        "blue_min_b": BLUE_MIN_B,
        "blue_min_channel_dominance": BLUE_MIN_DOMINANCE,
        "v7b_dilation_px": 0,
        "blue_candidate_pixels": int(np.count_nonzero(blue)),
        "removed_pixels": int(np.count_nonzero(remove)),
        "retained_building_pixels": int(np.count_nonzero(refined)),
    }
    arrays = {
        "blue_candidate": blue,
        "v7b_support": ~v7b_black,
        "remove": remove,
    }
    return refined, metadata, arrays


def retain_existing_ids(
    baseline_ids: np.ndarray,
    refined_building: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep current E00 identities and zero only pixels rejected by a filter."""

    ids = baseline_ids.copy()
    ids[~refined_building] = 0
    remaining = np.unique(ids)
    remaining = remaining[remaining > 0]
    return ids, {
        "assignment": (
            "frozen E00 provisional IDs on retained pixels; no seed recomputation, "
            "so foreground filtering is the only ablated variable"
        ),
        "provisional_instance_count": int(remaining.size),
        "remaining_instance_ids": [int(value) for value in remaining],
    }


def building_only_colors(rgb: np.ndarray, building: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    result[~building] = 0
    return result


def binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    tp = int(np.count_nonzero(prediction & reference))
    fp = int(np.count_nonzero(prediction & ~reference))
    fn = int(np.count_nonzero(~prediction & reference))
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "iou": float(tp / max(1, tp + fp + fn)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }




def fit_panel(array: np.ndarray, size: int = 330, nearest: bool = True) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    image.thumbnail(
        (size, size), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    )
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return panel


def comparison_figure(
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
        f"{sample_id} / SOURCE BLUE-LINE FILTER ABLATION",
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
                (x, header_y, x + tile - 1, header_y + header - 1), fill=(35, 38, 44)
            )
            draw.text((x + 8, header_y + 5), label, font=font(17), fill=(255, 255, 255))
            draw.text((x + 8, header_y + 36), subtitle, font=font(12), fill=(190, 197, 207))
            canvas.paste(fit_panel(array, tile, nearest=True), (x, header_y + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def build_overview(paths: list[Path], destination: Path) -> None:
    columns = 2
    tile_width = 1000
    tile_height = 820
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


def aggregate_metrics(records: list[dict[str, Any]], variant: str) -> dict[str, float | int]:
    tp = sum(item["metrics"][variant]["true_positive"] for item in records)
    fp = sum(item["metrics"][variant]["false_positive"] for item in records)
    fn = sum(item["metrics"][variant]["false_negative"] for item in records)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "iou": float(tp / max(1, tp + fp + fn)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }




