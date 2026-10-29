#!/usr/bin/env python3
"""Shared mask geometry and comparison-image helpers."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.camera_geometry import (
    comparison_panel,
    contact_sheet,
    font,
    height_preview,
    LANCZOS,
    lift_camera_forward_depth,
    load_rgb,
    load_u8,
    read_json,
    save_rgb,
    save_u8,
    zbuffer_world_xy,
)


DEFAULT_TEST_ROOT = Path("UNCONFIGURED_DEFAULT_TEST_ROOT")
DEFAULT_DATASET_ROOT = Path("UNCONFIGURED_DEFAULT_DATASET_ROOT")
DEFAULT_DEPTH_RUN = Path("UNCONFIGURED_DEFAULT_DEPTH_RUN")
DEFAULT_OUTPUT_ROOT = Path("UNCONFIGURED_DEFAULT_OUTPUT_ROOT")
DEFAULT_THRESHOLDS_M = (2.0, 3.0, 5.0, 8.0)
DEFAULT_SELECTED_THRESHOLD_M = 5.0
DEFAULT_MIN_COMPONENT_AREA_M2 = 8.0
BUILDING_SEMANTIC_ID = 6




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def threshold_tag(value: float) -> str:
    return f"{value:g}m".replace(".", "p")




def remove_small_components(
    mask: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, dict[str, int]]:
    labels, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    counts = np.bincount(labels.reshape(-1))
    keep = counts >= min_pixels
    if keep.size:
        keep[0] = False
    output = keep[labels]
    removed = mask & ~output
    return output, {
        "component_count_before": int(component_count),
        "component_count_kept": int(np.count_nonzero(keep)),
        "removed_pixel_count": int(np.count_nonzero(removed)),
    }


def binary_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, float | int | None]:
    pred = prediction & evaluation_mask
    ref = reference & evaluation_mask
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "evaluation_pixel_count": int(np.count_nonzero(evaluation_mask)),
        "prediction_pixel_count": int(np.count_nonzero(pred)),
        "reference_pixel_count": int(np.count_nonzero(ref)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / union if union else None,
    }


def full_grid_iou(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    intersection = int(np.count_nonzero(prediction & reference))
    union = int(np.count_nonzero(prediction | reference))
    return intersection / union if union else None


def red_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    red = np.zeros_like(output)
    red[..., 0] = 255.0
    output[mask] = (1.0 - alpha) * output[mask] + alpha * red[mask]
    return np.clip(output, 0, 255).astype(np.uint8)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def source_pixels_in_selected_bev(
    world_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds_xy_m: Iterable[float],
    output_size: tuple[int, int],
    selected_bev_mask: np.ndarray,
    ground_z_m: float,
    threshold_m: float,
) -> np.ndarray:
    """Map the selected BEV components back to their eligible source pixels.

    A source pixel is kept when its world XY lands inside a selected BEV cell
    and its own world Z is also above the physical height threshold.  This is
    intentionally a height-derived source-view candidate, not an instance mask.
    """

    width, height = output_size
    if selected_bev_mask.shape != (height, width):
        raise ValueError("Selected BEV mask and output size differ")
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_xy_m]
    cell_x = (max_x - min_x) / width
    cell_y = (max_y - min_y) / height
    world_x, world_y, world_z = world_xyz
    target_u = np.floor((world_x - min_x) / cell_x).astype(np.int32)
    target_v = np.floor((world_y - min_y) / cell_y).astype(np.int32)
    inside = (
        np.isfinite(world_x)
        & np.isfinite(world_y)
        & np.isfinite(world_z)
        & (target_u >= 0)
        & (target_u < width)
        & (target_v >= 0)
        & (target_v < height)
        & ((world_z - np.float32(ground_z_m)) >= np.float32(threshold_m))
    )
    result = np.zeros(world_x.shape, dtype=bool)
    result[inside] = selected_bev_mask[target_v[inside], target_u[inside]]
    return result


def two_row_comparison(
    title: str,
    perspective_row: list[tuple[str, str, np.ndarray, bool]],
    bev_row: list[tuple[str, str, np.ndarray, bool]],
    destination: Path,
    tile: int = 300,
    row_names: tuple[str, str] = ("PERSPECTIVE / INPUT VIEW", "ORTHOGRAPHIC / BEV VIEW"),
) -> None:
    """Write a two-row process panel with aligned semantic stages."""

    if len(perspective_row) != len(bev_row):
        raise ValueError("Perspective and BEV rows must have the same column count")
    columns = len(perspective_row)
    title_height = 48
    row_label_height = 25
    header = 62
    column_gap = 7
    row_gap = 10
    width = columns * tile + (columns - 1) * column_gap
    row_height = row_label_height + header + tile
    height = title_height + 2 * row_height + row_gap
    canvas = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 7), title, font=font(24), fill=(245, 245, 245))

    for row_index, (row_name, panels) in enumerate(
        ((row_names[0], perspective_row), (row_names[1], bev_row))
    ):
        row_y = title_height + row_index * (row_height + row_gap)
        draw.rectangle((0, row_y, width, row_y + row_label_height - 1), fill=(25, 31, 39))
        draw.text((10, row_y + 3), row_name, font=font(15), fill=(205, 219, 235))
        panel_y = row_y + row_label_height
        for column, (label, subtitle, array, nearest) in enumerate(panels):
            x = column * (tile + column_gap)
            draw.rectangle((x, panel_y, x + tile - 1, panel_y + header - 1), fill=(35, 38, 44))
            draw.text((x + 8, panel_y + 5), label, font=font(17), fill=(255, 255, 255))
            draw.text((x + 8, panel_y + 34), subtitle, font=font(12), fill=(190, 197, 207))
            resample = getattr(Image, "Resampling", Image).NEAREST if nearest else LANCZOS
            image = Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
                (tile, tile),
                resample,
            )
            canvas.paste(image, (x, panel_y + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def aggregate(values: Iterable[float | None]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }




