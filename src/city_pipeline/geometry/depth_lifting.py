#!/usr/bin/env python3
"""Build an instance-aware observed 2.5D city representation.

The experiment combines four independently produced inputs for the fixed
UECS_ST test set:

1. source-view RGB and exact camera geometry;
2. saved raw MoGe2-Aerial camera-forward depth;
3. Prompt V7B background surface classes for class-aware scale calibration;
4. Prompt V9.2 joint masks, from which only building instances are retained.

The output is deliberately an *observed* single-view 2.5D representation.  It
does not hallucinate occluded geometry or complete footprints.  Each observed
BEV cell carries world Z, RGB and a provisional building instance ID, and the
same data are exported as point clouds for inspection.

GT depth and structured building instance annotations are used only for audit
metrics.  They are never used to construct the predicted depth, instance IDs,
BEV maps or point clouds.
"""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT
from city_pipeline.geometry.priors import read_plane_priors

import argparse
import colorsys
import hashlib
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
    load_u8,
    read_json,
    save_rgb,
    save_u8,
    zbuffer_world_xy,
)


DEFAULT_TEST_ROOT = Path("UNCONFIGURED_DEFAULT_TEST_ROOT")
DEFAULT_DATASET_ROOT = Path("UNCONFIGURED_DEFAULT_DATASET_ROOT")
DEFAULT_DEPTH_RUN = Path("UNCONFIGURED_DEFAULT_DEPTH_RUN")
DEFAULT_V92_RUN = Path("UNCONFIGURED_DEFAULT_V92_RUN")
DEFAULT_V7B_RUN = Path("UNCONFIGURED_DEFAULT_V7B_RUN")
DEFAULT_OUTPUT_ROOT = Path("UNCONFIGURED_DEFAULT_OUTPUT_ROOT")

V7B_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 0, 0],
    ],
    dtype=np.uint8,
)
V7B_CLASS_NAMES = {
    0: "non_target_or_unknown",
    1: "ground_level_road",
    2: "water",
    3: "other_ground_hardscape",
    4: "elevated_road_or_bridge",
}

# Explicit calibration-plane priors supplied with the input dataset.
CALIBRATION_PLANE_Z_M = {
    1: 0.73,
    2: -3.50,
    3: 0.73,
}


CALIBRATION_CLASSES = (1, 2, 3)
EXCLUDED_CALIBRATION_CLASSES = (0, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build class-calibrated instance-aware 2.5D UECS_ST outputs."
    )
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--depth-run", type=Path, default=DEFAULT_DEPTH_RUN)
    parser.add_argument("--v9-2-run", type=Path, default=DEFAULT_V92_RUN)
    parser.add_argument("--v7b-run", type=Path, default=DEFAULT_V7B_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--anchor-erosion-radius-px",
        type=int,
        default=3,
        help="Erosion radius at the native source/depth resolution.",
    )
    parser.add_argument(
        "--max-balanced-ratios-per-class",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--instance-gradient-threshold-rgb",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--instance-seed-min-area-1024px",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--calibration-plane-policy",
        choices=("fixed-smallcity", "camera-ground"),
        default="fixed-smallcity",
        help=(
            "Use the historical SmallCity absolute plane priors or anchor all "
            "real-scene ground classes to each camera's local ground reference."
        ),
    )
    parser.add_argument("--allow-cached-smoke-recalibration", action="store_true", help="Explicit legacy numerical-fixture replay, never a fresh raw-model baseline")
    parser.add_argument("--priors-path", type=Path, help="Explicit fixed-plane priors JSON; defaults to dataset-root/priors.json")
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    return parser.parse_args()


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


def save_u16(path: Path, array: np.ndarray) -> None:
    if array.dtype != np.uint16:
        array = array.astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def rgb_overlay(rgb: np.ndarray, color: np.ndarray, mask: np.ndarray, alpha: float = 0.56) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    output[mask] = (
        (1.0 - alpha) * output[mask]
        + alpha * color.astype(np.float32)[mask]
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def discover_test_samples(test_root: Path) -> list[tuple[str, str, str, Path]]:
    samples: list[tuple[str, str, str, Path]] = []
    for path in sorted(test_root.glob("*__tilt_*.png")):
        patch_id, view_id = path.stem.rsplit("__", 1)
        samples.append((path.stem, patch_id, view_id, path))
    if not samples:
        raise ValueError(f"No *__tilt_*.png inputs found in {test_root}")
    return samples


def load_depth_samples(depth_run: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load a single depth manifest or all immediate profile manifests."""

    manifest_paths = [depth_run / "manifest.json"]
    if not manifest_paths[0].is_file():
        manifest_paths = sorted(depth_run.glob("*/manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No depth manifest found under {depth_run}")
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        base = manifest_path.parent if manifest.get("path_resolution") == "manifest_directory" or str(manifest.get("schema", "")).startswith("citycode.") else PROJECT_ROOT
        for original in manifest["samples"]:
            item = {**original, "outputs": dict(original["outputs"])}
            stage = original.get("depth_stage", manifest.get("depth_stage", "raw_metric_prediction"))
            if "already-calibrated" in str(original.get("scale_calibration", {}).get("method", "")).lower():
                stage = "already_calibrated_prediction"
            item["depth_stage"] = stage
            for key in ("depth_raw_zc_m", "depth_ground_scaled_zc_m"):
                if item["outputs"].get(key):
                    item["outputs"][key] = str((base / item["outputs"][key]).resolve())
            sample_id = f"{item['patch_id']}__{item['view_id']}"
            if sample_id in seen:
                raise ValueError(f"Duplicate depth sample across profiles: {sample_id}")
            seen.add(sample_id)
            samples.append(item)
    return samples, manifest_paths


def validate_depth_stage(samples, allow_cached_smoke_recalibration=False, sample_ids=None):
    """Already-calibrated fixtures are never silently relabeled as raw inference."""
    if sample_ids is not None:
        requested = set(sample_ids)
        samples = [item for item in samples if f"{item['patch_id']}__{item['view_id']}" in requested]
        found = {f"{item['patch_id']}__{item['view_id']}" for item in samples}
        if requested - found:
            raise ValueError(f"Missing selected depth samples: {sorted(requested - found)}")
    if any(item.get('depth_stage') == 'already_calibrated_prediction' for item in samples) and not allow_cached_smoke_recalibration:
        raise ValueError('Already-calibrated depth is not raw model output. Use cached E10 observations, or explicitly allow a legacy cached-smoke numerical replay.')


def v92_results_by_id(v92_run: Path) -> dict[str, Path]:
    manifest = read_json(v92_run / "manifest.json")
    result: dict[str, Path] = {}
    for item in manifest["samples"]:
        result[item["id"]] = v92_run / item["result"]
    return result


def v7b_results_by_id(v7b_run: Path) -> dict[str, Path]:
    manifest_path = v7b_run / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        return {
            item["id"]: v7b_run / item["result"]
            for item in manifest["samples"]
        }
    return {}


def normalize_v7b(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        actual_format = image.format
        raw = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if raw.shape != (1024, 1024, 3):
        raise ValueError(f"Unexpected V7B shape: {path}: {raw.shape}")
    difference = raw.astype(np.int32)[..., None, :] - V7B_PALETTE.astype(np.int32)
    squared = np.sum(difference * difference, axis=-1)
    class_ids = np.argmin(squared, axis=-1).astype(np.uint8)
    distance = np.sqrt(np.min(squared, axis=-1).astype(np.float64))
    normalized = V7B_PALETTE[class_ids]
    return class_ids, normalized, {
        "raw_actual_format": actual_format,
        "raw_unique_rgb_count": int(np.unique(raw.reshape(-1, 3), axis=0).shape[0]),
        "nearest_palette_distance_rgb": {
            "p50": float(np.percentile(distance, 50)),
            "p90": float(np.percentile(distance, 90)),
            "p95": float(np.percentile(distance, 95)),
            "p99": float(np.percentile(distance, 99)),
            "max": float(np.max(distance)),
        },
        "class_fraction": {
            V7B_CLASS_NAMES[index]: float(np.mean(class_ids == index))
            for index in range(len(V7B_PALETTE))
        },
    }


def resize_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(array).resize(size, Image.Resampling.NEAREST)
    )


def expected_plane_depth_zc(camera: dict[str, Any], plane_z_m: float) -> np.ndarray:
    width, height = [int(value) for value in camera["image_size_px"]]
    intrinsics = camera["intrinsics"]
    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    cx = float(intrinsics["cx_px"])
    cy = float(intrinsics["cy_px"])
    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    x = (u - cx) / fx
    y = (v - cy) / fy
    transform = np.asarray(camera["camera_to_world_4x4_m"], dtype=np.float64)
    world_ray_z = (
        transform[2, 0] * x[None, :]
        + transform[2, 1] * y[:, None]
        + transform[2, 2]
    )
    numerator = float(plane_z_m) - transform[2, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = numerator / world_ray_z
    depth[(depth <= 0.0) | ~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)


def anchor_core_masks(
    class_ids: np.ndarray,
    erosion_radius_px: int,
) -> dict[int, np.ndarray]:
    if erosion_radius_px < 0:
        raise ValueError("Anchor erosion radius cannot be negative")
    size = 2 * erosion_radius_px + 1
    structure = np.ones((size, size), dtype=bool)
    return {
        class_id: ndimage.binary_erosion(
            class_ids == class_id,
            structure=structure,
            border_value=0,
        )
        for class_id in CALIBRATION_CLASSES
    }


def class_aware_scale(
    raw_depth_zc: np.ndarray,
    camera: dict[str, Any],
    cores: dict[int, np.ndarray],
    max_per_class: int,
    plane_z_m: dict[int, float],
) -> tuple[float, dict[str, Any]]:
    balanced: list[np.ndarray] = []
    class_records: dict[str, Any] = {}
    active_classes = tuple(sorted(plane_z_m))
    for class_id in active_classes:
        expected = expected_plane_depth_zc(camera, plane_z_m[class_id])
        core = cores[class_id]
        valid = (
            core
            & np.isfinite(raw_depth_zc)
            & (raw_depth_zc > 0.0)
            & np.isfinite(expected)
            & (expected > 0.0)
        )
        ratios = expected[valid] / raw_depth_zc[valid]
        if ratios.size == 0:
            class_records[V7B_CLASS_NAMES[class_id]] = {
                "plane_z_m": plane_z_m[class_id],
                "raw_ratio_count": 0,
                "used": False,
            }
            continue
        median = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - median)))
        threshold = max(3.0 * mad, 0.01)
        robust = ratios[np.abs(ratios - median) <= threshold]
        if robust.size < 4096:
            class_records[V7B_CLASS_NAMES[class_id]] = {
                "plane_z_m": plane_z_m[class_id],
                "raw_ratio_count": int(ratios.size),
                "robust_ratio_count": int(robust.size),
                "used": False,
                "reason": "fewer than 4096 robust anchor pixels",
            }
            continue
        take = min(int(max_per_class), int(robust.size))
        indices = np.linspace(0, robust.size - 1, take, dtype=np.int64)
        sampled = robust[indices]
        balanced.append(sampled)
        class_records[V7B_CLASS_NAMES[class_id]] = {
            "plane_z_m": plane_z_m[class_id],
            "raw_ratio_count": int(ratios.size),
            "robust_ratio_count": int(robust.size),
            "balanced_ratio_count": int(sampled.size),
            "ratio_median": median,
            "ratio_mad": mad,
            "ratio_p05": float(np.percentile(ratios, 5)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "robust_keep_fraction": float(robust.size / ratios.size),
            "used": True,
        }
    if not balanced:
        raise ValueError("No V7B calibration class has enough robust anchor pixels")
    all_balanced = np.concatenate(balanced)
    scale = float(np.median(all_balanced))
    return scale, {
        "method": "class-balanced median of robust expected_plane_zc/raw_depth ratios",
        "class_plane_z_m": {
            V7B_CLASS_NAMES[class_id]: plane_z_m[class_id]
            for class_id in active_classes
        },
        "excluded_classes": [
            V7B_CLASS_NAMES[index]
            for index in sorted(set(V7B_CLASS_NAMES) - set(active_classes))
        ],
        "scale": scale,
        "balanced_ratio_count": int(all_balanced.size),
        "balanced_ratio_mad": float(np.median(np.abs(all_balanced - scale))),
        "classes": class_records,
        "uses_gt_depth": False,
        "uses_v7b_predictions": True,
        "uses_known_camera_and_asset_plane_priors": True,
    }


def local_rgb_gradient(rgb: np.ndarray) -> np.ndarray:
    work = rgb.astype(np.int32)
    gradient = np.zeros(rgb.shape[:2], dtype=np.float32)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        shifted = np.roll(work, (dy, dx), axis=(0, 1))
        difference = work - shifted
        gradient = np.maximum(
            gradient,
            np.sqrt(np.sum(difference * difference, axis=-1)),
        )
    gradient[[0, -1], :] = 441.7
    gradient[:, [0, -1]] = 441.7
    return gradient


def provisional_instance_ids(
    v92_rgb: np.ndarray,
    gradient_threshold: float,
    seed_min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Convert the generated RGB mask to provisional per-image integer IDs.

    This is a minimal normalization, not a geometric correction.  Black-like
    and road-blue-like pixels are excluded.  Large locally flat colour regions
    become instance seeds; every remaining building-coloured pixel is assigned
    to the nearest seed.  No building foreground pixel is deleted.
    """

    if v92_rgb.shape != (1024, 1024, 3):
        raise ValueError(f"Unexpected V9.2 mask shape: {v92_rgb.shape}")
    work = v92_rgb.astype(np.int32)
    black_distance = np.sqrt(np.sum(work * work, axis=-1))
    blue_delta = work - np.asarray([0, 0, 255], dtype=np.int32)
    blue_distance = np.sqrt(np.sum(blue_delta * blue_delta, axis=-1))
    background = black_distance <= 35.0
    road = (
        (blue_distance <= 60.0)
        & (work[..., 2] > work[..., 0] + 70)
        & (work[..., 2] > work[..., 1] + 70)
    )
    building = ~(background | road)
    gradient = local_rgb_gradient(v92_rgb)
    core = building & (gradient < float(gradient_threshold))
    core_labels, raw_seed_count = ndimage.label(
        core,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    core_counts = np.bincount(core_labels.reshape(-1))
    stable_core_ids = np.flatnonzero(core_counts >= int(seed_min_area))
    stable_core_ids = stable_core_ids[stable_core_ids > 0]

    marker = np.zeros_like(core_labels, dtype=np.int32)
    if stable_core_ids.size:
        selected = np.isin(core_labels, stable_core_ids)
        marker[selected] = np.searchsorted(stable_core_ids, core_labels[selected]) + 1
    next_marker = int(np.max(marker)) + 1

    # Guarantee at least one marker per disconnected building foreground island.
    building_components, building_component_count = ndimage.label(
        building,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    fallback_marker_count = 0
    for component_id in range(1, building_component_count + 1):
        component = building_components == component_id
        if np.any(marker[component] > 0):
            continue
        distance = ndimage.distance_transform_edt(component)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        marker[y, x] = next_marker
        next_marker += 1
        fallback_marker_count += 1

    if not np.any(marker > 0):
        raise ValueError("V9.2 result contains no provisional building instance seed")
    _, nearest = ndimage.distance_transform_edt(marker == 0, return_indices=True)
    ids = marker[nearest[0], nearest[1]].astype(np.uint16)
    ids[~building] = 0

    binary_labels, binary_count = ndimage.label(
        building,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    binary_counts = np.bincount(binary_labels.reshape(-1))[1:]
    seed_sizes = core_counts[1:]
    return ids, building, road, {
        "method": "flat-colour spatial seeds plus nearest-seed assignment",
        "black_rgb_distance_threshold": 35.0,
        "road_blue_rgb_distance_threshold": 60.0,
        "road_blue_channel_margin": 70,
        "local_rgb_gradient_threshold": float(gradient_threshold),
        "seed_min_area_at_1024px": int(seed_min_area),
        "raw_unique_rgb_count": int(
            np.unique(v92_rgb.reshape(-1, 3), axis=0).shape[0]
        ),
        "building_foreground_fraction": float(np.mean(building)),
        "road_fraction": float(np.mean(road)),
        "background_fraction": float(np.mean(background)),
        "building_binary_component_count": int(binary_count),
        "building_binary_component_area_1024px": distribution(binary_counts),
        "raw_flat_colour_seed_component_count": int(raw_seed_count),
        "raw_flat_colour_seed_area_1024px": distribution(seed_sizes),
        "stable_seed_count": int(stable_core_ids.size),
        "fallback_marker_count": int(fallback_marker_count),
        "provisional_instance_count": int(np.max(ids)),
        "building_foreground_pixels_removed": 0,
        "warning": (
            "IDs are provisional per-view IDs derived from generated colours; "
            "they are not cross-view identities and do not correct model-level merge/split errors."
        ),
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "count_lt_16": int(np.count_nonzero(values < 16)),
        "count_lt_64": int(np.count_nonzero(values < 64)),
        "count_lt_256": int(np.count_nonzero(values < 256)),
    }


def instance_palette(max_id: int) -> np.ndarray:
    palette = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for instance_id in range(1, max_id + 1):
        hue = (instance_id * 0.6180339887498949) % 1.0
        palette[instance_id] = (
            np.asarray(colorsys.hsv_to_rgb(hue, 0.72, 0.95)) * 255.0
        ).astype(np.uint8)
    return palette


def instance_preview(ids: np.ndarray) -> np.ndarray:
    return instance_palette(int(np.max(ids)))[ids]


def class_core_preview(cores: dict[int, np.ndarray]) -> np.ndarray:
    preview = np.zeros((*next(iter(cores.values())).shape, 3), dtype=np.uint8)
    for class_id, mask in cores.items():
        preview[mask] = V7B_PALETTE[class_id]
    return preview


def finite_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }




def calibration_planes(
    camera: dict[str, Any], policy: str, priors: dict[int, float] | None = None
) -> dict[int, float]:
    if policy == "fixed-smallcity":
        return {key: float(value) for key, value in (priors if priors is not None else CALIBRATION_PLANE_Z_M).items()}
    if policy == "camera-ground":
        ground = float(camera["ground_reference_z_m"])
        # Water can sit below the patch-local ground reference, and elevated
        # decks sit above it.  Real-scene scale uses road and hardscape only.
        return {class_id: ground for class_id in (1, 3)}
    raise ValueError(f"Unsupported calibration plane policy: {policy}")


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    max_points: int,
) -> dict[str, Any]:
    count = int(xyz.shape[0])
    if count == 0:
        raise ValueError("Cannot write an empty point cloud")
    take = min(count, int(max_points))
    indices = np.linspace(0, count - 1, take, dtype=np.int64)
    points = np.empty(
        take,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("instance_id", "<u2"),
            ]
        ),
    )
    points["x"] = xyz[indices, 0]
    points["y"] = xyz[indices, 1]
    points["z"] = xyz[indices, 2]
    points["red"] = rgb[indices, 0]
    points["green"] = rgb[indices, 1]
    points["blue"] = rgb[indices, 2]
    points["instance_id"] = instance_ids[indices]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {take}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property ushort instance_id\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        points.tofile(stream)
    return {
        "source_point_count": count,
        "written_point_count": take,
        "deterministic_uniform_subsample": take < count,
    }


def bev_point_cloud(
    surface_z: np.ndarray,
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    valid: np.ndarray,
    bounds_xy_m: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = surface_z.shape
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_xy_m]
    cell_x = (max_x - min_x) / width
    cell_y = (max_y - min_y) / height
    selected = valid & np.isfinite(surface_z) & (instance_ids > 0)
    v, u = np.nonzero(selected)
    xyz = np.stack(
        (
            min_x + (u.astype(np.float32) + 0.5) * cell_x,
            min_y + (v.astype(np.float32) + 0.5) * cell_y,
            surface_z[selected],
        ),
        axis=1,
    ).astype(np.float32)
    return xyz, rgb[selected].astype(np.uint8), instance_ids[selected].astype(np.uint16)


def per_instance_statistics(
    source_ids: np.ndarray,
    bev_ids: np.ndarray,
    surface_z: np.ndarray,
    ground_reference_z_m: float,
    bounds_xy_m: Iterable[float],
) -> list[dict[str, Any]]:
    height, width = bev_ids.shape
    min_x, min_y, max_x, max_y = [float(value) for value in bounds_xy_m]
    cell_x = (max_x - min_x) / width
    cell_y = (max_y - min_y) / height
    pixel_area = cell_x * cell_y
    source_counts = np.bincount(source_ids.reshape(-1))
    records: list[dict[str, Any]] = []
    for instance_id in np.unique(bev_ids):
        if instance_id == 0:
            continue
        mask = (bev_ids == instance_id) & np.isfinite(surface_z)
        v, u = np.nonzero(mask)
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
                "observed_bev_area_m2": float(z.size * pixel_area),
                "observed_world_z_m": finite_distribution(z),
                "observed_height_above_reference_ground_m": finite_distribution(
                    z - np.float32(ground_reference_z_m)
                ),
                "observed_xy_centroid_m": [
                    float(min_x + (np.mean(u) + 0.5) * cell_x),
                    float(min_y + (np.mean(v) + 0.5) * cell_y),
                ],
                "observed_xy_bbox_m": [
                    float(min_x + (np.min(u) + 0.5) * cell_x),
                    float(min_y + (np.min(v) + 0.5) * cell_y),
                    float(min_x + (np.max(u) + 0.5) * cell_x),
                    float(min_y + (np.max(v) + 0.5) * cell_y),
                ],
            }
        )
    return records


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
    test_root = args.test_root.resolve()
    dataset_root = args.dataset_root.resolve()
    depth_run = args.depth_run.resolve()
    v92_run = args.v9_2_run.resolve()
    v7b_run = args.v7b_run.resolve()
    output_root = args.output_root.resolve()
    priors_path = (args.priors_path or dataset_root / "priors.json").resolve() if args.calibration_plane_policy == "fixed-smallcity" else None
    configured_priors = read_plane_priors(priors_path) if priors_path else None

    depth_samples, depth_manifest_paths = load_depth_samples(depth_run)
    depth_by_id = {
        f"{item['patch_id']}__{item['view_id']}": item
        for item in depth_samples
    }
    v92_by_id = v92_results_by_id(v92_run)
    v7b_by_id = v7b_results_by_id(v7b_run)
    samples = discover_test_samples(test_root)
    validate_depth_stage(depth_samples, args.allow_cached_smoke_recalibration,
                         sample_ids=[item[0] for item in samples])

    directory_names = (
        "v7b_normalized_masks_1024",
        "v7b_class_ids_native",
        "ground_anchor_core_masks_native",
        "v92_building_binary_masks_native",
        "v92_building_only_raw_colors_native",
        "source_instance_ids_native",
        "source_instance_previews_native",
        "source_instance_overlays_native",
        "depth_class_aware_zc_m",
        "depth_previews",
        "bev_rgb",
        "bev_validity",
        "bev_surface_z_m",
        "bev_height_above_reference_ground_m",
        "bev_height_previews",
        "bev_instance_ids",
        "bev_instance_previews",
        "bev_instance_overlays",
        "point_cloud_npz",
        "point_cloud_ply_rgb",
        "point_cloud_ply_instance",
        "per_instance_statistics",
        "input_and_process_audit",
    )
    directories = {name: output_root / name for name in directory_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    audit_paths: list[Path] = []
    for sample_index, (sample_id, patch_id, view_id, test_rgb_path) in enumerate(
        samples, start=1
    ):
        print(f"[{sample_index}/{len(samples)}] {sample_id}", flush=True)
        depth_item = depth_by_id.get(sample_id)
        if depth_item is None:
            raise KeyError(f"Saved AerialMetric output is missing: {sample_id}")
        v92_path = v92_by_id.get(sample_id)
        if v92_path is None:
            raise KeyError(f"V9.2 output is missing: {sample_id}")
        v7b_path = v7b_by_id.get(sample_id, v7b_run / f"{sample_id}.jpg")
        patch_dir = dataset_root / "patches" / patch_id
        view_dir = patch_dir / "views" / view_id
        camera_path = view_dir / "camera.json"
        source_rgb_path = view_dir / "rgb.png"
        strict_valid_path = view_dir / str(
            read_json(camera_path).get("strict_valid_mask_path", "strict_valid_mask.png")
        )
        raw_depth_path = PROJECT_ROOT / depth_item["outputs"]["depth_raw_zc_m"]
        old_value = depth_item["outputs"].get("depth_ground_scaled_zc_m")
        old_depth_path = (PROJECT_ROOT / old_value) if old_value else None
        if old_depth_path is not None and not old_depth_path.is_file():
            old_depth_path = None
        for path in (
            test_rgb_path,
            source_rgb_path,
            camera_path,
            raw_depth_path,
            *([old_depth_path] if old_depth_path is not None else []),
            v92_path,
            v7b_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(test_rgb_path) != sha256(source_rgb_path):
            raise ValueError(f"Test RGB and released RGB differ: {sample_id}")

        camera = read_json(camera_path)
        sample_plane_z_m = calibration_planes(
            camera, args.calibration_plane_policy, configured_priors
        )
        source_rgb = load_rgb(source_rgb_path)
        source_height, source_width = source_rgb.shape[:2]
        source_size = (source_width, source_height)
        strict_required = bool(
            camera.get("strict_valid_mask_path")
            or read_json(patch_dir / "patch.json").get("strict_roi", {}).get(
                "required", False
            )
        )
        if strict_required and not strict_valid_path.is_file():
            raise FileNotFoundError(strict_valid_path)
        strict_valid = (
            load_u8(strict_valid_path) > 0
            if strict_valid_path.is_file()
            else np.ones(source_rgb.shape[:2], dtype=bool)
        )
        if strict_valid.shape != source_rgb.shape[:2]:
            raise ValueError(
                f"Strict-valid/RGB shape mismatch: {sample_id}: "
                f"{strict_valid.shape} versus {source_rgb.shape[:2]}"
            )
        raw_depth = np.load(raw_depth_path).astype(np.float32, copy=False)
        old_depth = np.load(old_depth_path).astype(np.float32, copy=False) if old_depth_path is not None else raw_depth.copy()
        if raw_depth.shape != source_rgb.shape[:2] or old_depth.shape != raw_depth.shape:
            raise ValueError(f"RGB/depth shape mismatch: {sample_id}")
        raw_depth = raw_depth.copy()
        old_depth = old_depth.copy()
        raw_depth[~strict_valid] = np.nan
        old_depth[~strict_valid] = np.nan

        # V7B: normalize JPEG output to its five legal semantic colours.
        v7b_ids_1024, v7b_normalized_1024, v7b_metadata = normalize_v7b(v7b_path)
        v7b_ids_2048 = resize_nearest(v7b_ids_1024, source_size).astype(np.uint8)
        v7b_ids_2048[~strict_valid] = 0
        cores = anchor_core_masks(v7b_ids_2048, args.anchor_erosion_radius_px)
        core_preview = class_core_preview(cores)
        v7b_normalized_path = (
            directories["v7b_normalized_masks_1024"]
            / f"{sample_id}__v7b_normalized.png"
        )
        v7b_ids_path = (
            directories["v7b_class_ids_native"]
            / f"{sample_id}__v7b_class_ids.png"
        )
        anchor_path = (
            directories["ground_anchor_core_masks_native"]
            / f"{sample_id}__ground_anchor_cores.png"
        )
        save_rgb(v7b_normalized_path, v7b_normalized_1024)
        save_u8(v7b_ids_path, v7b_ids_2048)
        save_rgb(anchor_path, core_preview)

        # V9.2: keep only building-coloured regions and derive provisional IDs.
        v92_rgb_1024 = np.asarray(Image.open(v92_path).convert("RGB"), dtype=np.uint8)
        ids_1024, building_1024, road_1024, instance_metadata = provisional_instance_ids(
            v92_rgb_1024,
            args.instance_gradient_threshold_rgb,
            args.instance_seed_min_area_1024px,
        )
        ids_2048 = resize_nearest(ids_1024, source_size).astype(np.uint16)
        ids_2048[~strict_valid] = 0
        building_2048 = ids_2048 > 0
        preview_1024 = instance_preview(ids_1024)
        preview_2048 = resize_nearest(preview_1024, source_size).astype(np.uint8)
        preview_2048[~strict_valid] = 0
        v92_raw_2048 = resize_nearest(v92_rgb_1024, source_size).astype(np.uint8)
        v92_raw_2048[~strict_valid] = 0
        building_only_raw = np.zeros_like(v92_raw_2048)
        building_only_raw[building_2048] = v92_raw_2048[building_2048]
        source_overlay = rgb_overlay(source_rgb, preview_2048, building_2048)
        building_binary_path = (
            directories["v92_building_binary_masks_native"]
            / f"{sample_id}__building_binary.png"
        )
        building_raw_path = (
            directories["v92_building_only_raw_colors_native"]
            / f"{sample_id}__building_only_raw_colors.png"
        )
        source_ids_path = (
            directories["source_instance_ids_native"]
            / f"{sample_id}__source_instance_ids.png"
        )
        source_preview_path = (
            directories["source_instance_previews_native"]
            / f"{sample_id}__source_instance_preview.png"
        )
        source_overlay_path = (
            directories["source_instance_overlays_native"]
            / f"{sample_id}__source_instance_overlay.png"
        )
        save_u8(building_binary_path, building_2048.astype(np.uint8) * 255)
        save_rgb(building_raw_path, building_only_raw)
        save_u16(source_ids_path, ids_2048)
        save_rgb(source_preview_path, preview_2048)
        save_rgb(source_overlay_path, source_overlay)

        # Class-aware metric scale from V7B cores and known UE asset plane priors.
        new_scale, scale_metadata = class_aware_scale(
            raw_depth,
            camera,
            cores,
            args.max_balanced_ratios_per_class,
            sample_plane_z_m,
        )
        calibrated_depth = (raw_depth * np.float32(new_scale)).astype(np.float32)
        calibrated_depth_path = (
            directories["depth_class_aware_zc_m"]
            / f"{sample_id}__depth_class_aware_zc_m.npy"
        )
        np.save(calibrated_depth_path, calibrated_depth)
        source_valid = np.isfinite(raw_depth) & (raw_depth > 0.0)
        raw_depth_preview = height_preview(raw_depth, source_valid)
        old_depth_preview = height_preview(old_depth, source_valid)
        calibrated_depth_preview = height_preview(calibrated_depth, source_valid)
        depth_preview_path = (
            directories["depth_previews"]
            / f"{sample_id}__raw_old_and_class_aware_depth.png"
        )
        two_row_comparison(
            f"{sample_id} / DEPTH SCALE AUDIT",
            [
                ("D1. Raw metric", "MoGe2-Aerial z_c", raw_depth_preview, False),
                ("D2. Old scale", f"scale={depth_item.get('scale_calibration', {}).get('scale', 1.0):.6f}" if old_depth_path is not None else "raw input; no prior calibrated diagnostic", old_depth_preview, False),
                ("D3. Class-aware", f"scale={new_scale:.6f}", calibrated_depth_preview, False),
            ],
            [
                ("A1. V7B normalized", "five legal classes", resize_nearest(v7b_normalized_1024, source_size), True),
                ("A2. Anchor cores", "road / water / hardscape", core_preview, True),
                ("A3. Excluded", "black + elevated are not anchors", mask_rgb(np.isin(v7b_ids_2048, EXCLUDED_CALIBRATION_CLASSES)), True),
            ],
            depth_preview_path,
            tile=360,
        )

        # Lift calibrated depth and carry instance identity through the same z-buffer.
        world_xyz, lift_metadata = lift_camera_forward_depth(calibrated_depth, camera)
        recovered, bev_valid, surface_z, rasterization = zbuffer_world_xy(
            {
                "rgb": source_rgb,
                "instance_id": ids_2048,
                "instance_preview": preview_2048,
            },
            world_xyz,
            camera["target_patch_bounds_xy_m"],
            tuple(int(value) for value in camera["image_size_px"]),
        )
        bev_ids = recovered["instance_id"].astype(np.uint16)
        bev_building = bev_valid & (bev_ids > 0)
        ground_reference_z_m = float(camera["ground_reference_z_m"])
        height_agl = surface_z - np.float32(ground_reference_z_m)
        source_agl = world_xyz[2] - np.float32(ground_reference_z_m)
        source_building_height_preview = height_preview(
            source_agl,
            source_valid & building_2048,
        )
        source_building_height_preview[~building_2048] = 0
        bev_height_preview = height_preview(height_agl, bev_building)
        bev_height_preview[~bev_building] = 0
        bev_instance_overlay = rgb_overlay(
            recovered["rgb"],
            recovered["instance_preview"],
            bev_building,
        )

        bev_rgb_path = directories["bev_rgb"] / f"{sample_id}__bev_rgb.png"
        bev_validity_path = directories["bev_validity"] / f"{sample_id}__bev_validity.png"
        surface_z_path = directories["bev_surface_z_m"] / f"{sample_id}__bev_surface_z_m.npy"
        height_agl_path = (
            directories["bev_height_above_reference_ground_m"]
            / f"{sample_id}__bev_height_above_reference_ground_m.npy"
        )
        bev_height_preview_path = (
            directories["bev_height_previews"]
            / f"{sample_id}__bev_building_height_preview.png"
        )
        bev_ids_path = directories["bev_instance_ids"] / f"{sample_id}__bev_instance_ids.png"
        bev_instance_preview_path = (
            directories["bev_instance_previews"]
            / f"{sample_id}__bev_instance_preview.png"
        )
        bev_overlay_path = (
            directories["bev_instance_overlays"]
            / f"{sample_id}__bev_instance_overlay.png"
        )
        save_rgb(bev_rgb_path, recovered["rgb"])
        save_u8(bev_validity_path, bev_valid.astype(np.uint8) * 255)
        np.save(surface_z_path, surface_z.astype(np.float32))
        np.save(height_agl_path, height_agl.astype(np.float32))
        save_rgb(bev_height_preview_path, bev_height_preview)
        save_u16(bev_ids_path, bev_ids)
        save_rgb(bev_instance_preview_path, recovered["instance_preview"])
        save_rgb(bev_overlay_path, bev_instance_overlay)

        # Dense observed building point data plus two lightweight PLY visualizations.
        xyz, point_rgb, point_ids = bev_point_cloud(
            surface_z,
            recovered["rgb"],
            bev_ids,
            bev_valid,
            camera["target_patch_bounds_xy_m"],
        )
        point_npz_path = directories["point_cloud_npz"] / f"{sample_id}__building_points.npz"
        np.savez_compressed(
            point_npz_path,
            xyz_m=xyz,
            rgb=point_rgb,
            instance_id=point_ids,
        )
        point_instance_rgb = instance_palette(int(np.max(point_ids)))[point_ids]
        ply_rgb_path = directories["point_cloud_ply_rgb"] / f"{sample_id}__building_rgb.ply"
        ply_instance_path = (
            directories["point_cloud_ply_instance"]
            / f"{sample_id}__building_instance_colors.ply"
        )
        ply_rgb_metadata = write_binary_ply(
            ply_rgb_path,
            xyz,
            point_rgb,
            point_ids,
            args.max_ply_points,
        )
        ply_instance_metadata = write_binary_ply(
            ply_instance_path,
            xyz,
            point_instance_rgb,
            point_ids,
            args.max_ply_points,
        )

        instance_stats = per_instance_statistics(
            ids_2048,
            bev_ids,
            surface_z,
            ground_reference_z_m,
            camera["target_patch_bounds_xy_m"],
        )
        instance_stats_path = (
            directories["per_instance_statistics"]
            / f"{sample_id}__per_instance_statistics.json"
        )
        instance_stats_path.write_text(
            json.dumps(instance_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Legacy manifest fields remain empty for downstream schema compatibility.
        old_metrics = new_metrics = {"status": "not_in_release", "abs_rel": None}
        class_height_audit = {"status": "not_in_release"}

        audit_path = (
            directories["input_and_process_audit"]
            / f"{sample_id}__instance_aware_25d_audit.png"
        )
        two_row_comparison(
            f"{sample_id} / INPUTS AND MINIMAL NORMALIZATION",
            [
                ("S1. Input RGB", f"{source_width} source view", source_rgb, False),
                ("S2. V9.2 raw", "joint generated mask", v92_raw_2048, True),
                ("S3. Building only", "black + road removed", building_only_raw, True),
                ("S4. Provisional IDs", f"{int(np.max(ids_2048))} per-view instances", preview_2048, True),
                ("S5. V7B normalized", "black / road / water / hard / elevated", resize_nearest(v7b_normalized_1024, source_size), True),
                ("S6. Anchor cores", "class interiors after erosion", core_preview, True),
            ],
            [
                ("G1. Raw depth", "uncalibrated z_c", raw_depth_preview, False),
                ("G2. Class-aware depth", f"scale={new_scale:.6f}", calibrated_depth_preview, False),
                ("G3. Source building Z", "world Z - reference ground", source_building_height_preview, False),
                ("G4. BEV RGB", f"coverage={np.mean(bev_valid):.1%}", recovered["rgb"], False),
                ("G5. BEV instances", "same IDs carried by z-buffer", recovered["instance_preview"], True),
                (
                    "G6. BEV building height",
                    f"Z - {ground_reference_z_m:.2f} m; nonbuilding black",
                    bev_height_preview,
                    False,
                ),
            ],
            audit_path,
            tile=320,
        )
        audit_paths.append(audit_path)

        record = {
            "id": sample_id,
            "patch_id": patch_id,
            "view_id": view_id,
            "input": {
                "rgb": relative(test_rgb_path),
                "camera": relative(camera_path),
                "aerialmetric_raw_depth_zc_m": relative(raw_depth_path),
                "aerialmetric_old_calibrated_depth_zc_m": relative(old_depth_path) if old_depth_path is not None else None,
                "old_calibrated_depth_is_diagnostic_only": True,
                "depth_input_stage": depth_item["depth_stage"],
                "v9_2_raw_joint_mask": relative(v92_path),
                "v7b_raw_background_mask": relative(v7b_path),
                "strict_valid_mask": (
                    relative(strict_valid_path) if strict_valid_path.is_file() else None
                ),
            },
            "strict_roi": {
                "required": strict_required,
                "applied": strict_valid_path.is_file(),
                "valid_pixel_count": int(np.count_nonzero(strict_valid)),
                "valid_fraction": float(np.mean(strict_valid)),
            },
            "v7b_normalization": v7b_metadata,
            "v7b_anchor_core_pixel_count": {
                V7B_CLASS_NAMES[class_id]: int(np.count_nonzero(cores[class_id]))
                for class_id in CALIBRATION_CLASSES
            },
            "instance_normalization": instance_metadata,
            "scale_calibration": {
                "old": depth_item["scale_calibration"],
                "class_aware": scale_metadata,
                "scale_change_fraction_vs_old": float(
                    new_scale / float(depth_item["scale_calibration"]["scale"]) - 1.0
                ),
            },
            "evaluation_only": {
                "old_calibrated_depth_all": old_metrics,
                "class_aware_calibrated_depth_all": new_metrics,
                "v7b_class_world_z_audit": class_height_audit,
            },
            "lifting": lift_metadata,
            "bev_rasterization": rasterization,
            "observed_building_point_count": int(xyz.shape[0]),
            "observed_building_instance_count": int(len(instance_stats)),
            "observed_building_height_distribution_m": finite_distribution(
                height_agl[bev_building]
            ),
            "outputs": {
                "v7b_normalized_mask": relative(v7b_normalized_path),
                "v7b_class_ids": relative(v7b_ids_path),
                "ground_anchor_cores": relative(anchor_path),
                "source_building_binary": relative(building_binary_path),
                "source_building_only_raw_colors": relative(building_raw_path),
                "source_instance_ids": relative(source_ids_path),
                "source_instance_preview": relative(source_preview_path),
                "source_instance_overlay": relative(source_overlay_path),
                "class_aware_depth_zc_m": relative(calibrated_depth_path),
                "depth_audit": relative(depth_preview_path),
                "bev_rgb": relative(bev_rgb_path),
                "bev_validity": relative(bev_validity_path),
                "bev_surface_z_m": relative(surface_z_path),
                "bev_height_above_reference_ground_m": relative(height_agl_path),
                "bev_height_preview": relative(bev_height_preview_path),
                "bev_instance_ids": relative(bev_ids_path),
                "bev_instance_preview": relative(bev_instance_preview_path),
                "bev_instance_overlay": relative(bev_overlay_path),
                "building_point_cloud_npz": relative(point_npz_path),
                "building_point_cloud_ply_rgb": relative(ply_rgb_path),
                "building_point_cloud_ply_instance": relative(ply_instance_path),
                "per_instance_statistics": relative(instance_stats_path),
                "input_and_process_audit": relative(audit_path),
            },
            "ply": {
                "rgb": ply_rgb_metadata,
                "instance_colors": ply_instance_metadata,
            },
        }
        records.append(record)

    overview_path = output_root / f"all_{len(records)}_instance_aware_25d_audits.png"
    build_overview(audit_paths, overview_path)
    old_absrel = [
        item["evaluation_only"]["old_calibrated_depth_all"]["abs_rel"]
        for item in records
    ]
    new_absrel = [
        item["evaluation_only"]["class_aware_calibrated_depth_all"]["abs_rel"]
        for item in records
    ]
    old_absrel_mean = None
    new_absrel_mean = None
    manifest = {
        "schema": "air2vectorcity.instance_aware_observed_25d.v1",
        "sample_count": len(records),
        "representation": {
            "name": "observed instance-aware 2.5D city surface",
            "dense_bev_fields": [
                "world surface Z in metres",
                "height above camera ground_reference_z_m",
                "RGB",
                "provisional per-view building instance ID",
                "observed/unknown validity",
            ],
            "point_record": "(X_world_m, Y_world_m, Z_world_m, RGB, instance_id)",
            "limitations": [
                "single-view visible surfaces only",
                "unknown/occluded geometry is not completed",
                "provisional IDs are not matched across views",
                "generated-mask merge/split errors remain",
                "height uses each camera's reference ground and is not yet a per-pixel local DTM height",
            ],
        },
        "construction": {
            "depth_manifests": [relative(path) for path in depth_manifest_paths],
            "v7b_palette": V7B_PALETTE.tolist(),
            "calibration_plane_policy": args.calibration_plane_policy,
            "cached_smoke_recalibration_explicit": args.allow_cached_smoke_recalibration,
            "input_depth_stages": sorted({item["depth_stage"] for item in depth_samples}),
            "scale_input": "raw metric prediction; explicit legacy smoke replay retains its already-calibrated provenance",
            "priors_json": relative(priors_path) if priors_path else None,
            "priors_sha256": sha256(priors_path) if priors_path else None,
            "fixed_smallcity_asset_plane_z_m": (
                {
                    V7B_CLASS_NAMES[key]: value
                    for key, value in (configured_priors or CALIBRATION_PLANE_Z_M).items()
                }
                if args.calibration_plane_policy == "fixed-smallcity"
                else None
            ),
            "anchor_erosion_radius_at_source_resolution_px": args.anchor_erosion_radius_px,
            "instance_gradient_threshold_rgb": args.instance_gradient_threshold_rgb,
            "instance_seed_min_area_at_1024px": args.instance_seed_min_area_1024px,
            "bev_rasterizer": "highest-world-Z forward splat",
            "gt_used_for_construction": False,
        },
        "summary": {
            "old_calibrated_depth_absrel_mean": old_absrel_mean,
            "class_aware_depth_absrel_mean": new_absrel_mean,
            "relative_absrel_change": (
                float(new_absrel_mean / old_absrel_mean - 1.0)
                if old_absrel_mean is not None and old_absrel_mean > 0.0
                else None
            ),
            "source_provisional_instance_count": finite_distribution(
                np.asarray(
                    [
                        item["instance_normalization"]["provisional_instance_count"]
                        for item in records
                    ],
                    dtype=np.float32,
                )
            ),
            "observed_bev_instance_count": finite_distribution(
                np.asarray(
                    [item["observed_building_instance_count"] for item in records],
                    dtype=np.float32,
                )
            ),
            "observed_building_point_count": finite_distribution(
                np.asarray(
                    [item["observed_building_point_count"] for item in records],
                    dtype=np.float32,
                )
            ),
        },
        "overview": relative(overview_path),
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} samples to {output_root}", flush=True)


if __name__ == "__main__":
    main()
