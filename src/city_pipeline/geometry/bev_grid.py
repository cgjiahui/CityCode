#!/usr/bin/env python3
"""Compare three orthographic raster frames without changing recovered UE XYZ.

E02 consumes the frozen E01 calibrated depth and cleaned source instances.  It
does not rerun a depth model and does not overwrite E01.  The three variants
are:

A. canonical UE +X/+Y, fixed 256 m patch;
B. camera right/image-down aligned, fixed 256 m square around the patch centre;
C. camera aligned, tight local AABB of the observations accepted by A.

All variants start from the same source RGB, depth and UE-world XYZ samples.
Every output stores an explicit pixel-boundary-to-UE-world affine transform.
"""

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


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.camera_geometry import (
    font,
    height_preview,
    lift_camera_forward_depth,
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.depth_lifting import (
    build_overview,
    instance_preview,
    save_u16,
)
from city_pipeline.geometry.depth_denoising import clean_bev_ids


DEFAULT_BASELINE_ROOT = Path("UNCONFIGURED_DEFAULT_BASELINE_ROOT")
DEFAULT_DENOISING_ROOT = DEFAULT_BASELINE_ROOT / "denoising_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_BASELINE_ROOT / "orientation_extent_ablation_e02"
CELL_SIZE_M = 0.125
FIXED_EXTENT_M = 256.0

VARIANTS = (
    ("a_world_fixed", "A. UE WORLD-ALIGNED / FIXED 256 M"),
    ("b_camera_fixed", "B. CAMERA-ALIGNED / FIXED 256 M"),
    ("c_camera_tight", "C. CAMERA-ALIGNED / TIGHT OBSERVED EXTENT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--denoising-root", type=Path, default=DEFAULT_DENOISING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Optional sample ID; may be repeated. Default: all 8 samples.",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def normalize_xy(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Camera axis has no usable horizontal projection")
    return vector / norm


def camera_aligned_axes(camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal XY frame matching source image right/down."""

    right_xy = normalize_xy(np.asarray(camera["basis_world"]["right"][:2]))
    down_xy = normalize_xy(np.asarray(camera["basis_world"]["down"][:2]))
    candidate = np.asarray([-right_xy[1], right_xy[0]], dtype=np.float64)
    if float(np.dot(candidate, down_xy)) < 0.0:
        candidate *= -1.0
    return right_xy, candidate


def local_coordinates(
    world_x: np.ndarray,
    world_y: np.ndarray,
    center_xy: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dx = world_x - np.float32(center_xy[0])
    dy = world_y - np.float32(center_xy[1])
    q_u = dx * np.float32(e_u[0]) + dy * np.float32(e_u[1])
    q_v = dx * np.float32(e_v[0]) + dy * np.float32(e_v[1])
    return q_u, q_v


def tight_bounds(
    q_u: np.ndarray,
    q_v: np.ndarray,
    source_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    if not np.any(source_mask):
        raise ValueError("No source samples are available for tight bounds")

    def aligned_range(values: np.ndarray) -> tuple[float, float]:
        minimum = math.floor(float(np.min(values)) / CELL_SIZE_M) * CELL_SIZE_M
        # Add one cell beyond the highest occupied coordinate so a value on a
        # cell boundary is retained by floor-based rasterization.
        maximum = (math.floor(float(np.max(values)) / CELL_SIZE_M) + 1) * CELL_SIZE_M
        return minimum, maximum

    min_u, max_u = aligned_range(q_u[source_mask])
    min_v, max_v = aligned_range(q_v[source_mask])
    return min_u, min_v, max_u, max_v


def frame_transforms(
    center_xy: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray,
    bounds_local_m: tuple[float, float, float, float],
    cell_size_m: float = CELL_SIZE_M,
) -> dict[str, Any]:
    min_u, min_v, max_u, max_v = bounds_local_m
    origin = center_xy + min_u * e_u + min_v * e_v
    boundary_to_world = np.asarray(
        [
            [cell_size_m * e_u[0], cell_size_m * e_v[0], origin[0]],
            [cell_size_m * e_u[1], cell_size_m * e_v[1], origin[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center_to_world = boundary_to_world.copy()
    center_to_world[:2, 2] += (
        0.5 * cell_size_m * e_u + 0.5 * cell_size_m * e_v
    )
    world_to_center = np.linalg.inv(center_to_world)
    probes = np.asarray(
        [[0.0, 0.0, 1.0], [17.25, 31.75, 1.0], [100.0, 200.0, 1.0]],
        dtype=np.float64,
    ).T
    recovered = world_to_center @ (center_to_world @ probes)
    error = float(np.max(np.abs(recovered - probes)))

    corners_local = np.asarray(
        [[min_u, min_v], [max_u, min_v], [max_u, max_v], [min_u, max_v]],
        dtype=np.float64,
    )
    corners_world = (
        center_xy[None, :]
        + corners_local[:, 0:1] * e_u[None, :]
        + corners_local[:, 1:2] * e_v[None, :]
    )
    return {
        "center_world_xy_m": center_xy.tolist(),
        "axis_u_world_xy": e_u.tolist(),
        "axis_v_world_xy": e_v.tolist(),
        "bounds_local_m": [min_u, min_v, max_u, max_v],
        "origin_world_xy_m_at_pixel_boundary": origin.tolist(),
        "pixel_boundary_to_ue_world_xy_3x3": boundary_to_world.tolist(),
        "pixel_center_index_to_ue_world_xy_3x3": center_to_world.tolist(),
        "ue_world_xy_to_pixel_center_index_3x3": world_to_center.tolist(),
        "raster_footprint_world_xy_m": corners_world.tolist(),
        "pixel_world_roundtrip_max_abs_error": error,
    }


def rasterize_local(
    attributes: dict[str, np.ndarray],
    world_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    center_xy: np.ndarray,
    e_u: np.ndarray,
    e_v: np.ndarray,
    bounds_local_m: tuple[float, float, float, float],
    source_filter: np.ndarray | None = None,
    cell_size_m: float = CELL_SIZE_M,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    world_x, world_y, world_z = world_xyz
    source_height, source_width = world_x.shape
    if any(value.shape[:2] != world_x.shape for value in attributes.values()):
        raise ValueError("All attributes must align with world XYZ")

    min_u, min_v, max_u, max_v = bounds_local_m
    width = int(round((max_u - min_u) / cell_size_m))
    height = int(round((max_v - min_v) / cell_size_m))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid output size {width}x{height}")

    q_u, q_v = local_coordinates(world_x, world_y, center_xy, e_u, e_v)
    finite_xy = np.isfinite(world_x) & np.isfinite(world_y)
    target_u = np.full(q_u.shape, -1, dtype=np.int32)
    target_v = np.full(q_v.shape, -1, dtype=np.int32)
    target_u[finite_xy] = np.floor(
        (q_u[finite_xy] - np.float32(min_u)) / cell_size_m
    ).astype(np.int32)
    target_v[finite_xy] = np.floor(
        (q_v[finite_xy] - np.float32(min_v)) / cell_size_m
    ).astype(np.int32)
    valid = (
        finite_xy
        & np.isfinite(world_z)
        & (target_u >= 0)
        & (target_u < width)
        & (target_v >= 0)
        & (target_v < height)
    )
    if source_filter is not None:
        valid &= source_filter

    source_flat = np.flatnonzero(valid).astype(np.int32)
    target_flat = (
        target_v[valid].astype(np.int64) * width + target_u[valid].astype(np.int64)
    )
    if not target_flat.size:
        raise ValueError("No lifted samples fall inside the local raster")

    z_valid = world_z[valid]
    target_count = width * height
    top_z = np.full(target_count, -np.inf, dtype=np.float32)
    np.maximum.at(top_z, target_flat, z_valid)
    is_top = z_valid == top_z[target_flat]
    selected_source = np.full(target_count, -1, dtype=np.int32)
    np.maximum.at(selected_source, target_flat[is_top], source_flat[is_top])
    observed = selected_source >= 0
    selected_target = np.flatnonzero(observed)
    selected_source_indices = selected_source[observed]

    outputs: dict[str, np.ndarray] = {}
    for name, source in attributes.items():
        source_values = source.reshape((source_height * source_width,) + source.shape[2:])
        target = np.zeros((target_count,) + source.shape[2:], dtype=source.dtype)
        target[selected_target] = source_values[selected_source_indices]
        outputs[name] = target.reshape((height, width) + source.shape[2:])

    surface_z = np.full(target_count, np.nan, dtype=np.float32)
    surface_z[observed] = top_z[observed]
    transforms = frame_transforms(center_xy, e_u, e_v, bounds_local_m, cell_size_m)
    metadata = {
        "method": "forward_local_orthographic_splat_with_highest_world_z_zbuffer",
        "output_size_px": [width, height],
        "meters_per_pixel": [cell_size_m, cell_size_m],
        "raw_target_valid_pixels": int(np.count_nonzero(observed)),
        "raw_target_valid_fraction": float(np.mean(observed)),
        "source_samples_inside_target": int(target_flat.size),
        "source_samples_inside_target_fraction": float(np.mean(valid)),
        "source_filter_applied": source_filter is not None,
        **transforms,
    }
    return outputs, observed.reshape(height, width), surface_z.reshape(height, width), metadata


def fit_panel(array: np.ndarray, size: int = 360, is_mask: bool = False) -> Image.Image:
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    image.thumbnail((size, size), resample)
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return panel


def world_frame_diagram(
    canonical_bounds: list[float],
    raster_polygon: list[list[float]],
    observed_xy: np.ndarray,
    size: int = 360,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = canonical_bounds
    canonical = np.asarray(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        dtype=np.float64,
    )
    raster = np.asarray(raster_polygon, dtype=np.float64)
    all_points = np.concatenate([canonical, raster], axis=0)
    lo = np.min(all_points, axis=0)
    hi = np.max(all_points, axis=0)
    span = np.maximum(hi - lo, 1.0)
    lo -= span * 0.08
    hi += span * 0.08

    def to_px(points: np.ndarray) -> list[tuple[int, int]]:
        normalized = (points - lo[None, :]) / (hi - lo)[None, :]
        return [
            (int(round(x * (size - 1))), int(round(y * (size - 1))))
            for x, y in normalized
        ]

    canvas = Image.new("RGB", (size, size), (17, 20, 24))
    draw = ImageDraw.Draw(canvas)
    if observed_xy.size:
        stride = max(1, observed_xy.shape[0] // 18_000)
        for point in to_px(observed_xy[::stride]):
            draw.point(point, fill=(34, 116, 130))
    canonical_px = to_px(canonical)
    raster_px = to_px(raster)
    draw.line(canonical_px + [canonical_px[0]], fill=(235, 235, 235), width=3)
    draw.line(raster_px + [raster_px[0]], fill=(255, 177, 45), width=4)
    label_font = font(18)
    draw.text((12, 10), "white: canonical 256 m patch", fill=(235, 235, 235), font=label_font)
    draw.text((12, 34), "orange: current raster", fill=(255, 177, 45), font=label_font)
    draw.text((12, 58), "cyan: accepted source samples", fill=(70, 190, 200), font=label_font)
    return np.asarray(canvas)


def comparison_grid(
    sample_id: str,
    source_rgb: np.ndarray,
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    tile = 360
    label_height = 78
    columns = 5
    canvas = Image.new("RGB", (columns * tile, len(rows) * (tile + label_height)), (12, 14, 17))
    draw = ImageDraw.Draw(canvas)
    title_font = font(19)
    small_font = font(16)
    for row_index, row in enumerate(rows):
        y = row_index * (tile + label_height)
        valid_fraction = row["metadata"]["raw_target_valid_fraction"]
        output_size = row["metadata"]["output_size_px"]
        title = (
            f"{sample_id} | {row['label']} | "
            f"{output_size[0]}x{output_size[1]} | observed={valid_fraction:.1%}"
        )
        draw.rectangle((0, y, columns * tile, y + label_height), fill=(31, 38, 47))
        draw.text((14, y + 10), title, fill=(248, 248, 248), font=title_font)
        draw.text(
            (14, y + 42),
            "same UE XYZ; orange frame is exactly mapped back by pixel_to_world",
            fill=(185, 199, 213),
            font=small_font,
        )
        panels = (
            (source_rgb, False),
            (row["frame_diagram"], True),
            (row["rgb"], False),
            (row["validity_rgb"], True),
            (row["instance_preview"], True),
        )
        for column, (array, is_mask) in enumerate(panels):
            canvas.paste(fit_panel(array, tile, is_mask), (column * tile, y + label_height))
        labels = ("SOURCE RGB", "UE WORLD FOOTPRINT", "ORTHO RGB", "VALIDITY", "BUILDING IDS")
        for column, label in enumerate(labels):
            draw.rectangle(
                (column * tile + 6, y + label_height + 6, column * tile + 196, y + label_height + 32),
                fill=(0, 0, 0),
            )
            draw.text(
                (column * tile + 12, y + label_height + 8),
                label,
                fill=(255, 255, 255),
                font=small_font,
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)


def save_variant(
    output_root: Path,
    variant_id: str,
    sample_id: str,
    rgb: np.ndarray,
    valid: np.ndarray,
    final_ids: np.ndarray,
    surface_z: np.ndarray,
    building_height: np.ndarray,
    instance_rgb: np.ndarray,
    height_rgb: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, str]:
    root = output_root / variant_id
    paths = {
        "rgb": root / "bev_rgb" / f"{sample_id}__bev_rgb.png",
        "validity": root / "bev_validity" / f"{sample_id}__bev_validity.png",
        "instance_ids": root / "bev_instance_ids" / f"{sample_id}__bev_instance_ids.png",
        "instance_preview": root / "bev_instance_previews" / f"{sample_id}__bev_instance_preview.png",
        "height_preview": root / "bev_building_height_previews" / f"{sample_id}__bev_building_height_preview.png",
        "surface": root / "surface_fields" / f"{sample_id}__surface_fields.npz",
        "transform": root / "transforms" / f"{sample_id}__pixel_to_ue_world.json",
    }
    save_rgb(paths["rgb"], rgb)
    save_u8(paths["validity"], valid.astype(np.uint8) * 255)
    save_u16(paths["instance_ids"], final_ids)
    save_rgb(paths["instance_preview"], instance_rgb)
    save_rgb(paths["height_preview"], height_rgb)
    paths["surface"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["surface"],
        surface_z_m=surface_z.astype(np.float32),
        building_height_above_reference_ground_m=building_height.astype(np.float32),
        valid=valid,
    )
    paths["transform"].parent.mkdir(parents=True, exist_ok=True)
    paths["transform"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {key: path.relative_to(PROJECT_ROOT).as_posix() for key, path in paths.items()}


def main() -> int:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    denoising_root = args.denoising_root.resolve()
    output_root = args.output_root.resolve()
    baseline = read_json(baseline_root / "manifest.json")
    denoising = read_json(denoising_root / "manifest.json")
    baseline_by_id = {sample["id"]: sample for sample in baseline["samples"]}
    denoising_by_id = {sample["id"]: sample for sample in denoising["samples"]}
    sample_ids = args.sample_id or [sample["id"] for sample in baseline["samples"]]
    if set(sample_ids) - set(denoising_by_id):
        raise ValueError("Requested sample is missing from E01")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        base_sample = baseline_by_id[sample_id]
        clean_sample = denoising_by_id[sample_id]
        source_rgb = load_rgb(project_path(base_sample["input"]["rgb"]))
        camera = read_json(project_path(base_sample["input"]["camera"]))
        depth = np.load(project_path(clean_sample["outputs"]["depth"])).astype(np.float32)
        with Image.open(project_path(clean_sample["outputs"]["source_ids"])) as image:
            source_ids = np.asarray(image, dtype=np.uint16)
        world_xyz, lifting = lift_camera_forward_depth(depth, camera)
        world_x, world_y, world_z = world_xyz
        finite = np.isfinite(world_x) & np.isfinite(world_y) & np.isfinite(world_z)

        patch_bounds = [float(value) for value in camera["target_patch_bounds_xy_m"]]
        min_x, min_y, max_x, max_y = patch_bounds
        canonical_source = (
            finite
            & (world_x >= min_x)
            & (world_x < max_x)
            & (world_y >= min_y)
            & (world_y < max_y)
        )
        center_xy = np.asarray(camera["target_center_world_m"][:2], dtype=np.float64)
        camera_u, camera_v = camera_aligned_axes(camera)
        q_u, q_v = local_coordinates(world_x, world_y, center_xy, camera_u, camera_v)
        tight = tight_bounds(q_u, q_v, canonical_source)
        half = FIXED_EXTENT_M / 2.0

        definitions = (
            {
                "id": "a_world_fixed",
                "label": VARIANTS[0][1],
                "center": np.asarray([0.0, 0.0], dtype=np.float64),
                "e_u": np.asarray([1.0, 0.0], dtype=np.float64),
                "e_v": np.asarray([0.0, 1.0], dtype=np.float64),
                "bounds": (min_x, min_y, max_x, max_y),
                "source_filter": None,
            },
            {
                "id": "b_camera_fixed",
                "label": VARIANTS[1][1],
                "center": center_xy,
                "e_u": camera_u,
                "e_v": camera_v,
                "bounds": (-half, -half, half, half),
                "source_filter": None,
            },
            {
                "id": "c_camera_tight",
                "label": VARIANTS[2][1],
                "center": center_xy,
                "e_u": camera_u,
                "e_v": camera_v,
                "bounds": tight,
                "source_filter": canonical_source,
            },
        )

        rows: list[dict[str, Any]] = []
        variant_records: dict[str, Any] = {}
        a_source_count = int(np.count_nonzero(canonical_source))
        accepted_xy = np.stack((world_x[canonical_source], world_y[canonical_source]), axis=1)
        for definition in definitions:
            recovered, valid, surface_z, raster = rasterize_local(
                {"rgb": source_rgb, "instance_id": source_ids},
                world_xyz,
                definition["center"],
                definition["e_u"],
                definition["e_v"],
                definition["bounds"],
                definition["source_filter"],
            )
            (
                component_ids,
                final_ids,
                clean_surface_z,
                clean_rgb,
                provenance,
                cleaning,
            ) = clean_bev_ids(
                recovered["instance_id"].astype(np.uint16),
                valid,
                surface_z,
                recovered["rgb"],
                (CELL_SIZE_M, CELL_SIZE_M),
            )
            del component_ids, provenance
            building_valid = (final_ids > 0) & np.isfinite(clean_surface_z)
            building_height = np.full(clean_surface_z.shape, np.nan, dtype=np.float32)
            building_height[building_valid] = (
                clean_surface_z[building_valid]
                - np.float32(camera["ground_reference_z_m"])
            )
            instance_rgb = instance_preview(final_ids)
            height_rgb = height_preview(building_height, building_valid)
            height_rgb[~building_valid] = 0

            q_u_def, q_v_def = local_coordinates(
                world_x, world_y, definition["center"], definition["e_u"], definition["e_v"]
            )
            b0, b1, b2, b3 = definition["bounds"]
            canonical_retained = canonical_source & (q_u_def >= b0) & (q_u_def < b2) & (q_v_def >= b1) & (q_v_def < b3)
            raster["canonical_source_sample_retention_fraction"] = (
                float(np.count_nonzero(canonical_retained) / a_source_count)
                if a_source_count
                else None
            )
            raster["canonical_patch_bounds_world_xy_m"] = patch_bounds
            raster["cleaning"] = cleaning
            outputs = save_variant(
                output_root,
                definition["id"],
                sample_id,
                clean_rgb,
                valid,
                final_ids,
                clean_surface_z,
                building_height,
                instance_rgb,
                height_rgb,
                raster,
            )
            frame_diagram = world_frame_diagram(
                patch_bounds,
                raster["raster_footprint_world_xy_m"],
                accepted_xy,
            )
            rows.append(
                {
                    "label": definition["label"],
                    "metadata": raster,
                    "rgb": clean_rgb,
                    "validity_rgb": np.repeat((valid.astype(np.uint8) * 255)[..., None], 3, axis=2),
                    "instance_preview": instance_rgb,
                    "frame_diagram": frame_diagram,
                }
            )
            variant_records[definition["id"]] = {"rasterization": raster, "outputs": outputs}

            if definition["id"] == "a_world_fixed":
                with Image.open(project_path(clean_sample["outputs"]["bev_final_ids"])) as image:
                    baseline_ids = np.asarray(image, dtype=np.uint16)
                baseline_height = np.load(project_path(clean_sample["outputs"]["height"]))
                variant_records[definition["id"]]["e01_exact_validation"] = {
                    "final_instance_ids_equal": bool(np.array_equal(final_ids, baseline_ids)),
                    "building_height_equal_nan_aware": bool(
                        np.allclose(building_height, baseline_height, rtol=0.0, atol=0.0, equal_nan=True)
                    ),
                }

        comparison = output_root / "comparisons" / f"{sample_id}__three_orthographic_frames.png"
        comparison_grid(sample_id, source_rgb, rows, comparison)
        comparisons.append(comparison)
        records.append(
            {
                "id": sample_id,
                "input_rgb": base_sample["input"]["rgb"],
                "camera": base_sample["input"]["camera"],
                "lifting": lifting,
                "canonical_source_sample_count": a_source_count,
                "variants": variant_records,
                "comparison": comparison.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    overview = output_root / "all_8_three_orthographic_frame_comparisons.png"
    build_overview(comparisons, overview)
    manifest = {
        "schema": "air2vectorcity.bev_orientation_extent_ablation_e02.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "completed_visual_ablation_not_a_new_baseline",
        "frozen_input_baseline": denoising_root.relative_to(PROJECT_ROOT).as_posix(),
        "cell_size_m": CELL_SIZE_M,
        "variants": {
            "a_world_fixed": "UE +X/+Y, fixed canonical 256 m patch",
            "b_camera_fixed": "camera right/image-down aligned, fixed 256 m square",
            "c_camera_tight": "camera aligned, tight AABB of A-accepted samples; canonical source filter retained",
        },
        "overview": overview.relative_to(PROJECT_ROOT).as_posix(),
        "samples": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
