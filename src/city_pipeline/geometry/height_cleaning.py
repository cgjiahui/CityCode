#!/usr/bin/env python3
"""Build E10 height-guided modeling instance masks from the frozen E09 run.

E09 remains the authoritative observation.  E10 derives a second instance-ID
raster for coarse modeling by finding spatially supported roof-height levels
and pruning only exterior, height-inconsistent, thin appendages.  Interior
height outliers are recorded but do not punch holes in the instance mask.
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
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.roof_masks import two_row_comparison  # noqa: E402
from city_pipeline.geometry.camera_geometry import (  # noqa: E402
    height_preview,
    load_rgb,
    read_json,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.camera_bev import (  # noqa: E402
    building_point_cloud,
)
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    build_overview,
    instance_palette,
    instance_preview,
    mask_rgb,
    rgb_overlay,
    save_u16,
    write_binary_ply,
)


DEFAULT_E09_ROOT = Path("UNCONFIGURED_DEFAULT_E09_ROOT")
DEFAULT_OUTPUT_ROOT = Path("UNCONFIGURED_DEFAULT_OUTPUT_ROOT")

CELL_SIZE_M = 0.125
PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
HEIGHT_BIN_M = 0.5
HEIGHT_WINDOW_M = 2.0
HEIGHT_STEP_M = 0.5
BASE_LAYER_FRACTION = 0.05
MIN_LAYER_AREA_M2 = 12.0
MAX_LAYER_AREA_REQUIREMENT_M2 = 60.0
MIN_CONNECTED_AREA_M2 = 8.0
MIN_CONNECTED_REQUIREMENT_FRACTION = 0.50
STRONG_MIN_INTERNAL_RADIUS_M = 1.50
CONDITIONAL_MIN_INTERNAL_RADIUS_M = 1.00
CONDITIONAL_LAYER_FRACTION = 0.10
CONDITIONAL_MIN_AREA_M2 = 20.0
CONDITIONAL_MAX_AREA_REQUIREMENT_M2 = 80.0
CONDITIONAL_CONNECTED_FRACTION = 0.70
LAYER_MERGE_DISTANCE_M = 2.50
MAX_STABLE_LAYERS = 3
LAYER_ASSIGNMENT_TOLERANCE_M = 2.0
TAIL_MAX_INTERNAL_RADIUS_M = 1.50
VERY_THIN_RADIUS_M = 1.00
TAIL_MAX_NECK_WIDTH_M = 1.50
TAIL_MIN_INCONSISTENT_FRACTION = 0.60
TAIL_MAX_AREA_FRACTION = 0.15
OBSERVED_HEIGHT_PROVENANCE = (1, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e09-root", type=Path, default=DEFAULT_E09_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument(
        "--merge-existing-manifest",
        action="store_true",
        help=(
            "When --sample-id is used, replace those sample records in an existing "
            "complete manifest instead of publishing a partial manifest."
        ),
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def save_float(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def colored_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[mask] = np.asarray(color, dtype=np.uint8)
    return output


def component_measure(mask: np.ndarray) -> tuple[float, float]:
    count = int(np.count_nonzero(mask))
    if not count:
        return 0.0, 0.0
    area = float(count * PIXEL_AREA_M2)
    radius = float(ndimage.distance_transform_edt(mask).max() * CELL_SIZE_M)
    return area, radius


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    labels, component_count = ndimage.label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if not component_count:
        return np.zeros_like(mask), 0.0, 0.0, 0
    sizes = np.bincount(labels.reshape(-1))
    label_id = 1 + int(np.argmax(sizes[1:]))
    selected = labels == label_id
    area, radius = component_measure(selected)
    return selected, area, radius, int(component_count)


def required_layer_area(observed_area_m2: float) -> float:
    return max(
        MIN_LAYER_AREA_M2,
        min(
            MAX_LAYER_AREA_REQUIREMENT_M2,
            observed_area_m2 * BASE_LAYER_FRACTION,
        ),
    )


def required_conditional_area(observed_area_m2: float) -> float:
    return max(
        CONDITIONAL_MIN_AREA_M2,
        min(
            CONDITIONAL_MAX_AREA_REQUIREMENT_M2,
            observed_area_m2 * CONDITIONAL_LAYER_FRACTION,
        ),
    )


def discover_height_layers(
    direct_observed: np.ndarray,
    surface_z: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = surface_z[direct_observed].astype(np.float64)
    observed_area = float(values.size * PIXEL_AREA_M2)
    if not values.size:
        return [], {
            "observed_area_m2": 0.0,
            "base_required_area_m2": 0.0,
            "conditional_required_area_m2": 0.0,
            "candidate_window_count": 0,
            "accepted_window_count": 0,
            "fallback": "no_provenance_1_height",
        }

    base_required = required_layer_area(observed_area)
    conditional_required = required_conditional_area(observed_area)
    connected_required = max(
        MIN_CONNECTED_AREA_M2,
        base_required * MIN_CONNECTED_REQUIREMENT_FRACTION,
    )
    highest_edge = math.ceil(float(np.max(values)) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    lowest_edge = math.floor(float(np.min(values)) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    windows: list[dict[str, Any]] = []
    upper = highest_edge
    while upper >= lowest_edge:
        lower = upper - HEIGHT_WINDOW_M
        band = direct_observed & (surface_z > lower) & (surface_z <= upper + 1e-5)
        area = float(np.count_nonzero(band) * PIXEL_AREA_M2)
        if area >= base_required:
            seed, largest_area, radius, component_count = largest_component(band)
            strong = (
                largest_area >= connected_required
                and radius >= STRONG_MIN_INTERNAL_RADIUS_M
            )
            conditional = (
                CONDITIONAL_MIN_INTERNAL_RADIUS_M
                <= radius
                < STRONG_MIN_INTERNAL_RADIUS_M
                and area >= conditional_required
                and largest_area >= area * CONDITIONAL_CONNECTED_FRACTION
            )
            if strong or conditional:
                seed_values = surface_z[seed]
                windows.append(
                    {
                        "lower_m": float(lower),
                        "upper_m": float(upper),
                        "center_m": float((lower + upper) / 2.0),
                        "height_m": float(np.median(seed_values)),
                        "support_area_m2": area,
                        "support_fraction": float(area / observed_area),
                        "largest_connected_area_m2": largest_area,
                        "internal_radius_m": radius,
                        "component_count": component_count,
                        "acceptance": "strong_radius" if strong else "conditional_radius",
                        "seed_mask": seed,
                    }
                )
        upper -= HEIGHT_STEP_M

    # Overlapping windows around the same histogram peak are one height layer.
    chosen: list[dict[str, Any]] = []
    for candidate in sorted(
        windows, key=lambda item: (item["support_area_m2"], item["height_m"]), reverse=True
    ):
        if any(
            abs(candidate["height_m"] - existing["height_m"])
            <= LAYER_MERGE_DISTANCE_M
            for existing in chosen
        ):
            continue
        chosen.append(candidate)
        if len(chosen) >= MAX_STABLE_LAYERS:
            break

    chosen.sort(key=lambda item: item["height_m"])
    for layer_index, layer in enumerate(chosen, start=1):
        layer["layer_index"] = layer_index
    metadata = {
        "observed_area_m2": observed_area,
        "base_required_area_m2": base_required,
        "conditional_required_area_m2": conditional_required,
        "connected_required_area_m2": connected_required,
        "candidate_window_count": len(windows),
        "accepted_window_count": len(chosen),
        "fallback": None if chosen else "no_window_passed; preserve_instance_without_pruning",
    }
    return chosen, metadata


def layer_regions(
    instance_mask: np.ndarray,
    surface_z: np.ndarray,
    provenance: np.ndarray,
    layers: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stable = np.zeros(instance_mask.shape, dtype=bool)
    layer_index_map = np.zeros(instance_mask.shape, dtype=np.uint8)
    residual = np.full(instance_mask.shape, np.nan, dtype=np.float32)
    assigned_distance = np.full(instance_mask.shape, np.inf, dtype=np.float32)
    observed_height = (
        instance_mask
        & np.isfinite(surface_z)
        & np.isin(provenance, np.asarray(OBSERVED_HEIGHT_PROVENANCE, dtype=np.uint8))
    )
    if not layers:
        return stable, layer_index_map, residual

    heights = np.asarray([layer["height_m"] for layer in layers], dtype=np.float32)
    observed_values = surface_z[observed_height]
    distances = np.abs(observed_values[:, None] - heights[None, :])
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(distances.shape[0]), nearest]
    residual[observed_height] = nearest_distance

    for zero_based, layer in enumerate(layers):
        consistent = (
            observed_height
            & (np.abs(surface_z - layer["height_m"]) <= LAYER_ASSIGNMENT_TOLERANCE_M)
        )
        labels, component_count = ndimage.label(
            consistent, structure=np.ones((3, 3), dtype=np.uint8)
        )
        seed = layer["seed_mask"]
        seed_touch = ndimage.binary_dilation(
            seed, structure=np.ones((3, 3), dtype=bool)
        )
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            area, radius = component_measure(component)
            keep = bool(np.any(component & seed_touch)) or (
                area >= MIN_CONNECTED_AREA_M2
                and radius >= CONDITIONAL_MIN_INTERNAL_RADIUS_M
            )
            if not keep:
                continue
            layer_distance = np.abs(surface_z - layer["height_m"])
            assign = component & (layer_distance < assigned_distance)
            stable[assign] = True
            layer_index_map[assign] = np.uint8(zero_based + 1)
            assigned_distance[assign] = layer_distance[assign]

    return stable, layer_index_map, residual


def maximum_height_slice_radius(
    component: np.ndarray,
    observed_height: np.ndarray,
    surface_z: np.ndarray,
) -> float:
    selected = component & observed_height
    values = surface_z[selected]
    if not values.size:
        return 0.0
    highest_edge = math.ceil(float(np.max(values)) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    lowest_edge = math.floor(float(np.min(values)) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    maximum = 0.0
    upper = highest_edge
    while upper >= lowest_edge:
        lower = upper - HEIGHT_WINDOW_M
        band = selected & (surface_z > lower) & (surface_z <= upper + 1e-5)
        _, _, radius, _ = largest_component(band)
        maximum = max(maximum, radius)
        upper -= HEIGHT_STEP_M
    return maximum


def clean_instance(
    instance_id: int,
    instance_mask: np.ndarray,
    surface_z: np.ndarray,
    provenance: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    y, x = np.nonzero(instance_mask)
    margin = 4
    y0 = max(0, int(np.min(y)) - margin)
    y1 = min(instance_mask.shape[0], int(np.max(y)) + margin + 1)
    x0 = max(0, int(np.min(x)) - margin)
    x1 = min(instance_mask.shape[1], int(np.max(x)) + margin + 1)
    mask = instance_mask[y0:y1, x0:x1]
    z = surface_z[y0:y1, x0:x1]
    prov = provenance[y0:y1, x0:x1]
    direct = mask & np.isfinite(z) & (prov == 1)
    layers, layer_metadata = discover_height_layers(direct, z)

    cleaned = mask.copy()
    stable, layer_indices, residual = layer_regions(mask, z, prov, layers)
    removed = np.zeros(mask.shape, dtype=bool)
    interior_outlier = np.zeros(mask.shape, dtype=bool)
    exterior_candidate = np.zeros(mask.shape, dtype=bool)
    component_records: list[dict[str, Any]] = []

    if layers:
        enclosed_stable = ndimage.binary_fill_holes(stable)
        unstable = mask & ~stable
        interior_outlier = unstable & enclosed_stable
        exterior_candidate = unstable & ~enclosed_stable
        labels, component_count = ndimage.label(
            exterior_candidate, structure=np.ones((3, 3), dtype=np.uint8)
        )
        observed_height = (
            mask
            & np.isfinite(z)
            & np.isin(prov, np.asarray(OBSERVED_HEIGHT_PROVENANCE, dtype=np.uint8))
        )
        stable_dilated = ndimage.binary_dilation(
            stable, structure=np.ones((3, 3), dtype=bool)
        )
        observed_area = max(layer_metadata["observed_area_m2"], PIXEL_AREA_M2)
        for component_id in range(1, component_count + 1):
            component = labels == component_id
            area, radius = component_measure(component)
            observed_component = component & observed_height
            observed_count = int(np.count_nonzero(observed_component))
            if observed_count:
                inconsistent = observed_component & (
                    ~np.isfinite(residual)
                    | (residual > LAYER_ASSIGNMENT_TOLERANCE_M)
                )
                inconsistent_fraction = float(
                    np.count_nonzero(inconsistent) / observed_count
                )
            else:
                inconsistent_fraction = 0.0
            contact = component & stable_dilated
            neck_width = float(np.count_nonzero(contact) * CELL_SIZE_M)
            slice_radius = maximum_height_slice_radius(
                component, observed_height, z
            )
            area_fraction = float(area / observed_area)
            very_thin = slice_radius < VERY_THIN_RADIUS_M
            thin_or_narrow = (
                slice_radius < TAIL_MAX_INTERNAL_RADIUS_M
                or (0.0 < neck_width < TAIL_MAX_NECK_WIDTH_M)
            )
            within_area_guard = area_fraction <= TAIL_MAX_AREA_FRACTION
            should_remove = (
                observed_count > 0
                and inconsistent_fraction >= TAIL_MIN_INCONSISTENT_FRACTION
                and thin_or_narrow
                and (within_area_guard or very_thin)
            )
            if should_remove:
                removed |= component
            component_records.append(
                {
                    "component_id": component_id,
                    "area_m2": area,
                    "area_fraction_of_observed": area_fraction,
                    "internal_radius_m": radius,
                    "maximum_2m_height_slice_radius_m": slice_radius,
                    "stable_contact_width_proxy_m": neck_width,
                    "observed_height_pixel_count": observed_count,
                    "height_inconsistent_fraction": inconsistent_fraction,
                    "removed": bool(should_remove),
                }
            )
        cleaned[removed] = False

    record_layers = []
    for layer in layers:
        record_layers.append(
            {key: value for key, value in layer.items() if key != "seed_mask"}
        )
    selected_top = max((layer["height_m"] for layer in layers), default=None)
    metadata = {
        "instance_id": instance_id,
        "layer_discovery": layer_metadata,
        "stable_layers": record_layers,
        "stable_layer_count": len(layers),
        "selected_top_z_m": selected_top,
        "mask_area_before_m2": float(np.count_nonzero(mask) * PIXEL_AREA_M2),
        "mask_area_after_m2": float(np.count_nonzero(cleaned) * PIXEL_AREA_M2),
        "removed_tail_area_m2": float(np.count_nonzero(removed) * PIXEL_AREA_M2),
        "interior_height_outlier_area_m2": float(
            np.count_nonzero(interior_outlier) * PIXEL_AREA_M2
        ),
        "automatic_pruning_applied": bool(layers),
        "exterior_components": component_records,
    }

    full_cleaned = np.zeros(instance_mask.shape, dtype=bool)
    full_stable = np.zeros(instance_mask.shape, dtype=bool)
    full_layers = np.zeros(instance_mask.shape, dtype=np.uint8)
    full_removed = np.zeros(instance_mask.shape, dtype=bool)
    full_interior = np.zeros(instance_mask.shape, dtype=bool)
    full_exterior = np.zeros(instance_mask.shape, dtype=bool)
    full_cleaned[y0:y1, x0:x1] = cleaned
    full_stable[y0:y1, x0:x1] = stable
    full_layers[y0:y1, x0:x1] = layer_indices
    full_removed[y0:y1, x0:x1] = removed
    full_interior[y0:y1, x0:x1] = interior_outlier
    full_exterior[y0:y1, x0:x1] = exterior_candidate
    return (
        full_cleaned,
        full_stable,
        full_layers,
        full_removed,
        full_interior,
        full_exterior,
        metadata,
    )


def main() -> int:
    global CELL_SIZE_M, PIXEL_AREA_M2
    args = parse_args()
    e09_root = args.e09_root.resolve()
    output_root = args.output_root.resolve()
    existing_manifest = None
    existing_manifest_path = output_root / "manifest.json"
    if args.merge_existing_manifest:
        if not args.sample_id:
            raise ValueError("--merge-existing-manifest requires --sample-id")
        if not existing_manifest_path.is_file():
            raise FileNotFoundError(existing_manifest_path)
        existing_manifest = read_json(existing_manifest_path)
    e09 = read_json(e09_root / "manifest.json")
    samples = e09["samples"]
    if args.sample_id:
        selected = set(args.sample_id)
        samples = [sample for sample in samples if sample["id"] in selected]
    cell_sizes = {
        float(sample["rasterization"]["meters_per_pixel"][0]) for sample in samples
    }
    if len(cell_sizes) != 1:
        raise ValueError(f"E10 requires one consistent BEV cell size: {sorted(cell_sizes)}")
    CELL_SIZE_M = cell_sizes.pop()
    PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M

    directory_names = (
        "modeling_instance_ids",
        "modeling_instance_previews",
        "stable_height_layer_codes",
        "stable_height_layer_previews",
        "stable_height_support",
        "tail_removed_masks",
        "interior_height_outlier_masks",
        "exterior_unstable_candidate_masks",
        "selected_top_support_ids",
        "modeling_building_height_m",
        "modeling_building_height_previews",
        "modeling_overlays",
        "building_point_cloud_npz",
        "building_point_cloud_ply_instance",
        "height_layer_tables",
        "comparisons",
    )
    directories = {name: output_root / name for name in directory_names}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    for sample_index, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        print(f"[{sample_index}/{len(samples)}] {sample_id}", flush=True)
        original_ids = np.asarray(
            Image.open(project_path(sample["outputs"]["final_ids"])), dtype=np.uint16
        )
        surface_z = np.load(project_path(sample["outputs"]["surface_z"])).astype(
            np.float32
        )
        provenance = np.asarray(
            Image.open(project_path(sample["outputs"]["provenance"])), dtype=np.uint8
        )
        bev_rgb = load_rgb(project_path(sample["outputs"]["bev_rgb"]))
        modeling_ids = original_ids.copy()
        stable_support = np.zeros(original_ids.shape, dtype=bool)
        stable_layer_codes = np.zeros(original_ids.shape, dtype=np.uint16)
        tail_removed = np.zeros(original_ids.shape, dtype=bool)
        interior_outlier = np.zeros(original_ids.shape, dtype=bool)
        exterior_candidate = np.zeros(original_ids.shape, dtype=bool)
        top_support_ids = np.zeros(original_ids.shape, dtype=np.uint16)
        instance_records: list[dict[str, Any]] = []

        for instance_id in np.unique(original_ids):
            if instance_id == 0:
                continue
            instance_mask = original_ids == instance_id
            (
                cleaned,
                stable,
                layer_indices,
                removed,
                interior,
                exterior,
                metadata,
            ) = clean_instance(
                int(instance_id), instance_mask, surface_z, provenance
            )
            modeling_ids[instance_mask & ~cleaned] = 0
            stable_support |= stable
            tail_removed |= removed
            interior_outlier |= interior
            exterior_candidate |= exterior
            stable_layer_codes[layer_indices > 0] = (
                np.uint16(instance_id) * np.uint16(MAX_STABLE_LAYERS + 1)
                + layer_indices[layer_indices > 0].astype(np.uint16)
            )
            if metadata["stable_layers"]:
                top_layer = max(
                    metadata["stable_layers"], key=lambda item: item["height_m"]
                )
                top_index = int(top_layer["layer_index"])
                top_support_ids[
                    (layer_indices == top_index) & cleaned
                ] = np.uint16(instance_id)
            instance_records.append(metadata)

        if np.any((modeling_ids > 0) & (original_ids == 0)):
            raise AssertionError("E10 may only delete E09 IDs; it cannot invent mask cells")
        if np.any((modeling_ids > 0) & (modeling_ids != original_ids)):
            raise AssertionError("E10 cannot change one non-zero instance ID into another")

        modeling_valid = (modeling_ids > 0) & np.isfinite(surface_z)
        ground_reference_z_m = float(sample.get("ground_reference_z_m", 0.48))
        modeling_height = np.full(surface_z.shape, np.nan, dtype=np.float32)
        modeling_height[modeling_valid] = (
            surface_z[modeling_valid] - np.float32(ground_reference_z_m)
        )
        original_preview = instance_preview(original_ids)
        modeling_preview = instance_preview(modeling_ids)
        layer_preview = instance_preview(stable_layer_codes)
        overlay = rgb_overlay(bev_rgb, modeling_preview, modeling_ids > 0)
        direct_height_mask = (
            (original_ids > 0) & np.isfinite(surface_z) & (provenance == 1)
        )
        direct_height_preview = height_preview(
            surface_z - np.float32(ground_reference_z_m), direct_height_mask
        )
        direct_height_preview[~direct_height_mask] = 0
        modeling_height_preview = height_preview(modeling_height, modeling_valid)
        modeling_height_preview[~modeling_valid] = 0

        paths = {
            "modeling_ids": directories["modeling_instance_ids"] / f"{sample_id}__modeling_instance_ids.png",
            "modeling_preview": directories["modeling_instance_previews"] / f"{sample_id}__modeling_instance_preview.png",
            "layer_codes": directories["stable_height_layer_codes"] / f"{sample_id}__stable_height_layer_codes.png",
            "layer_preview": directories["stable_height_layer_previews"] / f"{sample_id}__stable_height_layer_preview.png",
            "stable_support": directories["stable_height_support"] / f"{sample_id}__stable_height_support.png",
            "tail_removed": directories["tail_removed_masks"] / f"{sample_id}__tail_removed.png",
            "interior_outlier": directories["interior_height_outlier_masks"] / f"{sample_id}__interior_height_outlier.png",
            "exterior_candidate": directories["exterior_unstable_candidate_masks"] / f"{sample_id}__exterior_unstable_candidate.png",
            "top_support_ids": directories["selected_top_support_ids"] / f"{sample_id}__selected_top_support_ids.png",
            "height": directories["modeling_building_height_m"] / f"{sample_id}__modeling_building_height_m.npy",
            "height_preview": directories["modeling_building_height_previews"] / f"{sample_id}__modeling_building_height_preview.png",
            "overlay": directories["modeling_overlays"] / f"{sample_id}__modeling_overlay.png",
            "point_npz": directories["building_point_cloud_npz"] / f"{sample_id}__height_cleaned_building_points.npz",
            "point_ply": directories["building_point_cloud_ply_instance"] / f"{sample_id}__height_cleaned_building_instances.ply",
            "layers": directories["height_layer_tables"] / f"{sample_id}__height_layers.json",
            "comparison": directories["comparisons"] / f"{sample_id}__height_guided_instance_cleaning.png",
        }
        save_u16(paths["modeling_ids"], modeling_ids)
        save_rgb(paths["modeling_preview"], modeling_preview)
        save_u16(paths["layer_codes"], stable_layer_codes)
        save_rgb(paths["layer_preview"], layer_preview)
        save_u8(paths["stable_support"], stable_support.astype(np.uint8) * 255)
        save_u8(paths["tail_removed"], tail_removed.astype(np.uint8) * 255)
        save_u8(paths["interior_outlier"], interior_outlier.astype(np.uint8) * 255)
        save_u8(paths["exterior_candidate"], exterior_candidate.astype(np.uint8) * 255)
        save_u16(paths["top_support_ids"], top_support_ids)
        save_float(paths["height"], modeling_height)
        save_rgb(paths["height_preview"], modeling_height_preview)
        save_rgb(paths["overlay"], overlay)
        paths["layers"].write_text(
            json.dumps(instance_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        transform = sample["rasterization"]["pixel_center_index_to_ue_world_xy_3x3"]
        xyz, point_rgb, point_ids, point_provenance = building_point_cloud(
            surface_z, bev_rgb, modeling_ids, provenance, transform
        )
        np.savez_compressed(
            paths["point_npz"],
            xyz_m=xyz,
            rgb=point_rgb,
            instance_id=point_ids,
            provenance=point_provenance,
        )
        ply_meta = write_binary_ply(
            paths["point_ply"],
            xyz,
            instance_palette(int(np.max(point_ids)))[point_ids],
            point_ids,
            args.max_ply_points,
        )

        two_row_comparison(
            f"{sample_id} / E10 HEIGHT-GUIDED INSTANCE CLEANING",
            [
                ("E09 Final IDs", "frozen source mask", original_preview, True),
                ("Provenance-1 height", "direct z-buffer building heights", direct_height_preview, False),
                ("Accepted height layers", "0.5m bins / 2m windows", layer_preview, True),
                ("Stable layer support", "spatially connected support", mask_rgb(stable_support), True),
                ("Exterior unstable", "outside stable layer cores", mask_rgb(exterior_candidate), True),
                ("Tail removed", f"{np.count_nonzero(tail_removed)*PIXEL_AREA_M2:.1f} m2", colored_mask(tail_removed, (245, 73, 85)), True),
            ],
            [
                ("E10 Modeling IDs", f"{np.unique(modeling_ids[modeling_ids>0]).size} IDs", modeling_preview, True),
                ("Removed overlay", "red = removed from modeling mask", rgb_overlay(original_preview, colored_mask(tail_removed, (255, 48, 48)), tail_removed, 0.80), True),
                ("Interior height outlier", "ID kept; height ignored as a layer", colored_mask(interior_outlier, (255, 182, 43)), True),
                ("Selected top support", "highest accepted stable layer", instance_preview(top_support_ids), True),
                ("Cleaned building height", "raw observed Z retained", modeling_height_preview, False),
                ("E10 Overlay", f"points={xyz.shape[0]:,}", overlay, False),
            ],
            paths["comparison"],
            tile=320,
        )
        comparison_paths.append(paths["comparison"])
        records.append(
            {
                "id": sample_id,
                "input_e09_final_ids": sample["outputs"]["final_ids"],
                "input_e09_surface_z": sample["outputs"]["surface_z"],
                "input_e09_provenance": sample["outputs"]["provenance"],
                "input_e09_bev_rgb": sample["outputs"]["bev_rgb"],
                "transform": sample["transform"],
                "rasterization": sample["rasterization"],
                "ground_reference_z_m": ground_reference_z_m,
                "instance_count_before": int(np.unique(original_ids[original_ids > 0]).size),
                "instance_count_after": int(np.unique(modeling_ids[modeling_ids > 0]).size),
                "building_area_before_m2": float(np.count_nonzero(original_ids) * PIXEL_AREA_M2),
                "building_area_after_m2": float(np.count_nonzero(modeling_ids) * PIXEL_AREA_M2),
                "tail_removed_area_m2": float(np.count_nonzero(tail_removed) * PIXEL_AREA_M2),
                "interior_height_outlier_area_m2": float(np.count_nonzero(interior_outlier) * PIXEL_AREA_M2),
                "instances_without_stable_layer": int(sum(not row["stable_layers"] for row in instance_records)),
                "building_point_count": int(xyz.shape[0]),
                "building_ply": ply_meta,
                "outputs": {
                    "final_ids": relative(paths["modeling_ids"]),
                    "final_preview": relative(paths["modeling_preview"]),
                    "stable_height_layer_codes": relative(paths["layer_codes"]),
                    "stable_height_layer_preview": relative(paths["layer_preview"]),
                    "stable_height_support": relative(paths["stable_support"]),
                    "tail_removed": relative(paths["tail_removed"]),
                    "interior_height_outlier": relative(paths["interior_outlier"]),
                    "exterior_unstable_candidate": relative(paths["exterior_candidate"]),
                    "selected_top_support_ids": relative(paths["top_support_ids"]),
                    "surface_z": sample["outputs"]["surface_z"],
                    "provenance": sample["outputs"]["provenance"],
                    "bev_rgb": sample["outputs"]["bev_rgb"],
                    "building_height": relative(paths["height"]),
                    "building_height_preview": relative(paths["height_preview"]),
                    "overlay": relative(paths["overlay"]),
                    "height_layers": relative(paths["layers"]),
                    "building_point_cloud_npz": relative(paths["point_npz"]),
                    "building_point_cloud_ply_instance": relative(paths["point_ply"]),
                    "comparison": relative(paths["comparison"]),
                },
            }
        )

    if existing_manifest is not None:
        updated_by_id = {record["id"]: record for record in records}
        existing_by_id = {
            record["id"]: record for record in existing_manifest["samples"]
        }
        unknown = sorted(set(updated_by_id) - set(existing_by_id))
        if unknown:
            raise KeyError(f"Incremental E10 samples are absent from manifest: {unknown}")
        existing_by_id.update(updated_by_id)
        records = [
            existing_by_id[record["id"]]
            for record in existing_manifest["samples"]
        ]
        comparison_paths = [
            project_path(record["outputs"]["comparison"]) for record in records
        ]
    overview = output_root / f"all_{len(records)}_height_guided_instance_cleaning.png"
    build_overview(comparison_paths, overview)
    manifest = {
        "schema": "air2vectorcity.height_guided_instance_cleaning_e10.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "height-guided modeling-mask experiment",
        "incremental_update": (
            None
            if existing_manifest is None
            else {
                "merged_into_existing_complete_manifest": True,
                "updated_sample_ids": sorted(args.sample_id),
                "preserved_sample_count": len(records) - len(args.sample_id),
            }
        ),
        "input_e09": relative(e09_root),
        "policy": {
            "bev_cell_size_m": CELL_SIZE_M,
            "bev_pixel_area_m2": PIXEL_AREA_M2,
            "authoritative_e09_mask_is_unchanged": True,
            "height_seed_provenance": [1],
            "height_join_provenance": [1, 4],
            "geometry_fill_provenance_cannot_seed_layers": [5],
            "height_bin_m": HEIGHT_BIN_M,
            "height_window_m": HEIGHT_WINDOW_M,
            "height_step_m": HEIGHT_STEP_M,
            "base_layer_fraction": BASE_LAYER_FRACTION,
            "minimum_layer_area_m2": MIN_LAYER_AREA_M2,
            "maximum_layer_area_requirement_m2": MAX_LAYER_AREA_REQUIREMENT_M2,
            "minimum_connected_area_m2": MIN_CONNECTED_AREA_M2,
            "minimum_connected_requirement_fraction": MIN_CONNECTED_REQUIREMENT_FRACTION,
            "strong_minimum_internal_radius_m": STRONG_MIN_INTERNAL_RADIUS_M,
            "conditional_internal_radius_range_m": [
                CONDITIONAL_MIN_INTERNAL_RADIUS_M,
                STRONG_MIN_INTERNAL_RADIUS_M,
            ],
            "conditional_layer_fraction": CONDITIONAL_LAYER_FRACTION,
            "conditional_minimum_area_m2": CONDITIONAL_MIN_AREA_M2,
            "conditional_maximum_area_requirement_m2": CONDITIONAL_MAX_AREA_REQUIREMENT_M2,
            "conditional_connected_fraction": CONDITIONAL_CONNECTED_FRACTION,
            "layer_merge_distance_m": LAYER_MERGE_DISTANCE_M,
            "maximum_stable_layers": MAX_STABLE_LAYERS,
            "layer_assignment_tolerance_m": LAYER_ASSIGNMENT_TOLERANCE_M,
            "tail_maximum_height_slice_radius_m": TAIL_MAX_INTERNAL_RADIUS_M,
            "tail_maximum_neck_width_proxy_m": TAIL_MAX_NECK_WIDTH_M,
            "tail_minimum_height_inconsistent_fraction": TAIL_MIN_INCONSISTENT_FRACTION,
            "tail_maximum_area_fraction": TAIL_MAX_AREA_FRACTION,
            "interior_height_outliers_keep_instance_id": True,
            "height_values_are_not_smoothed": True,
        },
        "overview": relative(overview),
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview)
    print(output_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
