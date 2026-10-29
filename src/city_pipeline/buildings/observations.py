#!/usr/bin/env python3
"""Fuse P320 E10 building observations into predicted global buildings.

The prediction path uses recovered world-space footprints, heights, and quality
metadata only.
"""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import csv
import hashlib
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.strtree import STRtree


PROJECT_ROOT = DATA_ROOT
DEFAULT_RUN_ROOT = Path("UNCONFIGURED_DEFAULT_RUN_ROOT")
DEFAULT_DATASET_ROOT = Path("UNCONFIGURED_DEFAULT_DATASET_ROOT")
DEFAULT_OUTPUT_ROOT = DEFAULT_RUN_ROOT / "global_fused"
UE_Z_UP_TO_GLTF_Y_UP = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
STATUS_RANK = {
    "interior_keep": 3,
    "boundary_risk_review": 2,
    "boundary_clipped_exclude": 1,
}


@dataclass
class Observation:
    index: int
    observation_id: str
    sample_id: str
    patch_id: str
    view_id: str
    instance_id: int
    geometry: Polygon | MultiPolygon
    raster_area_m2: float
    geometry_area_m2: float
    centroid_x_m: float
    centroid_y_m: float
    component_count: int
    base_z_m: float
    top_z_m: float
    height_m: float
    height_method: str
    stable_layer_count: int
    observed_area_m2: float
    selected_support_fraction: float
    boundary_status_code: int
    boundary_status: str
    boundary_evidence: list[str]
    bev_edge_distance_m: float
    coverage_edge_distance_m: float
    source_touches_image_edge: bool
    source_edge_distance_px: int
    source_pixel_count: int
    source_mean_rgb: tuple[float, float, float]
    source_std_rgb: tuple[float, float, float]
    reliable_core_contains_centroid: bool
    reliable_core_edge_distance_m: float
    preliminary_quality: float
    paths: dict[str, str]
    ambiguity_count: int = 0
    canonical_score: float = 0.0
    global_building_id: str = ""
    global_building_index: int = 0


@dataclass
class MatchEdge:
    a: int
    b: int
    sample_pair: tuple[str, str]
    same_patch: bool
    intersection_m2: float
    union_m2: float
    iou: float
    containment: float
    overlap_evidence: float
    aligned_iou: float
    aligned_containment: float
    area_ratio: float
    geometry_distance_m: float
    centroid_distance_m: float
    top_difference_m: float
    height_tolerance_m: float
    color_distance_rgb: float
    color_similarity: float
    match_mode: str
    score: float
    accepted: bool
    selected_one_to_one: bool = False
    cluster_action: str = "not_selected"


@dataclass
class UnionFind:
    count: int
    parent: list[int] = field(init=False)
    members: dict[int, set[int]] = field(init=False)
    samples: dict[int, set[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.parent = list(range(self.count))
        self.members = {index: {index} for index in range(self.count)}
        self.samples = {index: set() for index in range(self.count)}

    def set_sample(self, index: int, sample_id: str) -> None:
        self.samples[index] = {sample_id}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, first: int, second: int) -> int:
        a = self.find(first)
        b = self.find(second)
        if a == b:
            return a
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a].update(self.members.pop(b))
        self.samples[a].update(self.samples.pop(b))
        return a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-buffer-m", type=float, default=12.0)
    parser.add_argument("--minimum-iou", type=float, default=0.12)
    parser.add_argument("--minimum-containment", type=float, default=0.45)
    parser.add_argument("--minimum-area-ratio", type=float, default=0.25)
    parser.add_argument("--maximum-centroid-distance-m", type=float, default=35.0)
    parser.add_argument("--minimum-match-score", type=float, default=0.45)
    parser.add_argument("--minimum-near-shape-iou", type=float, default=0.50)
    parser.add_argument("--minimum-near-color-similarity", type=float, default=0.80)
    parser.add_argument(
        "--enable-strong-overlap-height-override",
        action="store_true",
        help=(
            "Allow exceptionally strong direct footprint overlap to override noisy "
            "top-height disagreement. Intended for real reconstructed scenes without GT."
        ),
    )
    parser.add_argument("--strong-overlap-minimum-iou", type=float, default=0.45)
    parser.add_argument(
        "--strong-overlap-minimum-containment", type=float, default=0.65
    )
    parser.add_argument("--strong-overlap-minimum-area-ratio", type=float, default=0.35)
    parser.add_argument(
        "--strong-overlap-maximum-centroid-distance-m", type=float, default=20.0
    )
    parser.add_argument(
        "--enable-contained-fragment-reconciliation",
        action="store_true",
        help=(
            "After conservative one-to-one clustering, reconcile high-containment "
            "partial observations when doing so preserves the one-sample-per-cluster rule."
        ),
    )
    parser.add_argument("--reconciliation-minimum-iou", type=float, default=0.32)
    parser.add_argument(
        "--reconciliation-minimum-containment", type=float, default=0.75
    )
    parser.add_argument("--reconciliation-minimum-area-ratio", type=float, default=0.30)
    parser.add_argument(
        "--reconciliation-maximum-centroid-distance-m", type=float, default=45.0
    )
    parser.add_argument(
        "--enable-near-cluster-consensus",
        action="store_true",
        help="Allow multi-edge near-shape evidence to merge primary overlap clusters.",
    )
    parser.add_argument("--footprint-simplify-m", type=float, default=0.20)
    parser.add_argument(
        "--enable-low-confidence-fragment-filter",
        action="store_true",
        help=(
            "Exclude a global building from published outputs only when every member "
            "is outside the reliable core, every member is boundary-risk/excluded, "
            "every member lacks a stable height layer, and the canonical footprint is "
            "tiny or highly elongated."
        ),
    )
    parser.add_argument("--fragment-minimum-area-m2", type=float, default=20.0)
    parser.add_argument("--fragment-maximum-elongation", type=float, default=4.5)
    parser.add_argument(
        "--write-broad-fragment-quarantine",
        action="store_true",
        help="Write the broader third-tier candidate sidecar without excluding it.",
    )
    parser.add_argument("--quarantine-maximum-area-m2", type=float, default=75.0)
    parser.add_argument("--quarantine-maximum-elongation", type=float, default=4.5)
    parser.add_argument("--quarantine-minimum-width-m", type=float, default=5.5)
    parser.add_argument("--quarantine-minimum-compactness", type=float, default=0.20)
    parser.add_argument("--skip-glb", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_parquet_if_available(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def split_sample_id(sample_id: str) -> tuple[str, str]:
    patch_id, separator, view_id = sample_id.rpartition("__")
    if not separator or not patch_id or not view_id:
        raise ValueError(f"Unexpected sample ID: {sample_id}")
    return patch_id, view_id


def polygons_only(geometry: Any) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        result: list[Polygon] = []
        for child in geometry.geoms:
            result.extend(polygons_only(child))
        return result
    return []


def transform_points(points_uv: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = points_uv.astype(np.float64)
    u = points[:, 0]
    v = points[:, 1]
    return np.column_stack(
        (
            matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2],
            matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2],
        )
    )


def mask_crop_to_geometry(
    crop: np.ndarray,
    x_offset: int,
    y_offset: int,
    pixel_center_to_world: np.ndarray,
    half_cell_m: float,
    simplify_m: float,
) -> Polygon | MultiPolygon:
    contours, hierarchy = cv2.findContours(
        crop.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours or hierarchy is None:
        raise ValueError("Instance mask contains no contours")
    hierarchy = hierarchy[0]
    polygons: list[Polygon] = []
    for contour_index, contour in enumerate(contours):
        if int(hierarchy[contour_index][3]) != -1:
            continue
        shell_uv = contour[:, 0, :].astype(np.float64)
        if shell_uv.shape[0] < 3:
            continue
        shell_uv[:, 0] += x_offset
        shell_uv[:, 1] += y_offset
        shell = transform_points(shell_uv, pixel_center_to_world)
        holes: list[np.ndarray] = []
        child_index = int(hierarchy[contour_index][2])
        while child_index != -1:
            hole_uv = contours[child_index][:, 0, :].astype(np.float64)
            if hole_uv.shape[0] >= 3:
                hole_uv[:, 0] += x_offset
                hole_uv[:, 1] += y_offset
                holes.append(transform_points(hole_uv, pixel_center_to_world))
            child_index = int(hierarchy[child_index][0])
        candidate = Polygon(shell, holes)
        polygons.extend(polygons_only(make_valid(candidate)))
    if not polygons:
        raise ValueError("Instance mask did not produce a polygon")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
        geometry = make_valid(unary_union(polygons).buffer(half_cell_m, join_style=2))
    polygon_parts = polygons_only(geometry)
    if not polygon_parts:
        raise ValueError("Instance footprint became empty during repair")
    geometry = unary_union(polygon_parts)
    if simplify_m > 0:
        geometry = make_valid(geometry.simplify(simplify_m, preserve_topology=True))
    polygon_parts = polygons_only(geometry)
    if not polygon_parts:
        raise ValueError("Instance footprint became empty during simplification")
    result = unary_union(polygon_parts)
    if not result.is_valid:
        raise AssertionError("Re-polygonized footprint is invalid")
    return result


def preliminary_quality(
    status: dict[str, Any],
    height: dict[str, Any],
    geometry_area_m2: float,
    component_count: int,
    view_id: str,
) -> float:
    status_score = {
        "interior_keep": 55.0,
        "boundary_risk_review": 32.0,
        "boundary_clipped_exclude": 5.0,
    }.get(status["status"], 0.0)
    bev_distance = min(float(status.get("bev_canvas_edge_distance_m", 0.0)), 20.0)
    coverage_distance = min(
        float(status.get("coverage_hull_edge_distance_m", 0.0)), 20.0
    )
    source_view = status.get("source_view", {})
    source_edge_distance_px = source_view.get("largest_component_edge_distance_px")
    source_edge_distance_px = (
        float(source_edge_distance_px) if source_edge_distance_px is not None else 0.0
    )
    source_edge_score = 10.0 * min(source_edge_distance_px / 128.0, 1.0)
    stable_score = 8.0 if int(height.get("stable_layer_count", 0)) > 0 else 2.0
    observed_fraction = min(
        1.0,
        float(height.get("observed_area_m2", 0.0)) / max(geometry_area_m2, 1e-6),
    )
    component_penalty = min(max(component_count - 1, 0) * 2.0, 8.0)
    view_score = 2.0 if view_id == "tilt_light" else 0.0
    return float(
        status_score
        + 0.25 * bev_distance
        + 0.25 * coverage_distance
        + source_edge_score
        + stable_score
        + 8.0 * observed_fraction
        + view_score
        - component_penalty
    )


def load_observations(
    run_root: Path,
    dataset_root: Path,
    simplify_m: float,
) -> tuple[list[Observation], dict[str, Any]]:
    e10_root = run_root / "e10_height_guided"
    e10_manifest = read_json(e10_root / "manifest.json")
    e00_manifest = read_json(run_root / "e00_instance_aware_25d" / "manifest.json")
    prism_manifest = read_json(e10_root / "building_prisms" / "manifest.json")
    boundary_manifest = read_json(e10_root / "boundary_instance_tagging" / "manifest.json")
    e00_by_id = {sample["id"]: sample for sample in e00_manifest["samples"]}
    prism_by_id = {sample["id"]: sample for sample in prism_manifest["samples"]}
    boundary_by_id = {sample["id"]: sample for sample in boundary_manifest["samples"]}
    observations: list[Observation] = []
    sample_cache: dict[str, Any] = {}
    patch_core_bounds: dict[str, tuple[float, float, float, float]] = {}
    explicit_core_patch_count = 0
    for patch_dir in sorted(
        path for path in (dataset_root / "patches").iterdir() if path.is_dir()
    ):
        patch = read_json(patch_dir / "patch.json")
        bounds = patch.get("core_bounds_xy_world_m")
        if bounds is not None:
            explicit_core_patch_count += 1
        else:
            bounds = patch["bounds_xy_world_m"]
        patch_core_bounds[patch_dir.name] = tuple(float(value) for value in bounds)
    core_contract_active = explicit_core_patch_count > 0

    for sample_number, sample in enumerate(e10_manifest["samples"], start=1):
        sample_id = sample["id"]
        patch_id, view_id = split_sample_id(sample_id)
        print(f"[observations {sample_number:03d}/{len(e10_manifest['samples'])}] {sample_id}", flush=True)
        e00_sample = e00_by_id[sample_id]
        prism_sample = prism_by_id[sample_id]
        boundary_sample = boundary_by_id[sample_id]
        final_ids_path = project_path(sample["outputs"]["final_ids"])
        final_ids = np.asarray(Image.open(final_ids_path), dtype=np.uint16)
        source_ids_path = project_path(e00_sample["outputs"]["source_instance_ids"])
        source_ids = np.asarray(Image.open(source_ids_path), dtype=np.uint16)
        source_rgb = np.asarray(
            Image.open(project_path(e00_sample["input"]["rgb"])).convert("RGB"),
            dtype=np.uint8,
        )
        if final_ids.shape != source_ids.shape:
            raise AssertionError(f"Source/BEV shape mismatch: {sample_id}")
        if source_rgb.shape[:2] != source_ids.shape:
            raise AssertionError(f"Source RGB/Instance shape mismatch: {sample_id}")
        cell_sizes = sample["rasterization"]["meters_per_pixel"]
        if not math.isclose(float(cell_sizes[0]), float(cell_sizes[1]), abs_tol=1e-9):
            raise AssertionError(f"Non-square BEV cell: {sample_id}")
        cell_size_m = float(cell_sizes[0])
        pixel_area_m2 = cell_size_m * cell_size_m
        transform = np.asarray(
            sample["rasterization"]["pixel_center_index_to_ue_world_xy_3x3"],
            dtype=np.float64,
        )
        height_rows = read_json(project_path(prism_sample["outputs"]["height_json"]))
        status_rows = read_json(
            project_path(boundary_sample["outputs"]["instance_status_json"])
        )
        height_by_id = {int(row["instance_id"]): row for row in height_rows}
        status_by_id = {int(row["instance_id"]): row for row in status_rows}
        max_id = int(final_ids.max())
        slices = ndimage.find_objects(final_ids, max_label=max_id)
        source_counts = np.bincount(source_ids.reshape(-1), minlength=max_id + 1)
        source_flat = source_ids.reshape(-1)
        source_rgb_flat = source_rgb.reshape(-1, 3).astype(np.float64)
        source_channel_sums = np.stack(
            [
                np.bincount(
                    source_flat,
                    weights=source_rgb_flat[:, channel],
                    minlength=max_id + 1,
                )
                for channel in range(3)
            ],
            axis=1,
        )
        source_channel_square_sums = np.stack(
            [
                np.bincount(
                    source_flat,
                    weights=source_rgb_flat[:, channel] ** 2,
                    minlength=max_id + 1,
                )
                for channel in range(3)
            ],
            axis=1,
        )
        instance_ids = [int(value) for value in np.unique(final_ids) if value != 0]
        if set(instance_ids) != set(height_by_id) or set(instance_ids) != set(status_by_id):
            raise AssertionError(f"E10 height/status mismatch: {sample_id}")
        sample_cache[sample_id] = {
            "patch_id": patch_id,
            "view_id": view_id,
            "rgb": e00_sample["input"]["rgb"],
            "source_instance_ids": e00_sample["outputs"]["source_instance_ids"],
            "bev_instance_ids": sample["outputs"]["final_ids"],
            "building_point_cloud_npz": sample["outputs"]["building_point_cloud_npz"],
            "transform": sample["transform"],
        }
        for instance_id in instance_ids:
            instance_slice = slices[instance_id - 1]
            if instance_slice is None:
                raise AssertionError(f"Missing label slice: {sample_id} {instance_id}")
            y_slice, x_slice = instance_slice
            y0 = max(0, y_slice.start - 1)
            y1 = min(final_ids.shape[0], y_slice.stop + 1)
            x0 = max(0, x_slice.start - 1)
            x1 = min(final_ids.shape[1], x_slice.stop + 1)
            crop = final_ids[y0:y1, x0:x1] == instance_id
            geometry = mask_crop_to_geometry(
                crop,
                x0,
                y0,
                transform,
                cell_size_m / 2.0,
                simplify_m,
            )
            raster_area_m2 = float(np.count_nonzero(crop) * pixel_area_m2)
            status = status_by_id[instance_id]
            height = height_by_id[instance_id]
            source_view = status.get("source_view", {})
            observation_id = f"{sample_id}__instance_{instance_id:04d}"
            source_count = (
                int(source_counts[instance_id]) if instance_id < source_counts.size else 0
            )
            if source_count:
                mean_rgb = source_channel_sums[instance_id] / source_count
                variance_rgb = np.maximum(
                    source_channel_square_sums[instance_id] / source_count - mean_rgb**2,
                    0.0,
                )
                std_rgb = np.sqrt(variance_rgb)
            else:
                mean_rgb = np.zeros(3, dtype=np.float64)
                std_rgb = np.zeros(3, dtype=np.float64)
            centroid = geometry.centroid
            if patch_id not in patch_core_bounds:
                raise KeyError(f"Patch metadata missing for observation: {patch_id}")
            core_min_x, core_min_y, core_max_x, core_max_y = patch_core_bounds[patch_id]
            core_contains_centroid = bool(
                core_min_x <= centroid.x <= core_max_x
                and core_min_y <= centroid.y <= core_max_y
            )
            if not core_contract_active:
                core_contains_centroid = True
                core_edge_distance_m = 0.0
            elif core_contains_centroid:
                core_edge_distance_m = float(
                    min(
                        centroid.x - core_min_x,
                        core_max_x - centroid.x,
                        centroid.y - core_min_y,
                        core_max_y - centroid.y,
                    )
                )
            else:
                dx = max(core_min_x - centroid.x, 0.0, centroid.x - core_max_x)
                dy = max(core_min_y - centroid.y, 0.0, centroid.y - core_max_y)
                core_edge_distance_m = -float(math.hypot(dx, dy))
            observations.append(
                Observation(
                    index=len(observations),
                    observation_id=observation_id,
                    sample_id=sample_id,
                    patch_id=patch_id,
                    view_id=view_id,
                    instance_id=instance_id,
                    geometry=geometry,
                    raster_area_m2=raster_area_m2,
                    geometry_area_m2=float(geometry.area),
                    centroid_x_m=float(centroid.x),
                    centroid_y_m=float(centroid.y),
                    component_count=len(polygons_only(geometry)),
                    base_z_m=float(height["base_z_m"]),
                    top_z_m=float(height["top_z_m"]),
                    height_m=float(height["extrusion_height_m"]),
                    height_method=str(height["method"]),
                    stable_layer_count=int(height.get("stable_layer_count", 0)),
                    observed_area_m2=float(height.get("observed_area_m2", 0.0)),
                    selected_support_fraction=float(
                        height.get("selected_support_fraction", 0.0)
                    ),
                    boundary_status_code=int(status["status_code"]),
                    boundary_status=str(status["status"]),
                    boundary_evidence=list(status.get("evidence", [])),
                    bev_edge_distance_m=float(
                        status.get("bev_canvas_edge_distance_m", 0.0)
                    ),
                    coverage_edge_distance_m=float(
                        status.get("coverage_hull_edge_distance_m", 0.0)
                    ),
                    source_touches_image_edge=bool(
                        source_view.get("largest_component_touches_image_edge", False)
                    ),
                    source_edge_distance_px=int(
                        source_view.get("largest_component_edge_distance_px") or 0
                    ),
                    source_pixel_count=source_count,
                    source_mean_rgb=tuple(float(value) for value in mean_rgb),
                    source_std_rgb=tuple(float(value) for value in std_rgb),
                    reliable_core_contains_centroid=core_contains_centroid,
                    reliable_core_edge_distance_m=core_edge_distance_m,
                    preliminary_quality=preliminary_quality(
                        status,
                        height,
                        float(geometry.area),
                        len(polygons_only(geometry)),
                        view_id,
                    ),
                    paths={
                        "rgb": e00_sample["input"]["rgb"],
                        "source_instance_ids": e00_sample["outputs"]["source_instance_ids"],
                        "bev_instance_ids": sample["outputs"]["final_ids"],
                        "building_point_cloud_npz": sample["outputs"][
                            "building_point_cloud_npz"
                        ],
                        "sample_prism_glb": prism_sample["outputs"]["glb"],
                        "sample_prism_glb_ue_z_up": prism_sample["outputs"]["glb_ue_z_up"],
                    },
                )
            )
    if [observation.index for observation in observations] != list(range(len(observations))):
        raise AssertionError("Observation indices are not dense")
    context = {
        "e10_manifest": e10_manifest,
        "sample_cache": sample_cache,
        "e10_root": e10_root,
        "patch_core_bounds": patch_core_bounds,
        "explicit_core_patch_count": explicit_core_patch_count,
    }
    return observations, context


def edge_metrics(
    first: Observation,
    second: Observation,
    args: argparse.Namespace,
) -> MatchEdge | None:
    intersection = first.geometry.intersection(second.geometry)
    intersection_area = float(intersection.area)
    union_area = first.geometry_area_m2 + second.geometry_area_m2 - intersection_area
    smaller_area = min(first.geometry_area_m2, second.geometry_area_m2)
    larger_area = max(first.geometry_area_m2, second.geometry_area_m2)
    iou = intersection_area / max(union_area, 1e-6)
    containment = intersection_area / max(smaller_area, 1e-6)
    area_ratio = smaller_area / max(larger_area, 1e-6)
    if args.enable_near_cluster_consensus:
        second_aligned = translate(
            second.geometry,
            xoff=first.centroid_x_m - second.centroid_x_m,
            yoff=first.centroid_y_m - second.centroid_y_m,
        )
        aligned_intersection_area = float(
            first.geometry.intersection(second_aligned).area
        )
        aligned_union_area = (
            first.geometry_area_m2 + second.geometry_area_m2 - aligned_intersection_area
        )
        aligned_iou = aligned_intersection_area / max(aligned_union_area, 1e-6)
        aligned_containment = aligned_intersection_area / max(smaller_area, 1e-6)
    else:
        aligned_iou = iou
        aligned_containment = containment
    geometry_distance = float(first.geometry.distance(second.geometry))
    centroid_distance = float(first.geometry.centroid.distance(second.geometry.centroid))
    top_difference = abs(first.top_z_m - second.top_z_m)
    fallback = first.stable_layer_count == 0 or second.stable_layer_count == 0
    height_tolerance = max(
        12.0 if not fallback else 18.0,
        (0.25 if not fallback else 0.35) * max(first.height_m, second.height_m),
    )
    overlap_evidence = max(iou, containment * math.sqrt(max(area_ratio, 0.0)))
    height_similarity = max(0.0, 1.0 - top_difference / max(height_tolerance, 1e-6))
    centroid_similarity = max(
        0.0,
        1.0 - centroid_distance / max(args.maximum_centroid_distance_m, 1e-6),
    )
    overlap_score = (
        0.60 * overlap_evidence
        + 0.15 * area_ratio
        + 0.15 * height_similarity
        + 0.10 * centroid_similarity
    )
    color_distance = float(
        np.linalg.norm(
            np.asarray(first.source_mean_rgb, dtype=np.float64)
            - np.asarray(second.source_mean_rgb, dtype=np.float64)
        )
    )
    color_similarity = math.exp(-((color_distance / 90.0) ** 2))
    near_distance_limit = max(
        4.0,
        min(20.0, 0.45 * math.sqrt(max(larger_area, 1.0))),
    )
    geometry_distance_similarity = max(
        0.0, 1.0 - geometry_distance / max(near_distance_limit, 1e-6)
    )
    aligned_shape_quality = max(
        aligned_iou,
        aligned_containment * math.sqrt(max(area_ratio, 0.0)),
    )
    near_score = (
        0.35 * aligned_shape_quality
        + 0.15 * area_ratio
        + 0.15 * height_similarity
        + 0.15 * centroid_similarity
        + 0.10 * geometry_distance_similarity
        + 0.10 * color_similarity
    )
    same_patch = first.patch_id == second.patch_id
    if same_patch:
        overlap_score += 0.03
        near_score += 0.03
    smaller = first if first.geometry_area_m2 <= second.geometry_area_m2 else second
    partial_allowed = smaller.boundary_status != "interior_keep"
    area_gate = area_ratio >= args.minimum_area_ratio or partial_allowed or iou >= 0.35
    overlap_accepted = bool(
        intersection_area > 0.0
        and (iou >= args.minimum_iou or containment >= args.minimum_containment)
        and area_gate
        and centroid_distance <= args.maximum_centroid_distance_m
        and top_difference <= height_tolerance
        and overlap_score >= args.minimum_match_score
    )
    strong_overlap_accepted = bool(
        not overlap_accepted
        and args.enable_strong_overlap_height_override
        and intersection_area > 0.0
        and iou >= args.strong_overlap_minimum_iou
        and containment >= args.strong_overlap_minimum_containment
        and area_ratio >= args.strong_overlap_minimum_area_ratio
        and centroid_distance <= args.strong_overlap_maximum_centroid_distance_m
        and overlap_score >= args.minimum_match_score
    )
    contained_fragment_accepted = bool(
        not overlap_accepted
        and not strong_overlap_accepted
        and args.enable_contained_fragment_reconciliation
        and intersection_area > 0.0
        and iou >= args.reconciliation_minimum_iou
        and containment >= args.reconciliation_minimum_containment
        and area_ratio >= args.reconciliation_minimum_area_ratio
        and centroid_distance <= args.reconciliation_maximum_centroid_distance_m
        and overlap_score >= args.minimum_match_score
    )
    near_accepted = bool(
        not overlap_accepted
        and not strong_overlap_accepted
        and not contained_fragment_accepted
        and args.enable_near_cluster_consensus
        and geometry_distance <= near_distance_limit
        and centroid_distance <= args.maximum_centroid_distance_m
        and aligned_iou >= args.minimum_near_shape_iou
        and area_gate
        and top_difference <= height_tolerance
        and color_similarity >= args.minimum_near_color_similarity
        and near_score >= args.minimum_match_score
    )
    accepted = (
        overlap_accepted
        or strong_overlap_accepted
        or contained_fragment_accepted
        or near_accepted
    )
    if overlap_accepted:
        score = overlap_score
    elif strong_overlap_accepted:
        score = overlap_score
    elif contained_fragment_accepted:
        score = overlap_score
    elif near_accepted:
        score = near_score
    else:
        score = max(overlap_score, near_score)
    if overlap_accepted:
        match_mode = "overlap"
    elif strong_overlap_accepted:
        match_mode = "strong_overlap_height_override"
    elif contained_fragment_accepted:
        match_mode = "contained_fragment_reconciliation"
    elif near_accepted:
        match_mode = "near_aligned_shape"
    else:
        match_mode = "rejected"
    return MatchEdge(
        a=first.index,
        b=second.index,
        sample_pair=tuple(sorted((first.sample_id, second.sample_id))),
        same_patch=same_patch,
        intersection_m2=intersection_area,
        union_m2=union_area,
        iou=iou,
        containment=containment,
        overlap_evidence=overlap_evidence,
        aligned_iou=aligned_iou,
        aligned_containment=aligned_containment,
        area_ratio=area_ratio,
        geometry_distance_m=geometry_distance,
        centroid_distance_m=centroid_distance,
        top_difference_m=top_difference,
        height_tolerance_m=height_tolerance,
        color_distance_rgb=color_distance,
        color_similarity=color_similarity,
        match_mode=match_mode,
        score=score,
        accepted=accepted,
    )


def build_candidate_edges(
    observations: list[Observation], args: argparse.Namespace
) -> list[MatchEdge]:
    geometries = [observation.geometry for observation in observations]
    tree = STRtree(geometries)
    edges: list[MatchEdge] = []
    for first_index, observation in enumerate(observations):
        if first_index % 250 == 0:
            print(f"[matching candidates] {first_index}/{len(observations)}", flush=True)
        query_geometry = (
            observation.geometry.buffer(args.candidate_buffer_m)
            if args.enable_near_cluster_consensus
            else observation.geometry
        )
        candidate_indices = tree.query(query_geometry)
        for second_index_value in candidate_indices:
            second_index = int(second_index_value)
            if second_index <= first_index:
                continue
            second = observations[second_index]
            if observation.sample_id == second.sample_id:
                continue
            edge = edge_metrics(observation, second, args)
            if edge is not None:
                edges.append(edge)
    return edges


def choose_one_to_one_edges(
    edges: list[MatchEdge], observations: list[Observation], minimum_score: float
) -> tuple[list[MatchEdge], list[dict[str, Any]]]:
    accepted_by_pair: dict[tuple[str, str], list[MatchEdge]] = defaultdict(list)
    for edge in edges:
        if edge.accepted and edge.match_mode in {
            "overlap",
            "strong_overlap_height_override",
        }:
            accepted_by_pair[edge.sample_pair].append(edge)
    selected: list[MatchEdge] = []
    ambiguity_records: list[dict[str, Any]] = []
    for sample_pair, pair_edges in sorted(accepted_by_pair.items()):
        first_sample, second_sample = sample_pair
        first_indices = sorted(
            {
                edge.a if observations[edge.a].sample_id == first_sample else edge.b
                for edge in pair_edges
            }
        )
        second_indices = sorted(
            {
                edge.a if observations[edge.a].sample_id == second_sample else edge.b
                for edge in pair_edges
            }
        )
        first_lookup = {value: index for index, value in enumerate(first_indices)}
        second_lookup = {value: index for index, value in enumerate(second_indices)}
        matrix = np.full((len(first_indices), len(second_indices)), -1e6, dtype=np.float64)
        edge_lookup: dict[tuple[int, int], MatchEdge] = {}
        degree_first: dict[int, int] = defaultdict(int)
        degree_second: dict[int, int] = defaultdict(int)
        for edge in pair_edges:
            if observations[edge.a].sample_id == first_sample:
                first_index, second_index = edge.a, edge.b
            else:
                first_index, second_index = edge.b, edge.a
            row = first_lookup[first_index]
            column = second_lookup[second_index]
            if edge.score > matrix[row, column]:
                matrix[row, column] = edge.score
                edge_lookup[(row, column)] = edge
            degree_first[first_index] += 1
            degree_second[second_index] += 1
        rows, columns = linear_sum_assignment(-matrix)
        for row, column in zip(rows, columns):
            if matrix[row, column] < minimum_score:
                continue
            edge = edge_lookup[(int(row), int(column))]
            edge.selected_one_to_one = True
            selected.append(edge)
        ambiguous_indices = {
            index for index, degree in degree_first.items() if degree > 1
        } | {index for index, degree in degree_second.items() if degree > 1}
        if ambiguous_indices:
            for index in ambiguous_indices:
                observations[index].ambiguity_count += 1
            ambiguity_records.append(
                {
                    "sample_pair": list(sample_pair),
                    "ambiguous_observation_ids": [
                        observations[index].observation_id
                        for index in sorted(ambiguous_indices)
                    ],
                    "candidate_edges": [
                        {
                            "a": observations[edge.a].observation_id,
                            "b": observations[edge.b].observation_id,
                            "score": edge.score,
                            "iou": edge.iou,
                            "containment": edge.containment,
                        }
                        for edge in sorted(pair_edges, key=lambda item: -item.score)
                        if edge.a in ambiguous_indices or edge.b in ambiguous_indices
                    ],
                }
            )
    return selected, ambiguity_records


def cluster_observations(
    observations: list[Observation],
    selected_edges: list[MatchEdge],
    near_edges: list[MatchEdge],
    reconciliation_edges: list[MatchEdge],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    union_find = UnionFind(len(observations))
    for observation in observations:
        union_find.set_sample(observation.index, observation.sample_id)
    rejected: list[dict[str, Any]] = []
    for edge in sorted(selected_edges, key=lambda item: (-item.score, -item.iou)):
        first_root = union_find.find(edge.a)
        second_root = union_find.find(edge.b)
        if first_root == second_root:
            edge.cluster_action = "already_joined"
            continue
        repeated_samples = union_find.samples[first_root] & union_find.samples[second_root]
        if repeated_samples:
            edge.cluster_action = "rejected_repeated_sample"
            rejected.append(
                {
                    "reason": edge.cluster_action,
                    "a": observations[edge.a].observation_id,
                    "b": observations[edge.b].observation_id,
                    "score": edge.score,
                    "repeated_samples": sorted(repeated_samples),
                }
            )
            continue
        first_tops = [observations[index].top_z_m for index in union_find.members[first_root]]
        second_tops = [observations[index].top_z_m for index in union_find.members[second_root]]
        median_difference = abs(float(np.median(first_tops)) - float(np.median(second_tops)))
        if (
            edge.match_mode != "strong_overlap_height_override"
            and median_difference > max(edge.height_tolerance_m, 20.0)
        ):
            edge.cluster_action = "rejected_cluster_height"
            rejected.append(
                {
                    "reason": edge.cluster_action,
                    "a": observations[edge.a].observation_id,
                    "b": observations[edge.b].observation_id,
                    "score": edge.score,
                    "cluster_top_median_difference_m": median_difference,
                }
            )
            continue
        union_find.union(first_root, second_root)
        edge.cluster_action = "merged"

    near_by_cluster_pair: dict[tuple[int, int], list[MatchEdge]] = defaultdict(list)
    for edge in near_edges:
        first_root = union_find.find(edge.a)
        second_root = union_find.find(edge.b)
        if first_root == second_root:
            edge.cluster_action = "near_already_joined"
            continue
        near_by_cluster_pair[tuple(sorted((first_root, second_root)))].append(edge)
    near_groups = sorted(
        near_by_cluster_pair.values(),
        key=lambda group: (-len(group), -float(np.mean([edge.score for edge in group]))),
    )
    for original_group in near_groups:
        current_groups: dict[tuple[int, int], list[MatchEdge]] = defaultdict(list)
        for edge in original_group:
            first_root = union_find.find(edge.a)
            second_root = union_find.find(edge.b)
            if first_root == second_root:
                edge.cluster_action = "near_already_joined"
                continue
            current_groups[tuple(sorted((first_root, second_root)))].append(edge)
        for (first_root, second_root), group in current_groups.items():
            sample_pairs = {edge.sample_pair for edge in group}
            first_endpoints: set[int] = set()
            second_endpoints: set[int] = set()
            for edge in group:
                if union_find.find(edge.a) == first_root:
                    first_endpoints.add(edge.a)
                    second_endpoints.add(edge.b)
                else:
                    first_endpoints.add(edge.b)
                    second_endpoints.add(edge.a)
            independent_support = (
                len(group) >= 2
                and len(sample_pairs) >= 2
                and len(first_endpoints) + len(second_endpoints) >= 3
            )
            if not independent_support:
                for edge in group:
                    edge.cluster_action = "near_rejected_insufficient_consensus"
                continue
            repeated_samples = union_find.samples[first_root] & union_find.samples[second_root]
            if repeated_samples:
                for edge in group:
                    edge.cluster_action = "near_rejected_repeated_sample"
                rejected.append(
                    {
                        "reason": "near_rejected_repeated_sample",
                        "support_edge_count": len(group),
                        "repeated_samples": sorted(repeated_samples),
                    }
                )
                continue
            first_tops = [
                observations[index].top_z_m for index in union_find.members[first_root]
            ]
            second_tops = [
                observations[index].top_z_m for index in union_find.members[second_root]
            ]
            median_difference = abs(
                float(np.median(first_tops)) - float(np.median(second_tops))
            )
            group_tolerance = max(
                20.0,
                float(np.median([edge.height_tolerance_m for edge in group])),
            )
            if median_difference > group_tolerance:
                for edge in group:
                    edge.cluster_action = "near_rejected_cluster_height"
                continue
            union_find.union(first_root, second_root)
            for edge in group:
                edge.cluster_action = "near_consensus_merged"

    # Run this after the conservative one-to-one and consensus stages so extra
    # partial-view evidence cannot change their assignments.  The repeated-sample
    # guard prevents a broad mask from swallowing two buildings seen separately
    # in any one source view.
    for edge in sorted(
        reconciliation_edges,
        key=lambda item: (-item.overlap_evidence, -item.containment, -item.iou),
    ):
        first_root = union_find.find(edge.a)
        second_root = union_find.find(edge.b)
        if first_root == second_root:
            edge.cluster_action = "reconciliation_already_joined"
            continue
        repeated_samples = union_find.samples[first_root] & union_find.samples[second_root]
        if repeated_samples:
            edge.cluster_action = "reconciliation_rejected_repeated_sample"
            rejected.append(
                {
                    "reason": edge.cluster_action,
                    "a": observations[edge.a].observation_id,
                    "b": observations[edge.b].observation_id,
                    "score": edge.score,
                    "repeated_samples": sorted(repeated_samples),
                }
            )
            continue
        union_find.union(first_root, second_root)
        edge.cluster_action = "contained_fragment_reconciled"
    roots = sorted({union_find.find(index) for index in range(len(observations))})
    clusters = [sorted(union_find.members[root]) for root in roots]
    return clusters, rejected


def choose_canonical(
    member_indices: list[int],
    observations: list[Observation],
    edge_lookup: dict[tuple[int, int], MatchEdge],
) -> int:
    areas = np.asarray(
        [observations[index].geometry_area_m2 for index in member_indices],
        dtype=np.float64,
    )
    median_area = float(np.median(areas))
    for index in member_indices:
        observation = observations[index]
        pair_scores: list[float] = []
        for other_index in member_indices:
            if other_index == index:
                continue
            edge = edge_lookup.get(tuple(sorted((index, other_index))))
            if edge is not None:
                pair_scores.append(edge.score)
        consistency = float(np.mean(pair_scores)) if pair_scores else 0.5
        area_consistency = math.exp(
            -abs(math.log(max(observation.geometry_area_m2, 1e-6) / max(median_area, 1e-6)))
        )
        ambiguity_penalty = min(observation.ambiguity_count * 4.0, 16.0)
        observation.canonical_score = float(
            observation.preliminary_quality
            + 15.0 * consistency
            + 10.0 * area_consistency
            - ambiguity_penalty
        )
    return max(
        member_indices,
        key=lambda index: (
            observations[index].reliable_core_contains_centroid,
            observations[index].canonical_score,
            STATUS_RANK.get(observations[index].boundary_status, 0),
            observations[index].source_pixel_count,
            observations[index].observation_id,
        ),
    )


def assign_global_ids(
    clusters: list[list[int]],
    observations: list[Observation],
    edge_lookup: dict[tuple[int, int], MatchEdge],
) -> list[dict[str, Any]]:
    provisional: list[dict[str, Any]] = []
    for members in clusters:
        canonical_index = choose_canonical(members, observations, edge_lookup)
        signature = hashlib.sha256(
            "\n".join(sorted(observations[index].observation_id for index in members)).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        cluster_centroid_x_m = float(
            np.median([observations[index].centroid_x_m for index in members])
        )
        cluster_centroid_y_m = float(
            np.median([observations[index].centroid_y_m for index in members])
        )
        provisional.append(
            {
                "members": members,
                "canonical_index": canonical_index,
                "sort_key": (
                    round(cluster_centroid_x_m, 3),
                    round(cluster_centroid_y_m, 3),
                    signature,
                ),
                "cluster_centroid_x_m": cluster_centroid_x_m,
                "cluster_centroid_y_m": cluster_centroid_y_m,
                "membership_signature": signature,
            }
        )
    provisional.sort(key=lambda record: record["sort_key"])
    for global_index, record in enumerate(provisional, start=1):
        global_id = f"PGB_SC_{global_index:06d}"
        record["global_building_id"] = global_id
        record["global_building_index"] = global_index
        for member_index in record["members"]:
            observations[member_index].global_building_id = global_id
            observations[member_index].global_building_index = global_index
    return provisional


def footprint_shape_metrics(geometry: Polygon | MultiPolygon) -> dict[str, float]:
    """Return stable footprint shape metrics used only for publication filtering."""

    rectangle = geometry.minimum_rotated_rectangle
    coordinates = np.asarray(rectangle.exterior.coords, dtype=np.float64)
    edge_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    positive = edge_lengths[edge_lengths > 1e-9]
    if positive.size:
        length_m = float(np.max(positive))
        width_m = float(np.min(positive))
    else:
        length_m = 0.0
        width_m = 0.0
    area_m2 = float(geometry.area)
    perimeter_m = float(geometry.length)
    return {
        "area_m2": area_m2,
        "rotated_length_m": length_m,
        "rotated_width_m": width_m,
        "elongation": length_m / max(width_m, 1e-9),
        "compactness": (
            4.0 * math.pi * area_m2 / max(perimeter_m * perimeter_m, 1e-9)
        ),
    }


def apply_global_publication_policy(
    global_records: list[dict[str, Any]],
    observations: list[Observation],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify records after stable IDs are assigned, preserving ID traceability."""

    published: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    risky_statuses = {"boundary_risk_review", "boundary_clipped_exclude"}
    for record in global_records:
        canonical = observations[record["canonical_index"]]
        members = [observations[index] for index in record["members"]]
        metrics = footprint_shape_metrics(canonical.geometry)
        all_outside_core = all(
            not member.reliable_core_contains_centroid for member in members
        )
        all_boundary_risk_or_exclude = all(
            member.boundary_status in risky_statuses for member in members
        )
        all_without_stable_layer = all(
            member.stable_layer_count <= 0 for member in members
        )
        evidence_gate = (
            all_outside_core
            and all_boundary_risk_or_exclude
            and all_without_stable_layer
        )
        aggressive_shape_gate = (
            metrics["area_m2"] < args.fragment_minimum_area_m2
            or metrics["elongation"] > args.fragment_maximum_elongation
        )
        auto_removed = bool(
            args.enable_low_confidence_fragment_filter
            and evidence_gate
            and aggressive_shape_gate
        )
        broad_shape_gate = (
            metrics["area_m2"] < args.quarantine_maximum_area_m2
            or metrics["elongation"] > args.quarantine_maximum_elongation
            or metrics["rotated_width_m"] < args.quarantine_minimum_width_m
            or metrics["compactness"] < args.quarantine_minimum_compactness
        )
        broad_quarantine_candidate = bool(
            args.write_broad_fragment_quarantine
            and evidence_gate
            and broad_shape_gate
        )
        reason_parts: list[str] = []
        if metrics["area_m2"] < args.fragment_minimum_area_m2:
            reason_parts.append("area_below_threshold")
        if metrics["elongation"] > args.fragment_maximum_elongation:
            reason_parts.append("elongation_above_threshold")
        decision = {
            "predicted_global_building_id": record["global_building_id"],
            "global_building_index": record["global_building_index"],
            "membership_signature": record["membership_signature"],
            "cluster_centroid_x_m": record["cluster_centroid_x_m"],
            "cluster_centroid_y_m": record["cluster_centroid_y_m"],
            "canonical_observation_id": canonical.observation_id,
            "member_observation_ids_json": json.dumps(
                [member.observation_id for member in members], ensure_ascii=False
            ),
            "member_count": len(members),
            "all_members_outside_reliable_core": all_outside_core,
            "all_members_boundary_risk_or_exclude": all_boundary_risk_or_exclude,
            "all_members_without_stable_height_layer": all_without_stable_layer,
            "low_confidence_evidence_gate": evidence_gate,
            **metrics,
            "aggressive_shape_gate": aggressive_shape_gate,
            "auto_removed_from_published_outputs": auto_removed,
            "auto_remove_reason": ";".join(reason_parts) if auto_removed else "",
            "broad_quarantine_candidate": broad_quarantine_candidate,
        }
        record["publication_decision"] = decision
        decisions.append(decision)
        if broad_quarantine_candidate:
            quarantine.append(decision)
        if not auto_removed:
            published.append(record)
    return published, decisions, quarantine


def observation_row(observation: Observation) -> dict[str, Any]:
    return {
        "observation_id": observation.observation_id,
        "predicted_global_building_id": observation.global_building_id,
        "global_building_index": observation.global_building_index,
        "sample_id": observation.sample_id,
        "patch_id": observation.patch_id,
        "view_id": observation.view_id,
        "rgb_instance_id": observation.instance_id,
        "bev_instance_id": observation.instance_id,
        "boundary_status": observation.boundary_status,
        "raster_area_m2": observation.raster_area_m2,
        "repaired_footprint_area_m2": observation.geometry_area_m2,
        "centroid_x_m": observation.centroid_x_m,
        "centroid_y_m": observation.centroid_y_m,
        "base_z_m": observation.base_z_m,
        "top_z_m": observation.top_z_m,
        "height_m": observation.height_m,
        "height_method": observation.height_method,
        "stable_layer_count": observation.stable_layer_count,
        "preliminary_quality": observation.preliminary_quality,
        "canonical_score": observation.canonical_score,
        "ambiguity_count": observation.ambiguity_count,
        "source_pixel_count": observation.source_pixel_count,
        "source_edge_distance_px": observation.source_edge_distance_px,
        "reliable_core_contains_centroid": observation.reliable_core_contains_centroid,
        "reliable_core_edge_distance_m": observation.reliable_core_edge_distance_m,
        "source_mean_rgb_json": json.dumps(observation.source_mean_rgb),
        "source_std_rgb_json": json.dumps(observation.source_std_rgb),
        "rgb_path": observation.paths["rgb"],
        "source_instance_ids_path": observation.paths["source_instance_ids"],
        "bev_instance_ids_path": observation.paths["bev_instance_ids"],
        "building_point_cloud_npz_path": observation.paths["building_point_cloud_npz"],
    }


def edge_row(edge: MatchEdge, observations: list[Observation]) -> dict[str, Any]:
    return {
        "observation_a": observations[edge.a].observation_id,
        "observation_b": observations[edge.b].observation_id,
        "sample_a": observations[edge.a].sample_id,
        "sample_b": observations[edge.b].sample_id,
        "same_patch": edge.same_patch,
        "intersection_m2": edge.intersection_m2,
        "union_m2": edge.union_m2,
        "iou": edge.iou,
        "containment": edge.containment,
        "overlap_evidence": edge.overlap_evidence,
        "aligned_iou": edge.aligned_iou,
        "aligned_containment": edge.aligned_containment,
        "area_ratio": edge.area_ratio,
        "geometry_distance_m": edge.geometry_distance_m,
        "centroid_distance_m": edge.centroid_distance_m,
        "top_difference_m": edge.top_difference_m,
        "height_tolerance_m": edge.height_tolerance_m,
        "color_distance_rgb": edge.color_distance_rgb,
        "color_similarity": edge.color_similarity,
        "match_mode": edge.match_mode,
        "association_score": edge.score,
        "score": edge.score,
        "accepted": edge.accepted,
        "selected_one_to_one": edge.selected_one_to_one,
        "cluster_action": edge.cluster_action,
    }


def create_identity_rasters(
    output_root: Path,
    observations: list[Observation],
    sample_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    source_root = output_root / "source_rgb_global_building_index"
    bev_root = output_root / "bev_global_building_index"
    sample_map_root = output_root / "per_sample"
    for path in (source_root, bev_root, sample_map_root):
        path.mkdir(parents=True, exist_ok=True)
    by_sample: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_sample[observation.sample_id].append(observation)
    sample_records: list[dict[str, Any]] = []
    for sample_id in sorted(sample_cache):
        sample_observations = sorted(by_sample[sample_id], key=lambda item: item.instance_id)
        source_ids = np.asarray(
            Image.open(project_path(sample_cache[sample_id]["source_instance_ids"])),
            dtype=np.uint16,
        )
        bev_ids = np.asarray(
            Image.open(project_path(sample_cache[sample_id]["bev_instance_ids"])),
            dtype=np.uint16,
        )
        maximum_local_id = max(
            int(source_ids.max()),
            int(bev_ids.max()),
            max((item.instance_id for item in sample_observations), default=0),
        )
        lookup = np.zeros(maximum_local_id + 1, dtype=np.uint16)
        for observation in sample_observations:
            lookup[observation.instance_id] = observation.global_building_index
        source_global = lookup[source_ids]
        bev_global = lookup[bev_ids]
        source_path = source_root / f"{sample_id}__source_global_building_index.png"
        bev_path = bev_root / f"{sample_id}__bev_global_building_index.png"
        Image.fromarray(source_global, mode="I;16").save(source_path)
        Image.fromarray(bev_global, mode="I;16").save(bev_path)
        mapping_path = sample_map_root / f"{sample_id}__observation_to_global.json"
        record = {
            "schema": "air2vectorcity.predicted_sample_global_building_map.v1",
            "sample_id": sample_id,
            "patch_id": sample_cache[sample_id]["patch_id"],
            "view_id": sample_cache[sample_id]["view_id"],
            "rgb": sample_cache[sample_id]["rgb"],
            "source_instance_ids": sample_cache[sample_id]["source_instance_ids"],
            "source_global_building_index": relative(source_path),
            "bev_instance_ids": sample_cache[sample_id]["bev_instance_ids"],
            "bev_global_building_index": relative(bev_path),
            "building_point_cloud_npz": sample_cache[sample_id][
                "building_point_cloud_npz"
            ],
            "instance_to_global": [
                {
                    "observation_id": observation.observation_id,
                    "instance_id": observation.instance_id,
                    "global_building_index": observation.global_building_index,
                    "predicted_global_building_id": observation.global_building_id,
                }
                for observation in sample_observations
            ],
        }
        write_json(mapping_path, record)
        sample_records.append(
            {
                "sample_id": sample_id,
                "patch_id": record["patch_id"],
                "view_id": record["view_id"],
                "observation_count": len(sample_observations),
                "mapping": relative(mapping_path),
                "source_global_building_index": relative(source_path),
                "bev_global_building_index": relative(bev_path),
            }
        )
    return sample_records


def deterministic_color(global_id: str) -> np.ndarray:
    digest = hashlib.sha256(global_id.encode("utf-8")).digest()
    return np.asarray([55 + digest[0] % 180, 55 + digest[1] % 180, 55 + digest[2] % 180], dtype=np.uint8)


def cap_geometry(vertices,faces):
    return unary_union([Polygon(t) for t in vertices[faces] if Polygon(t).area>0.])


class _ExactTriangulationMismatch(ValueError):
    """A triangle set does not reproduce the immutable polygon."""


def _validated_exact_triangles(polygon,vertices,faces,label):
    vertices=np.asarray(vertices,np.float64);faces=np.asarray(faces,np.int64)
    # Exact aliases only: no rounding, snapping, simplification or buffering.
    vertices,inv=np.unique(vertices,axis=0,return_inverse=True);faces=inv[faces]
    tri=vertices[faces];cross=(tri[:,1,0]-tri[:,0,0])*(tri[:,2,1]-tri[:,0,1])-(tri[:,1,1]-tri[:,0,1])*(tri[:,2,0]-tri[:,0,0])
    faces=faces[cross!=0.];cross=cross[cross!=0.]
    faces[cross<0.]=faces[cross<0.,::-1]
    cap=cap_geometry(vertices,faces);tol=max(1e-8,polygon.area*1e-10)
    delta=cap.symmetric_difference(polygon).area
    triangle_area=float(np.abs(cross).sum()/2)
    if delta>tol or abs(triangle_area-polygon.area)>tol:
        raise _ExactTriangulationMismatch(f'{label} changed polygon: delta={delta}, area={triangle_area-polygon.area}')
    return vertices,faces


def triangulate_exact(polygon):
    import trimesh
    vertices,faces=trimesh.creation.triangulate_polygon(polygon,engine='earcut')
    try:
        # Every formerly successful path retains the same vertices and indices.
        return _validated_exact_triangles(polygon,vertices,faces,'Earcut')
    except _ExactTriangulationMismatch as error:
        # Touching valid rings can defeat earcut. Triangulate the same polygon
        # with existing GEOS constrained Delaunay; never repair its footprint.
        from shapely import constrained_delaunay_triangles
        triangles=constrained_delaunay_triangles(polygon)
        if triangles.geom_type!='GeometryCollection' or triangles.is_empty:
            raise ValueError('Constrained Delaunay returned no polygon triangles') from error
        coordinates=[]
        for triangle in triangles.geoms:
            if triangle.geom_type!='Polygon' or len(triangle.interiors) or len(triangle.exterior.coords)!=4 or triangle.area<=0:
                raise ValueError('Constrained Delaunay returned a non-triangle') from error
            coordinates.extend(list(triangle.exterior.coords)[:3])
        vertices=np.asarray(coordinates,np.float64)
        source_vertices={tuple(xy) for ring in [polygon.exterior,*polygon.interiors] for xy in ring.coords}
        if any(tuple(xy) not in source_vertices for xy in vertices):
            raise ValueError('Constrained Delaunay introduced a new or moved vertex') from error
        faces=np.arange(len(vertices),dtype=np.int64).reshape(-1,3)
        vertices,faces=_validated_exact_triangles(polygon,vertices,faces,'Constrained Delaunay')
        print(json.dumps({'event':'exact_triangulation_fallback','method':'constrained_delaunay',
            'polygon_wkb_sha256':hashlib.sha256(polygon.wkb).hexdigest(),
            'polygon_area_m2':polygon.area,'holes':len(polygon.interiors),
            'original_failure':str(error),'triangle_count':len(faces)}),flush=True)
        return vertices,faces


def split_boundary_vertex_fans(vertices,faces):
    """Split coincident boundary fans by index, never by spatial displacement.

    Valid polygon rings may touch at a point; earcut can also leave separate
    triangle fans after exact zero-area triangles are discarded. Sharing their
    vertical edge creates a nonmanifold solid. Separate indices retain the
    identical cap geometry while making each boundary fan a closed surface.
    """
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,inv,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edges[counts[inv]==1];ids,degree=np.unique(boundary,return_counts=True)
    vertices=list(vertices);faces=faces.copy()
    for vertex in ids[degree>2]:
        incident=np.flatnonzero(np.any(faces==vertex,axis=1));links={int(i):set() for i in incident};by_other={}
        for i in incident:
            for other in faces[i][faces[i]!=vertex]:by_other.setdefault(int(other),[]).append(int(i))
        for related in by_other.values():
            for i in related:links[i].update(related)
        unseen=set(links);components=[]
        while unseen:
            component=set();pending=[min(unseen)]
            while pending:
                i=pending.pop()
                if i in component:continue
                component.add(i);pending.extend(links[i]-component)
            unseen-=component;components.append(sorted(component))
        for component in components[1:]:
            new=len(vertices);vertices.append(np.asarray(vertices[int(vertex)]).copy())
            for i in component:faces[i,faces[i]==vertex]=new
    return np.asarray(vertices,np.float64),faces


def extrude_polygon_exact(polygon,height):
    import trimesh
    vertices,faces=triangulate_exact(orient(polygon,sign=1.))
    vertices,faces=split_boundary_vertex_fans(vertices,faces)
    used=np.unique(faces);remap=np.full(len(vertices),-1,np.int64);remap[used]=np.arange(len(used));vertices=vertices[used];faces=remap[faces]
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);sorted_edges=np.sort(edges,axis=1)
    _,inv,counts=np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Triangulation nonmanifold edge')
    boundary=edges[counts[inv]==1];n=len(vertices)
    wall=np.concatenate([np.column_stack([boundary[:,0],boundary[:,1],boundary[:,1]+n]),np.column_stack([boundary[:,0],boundary[:,1]+n,boundary[:,0]+n])])
    points=np.concatenate([np.column_stack([vertices,np.zeros(n)]),np.column_stack([vertices,np.full(n,float(height))])])
    mesh=trimesh.Trimesh(vertices=points,faces=np.concatenate([faces[:,::-1],faces+n,wall]),process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Exact indexed extrusion topology failed')
    expected=polygon.area*height
    if abs(mesh.volume-expected)>max(1e-6,expected*1e-9):raise ValueError('Extrusion volume changed')
    return mesh


def extrude_geometry(geometry,base_z_m,top_z_m,color,metadata):
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("GLB export requires the trimesh package") from error
    height=float(top_z_m-base_z_m)
    if not np.isfinite([height,base_z_m,top_z_m]).all() or height<=0.:raise ValueError('Building extrusion height must be finite and positive')
    if not geometry.is_valid or geometry.is_empty:raise ValueError('Extrusion requires valid nonempty immutable footprint')
    if geometry.geom_type=='Polygon':polygons=[geometry]
    elif geometry.geom_type=='MultiPolygon':polygons=list(geometry.geoms)
    else:raise ValueError('Extrusion requires Polygon or MultiPolygon')
    meshes=[]
    for polygon in polygons:
        if polygon.area<=0.:raise ValueError('Degenerate zero-area polygon in frozen footprint')
        mesh=extrude_polygon_exact(polygon,height);mesh.apply_translation((0.,0.,float(base_z_m)));meshes.append(mesh)
    mesh=trimesh.util.concatenate(meshes)
    rgba=np.asarray([color[0],color[1],color[2],255],np.uint8);mesh.visual.face_colors=np.repeat(rgba[None,:],len(mesh.faces),axis=0);mesh.metadata.update(metadata)
    return mesh


def export_global_glbs(
    output_root: Path,
    global_records: list[dict[str, Any]],
    observations: list[Observation],
) -> dict[str, Any]:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("GLB export requires the trimesh package") from error
    per_building_root = output_root / "canonical_building_glb"
    per_building_ue_root = output_root / "canonical_building_glb_ue_z_up"
    per_building_root.mkdir(parents=True, exist_ok=True)
    per_building_ue_root.mkdir(parents=True, exist_ok=True)
    for stale_path in per_building_root.glob("*.glb"):
        stale_path.unlink()
    for stale_path in per_building_ue_root.glob("*.glb"):
        stale_path.unlink()
    scene_ue = trimesh.Scene()
    scene_viewer = trimesh.Scene()
    total_faces = 0
    total_vertices = 0
    for number, record in enumerate(global_records, start=1):
        if number % 100 == 0:
            print(f"[GLB] {number}/{len(global_records)}", flush=True)
        canonical = observations[record["canonical_index"]]
        global_id = record["global_building_id"]
        color = deterministic_color(global_id)
        metadata = {
            "predicted_global_building_id": global_id,
            "canonical_observation_id": canonical.observation_id,
            "global_building_index": record["global_building_index"],
        }
        mesh_ue = extrude_geometry(
            canonical.geometry,
            canonical.base_z_m,
            canonical.top_z_m,
            color,
            metadata,
        )
        mesh_viewer = mesh_ue.copy()
        mesh_viewer.apply_transform(UE_Z_UP_TO_GLTF_Y_UP)
        mesh_ue.metadata.update(metadata)
        mesh_viewer.metadata.update(metadata)
        scene_ue.add_geometry(mesh_ue, node_name=global_id, geom_name=global_id)
        scene_viewer.add_geometry(mesh_viewer, node_name=global_id, geom_name=global_id)
        per_ue_scene = trimesh.Scene()
        per_viewer_scene = trimesh.Scene()
        per_ue_scene.add_geometry(mesh_ue, node_name=global_id, geom_name=global_id)
        per_viewer_scene.add_geometry(mesh_viewer, node_name=global_id, geom_name=global_id)
        ue_path = per_building_ue_root / f"{global_id}.glb"
        viewer_path = per_building_root / f"{global_id}.glb"
        ue_path.write_bytes(per_ue_scene.export(file_type="glb"))
        viewer_path.write_bytes(per_viewer_scene.export(file_type="glb"))
        record["canonical_glb"] = relative(viewer_path)
        record["canonical_glb_ue_z_up"] = relative(ue_path)
        total_faces += int(mesh_ue.faces.shape[0])
        total_vertices += int(mesh_ue.vertices.shape[0])
    full_viewer_path = output_root / "full_city_coarse.glb"
    full_ue_path = output_root / "full_city_coarse_ue_z_up.glb"
    full_viewer_path.write_bytes(scene_viewer.export(file_type="glb"))
    full_ue_path.write_bytes(scene_ue.export(file_type="glb"))
    return {
        "full_city_coarse_glb": relative(full_viewer_path),
        "full_city_coarse_glb_ue_z_up": relative(full_ue_path),
        "canonical_building_glb_root": relative(per_building_root),
        "canonical_building_glb_ue_z_up_root": relative(per_building_ue_root),
        "mesh_count": len(global_records),
        "vertex_count": total_vertices,
        "face_count": total_faces,
    }


def build_global_rows_and_geojson(
    global_records: list[dict[str, Any]], observations: list[Observation]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for record in global_records:
        canonical = observations[record["canonical_index"]]
        members = [observations[index] for index in record["members"]]
        row = {
            "predicted_global_building_id": record["global_building_id"],
            "global_building_index": record["global_building_index"],
            "membership_signature": record["membership_signature"],
            "canonical_observation_id": canonical.observation_id,
            "member_observation_ids_json": json.dumps(
                [member.observation_id for member in members], ensure_ascii=False
            ),
            "member_count": len(members),
            "patch_count": len({member.patch_id for member in members}),
            "sample_count": len({member.sample_id for member in members}),
            "canonical_patch_id": canonical.patch_id,
            "canonical_view_id": canonical.view_id,
            "canonical_instance_id": canonical.instance_id,
            "canonical_boundary_status": canonical.boundary_status,
            "canonical_centroid_in_reliable_core": (
                canonical.reliable_core_contains_centroid
            ),
            "canonical_reliable_core_edge_distance_m": (
                canonical.reliable_core_edge_distance_m
            ),
            "canonical_score": canonical.canonical_score,
            "footprint_area_m2": canonical.geometry_area_m2,
            "centroid_x_m": canonical.centroid_x_m,
            "centroid_y_m": canonical.centroid_y_m,
            "base_z_m": canonical.base_z_m,
            "top_z_m": canonical.top_z_m,
            "height_m": canonical.height_m,
            "height_method": canonical.height_method,
            "has_matching_conflict": any(member.ambiguity_count > 0 for member in members),
            "canonical_rgb_path": canonical.paths["rgb"],
            "canonical_source_instance_ids_path": canonical.paths["source_instance_ids"],
            "canonical_bev_instance_ids_path": canonical.paths["bev_instance_ids"],
            "canonical_building_point_cloud_npz_path": canonical.paths[
                "building_point_cloud_npz"
            ],
            "canonical_glb": record.get("canonical_glb", ""),
            "canonical_glb_ue_z_up": record.get("canonical_glb_ue_z_up", ""),
        }
        rows.append(row)
        feature_properties = dict(row)
        feature_properties["member_observation_ids"] = json.loads(
            feature_properties.pop("member_observation_ids_json")
        )
        features.append(
            {
                "type": "Feature",
                "properties": feature_properties,
                "geometry": mapping(canonical.geometry),
            }
        )
    return rows, {"type": "FeatureCollection", "features": features}


def build_observation_geojson(observations: list[Observation]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for observation in observations:
        properties = observation_row(observation)
        properties["boundary_evidence"] = observation.boundary_evidence
        properties["source_mean_rgb"] = list(observation.source_mean_rgb)
        properties["source_std_rgb"] = list(observation.source_std_rgb)
        properties.pop("source_mean_rgb_json")
        properties.pop("source_std_rgb_json")
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": mapping(observation.geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    observations, context = load_observations(
        run_root, dataset_root, args.footprint_simplify_m
    )
    print(f"Loaded {len(observations)} E10 observations", flush=True)
    candidate_edges = build_candidate_edges(observations, args)
    selected_edges, ambiguity_records = choose_one_to_one_edges(
        candidate_edges, observations, args.minimum_match_score
    )
    near_edges = [
        edge
        for edge in candidate_edges
        if edge.accepted and edge.match_mode == "near_aligned_shape"
    ]
    reconciliation_edges = [
        edge
        for edge in candidate_edges
        if edge.accepted and edge.match_mode == "contained_fragment_reconciliation"
    ]
    clusters, cluster_rejections = cluster_observations(
        observations, selected_edges, near_edges, reconciliation_edges
    )
    edge_lookup = {
        tuple(sorted((edge.a, edge.b))): edge for edge in candidate_edges
    }
    all_global_records = assign_global_ids(clusters, observations, edge_lookup)
    global_records, publication_decisions, quarantine_candidates = (
        apply_global_publication_policy(all_global_records, observations, args)
    )
    published_observation_indices = {
        index for record in global_records for index in record["members"]
    }
    published_observations = [
        observation
        for observation in observations
        if observation.index in published_observation_indices
    ]
    sample_records = create_identity_rasters(
        output_root, published_observations, context["sample_cache"]
    )

    glb_outputs: dict[str, Any] = {}
    if not args.skip_glb:
        glb_outputs = export_global_glbs(output_root, global_records, observations)
    global_rows, global_geojson = build_global_rows_and_geojson(
        global_records, observations
    )
    observation_rows = [observation_row(item) for item in published_observations]
    candidate_rows = [edge_row(edge, observations) for edge in candidate_edges]
    write_csv(output_root / "observation_to_global.csv", observation_rows)
    write_jsonl(output_root / "building_observations.jsonl", observation_rows)
    write_csv(output_root / "global_buildings.csv", global_rows)
    write_jsonl(output_root / "global_buildings.jsonl", global_rows)
    write_json(
        output_root / "building_observation_footprints.geojson",
        build_observation_geojson(published_observations),
    )
    write_json(output_root / "global_building_footprints.geojson", global_geojson)
    write_csv(output_root / "candidate_matches.csv", candidate_rows)
    write_json(
        output_root / "matching_conflicts.json",
        {
            "ambiguous_sample_pair_matches": ambiguity_records,
            "cluster_merge_rejections": cluster_rejections,
        },
    )
    write_csv(output_root / "global_building_publication_decisions.csv", publication_decisions)
    write_jsonl(
        output_root / "global_building_publication_decisions.jsonl",
        publication_decisions,
    )
    write_csv(output_root / "broad_fragment_quarantine_candidates.csv", quarantine_candidates)
    write_jsonl(
        output_root / "broad_fragment_quarantine_candidates.jsonl",
        quarantine_candidates,
    )
    quarantine_by_id = {
        row["predicted_global_building_id"]: row for row in quarantine_candidates
    }
    write_json(
        output_root / "broad_fragment_quarantine_candidates.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": quarantine_by_id[record["global_building_id"]],
                    "geometry": mapping(
                        observations[record["canonical_index"]].geometry
                    ),
                }
                for record in all_global_records
                if record["global_building_id"] in quarantine_by_id
            ],
        },
    )
    observation_parquet = write_parquet_if_available(
        output_root / "observation_to_global.parquet", observation_rows
    )
    global_parquet = write_parquet_if_available(
        output_root / "global_buildings.parquet", global_rows
    )

    duplicate_groups = [
        {
            "predicted_global_building_id": record["global_building_id"],
            "canonical_observation_id": observations[
                record["canonical_index"]
            ].observation_id,
            "member_observation_ids": [
                observations[index].observation_id for index in record["members"]
            ],
        }
        for record in global_records
        if len(record["members"]) > 1
    ]
    write_json(output_root / "duplicate_groups.json", duplicate_groups)
    global_id_index = {
        str(record["global_building_index"]): record["global_building_id"]
        for record in global_records
    }
    write_json(output_root / "global_building_index.json", global_id_index)

    accepted_count = sum(edge.accepted for edge in candidate_edges)
    merged_edge_count = sum(edge.cluster_action == "merged" for edge in candidate_edges)
    near_consensus_edge_count = sum(
        edge.cluster_action == "near_consensus_merged" for edge in candidate_edges
    )
    reconciliation_edge_count = sum(
        edge.cluster_action == "contained_fragment_reconciled"
        for edge in candidate_edges
    )
    singleton_count = sum(len(record["members"]) == 1 for record in global_records)
    removed_records = [
        record
        for record in all_global_records
        if record["publication_decision"]["auto_removed_from_published_outputs"]
    ]
    manifest = {
        "schema": "air2vectorcity.predicted_global_building_fusion.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete",
        "prediction_contract": {
            "global_id_source": "predicted E10 recovered geometry only",
            "observation_id": "sample_id + E10 instance_id",
            "global_id": "deterministic spatial ordering by cluster-median world centroid plus membership signature; independent of canonical selection",
            "canonical_geometry": "best-quality repaired E10 raster footprint plus E10 base/top height",
        },
        "inputs": {
            "run_root": relative(run_root),
            "e10_manifest": relative(context["e10_root"] / "manifest.json"),
            "dataset_root": relative(dataset_root),
        },
        "footprint_policy": {
            "source": "E10 modeling_instance_ids raster",
            "polygonization": "OpenCV component/hole contours, half-cell expansion, Shapely make_valid",
            "simplify_m": args.footprint_simplify_m,
            "all_output_footprints_valid": all(item.geometry.is_valid for item in observations),
        },
        "matching_policy": {
            "candidate_buffer_m": args.candidate_buffer_m,
            "minimum_iou": args.minimum_iou,
            "minimum_containment": args.minimum_containment,
            "minimum_area_ratio": args.minimum_area_ratio,
            "maximum_centroid_distance_m": args.maximum_centroid_distance_m,
            "minimum_match_score": args.minimum_match_score,
            "minimum_near_shape_iou": args.minimum_near_shape_iou,
            "minimum_near_color_similarity": args.minimum_near_color_similarity,
            "strong_overlap_height_override_enabled": args.enable_strong_overlap_height_override,
            "strong_overlap_minimum_iou": args.strong_overlap_minimum_iou,
            "strong_overlap_minimum_containment": args.strong_overlap_minimum_containment,
            "strong_overlap_minimum_area_ratio": args.strong_overlap_minimum_area_ratio,
            "strong_overlap_maximum_centroid_distance_m": args.strong_overlap_maximum_centroid_distance_m,
            "contained_fragment_reconciliation_enabled": args.enable_contained_fragment_reconciliation,
            "reconciliation_minimum_iou": args.reconciliation_minimum_iou,
            "reconciliation_minimum_containment": args.reconciliation_minimum_containment,
            "reconciliation_minimum_area_ratio": args.reconciliation_minimum_area_ratio,
            "reconciliation_maximum_centroid_distance_m": args.reconciliation_maximum_centroid_distance_m,
            "overlap_evidence_formula": "max(IoU, containment * sqrt(area_ratio))",
            "association_score_formula": "0.60*overlap_evidence + 0.15*area_ratio + 0.15*height_similarity + 0.10*centroid_similarity + 0.03_if_same_patch",
            "semantics": "overlap evidence and association score estimate whether two observations represent the same building; they are not observation-quality scores",
            "near_match_evidence": [
                "centroid-aligned footprint shape",
                "geometry distance",
                "area ratio",
                "top-height consistency",
                "source RGB mean-colour similarity",
            ],
            "near_cluster_consensus_enabled": args.enable_near_cluster_consensus,
            "near_cluster_consensus": "at least two edges from at least two sample pairs and three distinct endpoints",
            "contained_fragment_reconciliation": "post-assignment high-containment pass that preserves one observation per source sample",
            "sample_pair_assignment": "maximum-score one-to-one Hungarian assignment",
            "cluster_constraints": [
                "no repeated observation sample in one cluster",
                "cluster median top-height consistency except explicitly audited strong planar-overlap overrides",
                "ambiguous one-to-many candidates are retained for audit",
            ],
        },
        "canonical_policy": {
            "priority": [
                "centroid inside dataset reliable core when core_bounds_xy_world_m is present",
                "boundary status",
                "distance from BEV and coverage edges",
                "continuous source RGB instance edge distance up to 128 px",
                "stable height layer",
                "observed footprint support",
                "agreement with cluster members",
                "area near cluster median",
                "ambiguity penalty",
            ],
            "preliminary_quality_weights": {
                "boundary_status_interior_review_exclude": [55.0, 32.0, 5.0],
                "bev_edge_distance": "0.25 per metre, capped at 20 m",
                "coverage_edge_distance": "0.25 per metre, capped at 20 m",
                "rgb_edge_distance": "10 * min(distance_px / 128, 1)",
                "stable_layer": "8 if present, otherwise 2",
                "observed_area_fraction": "up to 8",
                "tilt_light": 2.0,
                "extra_component_penalty": "2 each, capped at 8",
            },
            "uses_union_geometry": False,
            "reliable_core": {
                "explicit_core_patch_count": context["explicit_core_patch_count"],
                "fallback": "bounds_xy_world_m when core bounds are absent",
                "use": "halo observations remain matching evidence; a core-centred member is preferred as canonical",
            },
        },
        "publication_filter": {
            "enabled": args.enable_low_confidence_fragment_filter,
            "id_assignment": "IDs are assigned before publication filtering; gaps preserve traceability.",
            "all_member_evidence": [
                "outside reliable core",
                "boundary_risk_review or boundary_clipped_exclude",
                "stable_layer_count == 0",
            ],
            "shape_gate": {
                "area_below_m2": args.fragment_minimum_area_m2,
                "or_elongation_above": args.fragment_maximum_elongation,
            },
            "broad_quarantine_applied_to_deletion": False,
            "broad_quarantine_sidecar_enabled": args.write_broad_fragment_quarantine,
            "broad_quarantine_shape_gate": {
                "area_below_m2": args.quarantine_maximum_area_m2,
                "or_elongation_above": args.quarantine_maximum_elongation,
                "or_rotated_width_below_m": args.quarantine_minimum_width_m,
                "or_compactness_below": args.quarantine_minimum_compactness,
            },
        },
        "mesh_policy": {
            "structured_footprint": "canonical repaired Polygon/MultiPolygon remains authoritative",
            "minimum_component_area_m2": 0.0,
            "interior_holes": "all original holes preserved",
            "topology_preserving_simplify_m": 0.0,
            "triangulation": "Earcut with exact cap checks; constrained Delaunay fallback on cap mismatch",
            "extrusion": "canonical base_z_m to top_z_m",
        },
        "totals": {
            "sample_views": len(sample_records),
            "observations_before_publication_filter": len(observations),
            "observations": len(published_observations),
            "candidate_edges": len(candidate_edges),
            "accepted_candidate_edges": accepted_count,
            "selected_one_to_one_edges": len(selected_edges),
            "merged_edges": merged_edge_count,
            "near_consensus_edges": near_consensus_edge_count,
            "contained_fragment_reconciliation_edges": reconciliation_edge_count,
            "predicted_global_buildings_before_publication_filter": len(all_global_records),
            "predicted_global_buildings": len(global_records),
            "singleton_global_buildings": singleton_count,
            "duplicate_groups": len(duplicate_groups),
            "global_buildings_removed": len(removed_records),
            "observations_removed": len(observations) - len(published_observations),
            "broad_quarantine_candidates": len(quarantine_candidates),
        },
        "identity_chain": {
            "sample_records": sample_records,
            "numeric_index_sidecar": relative(output_root / "global_building_index.json"),
            "source_rgb_global_index_root": relative(
                output_root / "source_rgb_global_building_index"
            ),
            "bev_global_index_root": relative(output_root / "bev_global_building_index"),
        },
        "outputs": {
            "observation_to_global_csv": relative(
                output_root / "observation_to_global.csv"
            ),
            "observation_to_global_parquet": relative(
                output_root / "observation_to_global.parquet"
            )
            if observation_parquet
            else None,
            "building_observations_jsonl": relative(
                output_root / "building_observations.jsonl"
            ),
            "global_buildings_csv": relative(output_root / "global_buildings.csv"),
            "global_buildings_parquet": relative(
                output_root / "global_buildings.parquet"
            )
            if global_parquet
            else None,
            "global_buildings_jsonl": relative(
                output_root / "global_buildings.jsonl"
            ),
            "global_building_footprints_geojson": relative(
                output_root / "global_building_footprints.geojson"
            ),
            "building_observation_footprints_geojson": relative(
                output_root / "building_observation_footprints.geojson"
            ),
            "candidate_matches_csv": relative(output_root / "candidate_matches.csv"),
            "duplicate_groups": relative(output_root / "duplicate_groups.json"),
            "matching_conflicts": relative(output_root / "matching_conflicts.json"),
            "global_building_publication_decisions_csv": relative(
                output_root / "global_building_publication_decisions.csv"
            ),
            "global_building_publication_decisions_jsonl": relative(
                output_root / "global_building_publication_decisions.jsonl"
            ),
            "broad_fragment_quarantine_candidates_csv": relative(
                output_root / "broad_fragment_quarantine_candidates.csv"
            ),
            "broad_fragment_quarantine_candidates_jsonl": relative(
                output_root / "broad_fragment_quarantine_candidates.jsonl"
            ),
            "broad_fragment_quarantine_candidates_geojson": relative(
                output_root / "broad_fragment_quarantine_candidates.geojson"
            ),
            **glb_outputs,
        },
    }
    write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
