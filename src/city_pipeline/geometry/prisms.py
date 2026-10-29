#!/usr/bin/env python3
"""Estimate robust instance top levels and extrude observed footprints into prisms."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.camera_geometry import read_json
from city_pipeline.geometry.depth_lifting import (
    build_overview,
    instance_palette,
    save_u16,
)


DEFAULT_E04_ROOT = Path("UNCONFIGURED_DEFAULT_E04_ROOT")
CELL_SIZE_M = 0.125
PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
BASE_Z_M = 0.48
HEIGHT_BIN_M = 0.5
TOP_SUPPORT_WINDOW_M = 1.0
MIN_TOP_SUPPORT_AREA_M2 = 4.0
MAX_TOP_SUPPORT_AREA_M2 = 20.0
RELATIVE_TOP_SUPPORT_FRACTION = 0.005
MIN_TOP_CONNECTED_AREA_M2 = 2.0
MIN_FOOTPRINT_COMPONENT_AREA_M2 = 5.0
MESH_SIMPLIFY_EPSILON_PX = 2.0
OBSERVED_HEIGHT_PROVENANCE = (1, 2, 4)
UE_Z_UP_TO_GLTF_Y_UP = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e04-root", type=Path, default=DEFAULT_E04_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--run-label",
        default="E04 aggressive",
        help="Human-readable input run label used in previews and metadata.",
    )
    parser.add_argument(
        "--schema",
        default="air2vectorcity.coarse_building_prisms_e04.v1",
        help="Output manifest schema identifier.",
    )
    parser.add_argument(
        "--input-manifest-key",
        default="input_e04",
        help="Manifest key used for the input run path.",
    )
    parser.add_argument(
        "--use-precomputed-height-layers",
        action="store_true",
        help=(
            "Use per-instance selected_top_z_m and selected_top_support_ids from "
            "the input run instead of running the legacy robust-top selector."
        ),
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def robust_top_level(
    instance_mask: np.ndarray,
    surface_z: np.ndarray,
    provenance: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    observed = (
        instance_mask
        & np.isfinite(surface_z)
        & np.isin(provenance, np.asarray(OBSERVED_HEIGHT_PROVENANCE, dtype=np.uint8))
    )
    values = surface_z[observed].astype(np.float64)
    if not values.size:
        observed = instance_mask & np.isfinite(surface_z)
        values = surface_z[observed].astype(np.float64)
    if not values.size:
        raise ValueError("Instance has no finite height samples")

    observed_area = float(values.size * PIXEL_AREA_M2)
    min_support_area = max(
        MIN_TOP_SUPPORT_AREA_M2,
        min(MAX_TOP_SUPPORT_AREA_M2, observed_area * RELATIVE_TOP_SUPPORT_FRACTION),
    )
    min_support_pixels = int(math.ceil(min_support_area / PIXEL_AREA_M2))
    min_connected_pixels = int(
        math.ceil(MIN_TOP_CONNECTED_AREA_M2 / PIXEL_AREA_M2)
    )
    highest_edge = math.ceil(float(np.max(values)) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    lowest_edge = math.floor(float(np.min(values)) / HEIGHT_BIN_M) * HEIGHT_BIN_M
    selected_mask: np.ndarray | None = None
    selected_interval: tuple[float, float] | None = None
    selected_largest_component = 0

    upper = highest_edge
    while upper >= lowest_edge:
        lower = upper - TOP_SUPPORT_WINDOW_M
        band = observed & (surface_z > lower) & (surface_z <= upper + 1e-5)
        count = int(np.count_nonzero(band))
        if count >= min_support_pixels:
            labels, component_count = ndimage.label(
                band, structure=np.ones((3, 3), dtype=np.uint8)
            )
            sizes = np.bincount(labels.reshape(-1))
            largest = int(np.max(sizes[1:])) if component_count else 0
            if largest >= min_connected_pixels:
                selected_mask = band
                selected_interval = (lower, upper)
                selected_largest_component = largest
                break
        upper -= HEIGHT_BIN_M

    if selected_mask is None:
        # Fallback remains robust to isolated maxima: the top is a quantile,
        # never the single maximum pixel.
        top_z = float(np.percentile(values, 98.0))
        tolerance = max(0.5, TOP_SUPPORT_WINDOW_M)
        selected_mask = observed & (surface_z >= top_z - tolerance)
        method = "p98_fallback"
        selected_interval = (top_z - tolerance, top_z + tolerance)
        labels, component_count = ndimage.label(
            selected_mask, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.reshape(-1))
        selected_largest_component = (
            int(np.max(sizes[1:])) if component_count else 0
        )
    else:
        top_z = float(np.median(surface_z[selected_mask]))
        method = "highest_supported_1m_band_median"

    top_z = max(top_z, BASE_Z_M + 0.5)
    metadata = {
        "method": method,
        "base_z_m": BASE_Z_M,
        "top_z_m": top_z,
        "extrusion_height_m": top_z - BASE_Z_M,
        "raw_max_z_m": float(np.max(values)),
        "raw_p99_z_m": float(np.percentile(values, 99.0)),
        "observed_height_sample_count": int(values.size),
        "observed_area_m2": observed_area,
        "minimum_top_support_area_m2": min_support_area,
        "minimum_top_support_pixels": min_support_pixels,
        "selected_top_band_m": list(selected_interval),
        "selected_support_pixel_count": int(np.count_nonzero(selected_mask)),
        "selected_support_area_m2": float(
            np.count_nonzero(selected_mask) * PIXEL_AREA_M2
        ),
        "selected_largest_connected_area_m2": float(
            selected_largest_component * PIXEL_AREA_M2
        ),
        "raw_max_minus_selected_top_m": float(np.max(values) - top_z),
        "observed_provenance_ids": list(OBSERVED_HEIGHT_PROVENANCE),
    }
    return top_z, selected_mask, metadata


def pixel_to_world(
    points_uv: np.ndarray,
    pixel_center_to_world: list[list[float]],
) -> np.ndarray:
    matrix = np.asarray(pixel_center_to_world, dtype=np.float64)
    u = points_uv[:, 0].astype(np.float64)
    v = points_uv[:, 1].astype(np.float64)
    return np.stack(
        (
            matrix[0, 0] * u + matrix[0, 1] * v + matrix[0, 2],
            matrix[1, 0] * u + matrix[1, 1] * v + matrix[1, 2],
        ),
        axis=1,
    )


def signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    def cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        first = p2 - p1
        second = p3 - p1
        return float(first[0] * second[1] - first[1] * second[0])

    c1 = cross(a, b, point)
    c2 = cross(b, c, point)
    c3 = cross(c, a, point)
    eps = 1e-10
    return c1 >= -eps and c2 >= -eps and c3 >= -eps


def ear_clip(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Triangulate a simple polygon; return triangles and fallback flag."""

    if points.shape[0] < 3:
        raise ValueError("Polygon needs at least three vertices")
    polygon = points.copy()
    if signed_area(polygon) < 0:
        polygon = polygon[::-1]
    indices = list(range(polygon.shape[0]))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < polygon.shape[0] ** 2:
        ear_found = False
        for position, current in enumerate(indices):
            previous = indices[position - 1]
            following = indices[(position + 1) % len(indices)]
            a, b, c = polygon[previous], polygon[current], polygon[following]
            first = b - a
            second = c - b
            cross = float(first[0] * second[1] - first[1] * second[0])
            if cross <= 1e-10:
                continue
            contains = False
            for other in indices:
                if other in (previous, current, following):
                    continue
                if point_in_triangle(polygon[other], a, b, c):
                    contains = True
                    break
            if contains:
                continue
            triangles.append((previous, current, following))
            del indices[position]
            ear_found = True
            break
        if not ear_found:
            break
        guard += 1
    if len(indices) == 3:
        triangles.append(tuple(indices))
        return np.asarray(triangles, dtype=np.int64), polygon, False

    # Rare numerical fallback: a convex hull is safer than exporting an
    # invalid self-intersecting mesh.  The exact raster boundary remains saved.
    hull_indices = cv2.convexHull(
        polygon.astype(np.float32), returnPoints=False
    ).reshape(-1)
    hull = polygon[hull_indices]
    if signed_area(hull) < 0:
        hull = hull[::-1]
    triangles = [(0, i, i + 1) for i in range(1, hull.shape[0] - 1)]
    return np.asarray(triangles, dtype=np.int64), hull, True


def prism_mesh(
    polygon_xy: np.ndarray,
    base_z: float,
    top_z: float,
    color_rgb: np.ndarray,
) -> tuple[trimesh.Trimesh, bool]:
    points = polygon_xy.copy()
    if signed_area(points) < 0:
        points = points[::-1]
    triangles, points, fallback = ear_clip(points)
    count = points.shape[0]
    bottom = np.column_stack((points, np.full(count, base_z)))
    top = np.column_stack((points, np.full(count, top_z)))
    vertices = np.concatenate((bottom, top), axis=0).astype(np.float32)
    faces: list[tuple[int, int, int]] = []
    for a, b, c in triangles:
        faces.append((c, b, a))
        faces.append((a + count, b + count, c + count))
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, following + count))
        faces.append((index, following + count, index + count))
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    rgba = np.concatenate((color_rgb.astype(np.uint8), np.asarray([255], dtype=np.uint8)))
    mesh.visual.face_colors = np.repeat(rgba[None, :], mesh.faces.shape[0], axis=0)
    return mesh, fallback


def extract_boundaries_and_meshes(
    instance_id: int,
    mask: np.ndarray,
    top_z: float,
    transform: dict[str, Any],
    color: np.ndarray,
) -> tuple[list[dict[str, Any]], list[trimesh.Trimesh]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    records: list[dict[str, Any]] = []
    meshes: list[trimesh.Trimesh] = []
    for component_index, contour in enumerate(contours, start=1):
        area_px = float(cv2.contourArea(contour))
        area_m2 = area_px * PIXEL_AREA_M2
        if area_m2 < MIN_FOOTPRINT_COMPONENT_AREA_M2:
            continue
        exact_uv = contour[:, 0, :].astype(np.int32)
        exact_world = pixel_to_world(
            exact_uv, transform["pixel_center_index_to_ue_world_xy_3x3"]
        )
        simplified = cv2.approxPolyDP(
            contour, epsilon=MESH_SIMPLIFY_EPSILON_PX, closed=True
        )[:, 0, :]
        if simplified.shape[0] < 3:
            simplified = cv2.convexHull(contour)[:, 0, :]
        mesh_world = pixel_to_world(
            simplified, transform["pixel_center_index_to_ue_world_xy_3x3"]
        )
        mesh, fallback = prism_mesh(mesh_world, BASE_Z_M, top_z, color)
        meshes.append(mesh)
        records.append(
            {
                "component_index": component_index,
                "raster_contour_area_m2": area_m2,
                "boundary_pixel_count": int(exact_uv.shape[0]),
                "boundary_pixels_uv": exact_uv.tolist(),
                "boundary_world_xy_m": exact_world.tolist(),
                "mesh_boundary_pixels_uv": simplified.astype(int).tolist(),
                "mesh_boundary_world_xy_m": mesh_world.tolist(),
                "mesh_vertex_count_2d": int(mesh_world.shape[0]),
                "triangulation_convex_hull_fallback": bool(fallback),
            }
        )
    return records, meshes


def preview_prisms(
    sample_id: str,
    ids_preview_path: Path,
    mesh_records: list[tuple[int, trimesh.Trimesh, np.ndarray]],
    destination: Path,
    run_label: str = "E04 aggressive",
) -> None:
    figure = plt.figure(figsize=(14, 7), facecolor="#111419")
    axis_mask = figure.add_subplot(1, 2, 1)
    axis_mask.imshow(Image.open(ids_preview_path))
    axis_mask.set_title(f"{run_label} instance footprint", color="white", fontsize=14)
    axis_mask.axis("off")

    axis = figure.add_subplot(1, 2, 2, projection="3d")
    all_vertices: list[np.ndarray] = []
    for _, mesh, color in mesh_records:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        polygons = vertices[faces]
        collection = Poly3DCollection(
            polygons,
            facecolor=np.asarray(color, dtype=np.float32) / 255.0,
            edgecolor=(0.08, 0.08, 0.10, 0.55),
            linewidth=0.15,
            alpha=0.92,
        )
        axis.add_collection3d(collection)
        all_vertices.append(vertices)
    if all_vertices:
        vertices = np.concatenate(all_vertices, axis=0)
        lo = np.min(vertices, axis=0)
        hi = np.max(vertices, axis=0)
        center = (lo + hi) / 2.0
        span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1.0)
        axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
        axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
        axis.set_zlim(float(lo[2]), float(hi[2]))
        z_span = max(float(hi[2] - lo[2]), 1.0)
        axis.set_box_aspect((1.0, 1.0, max(0.25, min(0.7, 1.5 * z_span / span))))
    axis.view_init(elev=38, azim=-58)
    axis.set_title("Robust-top footprint extrusion", color="white", fontsize=14)
    axis.set_xlabel("UE world X (m)", color="white")
    axis.set_ylabel("UE world Y (m)", color="white")
    axis.set_zlabel("world Z (m)", color="white")
    axis.tick_params(colors="#d7dde8", labelsize=7)
    axis.set_facecolor("#111419")
    for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane.pane.set_facecolor((0.07, 0.08, 0.10, 1.0))
        pane.pane.set_edgecolor((0.20, 0.23, 0.28, 1.0))
    figure.suptitle(
        f"{sample_id} — coarse per-instance building prisms",
        color="white",
        fontsize=17,
        y=0.96,
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.86, wspace=0.04)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> int:
    global CELL_SIZE_M, PIXEL_AREA_M2, MESH_SIMPLIFY_EPSILON_PX, BASE_Z_M
    args = parse_args()
    e04_root = args.e04_root.resolve()
    manifest = read_json(e04_root / "manifest.json")
    samples = manifest["samples"]
    if args.sample_id:
        requested = set(args.sample_id)
        samples = [sample for sample in samples if sample["id"] in requested]
    cell_sizes = {
        float(sample["rasterization"]["meters_per_pixel"][0]) for sample in samples
    }
    if len(cell_sizes) != 1:
        raise ValueError(
            f"Prism export requires one consistent BEV cell size: {sorted(cell_sizes)}"
        )
    CELL_SIZE_M = cell_sizes.pop()
    PIXEL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M
    MESH_SIMPLIFY_EPSILON_PX = 0.25 / CELL_SIZE_M

    output_root = e04_root / "building_prisms"
    for name in (
        "top_level_support_ids",
        "footprint_boundaries",
        "footprint_geojson",
        "height_tables",
        "glb",
        "glb_ue_z_up",
        "previews",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    preview_paths: list[Path] = []
    sample_records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        # Historical SmallCity manifests did not carry this field and retain
        # the 0.48 m fallback. Real-scene runs provide a patch-local ground Z.
        BASE_Z_M = float(sample.get("ground_reference_z_m", 0.48))
        print(f"[{sample_index}/{len(samples)}] {sample_id}", flush=True)
        ids = np.asarray(
            Image.open(project_path(sample["outputs"]["final_ids"])), dtype=np.uint16
        )
        surface_z = np.load(project_path(sample["outputs"]["surface_z"])).astype(
            np.float32
        )
        provenance = np.asarray(
            Image.open(project_path(sample["outputs"]["provenance"])), dtype=np.uint8
        )
        precomputed_heights: dict[int, dict[str, Any]] = {}
        precomputed_support_ids: np.ndarray | None = None
        if args.use_precomputed_height_layers:
            if "height_layers" not in sample["outputs"]:
                raise KeyError(f"Missing height_layers output for {sample_id}")
            if "selected_top_support_ids" not in sample["outputs"]:
                raise KeyError(f"Missing selected_top_support_ids for {sample_id}")
            rows = read_json(project_path(sample["outputs"]["height_layers"]))
            precomputed_heights = {int(row["instance_id"]): row for row in rows}
            precomputed_support_ids = np.asarray(
                Image.open(project_path(sample["outputs"]["selected_top_support_ids"])),
                dtype=np.uint16,
            )
        transform = read_json(project_path(sample["transform"]))
        palette = instance_palette(int(np.max(ids)))
        support_ids = np.zeros_like(ids, dtype=np.uint16)
        scene_ue = trimesh.Scene()
        scene_viewer = trimesh.Scene()
        boundary_instances: list[dict[str, Any]] = []
        height_rows: list[dict[str, Any]] = []
        geojson_features: list[dict[str, Any]] = []
        preview_meshes: list[tuple[int, trimesh.Trimesh, np.ndarray]] = []

        for instance_id in np.unique(ids):
            if instance_id == 0:
                continue
            mask = ids == instance_id
            precomputed = precomputed_heights.get(int(instance_id))
            if (
                args.use_precomputed_height_layers
                and precomputed is not None
                and precomputed.get("selected_top_z_m") is not None
            ):
                top_z = max(float(precomputed["selected_top_z_m"]), BASE_Z_M + 0.5)
                assert precomputed_support_ids is not None
                support = precomputed_support_ids == instance_id
                selected_layer = max(
                    precomputed["stable_layers"], key=lambda item: item["height_m"]
                )
                observed = (
                    mask
                    & np.isfinite(surface_z)
                    & np.isin(
                        provenance,
                        np.asarray(OBSERVED_HEIGHT_PROVENANCE, dtype=np.uint8),
                    )
                )
                observed_values = surface_z[observed].astype(np.float64)
                height_metadata = {
                    "method": "e10_highest_height_guided_stable_layer",
                    "base_z_m": BASE_Z_M,
                    "top_z_m": top_z,
                    "extrusion_height_m": top_z - BASE_Z_M,
                    "raw_max_z_m": float(np.max(observed_values)),
                    "raw_p99_z_m": float(np.percentile(observed_values, 99.0)),
                    "observed_height_sample_count": int(observed_values.size),
                    "observed_area_m2": float(observed_values.size * PIXEL_AREA_M2),
                    "minimum_top_support_area_m2": float(
                        precomputed["layer_discovery"]["base_required_area_m2"]
                    ),
                    "minimum_top_support_pixels": int(
                        math.ceil(
                            precomputed["layer_discovery"]["base_required_area_m2"]
                            / PIXEL_AREA_M2
                        )
                    ),
                    "selected_top_band_m": [
                        selected_layer["lower_m"],
                        selected_layer["upper_m"],
                    ],
                    "selected_support_pixel_count": int(np.count_nonzero(support)),
                    "selected_support_area_m2": float(
                        np.count_nonzero(support) * PIXEL_AREA_M2
                    ),
                    "selected_largest_connected_area_m2": float(
                        selected_layer["largest_connected_area_m2"]
                    ),
                    "selected_internal_radius_m": float(
                        selected_layer["internal_radius_m"]
                    ),
                    "selected_support_fraction": float(
                        selected_layer["support_fraction"]
                    ),
                    "selected_layer_acceptance": selected_layer["acceptance"],
                    "stable_layer_count": int(precomputed["stable_layer_count"]),
                    "raw_max_minus_selected_top_m": float(
                        np.max(observed_values) - top_z
                    ),
                    "observed_provenance_ids": list(OBSERVED_HEIGHT_PROVENANCE),
                }
            else:
                top_z, support, height_metadata = robust_top_level(
                    mask, surface_z, provenance
                )
                if args.use_precomputed_height_layers:
                    height_metadata["method"] = (
                        "legacy_robust_top_fallback_no_e10_stable_layer"
                    )
            support_ids[support] = instance_id
            components, meshes = extract_boundaries_and_meshes(
                int(instance_id),
                mask,
                top_z,
                transform,
                palette[instance_id],
            )
            if not components:
                continue
            for component, mesh in zip(components, meshes, strict=True):
                component_index = component["component_index"]
                name = f"{sample_id}__instance_{int(instance_id):04d}__part_{component_index:02d}"
                scene_ue.add_geometry(mesh, node_name=name, geom_name=name)
                viewer_mesh = mesh.copy()
                viewer_mesh.apply_transform(UE_Z_UP_TO_GLTF_Y_UP)
                scene_viewer.add_geometry(
                    viewer_mesh, node_name=name, geom_name=name
                )
                preview_meshes.append((int(instance_id), mesh, palette[instance_id]))
                coordinates = component["mesh_boundary_world_xy_m"]
                geojson_features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "sample_id": sample_id,
                            "instance_id": int(instance_id),
                            "component_index": component_index,
                            "base_z_m": BASE_Z_M,
                            "top_z_m": top_z,
                            "height_m": top_z - BASE_Z_M,
                            "top_height_method": height_metadata["method"],
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[*coordinates, coordinates[0]]],
                        },
                    }
                )
            boundary_instances.append(
                {
                    "instance_id": int(instance_id),
                    "height": height_metadata,
                    "components": components,
                }
            )
            height_rows.append(
                {
                    "sample_id": sample_id,
                    "instance_id": int(instance_id),
                    "component_count": len(components),
                    **height_metadata,
                }
            )

        paths = {
            "support_ids": output_root / "top_level_support_ids" / f"{sample_id}__top_level_support_ids.png",
            "boundaries": output_root / "footprint_boundaries" / f"{sample_id}__footprint_boundaries.json",
            "geojson": output_root / "footprint_geojson" / f"{sample_id}__footprints.geojson",
            "height_json": output_root / "height_tables" / f"{sample_id}__instance_heights.json",
            "height_csv": output_root / "height_tables" / f"{sample_id}__instance_heights.csv",
            "glb": output_root / "glb" / f"{sample_id}__building_prisms.glb",
            "glb_ue_z_up": output_root / "glb_ue_z_up" / f"{sample_id}__building_prisms_ue_z_up.glb",
            "preview": output_root / "previews" / f"{sample_id}__building_prisms.png",
        }
        save_u16(paths["support_ids"], support_ids)
        paths["boundaries"].write_text(
            json.dumps(
                {
                    "schema": "air2vectorcity.raster_footprint_boundaries.v1",
                    "sample_id": sample_id,
                    "coordinate_system": (
                        "UE world XY metres; boundary pixels use the camera-aligned "
                        f"u/v raster from {args.run_label}"
                    ),
                    "pixel_to_world_transform": sample["transform"],
                    "instances": boundary_instances,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["geojson"].write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": geojson_features},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["height_json"].write_text(
            json.dumps(height_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with paths["height_csv"].open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "sample_id",
                "instance_id",
                "component_count",
                "base_z_m",
                "top_z_m",
                "extrusion_height_m",
                "method",
                "raw_max_z_m",
                "raw_p99_z_m",
                "raw_max_minus_selected_top_m",
                "observed_area_m2",
                "minimum_top_support_area_m2",
                "selected_support_area_m2",
                "selected_largest_connected_area_m2",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(height_rows)

        # Default GLB follows the glTF Y-up convention for direct viewing.
        # The second GLB preserves raw UE X/Y/Z metres for geometry exchange.
        paths["glb"].write_bytes(scene_viewer.export(file_type="glb"))
        paths["glb_ue_z_up"].write_bytes(scene_ue.export(file_type="glb"))
        preview_prisms(
            sample_id,
            project_path(sample["outputs"]["final_preview"]),
            preview_meshes,
            paths["preview"],
            args.run_label,
        )
        preview_paths.append(paths["preview"])
        sample_records.append(
            {
                "id": sample_id,
                "instance_count": len(height_rows),
                "mesh_component_count": len(preview_meshes),
                "ground_reference_z_m": BASE_Z_M,
                "glb_size_bytes": paths["glb"].stat().st_size,
                "height_summary": {
                    "min_m": min(row["extrusion_height_m"] for row in height_rows),
                    "median_m": float(
                        np.median([row["extrusion_height_m"] for row in height_rows])
                    ),
                    "max_m": max(row["extrusion_height_m"] for row in height_rows),
                },
                "outputs": {
                    key: relative(path)
                    for key, path in paths.items()
                },
            }
        )

    overview = output_root / f"all_{len(sample_records)}_building_prism_previews.png"
    build_overview(preview_paths, overview)
    output_manifest = {
        "schema": args.schema,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        args.input_manifest_key: relative(e04_root),
        "input_run_label": args.run_label,
        "representation": (
            "per-instance observed raster footprint extruded from each sample's "
            "camera ground_reference_z_m to a robust supported top level"
        ),
        "height_policy": {
            "precomputed_height_layers_enabled": bool(
                args.use_precomputed_height_layers
            ),
            "height_bin_m": HEIGHT_BIN_M,
            "top_support_window_m": TOP_SUPPORT_WINDOW_M,
            "minimum_support_area_m2": MIN_TOP_SUPPORT_AREA_M2,
            "maximum_area_threshold_m2": MAX_TOP_SUPPORT_AREA_M2,
            "relative_support_fraction": RELATIVE_TOP_SUPPORT_FRACTION,
            "minimum_connected_support_area_m2": MIN_TOP_CONNECTED_AREA_M2,
            "observed_provenance_ids": list(OBSERVED_HEIGHT_PROVENANCE),
            "single_max_pixel_used": False,
        },
        "footprint_policy": {
            "source": f"{args.run_label} final per-view observed instance raster",
            "exact_boundary_pixels_saved": True,
            "mesh_simplification_epsilon_px": MESH_SIMPLIFY_EPSILON_PX,
            "minimum_component_area_m2": MIN_FOOTPRINT_COMPONENT_AREA_M2,
            "holes": "large remaining raster holes are not cut from the coarse mesh; external contours only",
        },
        "recommended_download_format": "Y-up GLB for direct viewing; UE-Z-up GLB for raw coordinates; GeoJSON+JSON+CSV for structured analysis; NPZ/PLY for point clouds",
        "coordinate_exports": {
            "glb": "standard glTF Y-up; coordinates are (UE X, UE Z, -UE Y)",
            "glb_ue_z_up": "raw UE world (X,Y,Z) metres; some generic glTF viewers may display Z-up data sideways",
            "geojson_and_boundary_json": "raw UE world XY metres",
            "ue_z_up_to_gltf_y_up_4x4": UE_Z_UP_TO_GLTF_Y_UP.tolist(),
        },
        "overview": relative(overview),
        "samples": sample_records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
