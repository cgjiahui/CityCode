#!/usr/bin/env python3
"""Tag E10 instances that may be incomplete at image, coverage, or BEV boundaries."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.roof_masks import two_row_comparison  # noqa: E402
from city_pipeline.geometry.camera_geometry import (  # noqa: E402
    load_rgb,
    save_rgb,
    save_u8,
)
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    build_overview,
    instance_preview,
    rgb_overlay,
    save_u16,
    write_binary_ply,
)


EXPERIMENT_ROOT = Path("UNCONFIGURED_EXPERIMENT_ROOT")
DEFAULT_E09_ROOT = EXPERIMENT_ROOT / "camera_aligned_25d_e09_full_pipeline_p5_close7"
DEFAULT_E10_ROOT = EXPERIMENT_ROOT / "camera_aligned_25d_e10_height_guided_instance_cleaning"
DEFAULT_OUTPUT_ROOT = DEFAULT_E10_ROOT / "boundary_instance_tagging"

METERS_PER_PIXEL = 0.125
BEV_NEAR_BOUNDARY_M = 2.0
COVERAGE_CONFIRMED_M = 0.25
COVERAGE_NEAR_BOUNDARY_M = 2.0
SOURCE_EDGE_BAND_PX = 1

STATUS_INTERIOR = 1
STATUS_REVIEW = 2
STATUS_EXCLUDE = 3
STATUS_NAMES = {
    STATUS_INTERIOR: "interior_keep",
    STATUS_REVIEW: "boundary_risk_review",
    STATUS_EXCLUDE: "boundary_clipped_exclude",
}
STATUS_COLORS = np.asarray(
    [
        [0, 0, 0],
        [43, 205, 139],
        [245, 181, 56],
        [244, 78, 78],
    ],
    dtype=np.uint8,
)
INSTANCE_NAME_PATTERN = re.compile(r"__instance_(\d{4})__part_\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e09-root", type=Path, default=DEFAULT_E09_ROOT)
    parser.add_argument("--e10-root", type=Path, default=DEFAULT_E10_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
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


def load_u16(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def source_largest_component_metrics(mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "available": False,
            "total_area_px": 0,
            "largest_component_area_px": 0,
            "largest_component_edge_distance_px": None,
            "largest_component_touches_image_edge": False,
        }
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    largest = labels == largest_label
    rows, cols = np.nonzero(largest)
    edge_distance = int(
        min(
            rows.min(),
            cols.min(),
            mask.shape[0] - 1 - rows.max(),
            mask.shape[1] - 1 - cols.max(),
        )
    )
    return {
        "available": True,
        "component_count": int(count),
        "total_area_px": int(np.count_nonzero(mask)),
        "largest_component_area_px": int(np.count_nonzero(largest)),
        "largest_component_fraction": float(np.count_nonzero(largest) / np.count_nonzero(mask)),
        "largest_component_edge_distance_px": edge_distance,
        "largest_component_touches_image_edge": edge_distance < SOURCE_EDGE_BAND_PX,
    }


def coverage_hull(validity: np.ndarray) -> np.ndarray:
    boundary = validity & ~ndi.binary_erosion(
        validity, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    rows, cols = np.nonzero(boundary)
    points = np.column_stack((cols, rows))
    if points.shape[0] < 3:
        raise ValueError("Cannot construct a coverage hull from fewer than three points")
    hull = ConvexHull(points)
    return points[hull.vertices].astype(np.float64)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )


def minimum_distance_to_polygon_edges(mask: np.ndarray, polygon_xy: np.ndarray) -> float:
    rows, cols = np.nonzero(mask_boundary(mask))
    points = np.column_stack((cols, rows)).astype(np.float64)
    minimum = np.full(points.shape[0], np.inf, dtype=np.float64)
    for index, start in enumerate(polygon_xy):
        end = polygon_xy[(index + 1) % polygon_xy.shape[0]]
        vector = end - start
        length_squared = float(np.dot(vector, vector))
        if length_squared == 0.0:
            continue
        parameter = np.clip(((points - start) @ vector) / length_squared, 0.0, 1.0)
        projection = start + parameter[:, None] * vector
        minimum = np.minimum(minimum, np.linalg.norm(points - projection, axis=1))
    return float(np.min(minimum))


def canvas_edge_distance_px(mask: np.ndarray) -> int:
    rows, cols = np.nonzero(mask)
    return int(
        min(
            rows.min(),
            cols.min(),
            mask.shape[0] - 1 - rows.max(),
            mask.shape[1] - 1 - cols.max(),
        )
    )


def status_for_evidence(
    canvas_distance_m: float,
    coverage_distance_m: float,
    source_touches_edge: bool,
) -> tuple[int, list[str]]:
    evidence: list[str] = []
    exact_bev_clip = canvas_distance_m < METERS_PER_PIXEL
    near_bev = canvas_distance_m <= BEV_NEAR_BOUNDARY_M
    coverage_clip_with_source = (
        coverage_distance_m <= COVERAGE_CONFIRMED_M and source_touches_edge
    )
    near_coverage = coverage_distance_m <= COVERAGE_NEAR_BOUNDARY_M
    if exact_bev_clip:
        evidence.append("touches_bev_canvas_edge")
    elif near_bev:
        evidence.append("within_2m_of_bev_canvas_edge")
    if source_touches_edge:
        evidence.append("largest_source_component_touches_input_edge")
    if coverage_distance_m <= COVERAGE_CONFIRMED_M:
        evidence.append("touches_projected_coverage_hull")
    elif near_coverage:
        evidence.append("within_2m_of_projected_coverage_hull")

    if exact_bev_clip or coverage_clip_with_source:
        return STATUS_EXCLUDE, evidence
    if near_bev or near_coverage or source_touches_edge:
        return STATUS_REVIEW, evidence
    return STATUS_INTERIOR, evidence


def draw_coverage_hull(rgb: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    points = [tuple(map(float, point)) for point in polygon_xy]
    draw.line(points + [points[0]], fill=(58, 190, 255), width=6)
    return np.asarray(image)


def stratified_indices(status: np.ndarray, maximum: int) -> np.ndarray:
    groups = [np.flatnonzero(status == code) for code in STATUS_NAMES]
    selected: list[np.ndarray] = []
    remaining = int(maximum)
    active = [group for group in groups if group.size]
    for group_index, group in enumerate(active):
        allocation = min(group.size, remaining // (len(active) - group_index))
        if allocation:
            positions = np.linspace(0, group.size - 1, allocation, dtype=np.int64)
            selected.append(group[positions])
            remaining -= allocation
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected))


def write_optional_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    maximum: int,
) -> dict[str, Any] | None:
    if xyz.shape[0] == 0:
        return None
    return write_binary_ply(path, xyz, rgb, instance_ids, maximum)


def augment_geojson(
    source: Path,
    destination: Path,
    status_by_instance: dict[int, dict[str, Any]],
) -> None:
    data = read_json(source)
    for feature in data["features"]:
        instance_id = int(feature["properties"]["instance_id"])
        status = status_by_instance[instance_id]
        feature["properties"].update(
            {
                "boundary_status_code": status["status_code"],
                "boundary_status": status["status"],
                "boundary_evidence": status["evidence"],
                "recommended_reconstruct_strict": status[
                    "recommended_reconstruct_strict"
                ],
                "recommended_reconstruct_conservative": status[
                    "recommended_reconstruct_conservative"
                ],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_boundary_colored_glb(
    source: Path,
    destination: Path,
    status_by_instance: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    import trimesh

    scene = trimesh.load(source, force="scene")
    instance_ids: set[int] = set()
    for geometry_name, geometry in scene.geometry.items():
        match = INSTANCE_NAME_PATTERN.search(geometry_name)
        if match is None:
            raise ValueError(f"Cannot parse Instance ID from GLB geometry: {geometry_name}")
        instance_id = int(match.group(1))
        if instance_id not in status_by_instance:
            raise KeyError(f"GLB Instance {instance_id} has no boundary status")
        status_code = int(status_by_instance[instance_id]["status_code"])
        rgba = np.concatenate((STATUS_COLORS[status_code], np.asarray([255], dtype=np.uint8)))
        geometry.visual.face_colors = np.broadcast_to(
            rgba, (geometry.faces.shape[0], 4)
        ).copy()
        geometry.metadata["building_instance_id"] = instance_id
        geometry.metadata["boundary_status_code"] = status_code
        geometry.metadata["boundary_status"] = STATUS_NAMES[status_code]
        instance_ids.add(instance_id)
    if instance_ids != set(status_by_instance):
        raise AssertionError(
            f"GLB/status Instance mismatch: GLB={sorted(instance_ids)}, "
            f"status={sorted(status_by_instance)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.export(destination)
    return {
        "geometry_count": len(scene.geometry),
        "instance_count": len(instance_ids),
        "size_bytes": destination.stat().st_size,
        "colors": {
            STATUS_NAMES[code]: STATUS_COLORS[code].tolist() for code in STATUS_NAMES
        },
    }


def main() -> int:
    global METERS_PER_PIXEL
    args = parse_args()
    e09_root = args.e09_root.resolve()
    e10_root = args.e10_root.resolve()
    output_root = args.output_root.resolve()
    e09 = read_json(e09_root / "manifest.json")
    e10 = read_json(e10_root / "manifest.json")
    prisms = read_json(e10_root / "building_prisms" / "manifest.json")
    e09_by_id = {sample["id"]: sample for sample in e09["samples"]}
    e10_by_id = {sample["id"]: sample for sample in e10["samples"]}
    prism_by_id = {sample["id"]: sample for sample in prisms["samples"]}
    sample_ids = args.sample_id or [sample["id"] for sample in e10["samples"]]
    cell_sizes = {
        float(e10_by_id[sample_id]["rasterization"]["meters_per_pixel"][0])
        for sample_id in sample_ids
    }
    if len(cell_sizes) != 1:
        raise ValueError(
            f"Boundary tagging requires one consistent BEV cell size: {sorted(cell_sizes)}"
        )
    METERS_PER_PIXEL = cell_sizes.pop()

    directory_names = (
        "instance_status_tables",
        "instance_status_csv",
        "boundary_status_ids",
        "boundary_status_previews",
        "boundary_status_overlays",
        "coverage_hull_previews",
        "boundary_risk_instance_ids",
        "reconstructable_ids_strict",
        "reconstructable_ids_conservative",
        "point_boundary_status_npz",
        "point_cloud_ply_status_all",
        "point_cloud_ply_interior",
        "point_cloud_ply_review",
        "point_cloud_ply_exclude",
        "prism_instance_status",
        "footprint_geojson_with_boundary_status",
        "building_prisms_glb_boundary_status",
        "building_prisms_glb_boundary_status_ue_z_up",
        "comparisons",
    )
    directories = {name: output_root / name for name in directory_names}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    comparisons: list[Path] = []
    totals = {"interior": 0, "review": 0, "exclude": 0}
    for sample_index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{sample_index}/{len(sample_ids)}] {sample_id}", flush=True)
        original = e09_by_id[sample_id]
        cleaned = e10_by_id[sample_id]
        prism = prism_by_id[sample_id]
        ids = load_u16(project_path(cleaned["outputs"]["final_ids"]))
        source_ids = load_u16(project_path(original["input"]["source_ids_120_2_5"]))
        surface_z = np.load(project_path(cleaned["outputs"]["surface_z"])).astype(np.float32)
        bev_rgb = load_rgb(project_path(cleaned["outputs"]["bev_rgb"]))
        hull_xy = coverage_hull(np.isfinite(surface_z))

        status_by_id: dict[int, dict[str, Any]] = {}
        status_raster = np.zeros(ids.shape, dtype=np.uint8)
        risk_ids = np.zeros(ids.shape, dtype=np.uint16)
        strict_ids = ids.copy()
        conservative_ids = ids.copy()
        for instance_value in np.unique(ids):
            instance_id = int(instance_value)
            if instance_id == 0:
                continue
            mask = ids == instance_id
            canvas_px = canvas_edge_distance_px(mask)
            coverage_px = minimum_distance_to_polygon_edges(mask, hull_xy)
            source = source_largest_component_metrics(source_ids == instance_id)
            status_code, evidence = status_for_evidence(
                canvas_px * METERS_PER_PIXEL,
                coverage_px * METERS_PER_PIXEL,
                bool(source["largest_component_touches_image_edge"]),
            )
            status_name = STATUS_NAMES[status_code]
            row = {
                "sample_id": sample_id,
                "instance_id": instance_id,
                "status_code": status_code,
                "status": status_name,
                "evidence": evidence,
                "likely_boundary_incomplete": status_code >= STATUS_REVIEW,
                "recommended_reconstruct_strict": status_code != STATUS_EXCLUDE,
                "recommended_reconstruct_conservative": status_code == STATUS_INTERIOR,
                "bev_area_m2": float(np.count_nonzero(mask) * METERS_PER_PIXEL**2),
                "bev_canvas_edge_distance_px": canvas_px,
                "bev_canvas_edge_distance_m": canvas_px * METERS_PER_PIXEL,
                "coverage_hull_edge_distance_px": coverage_px,
                "coverage_hull_edge_distance_m": coverage_px * METERS_PER_PIXEL,
                "source_view": source,
                "glb_mesh_name_prefix": f"{sample_id}__instance_{instance_id:04d}__",
            }
            status_by_id[instance_id] = row
            all_rows.append(row)
            totals[
                "interior"
                if status_code == STATUS_INTERIOR
                else "review"
                if status_code == STATUS_REVIEW
                else "exclude"
            ] += 1
            status_raster[mask] = np.uint8(status_code)
            if status_code >= STATUS_REVIEW:
                risk_ids[mask] = np.uint16(instance_id)
            if status_code == STATUS_EXCLUDE:
                strict_ids[mask] = 0
            if status_code != STATUS_INTERIOR:
                conservative_ids[mask] = 0

        status_preview = STATUS_COLORS[status_raster]
        status_overlay = rgb_overlay(bev_rgb, status_preview, status_raster > 0, 0.70)
        hull_preview = draw_coverage_hull(bev_rgb, hull_xy)
        paths = {
            "status_json": directories["instance_status_tables"] / f"{sample_id}__instance_boundary_status.json",
            "status_csv": directories["instance_status_csv"] / f"{sample_id}__instance_boundary_status.csv",
            "status_ids": directories["boundary_status_ids"] / f"{sample_id}__boundary_status_ids.png",
            "status_preview": directories["boundary_status_previews"] / f"{sample_id}__boundary_status_preview.png",
            "status_overlay": directories["boundary_status_overlays"] / f"{sample_id}__boundary_status_overlay.png",
            "hull_preview": directories["coverage_hull_previews"] / f"{sample_id}__coverage_hull_preview.png",
            "risk_ids": directories["boundary_risk_instance_ids"] / f"{sample_id}__boundary_risk_instance_ids.png",
            "strict_ids": directories["reconstructable_ids_strict"] / f"{sample_id}__reconstructable_ids_strict.png",
            "conservative_ids": directories["reconstructable_ids_conservative"] / f"{sample_id}__reconstructable_ids_conservative.png",
            "point_status": directories["point_boundary_status_npz"] / f"{sample_id}__point_boundary_status.npz",
            "ply_all": directories["point_cloud_ply_status_all"] / f"{sample_id}__point_boundary_status.ply",
            "ply_interior": directories["point_cloud_ply_interior"] / f"{sample_id}__interior_points.ply",
            "ply_review": directories["point_cloud_ply_review"] / f"{sample_id}__review_points.ply",
            "ply_exclude": directories["point_cloud_ply_exclude"] / f"{sample_id}__exclude_points.ply",
            "prism_status": directories["prism_instance_status"] / f"{sample_id}__prism_instance_status.json",
            "geojson": directories["footprint_geojson_with_boundary_status"] / f"{sample_id}__footprints_with_boundary_status.geojson",
            "glb_status": directories["building_prisms_glb_boundary_status"] / f"{sample_id}__building_prisms_boundary_status.glb",
            "glb_status_ue": directories["building_prisms_glb_boundary_status_ue_z_up"] / f"{sample_id}__building_prisms_boundary_status_ue_z_up.glb",
            "comparison": directories["comparisons"] / f"{sample_id}__boundary_instance_tagging.png",
        }

        status_records = [status_by_id[key] for key in sorted(status_by_id)]
        paths["status_json"].write_text(
            json.dumps(status_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        csv_fields = [
            "sample_id",
            "instance_id",
            "status_code",
            "status",
            "likely_boundary_incomplete",
            "recommended_reconstruct_strict",
            "recommended_reconstruct_conservative",
            "bev_area_m2",
            "bev_canvas_edge_distance_m",
            "coverage_hull_edge_distance_m",
            "evidence",
        ]
        with paths["status_csv"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields)
            writer.writeheader()
            for row in status_records:
                flat = {key: row[key] for key in csv_fields}
                flat["evidence"] = "|".join(row["evidence"])
                writer.writerow(flat)
        save_u8(paths["status_ids"], status_raster)
        save_rgb(paths["status_preview"], status_preview)
        save_rgb(paths["status_overlay"], status_overlay)
        save_rgb(paths["hull_preview"], hull_preview)
        save_u16(paths["risk_ids"], risk_ids)
        save_u16(paths["strict_ids"], strict_ids)
        save_u16(paths["conservative_ids"], conservative_ids)

        dense_npz = np.load(project_path(cleaned["outputs"]["building_point_cloud_npz"]))
        xyz = dense_npz["xyz_m"]
        point_rgb = dense_npz["rgb"]
        point_ids = dense_npz["instance_id"]
        point_status = np.zeros(point_ids.shape, dtype=np.uint8)
        for instance_id, row in status_by_id.items():
            point_status[point_ids == instance_id] = np.uint8(row["status_code"])
        if np.any(point_status == 0):
            raise AssertionError(f"Unlabelled E10 points: {sample_id}")
        np.savez_compressed(
            paths["point_status"],
            instance_id=point_ids.astype(np.uint16),
            boundary_status=point_status,
            recommended_reconstruct_strict=(point_status != STATUS_EXCLUDE),
            recommended_reconstruct_conservative=(point_status == STATUS_INTERIOR),
        )

        state_rgb = STATUS_COLORS[point_status]
        selected = stratified_indices(point_status, args.max_ply_points)
        all_meta = write_binary_ply(
            paths["ply_all"], xyz[selected], state_rgb[selected], point_ids[selected], selected.size
        )
        class_outputs: dict[str, Any] = {}
        for code, name, path in (
            (STATUS_INTERIOR, "interior", paths["ply_interior"]),
            (STATUS_REVIEW, "review", paths["ply_review"]),
            (STATUS_EXCLUDE, "exclude", paths["ply_exclude"]),
        ):
            chosen = point_status == code
            meta = write_optional_ply(
                path,
                xyz[chosen],
                point_rgb[chosen],
                point_ids[chosen],
                args.max_ply_points,
            )
            class_outputs[name] = {
                "point_count": int(np.count_nonzero(chosen)),
                "ply": relative(path) if meta else None,
                "ply_metadata": meta,
            }

        prism_status = {
            "sample_id": sample_id,
            "source_glb": prism["outputs"]["glb"],
            "source_glb_ue_z_up": prism["outputs"]["glb_ue_z_up"],
            "mesh_name_rule": f"{sample_id}__instance_{{instance_id:04d}}__part_{{component_index:02d}}",
            "instances": status_records,
        }
        paths["prism_status"].write_text(
            json.dumps(prism_status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        augment_geojson(
            project_path(prism["outputs"]["geojson"]),
            paths["geojson"],
            status_by_id,
        )
        glb_status_meta = export_boundary_colored_glb(
            project_path(prism["outputs"]["glb"]), paths["glb_status"], status_by_id
        )
        glb_status_ue_meta = export_boundary_colored_glb(
            project_path(prism["outputs"]["glb_ue_z_up"]),
            paths["glb_status_ue"],
            status_by_id,
        )

        two_row_comparison(
            f"{sample_id} / E10 BOUNDARY QUALITY TAGGING",
            [
                ("BEV RGB", "E10 camera-aligned surface", bev_rgb, False),
                ("E10 Instance IDs", f"instances={len(status_records)}", instance_preview(ids), True),
                ("Coverage hull", "cyan = projected coverage envelope", hull_preview, False),
                ("Boundary status", "green keep / yellow review / red exclude", status_preview, True),
            ],
            [
                ("Status overlay", "boundary evidence on BEV", status_overlay, False),
                ("Strict reconstruct", "remove red only", instance_preview(strict_ids), True),
                ("Conservative reconstruct", "keep green only", instance_preview(conservative_ids), True),
                ("Boundary-risk IDs", "yellow + red instances", instance_preview(risk_ids), True),
            ],
            paths["comparison"],
            tile=360,
            row_names=("BOUNDARY EVIDENCE", "RECONSTRUCTION SELECTION"),
        )
        comparisons.append(paths["comparison"])
        sample_counts = {
            name: sum(row["status_code"] == code for row in status_records)
            for code, name in (
                (STATUS_INTERIOR, "interior_keep"),
                (STATUS_REVIEW, "boundary_risk_review"),
                (STATUS_EXCLUDE, "boundary_clipped_exclude"),
            )
        }
        records.append(
            {
                "id": sample_id,
                "instance_count": len(status_records),
                "status_counts": sample_counts,
                "outputs": {
                    "instance_status_json": relative(paths["status_json"]),
                    "instance_status_csv": relative(paths["status_csv"]),
                    "boundary_status_ids": relative(paths["status_ids"]),
                    "boundary_status_preview": relative(paths["status_preview"]),
                    "boundary_status_overlay": relative(paths["status_overlay"]),
                    "coverage_hull_preview": relative(paths["hull_preview"]),
                    "boundary_risk_instance_ids": relative(paths["risk_ids"]),
                    "reconstructable_ids_strict": relative(paths["strict_ids"]),
                    "reconstructable_ids_conservative": relative(paths["conservative_ids"]),
                    "point_boundary_status_npz": relative(paths["point_status"]),
                    "point_cloud_status_all": relative(paths["ply_all"]),
                    "point_cloud_status_all_metadata": all_meta,
                    "point_cloud_classes": class_outputs,
                    "prism_instance_status": relative(paths["prism_status"]),
                    "footprint_geojson_with_boundary_status": relative(paths["geojson"]),
                    "building_prisms_glb_boundary_status": relative(paths["glb_status"]),
                    "building_prisms_glb_boundary_status_metadata": glb_status_meta,
                    "building_prisms_glb_boundary_status_ue_z_up": relative(paths["glb_status_ue"]),
                    "building_prisms_glb_boundary_status_ue_z_up_metadata": glb_status_ue_meta,
                    "comparison": relative(paths["comparison"]),
                },
            }
        )

    overview = output_root / f"all_{len(records)}_boundary_instance_tagging.png"
    build_overview(comparisons, overview)
    global_csv = output_root / "all_instance_boundary_status.csv"
    csv_fields = [
        "sample_id",
        "instance_id",
        "status_code",
        "status",
        "likely_boundary_incomplete",
        "recommended_reconstruct_strict",
        "recommended_reconstruct_conservative",
        "bev_area_m2",
        "bev_canvas_edge_distance_m",
        "coverage_hull_edge_distance_m",
        "evidence",
    ]
    with global_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            flat = {key: row[key] for key in csv_fields}
            flat["evidence"] = "|".join(row["evidence"])
            writer.writerow(flat)

    manifest = {
        "schema": "air2vectorcity.height_guided_instance_cleaning_e10.boundary_tagging.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete E10 boundary-quality module; original E10 geometry unchanged",
        "inputs": {
            "e09": relative(e09_root),
            "e10": relative(e10_root),
            "e10_building_prisms": relative(e10_root / "building_prisms"),
        },
        "policy": {
            "bev_meters_per_pixel": METERS_PER_PIXEL,
            "unit_of_classification": "sample-local building instance ID",
            "bev_exact_clip": "instance touches the 2048x2048 canvas edge",
            "bev_near_boundary_m": BEV_NEAR_BOUNDARY_M,
            "coverage_envelope": "convex hull of finite E09/E10 projected surface cells",
            "coverage_confirmed_m": COVERAGE_CONFIRMED_M,
            "coverage_near_boundary_m": COVERAGE_NEAR_BOUNDARY_M,
            "source_crop_evidence": "largest connected source-view component touches the source image edge",
            "exclude_rule": "exact BEV clip OR (coverage hull contact <=0.25m AND source crop evidence)",
            "review_rule": "not excluded AND any source crop, <=2m BEV edge, or <=2m coverage hull evidence",
            "interior_rule": "none of the boundary-risk evidence above",
        },
        "status_codes": {
            "0": "background",
            "1": "interior_keep",
            "2": "boundary_risk_review",
            "3": "boundary_clipped_exclude",
        },
        "reconstruction_modes": {
            "strict": "keep green and yellow; exclude red",
            "conservative": "keep green only; exclude yellow and red",
        },
        "totals": {
            "instance_view_pairs": len(all_rows),
            "interior_keep": totals["interior"],
            "boundary_risk_review": totals["review"],
            "boundary_clipped_exclude": totals["exclude"],
        },
        "all_instance_csv": relative(global_csv),
        "overview": relative(overview),
        "samples": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    e10_manifest_path = e10_root / "manifest.json"
    e10["quality_modules"] = e10.get("quality_modules", {})
    e10["quality_modules"]["boundary_instance_tagging"] = {
        "status": "complete",
        "manifest": relative(manifest_path),
        "instance_view_pairs": len(all_rows),
        "interior_keep": totals["interior"],
        "boundary_risk_review": totals["review"],
        "boundary_clipped_exclude": totals["exclude"],
        "standard_status_glb_directory": relative(
            directories["building_prisms_glb_boundary_status"]
        ),
        "ue_z_up_status_glb_directory": relative(
            directories["building_prisms_glb_boundary_status_ue_z_up"]
        ),
        "updated_at": manifest["created_at"],
    }
    e10_manifest_path.write_text(
        json.dumps(e10, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overview)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
