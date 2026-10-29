#!/usr/bin/env python3
"""Build an explicit per-instance footprint-plus-height representation for E10."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = DATA_ROOT
EXPERIMENT_ROOT = Path("UNCONFIGURED_EXPERIMENT_ROOT")
DEFAULT_E10_ROOT = EXPERIMENT_ROOT / "camera_aligned_25d_e10_height_guided_instance_cleaning"
DEFAULT_OUTPUT_ROOT = DEFAULT_E10_ROOT / "footprint_height_representation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e10-root", type=Path, default=DEFAULT_E10_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
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


def close_ring(coordinates: list[list[float]] | list[list[int]]) -> list[list[float]] | list[list[int]]:
    if not coordinates:
        raise ValueError("A footprint ring cannot be empty")
    return coordinates if coordinates[0] == coordinates[-1] else [*coordinates, coordinates[0]]


def geojson_geometry(components: list[dict[str, Any]], key: str) -> dict[str, Any]:
    polygons = [[[point for point in close_ring(component[key])]] for component in components]
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def main() -> int:
    args = parse_args()
    e10_root = args.e10_root.resolve()
    output_root = args.output_root.resolve()
    e10 = read_json(e10_root / "manifest.json")
    prism_manifest = read_json(e10_root / "building_prisms" / "manifest.json")
    boundary_manifest_path = e10_root / "boundary_instance_tagging" / "manifest.json"
    boundary_manifest = read_json(boundary_manifest_path)
    e10_by_id = {sample["id"]: sample for sample in e10["samples"]}
    prism_by_id = {sample["id"]: sample for sample in prism_manifest["samples"]}
    boundary_by_id = {sample["id"]: sample for sample in boundary_manifest["samples"]}
    sample_ids = args.sample_id or [sample["id"] for sample in e10["samples"]]

    records_root = output_root / "instance_records"
    exact_geojson_root = output_root / "geojson_exact_world_xy"
    modeling_geojson_root = output_root / "geojson_modeling_world_xy"
    for path in (records_root, exact_geojson_root, modeling_geojson_root):
        path.mkdir(parents=True, exist_ok=True)

    sample_records: list[dict[str, Any]] = []
    all_exact_features: list[dict[str, Any]] = []
    all_modeling_features: list[dict[str, Any]] = []
    total_instances = 0
    total_components = 0
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        e10_sample = e10_by_id[sample_id]
        prism_sample = prism_by_id[sample_id]
        boundary_sample = boundary_by_id[sample_id]
        boundaries_path = project_path(prism_sample["outputs"]["boundaries"])
        heights_path = project_path(prism_sample["outputs"]["height_json"])
        status_path = project_path(boundary_sample["outputs"]["instance_status_json"])
        boundaries = read_json(boundaries_path)
        heights = read_json(heights_path)
        statuses = read_json(status_path)
        boundary_by_instance = {
            int(record["instance_id"]): record for record in boundaries["instances"]
        }
        height_by_instance = {int(record["instance_id"]): record for record in heights}
        status_by_instance = {int(record["instance_id"]): record for record in statuses}
        final_ids = np.asarray(
            Image.open(project_path(e10_sample["outputs"]["final_ids"])), dtype=np.uint16
        )
        raster_instance_ids = {int(value) for value in np.unique(final_ids) if value != 0}
        if not (
            raster_instance_ids
            == set(boundary_by_instance)
            == set(height_by_instance)
            == set(status_by_instance)
        ):
            raise AssertionError(f"E10 footprint/height/status ID mismatch: {sample_id}")

        buildings: list[dict[str, Any]] = []
        exact_features: list[dict[str, Any]] = []
        modeling_features: list[dict[str, Any]] = []
        for instance_id in sorted(raster_instance_ids):
            boundary = boundary_by_instance[instance_id]
            height = height_by_instance[instance_id]
            status = status_by_instance[instance_id]
            components = boundary["components"]
            component_count = len(components)
            total_area_m2 = float(
                sum(float(component["raster_contour_area_m2"]) for component in components)
            )
            exact_geometry = geojson_geometry(components, "boundary_world_xy_m")
            modeling_geometry = geojson_geometry(components, "mesh_boundary_world_xy_m")
            common_properties = {
                "sample_id": sample_id,
                "sample_instance_key": f"{sample_id}__instance_{instance_id:04d}",
                "instance_id": instance_id,
                "component_count": component_count,
                "footprint_area_m2": total_area_m2,
                "base_z_m": float(height["base_z_m"]),
                "top_z_m": float(height["top_z_m"]),
                "height_m": float(height["extrusion_height_m"]),
                "height_method": height["method"],
                "boundary_status_code": int(status["status_code"]),
                "boundary_status": status["status"],
                "recommended_reconstruct_strict": bool(
                    status["recommended_reconstruct_strict"]
                ),
                "recommended_reconstruct_conservative": bool(
                    status["recommended_reconstruct_conservative"]
                ),
            }
            exact_feature = {
                "type": "Feature",
                "properties": {**common_properties, "footprint_variant": "exact_raster_boundary"},
                "geometry": exact_geometry,
            }
            modeling_feature = {
                "type": "Feature",
                "properties": {**common_properties, "footprint_variant": "mesh_simplified_boundary"},
                "geometry": modeling_geometry,
            }
            exact_features.append(exact_feature)
            modeling_features.append(modeling_feature)
            all_exact_features.append(exact_feature)
            all_modeling_features.append(modeling_feature)
            buildings.append(
                {
                    **common_properties,
                    "representation": "base footprint vertically extruded from base_z_m to top_z_m",
                    "exact_raster_footprint": {
                        "geometry_world_xy_m": exact_geometry,
                        "components": [
                            {
                                "component_index": int(component["component_index"]),
                                "raster_contour_area_m2": float(
                                    component["raster_contour_area_m2"]
                                ),
                                "boundary_pixel_count": int(
                                    component["boundary_pixel_count"]
                                ),
                                "boundary_pixels_uv": close_ring(
                                    component["boundary_pixels_uv"]
                                ),
                                "boundary_world_xy_m": close_ring(
                                    component["boundary_world_xy_m"]
                                ),
                            }
                            for component in components
                        ],
                    },
                    "modeling_footprint": {
                        "geometry_world_xy_m": modeling_geometry,
                        "simplification_epsilon_px": prism_manifest["footprint_policy"][
                            "mesh_simplification_epsilon_px"
                        ],
                        "components": [
                            {
                                "component_index": int(component["component_index"]),
                                "mesh_boundary_pixels_uv": close_ring(
                                    component["mesh_boundary_pixels_uv"]
                                ),
                                "mesh_boundary_world_xy_m": close_ring(
                                    component["mesh_boundary_world_xy_m"]
                                ),
                                "mesh_vertex_count_2d": int(
                                    component["mesh_vertex_count_2d"]
                                ),
                                "triangulation_convex_hull_fallback": bool(
                                    component["triangulation_convex_hull_fallback"]
                                ),
                            }
                            for component in components
                        ],
                    },
                    "height": height,
                    "boundary_quality": {
                        "status_code": int(status["status_code"]),
                        "status": status["status"],
                        "evidence": status["evidence"],
                        "recommended_reconstruct_strict": bool(
                            status["recommended_reconstruct_strict"]
                        ),
                        "recommended_reconstruct_conservative": bool(
                            status["recommended_reconstruct_conservative"]
                        ),
                    },
                    "mesh_names": [
                        f"{sample_id}__instance_{instance_id:04d}__part_{int(component['component_index']):02d}"
                        for component in components
                    ],
                }
            )

        record_path = records_root / f"{sample_id}__footprint_height_instances.json"
        exact_path = exact_geojson_root / f"{sample_id}__footprint_height_exact.geojson"
        modeling_path = (
            modeling_geojson_root / f"{sample_id}__footprint_height_modeling.geojson"
        )
        record_path.write_text(
            json.dumps(
                {
                    "schema": "air2vectorcity.e10.instance_footprint_height.v1",
                    "sample_id": sample_id,
                    "coordinate_system": "UE world XY/Z metres; raster boundaries use camera-aligned pixel u/v",
                    "pixel_to_world_transform": boundaries["pixel_to_world_transform"],
                    "source_modeling_instance_ids": e10_sample["outputs"]["final_ids"],
                    "source_exact_boundaries": relative(boundaries_path),
                    "source_height_table": relative(heights_path),
                    "source_boundary_status": relative(status_path),
                    "buildings": buildings,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        exact_path.write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": exact_features},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        modeling_path.write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": modeling_features},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        total_instances += len(buildings)
        total_components += sum(building["component_count"] for building in buildings)
        sample_records.append(
            {
                "id": sample_id,
                "instance_count": len(buildings),
                "component_count": sum(
                    building["component_count"] for building in buildings
                ),
                "outputs": {
                    "instance_records": relative(record_path),
                    "exact_world_xy_geojson": relative(exact_path),
                    "modeling_world_xy_geojson": relative(modeling_path),
                },
            }
        )

    all_exact_path = output_root / "all_instances_exact_world_xy.geojson"
    all_modeling_path = output_root / "all_instances_modeling_world_xy.geojson"
    all_exact_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": all_exact_features},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    all_modeling_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": all_modeling_features},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "air2vectorcity.e10.footprint_height_representation.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete E10 footprint-plus-height representation",
        "input_e10": relative(e10_root),
        "semantics": {
            "building": "one sample-local Instance represented by one Polygon or MultiPolygon footprint plus one base/top height pair",
            "exact_footprint": "full pixel contour from the final E10 modeling Instance mask",
            "modeling_footprint": "the 2-pixel simplified contour used to construct the existing prism Mesh",
            "height": "base_z_m, top_z_m, and extrusion_height_m from the E10 stable-height policy",
            "mesh_relation": "existing GLB is the modeling footprint vertically extruded over the saved height interval",
        },
        "totals": {
            "instance_view_pairs": total_instances,
            "footprint_components": total_components,
        },
        "all_exact_world_xy_geojson": relative(all_exact_path),
        "all_modeling_world_xy_geojson": relative(all_modeling_path),
        "samples": sample_records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    e10["representations"] = e10.get("representations", {})
    e10["representations"]["footprint_height"] = {
        "status": "complete",
        "manifest": relative(manifest_path),
        "instance_view_pairs": total_instances,
        "footprint_components": total_components,
        "updated_at": manifest["created_at"],
    }
    (e10_root / "manifest.json").write_text(
        json.dumps(e10, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
