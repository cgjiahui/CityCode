#!/usr/bin/env python3
"""Create a hole-free base mesh plus a true elevated-road mesh layer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--residual", type=float, default=0.01)
    return parser.parse_args()


def query_layer_arrays(local_points, *, frame, cx, cy, zc, x0, y0, source_mpp, alpha, arrays):
    world_x = cx + frame * local_points[:, 0]
    world_y = cy + frame * local_points[:, 2]
    su = np.floor((world_x - x0) / source_mpp).astype(np.int64)
    sv = np.floor((world_y - y0) / source_mpp).astype(np.int64)
    inside = (su >= 0) & (su < alpha.shape[1]) & (sv >= 0) & (sv < alpha.shape[0])
    alpha_valid = np.zeros(len(local_points), bool)
    alpha_valid[inside] = alpha[sv[inside], su[inside]] > 204

    result = {}
    for layer in ("base", "elevated"):
        height = np.full(len(local_points), np.nan, np.float32)
        confidence = np.zeros(len(local_points), np.float32)
        height[inside] = arrays[f"{layer}_height"][sv[inside], su[inside]]
        confidence[inside] = arrays[f"{layer}_confidence"][sv[inside], su[inside]]
        result[f"{layer}_height"] = height
        result[f"{layer}_confidence"] = confidence
        # Validity and confidence are deliberately independent.
        result[f"{layer}_valid"] = inside & alpha_valid & np.isfinite(height)
    result["predicted_height"] = zc + frame * local_points[:, 1]
    return result


def project_geometry(geometry, layer, query, frame, zc, residual):
    vertices = np.asarray(geometry.vertices).copy()
    faces = np.asarray(geometry.faces)
    sampled = query(vertices)
    valid = sampled[f"{layer}_valid"]
    target = sampled[f"{layer}_height"].copy()
    confidence = sampled[f"{layer}_confidence"].copy()

    direct_targets = np.where(valid, target, np.nan)
    face_targets = np.nanmean(direct_targets[faces], axis=1)
    direct_confidence = np.where(valid, confidence, np.nan)
    face_confidence = np.nanmean(direct_confidence[faces], axis=1)
    neighbor_sum = np.zeros(len(vertices), np.float64)
    confidence_sum = np.zeros(len(vertices), np.float64)
    neighbor_count = np.zeros(len(vertices), np.int32)
    finite_face = np.isfinite(face_targets)
    for corner in range(3):
        indices = faces[:, corner]
        np.add.at(neighbor_sum, indices, np.nan_to_num(face_targets))
        np.add.at(confidence_sum, indices, np.nan_to_num(face_confidence))
        np.add.at(neighbor_count, indices, finite_face.astype(np.int32))
    boundary = ~valid & (neighbor_count > 0)
    target[boundary] = neighbor_sum[boundary] / neighbor_count[boundary]
    confidence[boundary] = confidence_sum[boundary] / neighbor_count[boundary]
    projectable = valid | boundary

    predicted = sampled["predicted_height"]
    before = np.abs(predicted[projectable] - target[projectable])
    # Confidence affects strength without turning low-confidence geometry into a
    # hole. Even confidence=0 keeps 80% of the teacher correction.
    confidence_scale = 0.05 if layer == "elevated" else 0.20
    residual_cap = 0.12 if layer == "elevated" else 0.35
    effective_residual = np.clip(
        residual + (1.0 - confidence[projectable]) * confidence_scale,
        residual, residual_cap,
    )
    projected = target[projectable] + effective_residual * (
        predicted[projectable] - target[projectable]
    )
    vertices[projectable, 1] = (projected - zc) / frame
    geometry.vertices = vertices
    return {
        "projected_count": int(projectable.sum()),
        "before": before,
        "after": np.abs(projected - target[projectable]),
        "confidence": confidence[projectable],
    }


def main():
    args = parse_args()
    metadata_path = args.run / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    patch_id = metadata["patch_id"]
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    item = next(record for record in inputs["patches"] if record["patch_id"] == patch_id)
    teacher_manifest = json.loads((args.teacher / "manifest.json").read_text())
    with np.load(item["height_bundle"]) as bundle:
        arrays = {
            "base_height": np.asarray(bundle["base_surface_z_m"], np.float32),
            "base_confidence": np.asarray(bundle["base_confidence"], np.float32),
            "elevated_height": np.asarray(bundle["elevated_surface_z_m"], np.float32),
            "elevated_confidence": np.asarray(bundle["elevated_confidence"], np.float32),
        }
    alpha = np.asarray(Image.open(item["input"]))[..., 3]
    x0, y0, _, _ = map(float, item["data_bounds_xy_world_m"])
    source_mpp = float(item["ortho_meters_per_pixel"])
    contract = metadata["geometry_contract"]
    frame = float(contract["metric_frame_m"])
    cx, cy = map(float, contract["center_world_xy_m"])
    zc = float(contract["z_center_world_m"])

    def query(points):
        return query_layer_arrays(
            points, frame=frame, cx=cx, cy=cy, zc=zc,
            x0=x0, y0=y0, source_mpp=source_mpp, alpha=alpha, arrays=arrays,
        )

    source = args.run / "mesh_pbr_local.glb"
    scene = trimesh.load(source, force="scene", process=False)
    face_before = vertex_before = base_faces = base_vertices = 0
    elevated_faces = elevated_vertices = 0
    base_records = []
    elevated_records = []
    additions = []

    for geometry_name, geometry in list(scene.geometry.items()):
        vertices = np.asarray(geometry.vertices)
        faces = np.asarray(geometry.faces)
        vertex_before += len(vertices)
        face_before += len(faces)

        vertex_samples = query(vertices)
        centroids = vertices[faces].mean(axis=1)
        centroid_samples = query(centroids)
        keep_base = (
            centroid_samples["base_valid"] &
            (vertex_samples["base_valid"][faces].sum(axis=1) >= 2)
        )
        geometry.update_faces(keep_base)
        geometry.remove_unreferenced_vertices()

        vertices = np.asarray(geometry.vertices)
        faces = np.asarray(geometry.faces)
        vertex_samples = query(vertices)
        centroid_samples = query(vertices[faces].mean(axis=1))
        keep_elevated = (
            centroid_samples["elevated_valid"] &
            (vertex_samples["elevated_valid"][faces].sum(axis=1) >= 2)
        )
        if keep_elevated.any():
            elevated_geometry = geometry.submesh([keep_elevated], append=True, repair=False)
            elevated_record = project_geometry(
                elevated_geometry, "elevated", query, frame, zc, args.residual,
            )
            elevated_faces += int(len(elevated_geometry.faces))
            elevated_vertices += int(len(elevated_geometry.vertices))
            elevated_records.append(elevated_record)
            additions.append((f"{geometry_name}_elevated", elevated_geometry))

        base_record = project_geometry(geometry, "base", query, frame, zc, args.residual)
        base_faces += int(len(geometry.faces))
        base_vertices += int(len(geometry.vertices))
        base_records.append(base_record)

    for name, geometry in additions:
        scene.add_geometry(geometry, node_name=name, geom_name=name)

    local_destination = args.run / "mesh_pbr_local_canonical_axis_v3.glb"
    scene.export(local_destination, extension_webp=True)
    world = scene.copy()
    transform = np.array([
        [frame, 0, 0, cx],
        [0, frame, 0, zc],
        [0, 0, frame, cy],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    world.apply_transform(transform)
    world_destination = args.run / "mesh_pbr_world_canonical_axis_v3.glb"
    world.export(world_destination, extension_webp=True)

    def concatenate(records, key):
        chunks = [record[key] for record in records if record[key].size]
        return np.concatenate(chunks) if chunks else np.empty(0)

    base_before = concatenate(base_records, "before")
    base_after = concatenate(base_records, "after")
    elevated_before = concatenate(elevated_records, "before")
    elevated_after = concatenate(elevated_records, "after")
    all_before = np.concatenate([x for x in (base_before, elevated_before) if x.size]) if (base_before.size or elevated_before.size) else np.empty(0)
    all_after = np.concatenate([x for x in (base_after, elevated_after) if x.size]) if (base_after.size or elevated_after.size) else np.empty(0)

    def percentiles(values):
        return np.percentile(values, [50, 90, 99]).tolist() if values.size else []

    stats = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_canonical_mesh.v3",
        "axis_mapping_version": "pbr_local_z_to_ue_y_positive_v3",
        "axis_mapping": "PBR local (x,y-up,z-image-row) -> world (UE X, UE Z-up, UE +Y)",
        "source": str(source),
        "local": str(local_destination),
        "world": str(world_destination),
        "face_policy": (
            "continuous fully-opaque base; elevated semantic faces duplicate the generated "
            "Pixal3D topology/UV/PBR as an independent second mesh layer"
        ),
        "building_mask_policy": "QA only; never an alpha, face-deletion, or height-validity gate",
        "height_layers": ["filled_base", "elevated_road"],
        "teacher_schema": teacher_manifest["schema"],
        "height_residual_scale": args.residual,
        "confidence_residual_policy": "base: r<=0.35 with scale 0.20; elevated: r<=0.12 with scale 0.05",
        "faces_before": face_before,
        "base_faces_after": base_faces,
        "elevated_faces_added": elevated_faces,
        "faces_after": base_faces + elevated_faces,
        "faces_kept_fraction": base_faces / max(face_before, 1),
        "vertices_before": vertex_before,
        "base_vertices_after": base_vertices,
        "elevated_vertices_added": elevated_vertices,
        "vertices_after": base_vertices + elevated_vertices,
        "vertices_projected": sum(x["projected_count"] for x in base_records + elevated_records),
        "base_height_abs_error_before_m_p50_p90_p99": percentiles(base_before),
        "base_height_abs_error_after_m_p50_p90_p99": percentiles(base_after),
        "elevated_height_abs_error_before_m_p50_p90_p99": percentiles(elevated_before),
        "elevated_height_abs_error_after_m_p50_p90_p99": percentiles(elevated_after),
        "height_abs_error_before_m_p50_p90_p99": percentiles(all_before),
        "height_abs_error_after_m_p50_p90_p99": percentiles(all_after),
        "world_bounds": world.bounds.tolist(),
        "world_size_bytes": world_destination.stat().st_size,
    }
    canonical_metadata_path = args.run / "canonical_metadata.json"
    historical_metadata_path = args.run / "canonical_metadata_single_surface_v1.json"
    if canonical_metadata_path.is_file():
        previous = json.loads(canonical_metadata_path.read_text())
        if previous.get("schema") != stats["schema"] and not historical_metadata_path.exists():
            shutil.copy2(canonical_metadata_path, historical_metadata_path)
    canonical_metadata_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    if "canonical" in metadata and metadata["canonical"].get("schema") != stats["schema"]:
        metadata.setdefault("canonical_single_surface_v1", metadata["canonical"])
    metadata["canonical"] = stats
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()

