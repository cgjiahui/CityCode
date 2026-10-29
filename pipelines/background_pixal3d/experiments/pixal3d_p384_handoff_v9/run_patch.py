#!/usr/bin/env python3
"""Run P384 Pixal3D with a filled base and optional elevated-road layer.

The guided mode injects the dataset world-height surfaces into Pixal3D's 32^3
sparse support before either LR or HR shape SLat is sampled.  This is therefore
a generation-time constraint, not a post-hoc mesh height crop.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.pixal3d_full_detail import run_batch as base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=ROOT / "experiments/pixal3d_p384_handoff_v9/outputs/inputs_all_filled_v1")
    parser.add_argument("--teacher", type=Path, default=ROOT / "experiments/pixal3d_p384_handoff_v9/outputs/global_teacher_filled_v1_015m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "structure_guided"), required=True)
    parser.add_argument("--patch-id", action="append")
    parser.add_argument("--model", type=Path, default=ROOT / "third_party/models/Pixal3D")
    parser.add_argument("--dino", type=Path, default=ROOT / "third_party/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--moge", type=Path, default=ROOT / "third_party/models/moge-2-vitl")
    parser.add_argument("--birefnet", type=Path, default=ROOT / "third_party/models/BiRefNet")
    parser.add_argument("--naf", type=Path, default=ROOT / "third_party/NAF")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1536)
    parser.add_argument("--surface-band-voxels", type=int, default=0)
    parser.add_argument("--hr-surface-band-voxels", type=int, default=0)
    parser.add_argument("--height-projection-residual", type=float, default=0.05)
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1000000)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def preprocessing_transform(source: Image.Image) -> dict:
    """Reproduce Pixal3D's alpha-bbox preprocessing transform exactly."""
    max_size = max(source.size)
    scale = min(1.0, 1024.0 / max_size)
    resized = source
    if scale < 1:
        resized = source.resize(
            (int(source.width * scale), int(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
    rgba = np.asarray(resized)
    alpha = rgba[..., 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if not foreground.size:
        raise ValueError("Input alpha contains no foreground")
    raw_bbox = [
        int(np.min(foreground[:, 1])), int(np.min(foreground[:, 0])),
        int(np.max(foreground[:, 1])), int(np.max(foreground[:, 0])),
    ]
    center = ((raw_bbox[0] + raw_bbox[2]) / 2, (raw_bbox[1] + raw_bbox[3]) / 2)
    size = int(max(raw_bbox[2] - raw_bbox[0], raw_bbox[3] - raw_bbox[1]) * 1.1)
    crop = [
        int(center[0] - size // 2), int(center[1] - size // 2),
        int(center[0] + size // 2), int(center[1] + size // 2),
    ]
    return {
        "source_size_px": list(source.size),
        "resize_scale": scale,
        "resized_size_px": list(resized.size),
        "alpha_bbox_resized_px": raw_bbox,
        "crop_bbox_resized_px": crop,
        "crop_size_px": [crop[2] - crop[0], crop[3] - crop[1]],
    }


def load_array(path: Path):
    try:
        return np.load(path, mmap_mode="r")
    except OSError as error:
        if error.errno != 19:
            raise
        return np.load(path)


class MultiLayerHeightGuide:
    def __init__(self, teacher: Path, band_voxels: int):
        self.root = teacher
        self.manifest = json.loads((teacher / "manifest.json").read_text())
        outputs = self.manifest["outputs"]
        self.base_height = load_array(teacher / outputs["base_height_m"])
        self.base_confidence = load_array(teacher / outputs["base_confidence"])
        self.elevated_height = load_array(teacher / outputs["elevated_height_m"])
        self.elevated_confidence = load_array(teacher / outputs["elevated_confidence"])
        self.bounds = np.asarray(self.manifest["world_bounds_xy_m"], np.float64)
        self.mpp = float(self.manifest["resolution_m"])
        self.band = int(band_voxels)
        self.context = None
        self.last_stats = None
        self.stage_stats = {}

    def set_context(self, item: dict, source: Image.Image, transform: dict):
        bounds = np.asarray(item["data_bounds_xy_world_m"], np.float64)
        source_mpp = float(item["ortho_meters_per_pixel"])
        crop = transform["crop_bbox_resized_px"]
        scale = float(transform["resize_scale"])
        frame_width_m = (crop[2] - crop[0]) / scale * source_mpp
        frame_height_m = (crop[3] - crop[1]) / scale * source_mpp
        if abs(frame_width_m - frame_height_m) > source_mpp * 2:
            raise ValueError(f"Non-square preprocessed metric frame: {frame_width_m}, {frame_height_m}")
        frame_m = (frame_width_m + frame_height_m) / 2
        source_center_u = ((crop[0] + crop[2]) / 2) / scale
        source_center_v = ((crop[1] + crop[3]) / 2) / scale
        center_world_x = bounds[0] + source_center_u * source_mpp
        center_world_y = bounds[1] + source_center_v * source_mpp
        self.context = {
            "item": item,
            "alpha": np.asarray(source)[..., 3],
            "source_mpp": source_mpp,
            "frame_m": frame_m,
            "center_world_xy_m": [center_world_x, center_world_y],
            "preprocess_transform": transform,
            "z_anchor_m": None,
        }
        self.last_stats = None
        self.stage_stats = {}

    def _sample_world(self, world_x, world_y):
        gx = np.floor((world_x - self.bounds[0]) / self.mpp).astype(np.int64)
        gy = np.floor((world_y - self.bounds[1]) / self.mpp).astype(np.int64)
        inside = (
            (gx >= 0) & (gx < self.base_height.shape[1]) &
            (gy >= 0) & (gy < self.base_height.shape[0])
        )
        base_z = np.full(world_x.shape, np.nan, np.float32)
        base_conf = np.zeros(world_x.shape, np.float32)
        elevated_z = np.full(world_x.shape, np.nan, np.float32)
        elevated_conf = np.zeros(world_x.shape, np.float32)
        base_z[inside] = self.base_height[gy[inside], gx[inside]]
        base_conf[inside] = self.base_confidence[gy[inside], gx[inside]]
        elevated_z[inside] = self.elevated_height[gy[inside], gx[inside]]
        elevated_conf[inside] = self.elevated_confidence[gy[inside], gx[inside]]
        return base_z, base_conf, elevated_z, elevated_conf

    def select_targets(self, world_x, world_y, predicted_z):
        base_z, base_conf, elevated_z, elevated_conf = self._sample_world(world_x, world_y)
        base_valid = np.isfinite(base_z)
        elevated_valid = np.isfinite(elevated_z)
        base_error = np.where(base_valid, np.abs(predicted_z - base_z), np.inf)
        elevated_error = np.where(elevated_valid, np.abs(predicted_z - elevated_z), np.inf)
        choose_elevated = elevated_error < base_error
        target_z = np.where(choose_elevated, elevated_z, base_z)
        confidence = np.where(choose_elevated, elevated_conf, base_conf)
        valid = base_valid | elevated_valid
        layer = np.where(choose_elevated, 1, 0).astype(np.uint8)
        return valid, target_z, confidence, layer

    def make_surface(self, resolution: int):
        if self.context is None:
            raise RuntimeError("Guide context was not set")
        axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution - 0.5
        mesh_x, mesh_y = np.meshgrid(axis, axis, indexing="xy")
        frame_m = self.context["frame_m"]
        center_x, center_y = self.context["center_world_xy_m"]
        world_x = center_x + mesh_x * frame_m
        # Pixal3D's exported local Z follows the ortho image row direction.
        # Air2VectorCity defines increasing image rows as UE +Y, so this must
        # be a positive mapping.  The previous negative sign mirrored every
        # patch vertically in the final world scene.
        world_y = center_y + mesh_y * frame_m
        base_z, base_conf, elevated_z, elevated_conf = self._sample_world(world_x, world_y)

        item = self.context["item"]
        bounds = np.asarray(item["data_bounds_xy_world_m"], np.float64)
        source_mpp = self.context["source_mpp"]
        source_u = np.floor((world_x - bounds[0]) / source_mpp).astype(np.int64)
        source_v = np.floor((world_y - bounds[1]) / source_mpp).astype(np.int64)
        alpha = self.context["alpha"]
        source_inside = (
            (source_u >= 0) & (source_u < alpha.shape[1]) &
            (source_v >= 0) & (source_v < alpha.shape[0])
        )
        alpha_valid = np.zeros(source_inside.shape, dtype=bool)
        alpha_valid[source_inside] = alpha[source_v[source_inside], source_u[source_inside]] > 204
        base_valid = source_inside & alpha_valid & np.isfinite(base_z)
        elevated_valid = source_inside & alpha_valid & np.isfinite(elevated_z)
        if not base_valid.any():
            raise ValueError("No valid global teacher samples map into the Pixal3D frame")

        if self.context["z_anchor_m"] is None:
            self.context["z_anchor_m"] = float(np.median(base_z[base_valid]))
        z_center = float(self.context["z_anchor_m"])
        world_z = np.stack((base_z, elevated_z), axis=0)
        confidence = np.stack((base_conf, elevated_conf), axis=0)
        valid = np.stack((base_valid, elevated_valid), axis=0)
        mesh_z = (world_z - z_center) / frame_m
        target_z = np.full(world_z.shape, -1, np.int32)
        finite_z = np.isfinite(mesh_z)
        target_z[finite_z] = np.rint((mesh_z[finite_z] + 0.5) * (resolution - 1)).astype(np.int32)
        valid &= (target_z >= 0) & (target_z < resolution)
        return {
            "valid": valid,
            "target_z": target_z,
            "world_z": world_z,
            "confidence": confidence,
            "z_center_m": z_center,
            "frame_m": frame_m,
            "center_world_xy_m": [center_x, center_y],
        }

    def constrain(self, coords: torch.Tensor, resolution: int, *, stage: str, band: int | None = None):
        surface = self.make_surface(resolution)
        band = self.band if band is None else int(band)
        valid = surface["valid"]
        target_z = surface["target_z"]
        confidence = surface["confidence"]
        original = coords.detach().cpu().numpy().astype(np.int32)
        ys0, xs0, zs0 = original[:, 2], original[:, 1], original[:, 3]
        layer_distance = np.full((2, len(original)), np.iinfo(np.int32).max, np.int32)
        layer_confidence = np.zeros((2, len(original)), np.float32)
        for layer in range(2):
            layer_valid = valid[layer, ys0, xs0]
            layer_distance[layer, layer_valid] = np.abs(
                zs0[layer_valid] - target_z[layer, ys0[layer_valid], xs0[layer_valid]]
            )
            layer_confidence[layer, layer_valid] = confidence[layer, ys0[layer_valid], xs0[layer_valid]]
        chosen_layer = np.argmin(layer_distance, axis=0)
        chosen_distance = layer_distance[chosen_layer, np.arange(len(original))]
        chosen_confidence = layer_confidence[chosen_layer, np.arange(len(original))]
        applicable = chosen_distance < np.iinfo(np.int32).max
        # A 32^3 voxel is about 13 m, so only 96^3 low-confidence support gets
        # one extra voxel of model freedom.
        confidence_slack = ((resolution >= 64) & (chosen_confidence < 0.45)).astype(np.int32)
        keep = applicable & (chosen_distance <= band + confidence_slack)
        kept = original[keep]

        prior = []
        layer_prior_counts = []
        for layer in range(2):
            ys, xs = np.nonzero(valid[layer])
            center_z = target_z[layer, ys, xs]
            layer_conf = confidence[layer, ys, xs]
            layer_prior = []
            for dz in range(-band, band + 1):
                accepted = (center_z + dz >= 0) & (center_z + dz < resolution)
                if dz:
                    accepted &= layer_conf >= 0.70
                layer_prior.append(np.stack([
                    np.zeros(int(accepted.sum()), np.int32), xs[accepted], ys[accepted], center_z[accepted] + dz
                ], axis=1))
            layer_prior = np.concatenate(layer_prior, axis=0) if layer_prior else np.empty((0, 4), np.int32)
            prior.append(layer_prior)
            layer_prior_counts.append(int(len(layer_prior)))
        prior = np.concatenate(prior, axis=0)
        guided = np.unique(np.concatenate([kept, prior], axis=0), axis=0)
        self.last_stats = {
            "mode": "structure_guided",
            "stage": stage,
            "resolution": resolution,
            "surface_band_voxels": band,
            "outside_teacher_policy": "prune",
            "original_tokens": int(len(original)),
            "original_tokens_in_teacher_xy": int(applicable.sum()),
            "pruned_outside_teacher_tokens": int((~applicable).sum()),
            "pruned_off_surface_tokens": int((applicable & ~keep).sum()),
            "kept_original_tokens": int(len(kept)),
            "prior_surface_tokens_before_union": int(len(prior)),
            "guided_tokens": int(len(guided)),
            "valid_surface_xy_cells": int(valid.sum()),
            "valid_base_xy_cells": int(valid[0].sum()),
            "valid_elevated_xy_cells": int(valid[1].sum()),
            "valid_surface_xy_fraction": float(valid.any(axis=0).mean()),
            "layer_prior_tokens": {"base": layer_prior_counts[0], "elevated": layer_prior_counts[1]},
            "z_center_world_m": surface["z_center_m"],
            "metric_frame_m": surface["frame_m"],
            "center_world_xy_m": surface["center_world_xy_m"],
            "teacher_height_percentiles_m": np.percentile(surface["world_z"][valid], [0, 1, 50, 99, 100]).tolist(),
            "teacher_confidence_percentiles": np.percentile(surface["confidence"][valid], [0, 50, 100]).tolist(),
        }
        self.stage_stats[stage] = self.last_stats
        return torch.as_tensor(guided, dtype=coords.dtype, device=coords.device)


def save_raw_mesh(path: Path, mesh, shape, texture, grid, camera, geometry_contract, guide_stats):
    torch.save({
        "vertices": mesh.vertices.detach().cpu(),
        "faces": mesh.faces.detach().cpu(),
        "coords": mesh.coords.detach().cpu(),
        "attrs": mesh.attrs.detach().cpu(),
        "grid_size": int(grid),
        "origin": mesh.origin.detach().cpu(),
        "voxel_size": float(mesh.voxel_size),
        "voxel_shape": list(mesh.voxel_shape),
        "layout": {key: [value.start, value.stop] for key, value in mesh.layout.items()},
        "camera_params": camera,
        "geometry_contract": geometry_contract,
        "structure_guide": guide_stats,
        "shape_slat_coords": shape.coords.detach().cpu(),
        "texture_slat_coords": texture.coords.detach().cpu(),
    }, path)


def export_pbr(args, pipeline, mesh, grid, output: Path, geometry_contract: dict, guide: MultiLayerHeightGuide):
    generated = base.official.o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=grid,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    local_path = output / "mesh_pbr_local.glb"
    generated.export(local_path, extension_webp=True)
    frame = float(geometry_contract["metric_frame_m"])
    cx, cy = geometry_contract["center_world_xy_m"]
    zc = float(geometry_contract["z_center_world_m"])
    projected = generated.copy()
    projection_stats = {
        "enabled": args.mode == "structure_guided",
        "residual_scale": float(args.height_projection_residual),
    }
    if args.mode == "structure_guided":
        changed = 0
        total = 0
        before_error = []
        after_error = []
        geometries = projected.geometry.values() if isinstance(projected, trimesh.Scene) else [projected]
        for geometry in geometries:
            vertices = np.asarray(geometry.vertices).copy()
            world_x = cx + frame * vertices[:, 0]
            world_y = cy + frame * vertices[:, 2]
            predicted_z = zc + frame * vertices[:, 1]
            valid, teacher_z, confidence, layer = guide.select_targets(world_x, world_y, predicted_z)
            before_error.append(np.abs(predicted_z[valid] - teacher_z[valid]))
            effective_residual = np.clip(
                args.height_projection_residual + (1.0 - confidence[valid]) * 0.20,
                args.height_projection_residual, 0.35,
            )
            projected_z = teacher_z[valid] + effective_residual * (
                predicted_z[valid] - teacher_z[valid]
            )
            vertices[valid, 1] = (projected_z - zc) / frame
            after_error.append(np.abs(projected_z - teacher_z[valid]))
            geometry.vertices = vertices
            changed += int(valid.sum())
            total += int(len(valid))
            projection_stats.setdefault("base_vertices", 0)
            projection_stats.setdefault("elevated_vertices", 0)
            projection_stats["base_vertices"] += int((valid & (layer == 0)).sum())
            projection_stats["elevated_vertices"] += int((valid & (layer == 1)).sum())
        before = np.concatenate(before_error) if before_error else np.empty(0)
        after = np.concatenate(after_error) if after_error else np.empty(0)
        projection_stats.update({
            "vertices_projected": changed,
            "vertices_total": total,
            "projected_fraction": changed / max(total, 1),
            "abs_error_before_m_p50_p90_p99": np.percentile(before, [50, 90, 99]).tolist() if before.size else [],
            "abs_error_after_m_p50_p90_p99": np.percentile(after, [50, 90, 99]).tolist() if after.size else [],
            "confidence_residual_policy": "r = clip(base_r + (1-confidence)*0.20, base_r, 0.35)",
        })
        projected.export(output / "mesh_pbr_local_height_projected.glb", extension_webp=True)

    world = projected
    # Pixal3D's PBR exporter converts raw camera-depth Z to glTF Y-up and
    # raw image-row Y to glTF Z.  The world transform must operate on those
    # exported axes, not on the pre-export raw tensor axes.
    transform = np.array([
        [frame, 0, 0, cx],
        [0, frame, 0, zc],
        [0, 0, frame, cy],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    world.apply_transform(transform)
    world_path = output / "mesh_pbr_world_axes_v3.glb"
    world.export(world_path, extension_webp=True)
    return local_path, world_path, projection_stats


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    selected = set(args.patch_id or [])
    records = [item for item in manifest["patches"] if not selected or item["patch_id"] in selected]
    found = {item["patch_id"] for item in records}
    if selected - found:
        raise ValueError(f"Unknown patch ids: {sorted(selected - found)}")

    base.configure_local_models(args)
    print("[Pixal3D] loading generation pipeline once", flush=True)
    pipeline = base.official.init_pipeline(str(args.model.resolve()), device="cuda", low_vram=False)
    print("[MoGe-2] loading camera estimator once", flush=True)
    checkpoint = args.moge / "model.pt" if args.moge.is_dir() else args.moge
    moge = base.official.load_moge_model(device="cuda", model_name=str(checkpoint.resolve()))

    guide = MultiLayerHeightGuide(args.teacher, args.surface_band_voxels)
    original_sparse_sampler = pipeline.sample_sparse_structure
    if args.mode == "structure_guided":
        def guided_sparse_sampler(cond, resolution, num_samples=1, sampler_params={}):
            original = original_sparse_sampler(cond, resolution, num_samples, sampler_params)
            constrained = guide.constrain(original, resolution, stage="sparse_structure_32")
            print("[StructureGuide] " + json.dumps(guide.last_stats), flush=True)
            return constrained
        pipeline.sample_sparse_structure = guided_sparse_sampler

        decoder = pipeline.models["shape_slat_decoder"]
        original_upsample = decoder.upsample

        def guided_hr_upsample(slat, upsample_times):
            raw_coords = original_upsample(slat, upsample_times)
            # The 1536 cascade quantizes decoder coordinates from the LR 512
            # lattice into a 96^3 HR lattice immediately after this method.
            hr_resolution = args.resolution // 16
            quantized = torch.cat([
                raw_coords[:, :1],
                ((raw_coords[:, 1:] + 0.5) / 512 * (hr_resolution - 1)).round().int(),
            ], dim=1).unique(dim=0)
            constrained = guide.constrain(
                quantized, hr_resolution,
                stage=f"shape_hr_support_{hr_resolution}",
                band=args.hr_surface_band_voxels,
            )
            # Map representatives back to the 512 lattice. The unmodified
            # pipeline then performs its normal quantization and obtains the
            # exact constrained HR coordinates.
            restored = torch.cat([
                constrained[:, :1],
                ((constrained[:, 1:].float() / (hr_resolution - 1)) * 512 - 0.5).round().int(),
            ], dim=1)
            print("[ShapeHRGuide] " + json.dumps(guide.last_stats), flush=True)
            return restored

        decoder.upsample = guided_hr_upsample

    run_records = []
    for index, item in enumerate(records, 1):
        patch_id = item["patch_id"]
        output = args.output / patch_id
        output.mkdir(exist_ok=True)
        if (output / "run_metadata.json").is_file() and (output / "mesh_pbr_world_axes_v3.glb").is_file():
            run_records.append(json.loads((output / "run_metadata.json").read_text()))
            print(f"[{index}/{len(records)}] skip complete {patch_id}", flush=True)
            continue
        started = time.time()
        try:
            source_path = Path(item["input"])
            source = Image.open(source_path).convert("RGBA")
            # Pixal3D treats fully opaque RGBA as RGB and runs rembg.  These
            # images are complete background scenes, not isolated objects, so
            # rembg is both semantically wrong and can return an empty alpha
            # mask.  A single 254 alpha sample makes Pixal3D honor the supplied
            # full-frame alpha; it remains above the 0.8 bbox threshold and is
            # visually indistinguishable from opaque.
            alpha = source.getchannel("A")
            if alpha.getextrema() == (255, 255):
                alpha.putpixel((0, 0), 254)
                source.putalpha(alpha)
            prep_transform = preprocessing_transform(source)
            guide.set_context(item, source, prep_transform)
            preprocessed = pipeline.preprocess_image(source)
            preprocessed_path = output / "model_preprocessed.png"
            preprocessed.save(preprocessed_path)
            camera = base.official.get_camera_params_wild_moge(
                str(preprocessed_path), moge, device="cuda", mesh_scale=1.0,
                extend_pixel=0, image_resolution=512,
            )
            camera = {key: float(value) for key, value in camera.items()}
            # Compute the metric frame and Z anchor in baseline mode as well, so
            # paired outputs share the exact same world transform.
            surface = guide.make_surface(32)
            geometry_contract = {
                "metric_frame_m": float(surface["frame_m"]),
                "center_world_xy_m": surface["center_world_xy_m"],
                "z_center_world_m": float(surface["z_center_m"]),
                "axis_mapping_version": "pbr_local_z_to_ue_y_positive_v3",
                "axis_mapping": "PBR-exported glTF local (x,y-up,z-image-row) -> world (UE X, UE Z-up, UE +Y)",
                "preprocess_transform": prep_transform,
                "teacher": str(args.teacher.resolve()),
                "teacher_resolution_m": guide.mpp,
            }
            print(
                f"[{index}/{len(records)}] infer {args.mode} {patch_id}: "
                f"FOV={math.degrees(camera['camera_angle_x']):.3f} deg, frame={surface['frame_m']:.3f}m",
                flush=True,
            )
            generation_started = time.time()
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            meshes, (shape, texture, grid) = pipeline.run(
                preprocessed,
                camera_params=camera,
                seed=args.seed,
                preprocess_image=False,
                return_latent=True,
                pipeline_type=f"{args.resolution}_cascade",
                sparse_structure_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0},
                shape_slat_sampler_params={"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0},
                tex_slat_sampler_params={"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0},
                max_num_tokens=49152,
            )
            generation_seconds = time.time() - generation_started
            mesh = meshes[0]
            structure_stats = dict(guide.stage_stats) if args.mode == "structure_guided" else {
                "mode": "baseline", "generation_time_dataset_geometry_injection": False,
            }
            save_raw_mesh(output / "mesh_raw.pt", mesh, shape, texture, grid, camera, geometry_contract, structure_stats)
            base.save_latent(output / "shape_slat.pt", shape)
            base.save_latent(output / "texture_slat.pt", texture)
            (output / "structure_guide.json").write_text(json.dumps(structure_stats, indent=2) + "\n")

            export_started = time.time()
            local_path, world_path, projection_stats = export_pbr(
                args, pipeline, mesh, grid, output, geometry_contract, guide
            )
            metadata = {
                "schema": "air2vectorcity.pixal3d_p384_handoff_v9_patch.v2",
                "status": "complete",
                "patch_id": patch_id,
                "row": item["row"],
                "column": item["column"],
                "mode": args.mode,
                "input": str(source_path),
                "input_sha256": base.sha256(source_path),
                "model": "TencentARC/Pixal3D main",
                "pipeline_type": f"{args.resolution}_cascade",
                "seed": args.seed,
                "camera_mode": "MoGe-2 automatic per patch",
                "height_layers": ["filled_base", "elevated_road"],
                "image_preprocess_policy": "full-frame supplied alpha; rembg bypassed",
                "building_mask_policy": "QA only; not used for alpha, structure pruning, or face deletion",
                "camera_params": camera,
                "geometry_contract": geometry_contract,
                "structure_guide": structure_stats,
                "height_projection": projection_stats,
                "raw": {
                    "vertices": int(mesh.vertices.shape[0]),
                    "faces": int(mesh.faces.shape[0]),
                    "shape_tokens": int(shape.coords.shape[0]),
                    "texture_tokens": int(texture.coords.shape[0]),
                },
                "pbr_local": base.pbr_summary(local_path),
                "pbr_world": base.pbr_summary(world_path),
                "export": {
                    "texture_size": args.texture_size,
                    "decimation_target": args.decimation_target,
                    "remesh": True,
                    "texture_extension": "webp",
                },
                "timing_seconds": {
                    "generation": round(generation_seconds, 3),
                    "pbr_export": round(time.time() - export_started, 3),
                    "total": round(time.time() - started, 3),
                },
                "environment": {
                    "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "python": platform.python_version(),
                },
            }
            (output / "failure_metadata.json").unlink(missing_ok=True)
            (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            run_records.append(metadata)
            print(f"[{index}/{len(records)}] complete {patch_id} total={metadata['timing_seconds']['total']}s", flush=True)
            del meshes, mesh, shape, texture
            torch.cuda.empty_cache()
        except Exception as error:
            failure = {
                "status": "failed", "patch_id": patch_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            (output / "failure_metadata.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
            if args.continue_on_error:
                run_records.append(failure)
                print(f"[{index}/{len(records)}] failed {patch_id}: {failure['error']}", flush=True)
                torch.cuda.empty_cache()
                continue
            raise

    output_manifest = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_run.v1",
        "status": "complete",
        "mode": args.mode,
        "patch_count": len(run_records),
        "complete_count": sum(record.get("status") == "complete" for record in run_records),
        "failed_count": sum(record.get("status") == "failed" for record in run_records),
        "generation_time_geometry_injection": args.mode == "structure_guided",
        "patches": run_records,
    }
    (args.output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "mode": args.mode, "patch_count": len(run_records)}), flush=True)


if __name__ == "__main__":
    main()
