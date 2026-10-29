#!/usr/bin/env python3
"""Run Pixal3D 1536 + MoGe-2 auto camera and save full PBR mesh per patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn_3")

import numpy as np
import torch
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parents[1]
PIXAL = RELEASE / "third_party/Pixal3D"
sys.path.insert(0, str(PIXAL))
os.environ.setdefault("TORCH_HOME", str(ROOT / "third_party/torch_cache"))

import inference as official
from pixal3d.pipelines import rembg as rembg_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "third_party/models/Pixal3D")
    parser.add_argument("--dino", type=Path, default=ROOT / "third_party/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--moge", type=Path, default=ROOT / "third_party/models/moge-2-vitl")
    parser.add_argument("--birefnet", type=Path, default=ROOT / "third_party/models/BiRefNet")
    parser.add_argument("--naf", type=Path, default=ROOT / "third_party/NAF")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1536)
    parser.add_argument("--patch-id", action="append", help="Optional repeated pilot subset")
    parser.add_argument("--texture-size", type=int, default=4096)
    parser.add_argument("--decimation-target", type=int, default=1000000)
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_local_models(args):
    for config in official.IMAGE_COND_CONFIGS.values():
        config["model_name"] = str(args.dino.resolve())
    original_birefnet = rembg_module.BiRefNet.__init__

    def local_birefnet(self, model_name="ZhengPeng7/BiRefNet"):
        return original_birefnet(self, model_name=str(args.birefnet.resolve()))

    rembg_module.BiRefNet.__init__ = local_birefnet
    original_hub = torch.hub.load

    def local_hub(repo_or_dir, model, *hub_args, **hub_kwargs):
        if repo_or_dir == "valeoai/NAF":
            repo_or_dir = str(args.naf.resolve())
            hub_kwargs["source"] = "local"
        return original_hub(repo_or_dir, model, *hub_args, **hub_kwargs)

    torch.hub.load = local_hub


def save_latent(path: Path, latent):
    torch.save({"coords": latent.coords.detach().cpu(), "feats": latent.feats.detach().cpu()}, path)


def pbr_summary(path: Path):
    scene = trimesh.load(path, force="scene", process=False)
    geometries = list(scene.geometry.values())
    return {
        "geometry_count": len(geometries),
        "vertices": sum(len(mesh.vertices) for mesh in geometries),
        "faces": sum(len(mesh.faces) for mesh in geometries),
        "material_types": [type(getattr(mesh.visual, "material", None)).__name__ for mesh in geometries],
        "bounds": scene.bounds.tolist(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_manifest = json.loads((args.inputs / "manifest.json").read_text())
    selected = set(args.patch_id or [])
    records = [item for item in input_manifest["patches"] if not selected or item["patch_id"] in selected]
    if selected and selected != {item["patch_id"] for item in records}:
        raise ValueError(f"Unknown patch ids: {sorted(selected - {item['patch_id'] for item in records})}")
    configure_local_models(args)

    print("[Pixal3D] loading generation pipeline once", flush=True)
    pipeline = official.init_pipeline(str(args.model.resolve()), device="cuda", low_vram=False)
    print("[MoGe-2] loading camera estimator once", flush=True)
    checkpoint = args.moge / "model.pt" if args.moge.is_dir() else args.moge
    moge = official.load_moge_model(device="cuda", model_name=str(checkpoint.resolve()))
    summaries = []

    for index, item in enumerate(records, 1):
        patch_id = item["patch_id"]
        source = Path(item["input"])
        output = args.output / patch_id
        output.mkdir(exist_ok=True)
        metadata_path = output / "run_metadata.json"
        if metadata_path.is_file() and (output / "mesh_pbr.glb").is_file():
            metadata = json.loads(metadata_path.read_text())
            summaries.append(metadata)
            print(f"[{index}/{len(records)}] skip complete {patch_id}", flush=True)
            continue
        started = time.time()
        try:
            print(f"[{index}/{len(records)}] preprocess {patch_id}", flush=True)
            preprocessed = pipeline.preprocess_image(Image.open(source))
            preprocessed_path = output / "model_preprocessed.png"
            preprocessed.save(preprocessed_path)
            camera = official.get_camera_params_wild_moge(
                str(preprocessed_path), moge, device="cuda", mesh_scale=1.0,
                extend_pixel=0, image_resolution=512,
            )
            camera = {key: float(value) for key, value in camera.items()}
            print(
                f"[{index}/{len(records)}] infer {patch_id}: "
                f"FOV={math.degrees(camera['camera_angle_x']):.3f} deg distance={camera['distance']:.5f}",
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
            }, output / "mesh_raw.pt")
            save_latent(output / "shape_slat.pt", shape)
            save_latent(output / "texture_slat.pt", texture)

            print(f"[{index}/{len(records)}] export 4096 PBR {patch_id}", flush=True)
            export_started = time.time()
            glb = official.o_voxel.postprocess.to_glb(
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
            rotation = np.array([
                [-1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)
            glb.apply_transform(rotation)
            pbr_path = output / "mesh_pbr.glb"
            glb.export(pbr_path, extension_webp=True)
            metadata = {
                "schema": "air2vectorcity.pixal3d_full_detail_patch.v1",
                "status": "complete",
                "patch_id": patch_id,
                "row": item["row"],
                "column": item["column"],
                "world_bounds_xy_m": item["world_bounds_xy_m"],
                "input": str(source),
                "input_sha256": sha256(source),
                "model": "TencentARC/Pixal3D",
                "pipeline_type": f"{args.resolution}_cascade",
                "camera_mode": "MoGe-2 automatic per patch",
                "camera_params": camera,
                "seed": args.seed,
                "sampling_steps": {"sparse_structure": 12, "shape": 12, "texture": 12},
                "raw": {
                    "vertices": int(mesh.vertices.shape[0]),
                    "faces": int(mesh.faces.shape[0]),
                    "shape_tokens": int(shape.coords.shape[0]),
                    "texture_tokens": int(texture.coords.shape[0]),
                    "mesh_pt_bytes": (output / "mesh_raw.pt").stat().st_size,
                },
                "pbr": pbr_summary(pbr_path),
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
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summaries.append(metadata)
            print(f"[{index}/{len(records)}] complete {patch_id} total={metadata['timing_seconds']['total']}s", flush=True)
            del meshes, mesh, shape, texture, glb
            torch.cuda.empty_cache()
        except Exception as error:
            failure = {
                "schema": "air2vectorcity.pixal3d_full_detail_patch.v1",
                "status": "failed",
                "patch_id": patch_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            (output / "failure_metadata.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise

    manifest = {
        "schema": "air2vectorcity.pixal3d_full_detail_run.v1",
        "status": "complete",
        "patch_count": len(summaries),
        "quality_contract": {
            "true_pixal3d_geometry": True,
            "true_pixal3d_pbr": True,
            "dataset_rgb_used_as_output_texture": False,
            "per_patch_texture_size": args.texture_size,
            "per_patch_original_input_resolution": 2048,
        },
        "patches": summaries,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "patch_count": len(summaries)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
