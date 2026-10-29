#!/usr/bin/env python3
"""Check CPU dependencies, or the full generation imports on an existing GPU."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def check_imports(project: Path, cache: Path, *, require_cuda: bool = False) -> dict:
    """Never construct a model; full lazy imports require an existing CUDA driver."""
    cache.mkdir(parents=True, exist_ok=True)
    # Set these before importing torch, transformers, or the TRELLIS package.
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", WANDB_MODE="offline", HF_HOME=str(cache / "hf"),
        HF_MODULES_CACHE=str(cache / "hf_modules"), TORCH_HOME=str(cache / "torch"),
        FLEX_GEMM_USE_AUTOTUNE_CACHE="0", FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0",
        FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(cache / "flex_gemm.json"),
    )
    if not require_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(project / "src"))
    checks = []

    def check(name, action):
        try:
            value = action()
            checks.append({"name": name, "status": "passed",
                           "resolved": getattr(value, "__name__", type(value).__name__)})
        except Exception as exc:
            checks.append({"name": name, "status": "failed",
                           "error": type(exc).__name__ + ": " + str(exc),
                           "traceback": traceback.format_exc()})

    # Import torch first so native extensions resolve the PyTorch shared libraries.
    # These imports do not depend on flex_gemm/Triton driver discovery.
    for name in ("torch", "numpy", "shapely", "cv2", "ninja", "zstandard",
                 "PIL", "scipy", "trimesh", "transformers", "torchvision",
                 "timm", "kornia", "einops", "safetensors", "easydict", "tqdm"):
        check(name, lambda name=name: importlib.import_module(name))

    def configure():
        module = importlib.import_module("citycode_assets.generation")
        module.configure_offline_runtime(project / "third_party/TRELLIS2")
        return module

    check("standard_offline_runtime", configure)

    def discover(name):
        # Only top-level names: find_spec on a child can import its parent and
        # accidentally trigger flex_gemm's Triton benchmarker on a CPU machine.
        specification = importlib.util.find_spec(name)
        if specification is None:
            raise ModuleNotFoundError("Required generation package is absent: " + name)
        return specification

    for name in ("o_voxel", "cumesh", "nvdiffrast", "_nvdiffrast_c",
                 "flash_attn", "flash_attn_2_cuda", "flex_gemm"):
        check("discover_only:" + name, lambda name=name: discover(name))

    def model_class_names():
        specification = json.loads((project / "workflow/02_building_assets/pipeline.standard.json").read_text())["args"]
        names = [json.loads((project / (path + ".json")).read_text())["name"]
                 for path in specification["models"].values()]
        selected = json.loads((project / "models/building_generation/standard/MANIFEST.json").read_text())
        names += [json.loads((project / row["architecture"]["path"]).read_text())["name"]
                  for row in selected["models"]]
        return list(dict.fromkeys(names))

    try:
        names = model_class_names()
    except Exception as exc:
        names = []
        checks.append({"name": "standard_model_architectures", "status": "failed",
                       "error": type(exc).__name__ + ": " + str(exc),
                       "traceback": traceback.format_exc()})
    for module, attribute in (
        ("trellis2.modules.image_feature_extractor", "DinoV3FeatureExtractor"),
        ("trellis2.pipelines.rembg", "BiRefNet"),
        ("transformers", "DINOv3ViTModel"),
    ):
        check(module + "." + attribute,
              lambda module=module, attribute=attribute: getattr(importlib.import_module(module), attribute))

    def local_birefnet_class():
        specification = json.loads((project / "workflow/02_building_assets/pipeline.standard.json").read_text())["args"]
        directory = (project / specification["rembg_model"]["args"]["model_name"]).resolve()
        if not directory.is_dir():
            raise FileNotFoundError("Local background-removal code is missing: " + str(directory))
        config = json.loads((directory / "config.json").read_text())
        reference = config["auto_map"]["AutoModelForImageSegmentation"]
        if not isinstance(reference, str) or "--" in reference:
            raise ValueError("The standard check requires a local BiRefNet class reference")
        loader = importlib.import_module("transformers.dynamic_module_utils")
        return loader.get_class_from_dynamic_module(reference, str(directory), local_files_only=True)

    check("local_BiRefNet_dynamic_class", local_birefnet_class)

    if require_cuda:
        def existing_cuda():
            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                raise RuntimeError("--require-cuda needs an already allocated CUDA worker")
            return torch

        check("existing_CUDA_worker", existing_cuda)
        if checks[-1]["status"] == "passed":
            for name in (
                "o_voxel.convert", "o_voxel.postprocess", "o_voxel._C",
                "cumesh", "cumesh._C", "nvdiffrast.torch", "_nvdiffrast_c",
                "flash_attn", "flash_attn_2_cuda", "flex_gemm.ops.grid_sample",
                "flex_gemm.kernels.cuda",
            ):
                check(name, lambda name=name: importlib.import_module(name))
            for name in names:
                check("trellis2.models." + name,
                      lambda name=name: getattr(importlib.import_module("trellis2.models"), name))
            for module, attribute in (
                ("trellis2.pipelines", "Trellis2ImageTo3DPipeline"),
                ("trellis2.pipelines.samplers", "FlowEulerGuidanceIntervalSampler"),
                ("o_voxel.postprocess", "to_glb"),
                ("cumesh", "CuMesh"),
                ("nvdiffrast.torch", "RasterizeCudaContext"),
                ("flex_gemm.ops.grid_sample", "grid_sample_3d"),
            ):
                check(module + "." + attribute,
                      lambda module=module, attribute=attribute: getattr(importlib.import_module(module), attribute))

    def no_cuda_initialization():
        torch = importlib.import_module("torch")
        if torch.cuda.is_initialized():
            raise RuntimeError("An import unexpectedly initialized CUDA")
        return torch

    if not require_cuda:
        check("CUDA_remained_uninitialized", no_cuda_initialization)
    passed = all(row["status"] == "passed" for row in checks)
    return {
        "schema": "citycode.generation_import_check.v1",
        "status": "passed" if passed else "failed",
        "mode": "existing_CUDA_worker" if require_cuda else "CPU",
        "checks": checks, "GPU_allocation_requested": False, "weights_loaded": False,
        "network_allowed": False, "GPU_execution_verified": False,
        "full_GPU_imports_pending": not (require_cuda and passed),
        "expected_model_classes": names,
        "scope": ("Full generation imports on an existing CUDA worker; weights, inference and GLB baking remain unverified"
                  if require_cuda else "CPU-safe imports and GPU package discovery only; full generation imports require CUDA"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        help="Local import/cache directory; defaults to a temporary directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Import the full model/export closure on an already allocated CUDA worker")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    with ExitStack() as stack:
        cache = args.cache_dir.resolve() if args.cache_dir else Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="citycode-imports-")))
        report = check_imports(project, cache, require_cuda=args.require_cuda)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
