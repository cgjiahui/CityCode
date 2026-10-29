"""Raw MoGe v2 depth inference and explicit predicted-depth fixture import."""

from __future__ import annotations

import importlib
import copy
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .model_manifest import checkpoint_resource_record, file_sha256, model_config_sha256


class DepthBackend(Protocol):
    def predict(self, rgb: np.ndarray, *, fov_x: float, resolution_level: int) -> np.ndarray: ...


class MoGeBackend:
    """Wrap a loaded model; torch remains optional until actual inference is requested."""

    def __init__(self, model: Any, device: str, fp16: bool, resource: dict[str, Any]):
        self.model, self.device, self.fp16, self.resource = model, device, fp16, resource

    def predict(self, rgb: np.ndarray, *, fov_x: float, resolution_level: int) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float().div(255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.fp16:
            tensor = tensor.half()
        with torch.inference_mode():
            output = self.model.infer(
                tensor, resolution_level=resolution_level, force_projection=True,
                apply_mask=False, use_fp16=self.fp16, fov_x=fov_x,
            )
        return output["depth"].detach().cpu().float().numpy().reshape(rgb.shape[:2])


def _model_import_root(model_source: Path) -> Path:
    root = Path(model_source).resolve()
    for candidate in (root, root / "MoGe"):
        if (candidate / "moge" / "model" / "__init__.py").is_file():
            return candidate
    raise ValueError("model_source must contain moge/model or MoGe/moge/model")


def load_full_model(
    checkpoint: Path, model_source: Path | None = None, device: str = "cuda", *,
    config_path: Path | None = None, expected_sha256: str | None = None,
    fp16: bool | None = None,
) -> MoGeBackend:
    """Load the recorded full v2 payload with strict keys and no implicit download."""
    resource = checkpoint_resource_record(checkpoint, expected_sha256)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Depth inference requires torch and a compatible moge installation") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable; choose an available device explicitly")
    if model_source is not None:
        root = _model_import_root(model_source)
        loaded = sys.modules.get("moge")
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
            raise RuntimeError("A different moge source is already imported; use a fresh process")
        sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("moge.model")
    except ImportError as exc:
        raise RuntimeError("Install a compatible moge package or provide model_source") from exc
    # The full training exporter saves plain configuration, tensors and metadata.
    # Do not fall back to arbitrary pickle loading for incompatible checkpoints.
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"model_config", "model"}.issubset(payload):
        raise ValueError("Full checkpoint must contain model_config and model; a LoRA-only checkpoint is unsupported")
    if not isinstance(payload["model_config"], dict) or not isinstance(payload["model"], dict):
        raise ValueError("Invalid full checkpoint configuration or state dictionary")
    configured_version = "v2"
    if config_path is not None:
        config = json.loads(Path(config_path).read_text())
        configured_version = str(config.get("model_version", "v2"))
    version = str(payload.get("model_version", configured_version))
    if version != "v2" or configured_version != "v2":
        raise ValueError("This depth adapter implements the recorded MoGe v2 full-checkpoint contract")
    model_class = module.import_model_class_by_version(version)
    model = model_class(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    use_fp16 = device.startswith("cuda") if fp16 is None else bool(fp16 and device.startswith("cuda"))
    if use_fp16:
        model = model.half()
    source_file = Path(module.__file__).resolve()
    resource.update(
        model_version=version, model_config_sha256=model_config_sha256(payload["model_config"]),
        model_api="moge.model.import_model_class_by_version", source_module_sha256=file_sha256(source_file),
        device=device, fp16=use_fp16,
    )
    return MoGeBackend(model, device, use_fp16, resource)


def inference_shape(width: int, height: int, inference_size: int = 1024) -> tuple[int, int]:
    """Return (width, height), matching the original area-resize/multiple-of-14 rule."""
    if min(width, height, inference_size) <= 0:
        raise ValueError("Image dimensions and inference_size must be positive")
    scale = inference_size / max(width, height)
    return max(14, int(width * scale) // 14 * 14), max(14, int(height * scale) // 14 * 14)


def horizontal_fov(camera: dict[str, Any], width: int) -> float:
    intrinsics = camera["intrinsics"]
    if "horizontal_fov_deg" in intrinsics:
        fov = float(intrinsics["horizontal_fov_deg"])
    else:
        fx = float(intrinsics["fx_px"])
        if not math.isfinite(fx) or fx <= 0:
            raise ValueError("camera.intrinsics.fx_px must be positive and finite")
        fov = math.degrees(2 * math.atan(width / (2 * fx)))
    if not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError("Camera horizontal FoV must be finite and between 0 and 180 degrees")
    return fov


def infer_depth(
    rgb: np.ndarray, camera: dict[str, Any], valid_mask: np.ndarray, backend: DepthBackend, *,
    inference_size: int = 1024, resolution_level: int = 9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer raw metric Zc from RGB and known FoV; apply strict ROI at native size."""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be an HxWx3 uint8 array")
    height, width = rgb.shape[:2]
    if list(camera["image_size_px"]) != [width, height]:
        raise ValueError("Camera/RGB native image size mismatch")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (height, width) or not valid.any():
        raise ValueError("Strict-valid mask must match RGB and contain valid pixels")
    if not 0 <= resolution_level <= 9:
        raise ValueError("resolution_level must be between 0 and 9")
    network_width, network_height = inference_shape(width, height, inference_size)
    process = cv2.resize(rgb, (network_width, network_height), interpolation=cv2.INTER_AREA)
    fov = horizontal_fov(camera, width)
    started = time.perf_counter()
    predicted = np.asarray(backend.predict(process, fov_x=fov, resolution_level=resolution_level), dtype=np.float32)
    if predicted.shape != (network_height, network_width):
        raise ValueError("Backend depth must match its input image dimensions")
    if predicted.shape != (height, width):
        predicted = cv2.resize(predicted, (width, height), interpolation=cv2.INTER_LINEAR)
    depth = np.array(predicted, dtype=np.float32, copy=True)
    depth[~valid | ~np.isfinite(depth) | (depth <= 0)] = np.nan
    if not np.isfinite(depth).any():
        raise ValueError("Depth inference produced no finite positive depth inside the ROI")
    metadata = {
        "resize_long_edge": inference_size, "network_size_px": [network_width, network_height],
        "native_size_px": [width, height], "resolution_level": resolution_level,
        "model_fov_mode": "known", "given_horizontal_fov_deg": fov,
        "force_projection": True, "model_mask_applied": False, "seconds": time.perf_counter() - started,
        "depth_definition": "camera-forward Zc", "units": "metre", "dtype": "float32",
    }
    return depth, metadata


def _sample_inputs(sample: Any) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    if sample.view_id == "ortho_reference":
        raise ValueError("Reference orthographic views are outside the prediction-input contract")
    with Image.open(sample.rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    camera = json.loads(Path(sample.camera_path).read_text())
    with Image.open(sample.valid_mask_path) as image:
        valid = np.asarray(image.convert("L")) > 0
    return rgb, camera, valid


def _save_sample(
    sample: Any, dataset_root: Path, output_root: Path, depth: np.ndarray,
    inference: dict[str, Any], *, imported_from: Path | None = None,
    depth_stage: str = "raw_prediction", source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", sample.stem) or sample.stem in {".", ".."}:
        raise ValueError("Invalid depth output sample identity")
    directory = "depth_raw_zc_m" if depth_stage == "raw_prediction" else "depth_imported_zc_m"
    path = output_root / directory / f"{sample.stem}__{directory}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth, allow_pickle=False)
    relative = path.relative_to(output_root).as_posix()
    record = {
        "sample_id": sample.stem, "patch_id": sample.patch_id, "view_id": sample.view_id,
        "depth_stage": depth_stage,
        "input": {key: Path(value).resolve().relative_to(dataset_root.resolve()).as_posix() for key, value in
                  (("rgb", sample.rgb_path), ("camera", sample.camera_path), ("strict_valid_mask", sample.valid_mask_path))},
        "input_sha256": {key: file_sha256(Path(value)) for key, value in
                         (("rgb", sample.rgb_path), ("camera", sample.camera_path), ("strict_valid_mask", sample.valid_mask_path))},
        "strict_roi": {"applied": True, "outside_depth_value": "NaN", "finite_pixel_count": int(np.isfinite(depth).sum())},
        "inference": inference,
        "scale_calibration": {"method": "identity_raw_compatibility_alias", "scale": 1.0,
                              "applied": False, "uses_gt_depth": False, "uses_semantic_labels": False},
        "outputs": {"depth_raw_zc_m": relative, "depth_ground_scaled_zc_m": relative},
        "output_sha256": {"depth_raw_zc_m": file_sha256(path)},
    }
    if imported_from is not None:
        record["imported_prediction"] = {"filename": imported_from.name, "sha256": file_sha256(imported_from)}
        if source_metadata:
            record["imported_prediction"].update(copy.deepcopy(source_metadata))
            calibration = source_metadata.get("source_scale_calibration")
            if calibration and depth_stage == "already_calibrated_prediction":
                record["scale_calibration"] = {
                    **copy.deepcopy(calibration), "applied_by_import": False,
                    "prior_calibration_preserved": True,
                }
        if depth_stage == "already_calibrated_prediction" and not record["scale_calibration"].get("prior_calibration_preserved"):
            record["scale_calibration"] = {
                "method": "source_already_calibrated_original_scale_not_recorded",
                "scale": 1.0, "applied_by_import": False, "prior_calibration_preserved": True,
            }
    return record


def _publish_manifest(output_root: Path, samples: list[dict[str, Any]], model: dict[str, Any], mode: str) -> Path:
    stages = {sample["depth_stage"] for sample in samples}
    stage = next(iter(stages)) if len(stages) == 1 else "mixed_prediction_stages"
    payload = {
        "schema": "citycode.perception.depth.v1", "status": "complete", "mode": mode,
        "path_resolution": "manifest_directory", "sample_count": len(samples), "model": model,
        "depth_stage": stage,
        "prediction_contract": {"input_fields": ["rgb", "camera", "strict_valid_mask"],
                                "uses_reference_labels": False, "uses_reference_depth": False,
                                "depth_definition": "camera-forward Zc", "units": "metre",
                                "depth_stage": stage,
                                "legacy_output_keys_reference_same_file": True,
                                "legacy_scaled_output_is_raw_alias": stage == "raw_prediction",
                                "raw_output_key_is_stage_compatibility_alias": stage != "raw_prediction"},
        "samples": samples,
    }
    target = output_root / "manifest.json"
    temporary = output_root / ".manifest.pending.json"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return target


def predict_dataset(
    dataset_root: Path, output_root: Path, checkpoint: Path, model_source: Path | None = None,
    device: str = "cuda", patch_ids: list[str] | None = None, views: list[str] | None = None,
    inference_size: int = 1024, *, config_path: Path | None = None,
    expected_sha256: str | None = None, resolution_level: int = 9,
) -> Path:
    """Predict selected public samples using an explicitly supplied full checkpoint."""
    from .dataset import discover_samples

    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("Choose a new depth output directory")
    samples = discover_samples(dataset_root, patch_ids=patch_ids, views=views)
    if not samples:
        raise ValueError("No selected depth prediction samples")
    if any(sample.view_id == "ortho_reference" for sample in samples):
        raise ValueError("Reference orthographic views are outside the prediction-input contract")
    backend = load_full_model(checkpoint, model_source, device, config_path=config_path, expected_sha256=expected_sha256)
    output_root.mkdir(parents=True)
    records = []
    for sample in samples:
        rgb, camera, valid = _sample_inputs(sample)
        depth, inference = infer_depth(rgb, camera, valid, backend, inference_size=inference_size, resolution_level=resolution_level)
        records.append(_save_sample(sample, dataset_root, output_root, depth, inference))
    return _publish_manifest(output_root, records, backend.resource, "model_prediction")


def _fixture_paths(input_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Path, dict[str, Any]]:
    manifest = input_root if input_root.is_file() else input_root / "manifest.json"
    base = manifest.parent if manifest.is_file() else input_root
    paths = {}
    provenance: dict[str, Any] = {}
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        provenance = {
            "source_manifest_sha256": file_sha256(manifest),
            "source_note": payload.get("note"),
            "source_depth_stage": payload.get("depth_stage") or payload.get("prediction_contract", {}).get("depth_stage"),
        }
        for sample in payload.get("samples", []):
            value = sample.get("outputs", {}).get("depth_raw_zc_m")
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = base / path
            if not path.is_file():
                portable = base / "depth_raw_zc_m" / Path(value).name
                if portable.is_file():
                    path = portable
            key = (sample["patch_id"], sample["view_id"])
            if key in paths:
                raise ValueError("Duplicate depth fixture sample identity")
            paths[key] = {"path": path, "depth_stage": sample.get("depth_stage"),
                          "source_scale_calibration": sample.get("scale_calibration"),
                          "source_note": sample.get("source_note") or sample.get("note") or provenance.get("source_note")}
    return paths, base, provenance


def _import_depth_stage(source: dict[str, Any], provenance: dict[str, Any], override: str | None) -> str:
    stage = override or source.get("depth_stage") or provenance.get("source_depth_stage")
    if stage is None:
        method = str((source.get("source_scale_calibration") or {}).get("method", "")).lower().replace("-", " ")
        note = str(source.get("source_note") or "").lower()
        # These are explicit source declarations, not inference from a filename.
        cached = method.startswith("already calibrated") or (
            "cached" in note and "calibrated" in note and "not original raw" in note
        )
        stage = "already_calibrated_prediction" if cached else "raw_prediction"
    if stage not in {"raw_prediction", "already_calibrated_prediction"}:
        raise ValueError("Imported depth_stage must be raw_prediction or already_calibrated_prediction")
    return stage


def import_predicted_depth(
    dataset_root: Path, input_root: Path, output_root: Path,
    patch_ids: list[str] | None = None, views: list[str] | None = None, *, depth_stage: str | None = None,
) -> Path:
    """Import predicted Zc arrays, preserving raw versus prior-calibrated lineage."""
    from .dataset import discover_samples

    dataset_root, input_root, output_root = map(lambda p: Path(p).resolve(), (dataset_root, input_root, output_root))
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        raise FileExistsError("Choose a new depth import output directory")
    samples = discover_samples(dataset_root, patch_ids=patch_ids, views=views)
    if not samples:
        raise ValueError("No selected depth import samples")
    paths, base, provenance = _fixture_paths(input_root)
    output_root.mkdir(parents=True)
    records = []
    for sample in samples:
        source_record = paths.get((sample.patch_id, sample.view_id), {})
        source = source_record.get("path", base / "depth_raw_zc_m" / f"{sample.stem}__depth_raw_zc_m.npy")
        stage = _import_depth_stage(source_record, provenance, depth_stage)
        if source.suffix != ".npy" or not source.is_file():
            raise FileNotFoundError(f"Missing declared predicted .npy depth for {sample.stem}")
        rgb, camera, valid = _sample_inputs(sample)
        array = np.load(source, allow_pickle=False)
        if array.shape != rgb.shape[:2] or list(camera["image_size_px"]) != [rgb.shape[1], rgb.shape[0]]:
            raise ValueError(f"Depth fixture/native camera shape mismatch: {sample.stem}")
        if not np.issubdtype(array.dtype, np.floating) or not valid.any():
            raise ValueError("Depth fixture must be a floating-point prediction with a nonempty ROI")
        depth = np.array(array, dtype=np.float32, copy=True)
        if valid.shape != depth.shape:
            raise ValueError("Depth fixture/strict ROI shape mismatch")
        depth[~valid | ~np.isfinite(depth) | (depth <= 0)] = np.nan
        if not np.isfinite(depth).any():
            raise ValueError("Depth fixture has no finite positive depth in the ROI")
        inference = {"mode": "imported_raw_prediction" if stage == "raw_prediction" else "imported_already_calibrated_prediction",
                     "depth_stage": stage, "native_size_px": camera["image_size_px"],
                     "depth_definition": "camera-forward Zc", "units": "metre", "dtype": "float32"}
        metadata = {key: value for key, value in {**provenance, **source_record}.items()
                    if key != "path" and value is not None}
        records.append(_save_sample(sample, dataset_root, output_root, depth, inference,
                                    imported_from=source, depth_stage=stage, source_metadata=metadata))
    model = {"source": "user_supplied_predicted_depth", "checkpoint_identity": None, **provenance}
    return _publish_manifest(output_root, records, model, "predicted_depth_import")
