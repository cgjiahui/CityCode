"""Depth model resource requirements, without downloading or loading weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def depth_model_requirements() -> dict[str, Any]:
    """Describe the full-checkpoint adapter and distinguish upstream LoRA weights."""
    return {
        "schema": "citycode.perception.depth_model_requirements.v1",
        "model_api": "moge.model.import_model_class_by_version",
        "model_version": "v2",
        "checkpoint_mode": "full",
        "checkpoint_required_keys": ["model_config", "model"],
        "source": {
            "repository": "https://github.com/kuieless/AerialMetric",
            "moge_repository": "https://github.com/microsoft/MoGe",
            "recorded_package_version": "2.0.0",
            "installation": "Install a compatible moge package or supply a checkout containing moge/ or MoGe/moge/.",
        },
        "runtime_dependencies": ["torch", "opencv-python", "numpy", "Pillow", "scipy", "utils3d", "huggingface_hub", "moge"],
        "optional_runtime_dependencies": ["xformers", "pandas", "tqdm"],
        "recorded_dependency_pins": {
            "utils3d": "3fab839f0be9931dac7c8488eb0e1600c236e183",
            "pipeline": "866f059d2a05cde05e4a52211ec5051fd5f276d6",
        },
        "project_p384_full_checkpoint": {
            "selection": "big_city_p384_warmstart_full_sweep_v1_s1_lr2e6",
            "step": 4800,
            "filename": "step_00004800.pt",
            "sha256": "63cf3463631911ea3ce586d576926ec41230f93df547755e594994ded3bbc873",
            "bytes": 1323776820,
            "hash_status": "verified_read_only_streaming_sha256",
            "download_url": None,
            "availability": "The selected local full checkpoint has a verified hash; supply it explicitly because no public download is established.",
        },
        "upstream_aerial_lora_checkpoint": {
            "resource_page": "https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026",
            "filename": "Moge2-Aerial.pt",
            "sha256": "9c42837ebcf295af959bbdc4f2ddceb204286a7df9bb94f475a6d764099c6191",
            "bytes": 1484014928,
            "compatible_with_full_adapter": False,
            "note": "This recorded upstream LoRA hash is not the P384 full-finetuned checkpoint hash.",
        },
        "inference": {
            "known_horizontal_fov": True,
            "requested_long_edge": 1024,
            "round_dimensions_down_to_multiple": 14,
            "resolution_level": 9,
            "force_projection": True,
            "model_mask_applied": False,
            "output": "native image size, float32 camera-forward Zc in metres, NaN outside strict ROI",
        },
    }


def checkpoint_resource_record(checkpoint: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Record an explicitly supplied resource; verify only a supplied hash."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = None
    if expected_sha256 is not None:
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256.lower()):
            raise ValueError("expected_sha256 must be a SHA-256 hexadecimal digest")
        actual = file_sha256(checkpoint)
        if actual != expected_sha256.lower():
            raise ValueError("Depth checkpoint checksum mismatch")
    return {
        "filename": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": actual,
        "hash_status": "verified" if actual is not None else "not_provided",
        "checkpoint_mode": "full",
    }


def model_config_sha256(config: dict[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()
