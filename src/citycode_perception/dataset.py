"""Discover explicit observation inputs without consulting reference labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    patch_id: str
    view_id: str
    stem: str
    rgb_path: Path
    camera_path: Path
    valid_mask_path: Path
    patch_path: Path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Observation path escapes dataset: {path}")
    return resolved


def discover_samples(dataset_root, patch_ids=None, views=None) -> list[Sample]:
    root = Path(dataset_root).resolve()
    index = read_json(root / "patch_index.json")
    manifest = read_json(root / "dataset_manifest.json")
    selected_views = list(views or manifest.get("selected_views") or ["tilt_light"])
    requested = set(patch_ids or [])
    samples = []
    seen = set()
    known_patches = set()
    for entry in index["patches"]:
        patch_id = str(entry["patch_id"])
        if patch_id in known_patches:
            raise ValueError(f"Duplicate patch ID: {patch_id}")
        known_patches.add(patch_id)
        if requested and patch_id not in requested:
            continue
        patch = _inside(root / entry.get("directory", f"patches/{patch_id}"), root)
        patch_path = _inside(patch / "patch.json", root)
        if not patch_path.is_file():
            raise FileNotFoundError(patch_path)
        for view_id in selected_views:
            view = _inside(patch / "views" / view_id, root)
            paths = [_inside(view / name, root) for name in
                     ("rgb.png", "camera.json", "strict_valid_mask.png")]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
            stem = f"{patch_id}__{view_id}"
            if stem in seen:
                raise ValueError(f"Duplicate selected observation: {stem}")
            seen.add(stem)
            samples.append(Sample(patch_id, view_id, stem, *paths, patch_path))
    unknown = requested - known_patches
    if unknown:
        raise ValueError(f"Unknown patch IDs: {sorted(unknown)}")
    if not samples:
        raise ValueError("No observations selected")
    return samples


def validate_dataset(dataset_root, patch_ids=None, views=None,
                     verify_checksums=False) -> dict:
    from PIL import Image

    root = Path(dataset_root).resolve()
    samples = discover_samples(root, patch_ids, views)
    for sample in samples:
        camera = read_json(sample.camera_path)
        patch = read_json(sample.patch_path)
        expected = tuple(camera["image_size_px"])
        for path in (sample.rgb_path, sample.valid_mask_path):
            with Image.open(path) as picture:
                if picture.size != expected:
                    raise ValueError(f"Camera/image size mismatch: {path}")
        with Image.open(sample.valid_mask_path) as mask:
            if mask.mode not in ("1", "L", "I", "I;16"):
                raise ValueError(f"Valid mask must be single-channel: {sample.valid_mask_path}")
        for key, shape in (("camera_to_world_4x4_m", (4, 4)),
                           ("world_to_camera_4x4_m", (4, 4))):
            matrix = camera[key]
            if len(matrix) != shape[0] or any(len(row) != shape[1] for row in matrix):
                raise ValueError(f"Invalid camera transform: {sample.camera_path}")
        for key in ("data_bounds_xy_world_m", "core_bounds_xy_world_m"):
            if len(patch[key]) != 4:
                raise ValueError(f"Invalid world bounds: {sample.patch_path}")
    checked = 0
    if verify_checksums:
        for line in (root / "checksums.sha256").read_text().splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            path = _inside(root / name.lstrip("*"), root)
            if sha256(path) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            checked += 1
    return {"status": "PASS", "patch_count": len({s.patch_id for s in samples}),
            "view_count": len(samples), "selected_views": sorted({s.view_id for s in samples}),
            "checksums_verified": checked,
            "inputs": ["rgb.png", "camera.json", "strict_valid_mask.png", "patch metadata"]}
