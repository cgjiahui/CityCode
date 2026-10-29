"""Portable LOD delivery metadata without private research file dependencies."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


_PRIVATE_PATH_FIELDS = {
    "source_glb", "original_source_glb", "raw_cache_path", "raw_manifest_path",
    "fit_record", "predicted_building_path", "code_path", "source_root",
    "output_root", "trellis_root", "source_directory", "private_source_path",
}
_ORIGIN_FIELDS = {
    "building_id", "old_asset_id", "asset_id", "original_asset_id",
    "normal_repair_applied", "raw_cache_hash_verified_here",
    "raw_cache_hash_must_be_verified_before_gpu_export",
}


def _private_locator(value):
    if not isinstance(value, str):
        return False
    return (value.startswith(("/", "~", "file://", "\\\\"))
            or bool(re.search(r"(?:^|\s)[A-Za-z]:[\\/]", value))
            or bool(re.search(r"(?:^|[\s(\"'])/(?:mnt|tmp|home|root|data|opt|usr|var|workspace)/", value)))


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(child) for key, child in value.items()
                if key not in _PRIVATE_PATH_FIELDS and not _private_locator(key)
                and not key.endswith(("_path", "_paths", "_root", "_directory"))
                and not _private_locator(child)}
    if isinstance(value, list):
        return [_clean(child) for child in value if not _private_locator(child)]
    return copy.deepcopy(value)


def sanitize_delivery_metadata(result: dict, fit: dict | None = None) -> dict:
    """Retain cache contracts and actual checks while removing private paths.

    Asset paths must already be relative to the delivery package. Provenance
    identities and hashes remain under source; numeric registration evidence is
    retained separately. The caller retains the complete research record.
    """
    if not isinstance(result, dict):
        raise TypeError("LOD metadata must be an object")
    lods = result.get("lods", {})
    for label, row in lods.items():
        path = row.get("path")
        if (not isinstance(path, str) or not path or "\\" in path
                or re.match(r"^[A-Za-z]:", path) or "://" in path
                or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts):
            raise ValueError(f"LOD {label} needs an owned relative delivery path")
    cleaned = _clean(result)
    source = cleaned.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("Source provenance must be an object")
    origin = {key: value for key, value in source.items()
              if key in _ORIGIN_FIELDS or key.endswith(("_sha256", "_bytes"))}
    evidence = {key: value for key, value in source.items() if key not in origin}
    existing = cleaned.get("registration", {})
    if not isinstance(existing, dict):
        raise ValueError("Registration evidence must be an object")
    if fit is not None:
        if not isinstance(fit, dict):
            raise ValueError("Additional fit evidence must be an object")
        evidence.update(_clean(fit))
    if existing or evidence:
        cleaned["registration"] = {**existing, **evidence}
    cleaned["source"] = origin
    cleaned["delivery_metadata_schema"] = "citycode.asset_lod_delivery.v1"
    if "delivery_metadata_code_version" not in cleaned:
        from citycode_perception import __version__
        cleaned["delivery_metadata_code_version"] = __version__
    # Existing model checks and cache contract values are never reconstructed.
    for key in ("source_sha256", "raw_cache_sha256", "code_sha256", "lods", "status",
                "started_utc", "completed_utc", "elapsed_seconds"):
        if key in result and cleaned.get(key) != result[key]:
            raise ValueError(f"Sanitization would alter the retained LOD contract: {key}")
    return cleaned


def _fingerprint(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def sanitize_completed_sidecars(asset_package: str | Path) -> dict:
    """Rewrite only completed assets/*/lod_manifest.json, safely and repeatably."""
    root = Path(asset_package).resolve()
    changed = skipped = unchanged = 0
    errors = []
    for path in sorted((root / "assets").glob("*/lod_manifest.json")):
        if not path.resolve().is_relative_to(root):
            errors.append({"building_id": path.parent.name, "message": "Sidecar symlink escapes delivery package"})
            continue
        temporary = None
        try:
            before = _fingerprint(path)
            original = json.loads(path.read_text())
            if original.get("status") != "completed":
                skipped += 1
                continue
            cleaned = sanitize_delivery_metadata(original)
            if cleaned == original:
                unchanged += 1
                continue
            if _fingerprint(path) != before:
                skipped += 1
                continue
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".delivery_metadata_", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(cleaned, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            if _fingerprint(path) != before:
                skipped += 1
                continue
            os.replace(temporary, path)
            temporary = None
            changed += 1
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"building_id": path.parent.name, "message": str(exc)})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"schema": "citycode.delivery_metadata_sanitization.v1",
            "status": "passed" if not errors else "failed", "changed": changed,
            "unchanged": unchanged, "skipped_incomplete_or_changed": skipped, "errors": errors,
            "model_files_modified": False, "private_research_records_modified": False}
