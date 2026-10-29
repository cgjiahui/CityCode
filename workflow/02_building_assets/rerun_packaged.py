#!/usr/bin/env python3
"""Check packaged regeneration inputs, or explicitly dispatch a generation phase.

The default check uses only the Python standard library and writes no outputs.
GPU execution is opt-in via --phase gpu/run and uses city-buildings unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "data/smallcity_p384_reused_generation_inputs/manifest.json"
SCOPES = ("reused", "generated", "components")
PHASES = ("check-inputs", "prepare", "review-images", "freeze", "gpu",
          "recover-exports", "validate", "review-models", "finalize", "status", "run")
FORBIDDEN_CONFIG_KEYS = {
    "asset_cache", "asset_cache_manifest", "asset_cache_restore_output",
    "massing_completion_recipe", "massing_completion_support",
}


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_file(base, value, allowed, *, field):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(field + " must be a nonempty relative file path")
    path = (base / value).resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError(field + " is missing or outside its allowed package: " + value)
    return path


def _verify_proof(base, item, allowed, seen, *, field):
    if not isinstance(item, dict):
        raise ValueError(field + " must contain file proofs")
    path = _relative_file(base, item.get("path"), allowed, field=field)
    expected_size, expected_hash = item.get("bytes"), item.get("sha256")
    if (type(expected_size) is not int or expected_size < 0
            or not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_hash)):
        raise ValueError(field + " has an invalid byte count or SHA256")
    if path.stat().st_size != expected_size:
        raise ValueError("Packaged input size changed: " + item["path"]
                         + f" (manifest={expected_size}, actual={path.stat().st_size}). "
                         "Keep input verification enabled; maintainers can inspect "
                         "release/refresh_controls.py to refresh reviewed control-file proofs.")
    if path not in seen:
        seen[path] = _sha256(path)
    if seen[path] != expected_hash:
        raise ValueError("Packaged input checksum changed: " + item["path"])


def check_inputs(manifest_path=None, *, scope="all", project_root=PROJECT_ROOT):
    """Verify file identities and generation-only configs without importing GPU code."""
    project = Path(project_root).resolve()
    manifest_path = Path(manifest_path or project / DEFAULT_MANIFEST).resolve()
    if not manifest_path.is_relative_to(project) or not manifest_path.is_file():
        raise ValueError("Regeneration manifest must be a file inside this release")
    if scope not in ("all",) + SCOPES:
        raise ValueError("Unknown regeneration scope: " + str(scope))
    manifest = _read(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != "citycode.reused_generation_inputs.v1":
        raise ValueError("Unsupported packaged regeneration manifest schema")
    files, required = manifest.get("files"), manifest.get("required_files", [])
    if not isinstance(files, list) or not files or not isinstance(required, list):
        raise ValueError("Manifest needs nonempty files and a required_files list")
    bundle = manifest_path.parent
    seen = {}
    for item in files:
        _verify_proof(bundle, item, bundle, seen, field="files")
    # Shared structure/config resources may be siblings of the bundle, but all
    # references must resolve under the release after symlink normalization.
    for item in required:
        _verify_proof(bundle, item, project, seen, field="required_files")
    configured = manifest.get("rerun_configs")
    if not isinstance(configured, dict) or not set(SCOPES) <= set(configured):
        raise ValueError("Manifest must name reused, generated and components rerun_configs")
    selected = SCOPES if scope == "all" else (scope,)
    configs = {}
    for name in selected:
        path = _relative_file(project, configured[name], project, field="rerun_configs." + name)
        config = _read(path)
        if (not isinstance(config, dict) or config.get("schema") != "citycode.unified_generation_config.v1"
                or config.get("execution_mode") != "generate"):
            raise ValueError("Rerun configuration must explicitly select generate mode: " + name)
        if FORBIDDEN_CONFIG_KEYS & set(config) or any(k.startswith("massing_completion_") for k in config):
            raise ValueError("Rerun configuration cannot restore an asset cache or inherit a massing recipe: " + name)
        if config.get("historical_models_allowed") is True:
            raise ValueError("Rerun configuration cannot substitute historical model results: " + name)
        relative_root = config.get("project_root")
        if (not isinstance(relative_root, str) or Path(relative_root).is_absolute()
                or (path.parent / relative_root).resolve() != project):
            raise ValueError("Rerun project_root must resolve to this release: " + name)
        resolved = {}
        for key in ("input_manifest", "dispositions", "provider_config", "runtime_config",
                    "completed_images_manifest"):
            resolved[key] = _relative_file(project, config.get(key), project, field=name + "." + key)
        output_value = config.get("output")
        if not isinstance(output_value, str) or not output_value or Path(output_value).is_absolute():
            raise ValueError("Rerun output must be relative to this release: " + name)
        output = (project / output_value).resolve()
        if (not output.is_relative_to(project)
                or output.parent != resolved["input_manifest"].parents[2]
                or output == resolved["input_manifest"].parents[1]):
            raise ValueError("Rerun output must be a separate sibling of its structure package: " + name)
        for protected in ("smallcity_p384_building_assets", "smallcity_p384_building_assets_completed"):
            if output == project / "data" / protected:
                raise ValueError("Rerun cannot overwrite the provided building result: " + name)
        configs[name] = {"path": path.relative_to(project).as_posix(), "sha256": _sha256(path),
                         "execution_mode": "generate", "output": output.relative_to(project).as_posix()}
    if len({item["output"] for item in configs.values()}) != len(configs):
        raise ValueError("The selected regeneration scopes need separate output directories")
    return {"schema": "citycode.packaged_rerun_check.v1", "status": "inputs_verified",
            "phase": "check-inputs", "scope": scope,
            "manifest": manifest_path.relative_to(project).as_posix(),
            "manifest_sha256": _sha256(manifest_path),
            "input_file_proofs_verified": len(files), "required_file_proofs_verified": len(required),
            "unique_files_verified": len(seen), "configs_verified": configs,
            "outputs_created": False, "API_calls": 0, "GPU_jobs": 0,
            "full_GPU_regeneration_revalidated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / DEFAULT_MANIFEST)
    parser.add_argument("--scope", choices=("all",) + SCOPES, default="all")
    parser.add_argument("--phase", choices=PHASES, default="check-inputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--deadline-seconds", type=int, default=None, help="Optional explicit GPU runtime limit; omitted means no fixed cutoff")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 1 <= args.gpu_count <= 16 or (args.deadline_seconds is not None and args.deadline_seconds <= 0):
        parser.error("workers/gpu-count must be 1..16 and an explicit deadline-seconds must be positive")
    try:
        report = check_inputs(args.manifest, scope=args.scope)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, "Packaged input check failed: " + str(exc) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.phase == "check-inputs":
        return 0
    # Prefer the release beside this script; check-inputs above needs no project
    # installation, NumPy, CUDA imports, network or filesystem mutations.
    source = PROJECT_ROOT / "src"
    if source.is_dir():
        sys.path.insert(0, str(source))
    from citycode_assets.batch import main as batch_main
    for name, config in report["configs_verified"].items():
        path = PROJECT_ROOT / config["path"]
        if _sha256(path) != config["sha256"]:
            raise ValueError("Rerun config changed after verification: " + name)
        print(json.dumps({"scope": name, "phase": args.phase, "config": config["path"]}, ensure_ascii=False))
        if args.phase == "recover-exports":
            from recover_exports import main as recovery_main
            result = recovery_main(["--config", str(path)])
            if result:
                return result
            continue
        phase_argv = ["--config", str(path), "--phase", args.phase,
                      "--workers", str(args.workers), "--gpu-count", str(args.gpu_count)]
        if args.deadline_seconds is not None:
            phase_argv += ["--deadline-seconds", str(args.deadline_seconds)]
        result = batch_main(phase_argv)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
