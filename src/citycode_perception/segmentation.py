"""V9.2/V7B request export and offline mask import (Pillow + stdlib).

The geometry kernel consumes 1024-square RGB masks via samples[].id/result.
Provider responses are retained separately; mask resizing uses nearest neighbor.
No network access occurs in prepare_requests or import_results.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from .dataset import discover_samples, sha256


STAGES = {
    "structure_v9_2": ("v9.2", "prompt_v9_2_joint_building_instance.txt"),
    "structure_v7b": ("v7b", "prompt_v7b_planar_background_semantic.txt"),
}
MASK_SIZE = (1024, 1024)
_RESAMPLING = getattr(Image, "Resampling", Image)
VERTEX_DOCS = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/image-generation"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _local(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes its root: {relative}")
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _prompts(prompt_root: Path | str | None) -> Path:
    if prompt_root is not None:
        return Path(prompt_root).resolve()
    return Path(__file__).resolve().parent / "prompts"


def _config(working_size: int) -> dict:
    if not isinstance(working_size, int) or working_size < 1:
        raise ValueError("working_size must be a positive integer")
    return {
        "working_input_long_edge": working_size,
        "input_resampling": "LANCZOS",
        "requested_output": {"format": "PNG", "size": list(MASK_SIZE)},
        "geometry_mask_resampling": "NEAREST",
        "reference_model": "gemini-3-pro-image",
        "model_selection": "Provider and model must be recorded by the caller; export is provider-neutral.",
    }


def prepare_requests(dataset_root, output_root, prompt_root=None, patch_ids=None,
                     views=None, *, working_size=2048) -> Path:
    """Export image-edit jobs and geometry-compatible manifests for both stages.

    Prompt text is copied verbatim. The original RGB hash, deterministic working
    image hash and complete non-secret request configuration are frozen. Existing
    results are reusable only when that preparation identity remains unchanged.
    """
    dataset_root, output_root = Path(dataset_root).resolve(), Path(output_root).resolve()
    samples = discover_samples(dataset_root, patch_ids, views)
    config = _config(working_size)
    source_hashes = {sample.stem: sha256(sample.rgb_path) for sample in samples}
    stage_records = []
    for stage, (version, filename) in STAGES.items():
        prompt_path = _prompts(prompt_root) / filename
        prompt_bytes = prompt_path.read_bytes()
        prompt_text = prompt_bytes.decode("utf-8")
        prompt_hash = sha256(prompt_path)
        run_dir = output_root / stage
        manifest_path = run_dir / "manifest.json"
        old = _read(manifest_path) if manifest_path.is_file() else {}
        identity = {"prompt_sha256": prompt_hash, "config": config, "sources": source_hashes}
        if old and old.get("preparation_identity") != identity:
            raise ValueError(f"Prepared inputs/config changed; choose a fresh output root: {run_dir}")
        old_samples = {item["id"]: item for item in old.get("samples", [])}
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(exist_ok=True)
        snapshot = run_dir / f"prompt_{version}_snapshot.txt"
        if snapshot.exists() and snapshot.read_bytes() != prompt_bytes:
            raise ValueError(f"Prompt snapshot changed: {snapshot}")
        snapshot.write_bytes(prompt_bytes)
        records, jobs = [], []
        for sample in samples:
            if not re.fullmatch(r"[A-Za-z0-9_.+\-]+__[A-Za-z0-9_.+\-]+", sample.stem):
                raise ValueError(f"Unsafe observation ID: {sample.stem}")
            input_rel = f"inputs/{sample.stem}.png"
            input_path = _local(run_dir, input_rel)
            with Image.open(sample.rgb_path) as original:
                source_size = original.size
                if source_size[0] != source_size[1]:
                    raise ValueError(f"The fixed square mask recipe needs a square RGB: {sample.rgb_path}")
                image = original.convert("RGB")
                if image.size != (working_size, working_size):
                    image = image.resize((working_size, working_size), _RESAMPLING.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
            input_path.parent.mkdir(exist_ok=True)
            if input_path.exists() and input_path.read_bytes() != buffer.getvalue():
                raise ValueError(f"Working input changed: {input_path}")
            if not input_path.exists():
                input_path.write_bytes(buffer.getvalue())
            record = {
                "id": sample.stem, "patch_id": sample.patch_id, "view_id": sample.view_id,
                "source_path": os.path.relpath(sample.rgb_path, run_dir),
                "source_sha256": source_hashes[sample.stem], "source_size": list(source_size),
                "input": input_rel, "input_sha256": sha256(input_path),
                "input_size": [working_size, working_size],
                "result": f"results/{sample.stem}__mask-prompt-{version}__raw.png",
            }
            previous = old_samples.get(sample.stem, {})
            if "result_sha256" in previous:
                result_path = _local(run_dir, record["result"])
                if not result_path.is_file() or sha256(result_path) != previous["result_sha256"]:
                    raise ValueError(f"Imported result changed: {result_path}")
                for key in ("result_sha256", "result_size", "import", "status"):
                    if key in previous:
                        record[key] = previous[key]
            records.append(record)
            jobs.append({"id": sample.stem, "stage": stage, "edit_target": input_rel,
                         "input_sha256": record["input_sha256"], "source_sha256": record["source_sha256"],
                         "prompt_file": snapshot.name, "prompt": prompt_text,
                         "prompt_sha256": prompt_hash, "configuration": config,
                         "result_destination": record["result"]})
        ready = sum("result_sha256" in record for record in records)
        manifest = {
            "schema_version": 1, "experiment": stage, "prompt_version": version,
            "prompt_snapshot": snapshot.name, "prompt_sha256": prompt_hash,
            "source_directory": os.path.relpath(dataset_root, run_dir),
            "source_id_mode": "air2vectorcity-view", "working_input_long_edge": working_size,
            "requested_output": config["requested_output"], "configuration": config,
            "preparation_identity": identity, "sample_count": len(records),
            "completed_sample_count": ready,
            "status": "complete" if ready == len(records) else "prepared",
            "generation_backend": "provider-neutral request export / explicit PNG import",
            "samples": records, "updated_at": _now(),
        }
        _write(manifest_path, manifest)
        requests_path = run_dir / "requests.jsonl"
        requests_path.write_text("".join(json.dumps(job, ensure_ascii=False) + "\n" for job in jobs), encoding="utf-8")
        stage_records.append({"stage": stage, "manifest": f"{stage}/manifest.json",
                              "requests": f"{stage}/requests.jsonl", "sample_count": len(records)})
    destination = output_root / "requests_manifest.json"
    _write(destination, {"schema": "citycode.segmentation_requests.v1", "sample_count": len(samples),
                         "request_count": len(samples) * len(STAGES), "stages": stage_records,
                         "network_called": False, "updated_at": _now()})
    return destination


def _result_source(results_root: Path, stage: str, record: dict, version: str) -> tuple[Path, dict]:
    run_dir = results_root / stage
    source_manifest = run_dir / "manifest.json"
    if source_manifest.is_file():
        manifest = _read(source_manifest)
        selected = [item for item in manifest["samples"] if item["id"] == record["id"]]
        if len(selected) != 1:
            raise ValueError(f"Missing or duplicate selected mask ID: {record['id']} in {source_manifest}")
        source_record = selected[0]
        for key in ("source_sha256",):
            if source_record.get(key) and source_record[key] != record[key]:
                raise ValueError(f"Mask source RGB mismatch: {record['id']}")
        if manifest.get("prompt_sha256") and manifest["prompt_sha256"] != sha256(_local(run_dir, manifest["prompt_snapshot"])):
            raise ValueError(f"Source prompt snapshot changed: {source_manifest}")
        source = _local(run_dir, source_record["result"])
        if source_record.get("result_sha256") and sha256(source) != source_record["result_sha256"]:
            raise ValueError(f"Source result hash mismatch: {source}")
        # Re-importing this module's output must use the retained original PNG,
        # rather than treat the normalized geometry mask as a new provider result.
        imported = source_record.get("import", {})
        if imported.get("raw"):
            source = _local(run_dir, imported["raw"])
            if sha256(source) != imported["raw_sha256"]:
                raise ValueError(f"Retained raw result hash mismatch: {source}")
        return source, {
            "source_manifest": os.path.relpath(source_manifest, results_root),
            "source_manifest_sha256": sha256(source_manifest),
            "source_prompt_sha256": manifest.get("prompt_sha256"),
            "source_call_configuration": {key: manifest[key] for key in (
                "generation_backend", "generation_model", "working_input_long_edge",
                "requested_output", "inline_image_encoding", "configuration",
            ) if key in manifest},
            "source_api_metadata_sha256": source_record.get("api_metadata_sha256"),
        }
    candidates = [run_dir / "results" / f"{record['id']}.png", run_dir / "results" / Path(record["result"]).name,
                  run_dir / f"{record['id']}.png"]
    found = [candidate for candidate in candidates if candidate.is_file()]
    if not found:
        raise FileNotFoundError(f"No {stage} PNG for {record['id']} under {results_root}")
    if len({sha256(candidate) for candidate in found}) != 1:
        raise ValueError(f"Ambiguous differing PNGs for {stage}/{record['id']}")
    return found[0], {"source_manifest": None, "source_call_configuration": None}


def _mask_image(path: Path, allowed_sizes: set[tuple[int, int]]) -> tuple[Image.Image, dict]:
    with Image.open(path) as original:
        original.load()
        if original.format != "PNG":
            raise ValueError(f"Mask import requires a real PNG: {path}")
        if original.size not in allowed_sizes:
            raise ValueError(f"Unexpected mask dimensions {original.size}: {path}; allowed {sorted(allowed_sizes)}")
        if original.mode not in ("RGB", "RGBA", "P"):
            raise ValueError(f"Mask must contain RGB categories, not {original.mode}: {path}")
        if "A" in original.getbands() or "transparency" in original.info:
            if original.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"Transparent category mask is ambiguous: {path}")
        metadata = {"original_format": "PNG", "original_mode": original.mode,
                    "original_size": list(original.size), "target_size": list(MASK_SIZE),
                    "resampling": "none" if original.size == MASK_SIZE else "NEAREST",
                    "color_quantization": "none"}
        image = original.convert("RGB")
        if image.size != MASK_SIZE:
            image = image.resize(MASK_SIZE, _RESAMPLING.NEAREST)
        return image, metadata


def import_results(dataset_root, results_root, output_root, prompt_root=None,
                   patch_ids=None, views=None, *, working_size=2048) -> Path:
    """Import selected V9.2/V7B PNGs and return the two-manifest mask root.

    Results may be placed in stage/results/{patch}__{view}.png, use the exported
    result filename, or be declared by an existing stage manifest. Import checks
    available source hashes. Missing provenance is recorded, never invented.
    """
    results_root, output_root = Path(results_root).resolve(), Path(output_root).resolve()
    # Preflight both stages before writing geometry outputs.
    selected = discover_samples(dataset_root, patch_ids, views)
    planned = []
    for stage, (version, filename) in STAGES.items():
        target_prompt_hash = sha256(_prompts(prompt_root) / filename)
        for sample in selected:
            record = {"id": sample.stem, "source_sha256": sha256(sample.rgb_path),
                      "result": f"results/{sample.stem}__mask-prompt-{version}__raw.png"}
            source, provenance = _result_source(results_root, stage, record, version)
            source_manifest = results_root / stage / "manifest.json"
            if source_manifest.is_file():
                source_meta = _read(source_manifest)
                if source_meta.get("prompt_sha256") and source_meta["prompt_sha256"] != target_prompt_hash:
                    raise ValueError(f"Mask prompt differs from the formal {version} prompt: {source_manifest}")
            with Image.open(sample.rgb_path) as rgb:
                allowed = {MASK_SIZE, rgb.size, (working_size, working_size)}
            _, image_metadata = _mask_image(source, allowed)
            planned.append((stage, sample.stem, source, provenance, image_metadata))
    prepare_requests(dataset_root, output_root, prompt_root, patch_ids, views, working_size=working_size)
    manifests = {stage: _read(output_root / stage / "manifest.json") for stage in STAGES}
    for stage, sample_id, source, provenance, image_metadata in planned:
        run_dir = output_root / stage
        record = next(item for item in manifests[stage]["samples"] if item["id"] == sample_id)
        raw_path = run_dir / "raw_imports" / f"{sample_id}.png"
        raw_path.parent.mkdir(exist_ok=True)
        source_hash = sha256(source)
        if raw_path.exists() and sha256(raw_path) != source_hash:
            raise ValueError(f"Different raw mask already imported: {raw_path}; choose a fresh output root")
        if not raw_path.exists():
            shutil.copyfile(source, raw_path)
        image, _ = _mask_image(raw_path, {tuple(image_metadata["original_size"])})
        target = _local(run_dir, record["result"])
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        if target.exists() and target.read_bytes() != buffer.getvalue():
            # Existing native provider PNG in the prepared request destination
            # is safe to normalize only when it is the exact imported source.
            if target != source or sha256(target) != source_hash or "result_sha256" in record:
                raise ValueError(f"Different geometry mask already exists: {target}")
        target.write_bytes(buffer.getvalue())
        record.update({"status": "imported", "result_sha256": sha256(target), "result_size": list(MASK_SIZE),
                       "import": {"raw_source": os.path.relpath(source, results_root),
                                  "raw": os.path.relpath(raw_path, run_dir), "raw_sha256": source_hash,
                                  "image": image_metadata, "provenance": provenance,
                                  "provider": "external; consult source call metadata", "imported_at": _now()}})
    for stage, manifest in manifests.items():
        manifest.update(status="complete", completed_sample_count=len(manifest["samples"]), updated_at=_now())
        _write(output_root / stage / "manifest.json", manifest)
    return output_root


def run_vertex_requests(requests_root, *, project=None, location=None, model=None,
                        timeout=180, max_requests=None) -> dict:
    """Explicit opt-in Google Vertex generateContent adapter; never called by export/import.

    Uses GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, CITYCODE_VERTEX_MODEL and
    GOOGLE_OAUTH_ACCESS_TOKEN. Tokens remain in memory and are never persisted.
    This function issues one request per missing result, without implicit retries.
    REST contract: VERTEX_DOCS. Network execution needs the caller's authorization.
    """
    root = Path(requests_root).resolve()
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
    model = model or os.environ.get("CITYCODE_VERTEX_MODEL")
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if not all((project, location, model, token)):
        raise ValueError("Set project, location, model and GOOGLE_OAUTH_ACCESS_TOKEN explicitly")
    for value in (project, location, model):
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", value):
            raise ValueError("Invalid Vertex project/location/model identifier")
    if max_requests is not None and (not isinstance(max_requests, int) or max_requests < 1):
        raise ValueError("max_requests must be a positive integer")
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    endpoint = f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
    completed = 0
    for stage in STAGES:
        run_dir = root / stage
        manifest = _read(run_dir / "manifest.json")
        for record in manifest["samples"]:
            target = _local(run_dir, record["result"])
            receipt = run_dir / "api_metadata" / f"{record['id']}.json"
            if target.exists():
                if record.get("result_sha256") == sha256(target):
                    continue
                if receipt.is_file() and _read(receipt).get("raw_sha256") == sha256(target):
                    continue
                raise ValueError(f"Untracked/different existing result; import it or use a fresh run: {target}")
            if max_requests is not None and completed >= max_requests:
                return {"completed_requests": completed, "status": "limit_reached"}
            image_path = _local(run_dir, record["input"])
            prompt_path = _local(run_dir, manifest["prompt_snapshot"])
            if sha256(image_path) != record["input_sha256"] or sha256(prompt_path) != manifest["prompt_sha256"]:
                raise ValueError(f"Prepared request changed: {record['id']}")
            payload = {"contents": [{"role": "user", "parts": [
                {"text": prompt_path.read_text(encoding="utf-8")},
                {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image_path.read_bytes()).decode("ascii")}}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                                     "imageConfig": {"aspectRatio": "1:1", "imageSize": "1K"}}}
            request = Request(endpoint, data=json.dumps(payload).encode(),
                              headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(request, timeout=timeout) as response:
                    result = json.load(response)
            except HTTPError as error:
                raise RuntimeError(f"Vertex request failed with HTTP {error.code}; no retry was issued") from None
            images = []
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    inline = part.get("inlineData", {})
                    if inline.get("mimeType", "").startswith("image/") and inline.get("data"):
                        images.append(base64.b64decode(inline["data"], validate=True))
            if len(images) != 1:
                raise ValueError(f"Expected one mask image; got {len(images)} for {record['id']}")
            with Image.open(BytesIO(images[0])) as image:
                image.load()
                if image.size != MASK_SIZE:
                    raise ValueError(f"Vertex output is not the requested 1024-square mask: {image.size}")
                buffer = BytesIO()
                image.save(buffer, format="PNG")
            target.write_bytes(buffer.getvalue())
            _write(receipt, {"provider": "Google Vertex generateContent", "model": model,
                             "project": project, "location": location, "id": record["id"],
                             "source_sha256": record["source_sha256"], "prompt_sha256": manifest["prompt_sha256"],
                             "input_sha256": record["input_sha256"], "raw_sha256": sha256(target),
                             "request_config": payload["generationConfig"], "response_id": result.get("responseId"),
                             "usage_metadata": result.get("usageMetadata"), "completed_at": _now(), "docs": VERTEX_DOCS})
            completed += 1
    return {"completed_requests": completed, "status": "complete"}
