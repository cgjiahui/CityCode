"""Native image conditioning with explicit HTTPS configuration and hash receipts."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from PIL import Image, __version__ as PILLOW_VERSION

MODEL = "gemini-3-pro-image-preview"
QUALITY = "native_1k"
GENERATION_CONFIG = {"responseModalities": ["IMAGE"],
                     "imageConfig": {"aspectRatio": "1:1", "imageSize": "1K"}}
DEFAULT_CONFIG = {"schema": "citycode.appearance_provider.v1", "protocol": "gemini_native",
                  "model": MODEL, "quality": QUALITY, "size": "1024x1024",
                  "base_url": "https://api.boyuerichdata.opensphereai.com/v1",
                  "api_key_env": "OPENAI_API_KEY", "timeout_seconds": 600,
                  "generation_config": GENERATION_CONFIG}
INPUT_NAMES = ("lod1_main_oblique.png", "source_primary.png", "roof_evidence.png", "context.png")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def provider_config(value=None):
    """Validate non-secret configuration; no environment or network is read."""
    config = copy.deepcopy(DEFAULT_CONFIG if value is None else value)
    if not isinstance(config, dict) or set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Provider configuration must contain only the documented non-secret fields")
    for key in ("schema", "protocol", "quality", "size", "generation_config"):
        if config[key] != DEFAULT_CONFIG[key]:
            raise ValueError("Unsupported standard appearance setting: " + key)
    if not isinstance(config["model"], str) or not re.fullmatch(r"gemini-[A-Za-z0-9._-]+", config["model"]):
        raise ValueError("Native provider model must be an explicit Gemini model identifier")
    if not isinstance(config["base_url"], str):
        raise ValueError("Provider base URL must be an HTTPS string")
    parsed = urlsplit(config["base_url"])
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in ("", "/v1")):
        raise ValueError("Provider base URL must be HTTPS without credentials, query or redirects")
    if not isinstance(config["api_key_env"], str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", config["api_key_env"]):
        raise ValueError("api_key_env must name an environment variable")
    if type(config["timeout_seconds"]) is not int or not 1 <= config["timeout_seconds"] <= 900:
        raise ValueError("Provider timeout must be an integer in 1..900 seconds")
    return config


def transport_for(config=None):
    config = provider_config(config)
    return {"protocol": "gemini_native", "provider_host": urlsplit(config["base_url"]).hostname,
            "endpoint_path": f"/v1beta/models/{config['model']}:generateContent",
            "generation_config": copy.deepcopy(config["generation_config"])}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError("Provider redirects are not allowed")


def build_native_request(images, prompt, *, root, config=None):
    """Build exact four-PNG-then-text native bytes without storing base64 copies."""
    config = provider_config(config)
    root = Path(root).resolve()
    if len(images) != 4 or len({Path(p).resolve() for p in images}) != 4:
        raise ValueError("Exactly four distinct prepared PNG references are required")
    parts, inputs = [], []
    for value in images:
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Reference must remain inside the prepared appearance directory")
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError("Every reference must be an actual PNG")
            mode, size = image.mode, list(image.size)
        inputs.append({"path": path.relative_to(root).as_posix(), "sha256": _sha(data),
                       "bytes": len(data), "mode": mode, "size": size, "mime_type": "image/png"})
        parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(data).decode()}})
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("The prepared prompt cannot be empty")
    parts.append({"text": prompt})
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config["generation_config"]}
    wire = json.dumps(body).encode()
    transport = transport_for(config)
    metadata = {"schema": "citycode.native_image_request.v1", "model": config["model"],
                "quality": config["quality"], "provider_host": transport["provider_host"],
                "endpoint_path": transport["endpoint_path"], "method": "POST", "transport": transport,
                "inputs": inputs, "prompt_sha256": _sha(prompt.encode()),
                "actual_http_request_sha256": _sha(wire), "actual_http_request_bytes": len(wire),
                "generationConfig": config["generation_config"]}
    return wire, metadata


def _png_contract(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.format != "PNG" or image.size != (1024, 1024):
            raise ValueError("Provider output must already be an actual 1024 x 1024 PNG")
    return {"format": "PNG", "size": [1024, 1024], "image_sha256": _sha(data), "image_bytes": len(data)}


class _NativeResponseError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _payload_info(data, mime_type, *, path=None):
    info = {"path": path, "mime_type": mime_type, "sha256": _sha(data), "bytes": len(data),
            "format": None, "mode": None, "size": None}
    try:
        with Image.open(io.BytesIO(data)) as image:
            info.update(format=image.format, mode=image.mode, size=list(image.size))
    except (OSError, ValueError):
        pass
    return info


def _retain_payload(directory, data, info):
    if directory is None:
        return None
    suffix = {"PNG": ".png", "JPEG": ".jpg"}.get(info["format"], ".bin")
    relative = Path("provider_payload") / (info["sha256"] + suffix)
    path = Path(directory) / relative
    if not path.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("Retained provider payload must remain inside its owned directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Retained provider payload bytes changed")
    path.write_bytes(data)
    return relative.as_posix()


def _normalize_payload(data, mime_type, *, original_path=None):
    original = _payload_info(data, mime_type, path=original_path)
    expected = {"image/png": "PNG", "image/jpeg": "JPEG"}.get(mime_type)
    if expected is None or original["format"] != expected:
        raise ValueError("Provider image MIME must match its actual PNG or JPEG encoding")
    if original["size"] != [1024, 1024]:
        raise ValueError("Provider output must already be an actual 1024 x 1024 PNG or JPEG")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        pixel_mode = "RGB" if expected == "JPEG" else "RGBA"
        decoded = image.convert(pixel_mode)
        source_pixel_sha = _sha(decoded.tobytes())
        if expected == "PNG":
            canonical = data
        else:
            stream = io.BytesIO()
            decoded.save(stream, format="PNG")
            canonical = stream.getvalue()
    with Image.open(io.BytesIO(canonical)) as image:
        image.load()
        canonical_mode = image.mode
        canonical_pixel_sha = _sha(image.convert(pixel_mode).tobytes())
    if source_pixel_sha != canonical_pixel_sha:
        raise ValueError("Payload normalization changed decoded pixels")
    proof = {"schema": "citycode.provider_payload_normalization.v1", "original_payload": original,
             "canonical_png": {"path": "generated_main_oblique.png", "mime_type": "image/png", "format": "PNG",
                               "sha256": _sha(canonical), "bytes": len(canonical), "mode": canonical_mode,
                               "size": [1024, 1024]},
             "conversion": {"operation": "jpeg_decode_to_png" if expected == "JPEG" else "none_original_png",
                            "resized": False, "edited": False, "orientation_transformed": False,
                            "pixel_mode": pixel_mode, "source_pixel_sha256": source_pixel_sha,
                            "canonical_pixel_sha256": canonical_pixel_sha, "pixels_equal": True,
                            "pillow_version": PILLOW_VERSION}}
    return canonical, proof


def _validate_payload_provenance(directory, proof, canonical):
    original = proof.get("original_payload", {})
    relative = Path(original.get("path") or "")
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("provider_payload",)
            or not path.is_relative_to(directory)):
        raise ValueError("Original provider payload must remain inside its owned directory")
    data = path.read_bytes()
    actual_original = _payload_info(data, original.get("mime_type"), path=relative.as_posix())
    actual_canonical = _payload_info(canonical, "image/png", path="generated_main_oblique.png")
    expected_format = {"image/png": "PNG", "image/jpeg": "JPEG"}.get(original.get("mime_type"))
    if (proof.get("schema") != "citycode.provider_payload_normalization.v1" or actual_original != original
            or expected_format is None or actual_original["format"] != expected_format
            or actual_original["size"] != [1024, 1024] or actual_canonical != proof.get("canonical_png")
            or actual_canonical["format"] != "PNG" or actual_canonical["size"] != [1024, 1024]):
        raise ValueError("Original provider payload or actual canonical PNG proof changed")
    conversion = proof.get("conversion", {})
    pixel_mode = "RGB" if expected_format == "JPEG" else "RGBA"
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        source_pixel_sha = _sha(image.convert(pixel_mode).tobytes())
    with Image.open(io.BytesIO(canonical)) as image:
        image.load()
        canonical_pixel_sha = _sha(image.convert(pixel_mode).tobytes())
    # The historical encoder version is provenance. Cache verification uses
    # actual file hashes and decoded pixels, never a new PNG encoding.
    expected_conversion = {"operation": "jpeg_decode_to_png" if expected_format == "JPEG" else "none_original_png",
                           "resized": False, "edited": False, "orientation_transformed": False,
                           "pixel_mode": pixel_mode, "source_pixel_sha256": source_pixel_sha,
                           "canonical_pixel_sha256": canonical_pixel_sha, "pixels_equal": True,
                           "pillow_version": conversion.get("pillow_version")}
    if (not isinstance(conversion.get("pillow_version"), str) or not conversion["pillow_version"]
            or conversion != expected_conversion or source_pixel_sha != canonical_pixel_sha
            or (expected_format == "PNG" and data != canonical)
            or (expected_format == "JPEG" and actual_canonical["mode"] != "RGB")):
        raise ValueError("Original provider payload or decoded pixel conversion proof changed")


def decode_native_response(result, *, directory=None):
    images, candidates = [], []
    diagnostics = {"candidates": candidates, "non_thought_image_count": 0,
                   "usageMetadata": result.get("usageMetadata"), "modelVersion": result.get("modelVersion"),
                   "responseId": result.get("responseId")}
    for candidate_index, candidate in enumerate(result.get("candidates", [])):
        parts = []
        candidates.append({"candidate_index": candidate_index, "finishReason": candidate.get("finishReason"), "parts": parts})
        for part_index, part in enumerate(candidate.get("content", {}).get("parts", [])):
            inline = part.get("inlineData", part.get("inline_data"))
            if inline:
                try:
                    data = base64.b64decode(inline["data"], validate=True)
                except (KeyError, TypeError, ValueError):
                    parts.append({"part_index": part_index, "invalid_base64": True, "thought": bool(part.get("thought"))})
                    raise _NativeResponseError("Provider image base64 is invalid", diagnostics) from None
                mime_type = inline.get("mimeType", inline.get("mime_type"))
                info = _payload_info(data, mime_type)
                info["path"] = _retain_payload(directory, data, info)
                parts.append({"part_index": part_index, "mimeType": mime_type,
                              "image_payload_sha256": _sha(data), "image_payload_bytes": len(data),
                              "original_payload": info, "thought": bool(part.get("thought"))})
                if not part.get("thought"):
                    images.append((data, mime_type, info["path"]))
                    diagnostics["non_thought_image_count"] += 1
            elif "text" in part:
                parts.append({"text_sha256": _sha(part["text"].encode()), "text_characters": len(part["text"]),
                              "thought": bool(part.get("thought"))})
    if len(images) != 1:
        raise _NativeResponseError("Provider must return exactly one non-thought image", diagnostics)
    data, mime_type, original_path = images[0]
    try:
        canonical, proof = _normalize_payload(data, mime_type, original_path=original_path)
    except (OSError, ValueError) as error:
        raise _NativeResponseError(str(error), diagnostics) from None
    return canonical, {"status": "provider_returned", **diagnostics,
                       "payload_provenance": proof, **_png_contract(canonical)}


def _send_native(wire, config, secret):
    transport = transport_for(config)
    endpoint = "https://" + transport["provider_host"] + transport["endpoint_path"]
    request = urllib.request.Request(endpoint, data=wire, method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": secret})
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=config["timeout_seconds"]) as response:
        return json.loads(response.read()), response.status


def _failure(error, secret):
    result = {"status": "failed", "exception": type(error).__name__, "status_code": getattr(error, "code", None)}
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = json.loads(error.read(16000))
            detail = body.get("error", body)
            if isinstance(detail, dict):
                def redact(value):
                    value = str(value).replace(secret, "<redacted>")
                    return re.sub(r"[A-Za-z0-9+/]{100,}={0,2}", "<payload-redacted>", value)[:1000]
                result["provider_error"] = {key: redact(detail.get(key, "")) for key in ("type", "code", "message")}
        except (ValueError, OSError):
            pass
    return result


def complete_native(directory, record, *, config=None, timeout_seconds=None):
    """Make one explicit request or reuse a fully verified identical completion."""
    from .appearance import completion_contract, resolve_local, sha256_file, write_json

    directory = Path(directory).resolve()
    config = provider_config(config)
    if timeout_seconds is not None:
        config["timeout_seconds"] = timeout_seconds
        config = provider_config(config)
    transport = transport_for(config)
    if any(record.get(key) != config[key] for key in ("model", "quality", "size")) or record.get("transport") != transport:
        raise ValueError("Prepared appearance and configured native provider identities differ")
    if [row["path"] for row in record["inputs"]] != ["inputs/" + name for name in INPUT_NAMES]:
        raise ValueError("Prepared references must retain the standard four-image order")
    prompt = resolve_local(directory, record["prompt"]).read_text().strip()
    if _sha(prompt.encode()) != record["prompt_sha256"]:
        raise ValueError("Prompt changed after preparation")
    images = [resolve_local(directory, row["path"]) for row in record["inputs"]]
    wire, native_request = build_native_request(images, prompt, root=directory, config=config)
    for prepared, actual in zip(record["inputs"], native_request["inputs"]):
        if any(prepared[key] != actual[key] for key in ("path", "sha256", "mode", "size")):
            raise ValueError("Actual appearance evidence changed after preparation")
    identity = completion_contract(record)
    output = directory / "generated_main_oblique.png"
    receipt_path = directory / "completion.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous.get("status") == "complete":
            if previous.get("request_identity_sha256") != identity:
                raise ValueError("Completed appearance has a different request identity; use a new directory")
            for name, key in (("request.json", "request_sha256"), ("provider_request.json", "provider_request_sha256"),
                              ("provider_response.json", "provider_response_sha256"), (output.name, "output_sha256")):
                if sha256_file(resolve_local(directory, name)) != previous.get(key):
                    raise ValueError("Completed appearance evidence changed: " + name)
            canonical = json.loads((directory / "request.json").read_text())
            native = json.loads((directory / "provider_request.json").read_text())
            response = json.loads((directory / "provider_response.json").read_text())
            for item in (previous, canonical, native, response):
                if (item.get("actual_http_request_sha256") != native_request["actual_http_request_sha256"]
                        or item.get("actual_http_request_bytes") != len(wire)
                        or item.get("transport") != transport):
                    raise ValueError("Completed native wire evidence does not match the current request")
            if canonical.get("inputs") != record["inputs"] or canonical.get("prompt_sha256") != record["prompt_sha256"]:
                raise ValueError("Completed native request does not match the prepared inputs")
            if response.get("status") != "provider_returned" or response.get("image_sha256") != previous["output_sha256"]:
                raise ValueError("Completed native response does not bind the output image")
            canonical_data = output.read_bytes()
            _png_contract(canonical_data)
            proof = response.get("payload_provenance")
            if proof is not None or previous.get("payload_provenance") is not None:
                if previous.get("payload_provenance") != proof:
                    raise ValueError("Completed payload proof differs from its receipt")
                _validate_payload_provenance(directory, proof, canonical_data)
            return previous
    secret = os.environ.get(config["api_key_env"], "")
    if not secret:
        raise RuntimeError("Configure the provider API key in " + config["api_key_env"])
    native_request["provider_code_sha256"] = sha256_file(Path(__file__))
    write_json(directory / "provider_request.json", native_request)
    request = {"schema": "citycode.appearance_request.v1", "building_id": record["building_id"],
               "request_identity_sha256": identity, "model": config["model"], "quality": config["quality"],
               "size": config["size"], "n": 1, "output_format": "png", "prompt": prompt,
               "prompt_sha256": record["prompt_sha256"], "inputs": record["inputs"], "transport": transport,
               "provider_host": transport["provider_host"], "endpoint_path": transport["endpoint_path"],
               "generationConfig": config["generation_config"], "provider_code_sha256": native_request["provider_code_sha256"],
               "actual_http_request_sha256": native_request["actual_http_request_sha256"], "actual_http_request_bytes": len(wire)}
    write_json(directory / "request.json", request)
    started = time.monotonic()
    receipt = {"schema": "citycode.appearance_completion.v1", "building_id": record["building_id"],
               "request_identity_sha256": identity, "model": config["model"], "quality": config["quality"],
               "size": config["size"], "transport": transport, "provider_host": transport["provider_host"],
               "endpoint_path": transport["endpoint_path"], "request_sha256": sha256_file(directory / "request.json"),
               "provider_request_sha256": sha256_file(directory / "provider_request.json"),
               "actual_http_request_sha256": native_request["actual_http_request_sha256"], "actual_http_request_bytes": len(wire)}
    status = None
    try:
        result, status = _send_native(wire, config, secret)
        if status != 200:
            raise ValueError("Provider did not return HTTP200")
        data, response = decode_native_response(result, directory=directory)
        response.update(status_code=status, model=config["model"], quality=config["quality"], transport=transport,
                        provider_host=transport["provider_host"], endpoint_path=transport["endpoint_path"],
                        actual_http_request_sha256=native_request["actual_http_request_sha256"],
                        actual_http_request_bytes=len(wire), provider_request_sha256=receipt["provider_request_sha256"],
                        elapsed_seconds=time.monotonic() - started)
        write_json(directory / "provider_response.json", response)
        temporary = output.with_name("generated_main_oblique.partial.png")
        temporary.write_bytes(data)
        temporary.replace(output)
        receipt.update(status="complete", returncode=0, output=output.name, output_sha256=_sha(data),
                       payload_provenance=response["payload_provenance"])
    except Exception as error:
        response = _failure(error, secret)
        if status is not None:
            response["status_code"] = status
        if isinstance(error, _NativeResponseError):
            response.update(error.diagnostics, validation_error=str(error))
        response.update(model=config["model"], quality=config["quality"], transport=transport,
                        actual_http_request_sha256=native_request["actual_http_request_sha256"],
                        actual_http_request_bytes=len(wire), elapsed_seconds=time.monotonic() - started)
        write_json(directory / "provider_response.json", response)
        receipt.update(status="failed", returncode=1, error="Native provider completion failed; see provider_response.json")
        receipt.update(provider_response_sha256=sha256_file(directory / "provider_response.json"),
                       elapsed_seconds=time.monotonic() - started)
        write_json(receipt_path, receipt)
        raise RuntimeError(receipt["error"]) from None
    receipt.update(provider_response_sha256=sha256_file(directory / "provider_response.json"),
                   elapsed_seconds=time.monotonic() - started)
    write_json(receipt_path, receipt)
    return receipt
