#!/usr/bin/env python3
"""Transcode embedded WebP GLB textures to core glTF JPEG/PNG images.

Geometry buffers, accessors, UVs, materials, node transforms, and PBR texture
indices are preserved. Opaque base-color maps become high-quality 4:4:4 JPEG;
all data or alpha-bearing textures become lossless PNG. This avoids MeshLab's
STB failure on EXT_texture_webp while keeping the output as one self-contained
GLB.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from PIL import Image


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_glb(path: Path) -> tuple[dict, int, int]:
    with path.open("rb") as stream:
        magic, version, total = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2 or total != path.stat().st_size:
            raise ValueError(f"Invalid GLB header: {path}")
        json_length, json_kind = struct.unpack("<II", stream.read(8))
        if json_kind != JSON_CHUNK:
            raise ValueError("First GLB chunk is not JSON")
        document = json.loads(stream.read(json_length))
        binary_length, binary_kind = struct.unpack("<II", stream.read(8))
        if binary_kind != BIN_CHUNK:
            raise ValueError("Second GLB chunk is not BIN")
        binary_offset = stream.tell()
        if binary_offset + binary_length != total:
            raise ValueError("Unexpected extra or truncated GLB chunks")
    if len(document.get("buffers", [])) != 1 or document["buffers"][0].get("uri"):
        raise ValueError("Expected one embedded GLB buffer")
    return document, binary_offset, binary_length


def texture_source(texture: dict) -> int:
    source = texture.get("source")
    webp = texture.get("extensions", {}).get("EXT_texture_webp", {})
    if source is None:
        source = webp.get("source")
    if not isinstance(source, int):
        raise ValueError(f"Texture has no embedded image source: {texture}")
    return source


def classify_images(document: dict) -> tuple[set[int], set[int]]:
    """Return images safe for JPEG and images that must remain lossless."""
    textures = document.get("textures", [])
    opaque_color: set[int] = set()
    lossless: set[int] = set()
    for material in document.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        base = pbr.get("baseColorTexture")
        if base:
            source = texture_source(textures[base["index"]])
            if material.get("alphaMode", "OPAQUE") == "OPAQUE":
                opaque_color.add(source)
            else:
                lossless.add(source)
        for key, value in pbr.items():
            if key != "baseColorTexture" and isinstance(value, dict) and isinstance(value.get("index"), int):
                lossless.add(texture_source(textures[value["index"]]))
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            value = material.get(key)
            if isinstance(value, dict) and isinstance(value.get("index"), int):
                lossless.add(texture_source(textures[value["index"]]))
    opaque_color -= lossless
    all_images = set(range(len(document.get("images", []))))
    lossless |= all_images - opaque_color
    return opaque_color, lossless


def copy_range(source, destination, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("Source GLB ended inside a bufferView")
        destination.write(chunk)
        remaining -= len(chunk)


def remove_webp_extension(document: dict) -> None:
    for texture in document.get("textures", []):
        extensions = texture.get("extensions", {})
        webp = extensions.pop("EXT_texture_webp", None)
        if "source" not in texture and webp and isinstance(webp.get("source"), int):
            texture["source"] = webp["source"]
        if not extensions:
            texture.pop("extensions", None)
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in document:
            document[key] = [value for value in document[key] if value != "EXT_texture_webp"]
            if not document[key]:
                del document[key]


def write_glb(destination: Path, document: dict, input_path: Path,
              input_binary_offset: int, old_views: list[dict],
              converted: dict[int, Path]) -> None:
    json_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary_length = document["buffers"][0]["byteLength"]
    total = 12 + 8 + len(json_bytes) + 8 + binary_length
    if total >= 2 ** 32:
        raise ValueError(f"GLB exceeds the 4 GiB format limit: {total} bytes")
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with input_path.open("rb") as source, temporary.open("wb") as output:
        output.write(struct.pack("<4sII", b"glTF", 2, total))
        output.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        output.write(json_bytes)
        output.write(struct.pack("<II", binary_length, BIN_CHUNK))
        written = 0
        for index, (old_view, new_view) in enumerate(zip(old_views, document["bufferViews"])):
            target = new_view.get("byteOffset", 0)
            output.write(b"\0" * (target - written))
            if index in converted:
                with converted[index].open("rb") as image_stream:
                    shutil.copyfileobj(image_stream, output, 16 * 1024 * 1024)
            else:
                copy_range(
                    source,
                    output,
                    input_binary_offset + old_view.get("byteOffset", 0),
                    old_view["byteLength"],
                )
            written = target + new_view["byteLength"]
        output.write(b"\0" * (binary_length - written))
    os.replace(temporary, destination)


def convert(input_path: Path, destination: Path, quality: int) -> dict:
    source_document, binary_offset, _ = parse_glb(input_path)
    document = copy.deepcopy(source_document)
    old_views = copy.deepcopy(source_document.get("bufferViews", []))
    opaque_color, lossless = classify_images(document)
    image_views = [image["bufferView"] for image in document.get("images", [])]
    if len(image_views) != len(set(image_views)):
        raise ValueError("Multiple images share one bufferView; refusing ambiguous conversion")

    destination.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[int, Path] = {}
    jpeg_count = png_count = 0
    with tempfile.TemporaryDirectory(prefix="glb_texture_transcode_", dir=destination.parent) as temp_name:
        temp_root = Path(temp_name)
        with input_path.open("rb") as source:
            for image_index, image in enumerate(document.get("images", [])):
                view_index = image["bufferView"]
                view = old_views[view_index]
                source.seek(binary_offset + view.get("byteOffset", 0))
                payload = source.read(view["byteLength"])
                decoded = Image.open(io.BytesIO(payload))
                decoded.load()
                if image_index in opaque_color and image_index not in lossless:
                    target = temp_root / f"{image_index:04d}.jpg"
                    decoded.convert("RGB").save(
                        target, "JPEG", quality=quality, subsampling=0,
                        optimize=True, progressive=False,
                    )
                    image["mimeType"] = "image/jpeg"
                    jpeg_count += 1
                else:
                    target = temp_root / f"{image_index:04d}.png"
                    decoded.save(target, "PNG", compress_level=6, optimize=False)
                    image["mimeType"] = "image/png"
                    png_count += 1
                converted[view_index] = target
                print(json.dumps({
                    "converted": image_index + 1,
                    "total": len(document["images"]),
                    "mime_type": image["mimeType"],
                    "size_bytes": target.stat().st_size,
                }), flush=True)

        remove_webp_extension(document)
        cursor = 0
        for index, view in enumerate(document.get("bufferViews", [])):
            cursor = (cursor + 3) & ~3
            view["buffer"] = 0
            view["byteOffset"] = cursor
            view["byteLength"] = converted[index].stat().st_size if index in converted else old_views[index]["byteLength"]
            cursor += view["byteLength"]
        cursor = (cursor + 3) & ~3
        document["buffers"][0]["byteLength"] = cursor
        write_glb(destination, document, input_path, binary_offset, old_views, converted)

    verified, _, _ = parse_glb(destination)
    mime_types = sorted({image.get("mimeType") for image in verified.get("images", [])})
    if "EXT_texture_webp" in verified.get("extensionsUsed", []):
        raise ValueError("Output still declares EXT_texture_webp")
    if any(value not in {"image/jpeg", "image/png"} for value in mime_types):
        raise ValueError(f"Unexpected output image types: {mime_types}")
    return {
        "schema": "air2vectorcity.glb_texture_conversion.v1",
        "input": str(input_path.resolve()),
        "output": str(destination.resolve()),
        "input_size_bytes": input_path.stat().st_size,
        "output_size_bytes": destination.stat().st_size,
        "jpeg_quality": quality,
        "jpeg_images": jpeg_count,
        "png_images": png_count,
        "geometry_buffer_views_preserved": len(old_views) - len(converted),
        "image_buffer_views_transcoded": len(converted),
        "mime_types": mime_types,
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    metadata = convert(args.input, args.output, args.jpeg_quality)
    metadata_path = args.output.with_suffix(".conversion.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
