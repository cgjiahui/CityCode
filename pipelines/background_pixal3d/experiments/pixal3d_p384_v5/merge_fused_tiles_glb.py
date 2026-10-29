#!/usr/bin/env python3
"""Losslessly package fused GLB tiles as one browser-loadable GLB.

Unlike a trimesh import/export round trip, this merger preserves embedded WebP
images and every glTF material/UV reference byte-for-byte. Geometry buffers are
copied without decoding, quantization, simplification, or coordinate changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import struct
from pathlib import Path


JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--manifest-name", default="scene_manifest.json")
    parser.add_argument("--output-name", default="full_scene_single_owner.glb")
    return parser.parse_args()


def parse_glb(path: Path) -> tuple[dict, int, int]:
    with path.open("rb") as stream:
        magic, version, total = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2 or total != path.stat().st_size:
            raise ValueError(f"Invalid GLB header: {path}")
        json_length, json_kind = struct.unpack("<II", stream.read(8))
        if json_kind != JSON_CHUNK:
            raise ValueError(f"First chunk is not JSON: {path}")
        document = json.loads(stream.read(json_length))
        binary_length, binary_kind = struct.unpack("<II", stream.read(8))
        if binary_kind != BIN_CHUNK:
            raise ValueError(f"Second chunk is not BIN: {path}")
        binary_offset = stream.tell()
        if binary_offset + binary_length != total:
            raise ValueError(f"Unexpected chunks or length: {path}")
    buffers = document.get("buffers", [])
    if len(buffers) != 1 or buffers[0].get("uri"):
        raise ValueError(f"Expected one embedded buffer: {path}")
    if buffers[0]["byteLength"] > binary_length:
        raise ValueError(f"Declared buffer exceeds BIN chunk: {path}")
    return document, binary_offset, binary_length


def shift_material_texture_indices(value, texture_offset: int) -> None:
    if isinstance(value, dict):
        # Within a material document every `index` member is a texture-info
        # reference (base color, MR, normal, occlusion, emissive, or extension).
        if isinstance(value.get("index"), int):
            value["index"] += texture_offset
        for child in value.values():
            shift_material_texture_indices(child, texture_offset)
    elif isinstance(value, list):
        for child in value:
            shift_material_texture_indices(child, texture_offset)


def merge_documents(sources: list[tuple[Path, dict, int, int]], expected_faces: int) -> tuple[dict, list[int]]:
    combined = {
        "asset": {"version": "2.0", "generator": "Air2VectorCity lossless fused-tile packager"},
        "scene": 0,
        "scenes": [{"name": "P384 final single-owner fusion", "nodes": []}],
        "buffers": [{"byteLength": 0}],
        "bufferViews": [],
        "accessors": [],
        "images": [],
        "samplers": [],
        "textures": [],
        "materials": [],
        "meshes": [],
        "nodes": [],
    }
    extensions_used: set[str] = set()
    extensions_required: set[str] = set()
    binary_bases: list[int] = []
    binary_cursor = 0

    for path, document, _, binary_length in sources:
        if document.get("animations") or document.get("skins") or document.get("cameras"):
            raise ValueError(f"Unsupported animated/skinned/camera tile: {path}")
        if document.get("extensions"):
            raise ValueError(f"Unsupported top-level extensions: {path}")
        offsets = {key: len(combined[key]) for key in (
            "bufferViews", "accessors", "images", "samplers", "textures",
            "materials", "meshes", "nodes",
        )}
        binary_cursor = (binary_cursor + 3) & ~3
        binary_bases.append(binary_cursor)

        for source in document.get("bufferViews", []):
            item = copy.deepcopy(source)
            item["buffer"] = 0
            item["byteOffset"] = binary_cursor + item.get("byteOffset", 0)
            combined["bufferViews"].append(item)

        for source in document.get("accessors", []):
            item = copy.deepcopy(source)
            if "bufferView" in item:
                item["bufferView"] += offsets["bufferViews"]
            sparse = item.get("sparse")
            if sparse:
                sparse["indices"]["bufferView"] += offsets["bufferViews"]
                sparse["values"]["bufferView"] += offsets["bufferViews"]
            combined["accessors"].append(item)

        for source in document.get("images", []):
            item = copy.deepcopy(source)
            if item.get("uri"):
                raise ValueError(f"External image URI is not supported: {path}")
            item["bufferView"] += offsets["bufferViews"]
            combined["images"].append(item)

        combined["samplers"].extend(copy.deepcopy(document.get("samplers", [])))
        for source in document.get("textures", []):
            item = copy.deepcopy(source)
            if "sampler" in item:
                item["sampler"] += offsets["samplers"]
            if "source" in item:
                item["source"] += offsets["images"]
            webp = item.get("extensions", {}).get("EXT_texture_webp")
            if webp and "source" in webp:
                webp["source"] += offsets["images"]
            combined["textures"].append(item)

        for source in document.get("materials", []):
            item = copy.deepcopy(source)
            shift_material_texture_indices(item, offsets["textures"])
            combined["materials"].append(item)

        for source in document.get("meshes", []):
            item = copy.deepcopy(source)
            for primitive in item.get("primitives", []):
                primitive["attributes"] = {
                    name: accessor + offsets["accessors"]
                    for name, accessor in primitive.get("attributes", {}).items()
                }
                if "indices" in primitive:
                    primitive["indices"] += offsets["accessors"]
                if "material" in primitive:
                    primitive["material"] += offsets["materials"]
                for target in primitive.get("targets", []):
                    for name in tuple(target):
                        target[name] += offsets["accessors"]
                draco = primitive.get("extensions", {}).get("KHR_draco_mesh_compression")
                if draco:
                    draco["bufferView"] += offsets["bufferViews"]
            combined["meshes"].append(item)

        for source in document.get("nodes", []):
            item = copy.deepcopy(source)
            if "mesh" in item:
                item["mesh"] += offsets["meshes"]
            if "children" in item:
                item["children"] = [child + offsets["nodes"] for child in item["children"]]
            combined["nodes"].append(item)

        source_scene = document.get("scenes", [{}])[document.get("scene", 0)]
        combined["scenes"][0]["nodes"].extend(
            node + offsets["nodes"] for node in source_scene.get("nodes", [])
        )
        extensions_used.update(document.get("extensionsUsed", []))
        extensions_required.update(document.get("extensionsRequired", []))
        binary_cursor += binary_length

    combined["buffers"][0]["byteLength"] = binary_cursor
    if extensions_used:
        combined["extensionsUsed"] = sorted(extensions_used)
    if extensions_required:
        combined["extensionsRequired"] = sorted(extensions_required)

    triangles = 0
    for mesh in combined["meshes"]:
        for primitive in mesh.get("primitives", []):
            triangles += combined["accessors"][primitive["indices"]]["count"] // 3
    if triangles != expected_faces:
        raise ValueError(f"Face count mismatch: merged={triangles}, expected={expected_faces}")
    # glTF optional top-level arrays must be omitted when empty. In particular,
    # textures may legally omit a sampler, leaving the merged sampler list empty.
    for key in ("images", "samplers", "textures", "materials", "meshes", "nodes"):
        if not combined[key]:
            del combined[key]
    return combined, binary_bases


def write_glb(destination: Path, document: dict, sources, binary_bases: list[int]) -> None:
    json_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    binary_length = document["buffers"][0]["byteLength"]
    total = 12 + 8 + len(json_bytes) + 8 + binary_length
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as output:
        output.write(struct.pack("<4sII", b"glTF", 2, total))
        output.write(struct.pack("<II", len(json_bytes), JSON_CHUNK))
        output.write(json_bytes)
        output.write(struct.pack("<II", binary_length, BIN_CHUNK))
        written = 0
        for (path, _, binary_offset, source_length), base in zip(sources, binary_bases):
            if base < written:
                raise ValueError("Overlapping binary ranges")
            output.write(b"\0" * (base - written))
            with path.open("rb") as source:
                source.seek(binary_offset)
                remaining = source_length
                while remaining:
                    chunk = source.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"Truncated BIN chunk: {path}")
                    output.write(chunk)
                    remaining -= len(chunk)
            written = base + source_length
        output.write(b"\0" * (binary_length - written))
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    manifest_path = args.fusion / args.manifest_name
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("tiles", [])
    if not records:
        raise ValueError(f"No fused tiles in {manifest_path}")
    sources = []
    for index, record in enumerate(records, start=1):
        tile = Path(record.get("tile") or record.get("tile_url"))
        if not tile.is_absolute():
            # Older manifests may contain a cwd-relative path that already
            # includes the fusion directory, while tile_url is fusion-relative.
            tile = tile if tile.is_file() else args.fusion / tile
        document, binary_offset, binary_length = parse_glb(tile)
        sources.append((tile, document, binary_offset, binary_length))
        print(json.dumps({"scanned": index, "total": len(records), "patch_id": record["patch_id"]}), flush=True)

    expected_faces = int(manifest["total_faces_owned"])
    document, binary_bases = merge_documents(sources, expected_faces)
    destination = args.fusion / args.output_name
    write_glb(destination, document, sources, binary_bases)
    verified_document, _, _ = parse_glb(destination)
    # A tile can legitimately contain multiple glTF meshes (for example the
    # base and elevated surface components).  Compare against all source
    # meshes, not against the number of tile files.
    expected_meshes = sum(len(item[1].get("meshes", [])) for item in sources)
    if len(verified_document.get("meshes", [])) != expected_meshes:
        raise ValueError(
            "Output mesh count mismatch: "
            f"merged={len(verified_document.get('meshes', []))}, "
            f"expected={expected_meshes}, tiles={len(records)}"
        )
    if len(verified_document.get("images", [])) != sum(len(item[1].get("images", [])) for item in sources):
        raise ValueError("Output image count mismatch")

    manifest.update({
        "monolithic": str(destination.resolve()),
        "monolithic_size_bytes": destination.stat().st_size,
        "monolithic_faces": expected_faces,
        "monolithic_geometry_count": len(document["meshes"]),
        "monolithic_material_count": len(document["materials"]),
        "monolithic_image_count": len(document["images"]),
        "monolithic_packaging": "lossless GLB buffer concatenation; no geometry or texture re-encoding",
    })
    temporary_manifest = manifest_path.with_suffix(".json.part")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({
        "status": "complete",
        "output": str(destination),
        "size_bytes": destination.stat().st_size,
        "faces": expected_faces,
        "meshes": len(document["meshes"]),
        "materials": len(document["materials"]),
        "images": len(document["images"]),
    }), flush=True)


if __name__ == "__main__":
    main()
