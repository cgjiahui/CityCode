#!/usr/bin/env python3
"""Create resumable MeshLab-compatible JPEG/PNG copies of fused GLB tiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from convert_glb_textures import convert, parse_glb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output-dir", default="tiles_meshlab")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compatible(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        document, _, _ = parse_glb(path)
    except Exception:
        return False
    mime_types = {image.get("mimeType") for image in document.get("images", [])}
    return (
        bool(document.get("images"))
        and mime_types <= {"image/jpeg", "image/png"}
        and "EXT_texture_webp" not in document.get("extensionsUsed", [])
        and all(isinstance(texture.get("source"), int) for texture in document.get("textures", []))
    )


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    scene_path = args.fusion / "scene_manifest.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    source_records = scene.get("tiles", [])
    if not source_records:
        raise ValueError(f"No tiles in {scene_path}")
    output_root = args.fusion / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    total_input = total_output = jpeg_images = png_images = 0
    for index, source_record in enumerate(source_records, start=1):
        patch_id = source_record["patch_id"]
        source = Path(source_record["tile"])
        if not source.is_absolute():
            source = args.fusion / source
        destination = output_root / f"{patch_id}.glb"
        metadata_path = destination.with_suffix(".conversion.json")
        if args.overwrite or not compatible(destination) or not metadata_path.is_file():
            metadata = convert(source, destination, args.jpeg_quality)
            atomic_json(metadata_path, metadata)
            action = "converted"
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            action = "reused"
        record = {
            "patch_id": patch_id,
            "source": str(source.resolve()),
            "output": str(destination.resolve()),
            "url": f"{args.output_dir}/{destination.name}",
            "input_size_bytes": source.stat().st_size,
            "output_size_bytes": destination.stat().st_size,
            "faces_owned": source_record.get("faces_owned"),
            "jpeg_images": metadata["jpeg_images"],
            "png_images": metadata["png_images"],
        }
        records.append(record)
        total_input += record["input_size_bytes"]
        total_output += record["output_size_bytes"]
        jpeg_images += record["jpeg_images"]
        png_images += record["png_images"]
        print(json.dumps({
            "tile": index,
            "total": len(source_records),
            "patch_id": patch_id,
            "action": action,
            "output_size_bytes": record["output_size_bytes"],
        }), flush=True)

    manifest = {
        "schema": "air2vectorcity.meshlab_compatible_tiles.v1",
        "status": "complete",
        "source_scene_manifest": str(scene_path.resolve()),
        "jpeg_quality": args.jpeg_quality,
        "tile_count_expected": len(source_records),
        "tile_count_available": len(records),
        "total_input_size_bytes": total_input,
        "total_output_size_bytes": total_output,
        "jpeg_images": jpeg_images,
        "png_images": png_images,
        "geometry_and_uv_preserved": True,
        "records": records,
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest | {"records": f"{len(records)} records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
