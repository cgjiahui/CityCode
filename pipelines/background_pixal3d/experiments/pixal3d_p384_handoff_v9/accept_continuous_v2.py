#!/usr/bin/env python3
"""Acceptance contract for the hole-free continuous V9.1 fusion."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def glb_document(path):
    path = Path(path)
    with path.open("rb") as stream:
        magic, version, total = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2 or total != path.stat().st_size:
            raise ValueError(f"Invalid GLB header: {path}")
        length, kind = struct.unpack("<II", stream.read(8))
        if kind != 0x4E4F534A:
            raise ValueError(f"Missing JSON chunk: {path}")
        return json.loads(stream.read(length))


def check(name, value, operator, threshold, passed):
    return {"name": name, "value": value, "operator": operator, "threshold": threshold, "pass": bool(passed)}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = read_json(args.fusion / "scene_manifest.json")
    preview = read_json(args.fusion / "full_scene_top_surface_2m.json")
    reference = read_json(args.reference_validation)
    tile_metadata = [read_json(path) for path in sorted((args.fusion / "tile_metadata").glob("*.json"))]
    tiles = sorted((args.fusion / "tiles").glob("*.glb"))
    meshlab_tiles = sorted((args.fusion / "tiles_meshlab").glob("*.glb"))
    monolithic = args.fusion / "full_scene_single_owner.glb"
    monolithic_meshlab = args.fusion / "full_scene_single_owner_meshlab.glb"
    preview_glb = args.fusion / "full_scene_top_surface_2m.glb"
    preview_png = args.fusion / "full_scene_top_surface_2m.png"
    generated_coverage = float(scene["generated_base_coverage_fraction"])
    final_coverage = float(scene["coverage_after_fraction"])
    projected_coverage = float(preview["projected_triangle_coverage_fraction"])
    global_metrics = reference["global"]
    seam_metrics = reference["seams"]
    checks = [
        check("scene_status", scene.get("status"), "==", "complete", scene.get("status") == "complete"),
        check("patch_count", scene.get("patch_count_available"), "==", 79, scene.get("patch_count_available") == 79),
        check("tile_metadata_count", len(tile_metadata), "==", 79, len(tile_metadata) == 79),
        check("tile_count", len(tiles), "==", 79, len(tiles) == 79),
        check("meshlab_tile_count", len(meshlab_tiles), "==", 79, len(meshlab_tiles) == 79),
        check("recorded_continuous_coverage", final_coverage, "==", 1.0, abs(final_coverage - 1.0) < 1e-12),
        check("rasterized_triangle_coverage", projected_coverage, ">=", 0.999, projected_coverage >= 0.999),
        check("all_patch_coverage", min(float(item["coverage_after_fraction"]) for item in tile_metadata), "==", 1.0, all(abs(float(item["coverage_after_fraction"]) - 1.0) < 1e-12 for item in tile_metadata)),
        check("gap_fill_faces", int(scene["base_gap_fill_cells"]) * 2, ">", 0, int(scene["base_gap_fill_cells"]) > 0),
        check("monolithic_exists", monolithic.stat().st_size if monolithic.is_file() else 0, ">", 0, monolithic.is_file() and monolithic.stat().st_size > 0),
        check("meshlab_monolithic_exists", monolithic_meshlab.stat().st_size if monolithic_meshlab.is_file() else 0, ">", 0, monolithic_meshlab.is_file() and monolithic_meshlab.stat().st_size > 0),
        check("preview_glb_exists", preview_glb.stat().st_size if preview_glb.is_file() else 0, ">", 0, preview_glb.is_file() and preview_glb.stat().st_size > 0),
        check("preview_png_exists", preview_png.stat().st_size if preview_png.is_file() else 0, ">", 0, preview_png.is_file() and preview_png.stat().st_size > 0),
        check("top_teacher_abs_error_p90_m", global_metrics["top_multilayer_teacher_abs_error_m"]["90"], "<=", 0.75, global_metrics["top_multilayer_teacher_abs_error_m"]["90"] <= 0.75),
        check("neighbor_seam_abs_delta_p90_m", seam_metrics["absolute_height_delta_m"]["90"], "<=", 1.0, seam_metrics["absolute_height_delta_m"]["90"] <= 1.0),
        check("vertical_span_over_5m_fraction", global_metrics["span_over_5m_fraction"], "<=", 0.05, global_metrics["span_over_5m_fraction"] <= 0.05),
    ]
    bad_meshlab_mimes = {}
    for path in meshlab_tiles + ([monolithic_meshlab] if monolithic_meshlab.is_file() else []):
        document = glb_document(path)
        mimes = sorted({item.get("mimeType") for item in document.get("images", [])})
        if not set(mimes).issubset({"image/jpeg", "image/png"}):
            bad_meshlab_mimes[path.name] = mimes
    checks.append(check("meshlab_unknown_image_formats", len(bad_meshlab_mimes), "==", 0, not bad_meshlab_mimes))
    partials = [str(path) for path in args.fusion.rglob("*.part")]
    checks.append(check("partial_files", len(partials), "==", 0, not partials))
    status = "passed" if all(item["pass"] for item in checks) else "failed"
    validation = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_continuous_validation.v1",
        "status": "complete",
        "generated_base_coverage_fraction": generated_coverage,
        "coverage_after_fraction": final_coverage,
        "projected_triangle_coverage_fraction": projected_coverage,
        "gap_fill_cells": int(scene["base_gap_fill_cells"]),
        "gap_fill_faces": int(scene["base_gap_fill_cells"]) * 2,
        "total_faces": int(scene["total_faces_owned"]),
        "reference_height_and_seam_validation": str(args.reference_validation.resolve()),
        "reference_global": global_metrics,
        "reference_seams": seam_metrics,
    }
    acceptance = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_continuous_acceptance.v1",
        "status": status,
        "checks": checks,
        "bad_meshlab_mime_types": bad_meshlab_mimes,
        "partial_files": partials,
        "scene_manifest": str((args.fusion / "scene_manifest.json").resolve()),
        "preview_manifest": str((args.fusion / "full_scene_top_surface_2m.json").resolve()),
    }
    (args.output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    (args.output / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(acceptance, ensure_ascii=False), flush=True)
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
