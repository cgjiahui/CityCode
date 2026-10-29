#!/usr/bin/env python3
"""Machine-check the complete P384 recommended run and preserve the verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = json.loads((args.fusion / "scene_manifest.json").read_text())
    validation = json.loads((args.validation / "validation.json").read_text())
    required = [
        "run_metadata.json", "mesh_raw.pt", "shape_slat.pt", "texture_slat.pt",
        "mesh_pbr_local.glb",
        "mesh_pbr_local_canonical_axis_v3.glb", "mesh_pbr_world_canonical_axis_v3.glb",
        "canonical_metadata.json",
    ]
    # Historical runs wrote V2; corrected positive image-row -> UE +Y runs write V3.
    # Either is archival evidence; fusion always consumes canonical axis V3.
    generated_world_alternatives = ("mesh_pbr_world_axes_v3.glb", "mesh_pbr_world_axes_v2.glb")
    patch_dirs = sorted(path.parent for path in args.runs.glob("*/run_metadata.json"))
    missing_artifacts = {
        patch.name: (
            [name for name in required if not (patch / name).is_file()] +
            ([] if any((patch / name).is_file() for name in generated_world_alternatives)
             else ["mesh_pbr_world_axes_v3.glb|mesh_pbr_world_axes_v2.glb"])
        )
        for patch in patch_dirs
    }
    missing_artifacts = {key: value for key, value in missing_artifacts.items() if value}
    global_stats = validation["global"]
    seam_stats = validation["seams"]
    input_audit = json.loads(args.input_audit.read_text()) if args.input_audit else None
    audit_patches = input_audit["patches"] if input_audit else []
    height_audit_patches = [item for item in audit_patches if "z_percentiles_m" in item]
    max_input_p99 = max((item["z_percentiles_m"][5] for item in height_audit_patches), default=None)
    max_input_above_100 = max(
        (item["above_100m_fraction"] for item in height_audit_patches), default=None
    )
    checks = [
        {"name": "generated_patch_count", "value": len(patch_dirs), "operator": "==", "threshold": 79, "pass": len(patch_dirs) == 79},
        {"name": "required_artifacts", "value": len(missing_artifacts), "operator": "==", "threshold": 0, "pass": not missing_artifacts},
        {"name": "scene_status", "value": scene["status"], "operator": "==", "threshold": "complete", "pass": scene["status"] == "complete"},
        {"name": "owned_tile_count", "value": scene["patch_count_available"], "operator": "==", "threshold": 79, "pass": scene["patch_count_available"] == 79},
        {"name": "missing_scene_patches", "value": len(scene["missing_patches"]), "operator": "==", "threshold": 0, "pass": not scene["missing_patches"]},
        {"name": "validation_status", "value": validation["status"], "operator": "==", "threshold": "complete", "pass": validation["status"] == "complete"},
        {"name": "world_coverage_fraction", "value": global_stats["coverage_fraction"], "operator": ">=", "threshold": 0.25, "pass": global_stats["coverage_fraction"] >= 0.25},
        {"name": "top_multilayer_teacher_abs_error_p90_m", "value": global_stats["top_multilayer_teacher_abs_error_m"].get("90"), "operator": "<=", "threshold": 0.75, "pass": global_stats["top_multilayer_teacher_abs_error_m"].get("90", float("inf")) <= 0.75},
        {"name": "neighbor_seam_abs_delta_p90_m", "value": seam_stats["absolute_height_delta_m"].get("90"), "operator": "<=", "threshold": 1.00, "pass": seam_stats["absolute_height_delta_m"].get("90", float("inf")) <= 1.00},
        {"name": "vertical_span_over_5m_fraction", "value": global_stats["span_over_5m_fraction"], "operator": "<=", "threshold": 0.05, "pass": global_stats["span_over_5m_fraction"] <= 0.05},
    ]
    if input_audit:
        checks.extend([
            {"name": "prepared_input_audit_patch_count", "value": len(audit_patches), "operator": "==", "threshold": 79, "pass": len(audit_patches) == 79},
        ])
        if height_audit_patches:
            checks.extend([
                {"name": "prepared_input_worst_patch_z_p99_m", "value": max_input_p99, "operator": "<=", "threshold": 25.0, "pass": max_input_p99 is not None and max_input_p99 <= 25.0},
                {"name": "prepared_input_worst_patch_above_100m_fraction", "value": max_input_above_100, "operator": "<=", "threshold": 0.0001, "pass": max_input_above_100 is not None and max_input_above_100 <= 0.0001},
            ])
        if input_audit.get("schema") == "air2vectorcity.pixal3d_p384_derived_validity_audit.v1":
            checks.extend([
                {"name": "derived_validity_audit_status", "value": input_audit.get("status"), "operator": "==", "threshold": "passed", "pass": input_audit.get("status") == "passed"},
                {"name": "derived_validity_total_violations", "value": input_audit.get("total_violations"), "operator": "==", "threshold": 0, "pass": input_audit.get("total_violations") == 0},
                {"name": "derived_validity_mean_line_recovery", "value": input_audit.get("mean_line_candidate_recovery_rate"), "operator": ">=", "threshold": 0.90, "pass": input_audit.get("mean_line_candidate_recovery_rate", 0) >= 0.90},
            ])
    verdict = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_acceptance.v1",
        "status": "passed" if all(item["pass"] for item in checks) else "failed",
        "checks": checks,
        "missing_artifacts": missing_artifacts,
        "scene_manifest": str((args.fusion / "scene_manifest.json").resolve()),
        "validation": str((args.validation / "validation.json").resolve()),
        "input_audit": str(args.input_audit.resolve()) if args.input_audit else None,
    }
    destination = args.output / "acceptance.json"
    destination.write_text(json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False), flush=True)
    if verdict["status"] != "passed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
