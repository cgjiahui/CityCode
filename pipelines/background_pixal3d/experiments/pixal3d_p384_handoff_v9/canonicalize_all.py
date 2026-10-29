#!/usr/bin/env python3
"""Resume-safe batch wrapper for canonical PBR postprocessing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--residual", type=float, default=0.01)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    script = Path(__file__).with_name("canonicalize_pbr.py")
    records = []
    for index, metadata_path in enumerate(sorted(args.runs.glob("*/run_metadata.json")), 1):
        run = metadata_path.parent
        destination = run / "mesh_pbr_world_canonical_axis_v3.glb"
        canonical_metadata = run / "canonical_metadata.json"
        if destination.is_file() and canonical_metadata.is_file():
            record = json.loads(canonical_metadata.read_text())
            if (
                record.get("schema") == "air2vectorcity.pixal3d_p384_handoff_v9_canonical_mesh.v3"
            ):
                records.append({"patch_id": run.name, "status": "complete", "cached": True, **record})
                print(json.dumps({"index": index, "patch_id": run.name, "status": "cached"}), flush=True)
                continue
        command = [
            sys.executable, str(script), "--run", str(run),
            "--inputs", str(args.inputs), "--teacher", str(args.teacher),
            "--residual", str(args.residual),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            failure = {"patch_id": run.name, "status": "failed", "stderr": result.stderr[-4000:]}
            records.append(failure)
            print(json.dumps({"index": index, **failure}), flush=True)
            if not args.continue_on_error:
                raise RuntimeError(f"Canonicalization failed for {run.name}: {result.stderr}")
            continue
        record = json.loads(canonical_metadata.read_text())
        records.append({"patch_id": run.name, "status": "complete", "cached": False, **record})
        print(json.dumps({"index": index, "patch_id": run.name, "status": "complete"}), flush=True)
    manifest = {
        "schema": "air2vectorcity.pixal3d_p384_handoff_v9_canonical_batch.v1",
        "axis_mapping_version": "pbr_local_z_to_ue_y_positive_v3",
        "run_count": len(records),
        "complete_count": sum(item["status"] == "complete" for item in records),
        "failed_count": sum(item["status"] == "failed" for item in records),
        "records": records,
    }
    (args.runs / "canonical_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("run_count", "complete_count", "failed_count")}), flush=True)


if __name__ == "__main__":
    main()
