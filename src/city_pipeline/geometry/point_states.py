#!/usr/bin/env python3
"""Export a viewer-oriented E09/E10 point-cloud state comparison package."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = DATA_ROOT


from city_pipeline.geometry.camera_geometry import load_rgb  # noqa: E402
from city_pipeline.geometry.camera_bev import (  # noqa: E402
    building_point_cloud,
)
from city_pipeline.geometry.depth_lifting import (  # noqa: E402
    instance_palette,
    write_binary_ply,
)


EXPERIMENT_ROOT = Path("UNCONFIGURED_EXPERIMENT_ROOT")
DEFAULT_E09_ROOT = EXPERIMENT_ROOT / "camera_aligned_25d_e09_full_pipeline_p5_close7"
DEFAULT_E10_ROOT = EXPERIMENT_ROOT / "camera_aligned_25d_e10_height_guided_instance_cleaning"
DEFAULT_OUTPUT_ROOT = DEFAULT_E10_ROOT / "point_cloud_state_comparison"

KEPT_COLOR = np.asarray([40, 205, 135], dtype=np.uint8)
REMOVED_COLOR = np.asarray([244, 78, 78], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e09-root", type=Path, default=DEFAULT_E09_ROOT)
    parser.add_argument("--e10-root", type=Path, default=DEFAULT_E10_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_u16(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def load_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint8)


def point_layer(
    layer_id: str,
    label: str,
    role: str,
    point_count: int,
    dense_npz: Path,
    preview_ply: Path,
    color_mode: str,
    description: str,
) -> dict[str, Any]:
    return {
        "id": layer_id,
        "label": label,
        "role": role,
        "point_count": int(point_count),
        "dense_npz": relative(dense_npz),
        "preview_ply": relative(preview_ply),
        "preview_color_mode": color_mode,
        "description": description,
    }


def main() -> int:
    args = parse_args()
    e09_root = args.e09_root.resolve()
    e10_root = args.e10_root.resolve()
    output_root = args.output_root.resolve()
    e09 = read_json(e09_root / "manifest.json")
    e10 = read_json(e10_root / "manifest.json")
    e09_by_id = {sample["id"]: sample for sample in e09["samples"]}
    e10_by_id = {sample["id"]: sample for sample in e10["samples"]}
    sample_ids = args.sample_id or [sample["id"] for sample in e10["samples"]]

    directories = {
        "removed_npz": output_root / "removed_points_npz",
        "removed_rgb": output_root / "removed_points_ply_rgb",
        "removed_instance": output_root / "removed_points_ply_instance",
        "removed_state": output_root / "removed_points_ply_state_red",
        "state": output_root / "initial_state_ply_kept_green_removed_red",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    totals = {"initial": 0, "kept": 0, "removed": 0}
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] {sample_id}", flush=True)
        original = e09_by_id[sample_id]
        cleaned = e10_by_id[sample_id]

        original_ids = load_u16(project_path(original["outputs"]["final_ids"]))
        cleaned_ids = load_u16(project_path(cleaned["outputs"]["final_ids"]))
        surface_z = np.load(project_path(original["outputs"]["surface_z"])).astype(np.float32)
        bev_rgb = load_rgb(project_path(original["outputs"]["bev_rgb"]))
        provenance = load_u8(project_path(original["outputs"]["provenance"]))
        transform = original["rasterization"]["pixel_center_index_to_ue_world_xy_3x3"]

        if original_ids.shape != cleaned_ids.shape:
            raise AssertionError(f"ID raster shape mismatch: {sample_id}")
        if np.any((cleaned_ids > 0) & (original_ids != cleaned_ids)):
            raise AssertionError(f"E10 is not an ID-preserving subset of E09: {sample_id}")
        if np.any((cleaned_ids > 0) & ~np.isfinite(surface_z)):
            raise AssertionError(f"E10 contains points without finite height: {sample_id}")

        removed_ids = np.where(
            (original_ids > 0) & (cleaned_ids == 0), original_ids, 0
        ).astype(np.uint16)
        removed_xyz, removed_rgb, removed_instance, removed_provenance = building_point_cloud(
            surface_z,
            bev_rgb,
            removed_ids,
            provenance,
            transform,
        )

        original_npz = project_path(original["outputs"]["building_point_cloud_npz"])
        cleaned_npz = project_path(cleaned["outputs"]["building_point_cloud_npz"])
        original_ply = project_path(original["outputs"]["building_point_cloud_ply_instance"])
        cleaned_ply = project_path(cleaned["outputs"]["building_point_cloud_ply_instance"])
        original_points = int(original["building_point_count"])
        kept_points = int(cleaned["building_point_count"])
        removed_points = int(removed_xyz.shape[0])
        if original_points != kept_points + removed_points:
            raise AssertionError(
                f"Point conservation failed for {sample_id}: "
                f"{original_points} != {kept_points} + {removed_points}"
            )

        removed_npz = directories["removed_npz"] / f"{sample_id}__removed_e09_to_e10_points.npz"
        removed_rgb_ply = directories["removed_rgb"] / f"{sample_id}__removed_points_rgb.ply"
        removed_instance_ply = directories["removed_instance"] / f"{sample_id}__removed_points_instance.ply"
        removed_state_ply = directories["removed_state"] / f"{sample_id}__removed_points_red.ply"
        state_ply = directories["state"] / f"{sample_id}__e09_points_kept_green_removed_red.ply"

        np.savez_compressed(
            removed_npz,
            xyz_m=removed_xyz,
            rgb=removed_rgb,
            instance_id=removed_instance,
            provenance=removed_provenance,
            point_state=np.full(removed_points, 2, dtype=np.uint8),
        )
        removed_rgb_meta = write_binary_ply(
            removed_rgb_ply,
            removed_xyz,
            removed_rgb,
            removed_instance,
            args.max_ply_points,
        )
        removed_instance_meta = write_binary_ply(
            removed_instance_ply,
            removed_xyz,
            instance_palette(int(np.max(original_ids)))[removed_instance],
            removed_instance,
            args.max_ply_points,
        )
        removed_state_meta = write_binary_ply(
            removed_state_ply,
            removed_xyz,
            np.broadcast_to(REMOVED_COLOR, (removed_points, 3)),
            removed_instance,
            args.max_ply_points,
        )

        original_dense = np.load(original_npz)
        original_xyz = original_dense["xyz_m"]
        original_instance = original_dense["instance_id"]
        original_count = int(original_xyz.shape[0])
        if original_count != original_points:
            raise AssertionError(f"E09 NPZ count mismatch: {sample_id}")
        kept_raster = cleaned_ids > 0
        original_selected = (original_ids > 0) & np.isfinite(surface_z)
        point_kept = kept_raster[original_selected]
        if int(np.count_nonzero(point_kept)) != kept_points:
            raise AssertionError(f"State mapping mismatch: {sample_id}")
        state_rgb = np.empty((original_count, 3), dtype=np.uint8)
        state_rgb[point_kept] = KEPT_COLOR
        state_rgb[~point_kept] = REMOVED_COLOR
        state_meta = write_binary_ply(
            state_ply,
            original_xyz,
            state_rgb,
            original_instance,
            args.max_ply_points,
        )

        totals["initial"] += original_points
        totals["kept"] += kept_points
        totals["removed"] += removed_points
        records.append(
            {
                "id": sample_id,
                "coordinate_system": "UE world XYZ metres; identical coordinates across all layers",
                "point_counts": {
                    "initial_e09": original_points,
                    "kept_e10": kept_points,
                    "removed_e09_to_e10": removed_points,
                    "removed_fraction": removed_points / original_points,
                    "conservation_check": original_points == kept_points + removed_points,
                },
                "layers": [
                    point_layer(
                        "initial_e09",
                        "E09 初始 Instance 点",
                        "initial",
                        original_points,
                        original_npz,
                        original_ply,
                        "instance_id",
                        "E09 修复 12/48 m² 与 Closing 7×7 后、进入高度清理前的全部建筑点。",
                    ),
                    point_layer(
                        "kept_e10",
                        "E10 高度清理后保留点",
                        "kept",
                        kept_points,
                        cleaned_npz,
                        cleaned_ply,
                        "instance_id",
                        "经过稳定高度层、空间连通与内部半径判断后保留的建模点。",
                    ),
                    point_layer(
                        "removed_e09_to_e10",
                        "E10 删除的高度噪声/细长尾巴点",
                        "removed",
                        removed_points,
                        removed_npz,
                        removed_instance_ply,
                        "instance_id",
                        "严格的 E09 点集减 E10 点集；所有点保留原始 XYZ、RGB、ID 和 provenance。",
                    ),
                ],
                "removed_point_previews": {
                    "original_rgb": relative(removed_rgb_ply),
                    "instance_color": relative(removed_instance_ply),
                    "state_red": relative(removed_state_ply),
                    "ply_metadata": {
                        "rgb": removed_rgb_meta,
                        "instance": removed_instance_meta,
                        "state": removed_state_meta,
                    },
                },
                "state_comparison": {
                    "preview_ply": relative(state_ply),
                    "color_semantics": {
                        "kept_e10": KEPT_COLOR.tolist(),
                        "removed_e09_to_e10": REMOVED_COLOR.tolist(),
                    },
                    "ply_metadata": state_meta,
                },
                "raster_evidence": {
                    "initial_ids_e09": original["outputs"]["final_ids"],
                    "cleaned_ids_e10": cleaned["outputs"]["final_ids"],
                    "removed_mask_e10": cleaned["outputs"]["tail_removed"],
                    "surface_z_m": original["outputs"]["surface_z"],
                    "provenance": original["outputs"]["provenance"],
                },
            }
        )

    if totals["initial"] != totals["kept"] + totals["removed"]:
        raise AssertionError("Dataset-level point conservation failed")
    manifest = {
        "schema": "air2vectorcity.e09_e10_point_cloud_state_comparison.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete",
        "purpose": "Structured selectable point-cloud layers for comparing E09 initial instance points with E10 height-guided modeling points.",
        "inputs": {
            "initial_e09": relative(e09_root),
            "cleaned_e10": relative(e10_root),
        },
        "layer_contract": {
            "initial_e09": "All E09 building points before height-guided tail removal.",
            "kept_e10": "The ID-preserving subset retained for E10 modeling.",
            "removed_e09_to_e10": "Exact set difference initial_e09 minus kept_e10.",
            "state_comparison": "One preview cloud of all E09 points: green kept, red removed.",
            "dense_npz_is_complete": True,
            "preview_ply_max_points": int(args.max_ply_points),
            "preview_ply_record_bytes": 17,
        },
        "point_state_codes_in_removed_npz": {
            "2": "removed_e09_to_e10"
        },
        "totals": {
            "initial_e09": totals["initial"],
            "kept_e10": totals["kept"],
            "removed_e09_to_e10": totals["removed"],
            "removed_fraction": totals["removed"] / totals["initial"],
            "conservation_check": totals["initial"] == totals["kept"] + totals["removed"],
        },
        "samples": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
