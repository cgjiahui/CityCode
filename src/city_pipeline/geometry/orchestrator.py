#!/usr/bin/env python3
"""Shared orchestration for a configured single-view E10 pipeline."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = DATA_ROOT
PYTHON = Path(sys.executable)
MASK_WRAPPER = Path("UNCONFIGURED_MASK_WRAPPER")
MASK_SCRIPT = Path("UNCONFIGURED_MASK_SCRIPT")
V92_PROMPT = Path("UNCONFIGURED_V92_PROMPT")
V7B_PROMPT = Path("UNCONFIGURED_V7B_PROMPT")
REFERENCE_ROOT = Path("UNCONFIGURED_REFERENCE_ROOT")
REFERENCE_V92 = REFERENCE_ROOT / "uecs_v9_2_gemini_3_pro_image"
REFERENCE_V7B = REFERENCE_ROOT / "uecs_v7b_gemini_pro"
VIEWS = ("tilt_light", "tilt_medium")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("p384",), required=True)
    parser.add_argument(
        "--phase",
        choices=("geometry",),
        default="geometry",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--priors-path", type=Path)
    parser.add_argument("--allow-cached-smoke-recalibration", action="store_true")
    parser.add_argument(
        "--mask-root",
        type=Path,
        help="Existing V9.2/V7B parent; defaults to run-root for legacy profiles.",
    )
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--include-view",
        action="append",
        choices=VIEWS,
        default=[],
        help=(
            "Restrict sample discovery to one or more tilt views. The default "
            "keeps both tilt_light and tilt_medium."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-ply-points", type=int, default=250_000)
    parser.add_argument("--point-preview-points", type=int, default=80_000)
    parser.add_argument(
        "--anchor-erosion-radius-px",
        type=int,
        default=3,
        help="Ground-anchor erosion radius at native RGB/depth resolution.",
    )
    parser.add_argument(
        "--bev-closing-size",
        type=int,
        default=0,
        help="Odd BEV Closing kernel; 0 chooses the closest odd size to the old 0.875 m span.",
    )
    parser.add_argument(
        "--minimum-total-instance-area-m2",
        type=float,
        default=None,
        help=(
            "E09 post-fill whole-instance threshold. Defaults to 20 m2 for P384 "
            "and 0 (disabled) for earlier profiles."
        ),
    )
    parser.add_argument("--mask-delay-seconds", type=float, default=10.0)
    parser.add_argument("--mask-resume-attempts", type=int, default=20)
    parser.add_argument("--mask-retry-cooldown-seconds", type=int, default=300)
    parser.add_argument("--v9-location", default="global")
    parser.add_argument("--v7b-location", default="us-central1")
    parser.add_argument("--v9-prompt", type=Path, default=V92_PROMPT)
    parser.add_argument("--v7b-prompt", type=Path, default=V7B_PROMPT)
    parser.add_argument(
        "--source-evaluation",
        choices=("none",),
        default="none",
        help=(
            "Reserved compatibility option; the public workflow uses none."
        ),
    )
    parser.add_argument(
        "--calibration-plane-policy",
        choices=("fixed-smallcity", "camera-ground"),
        default="fixed-smallcity",
        help=(
            "Depth scale anchor policy. camera-ground uses each real-scene "
            "camera's local ground reference instead of SmallCity absolute Z priors."
        ),
    )
    parser.add_argument(
        "--skip-release-verification",
        action="store_true",
        help="Skip the profile-specific final audit so an adapted dataset can run its own verifier.",
    )
    return parser.parse_args(argv)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_run_status(run_root: Path, status: str, **extra: Any) -> None:
    manifest_path = run_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["status"] = status
    manifest["updated_at"] = now()
    manifest.update(extra)
    write_json(manifest_path, manifest)


def mask_result_counts(mask_root: Path) -> dict[str, int]:
    return {
        name: len(list((mask_root / name / "results").glob("*.png")))
        for name in ("structure_v9_2", "structure_v7b")
    }


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def discover_samples(
    dataset_root: Path,
    requested: list[str],
    included_views: list[str] | None = None,
    require_reference_depth: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    requested_set = set(requested)
    active_views = tuple(dict.fromkeys(included_views or VIEWS))
    found: set[str] = set()
    for patch_dir in sorted(
        path for path in (dataset_root / "patches").iterdir() if path.is_dir()
    ):
        if not patch_dir.is_dir():
            continue
        for view_id in active_views:
            view_dir = patch_dir / "views" / view_id
            sample_id = f"{patch_dir.name}__{view_id}"
            if requested_set and sample_id not in requested_set:
                continue
            paths = {
                "rgb": view_dir / "rgb.png",
                "camera": view_dir / "camera.json",
            }
            for optional_name, optional_path in (
                ("semantic", view_dir / "semantic_ids.png"),
                ("building_instance", view_dir / "building_instance_ids.png"),
            ):
                if optional_path.is_file():
                    paths[optional_name] = optional_path
            strict_valid = view_dir / "strict_valid_mask.png"
            if strict_valid.is_file():
                paths["strict_valid"] = strict_valid
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{sample_id} missing {missing}")
            records.append(
                {
                    "id": sample_id,
                    "patch_id": patch_dir.name,
                    "view_id": view_id,
                    "paths": {key: relative(value) for key, value in paths.items()},
                }
            )
            found.add(sample_id)
    unknown = requested_set - found
    if unknown:
        raise KeyError(f"Unknown requested samples: {sorted(unknown)}")
    if not records:
        raise ValueError("No configured tilt samples discovered")
    return records


def stage_is_complete(manifest_path: Path, expected: int) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        data = read_json(manifest_path)
    except Exception:
        return False
    if "samples" in data:
        return len(data["samples"]) == expected
    return int(data.get("sample_count", -1)) == expected


def run_command(
    label: str,
    command: list[str],
    run_root: Path,
    dry_run: bool,
    environment: dict[str, str] | None = None,
) -> None:
    printable = " ".join(command)
    print(f"\n===== {label} =====\n{printable}", flush=True)
    if dry_run:
        return
    log_path = run_root / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] {printable}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def prepare_inputs(
    dataset_root: Path,
    depth_root: Path,
    run_root: Path,
    samples: list[dict[str, Any]],
    profile: str,
    mask_root: Path,
    v9_prompt: Path,
    v7b_prompt: Path,
    anchor_erosion_radius_px: int,
    dry_run: bool,
) -> None:
    flat_root = run_root / "inputs_flat"
    if not dry_run:
        flat_root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        destination = flat_root / f"{sample['id']}.png"
        source = PROJECT_ROOT / sample["paths"]["rgb"]
        if destination.exists():
            continue
        print(f"LINK {destination.name} -> {source}", flush=True)
        if not dry_run:
            destination.symlink_to(source)
    first_sample = samples[0]
    first_patch = read_json(
        dataset_root / "patches" / first_sample["patch_id"] / "patch.json"
    )
    first_camera = read_json(
        dataset_root
        / "patches"
        / first_sample["patch_id"]
        / "views"
        / first_sample["view_id"]
        / "camera.json"
    )
    model_selection_path = depth_root / "model_selection.json"
    model_selection = (
        read_json(model_selection_path) if model_selection_path.is_file() else None
    )
    patch_size_m = [float(value) for value in first_patch["size_m"]]
    patch_stride_m = [float(value) for value in first_patch["stride_m"]]
    patch_overlap_m = first_patch.get("overlap_m")
    if patch_overlap_m is None:
        patch_overlap_m = [
            size - stride for size, stride in zip(patch_size_m, patch_stride_m)
        ]
    ortho_meters_per_pixel = float(
        first_patch.get(
            "ortho_meters_per_pixel",
            patch_size_m[0] / float(first_camera["image_size_px"][0]),
        )
    )
    core_bounds = first_patch.get("core_bounds_xy_world_m")
    core_size_m = (
        [
            float(core_bounds[2]) - float(core_bounds[0]),
            float(core_bounds[3]) - float(core_bounds[1]),
        ]
        if core_bounds is not None
        else None
    )
    active_views = sorted({sample["view_id"] for sample in samples})
    manifest = {
        "schema": "air2vectorcity.configured_full_e10_run.v1",
        "created_at": now(),
        "status": "prepared",
        "pipeline_profile": profile,
        "mode": "predicted V9.2 building Instances + predicted V7B background",
        "dataset_root": relative(dataset_root),
        "depth_root": relative(depth_root),
        "sample_count": len(samples),
        "views": active_views,
        "patch_contract": {
            "size_m": patch_size_m,
            "stride_m": patch_stride_m,
            "overlap_m": patch_overlap_m,
            "rgb_resolution_px": first_camera["image_size_px"],
            "ortho_meters_per_pixel": ortho_meters_per_pixel,
            "core_size_m": core_size_m,
            "halo_m_per_side": first_patch.get("halo_m_per_side"),
        },
        "mask_contract": {
            "model": "gemini-3-pro-image",
            "v9_2_prompt": relative(v9_prompt),
            "v7b_prompt": relative(v7b_prompt),
            "output_resolution_px": [1024, 1024],
            "root": relative(mask_root),
        },
        "depth_selection": model_selection,
        "planned_geometry_policy": {
            "source_filter": "frozen 120/2/5 at unchanged 1024 mask resolution",
            "physical_area_thresholds_m2": {
                "minimum_component": 12.0,
                "minimum_total_instance": 20.0 if profile == "p384" else 0.0,
                "maximum_hole": 48.0,
            },
            "closing_kernel": (
                "automatic odd kernel nearest the P256 7-cell / 0.875 m span"
            ),
            "ground_anchor_erosion_radius_native_px": anchor_erosion_radius_px,
            "e10_height_policy": "frozen metric thresholds; cell size read from E09 rasterization",
            "strict_roi": (
                "required mask applied before scale calibration, lifting, recovery and fusion"
                if first_patch.get("strict_roi", {}).get("required", False)
                else "not required by dataset metadata"
            ),
            "reliable_core_bounds": first_patch.get("core_bounds_xy_world_m"),
        },
        "samples": samples,
        "stages": {
            "v9_2_masks": relative(mask_root / "structure_v9_2" / "manifest.json"),
            "v7b_masks": relative(mask_root / "structure_v7b" / "manifest.json"),
            "e00": relative(run_root / "e00_instance_aware_25d" / "manifest.json"),
            "e01": relative(run_root / "e01_denoising" / "manifest.json"),
            "source_filter": relative(run_root / "source_filter_120_2_5" / "manifest.json"),
            "e05": relative(run_root / "e05_bev_inputs" / "manifest.json"),
            "e09": relative(run_root / "e09_full_pipeline" / "manifest.json"),
            "e10": relative(run_root / "e10_height_guided" / "manifest.json"),
        },
    }
    if not dry_run:
        write_json(run_root / "run_manifest.json", manifest)
        profile_label = profile.upper()
        dataset_label = str(first_patch.get("scene_id", "configured dataset"))
        readme = f"# {dataset_label} {profile_label} E10 Predicted Result\n\n"
        readme += f"状态：**已准备 {len(samples)} 个单视图样本；待生成 E10 结果**。\n\n"
        readme += "## 正式结果范围\n\n"
        readme += "本次只发布 **E10 最新方案**，正式结果根目录为 `e10_height_guided/`。\n\n"
        readme += (
            "`e00_instance_aware_25d/`、`e01_denoising/`、`e05_bev_inputs/` 和 "
            "`e09_full_pipeline/` 只是 E10 的计算依赖/缓存，不作为本次实验结果发布。\n\n"
        )
        readme += "## 输入契约\n\n"
        readme += f"- Patch：`{first_patch['size_m'][0]} m x {first_patch['size_m'][1]} m`\n"
        readme += f"- 步长/重叠：`{patch_stride_m[0]} m / {patch_overlap_m[0]} m`\n"
        readme += f"- RGB：`{first_camera['image_size_px'][0]} x {first_camera['image_size_px'][1]}`\n"
        readme += f"- BEV：`{ortho_meters_per_pixel} m/px`\n"
        readme += "- 视角：" + "、".join(f"`{value}`" for value in active_views) + "；不使用 `perspective_nadir`\n\n"
        readme += "## 固定与换算\n\n"
        readme += "- Gemini：原 V9.2/V7B Prompt，`gemini-3-pro-image`，输出 1024 方形 Mask。\n"
        readme += "- 源图过滤：在固定 1024 Mask 空间继续使用 `120/2/5`。\n"
        readme += "- 米制阈值：12/48 m2 及 E10 高度/半径参数保持不变，按实际 cell 面积换算。\n"
        readme += "- Closing：按实际 BEV cell size 自动选择最接近 0.875 m 的奇数核。\n"
        readme += f"- 深度输入：`{relative(depth_root)}`。\n"
        width, height = [int(value) for value in first_camera["image_size_px"]]
        readme += (
            f"- 深度接口：`{width} x {height} float32`，camera-forward `z_c`，单位米。\n"
        )
        if first_patch.get("strict_roi", {}).get("required", False):
            readme += "- 严格 ROI：所有 depth/mask/lifting 输出应用 `strict_valid_mask.png`。\n"
        if first_patch.get("core_bounds_xy_world_m"):
            readme += "- 可靠范围：384 m 数据域中的中心 320 m core 用于全局 canonical/ownership。\n"
        readme += "\n"
        readme += "## 入口\n\n"
        readme += "- `e10_height_guided/manifest.json`：E10 核心清单。\n"
        readme += "- `e10_height_guided/modeling_instance_ids/`：最终建模 Instance ID。\n"
        readme += "- `e10_height_guided/modeling_building_height_m/`：建筑高度栅格。\n"
        readme += "- `e10_height_guided/building_point_cloud_npz/`：E10 2.5D 点云。\n"
        readme += "- `e10_height_guided/building_prisms/`：建筑棱柱 GLB / GeoJSON。\n"
        readme += "- `e10_height_guided/boundary_instance_tagging/`：边界风险标记。\n"
        readme += "- `e10_height_guided/footprint_height_representation/`：footprint + base/top height。\n"
        readme += "- `E10_RESULT_AUDIT.json`：E10-only 最终审计。\n"
        (run_root / "README.md").write_text(readme, encoding="utf-8")








def execute_stage(
    args: argparse.Namespace,
    run_root: Path,
    expected: int,
    label: str,
    manifest: Path,
    command: list[str],
) -> None:
    if not args.force and stage_is_complete(manifest, expected):
        print(f"SKIP {label}: complete manifest with {expected} samples", flush=True)
        return
    run_command(label, command, run_root, args.dry_run)
    if not args.dry_run and not stage_is_complete(manifest, expected):
        raise RuntimeError(f"{label} did not produce {expected} samples: {manifest}")


def run_geometry(
    args: argparse.Namespace,
    run_root: Path,
    mask_root: Path,
    samples: list[dict[str, Any]],
) -> None:
    expected = len(samples)
    flat = run_root / "inputs_flat"
    e00 = run_root / "e00_instance_aware_25d"
    e01 = run_root / "e01_denoising"
    source_filter = run_root / "source_filter_120_2_5"
    e05 = run_root / "e05_bev_inputs"
    e09 = run_root / "e09_full_pipeline"
    e10 = run_root / "e10_height_guided"
    python = str(PYTHON)
    camera = read_json(
        args.dataset_root
        / "patches"
        / samples[0]["patch_id"]
        / "views"
        / samples[0]["view_id"]
        / "camera.json"
    )
    bounds = [float(value) for value in camera["target_patch_bounds_xy_m"]]
    cell_size_m = (bounds[2] - bounds[0]) / int(camera["image_size_px"][0])
    closing_size = args.bev_closing_size
    if closing_size <= 0:
        target_cells = 0.875 / cell_size_m
        closing_size = max(1, int(2 * round((target_cells - 1) / 2) + 1))
    if closing_size % 2 != 1:
        raise ValueError("--bev-closing-size must be odd")
    anchor_erosion_radius_px = int(args.anchor_erosion_radius_px)
    if anchor_erosion_radius_px < 0:
        raise ValueError("--anchor-erosion-radius-px cannot be negative")
    minimum_total_instance_area_m2 = args.minimum_total_instance_area_m2
    if minimum_total_instance_area_m2 is None:
        minimum_total_instance_area_m2 = 20.0 if args.profile == "p384" else 0.0
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_e00",
        e00 / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.depth_lifting",
            "--test-root",
            str(flat),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--depth-run",
            str(args.depth_root.resolve()),
            "--v9-2-run",
            str(mask_root / "structure_v9_2"),
            "--v7b-run",
            str(mask_root / "structure_v7b"),
            "--output-root",
            str(e00),
            "--max-ply-points",
            str(args.max_ply_points),
            "--anchor-erosion-radius-px",
            str(anchor_erosion_radius_px),
            "--calibration-plane-policy",
            args.calibration_plane_policy,
            *(["--priors-path", str(args.priors_path)] if args.priors_path else []),
            *(["--allow-cached-smoke-recalibration"] if args.allow_cached_smoke_recalibration else []),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_e01",
        e01 / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.depth_denoising",
            "--baseline-root",
            str(e00),
            "--output-root",
            str(e01),
            "--max-ply-points",
            str(args.max_ply_points),
            "--anchor-erosion-radius-px",
            str(anchor_erosion_radius_px),
            "--calibration-plane-policy",
            args.calibration_plane_policy,
            *(["--priors-path", str(args.priors_path)] if args.priors_path else []),
            *(["--allow-cached-smoke-recalibration"] if args.allow_cached_smoke_recalibration else []),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_source_filter",
        source_filter / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.source_filter",
            "--baseline-root",
            str(e00),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--output-root",
            str(source_filter),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_e05",
        e05 / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.bev_inputs",
            "--baseline-root",
            str(e00),
            "--e01-root",
            str(e01),
            "--source-parameter-root",
            str(source_filter),
            "--output-root",
            str(e05),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_e09",
        e09 / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.surface_recovery",
            "--baseline-root",
            str(e00),
            "--e01-root",
            str(e01),
            "--e05-root",
            str(e05),
            "--skip-e08-reference",
            "--output-root",
            str(e09),
            "--max-ply-points",
            str(args.max_ply_points),
            "--closing-size",
            str(closing_size),
            "--minimum-total-instance-area-m2",
            str(minimum_total_instance_area_m2),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_e10",
        e10 / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.height_cleaning",
            "--e09-root",
            str(e09),
            "--output-root",
            str(e10),
            "--max-ply-points",
            str(args.max_ply_points),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_prisms",
        e10 / "building_prisms" / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.prisms",
            "--e04-root",
            str(e10),
            "--run-label",
            "E10 Full Height-Guided",
            "--schema",
            "air2vectorcity.coarse_building_prisms_e10.configured_full.v1",
            "--input-manifest-key",
            "input_e10_configured_full",
            "--use-precomputed-height-layers",
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_point_states",
        e10 / "point_cloud_state_comparison" / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.point_states",
            "--e09-root",
            str(e09),
            "--e10-root",
            str(e10),
            "--output-root",
            str(e10 / "point_cloud_state_comparison"),
            "--max-ply-points",
            str(args.max_ply_points),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_boundary_tags",
        e10 / "boundary_instance_tagging" / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.boundary_tags",
            "--e09-root",
            str(e09),
            "--e10-root",
            str(e10),
            "--output-root",
            str(e10 / "boundary_instance_tagging"),
            "--max-ply-points",
            str(args.max_ply_points),
        ],
    )
    execute_stage(
        args,
        run_root,
        expected,
        "geometry_footprint_height",
        e10 / "footprint_height_representation" / "manifest.json",
        [
            python,
            "-m", "city_pipeline.geometry.footprint_height",
            "--e10-root",
            str(e10),
            "--output-root",
            str(e10 / "footprint_height_representation"),
        ],
    )




def verify_masks(mask_root: Path, samples) -> None:
    wanted = {sample['id'] for sample in samples}
    for name in ('structure_v9_2', 'structure_v7b'):
        manifest_path = mask_root / name / 'manifest.json'
        manifest = read_json(manifest_path)
        records = {item['id']: item for item in manifest['samples']}
        if len(records) != len(manifest['samples']):
            raise ValueError('Duplicate segmentation sample IDs')
        missing = [sid for sid in sorted(wanted) if sid not in records or not (mask_root / name / records[sid]['result']).is_file()]
        if missing:
            raise RuntimeError(f'{name} missing selected results: {missing[:5]}')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dataset_root = args.dataset_root.resolve()
    args.depth_root = args.depth_root.resolve()
    if args.calibration_plane_policy == "fixed-smallcity":
        args.priors_path = (args.priors_path or args.dataset_root / "priors.json").resolve()
        from .priors import read_plane_priors
        read_plane_priors(args.priors_path)
    run_root = args.run_root.resolve()
    mask_root = (args.mask_root or run_root).resolve()
    samples = discover_samples(args.dataset_root, args.sample_id, args.include_view, require_reference_depth=args.source_evaluation != "none")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    prepare_inputs(
        args.dataset_root,
        args.depth_root,
        run_root,
        samples,
        args.profile,
        mask_root,
        args.v9_prompt.resolve(),
        args.v7b_prompt.resolve(),
        args.anchor_erosion_radius_px,
        args.dry_run,
    )
    if args.phase in ("geometry", "visualization", "all") and not args.dry_run:
        verify_masks(mask_root, samples)
    offline_environment = os.environ.copy()
    offline_environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "MPLCONFIGDIR": str(run_root / ".matplotlib"),
        }
    )
    os.environ.update(offline_environment)
    if args.phase in ("geometry", "all"):
        run_geometry(args, run_root, mask_root, samples)
        if not args.dry_run:
            update_run_status(run_root, "geometry_complete")
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
