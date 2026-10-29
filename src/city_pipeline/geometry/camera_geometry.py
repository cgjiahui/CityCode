#!/usr/bin/env python3
"""Camera-depth lifting, world-XY rasterization and RGB display helpers."""

from __future__ import annotations
from city_pipeline.runtime import DATA_ROOT

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = DATA_ROOT
DEFAULT_DATASET_ROOT = Path("UNCONFIGURED_DEFAULT_DATASET_ROOT")
DEFAULT_OUTPUT_ROOT = Path("UNCONFIGURED_DEFAULT_OUTPUT_ROOT")
DEFAULT_VIEWS = ("tilt_light", "tilt_medium")
ROAD_IDS = np.asarray([1, 2, 3, 4, 5], dtype=np.uint8)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        result = np.asarray(image)
    if result.ndim != 2:
        raise ValueError(f"Expected one-channel uint8 image: {path}")
    return result.astype(np.uint8, copy=False)


def save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def save_u8(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _read_c_string(data: bytes, cursor: int) -> tuple[str, int]:
    end = data.find(b"\0", cursor)
    if end < 0:
        raise ValueError("Malformed OpenEXR header: missing string terminator")
    return data[cursor:end].decode("ascii"), end + 1


def _parse_chlist(payload: bytes) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(payload) and payload[cursor] != 0:
        name, cursor = _read_c_string(payload, cursor)
        if cursor + 16 > len(payload):
            raise ValueError("Truncated OpenEXR channel list")
        pixel_type, p_linear, _reserved, x_sampling, y_sampling = struct.unpack_from(
            "<iB3sii", payload, cursor
        )
        cursor += 16
        channels.append(
            {
                "name": name,
                "pixel_type": int(pixel_type),
                "p_linear": int(p_linear),
                "x_sampling": int(x_sampling),
                "y_sampling": int(y_sampling),
            }
        )
    return channels


def read_uncompressed_float_exr(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the release dataset's single-channel, uncompressed float EXR.

    This deliberately small reader rejects compression, tiles, multipart/deep
    files, subsampling, multiple channels, and non-float pixels.  It is not a
    general OpenEXR replacement; the strict checks prevent silently reading a
    future, incompatible depth format incorrectly.
    """

    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError(f"Truncated OpenEXR file: {path}")
    magic, version_flags = struct.unpack_from("<II", data, 0)
    if magic != 20000630:
        raise ValueError(f"Invalid OpenEXR magic: {path}")
    version = version_flags & 0xFF
    unsupported_flags = version_flags & ~0xFF
    if version not in (1, 2) or unsupported_flags:
        raise ValueError(
            f"Unsupported OpenEXR version/flags {version_flags:#x}: {path}"
        )

    cursor = 8
    attributes: dict[str, tuple[str, bytes]] = {}
    while True:
        name, cursor = _read_c_string(data, cursor)
        if not name:
            break
        attribute_type, cursor = _read_c_string(data, cursor)
        if cursor + 4 > len(data):
            raise ValueError(f"Truncated OpenEXR attribute: {path}")
        size = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        end = cursor + size
        if end > len(data):
            raise ValueError(f"Truncated OpenEXR attribute payload: {path}")
        attributes[name] = (attribute_type, data[cursor:end])
        cursor = end

    required = {"channels", "compression", "dataWindow", "lineOrder"}
    missing = required.difference(attributes)
    if missing:
        raise ValueError(f"OpenEXR is missing {sorted(missing)}: {path}")
    channels = _parse_chlist(attributes["channels"][1])
    expected_channel = {
        "pixel_type": 2,
        "p_linear": 0,
        "x_sampling": 1,
        "y_sampling": 1,
    }
    if (
        len(channels) != 1
        or channels[0].get("name") not in {"R", "Z"}
        or any(channels[0].get(key) != value for key, value in expected_channel.items())
    ):
        raise ValueError(
            f"Expected one uncompressed full-float R or Z channel, got {channels}: {path}"
        )
    channel_name = str(channels[0]["name"])
    compression = attributes["compression"][1]
    if compression != b"\0":
        raise ValueError(f"Expected uncompressed OpenEXR scanlines: {path}")
    if attributes["lineOrder"][1] != b"\0":
        raise ValueError(f"Expected increasing-y OpenEXR scanlines: {path}")

    min_x, min_y, max_x, max_y = struct.unpack(
        "<4i", attributes["dataWindow"][1]
    )
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid OpenEXR data window: {path}")
    table_bytes = height * 8
    if cursor + table_bytes > len(data):
        raise ValueError(f"Truncated OpenEXR scanline table: {path}")
    offsets = np.frombuffer(data, dtype="<u8", count=height, offset=cursor)

    result = np.empty((height, width), dtype=np.float32)
    seen = np.zeros(height, dtype=bool)
    expected_payload = width * 4
    for offset_value in offsets:
        offset = int(offset_value)
        if offset + 8 > len(data):
            raise ValueError(f"Invalid OpenEXR scanline offset: {path}")
        y, payload_size = struct.unpack_from("<iI", data, offset)
        row = y - min_y
        if not 0 <= row < height or seen[row]:
            raise ValueError(f"Invalid/duplicate OpenEXR scanline {y}: {path}")
        if payload_size != expected_payload or offset + 8 + payload_size > len(data):
            raise ValueError(f"Unexpected OpenEXR scanline payload: {path}")
        result[row] = np.frombuffer(
            data, dtype="<f4", count=width, offset=offset + 8
        )
        seen[row] = True
    if not np.all(seen):
        raise ValueError(f"OpenEXR has missing scanlines: {path}")

    return result, {
        "reader": "strict_single_channel_uncompressed_float32_openexr",
        "version": version,
        "channel": channel_name,
        "shape_px": [width, height],
        "data_window": [min_x, min_y, max_x, max_y],
        "finite_fraction": float(np.mean(np.isfinite(result))),
        "positive_fraction": float(np.mean(np.isfinite(result) & (result > 0.0))),
        "min_m": float(np.nanmin(result)),
        "median_m": float(np.nanmedian(result)),
        "max_m": float(np.nanmax(result)),
    }


def lift_camera_forward_depth(
    depth_z: np.ndarray,
    camera: dict[str, Any],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Lift z_camera depth into UE world XYZ at source pixel centres."""

    height, width = depth_z.shape
    if [width, height] != [int(v) for v in camera["image_size_px"]]:
        raise ValueError("Depth dimensions do not match camera.json")
    intrinsics = camera["intrinsics"]
    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    cx = float(intrinsics["cx_px"])
    cy = float(intrinsics["cy_px"])

    u = np.arange(width, dtype=np.float32) + np.float32(0.5)
    v = np.arange(height, dtype=np.float32) + np.float32(0.5)
    x_camera = depth_z * ((u - np.float32(cx)) / np.float32(fx))[None, :]
    y_camera = depth_z * ((v - np.float32(cy)) / np.float32(fy))[:, None]
    z_camera = depth_z

    transform = np.asarray(camera["camera_to_world_4x4_m"], dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    world_x = (
        rotation[0, 0] * x_camera
        + rotation[0, 1] * y_camera
        + rotation[0, 2] * z_camera
        + translation[0]
    ).astype(np.float32)
    world_y = (
        rotation[1, 0] * x_camera
        + rotation[1, 1] * y_camera
        + rotation[1, 2] * z_camera
        + translation[1]
    ).astype(np.float32)
    world_z = (
        rotation[2, 0] * x_camera
        + rotation[2, 1] * y_camera
        + rotation[2, 2] * z_camera
        + translation[2]
    ).astype(np.float32)

    # Independently verify camera/world conventions on a deterministic sample.
    step = max(1, min(width, height) // 32)
    sample_world = np.stack(
        (
            world_x[::step, ::step],
            world_y[::step, ::step],
            world_z[::step, ::step],
            np.ones_like(world_z[::step, ::step]),
        ),
        axis=-1,
    ).reshape(-1, 4)
    world_to_camera = np.asarray(camera["world_to_camera_4x4_m"], dtype=np.float64)
    recovered = sample_world @ world_to_camera.T
    sampled_depth = depth_z[::step, ::step].reshape(-1)
    finite = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    depth_error = np.abs(recovered[finite, 2] - sampled_depth[finite])
    inverse_error = np.max(
        np.abs(world_to_camera @ transform - np.eye(4, dtype=np.float64))
    )
    return (world_x, world_y, world_z), {
        "depth_convention": "camera-forward z_c in metres",
        "pixel_centres": "(u+0.5, v+0.5) in image-boundary coordinates",
        "equations": {
            "x_camera": "(u+0.5-cx)/fx*z_camera",
            "y_camera": "(v+0.5-cy)/fy*z_camera",
            "world_xyz": "camera_to_world_4x4_m @ [x_c,y_c,z_c,1]",
        },
        "camera_transform_inverse_max_abs_error": float(inverse_error),
        "sampled_camera_z_roundtrip_max_abs_error_m": (
            float(np.max(depth_error)) if depth_error.size else None
        ),
        "sampled_camera_z_roundtrip_mean_abs_error_m": (
            float(np.mean(depth_error)) if depth_error.size else None
        ),
    }


def zbuffer_world_xy(
    attributes: dict[str, np.ndarray],
    world_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds_xy_m: Iterable[float],
    output_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """Forward-splat visible samples and retain the highest world-Z surface."""

    width, height = output_size
    min_x, min_y, max_x, max_y = [float(v) for v in bounds_xy_m]
    cell_x = (max_x - min_x) / width
    cell_y = (max_y - min_y) / height
    world_x, world_y, world_z = world_xyz
    source_height, source_width = world_x.shape
    if any(value.shape[:2] != (source_height, source_width) for value in attributes.values()):
        raise ValueError("All splatted attributes must align with depth")

    finite_xy = np.isfinite(world_x) & np.isfinite(world_y)
    target_u = np.full(world_x.shape, -1, dtype=np.int32)
    target_v = np.full(world_y.shape, -1, dtype=np.int32)
    target_u[finite_xy] = np.floor(
        (world_x[finite_xy] - min_x) / cell_x
    ).astype(np.int32)
    target_v[finite_xy] = np.floor(
        (world_y[finite_xy] - min_y) / cell_y
    ).astype(np.int32)
    valid = (
        finite_xy
        & np.isfinite(world_z)
        & (target_u >= 0)
        & (target_u < width)
        & (target_v >= 0)
        & (target_v < height)
    )
    source_flat = np.flatnonzero(valid).astype(np.int32)
    target_flat = (
        target_v[valid].astype(np.int64) * width
        + target_u[valid].astype(np.int64)
    )
    z_valid = world_z[valid]
    if target_flat.size == 0:
        raise ValueError("No lifted depth samples fall inside the target patch")

    target_count = width * height
    top_z = np.full(target_count, -np.inf, dtype=np.float32)
    np.maximum.at(top_z, target_flat, z_valid)
    is_top = z_valid == top_z[target_flat]
    selected_source = np.full(target_count, -1, dtype=np.int32)
    np.maximum.at(
        selected_source,
        target_flat[is_top],
        source_flat[is_top],
    )
    observed = selected_source >= 0
    selected_target = np.flatnonzero(observed)
    selected_source_indices = selected_source[observed]

    outputs: dict[str, np.ndarray] = {}
    for name, source in attributes.items():
        source_values = source.reshape((source_height * source_width,) + source.shape[2:])
        shape = (target_count,) + source.shape[2:]
        target = np.zeros(shape, dtype=source.dtype)
        target[selected_target] = source_values[selected_source_indices]
        outputs[name] = target.reshape((height, width) + source.shape[2:])

    surface_height = np.full(target_count, np.nan, dtype=np.float32)
    surface_height[observed] = top_z[observed]
    counts = np.zeros(target_count, dtype=np.uint16)
    np.add.at(counts, target_flat, 1)

    source_y, source_x = np.indices((source_height, source_width), dtype=np.float32)
    target_u_float = (world_x - min_x) / cell_x - 0.5
    target_v_float = (world_y - min_y) / cell_y - 0.5
    displacement = np.hypot(
        source_x[valid] - target_u_float[valid],
        source_y[valid] - target_v_float[valid],
    )
    return outputs, observed.reshape(height, width), surface_height.reshape(height, width), {
        "method": "forward_world_xyz_splat_with_highest_world_z_zbuffer",
        "target_bounds_xy_world_m": [min_x, min_y, max_x, max_y],
        "output_size_px": [width, height],
        "meters_per_pixel": [cell_x, cell_y],
        "axis_mapping": {"u": "UE +X", "v": "UE +Y"},
        "source_samples_inside_target": int(target_flat.size),
        "source_samples_inside_target_fraction": float(np.mean(valid)),
        "raw_target_valid_pixels": int(np.count_nonzero(observed)),
        "raw_target_valid_fraction": float(np.mean(observed)),
        "target_cells_with_collisions": int(np.count_nonzero(counts > 1)),
        "target_collision_fraction": float(np.mean(counts > 1)),
        "max_source_samples_per_target": int(np.max(counts)),
        "source_to_true_ortho_displacement_px": {
            "median": float(np.median(displacement)),
            "p90": float(np.percentile(displacement, 90)),
            "p99": float(np.percentile(displacement, 99)),
            "max": float(np.max(displacement)),
        },
        "source_to_true_ortho_displacement_m": {
            "median": float(np.median(displacement) * cell_x),
            "p90": float(np.percentile(displacement, 90) * cell_x),
            "p99": float(np.percentile(displacement, 99) * cell_x),
            "max": float(np.max(displacement) * cell_x),
        },
        "observed_surface_height_m": {
            "min": float(np.min(surface_height[observed])),
            "median": float(np.median(surface_height[observed])),
            "p99": float(np.percentile(surface_height[observed], 99)),
            "max": float(np.max(surface_height[observed])),
        },
    }


def fill_small_rgb_holes(
    rgb: np.ndarray,
    raw_valid: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diffuse RGB only a bounded number of pixels; keep provenance masks."""

    output = rgb.copy()
    valid = raw_valid.copy()
    height, width = valid.shape
    for _ in range(iterations):
        accum = np.zeros_like(output, dtype=np.uint32)
        counts = np.zeros((height, width), dtype=np.uint8)
        for dy, dx in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            sy0, sy1 = max(0, -dy), min(height, height - dy)
            sx0, sx1 = max(0, -dx), min(width, width - dx)
            dy0, dy1 = sy0 + dy, sy1 + dy
            dx0, dx1 = sx0 + dx, sx1 + dx
            neighbor_valid = valid[sy0:sy1, sx0:sx1]
            accum[dy0:dy1, dx0:dx1] += (
                output[sy0:sy1, sx0:sx1].astype(np.uint32)
                * neighbor_valid[..., None]
            )
            counts[dy0:dy1, dx0:dx1] += neighbor_valid.astype(np.uint8)
        newly_filled = (~valid) & (counts > 0)
        if not np.any(newly_filled):
            break
        output[newly_filled] = (
            accum[newly_filled] / counts[newly_filled, None]
        ).astype(np.uint8)
        valid[newly_filled] = True
    fill_mask = valid & (~raw_valid)
    return output, fill_mask, ~valid


def binary_iou(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    left = left & mask
    right = right & mask
    union = np.count_nonzero(left | right)
    if union == 0:
        return None
    return float(np.count_nonzero(left & right) / union)




def height_preview(height_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(height_m.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return result
    low = float(np.percentile(height_m[valid], 1))
    high = float(np.percentile(height_m[valid], 99))
    scale = np.nan_to_num(
        np.clip((height_m - low) / max(high - low, 1e-6), 0.0, 1.0),
        nan=0.0,
    )
    # Lightweight blue -> cyan -> yellow -> red height palette.
    result[..., 0] = np.clip(2.0 * scale - 0.5, 0.0, 1.0) * 255
    result[..., 1] = np.clip(1.5 - np.abs(2.0 * scale - 1.0), 0.0, 1.0) * 255
    result[..., 2] = np.clip(1.5 - 2.0 * scale, 0.0, 1.0) * 255
    result[~valid] = 0
    return result


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def comparison_panel(
    title: str,
    images: list[tuple[str, str, np.ndarray]],
    destination: Path,
    tile: int = 360,
) -> None:
    header = 70
    title_height = 50
    gap = 8
    width = tile * len(images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, title_height + header + tile), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font(25), fill=(245, 245, 245))
    for index, (label, subtitle, array) in enumerate(images):
        x = index * (tile + gap)
        draw.rectangle((x, title_height, x + tile - 1, title_height + header - 1), fill=(35, 38, 44))
        draw.text((x + 9, title_height + 6), label, font=font(19), fill=(255, 255, 255))
        draw.text((x + 9, title_height + 38), subtitle, font=font(13), fill=(190, 197, 207))
        image = Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
            (tile, tile), LANCZOS
        )
        canvas.paste(image, (x, title_height + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def contact_sheet(
    samples: list[tuple[str, np.ndarray]],
    destination: Path,
    columns: int = 8,
    tile: int = 220,
) -> None:
    rows = math.ceil(len(samples) / columns)
    label_height = 34
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_height)), (15, 17, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(samples):
        row, column = divmod(index, columns)
        x, y = column * tile, row * (tile + label_height)
        image = Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
            (tile, tile), LANCZOS
        )
        canvas.paste(image, (x, y))
        draw.text((x + 5, y + tile + 6), label, font=font(13), fill=(225, 228, 233))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def aggregate(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }




