"""Four-reference building appearance completion and the frozen LoD1 v4 renderer.

The renderer and height prompt retain the historical standard's numerical recipe.
Provider credentials are supplied only by the caller's environment. Source imagery
is evidence, never a substitute for the completed conditioning image.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

INPUT_NAMES = ("lod1_main_oblique.png", "source_primary.png", "roof_evidence.png", "context.png")
MODES = (("perspective_light_ao_rough",),)
RENDER_PROFILE = {"name": "lod1_v4_sculpted_rough", "elevation_deg": 34.0,
                  "roughness_scale": 1.65, "perspective_distance_scale": 2.80,
                  "canvas_size": 1024, "vertical_profile": "sculpted"}

def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values()]
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(loaded).__name__}")
    return loaded

def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm

def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)

def normalized_blur(values: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    weighted = Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255.0))
    weights = Image.fromarray(np.uint8(mask) * 255)
    weighted = weighted.filter(ImageFilter.GaussianBlur(radius=radius))
    weights = weights.filter(ImageFilter.GaussianBlur(radius=radius))
    numerator = np.asarray(weighted, dtype=np.float32) / 255.0
    denominator = np.asarray(weights, dtype=np.float32) / 255.0
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-4),
        out=np.zeros_like(numerator),
        where=denominator > 1e-4,
    )

def texture_field(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    resampling = getattr(Image, "Resampling", Image)
    low = rng.normal(127.5, 36.0, size=(64, 64)).clip(0, 255).astype(np.uint8)
    medium = rng.normal(127.5, 30.0, size=(256, 256)).clip(0, 255).astype(np.uint8)
    low_image = Image.fromarray(low).resize((size, size), resampling.BICUBIC)
    medium_image = Image.fromarray(medium).resize((size, size), resampling.BILINEAR)
    low_array = (np.asarray(low_image, dtype=np.float32) - 127.5) / 127.5
    medium_array = (np.asarray(medium_image, dtype=np.float32) - 127.5) / 127.5
    return 0.72 * low_array + 0.28 * medium_array

def mask_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    source = Image.fromarray(np.uint8(mask) * 255)
    kernel = max(3, width * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    dilated = np.asarray(source.filter(ImageFilter.MaxFilter(kernel)), dtype=np.int16)
    eroded = np.asarray(source.filter(ImageFilter.MinFilter(kernel)), dtype=np.int16)
    return (dilated - eroded) > 0

def render_spatial_proxy(
    mesh_path: Path,
    destination: Path,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    mode: str,
    seed: int,
    vertical_profile: str = "none",
    roughness_scale: float = 1.0,
    perspective_distance_scale: float = 3.35,
    canvas_size: int = 1024,
) -> None:
    valid_modes = {item[0] for item in MODES}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported spatial-cue mode: {mode}")
    if vertical_profile not in {
        "none",
        "soft",
        "medium",
        "soft_band",
        "strong",
        "sculpted",
    }:
        raise ValueError(f"Unsupported vertical shading profile: {vertical_profile}")
    if roughness_scale < 0.0:
        raise ValueError("roughness_scale must be non-negative")
    if perspective_distance_scale <= 1.25:
        raise ValueError("perspective_distance_scale must be greater than 1.25")

    use_lighting = mode != "flat_ortho"
    use_perspective = mode in {
        "perspective_light_ao",
        "perspective_light_ao_rough",
    }
    use_ao = use_perspective
    use_roughness = mode == "perspective_light_ao_rough"

    mesh = load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    centered = vertices - center

    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    camera_direction = normalized(
        np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float64,
        )
    )
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    screen_right = normalized(np.cross(world_up, camera_direction))
    screen_up = normalized(np.cross(camera_direction, screen_right))
    camera_x = centered @ screen_right
    camera_y = centered @ screen_up
    toward_camera = centered @ camera_direction

    if use_perspective:
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        camera_distance = max(perspective_distance_scale * diagonal, 1e-3)
        distance_to_camera = np.maximum(camera_distance - toward_camera, 1e-3)
        plane = np.column_stack(
            (camera_x / distance_to_camera, camera_y / distance_to_camera)
        )
    else:
        camera_distance = None
        plane = np.column_stack((camera_x, camera_y))

    projected = np.column_stack((plane, toward_camera))
    visible = (normals @ camera_direction) > 1e-6
    visible_indices = np.flatnonzero(visible)
    visible_indices = sorted(
        visible_indices,
        key=lambda index: float(toward_camera[faces[index]].mean()),
    )
    visible_vertices = projected[faces[visible_indices], :2].reshape(-1, 2)
    minimum = visible_vertices.min(axis=0)
    maximum = visible_vertices.max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-9)
    supersample = 2
    size = canvas_size * supersample
    margin = 0.12 * size
    scale = min((size - 2 * margin) / extent[0], (size - 2 * margin) / extent[1])
    midpoint = (minimum + maximum) * 0.5

    def pixel(point: np.ndarray) -> tuple[float, float]:
        x = (point[0] - midpoint[0]) * scale + size * 0.5
        y = size * 0.5 - (point[1] - midpoint[1]) * scale
        return float(x), float(y)

    light_direction = normalized(
        0.42 * camera_direction - 0.58 * screen_right + 0.92 * world_up
    )
    face_toward = np.array(
        [toward_camera[faces[index]].mean() for index in visible_indices],
        dtype=np.float64,
    )
    depth_min = float(face_toward.min())
    depth_span = max(float(face_toward.max() - depth_min), 1e-9)

    kind_image = Image.new("L", (size, size), 0)
    light_image = Image.new("L", (size, size), 0)
    depth_image = Image.new("L", (size, size), 0)
    kind_draw = ImageDraw.Draw(kind_image)
    light_draw = ImageDraw.Draw(light_image)
    depth_draw = ImageDraw.Draw(depth_image)
    for order, face_index in enumerate(visible_indices):
        polygon = [pixel(projected[index, :2]) for index in faces[face_index]]
        is_roof = normals[face_index, 2] > 0.65
        kind_draw.polygon(polygon, fill=2 if is_roof else 1)
        diffuse = max(0.0, float(normals[face_index] @ light_direction))
        # A low ambient term makes the directional response easy to see while
        # keeping every face readable on a white-background conditioning image.
        irradiance = 0.42 + 0.58 * diffuse
        light_draw.polygon(polygon, fill=int(round(255 * irradiance)))
        depth = (face_toward[order] - depth_min) / depth_span
        depth_draw.polygon(polygon, fill=int(round(24 + 231 * depth)))

    kind = np.asarray(kind_image, dtype=np.uint8)
    foreground = kind > 0
    facade = kind == 1
    roof = kind == 2
    raw_light = np.asarray(light_image, dtype=np.float32) / 255.0
    raw_depth = np.asarray(depth_image, dtype=np.float32) / 255.0

    # Blur the lighting field within each surface class. Large facade planes
    # retain a clear light/dark response, while tiny footprint facets merge
    # into a continuous tone instead of becoming vertical stripes.
    facade_light = normalized_blur(raw_light * facade, facade, radius=64.0)
    roof_light = normalized_blur(raw_light * roof, roof, radius=24.0)
    light = np.where(roof, roof_light, facade_light)
    facade_depth = normalized_blur(raw_depth * facade, facade, radius=48.0)
    roof_depth = normalized_blur(raw_depth * roof, roof, radius=24.0)
    depth = np.clip(np.where(roof, roof_depth, facade_depth), 0.0, 1.0)

    base_facade = np.array([207.0, 199.0, 190.0], dtype=np.float32)
    base_roof = np.array([91.0, 174.0, 204.0], dtype=np.float32)
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[facade] = base_facade
    rgb[roof] = base_roof

    if use_lighting:
        # Roughly 20-25% usable diffuse contrast across differently oriented
        # macro surfaces, plus a small near/far depth cue.
        light_factor = 0.68 + 0.42 * light
        depth_factor = 0.965 + 0.055 * depth
        vertical = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
        vertical_factor = np.where(facade, 1.015 - 0.055 * vertical, 1.0)
        rgb *= (light_factor * depth_factor * vertical_factor)[:, :, None]

    if use_ao:
        # A soft, local contact shadow just below the roof/facade transition.
        # It is self-shading only: no ground plane or cast ground shadow.
        roof_float = Image.fromarray(np.uint8(roof) * 255).filter(
            ImageFilter.GaussianBlur(radius=18.0)
        )
        contact = np.asarray(roof_float, dtype=np.float32) / 255.0
        contact = np.clip(contact * 2.15, 0.0, 1.0) * facade
        rgb *= (1.0 - 0.115 * contact)[:, :, None]

    if use_roughness:
        texture = texture_field(size=size, seed=seed)
        # Texture modulates irradiance rather than painting a material pattern.
        # The facade is slightly rougher than the roof, but both remain neutral.
        amplitude = roughness_scale * np.where(roof, 0.020, 0.032)
        rgb *= (1.0 + amplitude * texture)[:, :, None]

    if vertical_profile != "none":
        # Estimate relative facade height independently in each image column.
        # This follows the projected top and bottom contours under perspective,
        # while avoiding horizontal seams or any assumed story segmentation.
        relative_height = np.zeros((size, size), dtype=np.float32)
        columns = np.flatnonzero(facade.any(axis=0))
        for x in columns:
            ys = np.flatnonzero(facade[:, x])
            if len(ys) == 1:
                relative_height[ys[0], x] = 0.5
                continue
            top = int(ys[0])
            bottom = int(ys[-1])
            span = max(bottom - top, 1)
            relative_height[ys, x] = 1.0 - (ys - top) / span

        if vertical_profile == "soft":
            height_factor = 0.93 + 0.08 * relative_height
        elif vertical_profile == "medium":
            height_factor = 0.86 + 0.16 * relative_height
        elif vertical_profile == "soft_band":
            broad_shadow = np.exp(
                -0.5 * ((relative_height - 0.48) / 0.19) ** 2
            )
            height_factor = 0.88 + 0.14 * relative_height - 0.060 * broad_shadow
        elif vertical_profile == "strong":
            height_factor = 0.78 + 0.24 * relative_height
        else:
            # A continuous, non-linear sky-exposure profile. The upper third
            # receives a gentle lift; the lower third receives broad ambient
            # attenuation. There is no story boundary or horizontal hard edge.
            smooth_height = relative_height * relative_height * (
                3.0 - 2.0 * relative_height
            )
            base_occlusion = np.exp(-relative_height / 0.16)
            top_exposure = relative_height**3
            height_factor = (
                0.80
                + 0.22 * smooth_height
                - 0.050 * base_occlusion
                + 0.030 * top_exposure
            )
        rgb[facade] *= height_factor[facade, None]

    silhouette = mask_boundary(foreground, width=4)
    roof_boundary = mask_boundary(roof, width=3) & foreground
    outline = silhouette | roof_boundary
    outline_color = np.array([72.0, 67.0, 62.0], dtype=np.float32)
    rgb[outline] = 0.70 * outline_color + 0.30 * rgb[outline]

    alpha = np.zeros((size, size), dtype=np.uint8)
    alpha[foreground | silhouette] = 255
    rgba = np.dstack((np.uint8(np.clip(rgb, 0, 255)), alpha))
    image = Image.fromarray(rgba, mode="RGBA")
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((canvas_size, canvas_size), resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)

PROMPT_CALL_1_TEMPLATE = """Reconstruct a complete and realistic standalone building from the four reference images.

Image 1 is a reconstructed LoD1 view and serves as a coarse visual proxy for the target building. Use it to guide the approximate building massing, proportions, orientation, and output viewpoint, but do not reproduce its geometry literally, as it may be incomplete or noisy.

The target building is approximately {HEIGHT_M} metres tall. Use this physical height together with the visual evidence to infer a plausible vertical scale, facade proportions, and floor organization for this building type.

Image 2 is the primary appearance evidence for the target building. It is often a near-top-down observation showing part of the roof and limited visible portions of the building's facades within the context shown in Image 4. Use it to infer facade materials, colors, windows, patterns, architectural style, and any visible roof appearance.

Image 3 is a possibly incomplete recovered top-down roof image. Use it to infer roof materials, layout, equipment, and other roof features.

Image 4 shows the surrounding context from which the building in Image 2 was extracted. Use it to identify and understand the target building represented by Image 1, but do not reproduce the surrounding scene.

Combine the available evidence and plausibly complete any missing or uncertain parts to create a coherent full building. Prioritize Image 2 for facade appearance, Image 3 for roof appearance, and Image 1 as a coarse guide to the overall form and viewpoint.

Generate the completed building from a slightly elevated oblique viewpoint similar to Image 1, clearly showing one main facade, part of an adjacent facade, and some of the roof. Show the entire building in a square image with clear margins and no cropping.

Use a uniform pure white background, with no ground plane, cast shadow, surrounding scene, or added objects."""

def format_height_m(value: Any) -> str:
    height_m = float(value)
    if not math.isfinite(height_m) or height_m <= 0:
        raise ValueError(f"Invalid building height: {value!r}")
    return f"{height_m:.2f}".rstrip("0").rstrip(".")

def prompt_for_height(value: Any) -> str:
    return PROMPT_CALL_1_TEMPLATE.format(HEIGHT_M=format_height_m(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_local(root: Path, value: str) -> Path:
    root = Path(root).resolve()
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Evidence paths must remain relative to their package")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _shape_yaw(geometry) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return 0.0
    points = list(rectangle.exterior.coords)
    edges = [(math.hypot(b[0] - a[0], b[1] - a[1]),
              math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
             for a, b in zip(points[:-1], points[1:])]
    return float(max(edges, key=lambda item: item[0])[1] % 180.0)


def prepare_building(building_path: Path, destination: Path, *, provider_config: dict | None = None) -> dict:
    """Adapt a released building without dropping components or observed evidence.

    The released canonical observation is selected explicitly. Its RGBA crops
    remain unchanged; only the proxy is rendered with the frozen v4 recipe.
    """
    from shapely.geometry import shape
    from .provider import provider_config as normalize_provider_config, transport_for

    provider = normalize_provider_config(provider_config)
    provider_fields = {key: provider[key] for key in ("model", "quality", "size")}
    provider_fields["transport"] = transport_for(provider)
    building_path = Path(building_path).resolve()
    destination = Path(destination).resolve()
    building = json.loads(building_path.read_text())
    if building.get("schema") != "citycode.building_generation_input.v1" or building.get("status") != "complete":
        raise ValueError("Expected a complete released building-generation input")
    sid = building["building_id"]
    height = float(building["height_m"])
    format_height_m(height)
    geometry = shape(building["footprint_world"])
    if not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
        raise ValueError(f"Invalid released footprint: {sid}")
    canonical_id = building["canonical_observation_id"]
    observations = [o for o in building["observations"] if o["observation_id"] == canonical_id]
    if len(observations) != 1:
        raise ValueError(f"Canonical observation is not unique: {sid}")
    observation_path = resolve_local(building_path.parent, observations[0]["path"] + "/observation.json")
    observation = json.loads(observation_path.read_text())
    if observation["building_id"] != sid or observation["observation_id"] != canonical_id:
        raise ValueError("Observation identity mismatch")
    proxy = resolve_local(building_path.parent, building["paths"]["proxy_local_ue_z_up"])
    sources = {key: resolve_local(observation_path.parent, observation["paths"][key])
               for key in ("source_primary", "roof_evidence", "context")}
    for key, path in sources.items():
        with Image.open(path) as image:
            image.verify()
        if key != "context":
            with Image.open(path) as image:
                if image.mode != "RGBA":
                    raise ValueError(f"Expected released RGBA evidence: {key}")
    yaw = _shape_yaw(geometry)
    provenance = {
        "building_sha256": sha256_file(building_path),
        "observation_sha256": sha256_file(observation_path),
        "proxy_sha256": sha256_file(proxy),
        "evidence_sha256": {key: sha256_file(path) for key, path in sources.items()},
        "renderer": RENDER_PROFILE, "shape_yaw_world_deg": yaw,
        "renderer_source_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    record_path = destination / "appearance_input.json"
    if record_path.is_file():
        previous = json.loads(record_path.read_text())
        if previous.get("input_identity_sha256") != identity:
            raise ValueError(f"Existing input has a different identity; use a new directory: {sid}")
        if any(previous.get(key) != value for key, value in provider_fields.items()):
            raise ValueError(f"Existing input has a different provider identity; use a new directory: {sid}")
        if all((destination / row["path"]).is_file() and sha256_file(destination / row["path"]) == row["sha256"]
               for row in previous["inputs"]):
            return previous
        raise ValueError(f"Existing input files changed: {sid}")
    input_dir = destination / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for key, source in sources.items():
        shutil.copy2(source, input_dir / (key + ".png"))
    render_spatial_proxy(proxy, input_dir / INPUT_NAMES[0], azimuth_deg=yaw + 45.0,
                         elevation_deg=34.0, mode="perspective_light_ao_rough",
                         vertical_profile="sculpted", roughness_scale=1.65,
                         perspective_distance_scale=2.80,
                         seed=stable_seed("lod1_v4_sculpted_rough", sid, "main"))
    prompt = prompt_for_height(height)
    (destination / "prompt.txt").write_text(prompt + "\n")
    records = []
    for name in INPUT_NAMES:
        path = input_dir / name
        with Image.open(path) as image:
            records.append({"path": f"inputs/{name}", "sha256": sha256_file(path),
                            "mode": image.mode, "size": list(image.size)})
    parts = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    record = {"schema": "citycode.appearance_input.v1", "status": "prepared",
              "building_id": sid, "canonical_observation_id": canonical_id,
              "height_m": height, "footprint_area_m2": float(geometry.area),
              "footprint_components": len(parts), "footprint_holes": sum(len(p.interiors) for p in parts),
              "world_from_local_4x4": building["world_from_local_4x4"],
              "input_identity_sha256": identity, "provenance": provenance,
              "inputs": records, "prompt": "prompt.txt",
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              **provider_fields}
    write_json(record_path, record)
    return record


def completion_contract(record: dict) -> str:
    value = {key: record[key] for key in ("model", "quality", "size", "prompt_sha256")}
    value["input_sha256"] = [row["sha256"] for row in record["inputs"]]
    if "transport" in record:
        value["transport"] = record["transport"]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reuse_completed_image(directory: Path, candidate: Path) -> bool:
    """Reuse only an identical four-image, height-prompt and provider-model request.

    IDs alone are deliberately insufficient. Historical records without this
    complete contract require an explicit identity audit before adaptation.
    """
    directory, candidate = Path(directory), Path(candidate)
    current = json.loads((directory / "appearance_input.json").read_text())
    if current.get("transport", {}).get("protocol") == "gemini_native":
        # Native completions require their full canonical and response evidence.
        # Their in-place reuse is verified by complete_native, before any request.
        return False
    receipt_path = candidate / "completion.json"
    if not receipt_path.is_file():
        return False
    receipt = json.loads(receipt_path.read_text())
    source = candidate / "generated_main_oblique.png"
    if (receipt.get("status") != "complete" or receipt.get("request_identity_sha256") != completion_contract(current)
            or not source.is_file() or sha256_file(source) != receipt.get("output_sha256")):
        return False
    shutil.copy2(source, directory / source.name)
    write_json(directory / "completion.json", {**receipt, "reused": True,
                                                "source_receipt_sha256": sha256_file(receipt_path)})
    return True


def complete_building(directory: Path, *, provider_config: dict | None = None,
                      provider_python: Path | None = None, provider_cli: Path | None = None,
                      timeout_seconds: int | None = None) -> dict:
    """Complete the standard four-image request with the configured provider."""
    directory = Path(directory)
    record = json.loads((directory / "appearance_input.json").read_text())
    receipt_path = directory / "completion.json"
    if receipt_path.is_file():
        imported = json.loads(receipt_path.read_text())
        if imported.get("origin") == "external_completed_image":
            output = directory / "generated_main_oblique.png"
            if (imported.get("status") != "complete" or not output.is_file()
                    or imported.get("request_identity_sha256") != completion_contract(record)
                    or imported.get("output_sha256") != sha256_file(output)):
                raise ValueError("Imported completed image changed; use a new run")
            return imported
    if record.get("transport", {}).get("protocol") == "gemini_native":
        from .provider import complete_native
        return complete_native(directory, record, config=provider_config, timeout_seconds=timeout_seconds)
    if provider_python is None or provider_cli is None:
        raise ValueError("Legacy prepared requests require their original provider CLI adapter")
    request_identity = completion_contract(record)
    output = directory / "generated_main_oblique.png"
    receipt_path = directory / "completion.json"
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if (previous.get("status") == "complete" and previous.get("request_identity_sha256") == request_identity
                and output.is_file() and sha256_file(output) == previous.get("output_sha256")):
            return previous
    secret = os.environ.get("OPENAI_API_KEY", "")
    if not secret:
        raise RuntimeError("Provider credential wrapper must set OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    from urllib.parse import urlparse
    if urlparse(base_url).scheme != "https":
        raise ValueError("Configure an HTTPS provider endpoint")
    prompt_path = directory / record["prompt"]
    if hashlib.sha256(prompt_path.read_text().strip().encode()).hexdigest() != record["prompt_sha256"]:
        raise ValueError("Prompt changed after preparation")
    command = [str(provider_python), str(provider_cli), "edit"]
    for row in record["inputs"]:
        path = directory / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("Appearance evidence changed after preparation")
        command += ["--image", str(path)]
    temporary = directory / "generated_main_oblique.partial.png"
    command += ["--prompt-file", str(prompt_path), "--no-augment", "--model", record["model"],
                "--n", "1", "--size", record["size"], "--quality", record["quality"],
                "--output-format", "png", "--out", str(temporary), "--force"]
    request = {"schema": "citycode.appearance_request.v1", "building_id": record["building_id"],
               "request_identity_sha256": request_identity, "model": record["model"],
               "quality": record["quality"], "size": record["size"], "n": 1,
               "output_format": "png", "prompt": prompt_path.read_text().strip(),
               "prompt_sha256": record["prompt_sha256"], "inputs": record["inputs"],
               "provider_host": urlparse(base_url).hostname,
               "endpoint_path": urlparse(base_url).path.rstrip("/") + "/images/edits",
               "provider_cli_sha256": sha256_file(provider_cli)}
    write_json(directory / "request.json", request)
    start = time.monotonic()
    child_env = dict(os.environ, CITYCODE_PROVIDER_RESPONSE=str(directory / "provider_response.json"))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds or 900, check=False, env=child_env)
    receipt = {"schema": "citycode.appearance_completion.v1", "building_id": record["building_id"],
               "request_identity_sha256": request_identity, "model": record["model"],
               "quality": record["quality"], "provider_host": urlparse(base_url).hostname,
               "elapsed_seconds": time.monotonic() - start, "returncode": result.returncode}
    receipt["request_sha256"] = sha256_file(directory / "request.json")
    receipt["cli_stdout"] = result.stdout[-4000:].replace(secret, "<redacted>")
    receipt["cli_stderr"] = result.stderr[-4000:].replace(secret, "<redacted>")
    response_path = directory / "provider_response.json"
    if response_path.is_file():
        receipt["provider_response_sha256"] = sha256_file(response_path)
    valid = False
    if result.returncode == 0 and temporary.is_file():
        with Image.open(temporary) as image:
            image.load()
            valid = image.format == "PNG" and image.size == (1024, 1024)
    if not valid:
        receipt.update(status="failed", error=(result.stderr or result.stdout)[-1800:].replace(secret, "<redacted>"))
        write_json(receipt_path, receipt)
        raise RuntimeError(f"Appearance completion failed for {record['building_id']}; see completion.json")
    temporary.replace(output)
    receipt.update(status="complete", output="generated_main_oblique.png", output_sha256=sha256_file(output))
    write_json(receipt_path, receipt)
    return receipt
