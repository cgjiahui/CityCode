"""Source-bound CPU postprocessing of unfinished building assets.

Reviewed repairs retain detailed meshes. Structural replacements retain the
predicted geometry and observed roof appearance, with explicitly neutral walls.
Observed fragments are separate surface attachments, never invented buildings.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct

import numpy as np
from PIL import Image
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .lod import inspect_glb, sha256
from .registration import read_glb_scene_geometry


GLTF_FROM_UE = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                         [0., -1., 0., 0.], [0., 0., 0., 1.]])
UE_FROM_GLTF = np.linalg.inv(GLTF_FROM_UE)
METHODS = {"source_structure", "observed_fragment_surface", "local_mesh_repair", "retain_fragment_evidence"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def file_proof(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return {"path": os.path.relpath(path, root), "sha256": sha256(path), "bytes": path.stat().st_size}


def checked_file(item, root, allowed_root):
    _require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"}, "Exact file proof required")
    relative = item["path"]
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
             and "\\" not in relative and not re.match(r"^[A-Za-z]:", relative), "Relative POSIX evidence path required")
    path = (Path(root) / relative).resolve()
    _require(path.is_relative_to(Path(allowed_root).resolve()) and path.is_file()
             and not (Path(root) / relative).is_symlink(), "Evidence escaped the data package")
    _require(type(item["bytes"]) is int and path.stat().st_size == item["bytes"]
             and sha256(path) == item["sha256"], "Evidence bytes changed: " + relative)
    return path


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    owned = False
    try:
        with temporary.open("x") as stream:
            owned = True
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def observed_roof_material(building_path, observation_id=None):
    """Read an actual alpha-masked BEV crop and its pixel-center calibration."""
    building_path = Path(building_path)
    building = _read(building_path)
    identity = observation_id or building["canonical_observation_id"]
    selected = [x for x in building["observations"] if x["observation_id"] == identity]
    _require(len(selected) == 1, "Observation is absent or ambiguous")
    folder = (building_path.parent / selected[0]["path"]).resolve()
    _require(folder.is_relative_to(building_path.parent.resolve()), "Observation escaped its building")
    observation_path = folder / "observation.json"
    observation = _read(observation_path)
    _require(observation["building_id"] == building["building_id"]
             and observation["observation_id"] == identity, "Observation identity mismatch")
    image_path = folder / observation["paths"]["roof_evidence"]
    _require(image_path.resolve().is_relative_to(folder)
             and sha256(image_path) == observation["files"][image_path.name], "Observed roof crop changed")
    with Image.open(image_path) as image:
        rgba = np.array(image.convert("RGBA"))
    # Alpha denotes observed support. Unknown pixels receive a declared neutral
    # material, never a synthesized continuation of a roof or neighboring image.
    alpha = rgba[:, :, 3:4].astype(float) / 255.
    rgb = np.rint(rgba[:, :, :3] * alpha + 174. * (1. - alpha)).astype(np.uint8)
    rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), constant_values=174)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    calibration = np.asarray(observation["bev_crop_pixel_center_to_world_xy_3x3"], float)
    _require(calibration.shape == (3, 3) and np.isfinite(calibration).all()
             and abs(np.linalg.det(calibration)) > 1e-12, "Invalid BEV calibration")
    return {"png": buffer.getvalue(), "size": [rgb.shape[1], rgb.shape[0]],
            "pixel_from_world": np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]) @ np.linalg.inv(calibration),
            "observation_id": identity, "observation_path": observation_path,
            "image_path": image_path, "observed_pixels": int(np.count_nonzero(rgba[:, :, 3])),
            "unknown_pixel_policy": "neutral_174_RGB"}


def roof_uv(world_vertices, material):
    xy = np.column_stack([np.asarray(world_vertices)[:, :2], np.ones(len(world_vertices))])
    pixel = xy @ material["pixel_from_world"].T
    pixel = pixel[:, :2] / pixel[:, 2:3]
    # glTF UV origin is the top-left of the encoded image. The calibration maps
    # integer pixel-center indices, hence the half-pixel conversion to UVs.
    return (pixel + .5) / np.asarray(material["size"])


def write_textured_glb(path, vertices, faces, world_from_asset, materials, face_material=None):
    """Encode exact triangle positions with flat normals, UVs and embedded PBR."""
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype=np.int64)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
             and faces.ndim == 2 and faces.shape[1] == 3 and len(faces)
             and faces.min() >= 0 and faces.max() < len(vertices), "Invalid actual mesh")
    triangles = vertices[faces].astype(float)
    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(vectors, axis=1)
    normals = np.divide(vectors, lengths[:, None], out=np.tile([0., 1., 0.], (len(faces), 1)), where=lengths[:, None] > 1e-15)
    if face_material is None:
        face_material = np.where(normals[:, 1] > .5, 0, -1)
    face_material = np.asarray(face_material, dtype=int)
    _require(face_material.shape == (len(faces),) and set(face_material) <= set(range(len(materials))) | {-1},
             "Invalid material membership")
    binary = bytearray(); views = []; accessors = []
    document = {"asset": {"version": "2.0", "generator": "CityCode source postprocessing"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": []}], "bufferViews": views, "accessors": accessors,
        "images": [], "textures": [], "samplers": [{"wrapS": 33071, "wrapT": 33071,
            "minFilter": 9729, "magFilter": 9729}], "materials": []}
    def view(payload, target=None):
        index = len(views); item = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
        if target is not None: item["target"] = target
        views.append(item); binary.extend(payload); binary.extend(b"\x00" * (-len(binary) % 4))
        return index
    def accessor(array, kind, component=5126, target=34962):
        array = np.ascontiguousarray(array, dtype="<u4" if component == 5125 else "<f4")
        row = {"bufferView": view(array.tobytes(), target), "componentType": component,
               "count": len(array), "type": kind}
        if kind == "VEC3": row.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(row)
        return len(accessors) - 1
    neutral = io.BytesIO(); Image.new("RGB", (2, 2), (166, 164, 159)).save(neutral, format="PNG")
    textures = [x["png"] for x in materials] + [neutral.getvalue()]
    for i, png in enumerate(textures):
        document["images"].append({"bufferView": view(png), "mimeType": "image/png"})
        document["textures"].append({"source": i, "sampler": 0})
        document["materials"].append({"name": "observed_roof" if i < len(materials) else "neutral_unobserved_surface",
            "doubleSided": False, "pbrMetallicRoughness": {"baseColorTexture": {"index": i},
            "metallicFactor": 0., "roughnessFactor": .9}})
    transform = np.asarray(world_from_asset, float)
    for selected in sorted(set(face_material)):
        selected = int(selected)
        mask = face_material == selected
        positions = vertices[faces[mask]].reshape(-1, 3)
        normal = np.repeat(normals[mask], 3, axis=0)
        if selected >= 0:
            world = positions @ transform[:3, :3].T + transform[:3, 3]
            uv = roof_uv(world, materials[selected])
        else:
            uv = np.full((len(positions), 2), .5)
        document["meshes"][0]["primitives"].append({"attributes": {
            "POSITION": accessor(positions, "VEC3"), "NORMAL": accessor(normal, "VEC3"),
            "TEXCOORD_0": accessor(uv, "VEC2")},
            "indices": accessor(np.arange(len(positions)), "SCALAR", 5125, 34963),
            "material": selected if selected >= 0 else len(materials), "mode": 4})
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    body = struct.pack("<I4s", len(encoded), b"JSON") + encoded
    body += struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body)
    return inspect_glb(path)


def thin_surface(vertices_world, faces, thickness=.12):
    """Close a source-supported surface downwards, retaining every top vertex."""
    vertices = np.asarray(vertices_world, float); faces = np.asarray(faces, np.int64)
    _require(0 < thickness <= .25, "Only a small declared visualization thickness is allowed")
    count = len(vertices)
    bottom = vertices.copy(); bottom[:, 2] -= thickness
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    _require(counts.max() <= 2, "Source surface has nonmanifold shared edges")
    boundary = edges[counts[inverse] == 1]
    side = []
    for a, b in boundary:
        side.extend([[b, a, a + count], [b, a + count, b + count]])
    all_faces = np.vstack([faces, faces[:, ::-1] + count, np.asarray(side, np.int64).reshape(-1, 3)])
    return np.vstack([vertices, bottom]), all_faces


def _base_postprocess(building_path, asset_root, method, representation, decision):
    building = _read(building_path)
    proxy_path = building_path.parent / building["paths"]["proxy_local_gltf_y_up"]
    return {"schema": "citycode.asset_postprocess_record.v1", "method": method,
        "representation": representation, "recipe_version": "source_postprocess_v1",
        "source": {"building": file_proof(building_path, asset_root),
                   "proxy": file_proof(proxy_path, asset_root), "evidence": decision["evidence"]},
        "source_target_geometry": building["footprint_world"],
        "source_base_z_m": building["base_z_m"], "source_top_z_m": building["top_z_m"],
        "reason_code": decision["reason_code"], "reason": decision["reason"],
        "reviewer": decision["reviewer"], "fine_generation_claimed": False,
        "detailed_asset_accepted": representation == "repaired_detailed",
        "ground_truth_used_for_geometry": False}


def _lod_metadata(asset_root, paths):
    result = {}
    for label, path in paths.items():
        scene = read_glb_scene_geometry(path)
        result[label] = {**inspect_glb(path), "faces": len(scene["faces"]),
            "path": path.relative_to(asset_root).as_posix(),
            "target_faces": 250000 if label == "0.25M" else 100000}
    return result


def build_structure_record(row, building_path, decision, asset_root, folder):
    building = _read(building_path)
    proxy = building_path.parent / building["paths"]["proxy_local_gltf_y_up"]
    world = np.asarray(building["world_from_local_4x4"]) @ UE_FROM_GLTF
    geometry_source = proxy
    if "layered_mesh" in decision:
        geometry_source = checked_file(decision["layered_mesh"], asset_root, asset_root.parent)
    mesh = read_glb_scene_geometry(geometry_source)
    vertices = mesh["vertices"]
    if "layered_mesh" in decision:
        # The stage-01 canonical mesh stores world positions in glTF axes.
        # Rebase once into this building's local frame; do not translate twice.
        local_from_source = np.linalg.inv(world) @ UE_FROM_GLTF
        vertices = vertices @ local_from_source[:3, :3].T + local_from_source[:3, 3]
    material = observed_roof_material(building_path)
    high, low = folder / "model_025m.glb", folder / "model_01m.glb"
    write_textured_glb(high, vertices, mesh["faces"], world, [material])
    shutil.copyfile(high, low)
    result = copy.deepcopy(row)
    result.update(status="accepted_structure", reason=decision["reason"], asset_instance_id=row["building_id"],
        local_coordinate_system="glTF Y-up", world_coordinate_system="UE XYZ metres Z-up",
        world_from_local_4x4=world.tolist(), building_local_from_asset_4x4=UE_FROM_GLTF.tolist(),
        world_from_building_local_4x4=building["world_from_local_4x4"],
        normal_transform_world_3x3=np.linalg.inv(world[:3, :3]).T.tolist(),
        base_z_m=building["base_z_m"], top_z_m=building["top_z_m"],
        target_geometry=building["footprint_world"],
        lods=_lod_metadata(asset_root, {"0.25M": high, "0.1M": low}))
    # Previous candidate checks belong to another mesh and are never approvals
    # of this structural replacement.
    for key in ["validation", "semantic_review", "geometry_fit_metrics", "reuse_fit_metrics"]:
        result.pop(key, None)
    result["provenance"] = {"kind": "source_structure_postprocess", "source_proxy_sha256": sha256(proxy)}
    result["postprocess"] = _base_postprocess(building_path, asset_root, "source_structure", "structure", decision)
    if "layered_mesh" in decision:
        result["postprocess"]["source"].update(layered_mesh=file_proof(geometry_source, asset_root),
            geometry_source_frame="world_gltf_y_up")
        result["postprocess"].update(recorded_roof_layers_preserved=True,
            geometry_rebase="world_gltf_y_up_to_building_local_gltf_y_up",
            geometry_source_sha256=sha256(geometry_source))
    result["postprocess"].update(facade_detail_recovered=False, lods_identical=True,
        geometry_positions_and_faces_preserved=True, material_policy="observed_roof_with_neutral_walls",
        roof_observation=file_proof(material["observation_path"], asset_root),
        roof_image=file_proof(material["image_path"], asset_root), unknown_roof_pixels="neutral_174_RGB")
    return result


def build_fragment_record(row, building_path, decision, asset_root, folder, plan_root):
    building = _read(building_path)
    support_path = checked_file(decision["surface_support"], plan_root, asset_root.parent)
    support_metadata_path = checked_file(decision["surface_metadata"], plan_root, asset_root.parent)
    metadata = _read(support_metadata_path)
    _require(metadata["building_id"] == row["building_id"], "Surface belongs to another fragment")
    with np.load(support_path, allow_pickle=False) as payload:
        vertices = payload["vertices_world_xyz_m"].astype(float)
        faces = payload["faces"].astype(np.int64)
        # Each top triangle carries the original observed view, not a guessed
        # correspondence to a neighboring building.
        face_observation = payload["triangle_observation_index"].astype(int)
    _require(len(faces) and face_observation.shape == (len(faces),), "No source-supported fragment triangles")
    thickness = .12
    full_vertices, full_faces = thin_surface(vertices, faces, thickness)
    world = np.asarray(building["world_from_local_4x4"]) @ UE_FROM_GLTF
    inverse = np.linalg.inv(world)
    local_vertices = full_vertices @ inverse[:3, :3].T + inverse[:3, 3]
    materials = [observed_roof_material(building_path, o["observation_id"]) for o in metadata["source_observations"]]
    face_material = np.r_[face_observation, np.full(len(full_faces) - len(faces), -1)]
    high, low = folder / "model_025m.glb", folder / "model_01m.glb"
    write_textured_glb(high, local_vertices, full_faces, world, materials, face_material)
    shutil.copyfile(high, low)
    footprint = unary_union([shape({"type": "Polygon", "coordinates": [triangle[:, :2].tolist()]})
                             for triangle in vertices[faces]])
    _require(not footprint.is_empty and footprint.is_valid and footprint.area > 0, "Invalid observed fragment surface")
    result = copy.deepcopy(row)
    result.update(status="resolved_fragment", reason=decision["reason"],
        surface_instance_id="SURFACE_" + row["building_id"],
        local_coordinate_system="glTF Y-up", world_coordinate_system="UE XYZ metres Z-up",
        world_from_local_4x4=world.tolist(), building_local_from_asset_4x4=UE_FROM_GLTF.tolist(),
        world_from_building_local_4x4=building["world_from_local_4x4"],
        normal_transform_world_3x3=np.linalg.inv(world[:3, :3]).T.tolist(),
        base_z_m=float(vertices[:, 2].min() - thickness), top_z_m=float(vertices[:, 2].max()),
        target_geometry=mapping(footprint),
        lods=_lod_metadata(asset_root, {"0.25M": high, "0.1M": low}))
    for key in ["validation", "semantic_review", "geometry_fit_metrics", "reuse_fit_metrics", "asset_instance_id"]:
        result.pop(key, None)
    result["provenance"] = {"kind": "observed_fragment_surface_postprocess", "support_sha256": sha256(support_path)}
    result["postprocess"] = _base_postprocess(building_path, asset_root, "observed_fragment_surface", "observed_surface", decision)
    result["postprocess"].update(parent_building_id=decision.get("parent_building_id"),
        full_building_reconstructed=False, complete_building=False, facade_detail_recovered=False, lods_identical=True,
        source_footprint_observed_fraction=float(footprint.area / shape(building["footprint_world"]).area),
        material_policy="observed_roof_with_neutral_surface_edges",
        surface={"support": file_proof(support_path, asset_root),
            "metadata": file_proof(support_metadata_path, asset_root),
            "thickness_m": thickness, "thickness_is_visualization_only": True,
            "visible_source_coverage": float(footprint.area / shape(building["footprint_world"]).area),
            "assembly_enabled": decision.get("assembly_enabled", True),
            "parent_building_id": decision.get("parent_building_id")})
    return result


def build_repaired_record(row, building_path, decision, asset_root, folder, plan_root):
    replacement = checked_file(decision["repaired_record"], plan_root, asset_root.parent)
    result = _read(replacement)
    _require(result["building_id"] == row["building_id"] and result["status"] == "accepted_repaired",
             "Explicit reviewed repaired record required")
    paths = {}
    if "repair_recipe" in decision:
        from .mesh_repair import repair_lods
        recipe = checked_file(decision["repair_recipe"], plan_root, asset_root.parent)
        inputs = {label: checked_file(item, plan_root, asset_root.parent)
                  for label, item in decision["original_lods"].items()}
        repair_lods(inputs["0.25M"], inputs["0.1M"], result["world_from_local_4x4"],
                    folder / "lods", recipe, expected_recipe_sha256=decision["repair_recipe"]["sha256"])
        paths = {"0.25M": folder / "lods/model_025m.glb", "0.1M": folder / "lods/model_01m.glb"}
    else:
        for label, name in [("0.25M", "model_025m.glb"), ("0.1M", "model_01m.glb")]:
            source = checked_file(decision["repaired_lods"][label], plan_root, asset_root.parent)
            destination = folder / name
            shutil.copyfile(source, destination)
            paths[label] = destination
    result["lods"] = _lod_metadata(asset_root, paths)
    # Repaired inputs are already reviewed public, portable records; do not
    # reconstruct a numeric or semantic approval from the original failed fit.
    return result


def postprocess_assets(manifest_path, decisions_path, structure_root, output_path):
    """Apply a finite reviewed plan while preserving every accepted input row."""
    from .cli import ACCEPTED, _validate_accepted, _policy
    from .postprocess_validation import validate_postprocess_record, validate_surface_instances

    manifest_path, decisions_path = Path(manifest_path).resolve(), Path(decisions_path).resolve()
    output_path, structure_root = Path(output_path).resolve(), Path(structure_root).resolve()
    root = manifest_path.parent
    _require(output_path.parent == root and output_path != manifest_path,
             "Build a new manifest beside the input; replace production only after validation")
    _require(not output_path.exists(), "Postprocessed output already exists")
    before_sha = sha256(manifest_path); base = _read(manifest_path); plan = _read(decisions_path)
    _require(base.get("schema") == "citycode.building_asset_bindings.v1" and not base.get("postprocessing"),
             "A fresh unprocessed binding manifest is required")
    _require(plan.get("schema") == "citycode.asset_postprocess_decisions.v1"
             and plan.get("base_binding_sha256") == before_sha, "Plan is bound to other binding bytes")
    _require(sha256(structure_root / "manifest.json") == base["source_structure"]["manifest_sha256"],
             "Current structure manifest changed")
    records = plan.get("records")
    _require(isinstance(records, list) and records and len({r["building_id"] for r in records}) == len(records),
             "A finite unique decision list is required")
    by_id = {r["building_id"]: r for r in base["buildings"]}
    selected = {r["building_id"] for r in records}
    _require(selected <= set(by_id), "Plan has unknown source IDs")
    _require(all(by_id[sid]["status"] not in ACCEPTED for sid in selected), "Accepted assets cannot be replaced by this cleanup")
    for decision in records:
        _require(decision.get("action") in METHODS and decision.get("reviewer")
                 and isinstance(decision.get("reason"), str) and decision["reason"].strip()
                 and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", decision.get("reason_code", "")), "Unreviewed postprocess action")
        _require(isinstance(decision.get("evidence"), list) and decision["evidence"], "Decision lacks actual source evidence")
        for evidence in decision["evidence"]:
            checked_file(evidence, root, root.parent)
    result = copy.deepcopy(base); result_rows = {r["building_id"]: r for r in result["buildings"]}
    created = []; outputs = []; surfaces = []
    folder_root = root / "postprocessed_assets"
    _require(not folder_root.exists(), "Postprocessed assets already exist; use the recorded output")
    folder_root.mkdir(); created.append(folder_root)
    try:
        for decision in records:
            sid = decision["building_id"]; row = by_id[sid]
            building_path = (root / row["structure_path"]).resolve()
            _require(building_path.is_relative_to(structure_root) and building_path.is_file(), "Current building escaped the structure input")
            building = _read(building_path)
            _require(building["building_id"] == sid and shape(building["footprint_world"]).equals_exact(shape(row["target_geometry"]), 0),
                     "Source footprint or identity changed")
            folder = folder_root / sid; folder.mkdir()
            action = decision["action"]
            if action == "source_structure":
                updated = build_structure_record(row, building_path, decision, root, folder)
            elif action == "observed_fragment_surface":
                updated = build_fragment_record(row, building_path, decision, root, folder, decisions_path.parent)
            elif action == "local_mesh_repair":
                updated = build_repaired_record(row, building_path, decision, root, folder, decisions_path.parent)
            else:
                updated = copy.deepcopy(row)
                _require(updated["status"] in {"excluded_incomplete", "excluded_noise"}, "Evidence-only action needs a confirmed fragment/noise record")
                updated["reason"] = decision["reason"]
                updated["postprocess"] = _base_postprocess(building_path, root, action, "evidence_only", decision)
                updated["postprocess"]["parent_building_id"] = decision.get("parent_building_id")
            if action != "retain_fragment_evidence":
                source_check = validate_postprocess_record(root, updated, geometry=True)
                actual_checks = _validate_accepted(root, updated, _policy(base), geometry=True)
                updated["postprocess"]["validation"] = {"status": "passed", "source": source_check,
                    "actual_lods": actual_checks, "reference_geometry_read": False}
                _json_write(folder / "lod_manifest.json", {"schema": "citycode.postprocessed_lods.v1",
                    "status": "completed", "building_id": sid, "lods": updated["lods"],
                    "representation": updated["postprocess"]["representation"],
                    "source_building_sha256": updated["postprocess"]["source"]["building"]["sha256"]})
                if updated["status"] == "resolved_fragment":
                    surfaces.append({"id": updated["surface_instance_id"], "source_building_ids": [sid],
                        "status": "resolved_fragment", "world_from_local_4x4": updated["world_from_local_4x4"],
                        "lods": updated["lods"], "parent_building_id": decision.get("parent_building_id"),
                        "assembly_enabled": decision.get("assembly_enabled", True)})
                else:
                    result["asset_instances"].append({"id": updated["asset_instance_id"], "source_building_ids": [sid],
                        "status": updated["status"], "world_from_local_4x4": updated["world_from_local_4x4"],
                        "lods": updated["lods"]})
            result_rows[sid].clear(); result_rows[sid].update(updated)
            outputs.append({"building_id": sid, "before": row["status"], "after": updated["status"],
                "action": action, "reason": decision["reason"], "representation": updated["postprocess"]["representation"]})
        result["surface_instances"] = surfaces
        validate_surface_instances(result, result["buildings"])
        protected = [r for r in base["buildings"] if r["status"] in ACCEPTED]
        _require(all(result_rows[r["building_id"]] == r for r in protected), "An accepted source row changed")
        _require(result["asset_instances"][:len(base["asset_instances"])] == base["asset_instances"], "An existing physical instance changed")
        _require(result["source_structure"] == base["source_structure"], "Source structure contract changed")
        counts = Counter(r["status"] for r in result["buildings"])
        result["counts"] = dict(counts)
        from .cli import PENDING
        pending = [r["building_id"] for r in result["buildings"] if r["status"] in PENDING]
        result["status"] = "complete" if not pending else "in_progress"
        result["updated_utc"] = datetime.now(timezone.utc).isoformat()
        result["postprocessing"] = {"schema": "citycode.asset_postprocess_summary.v1", "recipe": "source_postprocess_v1",
            "input_binding_sha256": before_sha, "decisions": file_proof(decisions_path, root),
            "selected_count": len(records), "action_counts": dict(Counter(r["action"] for r in outputs)),
            "protected_existing_physical_assets": len(base["asset_instances"]),
            "protected_existing_source_records": len(protected),
            "building_asset_count": len(result["asset_instances"]), "surface_attachment_count": len(surfaces),
            "enabled_surface_attachments": sum(s.get("assembly_enabled", True) for s in surfaces),
            "reference_only_surface_attachments": sum(not s.get("assembly_enabled", True) for s in surfaces),
            "structural_assets": counts["accepted_structure"], "repaired_assets": counts["accepted_repaired"],
            "fine_generation_complete": counts["accepted_structure"] == 0 and not pending,
            "source_structures_modified": False, "paid_API_calls": 0, "GPU_jobs": 0,
            "quality_note": "Structural assets use observed roof imagery and neutral walls; surface attachments retain only directly observed geometry and are not independent buildings.",
            "records": outputs}
        result["assembly_contract"] = "Instantiate every asset_instances entry once. Add only surface_instances with assembly_enabled=true by default; the remaining surface entries are an optional evidence layer. A surface parent is a relation, not a second source-building membership."
        _require(sha256(manifest_path) == before_sha, "Input changed during postprocessing")
        _json_write(output_path, result)
    except BaseException:
        for folder in reversed(created):
            shutil.rmtree(folder)
        raise
    return result["postprocessing"]
