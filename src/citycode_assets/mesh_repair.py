"""Replay reviewed, SHA-bound CPU repairs between two static glTF LODs.

The caller supplies a reviewed recipe; this module does not infer which visible
openings are errors. Endpoint contractions keep existing UV-chart neighbours.
Patches copy actual triangles and their own UV, normal and PBR data from the other
LOD. The original binary chunks remain intact prefixes of each result.

``repair_lods(high, low, world_from_asset, output_dir, recipe,
expected_recipe_sha256=...)`` writes two GLBs and ``repair_report.json`` into a new
directory. A recipe binds both input hashes, the unchanged pose, ordered low-LOD
endpoint contractions, exact donor triangle IDs and fixed face budgets. Review
and independent source/geometry/PBR validation remain the caller's responsibility.
Residual shared openings may remain: this is not a watertightness guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any

import numpy as np


RECIPE_SCHEMA = "citycode.reviewed_mesh_repair_recipe.v1"
MAX_FACE_BUDGETS = {"0.25M": 250000, "0.1M": 100000}
_COMPONENTS = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _proof(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": _sha(data), "bytes": len(data)}


def _read(path: Path):
    data = path.read_bytes()
    if len(data) < 28 or struct.unpack_from("<4sII", data) != (b"glTF", 2, len(data)):
        raise ValueError("Invalid GLB v2 header/length")
    chunks, cursor = {}, 12
    while cursor < len(data):
        if cursor + 8 > len(data):
            raise ValueError("Truncated GLB chunk header")
        length, kind = struct.unpack_from("<II", data, cursor)
        cursor += 8
        if length % 4 or cursor + length > len(data) or kind in chunks:
            raise ValueError("Invalid or duplicate GLB chunk")
        chunks[kind] = data[cursor:cursor + length]
        cursor += length
    if set(chunks) != {0x4E4F534A, 0x004E4942}:
        raise ValueError("Repair requires JSON and one embedded BIN chunk")
    doc = json.loads(chunks[0x4E4F534A])
    binary = chunks[0x004E4942]
    if (len(doc.get("buffers", [])) != 1 or doc["buffers"][0].get("uri")
            or doc["buffers"][0]["byteLength"] > len(binary)):
        raise ValueError("Repair requires one self-contained GLB buffer")
    if doc.get("animations") or doc.get("skins"):
        raise ValueError("Animated/skinned meshes cannot enter a static repair")
    if len(doc.get("meshes", [])) != 1 or len(doc["meshes"][0]["primitives"]) != 1:
        raise ValueError("Recipe input must contain one original triangle primitive")
    if len(doc.get("scenes", [])) != 1 or len(doc["scenes"][0].get("nodes", [])) != 1:
        raise ValueError("Recipe requires one explicit scene root")
    nodes = doc.get("nodes", [])
    roots = doc["scenes"][0]["nodes"]
    visited, mesh_instances = set(), []

    def visit(index):
        if not isinstance(index, int) or index < 0 or index >= len(nodes) or index in visited:
            raise ValueError("Invalid, instanced or cyclic repair scene")
        visited.add(index)
        node = nodes[index]
        if any(key in node for key in ("matrix", "translation", "rotation", "scale", "skin", "weights")):
            raise ValueError("Repair currently requires identity mesh-node transforms")
        if "mesh" in node:
            mesh_instances.append(node["mesh"])
        for child in node.get("children", []):
            visit(child)

    visit(roots[0])
    if mesh_instances != [0] or len(visited) != len(nodes):
        raise ValueError("Repair requires one active mesh and no unused scene nodes")
    primitive = doc["meshes"][0]["primitives"][0]
    if (primitive.get("mode", 4) != 4 or primitive.get("extensions")
            or primitive.get("targets") or "indices" not in primitive):
        raise ValueError("Only static uncompressed indexed triangles are supported")
    for required in ("POSITION", "NORMAL", "TEXCOORD_0"):
        if required not in primitive.get("attributes", {}):
            raise ValueError("Positions, original normals and UVs are required")
    return doc, binary


def _accessor(doc, binary, index):
    spec = doc["accessors"][index]
    if spec.get("sparse") or spec.get("normalized"):
        raise ValueError("Sparse/normalized repair streams are unsupported")
    view = doc["bufferViews"][spec["bufferView"]]
    if view.get("buffer", 0) != 0 or view.get("extensions"):
        raise ValueError("External/compressed repair streams are unsupported")
    dtype = np.dtype(_COMPONENTS[spec["componentType"]])
    width, count = _WIDTHS[spec["type"]], int(spec["count"])
    stride = int(view.get("byteStride", dtype.itemsize * width))
    relative = int(spec.get("byteOffset", 0))
    length = max(0, count - 1) * stride + width * dtype.itemsize
    start = int(view.get("byteOffset", 0)) + relative
    if (count < 1 or relative < 0 or stride < width * dtype.itemsize
            or relative + length > view["byteLength"] or start + length > len(binary)):
        raise ValueError("Repair accessor exceeds its buffer view")
    return np.ndarray((count, width), dtype, buffer=binary, offset=start,
                      strides=(stride, dtype.itemsize)).copy()


def _arrays(doc, binary):
    primitive = doc["meshes"][0]["primitives"][0]
    attrs = {key: _accessor(doc, binary, index)
             for key, index in primitive["attributes"].items()}
    vertices, normals, uv = (attrs[key] for key in ("POSITION", "NORMAL", "TEXCOORD_0"))
    faces = _accessor(doc, binary, primitive["indices"]).reshape(-1)
    if (vertices.shape[1:] != (3,) or normals.shape != vertices.shape
            or uv.shape != (len(vertices), 2) or len(faces) % 3
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)
            or any(len(a) != len(vertices) or not np.isfinite(a).all() for a in attrs.values())
            or np.any(np.linalg.norm(normals, axis=1) < 1e-10)):
        raise ValueError("Invalid repair vertex, normal, UV or index streams")
    return attrs, faces.reshape(-1, 3).astype(np.int64)


def _blob(doc, binary: bytearray, payload: bytes, target=None):
    binary.extend(b"\0" * (-len(binary) % 4))
    view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
    if target is not None:
        view["target"] = target
    binary.extend(payload)
    doc["bufferViews"].append(view)
    return len(doc["bufferViews"]) - 1


def _append_indices(doc, binary, faces):
    view = _blob(doc, binary, np.asarray(faces, dtype="<u4").ravel().tobytes(), 34963)
    doc["accessors"].append({"bufferView": view, "componentType": 5125,
                             "count": int(np.asarray(faces).size), "type": "SCALAR"})
    return len(doc["accessors"]) - 1


def _write(path, doc, binary):
    doc["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    binary = bytes(binary) + b"\0" * (-len(binary) % 4)
    payload = (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary))
               + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
               + struct.pack("<II", len(binary), 0x004E4942) + binary)
    path.write_bytes(payload)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _replay_low(doc, binary, passes, matrix):
    """Replay explicit contractions; no angle, geometry or texture optimisation."""
    doc, binary = copy.deepcopy(doc), bytearray(binary)
    attrs, faces = _arrays(doc, binary)
    vertices, uv, normals = (attrs[key].astype(np.float64) for key in ("POSITION", "TEXCOORD_0", "NORMAL"))
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    summaries = []
    for pass_number, contraction_pass in enumerate(passes):
        if len(faces) != contraction_pass["original_triangles"]:
            raise ValueError("Contraction pass input triangle count changed")
        live = np.ones(len(faces), bool)
        removed_total, plane_max, uv_max = 0, 0., 0.
        for contraction in contraction_pass["endpoint_collapses"]:
            remap = {}
            for key, value in contraction["original_vertex_remap"].items():
                # JSON object keys are strings; require canonical decimal indices.
                a = int(key)
                if str(a) != str(key):
                    raise ValueError("Noncanonical endpoint index")
                q = _integer(value, "endpoint target")
                if min(a, q) < 0 or max(a, q) >= len(vertices) or a == q:
                    raise ValueError("Endpoint contraction index is invalid")
                remap[a] = q
            if not remap or set(remap) & set(remap.values()):
                raise ValueError("Endpoint contraction must contain distinct source/target vertices")
            ids = np.flatnonzero(live & np.isin(faces, list(remap)).any(axis=1))
            if not len(ids):
                raise ValueError("Endpoint contraction no longer addresses actual triangles")
            old, new = faces[ids].copy(), faces[ids].copy()
            for a, q in remap.items():
                if not np.any(np.any(old == a, axis=1) & np.any(old == q, axis=1)):
                    raise ValueError("Contraction endpoint is not an original chart edge neighbour")
                if np.dot(normals[a], normals[q]) < .97 or np.linalg.norm(uv[a] - uv[q]) > .025:
                    raise ValueError("Contraction crosses a normal/UV chart constraint")
                new[new == a] = q
            ow, nw = world[old], world[new]
            on = np.cross(ow[:, 1] - ow[:, 0], ow[:, 2] - ow[:, 0])
            nn = np.cross(nw[:, 1] - nw[:, 0], nw[:, 2] - nw[:, 0])
            ol, nl = np.linalg.norm(on, axis=1), np.linalg.norm(nn, axis=1)
            keep = nl > 1e-12
            removed = int((~keep).sum())
            if removed < 1 or removed > 4 or removed != contraction["removed_triangles"]:
                raise ValueError("Contraction removed triangle count changed")
            alignment = np.sum(on[keep] * nn[keep], axis=1) / np.maximum(ol[keep] * nl[keep], 1e-30)
            if np.any(alignment < .995):
                raise ValueError("Contraction would flip or rotate a surviving triangle")
            plane_error = np.abs(np.sum((nw - ow) * on[:, None, :], axis=2)) / np.maximum(ol[:, None], 1e-30)
            measured_plane = float(plane_error.max())
            if measured_plane > .02:
                raise ValueError("Contraction exceeds original-plane error bound")
            measured_uv = 0.
            for of, nf in zip(old, new):
                points, coords = world[of], uv[of]
                axes = np.stack([points[1] - points[0], points[2] - points[0]], axis=1)
                if np.linalg.matrix_rank(axes) < 2:
                    continue
                weights = np.linalg.lstsq(axes, (world[nf] - points[0]).T, rcond=None)[0].T
                predicted = (coords[0] + weights[:, 0, None] * (coords[1] - coords[0])
                             + weights[:, 1, None] * (coords[2] - coords[0]))
                measured_uv = max(measured_uv, float(np.linalg.norm(predicted - uv[nf], axis=1).max()))
            if measured_uv > .0015:
                raise ValueError("Contraction exceeds local UV-affine error bound")
            faces[ids] = new
            live[ids[~keep]] = False
            removed_total += removed
            plane_max, uv_max = max(plane_max, measured_plane), max(uv_max, measured_uv)
        faces = faces[live]
        if len(faces) != contraction_pass["result_triangles"]:
            raise ValueError("Contraction pass result triangle count changed")
        primitive = doc["meshes"][0]["primitives"][0]
        primitive["indices"] = _append_indices(doc, binary, faces)
        summaries.append({"pass": pass_number, "result_triangles": len(faces),
                          "endpoint_contractions": len(contraction_pass["endpoint_collapses"]),
                          "removed_triangles": removed_total,
                          "maximum_original_plane_error_m": plane_max,
                          "maximum_local_UV_affine_error": uv_max})
    return doc, binary, summaries


def _triangle_ids(values, face_count):
    ids = np.asarray([_integer(q, "donor triangle ID") for q in values], dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Donor triangle IDs must be a nonempty unique list")
    if ids.min() < 0 or ids.max() >= face_count:
        raise ValueError("Donor triangle ID is outside the SHA-bound original mesh")
    return ids


def _patch(destination_doc, destination_binary, donor_doc, donor_binary, donor_ids):
    doc, binary = copy.deepcopy(destination_doc), bytearray(destination_binary)
    donor_attrs, donor_faces = _arrays(donor_doc, donor_binary)
    source = donor_doc["meshes"][0]["primitives"][0]
    selected = donor_faces[donor_ids]
    used = np.unique(selected)
    remap = np.full(int(selected.max()) + 1, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    attributes = {}
    for name, values in donor_attrs.items():
        values = values[used]
        spec = copy.deepcopy(donor_doc["accessors"][source["attributes"][name]])
        spec.pop("byteOffset", None)
        spec["count"] = len(values)
        spec["bufferView"] = _blob(doc, binary, values.tobytes(), 34962)
        spec["min"], spec["max"] = values.min(axis=0).tolist(), values.max(axis=0).tolist()
        doc["accessors"].append(spec)
        attributes[name] = len(doc["accessors"]) - 1
    indices = _append_indices(doc, binary, remap[selected])
    image_offset, texture_offset, material_offset, sampler_offset = (
        len(doc.get(key, [])) for key in ("images", "textures", "materials", "samplers"))
    for image in donor_doc.get("images", []):
        image = copy.deepcopy(image)
        if "bufferView" not in image or "uri" in image:
            raise ValueError("Repair PBR images must be embedded")
        view = donor_doc["bufferViews"][image["bufferView"]]
        start, length = int(view.get("byteOffset", 0)), int(view["byteLength"])
        if start < 0 or start + length > len(donor_binary):
            raise ValueError("Embedded donor image exceeds its binary buffer")
        image["bufferView"] = _blob(doc, binary, donor_binary[start:start + length])
        doc.setdefault("images", []).append(image)
    for sampler in donor_doc.get("samplers", []):
        doc.setdefault("samplers", []).append(copy.deepcopy(sampler))
    for texture in donor_doc.get("textures", []):
        texture = copy.deepcopy(texture)
        if "sampler" in texture:
            texture["sampler"] += sampler_offset
        if "source" in texture:
            texture["source"] += image_offset
        for extension in texture.get("extensions", {}).values():
            if "source" in extension:
                extension["source"] += image_offset
        doc.setdefault("textures", []).append(texture)

    def material_textures(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("Texture"):
                    if not isinstance(child, dict) or "index" not in child:
                        raise ValueError("Unsupported donor material texture binding")
                    child["index"] += texture_offset
                else:
                    material_textures(child)
        elif isinstance(value, list):
            for child in value:
                material_textures(child)

    for material in donor_doc.get("materials", []):
        material = copy.deepcopy(material)
        material_textures(material)
        doc.setdefault("materials", []).append(material)
    if "material" not in source:
        raise ValueError("Donor patches require their original explicit material")
    primitive = {"attributes": attributes, "indices": indices, "mode": 4,
                 "material": source["material"] + material_offset}
    doc["meshes"][0]["primitives"].append(primitive)
    for key in ("extensionsUsed", "extensionsRequired"):
        if donor_doc.get(key):
            doc[key] = sorted(set(doc.get(key, [])) | set(donor_doc[key]))
    return doc, binary


def repair_lods(high: str | Path, low: str | Path, world_from_asset, output: str | Path,
                recipe: str | Path | dict[str, Any], *, expected_recipe_sha256: str | None = None):
    """Replay a reviewed recipe into a new directory, preserving input files.

    A recipe path is checked against its exact file bytes. A dictionary is checked
    against canonical JSON bytes (sorted keys, compact separators, UTF-8). Supply
    ``expected_recipe_sha256`` from the independently reviewed recipe. A digest
    binds content and is not a cryptographic signature or reviewer authentication.

    Final files are installed together only after every recipe operation passes.
    Existing outputs are never overwritten. Geometry/PBR/source acceptance is a
    separate check; the report always leaves ``release_approved`` false.
    """
    high, low, output = Path(high).resolve(), Path(low).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Repair output directory already exists")
    recipe_bytes = _canonical(recipe) if isinstance(recipe, dict) else Path(recipe).read_bytes()
    recipe_sha = _sha(recipe_bytes)
    if expected_recipe_sha256 is None or recipe_sha != expected_recipe_sha256:
        raise ValueError("An exact reviewed recipe SHA256 is required")
    plan = copy.deepcopy(recipe) if isinstance(recipe, dict) else json.loads(recipe_bytes)
    if (plan.get("schema") != RECIPE_SCHEMA
            or plan.get("status") != "reviewed_actual_triangle_selection"
            or not str(plan.get("reviewer", "")).strip()):
        raise ValueError("Recipe lacks explicit actual-triangle selection review")
    matrix = np.asarray(world_from_asset, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-12
            or not np.array_equal(matrix, np.asarray(plan["world_from_asset_4x4"], dtype=np.float64))):
        raise ValueError("Recipe requires the exact finite, affine, nonsingular original pose")
    paths = {"0.25M": high, "0.1M": low}
    inputs, documents = {}, {}
    for label, path in paths.items():
        inputs[label] = _proof(path)
        if any(inputs[label][key] != plan["inputs"][label][key] for key in ("sha256", "bytes")):
            raise ValueError("SHA-bound repair input changed: " + label)
        documents[label] = _read(path)
        _arrays(*documents[label])
    high_doc, high_bin = documents["0.25M"]
    low_doc, low_bin, contractions = _replay_low(*documents["0.1M"],
                                               plan.get("low_endpoint_passes", []), matrix)
    if high_doc["nodes"] != low_doc["nodes"] or high_doc["scenes"] != low_doc["scenes"]:
        raise ValueError("LOD patches require identical identity-node scene graphs")
    prepatch = {"0.25M": (high_doc, high_bin), "0.1M": (low_doc, low_bin)}
    patched, counts, selections = {}, {}, {}
    for label, donor in (("0.25M", "0.1M"), ("0.1M", "0.25M")):
        donor_arrays, donor_faces = _arrays(*prepatch[donor])
        del donor_arrays
        ids = _triangle_ids(plan["donor_triangle_ids"][label], len(donor_faces))
        _, original_faces = _arrays(*prepatch[label])
        budget = _integer(plan.get("face_budgets", MAX_FACE_BUDGETS)[label], "face budget")
        counts[label] = len(original_faces) + len(ids)
        if budget < 1 or budget > MAX_FACE_BUDGETS[label] or counts[label] > budget:
            raise ValueError("Repair exceeds the unchanged LOD face budget: " + label)
        patched[label] = _patch(*prepatch[label], *prepatch[donor], ids)
        selections[label] = {"donor_lod": donor, "donor_triangle_count": len(ids),
                             "donor_triangle_ids_sha256": _sha(_canonical(ids.tolist()))}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".citycode_mesh_repair_", dir=output.parent))
    try:
        names = {"0.25M": "model_025m.glb", "0.1M": "model_01m.glb"}
        outputs = {}
        for label, name in names.items():
            doc, binary = patched[label]
            # Original stream/image data remain untouched, even after low reindexing.
            original_doc, original_binary = documents[label]
            if bytes(binary[:len(original_binary)]) != original_binary:
                raise ValueError("Repair unexpectedly changed original binary data")
            if (doc["nodes"] != original_doc["nodes"] or doc["scenes"] != original_doc["scenes"]
                    or doc["materials"][:len(original_doc.get("materials", []))] != original_doc.get("materials", [])):
                raise ValueError("Repair unexpectedly changed original scene/materials")
            _write(stage / name, doc, binary)
            outputs[label] = {**_proof(stage / name), "path": name, "faces": counts[label]}
        # Check inputs again immediately before installing a complete result.
        if any(_proof(path) != inputs[label] for label, path in paths.items()):
            raise ValueError("Repair input changed during replay")
        relative_inputs = {label: {**item, "path": Path(os.path.relpath(paths[label], output)).as_posix()}
                           for label, item in inputs.items()}
        report = {"schema": "citycode.mesh_repair_report.v1", "status": "replayed_requires_independent_validation",
                  "path_base": "report_directory",
                  "building_id": plan.get("building_id"), "recipe_sha256": recipe_sha,
                  "method": "reviewed endpoint contractions and actual cross-LOD triangle transfer",
                  "inputs": relative_inputs, "outputs": outputs, "world_from_asset_4x4": matrix.tolist(),
                  "pose_changed": False, "face_budgets": plan.get("face_budgets", MAX_FACE_BUDGETS),
                  "low_endpoint_passes": contractions, "selections": selections,
                  "original_binary_prefixes_unchanged": True,
                  "original_POSITION_NORMAL_TEXCOORD_and_PBR_image_bytes_unchanged": True,
                  "new_patch_data_from_actual_donor_LOD": True,
                  "residual_openings": plan.get("residual_openings", "Not assessed by recipe replay"),
                  "watertightness_guaranteed": False, "release_approved": False,
                  "limitations": ["Shared holes can remain in both original LODs.",
                                  "Local plane/UV error bounds apply per contraction pass; no global Hausdorff bound is claimed.",
                                  "Independent source correspondence, LOD/PBR and scene validation remain required."],
                  "code_sha256": _sha(Path(__file__).read_bytes())}
        (stage / "repair_recipe.json").write_bytes(recipe_bytes)
        (stage / "repair_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stage.rename(output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)
