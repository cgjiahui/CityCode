"""Material-preserving GLB validation, normal repair and decoded-mesh LOD export."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


LOD_SPECS = {
    100000: ("0.1M", "model_01m.glb"),
    250000: ("0.25M", "model_025m.glb"),
    1000000: ("1M", "model_1m.glb"),
}


def generation_lod_targets(config):
    """Keep the standard pair, optionally adding the independently baked 1M LOD."""
    targets = config.get("lod_targets", [250000, 100000])
    if (not isinstance(targets, (list, tuple)) or not targets
            or any(type(value) is not int or value not in LOD_SPECS for value in targets)
            or len(set(targets)) != len(targets)
            or not {250000, 100000}.issubset(targets)):
        raise ValueError("lod_targets must contain 250000 and 100000, with optional 1000000, without duplicates")
    return tuple(sorted(targets, reverse=True))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_glb(path: Path):
    data = bytearray(path.read_bytes())
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError("Invalid GLB header")
    offset, document, binary_offset = 12, None, None
    while offset < len(data):
        size, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        if kind == b"JSON":
            document = json.loads(data[offset:offset + size])
        elif kind == b"BIN\x00":
            binary_offset = offset
        offset += size
    if document is None or binary_offset is None:
        raise ValueError("Embedded GLB document and binary required")
    return data, document, binary_offset


def _accessor(data, document, binary_offset, index):
    a = document["accessors"][index]
    if "sparse" in a:
        raise ValueError("Sparse GLB accessors are not supported by this exporter")
    view = document["bufferViews"][a["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise ValueError("External buffer not supported")
    dtype = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
    dims = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    item = np.dtype(dtype).itemsize
    stride = view.get("byteStride", dims * item)
    start = binary_offset + view.get("byteOffset", 0) + a.get("byteOffset", 0)
    array = np.ndarray((a["count"], dims), dtype=dtype, buffer=data, offset=start, strides=(stride, item))
    return array, a


def _replace_document(data, document):
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * ((-len(encoded)) % 4)
    chunks, offset = [], 12
    while offset < len(data):
        size, kind = struct.unpack_from("<I4s", data, offset)
        payload = encoded if kind == b"JSON" else bytes(data[offset + 8:offset + 8 + size])
        chunks.append(struct.pack("<I4s", len(payload), kind) + payload)
        offset += 8 + size
    body = b"".join(chunks)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def inspect_glb(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data, doc, binary = _read_glb(path)
    for resource in doc.get("buffers", []) + doc.get("images", []):
        if resource.get("uri") and not resource["uri"].startswith("data:"):
            raise ValueError("GLB references a nonportable external resource")
    faces, primitives, zero_normals = 0, 0, 0
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh in doc.get("meshes", []):
        for p in mesh["primitives"]:
            if p.get("mode", 4) != 4:
                raise ValueError("Triangle mesh required")
            attrs = p["attributes"]
            if not {"POSITION", "NORMAL", "TEXCOORD_0"}.issubset(attrs):
                raise ValueError("Positions, encoded normals and UVs are required")
            vertices, _ = _accessor(data, doc, binary, attrs["POSITION"])
            normals, _ = _accessor(data, doc, binary, attrs["NORMAL"])
            uv, _ = _accessor(data, doc, binary, attrs["TEXCOORD_0"])
            indices = _accessor(data, doc, binary, p["indices"])[0].ravel() if "indices" in p else np.arange(len(vertices))
            if len(indices) % 3 or not len(indices) or indices.max() >= len(vertices):
                raise ValueError("Invalid triangle indices")
            used = np.unique(indices)
            if not all(np.isfinite(a).all() for a in [vertices[used], normals[used], uv[used]]):
                raise ValueError("Nonfinite mesh attributes")
            zero_normals += int(np.count_nonzero(np.linalg.norm(normals[used], axis=1) < 1e-8))
            material = doc.get("materials", [])[p["material"]]
            if "baseColorTexture" not in material.get("pbrMetallicRoughness", {}):
                raise ValueError("Textured PBR material required")
            minimum = np.minimum(minimum, vertices[used].min(axis=0))
            maximum = np.maximum(maximum, vertices[used].max(axis=0))
            faces += len(indices) // 3
            primitives += 1
    if not primitives or zero_normals:
        raise ValueError(f"Empty mesh or {zero_normals} used zero normals")
    return {"faces": int(faces), "primitives": primitives, "material_count": len(doc.get("materials", [])),
            "bounds_local_accessor": [minimum.tolist(), maximum.tolist()], "encoded_normals_and_uv": True,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def repair_glb_normals(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Repair invalid normals and discard exact zero-area triangles; preserve geometry/UVs/materials."""
    source, destination = Path(source), Path(destination)
    data, doc, binary = _read_glb(source)
    changed_used = changed_unused = removed_zero_area = 0
    for mesh in doc.get("meshes", []):
        for p in mesh["primitives"]:
            normals, normal_accessor = _accessor(data, doc, binary, p["attributes"]["NORMAL"])
            if normal_accessor["componentType"] != 5126:
                raise ValueError("Normal repair requires FLOAT normals")
            vertices, _ = _accessor(data, doc, binary, p["attributes"]["POSITION"])
            bad = ~np.isfinite(normals).all(axis=1) | (np.linalg.norm(normals, axis=1) < 1e-8)
            if not bad.any():
                continue
            indices = _accessor(data, doc, binary, p["indices"])[0].ravel() if "indices" in p else np.arange(len(vertices))
            triangles = indices.reshape(-1, 3)
            v = np.asarray(vertices, dtype=np.float64)
            face_normals = np.cross(v[triangles[:, 1]] - v[triangles[:, 0]], v[triangles[:, 2]] - v[triangles[:, 0]])
            nondegenerate = np.linalg.norm(face_normals, axis=1) > 0
            if not nondegenerate.all():
                if "indices" not in p:
                    raise ValueError("Degenerate unindexed mesh requires an explicit mesh repair")
                kept_triangles = triangles[nondegenerate].copy()
                compact = kept_triangles.ravel()
                if not len(compact):
                    raise ValueError("All source triangles are degenerate")
                index_array, index_accessor = _accessor(data, doc, binary, p["indices"])
                index_array.ravel()[:len(compact)] = compact
                index_accessor["count"] = len(compact)
                index_accessor["min"] = [int(compact.min())]
                index_accessor["max"] = [int(compact.max())]
                removed_zero_area += int(np.count_nonzero(~nondegenerate))
                indices = compact
                triangles = kept_triangles
                face_normals = face_normals[nondegenerate]
            accumulator = np.zeros((len(v), 3), dtype=np.float64)
            for corner in range(3):
                np.add.at(accumulator, triangles[:, corner], face_normals)
            used = np.zeros(len(v), dtype=bool)
            used[indices] = True
            for index in np.flatnonzero(bad):
                vector = accumulator[index]
                length = np.linalg.norm(vector)
                if used[index] and length == 0:
                    adjacent = face_normals[(triangles == index).any(axis=1)]
                    if not len(adjacent) or np.linalg.norm(adjacent, axis=1).max() == 0:
                        raise ValueError("Cannot repair normal of a geometrically degenerate used vertex")
                    vector = adjacent[np.argmax(np.linalg.norm(adjacent, axis=1))]
                    length = np.linalg.norm(vector)
                if used[index]:
                    normals[index] = vector / length
                    changed_used += 1
                else:
                    normals[index] = [0, 1, 0]
                    changed_unused += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(_replace_document(data, doc) if removed_zero_area else data)
    temporary.replace(destination)
    result = inspect_glb(destination)
    result.update(source_sha256=sha256(source), repaired_used_normals=changed_used,
                  repaired_unused_normals=changed_unused, exact_zero_area_faces_removed=removed_zero_area,
                  position_uv_material_buffer_bytes_unchanged=True,
                  positive_area_triangles_unchanged=True)
    return result


def _install_closed_mesh_guard():
    import cumesh
    original = cumesh.CuMesh.fill_holes
    if getattr(original, "_citycode_closed_guard", False):
        return
    def guarded(self, max_hole_perimeter=3e-2):
        _, faces = self.read()
        triangles = faces.detach().cpu().numpy()
        if not triangles.size:
            return None
        edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)[1]
        if not np.any(counts == 1):
            return None
        self.get_boundary_loops()
        loops, edges, _ = self.read_boundary_loops()
        if int(loops) == 0 or edges.numel() == 0:
            return None
        return original(self, max_hole_perimeter=max_hole_perimeter)
    guarded._citycode_closed_guard = True
    cumesh.CuMesh.fill_holes = guarded


def export_decoded_lods(raw_cache: str | Path, output: str | Path, *, targets=(250000, 100000),
                        texture_size=2048, master_glb: str | Path | None = None,
                        allow_verified_existing: bool = False) -> dict[str, Any]:
    """Bake each requested resolution from the same decoded mesh; never upsample a low LOD."""
    import torch
    import o_voxel
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for CuMesh/xatlas and texture baking")
    _install_closed_mesh_guard()
    raw_cache, output = Path(raw_cache), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(raw_cache, map_location="cpu", weights_only=True)
    required = {"vertices", "faces", "attrs", "coords", "resolution", "attr_layout"}
    if not required.issubset(payload):
        raise ValueError("Decoded mesh cache fields missing")
    for key in ["vertices", "attrs"]:
        if not torch.isfinite(payload[key]).all():
            raise ValueError("Nonfinite decoded cache")
    results = {}
    for target in targets:
        if not isinstance(target, int) or target <= 0:
            raise ValueError("LOD face budgets must be positive integers")
        label, name = LOD_SPECS.get(target, (f"{target}_faces", f"model_{target}.glb"))
        destination = output / name
        if not destination.exists() or not allow_verified_existing:
            if target == 250000 and master_glb is not None:
                repair_glb_normals(master_glb, destination)
            else:
                temporary = output / (name + ".partial.glb")
                glb = o_voxel.postprocess.to_glb(
                    vertices=payload["vertices"].cuda(), faces=payload["faces"].cuda(),
                    attr_volume=payload["attrs"].cuda(), coords=payload["coords"].cuda(),
                    attr_layout={key: slice(*value) for key, value in payload["attr_layout"].items()},
                    grid_size=int(payload["resolution"]), aabb=[[-.5, -.5, -.5], [.5, .5, .5]],
                    decimation_target=target, texture_size=texture_size, remesh=True,
                    remesh_band=1, remesh_project=0, use_tqdm=False)
                glb.export(temporary, extension_webp=True)
                repair_glb_normals(temporary, destination)
                temporary.unlink()
                del glb
        result = inspect_glb(destination)
        if result["faces"] > target * 1.02:
            raise ValueError(f"LOD face count exceeds tolerance: {result['faces']} > {target}")
        result.update(path=str(destination), target_faces=target, texture_size=texture_size)
        results[label] = result
    if len(results) >= 2:
        high = max(results.values(), key=lambda value: value["target_faces"])
        span = np.maximum(np.ptp(np.asarray(high["bounds_local_accessor"]), axis=0), 1e-6)
        for low in results.values():
            bounds_delta = np.abs(np.asarray(low["bounds_local_accessor"]) - np.asarray(high["bounds_local_accessor"]))
            if np.max(bounds_delta / span) > .025:
                raise ValueError("LOD local coordinate bounds drift exceeds2.5 percent")
    torch.cuda.empty_cache()
    return {"status": "completed", "raw_cache_sha256": sha256(raw_cache), "lods": results,
            "shared_local_coordinate_frame": True, "positions_not_world_baked": True}
