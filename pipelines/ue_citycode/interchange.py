"""Lossless CPU preparation for UE 5.0 GLTFImporter (NumPy and Pillow only).

The output stores *local UE XYZ metres* as raw glTF numbers. The editor adapter
measures its importer basis with a fixture and cancels that basis on each actor.
This avoids guessing a plugin axis convention or decomposing a sheared binding.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import mmap
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from PIL import Image

VERSION = 1
COMPONENTS = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_glb(path):
    with Path(path).open("rb") as stream:
        payload = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    if len(payload) < 20 or struct.unpack_from("<4sII", payload) != (b"glTF", 2, len(payload)):
        raise ValueError("Invalid GLB v2: " + str(path))
    doc, binary, offset = None, None, 12
    while offset < len(payload):
        length, kind = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        if offset + length > len(payload):
            raise ValueError("GLB chunk exceeds file")
        if kind == b"JSON":
            doc = json.loads(payload[offset:offset + length])
        elif kind == b"BIN\0":
            binary = memoryview(payload)[offset:offset + length]
        offset += length
    if doc is None or binary is None:
        raise ValueError("Embedded JSON and BIN chunks are required")
    if len(doc.get("buffers", [])) != 1 or doc["buffers"][0].get("uri"):
        raise ValueError("Only self-contained GLB is supported")
    if doc.get("skins") or doc.get("animations"):
        raise ValueError("Animated/skinned input requires a dedicated baker")
    return doc, binary, digest_file(path)


def node_matrix(node):
    if "matrix" in node:
        if any(k in node for k in ("translation", "rotation", "scale")):
            raise ValueError("Node combines matrix and TRS")
        result = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
    else:
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        result = np.eye(4)
        result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        result[:3, :3] @= np.diag(node.get("scale", [1, 1, 1]))
        result[:3, 3] = node.get("translation", [0, 0, 0])
    if not np.isfinite(result).all() or not np.allclose(result[3], [0, 0, 0, 1]):
        raise ValueError("Non-finite/non-affine node transform")
    return result


def scene_meshes(doc):
    roots = doc["scenes"][doc.get("scene", 0)].get("nodes", [])
    def visit(index, parent, ancestors):
        if index in ancestors:
            raise ValueError("Cyclic glTF scene")
        node = doc["nodes"][index]
        matrix = parent @ node_matrix(node)
        if "mesh" in node:
            yield index, node["mesh"], matrix
        for child in node.get("children", []):
            yield from visit(child, matrix, ancestors | {index})
    for root in roots:
        yield from visit(root, np.eye(4), set())


def accessor(doc, binary, index):
    a = doc["accessors"][index]
    if "sparse" in a or "bufferView" not in a:
        raise ValueError("Sparse/accessor without a bufferView is unsupported")
    view = doc["bufferViews"][a["bufferView"]]
    if view.get("extensions") or view.get("buffer", 0) != 0:
        raise ValueError("Compressed/external bufferView is unsupported")
    dtype, width = np.dtype(COMPONENTS[a["componentType"]]), WIDTHS[a["type"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    end = start + max(0, a["count"] - 1) * stride + dtype.itemsize * width
    if end > view.get("byteOffset", 0) + view["byteLength"] or end > len(binary):
        raise ValueError("Accessor outside bufferView")
    result = np.ndarray((a["count"], width), dtype=dtype, buffer=binary,
                        offset=start, strides=(stride, dtype.itemsize)).copy()
    return result, a


class Writer:
    def __init__(self, directory=None):
        # A full background contains tens of millions of triangles. Keep its BIN
        # on disk: bytearray + GLB concatenation otherwise multiplies peak RAM.
        stream = tempfile.NamedTemporaryFile(prefix="citycode-bin-", suffix=".tmp", dir=directory, delete=False)
        self.binary = stream
        self.binary_path = Path(stream.name)
        self.doc = {"asset": {"version": "2.0", "generator": "CityCode UE CPU interchange v1"},
                    "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
                    "accessors": [], "bufferViews": []}

    def blob(self, value, target=None):
        self.binary.write(b"\0" * (-self.binary.tell() % 4))
        view = {"buffer": 0, "byteOffset": self.binary.tell(), "byteLength": len(value)}
        if target:
            view["target"] = target
        self.doc["bufferViews"].append(view)
        self.binary.write(value)
        return len(self.doc["bufferViews"]) - 1

    def array(self, value, source=None, target=34962):
        value = np.ascontiguousarray(value)
        a = {k: copy.deepcopy(v) for k, v in (source or {}).items()
             if k not in ("bufferView", "byteOffset", "min", "max", "sparse")}
        if source is None:
            a.update(componentType=5126, type="VEC3", count=len(value))
        a["bufferView"] = self.blob(memoryview(value).cast("B"), target)
        a["min"], a["max"] = value.min(axis=0).tolist(), value.max(axis=0).tolist()
        self.doc["accessors"].append(a)
        return len(self.doc["accessors"]) - 1

    def save(self, path):
        self.doc["buffers"] = [{"byteLength": self.binary.tell()}]
        encoded = json.dumps(self.doc, separators=(",", ":"), allow_nan=False).encode()
        encoded += b" " * (-len(encoded) % 4)
        self.binary.write(b"\0" * (-self.binary.tell() % 4))
        binary_length = self.binary.tell()
        self.binary.flush(); self.binary.seek(0)
        temporary = Path(str(path) + ".writing")
        with temporary.open("wb") as output:
            output.write(struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + binary_length))
            output.write(struct.pack("<I4s", len(encoded), b"JSON")); output.write(encoded)
            output.write(struct.pack("<I4s", binary_length, b"BIN\0"))
            shutil.copyfileobj(self.binary, output, length=4*1024*1024)
        temporary.replace(path)
        self.binary.close(); self.binary_path.unlink(missing_ok=True)

    def __del__(self):
        try:
            self.binary.close(); self.binary_path.unlink(missing_ok=True)
        except Exception:
            pass


def bake(source, destination, world_from_scene, anchor_m, expected_sha256=None):
    """Compose every scene-node transform before the supplied binding affine.

    Geometry/UV topology and full image resolution are preserved. Only positions,
    normals, tangents and winding affected by the affine are changed. WebP images
    are decoded to lossless PNG so UE 5.0 never needs EXT_texture_webp support.
    """
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc, binary, source_hash = read_glb(source)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("Source digest changed: " + str(source))
    matrix = np.asarray(world_from_scene, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Invalid binding affine")
    anchor = np.asarray(anchor_m, dtype=float)
    w = Writer(directory=destination.parent)
    for key in ("materials", "textures", "samplers"):
        if key in doc:
            w.doc[key] = copy.deepcopy(doc[key])
    for texture in w.doc.get("textures", []):
        webp = texture.get("extensions", {}).pop("EXT_texture_webp", None)
        if webp:
            texture["source"] = webp["source"]
        if not texture.get("extensions"):
            texture.pop("extensions", None)
    image_records = []
    for image in doc.get("images", []):
        if "bufferView" not in image:
            raise ValueError("External images are unsupported")
        view = doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        original = bytes(binary[start:start + view["byteLength"]])
        if image.get("mimeType") == "image/webp" or original[:4] == b"RIFF":
            with Image.open(io.BytesIO(original)) as decoded:
                image_records.append({"source_mode": decoded.mode, "size": list(decoded.size)})
                stream = io.BytesIO()
                decoded.save(stream, "PNG", compress_level=3)
                value, mime = stream.getvalue(), "image/png"
        else:
            value, mime = original, image["mimeType"]
        w.doc.setdefault("images", []).append({"bufferView": w.blob(value), "mimeType": mime})
    for key in ("extensionsUsed", "extensionsRequired"):
        value = [v for v in doc.get(key, []) if v != "EXT_texture_webp"]
        if value:
            w.doc[key] = value
    lows, highs, faces = [], [], 0
    for node_index, mesh_index, node_transform in scene_meshes(doc):
        combined = matrix @ node_transform
        linear = combined[:3, :3]
        determinant = float(np.linalg.det(linear))
        if abs(determinant) < 1e-15:
            raise ValueError("Singular mesh transform")
        normal_matrix = np.linalg.inv(linear).T
        mesh = {"name": "node_%04d_mesh_%04d" % (node_index, mesh_index), "primitives": []}
        for primitive in doc["meshes"][mesh_index]["primitives"]:
            if primitive.get("mode", 4) != 4 or primitive.get("targets") or primitive.get("extensions"):
                raise ValueError("Only uncompressed static triangle primitives are supported")
            p = {k: copy.deepcopy(v) for k, v in primitive.items() if k not in ("attributes", "indices")}
            p["attributes"] = {}
            for name, index in primitive["attributes"].items():
                array, descriptor = accessor(doc, binary, index)
                if name in ("POSITION", "NORMAL", "TANGENT"):
                    if descriptor["componentType"] != 5126 or descriptor.get("normalized"):
                        raise ValueError("Quantized geometry needs explicit decoding")
                    if name == "POSITION":
                        array = (array.astype(float) @ linear.T + combined[:3, 3] - anchor).astype("<f4")
                        lows.append(array.min(axis=0)); highs.append(array.max(axis=0))
                    else:
                        vector = array[:, :3].astype(float) @ (normal_matrix if name == "NORMAL" else linear).T
                        vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-30)
                        array[:, :3] = vector
                        if name == "TANGENT" and determinant < 0:
                            array[:, 3] *= -1
                p["attributes"][name] = w.array(array, descriptor)
            if "indices" in primitive:
                array, descriptor = accessor(doc, binary, primitive["indices"])
                if len(array) % 3:
                    raise ValueError("Invalid triangle indices")
                if determinant < 0:
                    array = array.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1, 1)
                p["indices"] = w.array(array, descriptor, 34963)
                faces += len(array) // 3
            else:
                if determinant < 0:
                    raise ValueError("Reflected non-indexed primitives require explicit indexing")
                faces += doc["accessors"][primitive["attributes"]["POSITION"]]["count"] // 3
            mesh["primitives"].append(p)
        w.doc["scenes"][0]["nodes"].append(len(w.doc["nodes"]))
        w.doc["nodes"].append({"mesh": len(w.doc["meshes"]), "name": mesh["name"]})
        w.doc["meshes"].append(mesh)
    if not lows:
        raise ValueError("Empty active glTF scene")
    destination.parent.mkdir(parents=True, exist_ok=True)
    w.save(destination)
    local_bounds = [np.min(lows, axis=0).tolist(), np.max(highs, axis=0).tolist()]
    return {"version": VERSION, "source_sha256": source_hash,
            "interchange_sha256": digest_file(destination), "interchange_bytes": destination.stat().st_size,
            "local_bounds_m": local_bounds, "world_bounds_m": (np.array(local_bounds) + anchor).tolist(),
            "mesh_count": len(w.doc["meshes"]), "face_count": faces,
            "image_count": len(w.doc.get("images", [])), "webp_decoded_to_png": image_records,
            "node_transforms_composed": True, "full_binding_baked": True}


def fixture(destination):
    w = Writer()
    positions = np.array([[x, y, z] for x in (1, 2) for y in (3, 5) for z in (7, 11)], dtype="<f4")
    indices = np.array([0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3, 0, 4, 5, 0, 5, 1,
                        2, 3, 7, 2, 7, 6, 0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5], dtype="<u4").reshape(-1, 1)
    a, b = w.array(positions), w.array(indices, {"componentType": 5125, "type": "SCALAR", "count": len(indices)}, 34963)
    w.doc["meshes"] = [{"name": "axis_fixture", "primitives": [{"attributes": {"POSITION": a}, "indices": b}]}]
    w.doc["nodes"] = [{"mesh": 0}]; w.doc["scenes"][0]["nodes"] = [0]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    w.save(destination)
    return {"path": str(destination), "bounds_m": [[1, 3, 7], [2, 5, 11]], "sha256": digest_file(destination)}
