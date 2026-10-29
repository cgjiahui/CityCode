"""Read prepared GLB normal requirements without loading geometry or Unreal.

The UE 5.0 importer names each prepared mesh ``<index>_<glTF mesh name>``.
Prepared CityCode GLBs have unique ASCII mesh names and one active node per mesh.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import struct


def read_requirements(filename):
    path = Path(filename)
    with path.open("rb") as stream:
        header = stream.read(20)
        if len(header) != 20:
            raise ValueError("Truncated GLB normal-policy input")
        magic, version, total, length, kind = struct.unpack("<4sIII4s", header)
        if magic != b"glTF" or version != 2 or total != path.stat().st_size or kind != b"JSON" or length > total-20:
            raise ValueError("Invalid GLB normal-policy input")
        doc = json.loads(stream.read(length))
    result = {}
    for index, mesh in enumerate(doc.get("meshes", [])):
        name = mesh.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("Prepared GLB mesh name cannot be matched unambiguously: " + name)
        primitives = mesh.get("primitives", [])
        if not primitives:
            raise ValueError("Prepared mesh has no primitives")
        missing = [i for i, primitive in enumerate(primitives) if "NORMAL" not in primitive.get("attributes", {})]
        result[str(index) + "_" + name] = {"source_mesh_index": index, "source_mesh_name": name,
            "primitive_count": len(primitives), "missing_normal_primitives": missing,
            "recompute_normals_required": bool(missing)}
    if not result:
        raise ValueError("Prepared GLB contains no mesh normal requirements")
    return result


def match_requirements(requirements, mesh_paths):
    result = {}
    for path in mesh_paths:
        if path in result:
            raise ValueError("Imported mesh cannot be matched to its GLB normal requirements: " + path)
        result[path] = mesh_requirement(requirements, path)
    if len(result) != len(requirements) or len({row["source_mesh_index"] for row in result.values()}) != len(requirements):
        raise ValueError("Imported mesh count/identity differs from GLB normal requirements")
    return result


def mesh_requirement(requirements, path):
    name = path.rsplit("/", 1)[-1].split(".", 1)[0]
    if name not in requirements:
        raise ValueError("Imported mesh cannot be matched to its GLB normal requirements: " + path)
    return requirements[name]
