#!/usr/bin/env python3
"""Split a baked UE-only background into small self-contained GLBs on CPU.

PNG bytes, materials and all retained triangles remain unchanged. Each output
contains only the textures it uses; unused vertices after index filtering are
removed together with matching normal/UV rows. No images are decoded and no
positions receive another coordinate conversion. This avoids multi-GB importer
input allocations in UE 5.0 and permits upload/import to overlap by tile.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from interchange import Writer, accessor, digest_file, read_glb


def texture_ids(value):
    result = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower().endswith("texture") and isinstance(child, dict) and "index" in child:
                result.add(child["index"])
            result.update(texture_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(texture_ids(child))
    return result


def remap_texture_ids(value, mapping):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower().endswith("texture") and isinstance(child, dict) and "index" in child:
                child["index"] = mapping[child["index"]]
            remap_texture_ids(child, mapping)
    elif isinstance(value, list):
        for child in value:
            remap_texture_ids(child, mapping)


def used_refs(doc, mesh_ids):
    materials, accessors = set(), set()
    for index in mesh_ids:
        for primitive in doc["meshes"][index]["primitives"]:
            count = doc["accessors"][primitive.get("indices", primitive["attributes"]["POSITION"])]["count"]
            if count == 0:
                continue
            if "material" in primitive:
                materials.add(primitive["material"])
            accessors.update(primitive["attributes"].values())
            if "indices" in primitive:
                accessors.add(primitive["indices"])
    textures = set()
    for index in materials:
        textures.update(texture_ids(doc["materials"][index]))
    images = {doc["textures"][i]["source"] for i in textures}
    image_views = {doc["images"][i]["bufferView"] for i in images}
    all_views = image_views | {doc["accessors"][i]["bufferView"] for i in accessors}
    return materials, textures, images, image_views, all_views


def split(source, output_directory, manifest_path, *, max_bytes=512*1024*1024, filter_proof=None):
    source, output_directory, manifest_path = Path(source), Path(output_directory), Path(manifest_path)
    output_directory.mkdir(parents=True, exist_ok=True); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    doc, binary, source_digest = read_glb(source)
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if old.get("source_interchange_sha256") == source_digest and old.get("maximum_part_bytes") == max_bytes:
            if all((manifest_path.parent/p["file"]).is_file() and digest_file(manifest_path.parent/p["file"]) == p["sha256"] for p in old.get("import_parts", [])):
                return old
    nodes = []
    for index in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        node = doc["nodes"][index]
        if "mesh" not in node or any(key in node for key in ("children", "translation", "rotation", "scale", "matrix")):
            raise ValueError("Background must already be baked with flat identity mesh nodes")
        mesh = doc["meshes"][node["mesh"]]
        if any(doc["accessors"][p.get("indices", p["attributes"]["POSITION"])]["count"] for p in mesh["primitives"]):
            nodes.append(node)
    groups, current, current_views = [], [], set()
    for node in nodes:
        refs = used_refs(doc, [node["mesh"]]); views = refs[4]
        combined = current_views | views
        # Reserve metadata and a possible 16->32 bit index expansion. Images are
        # copied unchanged; this conservative estimate usually stays below the
        # requested byte budget after unused vertices are removed.
        image_set = used_refs(doc, [n["mesh"] for n in current] + [node["mesh"]])[3]
        size = sum(doc["bufferViews"][i]["byteLength"]*(1 if i in image_set else 2) for i in combined) + 1024*1024
        if current and size > max_bytes:
            groups.append(current); current, current_views = [], set()
        current.append(node); current_views.update(views)
    if current:
        groups.append(current)
    parts = []
    global_low, global_high = np.full(3, np.inf), np.full(3, -np.inf)
    for part_index, group in enumerate(groups):
        materials, textures, images, image_views, _ = used_refs(doc, [n["mesh"] for n in group])
        material_map = {old: new for new, old in enumerate(sorted(materials))}
        texture_map = {old: new for new, old in enumerate(sorted(textures))}
        image_map = {old: new for new, old in enumerate(sorted(images))}
        writer = Writer(directory=output_directory)
        for key in ("extensionsUsed", "extensionsRequired", "samplers"):
            if key in doc:
                writer.doc[key] = copy.deepcopy(doc[key])
        if materials:
            writer.doc["materials"] = [copy.deepcopy(doc["materials"][i]) for i in sorted(materials)]
        for material in writer.doc.get("materials", []):
            remap_texture_ids(material, texture_map)
        if textures:
            writer.doc["textures"] = [copy.deepcopy(doc["textures"][i]) for i in sorted(textures)]
        for texture in writer.doc.get("textures", []):
            texture["source"] = image_map[texture["source"]]
        view_map = {}
        for index in sorted(image_views):
            view = doc["bufferViews"][index]; start = view.get("byteOffset", 0)
            view_map[index] = writer.blob(binary[start:start+view["byteLength"]])
        if images:
            writer.doc["images"] = [copy.deepcopy(doc["images"][i]) for i in sorted(images)]
        for image in writer.doc.get("images", []):
            image["bufferView"] = view_map[image["bufferView"]]
        low, high, face_count, vertex_count = np.full(3, np.inf), np.full(3, -np.inf), 0, 0
        for node in group:
            original = doc["meshes"][node["mesh"]]
            mesh = {"name": original.get("name", "background_mesh"), "primitives": []}
            for primitive in original["primitives"]:
                if primitive.get("mode", 4) != 4:
                    raise ValueError("Only triangle primitives are supported")
                if "indices" in primitive:
                    indices, _ = accessor(doc, binary, primitive["indices"]); indices = indices.reshape(-1)
                else:
                    indices = np.arange(doc["accessors"][primitive["attributes"]["POSITION"]]["count"])
                if not len(indices):
                    continue
                if len(indices) % 3:
                    raise ValueError("Invalid filtered triangle indices")
                used, inverse = np.unique(indices, return_inverse=True)
                item = {k: copy.deepcopy(v) for k, v in primitive.items() if k not in ("indices", "attributes")}
                item["attributes"] = {}
                if "material" in item:
                    item["material"] = material_map[item["material"]]
                for name, index in primitive["attributes"].items():
                    values, descriptor = accessor(doc, binary, index)
                    if used[-1] >= len(values):
                        raise ValueError("Filtered indices exceed attribute bounds")
                    values = values[used]
                    descriptor = {**descriptor, "count": len(values)}
                    item["attributes"][name] = writer.array(values, descriptor)
                    if name == "POSITION":
                        low = np.minimum(low, values.min(axis=0)); high = np.maximum(high, values.max(axis=0))
                        vertex_count += len(values)
                item["indices"] = writer.array(inverse.astype("<u4").reshape(-1, 1),
                    {"componentType": 5125, "type": "SCALAR", "count": len(inverse)}, 34963)
                face_count += len(inverse)//3; mesh["primitives"].append(item)
            if mesh["primitives"]:
                writer.doc["scenes"][0]["nodes"].append(len(writer.doc["nodes"]))
                writer.doc["nodes"].append({"name": node.get("name", mesh["name"]), "mesh": len(writer.doc["meshes"])})
                writer.doc["meshes"].append(mesh)
        filename = output_directory / ("BACKGROUND_GROUND_%03d.glb" % part_index)
        writer.save(filename)
        if filename.stat().st_size > 2*1024**3-1:
            raise ValueError("A single mesh/material exceeds the UE5.0 2GB import allocation safety limit")
        global_low = np.minimum(global_low, low); global_high = np.maximum(global_high, high)
        parts.append({"file": filename.relative_to(manifest_path.parent).as_posix(), "sha256": digest_file(filename),
            "bytes": filename.stat().st_size, "mesh_count": len(writer.doc["meshes"]), "face_count": face_count,
            "vertex_count": vertex_count, "local_bounds_m": [low.tolist(), high.tolist()]})
        print("Prepared background import part %d/%d (%d bytes)" % (part_index+1, len(groups), filename.stat().st_size), flush=True)
    result = {"schema": "citycode.ue_background_import_parts.v1", "source_interchange_sha256": source_digest,
        "source_interchange_file": str(source), "maximum_part_bytes": max_bytes, "part_count": len(parts),
        "mesh_count": sum(p["mesh_count"] for p in parts), "face_count": sum(p["face_count"] for p in parts),
        "world_bounds_m": [global_low.tolist(), global_high.tolist()], "import_parts": parts,
        "coordinate_contract": "already baked local UE XYZ metres; anchor zero; no additional swaps or binding",
        "image_processing": "raw PNG bufferView bytes copied unchanged; only referenced images/materials included",
        "geometry_processing": "all retained triangles preserved; unused attribute rows removed and indices remapped",
        "filter_provenance": json.loads(Path(filter_proof).read_text(encoding="utf-8-sig")) if filter_proof else None,
        "filter_proof_sha256": digest_file(filter_proof) if filter_proof else None, "ue_executed": False}
    manifest_path.write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--filter-proof", type=Path)
    parser.add_argument("--max-mb", type=float, default=512)
    args = parser.parse_args()
    result = split(args.input.resolve(), args.output_directory.resolve(), args.manifest.resolve(),
                   max_bytes=int(args.max_mb*1024**2), filter_proof=args.filter_proof)
    print(json.dumps({k: result[k] for k in ("part_count", "mesh_count", "face_count", "world_bounds_m")}))


if __name__ == "__main__":
    main()
