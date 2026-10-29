#!/usr/bin/env python3
"""Audit actual source and prepared geometry using an independent trimesh reader.

This does not stand in for UE saved-map readback. The default audits every source;
--sample chooses representative building/massing/background inputs for a smoke
check. No geometry/material source is changed and no network is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def measure(path, outer):
    scene = trimesh.load_scene(path, process=False, skip_materials=True, allow_remote=False)
    low, high, faces, nodes = np.full(3, np.inf), np.full(3, -np.inf), 0, 0
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        matrix = outer @ transform
        # A filtered derivative can retain unused attribute rows. Geometry bounds
        # are defined by referenced triangles rather than orphaned accessors.
        vertices = np.asarray(mesh.vertices)[np.unique(mesh.faces)] @ matrix[:3, :3].T + matrix[:3, 3]
        low = np.minimum(low, vertices.min(axis=0)); high = np.maximum(high, vertices.max(axis=0))
        faces += len(mesh.faces); nodes += 1
    return np.stack([low, high]), faces, nodes


def measure_background(path, outer, root):
    # The release's separate ground-sampling reader memory-maps the 15M-triangle
    # background. Using trimesh here would materialize its complete scene twice.
    sys.path.insert(0, str(root/"src"))
    from citycode_scene.ground import _BackgroundGLB, DEFAULT_ROLES
    scene = _BackgroundGLB(path, outer, DEFAULT_ROLES)
    low, high, faces = np.full(3, np.inf), np.full(3, -np.inf), 0
    try:
        for instance in scene.instances:
            for vertices, triangles in scene.primitives(instance):
                if not len(triangles):
                    continue
                used = vertices[np.unique(triangles)]
                low = np.minimum(low, used.min(axis=0)); high = np.maximum(high, used.max(axis=0))
                faces += len(triangles)
        return np.stack([low, high]), faces, len(scene.instances)
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()
    root = args.root.resolve(); path = args.manifest or root/"build/ue_citycode/assembly_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig")); rows = manifest["meshes"]
    manifest_digest = digest(path)
    if args.sample:
        chosen = np.linspace(0, len(rows)-1, min(args.sample, len(rows))).astype(int).tolist()
        chosen += [i for i, row in enumerate(rows) if row["kind"] == "massing"][:2]
        rows = [rows[i] for i in sorted(set(chosen))]
    results = []
    for index, row in enumerate(rows):
        source, prepared = root/row["source_file"], path.parent/row["import_file"]
        issues = []
        if digest(source) != row["source_sha256"]:
            issues.append("source_hash_mismatch")
        reader = (lambda file, transform: measure_background(file, transform, root)) if row["kind"] == "background" else measure
        if row.get("import_parts"):
            derivative = row["background_derivative"]
            if digest(root/derivative["file"]) != derivative["sha256"]:
                issues.append("background_derivative_hash_mismatch")
            source_bounds, source_faces, source_nodes = reader(root/derivative["file"], np.eye(4))
            low, high, target_faces, target_nodes = np.full(3, np.inf), np.full(3, -np.inf), 0, 0
            for part in row["import_parts"]:
                file = path.parent/part["file"]
                if digest(file) != part["sha256"]:
                    issues.append("background_part_hash_mismatch:"+part["file"])
                box, count, nodes = reader(file, np.eye(4))
                if count != part["face_count"] or nodes != part["mesh_count"]:
                    issues.append("background_part_count_mismatch:"+part["file"])
                low, high = np.minimum(low, box[0]), np.maximum(high, box[1])
                target_faces += count; target_nodes += nodes
            target_bounds = np.stack([low, high])
        else:
            if digest(prepared) != row["prepared"]["interchange_sha256"]:
                issues.append("interchange_hash_mismatch")
            source_bounds, source_faces, source_nodes = reader(source, np.asarray(row["world_from_scene_4x4"]))
            anchor = np.eye(4); anchor[:3, 3] = row["anchor_m"]
            target_bounds, target_faces, target_nodes = reader(prepared, anchor)
        delta = float(np.max(np.abs(source_bounds-target_bounds)))
        if delta > .0003:
            issues.append("geometry_bounds_changed_over_0.3mm")
        if source_faces != target_faces or target_faces != row["prepared"]["face_count"]:
            issues.append("triangle_count_changed")
        if np.max(np.abs(target_bounds-np.asarray(row["prepared"]["world_bounds_m"]))) > .0003:
            issues.append("proof_bounds_mismatch")
        results.append({"id": row["id"], "kind": row["kind"], "issues": issues,
            "maximum_bounds_error_m": delta, "face_count": target_faces,
            "independent_reader": "release_ground_mmap" if row["kind"] == "background" else "trimesh",
            "source_geometry_node_count": source_nodes, "interchange_geometry_node_count": target_nodes})
        if (index+1) % 25 == 0 or index+1 == len(rows):
            print("Independent CPU geometry audit %d/%d" % (index+1, len(rows)), flush=True)
    if digest(path) != manifest_digest:
        raise ValueError("Assembly manifest changed during independent geometry audit")
    result = {"schema": "citycode.ue_cpu_geometry_audit.v1", "status": "passed" if all(not r["issues"] for r in results) else "failed",
        "assembly_manifest_sha256": manifest_digest, "source_binding_sha256": manifest["source_binding_sha256"],
        "scope": "independent trimesh/release_ground_mmap geometry and hash audit; this command does not execute UE",
        "ue_executed": False, "sampled": bool(args.sample), "checked": len(results), "total_manifest_meshes": len(manifest["meshes"]),
        "maximum_bounds_error_m": max(r["maximum_bounds_error_m"] for r in results), "records": results}
    output = args.output or path.parent/"independent_cpu_validation.json"
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"output": str(output), "status": result["status"], "checked": len(results), "maximum_bounds_error_m": result["maximum_bounds_error_m"]}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
