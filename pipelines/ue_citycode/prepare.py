#!/usr/bin/env python3
"""Prepare a portable, explicit UE assembly manifest from the frozen city.

Examples:
  python pipelines/ue_citycode/prepare.py --bake --workers 3
  python pipelines/ue_citycode/prepare.py --proposal data/.../placement_proposal.json --catalog catalog.json --asset-bindings bindings.json --bake

Only the CPU interchange copies are rewritten. Official binding/assets are read
only. Unbound proposals remain visible in the report and block full UE assembly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from interchange import VERSION, bake, digest_file, fixture

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTENT_ROOT = "/Game/Research/CityCodeOpenSource"
BACKGROUND_TO_UE = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value))


def source_reference(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        return path.relative_to(root).as_posix(), "release_root"
    except ValueError:
        return str(path), "absolute_external_input"


def select_lod_sources(binding, binding_path, lod):
    """Validate every selected asset/donor before preparing any geometry.

    The requested LOD is exact. A missing 1M export is reported for each enabled
    row, never substituted with a lower-resolution resource.
    """
    if lod not in ("0.25M", "0.1M", "1M"):
        raise ValueError("Unsupported requested LOD: " + str(lod))
    selected, issues = [], []
    groups = (("building", binding.get("asset_instances", [])), ("massing", binding.get("massing_instances", [])),
        ("surface", [row for row in binding.get("surface_instances", []) if row.get("assembly_enabled", False)]))
    for kind, rows in groups:
        for row in rows:
            if not row.get("assembly_enabled", True):
                continue
            label = "%s %s" % (kind, row.get("id", "<missing-id>"))
            available = row.get("lods") or row.get("donor_lods", {})
            detail = available.get(lod)
            if not isinstance(detail, dict):
                issues.append(label + ": missing " + lod + " entry; available=" + ",".join(sorted(available)))
                continue
            if not isinstance(detail.get("path"), str) or not detail["path"]:
                issues.append(label + ": " + lod + " entry has no resource path")
                continue
            source = (Path(binding_path).parent/detail["path"]).resolve()
            if not source.is_file():
                issues.append(label + ": missing " + lod + " resource file: " + str(source))
                continue
            if not isinstance(detail.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", detail["sha256"]):
                issues.append(label + ": " + lod + " entry has no valid SHA256")
                continue
            selected.append((kind, row, detail, source))
    if issues:
        raise ValueError("Requested LOD %s is incomplete for %d enabled asset/donor rows; no fallback is permitted:\n%s" %
            (lod, len(issues), "\n".join("- " + issue for issue in issues)))
    if not selected:
        raise ValueError("Binding contains no enabled asset/donor rows")
    return selected


def finite_vector(value, count, label):
    if not isinstance(value, (list, tuple)) or len(value) != count or not all(math.isfinite(float(x)) for x in value):
        raise ValueError("Expected finite %s-vector: %s" % (count, label))
    return [float(x) for x in value]


def rotation_matrix(degrees):
    pitch, yaw, roll = np.radians(degrees)
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, -cr*sp*cy-sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, -cr*sp*sy+sr*cy],
                     [sp, -sr*cp, cr*cp]])


def quaternion(matrix):
    # Stable rotation-matrix conversion without SciPy (UE's Quat is XYZW).
    m = np.asarray(matrix, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [(m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(m))); j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(max(0, 1 + m[i, i] - m[j, j] - m[k, k])) * 2
        q = [0., 0., 0., 0.]; q[i] = s/4
        q[j] = (m[j, i]+m[i, j])/s; q[k] = (m[k, i]+m[i, k])/s
        q[3] = (m[k, j]-m[j, k])/s
    return (np.asarray(q) / np.linalg.norm(q)).tolist()


def catalog_rows(path):
    if not path:
        return {}
    payload = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix == ".jsonl":
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    else:
        doc = json.loads(payload)
        rows = doc if isinstance(doc, list) else doc.get("assets", doc.get("records", []))
    result = {}
    for row in rows:
        row = dict(row)
        if row.get("mesh") and not row.get("mesh_path"):
            row["mesh_path"] = row["mesh"]
        if "bounds_cm" not in row and "bounds_min_m" in row and "bounds_max_m" in row:
            row["bounds_cm"] = [[float(x)*100 for x in row[name]] for name in ("bounds_min_m", "bounds_max_m")]
        for key in (row.get("asset_id"), row.get("id"), row.get("mesh_path"), row.get("object_path")):
            if key:
                result[str(key)] = row
    return result


def bind_placement(row, catalog, bindings):
    row = dict(row)
    category = row.get("category", "prop")
    identity = row.get("placement_id", row.get("id"))
    if not identity:
        raise ValueError("Proposal placement lacks stable placement_id")
    if row.get("status", "").startswith("withheld") or row.get("status") == "junction_requires_connection_solution":
        return None, {"placement_id": identity, "status": "withheld", "reason": row.get("reason", row.get("status"))}
    if category in ("bridge_deck", "elevated_deck", "predicted_bridge_span", "bridge_support", "bridge_railing"):
        if row.get("placement_source") != "predicted_elevated_structure":
            raise ValueError("Bridge must be derived from predicted_elevated_structure: " + identity)
        forbidden = {"original_actor_transform", "gt_actor_transform", "source_actor_world_transform"}
        if forbidden.intersection(row):
            raise ValueError("Original bridge actor world transform is prohibited: " + identity)
    selected = bindings.get(row.get("asset_id"), bindings.get(category, {})) or {}
    if isinstance(selected, str):
        selected = {"mesh_path": selected}
    asset = catalog.get(selected.get("mesh_path", selected.get("asset_id", row.get("mesh_path", row.get("asset_id")))), {})
    mesh_path = selected.get("mesh_path", row.get("mesh_path", asset.get("mesh_path", asset.get("object_path"))))
    if not mesh_path:
        return None, {"placement_id": identity, "category": category, "asset_id": row.get("asset_id"), "status": "requires_asset_binding"}
    if not mesh_path.startswith("/Game/"):
        raise ValueError("Project mesh path must begin /Game/: " + mesh_path)
    if not asset:
        return None, {"placement_id": identity, "category": category, "mesh_path": mesh_path, "status": "mesh_not_verified_in_ue_catalog"}
    if category == "predicted_bridge_span" and row.get("prepared_world_geometry") is not True:
        raise ValueError("Native deformed span must declare prepared_world_geometry: " + identity)
    position = np.array(finite_vector(row.get("position_m"), 3, identity + ".position_m"))
    scale = np.array(finite_vector(row.get("scale", [1, 1, 1]), 3, identity + ".scale"))
    if np.any(scale <= 0):
        raise ValueError("Proposal scale must be positive")
    rotation = rotation_matrix(finite_vector(row.get("rotation_deg", [0, 0, 0]), 3, identity + ".rotation_deg"))
    pivot_rule = "proposal_position_is_mesh_pivot"
    if category in ("bridge_deck", "elevated_deck"):
        bounds = asset.get("bounds_cm")
        if not bounds:
            return None, {"placement_id": identity, "status": "bridge_catalog_bounds_missing"}
        bounds = np.asarray(bounds, dtype=float)
        axes = {**asset.get("local_axes", {}), **selected.get("local_axes", {})}
        length_axis = axes.get("length_axis")
        if length_axis not in ("X", "Y"):
            return None, {"placement_id": identity, "status": "bridge_local_length_axis_requires_confirmation"}
        if not selected.get("orientation_confirmed", asset.get("orientation_confirmed", False)):
            return None, {"placement_id": identity, "status": "bridge_orientation_requires_confirmation", "mesh_path": mesh_path}
        deck_top_cm = selected.get("deck_top_local_z_cm", asset.get("deck_top_local_z_cm"))
        if deck_top_cm is None or not math.isfinite(float(deck_top_cm)):
            return None, {"placement_id": identity, "status": "bridge_driving_surface_local_z_required", "mesh_path": mesh_path,
                          "reason": "Bounding-box max Z can be the top of railings; it is not assumed to be the road surface"}
        start = np.array(finite_vector(row.get("start_m"), 3, identity + ".start_m"))
        end = np.array(finite_vector(row.get("end_m"), 3, identity + ".end_m"))
        direction = end-start; length = float(np.linalg.norm(direction))
        if length < .01 or np.linalg.norm(direction[:2]) < .01:
            raise ValueError("Degenerate bridge segment: " + identity)
        forward = direction/length
        side = np.array([-forward[1], forward[0], 0.]); side /= np.linalg.norm(side)
        up = np.cross(forward, side)
        rotation = np.column_stack((forward, side, up)) if length_axis == "X" else np.column_stack((-side, forward, up))
        long_axis, short_axis = (0, 1) if length_axis == "X" else (1, 0)
        dimensions_m = (bounds[1]-bounds[0])/100
        if np.any(dimensions_m <= 0):
            raise ValueError("Invalid catalog mesh bounds")
        width = float(row["width_m"])
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Invalid bridge width")
        scale[long_axis] = length/dimensions_m[long_axis]
        scale[short_axis] = width/dimensions_m[short_axis]
        if not float(selected.get("minimum_width_scale", .5)) <= scale[short_axis] <= float(selected.get("maximum_width_scale", 1.5)):
            return None, {"placement_id": identity, "status": "bridge_requires_width_tiling_or_another_template", "mesh_path": mesh_path,
                          "requested_width_m": width, "template_width_m": float(dimensions_m[short_axis]), "required_width_scale": float(scale[short_axis])}
        if not float(selected.get("minimum_length_scale", .25)) <= scale[long_axis] <= float(selected.get("maximum_length_scale", 2.0)):
            return None, {"placement_id": identity, "status": "bridge_requires_length_tiling_or_another_template", "mesh_path": mesh_path,
                          "requested_length_m": length, "template_length_m": float(dimensions_m[long_axis]), "required_length_scale": float(scale[long_axis])}
        # The proposal is the predicted deck TOP centre. Source mesh pivot stays
        # local to the mesh and is corrected without consulting any placed actor.
        local_top = np.array([(bounds[0, 0]+bounds[1, 0])/200,
                              (bounds[0, 1]+bounds[1, 1])/200,
                              float(deck_top_cm)/100])
        position = position - rotation @ (scale * local_top)
        pivot_rule = "predicted_deck_top_center_minus_rotated_scaled_local_mesh_top"
    bound = {"id": identity, "kind": category, "mesh_path": mesh_path,
            "position_cm": (position*100).tolist(), "rotation_quat_xyzw": quaternion(rotation),
            "scale": scale.tolist(), "placement_source": row.get("placement_source", "rule_based_road_semantics"),
            "source_segment_id": row.get("source_segment_id"), "pivot_rule": pivot_rule,
            "proposal_position_m": row["position_m"], "catalog_mesh_verified": True}
    if row.get("group_id"):
        bound.update(group_id=row["group_id"], part_index=row.get("part_index"), source_asset_id=row.get("asset_id"))
    if category == "predicted_bridge_span":
        bound.update(prepared_world_geometry=True, logical_module_count=row["logical_module_count"],
            expected_native_bounds_m=row["expected_native_bounds_m"], native_material_paths=row["native_material_paths"],
            template_sha256=row["template_sha256"], inference=row.get("inference"))
    return bound, None


def bake_job(job):
    record, root, output = job
    destination = output / record["import_file"]
    proof_path = destination.with_suffix(".proof.json")
    key = hashlib.sha256(json.dumps({"version": VERSION, "source_sha256": record["source_sha256"],
        "matrix": record["world_from_scene_4x4"], "anchor": record["anchor_m"]}, sort_keys=True).encode()).hexdigest()
    if destination.is_file() and proof_path.is_file():
        proof = json.loads(proof_path.read_text(encoding="utf-8-sig"))
        if proof.get("cache_key") == key and proof.get("interchange_sha256") == digest_file(destination):
            return record["id"], proof
    proof = bake(root / record["source_file"], destination, record["world_from_scene_4x4"], record["anchor_m"], record["source_sha256"])
    proof["cache_key"] = key
    write_json(proof_path, proof)
    return record["id"], proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binding", type=Path, help="Physical binding manifest; relative resource paths resolve beside this file")
    parser.add_argument("--lod", choices=["0.25M", "0.1M", "1M"], help="Exact LOD; 1M requires every enabled asset/donor export and never falls back")
    parser.add_argument("--proposal", type=Path, action="append", default=[])
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--asset-bindings", type=Path)
    parser.add_argument("--background-parts", type=Path, help="Explicit split/filtered UE-only background manifest; formal source background remains unchanged")
    parser.add_argument("--bake", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Prepare only this many meshes; resulting manifest remains explicitly incomplete")
    parser.add_argument("--only-id", action="append", default=[], help="Explicit diagnostic subset (repeat for multiple ids); never presented as the full city")
    parser.add_argument("--without-background", action="store_true", help="Explicit building-only diagnostic manifest")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "build/ue_citycode").resolve(); output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "workflow/04_citycode_assembly/standard.json").read_text(encoding="utf-8-sig"))
    binding_path = (args.binding or root / config["inputs"]["building_binding"]).resolve()
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    binding_digest = hashlib.sha256(binding_bytes).hexdigest()
    lod = args.lod or config["default_scene"]["building_lod"]
    selected_sources = select_lod_sources(binding, binding_path, lod)
    records = []
    entity_for_asset = {row["id"]: row.get("building_entity_id", row["id"]) for row in binding["asset_instances"]}
    for kind, row, detail, source in selected_sources:
        matrix = np.asarray(row["world_from_local_4x4"], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Invalid binding: " + row["id"])
        entity_id = row.get("building_entity_id")
        if not entity_id:
            parent_id = row.get("parent_building_id")
            entity_id = entity_for_asset.get(parent_id, parent_id)
        if not entity_id and kind in ("building", "massing"):
            entity_id = "MASSING_BUILDING_" + row["source_building_ids"][0]
        source_file, source_base = source_reference(source, root)
        records.append({"id": row["id"], "kind": kind, "source_file": source_file, "source_file_path_base": source_base,
            "source_sha256": detail["sha256"].lower(), "source_bytes": source.stat().st_size,
            "world_from_scene_4x4": matrix.tolist(), "anchor_m": matrix[:3, 3].tolist(),
            "import_file": "interchange/" + safe_name(row["id"]) + ".glb", "nanite": True,
            "source_building_ids": row.get("source_building_ids", []), "building_entity_id": entity_id,
            "placement_source": "formal_physical_binding", "source_node_transforms": "composed_before_binding"})
    if not args.without_background:
        source = root / config["inputs"]["background_full"]
        records.append({"id": "BACKGROUND_COASTAL", "kind": "background", "source_file": source.relative_to(root).as_posix(),
            "source_sha256": digest_file(source), "source_bytes": source.stat().st_size,
            "world_from_scene_4x4": BACKGROUND_TO_UE, "anchor_m": [0, 0, 0],
            "import_file": "interchange/BACKGROUND_COASTAL.glb", "nanite": False,
            "placement_source": "formal_coastal_background", "background_yz_swap_count": 1})
    if args.only_id:
        unknown = set(args.only_id)-{r["id"] for r in records}
        if unknown:
            raise ValueError("Unknown diagnostic subset ids: " + str(sorted(unknown)))
        records = [r for r in records if r["id"] in args.only_id]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate mesh instance ids")
    catalog = catalog_rows(args.catalog)
    bindings = json.loads(args.asset_bindings.read_text(encoding="utf-8-sig")) if args.asset_bindings else {}
    placements, decisions, proposal_sources, proposal_categories = [], [], [], set()
    for path in args.proposal:
        proposal = json.loads(path.read_text(encoding="utf-8-sig"))
        coordinate = proposal.get("coordinate_system", {})
        if not isinstance(coordinate, dict) or coordinate.get("units") != "m" or coordinate.get("up") != "Z" or coordinate.get("world") != "UE":
            raise ValueError("Proposal must declare coordinate_system {world:UE,units:m,up:Z}")
        source_path = path.resolve()
        try:
            source_name, path_base = source_path.relative_to(root).as_posix(), "release_root"
        except ValueError:
            source_name, path_base = str(source_path), "absolute_external_input"
        source_record = {"file": source_name, "path_base": path_base, "sha256": digest_file(path), "schema": proposal.get("schema")}
        if proposal.get("agentic_review"):
            source_record["agentic_review"] = proposal["agentic_review"]
            decision_file = source_path.parent / proposal["agentic_review"]["decisions_file"]
            source_record["review_decisions_sha256"] = digest_file(decision_file)
            source_record["review_decisions_file"] = decision_file.relative_to(root).as_posix() if path_base == "release_root" else str(decision_file)
        proposal_sources.append(source_record)
        for row in proposal.get("withheld_placements", proposal.get("withheld", [])):
            decisions.append({"placement_id": row.get("placement_id", row.get("id", row.get("station_id"))), "status": "withheld",
                              "reason": row.get("reason", row.get("status", "withheld_by_prediction_uncertainty"))})
        proposal_rows = proposal.get("placements", proposal.get("rows"))
        if not isinstance(proposal_rows, list):
            raise ValueError("Proposal requires a placements/rows list")
        for row in proposal_rows:
            proposal_categories.add(row.get("category"))
            placement, decision = bind_placement(row, catalog, bindings)
            if placement:
                placements.append(placement)
            if decision:
                decisions.append(decision)
    if "predicted_bridge_span" in proposal_categories and proposal_categories.intersection({"bridge_deck", "elevated_deck"}):
        raise ValueError("Native continuous spans replace abstract bridge modules; do not include both proposals")
    placement_ids = [p["id"] for p in placements]
    if len(set(ids + placement_ids)) != len(ids + placement_ids):
        raise ValueError("Duplicate placement ids")
    if args.bake:
        selected = records[:args.limit] if args.limit is not None else records
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            tasks = [pool.submit(bake_job, (record, root, output)) for record in selected]
            proofs = {}
            for task in as_completed(tasks):
                identity, proof = task.result(); proofs[identity] = proof
                if len(proofs) % 25 == 0 or len(proofs) == len(selected):
                    print("Prepared %d/%d interchange files" % (len(proofs), len(selected)), flush=True)
        for record in records:
            if record["id"] in proofs:
                record["prepared"] = proofs[record["id"]]
    else:
        # A second manifest/binding pass can reuse already prepared files.
        for record in records:
            path = output / record["import_file"]
            if path.is_file() and path.with_suffix(".proof.json").is_file():
                proof = json.loads(path.with_suffix(".proof.json").read_text(encoding="utf-8-sig"))
                key = hashlib.sha256(json.dumps({"version": VERSION, "source_sha256": record["source_sha256"],
                    "matrix": record["world_from_scene_4x4"], "anchor": record["anchor_m"]}, sort_keys=True).encode()).hexdigest()
                if proof.get("cache_key") == key and proof.get("interchange_sha256") == digest_file(path):
                    record["prepared"] = proof
    background_parts_digest = None
    if args.background_parts:
        part_path = args.background_parts.resolve()
        part_manifest = json.loads(part_path.read_text(encoding="utf-8-sig"))
        if part_manifest.get("schema") != "citycode.ue_background_import_parts.v1":
            raise ValueError("Unsupported background-parts manifest")
        background = next((r for r in records if r["kind"] == "background"), None)
        if background is None or "prepared" not in background:
            raise ValueError("Split/filtered background requires the original prepared background provenance")
        background["prepared_before_background_filter"] = background["prepared"]
        background["import_parts"] = [{**part, "file": Path(os.path.relpath((part_path.parent/part["file"]).resolve(), output)).as_posix()}
                                      for part in part_manifest["import_parts"]]
        derivative = Path(part_manifest["source_interchange_file"]).resolve()
        background["background_derivative"] = {"file": derivative.relative_to(root).as_posix(),
            "sha256": part_manifest["source_interchange_sha256"], "filter_provenance": part_manifest.get("filter_provenance"),
            "filter_proof_sha256": part_manifest.get("filter_proof_sha256")}
        background_parts_digest = digest_file(part_path)
        background["prepared"] = {**background["prepared"], "interchange_sha256": part_manifest["source_interchange_sha256"],
            "interchange_bytes": sum(p["bytes"] for p in part_manifest["import_parts"]), "mesh_count": part_manifest["mesh_count"],
            "face_count": part_manifest["face_count"], "world_bounds_m": part_manifest["world_bounds_m"],
            "local_bounds_m": part_manifest["world_bounds_m"], "background_parts_manifest_sha256": background_parts_digest,
            "geometry_contract": "UE-only declared background derivative, split without changing retained triangles/materials/images"}
    calibration = fixture(output / "interchange/axis_fixture.glb"); calibration["path"] = "interchange/axis_fixture.glb"
    if digest_file(binding_path) != binding_digest:
        raise ValueError("Formal binding changed while preparing the handoff; retry from its frozen version")
    fingerprint = hashlib.sha256(json.dumps({"binding": binding_digest, "lod": lod,
        "sources": [p["sha256"] for p in proposal_sources], "placements": placements, "background": not args.without_background,
        "diagnostic_subset_ids": args.only_id, "background_parts_sha256": background_parts_digest}, sort_keys=True).encode()).hexdigest()[:12]
    missing = [r["id"] for r in records if "prepared" not in r]
    unresolved = [r for r in decisions if r["status"] != "withheld"]
    binding_reference, binding_base = source_reference(binding_path, root)
    result = {"schema": "citycode.ue_citycode_assembly.v1", "status": "cpu_ready_ue_execution_pending" if not missing and not unresolved else "preparation_incomplete",
        "engine_target": "5.0.3", "source_binding": binding_reference, "source_binding_path_base": binding_base, "source_binding_sha256": binding_digest,
        "path_contract": {"source_file_and_proposal_sources": "CityCode release root unless the input path_base declares absolute_external_input",
            "import_file_and_import_parts": "assembly manifest directory"},
        "catalog_sha256": digest_file(args.catalog) if args.catalog else None,
        "asset_bindings_sha256": digest_file(args.asset_bindings) if args.asset_bindings else None,
        "building_lod": lod, "content_root": CONTENT_ROOT, "scene_id": fingerprint,
        "diagnostic_subset_ids": args.only_id,
        "new_level_path": CONTENT_ROOT + "/Maps/CityCode_" + fingerprint,
        "coordinate_contract": {"baked_geometry": "local UE XYZ metres; original glTF scene nodes composed before binding", "actor_position_units": "cm",
            "importer_basis": "must be measured from axis_fixture in this UE installation and cancelled", "background_yz_swap_count": 0 if args.without_background else 1},
        "counts": {"building": sum(r["kind"] == "building" for r in records), "massing": sum(r["kind"] == "massing" for r in records),
            "surface": sum(r["kind"] == "surface" for r in records), "background": sum(r["kind"] == "background" for r in records),
            "proposal_bound": len(placements), "proposal_withheld": len(decisions)-len(unresolved), "proposal_unresolved": len(unresolved), "interchange_pending": len(missing)},
        "proposal_category_counts": {kind: sum(p["kind"] == kind for p in placements) for kind in sorted({p["kind"] for p in placements})},
        "logical_bridge_module_count": sum(p.get("logical_module_count", 0) for p in placements),
        "calibration": calibration, "meshes": records, "placements": placements, "proposal_sources": proposal_sources,
        "proposal_decisions": decisions, "pending_interchange_ids": missing,
        "ue_executed": False, "bridge_actor_world_transforms_used": False,
        "documentation": {"gltf_options": "https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/GLTFImportOptions?application_version=5.0",
            "mesh_subsystem": "https://dev.epicgames.com/documentation/en-us/unreal-engine/python-api/class/StaticMeshEditorSubsystem?application_version=5.0"}}
    write_json(output / "assembly_manifest.json", result)
    print(json.dumps({"manifest": str(output / "assembly_manifest.json"), "status": result["status"], "counts": result["counts"]}), flush=True)


if __name__ == "__main__":
    main()
