"""Apply a recorded scene review while preserving source evidence and meshes.

This module does not infer ownership or read reference geometry. A reviewed
decision file is an explicit input; reference-assisted review must be declared.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path


def apply_scene_review(manifest_path, decisions_path, output_path):
    from .cli import ACCEPTED

    manifest_path, decisions_path, output_path = map(Path, (manifest_path, decisions_path, output_path))
    if output_path.exists() or output_path.resolve() == manifest_path.resolve():
        raise ValueError("Write the reviewed manifest to a new distinct output path")
    if output_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("Reviewed manifest must retain the original asset path base")
    payload = manifest_path.read_bytes()
    original, decisions = json.loads(payload), json.loads(decisions_path.read_text())
    if decisions.get("schema") != "citycode.scene_asset_review.v1":
        raise ValueError("Unsupported scene review schema")
    if hashlib.sha256(payload).hexdigest() != decisions.get("input_manifest_sha256"):
        raise ValueError("Scene review is not bound to this exact input manifest")
    records = decisions["records"]
    selected = {r["building_id"]: r for r in records}
    if len(selected) != len(records) or any(r.get("action") not in {"include", "record_only", "hold"} for r in records):
        raise ValueError("Scene decisions require unique source IDs and recognized actions")
    instances = {r["id"]: r for r in original["asset_instances"]}
    baseline = decisions["baseline_asset_ids"]
    if len(set(baseline)) != len(baseline) or set(baseline) & set(selected) or set(baseline) | set(selected) != set(instances):
        raise ValueError("Baseline and reviewed assets must partition the input assembly table")
    result = copy.deepcopy(original)
    rows = {r["building_id"]: r for r in result["buildings"]}
    for sid, decision in selected.items():
        if sid not in rows or rows[sid].get("status") not in ACCEPTED or instances[sid]["source_building_ids"] != [sid]:
            raise ValueError("Review requires an existing accepted single-source candidate")
        if not decision.get("reason") or not decision.get("evidence"):
            raise ValueError("Each decision needs its reason and observed evidence")
        row = rows[sid]
        row["scene_review"] = copy.deepcopy(decision)
        if decision["action"] != "include":
            row["retained_candidate_lods"] = row.pop("lods")
            row["previous_candidate_status"] = row["status"]
            row["status"] = "excluded_existing_building" if decision["action"] == "record_only" else "pending_review"
            row["reason"] = decision["reason"]
            row["evidence"] = copy.deepcopy(decision["evidence"])
    enabled = set(baseline) | {sid for sid, d in selected.items() if d["action"] == "include"}
    result["asset_instances"] = [r for r in result["asset_instances"] if r["id"] in enabled]
    # Logical entities can contain several distinct, material-preserving mesh
    # parts. They differ from physical shared-asset groups, which instance one
    # mesh once for several source IDs.
    owners, entities = {}, []
    active = {r["id"]: r for r in result["asset_instances"]}
    for group in decisions.get("building_entity_groups", []):
        members = group["asset_instance_ids"]
        if len(set(members)) < 2 or len(set(members)) != len(members) or set(members) - enabled or set(members) & set(owners):
            raise ValueError("A building entity needs unique active parts without conflicting ownership")
        identity = group["id"]
        entities.append({**copy.deepcopy(group), "source_building_ids": sorted({s for a in members for s in active[a]["source_building_ids"]})})
        owners.update({a: identity for a in members})
    for instance in result["asset_instances"]:
        aid = instance["id"]
        if aid not in owners:
            owners[aid] = aid
            entities.append({"id": aid, "asset_instance_ids": [aid], "source_building_ids": instance["source_building_ids"]})
        instance["building_entity_id"] = owners[aid]
    result["building_entities"] = entities
    result["counts"] = dict(Counter(r["status"] for r in result["buildings"]))
    pending = any(r["status"].startswith("pending_") for r in result["buildings"])
    result["status"] = "in_progress" if pending else "complete"
    result["fine_generation_complete"] = not pending
    if "fine_refinement" in result:
        result["fine_refinement"]["scope"] = "Historical generation batch; current scene selection is recorded in scene_review."
    result["assembly_contract"] = (
        "Instantiate each active asset_instances mesh part exactly once. Build CityCode building objects from "
        "building_entities, attaching its listed mesh parts under one building identity. Physical asset_instance_groups "
        "still instantiate their shared mesh only once. Add only assembly_enabled surface attachments by default. "
        "Records excluded_existing_building and pending_review are evidence/candidates, not active placements."
    )
    result["scene_review"] = {"schema": decisions["schema"], "input_manifest_sha256": decisions["input_manifest_sha256"],
        "decisions_sha256": hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
        "reference_assisted_review": bool(decisions.get("reference_assisted_review", False)),
        "reviewed_candidates": len(records), "decision_counts": dict(Counter(r["action"] for r in records)),
        "physical_mesh_parts": len(result["asset_instances"]), "building_entities": len(entities),
        "source_instances_preserved": len(result["buildings"]), "mesh_bytes_modified": False,
        "geometry_generation_claimed": False, "pending_candidates": [s for s, d in selected.items() if d["action"] == "hold"]}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result["scene_review"]
