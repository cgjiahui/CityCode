"""Replayable, input-bound review of three new props near predicted railings.

The frozen baseline remains read-only. Try bounded translations along the
street, then withhold only a reviewed group when no safe candidate exists.
Railing conflicts use conservative complete native bounding boxes; they do not
claim an exact intersection of the native visible meshes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from props import (ROOT, Config, CollisionIndex, GroundMaps, PublishedSemantics,
                   evaluate_support, flatten_parts, footprint, json_write,
                   load_buildings, load_roads, overhead_reason, sha,
                   support_points, ue_rotation, validate_groups)
from shapely.geometry import shape

TARGETS = ("PROP_7d32949aac398dceeeb2", "PROP_ea171f6f53c28800453a", "PROP_a8a61b9ca5cdc14f726a")


def obb(bounds, position, rotation, scale):
    bounds = np.asarray(bounds, float)
    r = ue_rotation(rotation)
    scale = np.asarray(scale, float)
    middle = (bounds[0]+bounds[1])/2
    return {"center": np.asarray(position)+r @ (middle*scale), "axes": r,
            "half": (bounds[1]-bounds[0])*scale/2}


def obb_penetration(a, b):
    """Separating-axis test including cross axes, in metre projections."""
    axes = [a["axes"][:, i] for i in range(3)]+[b["axes"][:, i] for i in range(3)]
    axes += [np.cross(a["axes"][:, i], b["axes"][:, j]) for i in range(3) for j in range(3)]
    smallest = float("inf")
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis = axis/norm
        radius_a = np.dot(a["half"], np.abs(a["axes"].T @ axis))
        radius_b = np.dot(b["half"], np.abs(b["axes"].T @ axis))
        overlap = radius_a+radius_b-abs(np.dot(a["center"]-b["center"], axis))
        if overlap <= 0:
            return 0.
        smallest = min(smallest, float(overlap))
    return smallest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path)
    p.add_argument("--output-directory", type=Path,
                   help="Write review results to a separate directory; leave frozen inputs untouched")
    args = p.parse_args()
    e = ROOT/"data/smallcity_p384_scene_enrichment"
    output = args.output_directory.resolve() if args.output_directory else e
    if args.output_directory:
        if output == e.resolve():
            raise ValueError("Explicit review output must be separate from the frozen input directory")
        for filename in ("props_reviewed_proposal.json", "props_reviewed_groups.json",
                         "props_review_decisions.json", "props_review_validation.json"):
            if (output/filename).exists():
                raise ValueError("Refusing to overwrite review output: " + str(output/filename))
    paths = {"baseline_proposal": e/"props_proposal.json", "baseline_groups": e/"props_groups.json",
             "catalog": e/"props_asset_catalog.jsonl", "rails": e/"bridge_railings_proposal.json",
             "buildings": ROOT/"data/smallcity_p384_building_assets_completed/binding_manifest_physical.json",
             "land": ROOT/"data/smallcity_p384_background_coastal/land.geojson",
             "sections": ROOT/"data/smallcity_p384_gt_roads/source/road_sections.geojson",
             "lanes": ROOT/"data/smallcity_p384_gt_roads/source/lane_centerlines.geojson",
             "graph": ROOT/"data/smallcity_p384_gt_roads/ground_only/road_graph.json",
             "actual_ground_sampling": e/"props_ground/BACKGROUND_SAMPLING.json"}
    for name in ("continuous_top", "detail_near_ground_top", "detail_max_top"):
        paths["actual_ground_"+name] = e/"props_ground"/(name+".npy")
    semantic_folder = ROOT/"data/smallcity_p384_background_history/height/completed"
    for name in ("base_surface_z_m", "base_confidence", "base_provenance", "base_semantic_id"):
        paths["semantic_"+name] = semantic_folder/(name+".npy")
    paths["semantic_pixel_to_world"] = semantic_folder/"pixel_to_world.json"
    inputs = {k: {"path": str(v.relative_to(ROOT)), "sha256": sha(v)} for k, v in paths.items()}
    baseline = json.loads(paths["baseline_proposal"].read_text())
    groups = json.loads(paths["baseline_groups"].read_text())["groups"]
    bygroup = {g["group_id"]: g for g in groups}
    catalog = {a["asset_id"]: a for a in map(json.loads, paths["catalog"].read_text().splitlines())}
    rails = json.loads(paths["rails"].read_text())
    bounds = np.asarray(rails["native_asset"]["bounds_cm"])/100
    rail_boxes = [(r["placement_id"], obb(bounds, r["position_m"], r["rotation_deg"], r["scale"]))
                  for r in rails["placements"]]
    buildings = load_buildings(paths["buildings"])
    roads = load_roads(paths["sections"], paths["lanes"], paths["graph"])
    land = shape(json.loads(paths["land"].read_text())["geometry"])
    c = Config()
    collisions = CollisionIndex(buildings, roads[0], roads[1], land, c)
    for group in groups:
        if group["group_id"] not in TARGETS:
            collisions.add(shape(group["footprint"]), group["category"], group["position_m"][:2])
    ground = GroundMaps(e/"props_ground")
    semantics = PublishedSemantics(ROOT/"data/smallcity_p384_background_history/height/completed")

    def rail_conflicts(asset, position, yaw, scale):
        b = np.array([asset["bounds_min_m"], asset["bounds_max_m"]])
        candidate = obb(b, position, [0., yaw, 0.], [scale]*3)
        return [{"rail_placement_id": identity, "minimum_obb_axis_penetration_m": depth}
                for identity, rail in rail_boxes
                if (depth := obb_penetration(candidate, rail)) > .2]

    def candidate(group, along):
        asset = catalog[group["asset_id"]]
        tangent = ue_rotation([0, group["yaw_deg"], 0])[:2, 0]
        old = np.asarray(group["position_m"])
        xy = old[:2]+tangent*along
        poly = footprint(asset, xy, group["yaw_deg"], group["scale"])
        reason = collisions.reason(poly, group["category"], xy)
        if reason:
            return None, reason
        support, reason = evaluate_support(support_points(poly, group["category"], xy),
                                           group["category"], ground, semantics, c)
        if reason:
            return None, reason
        z = support["z_m"]-float(asset["bounds_min_m"][2])*group["scale"]
        position = np.array([*xy, z])
        if np.linalg.norm(position-old) > 1.+1e-9:
            return None, "total_translation_exceeds_1m"
        reason = overhead_reason(poly, support["z_m"], z+asset["bounds_max_m"][2]*group["scale"], ground)
        if reason:
            return None, reason
        conflicts = rail_conflicts(asset, position, group["yaw_deg"], group["scale"])
        if conflicts:
            return None, "conservative_railing_bounds_contact"
        changed = copy.deepcopy(group)
        changed["position_m"] = position.tolist(); changed["footprint"] = poly.__geo_interface__
        changed["evidence"]["support"] = support
        changed["evidence"]["agentic_review"] = {"bounded_street_parallel_translation_m": along,
                                                   "total_translation_m": float(np.linalg.norm(position-old)),
                                                   "conservative_railing_bounds_checked": len(rail_boxes)}
        return changed, None

    decisions = []
    if args.replay:
        frozen = json.loads(args.replay.read_text())
        if frozen["inputs"] != inputs:
            raise ValueError("Review replay input identity changed; regenerate and review decisions")
        frozen_by = {d["group_id"]: d for d in frozen["decisions"]}
    else:
        frozen_by = None
    changed_groups = {}
    withheld = set()
    for identity in TARGETS:
        group = bygroup[identity]
        asset = catalog[group["asset_id"]]
        before_conflicts = rail_conflicts(asset, group["position_m"], group["yaw_deg"], group["scale"])
        attempts = []
        selected = None
        if frozen_by:
            frozen_decision = frozen_by[identity]
            offsets = [frozen_decision["along_street_translation_m"]] if frozen_decision["action"] == "translate" else []
        else:
            offsets = [.25, -.25, .5, -.5, .75, -.75, 1., -1.]
        selected_offset = None
        for offset in offsets:
            selected, reason = candidate(group, offset)
            attempts.append({"along_street_translation_m": offset, "status": reason or "safe"})
            if selected:
                selected_offset = offset
                break
        if frozen_by:
            # Preserve the reviewed, inspectable action log exactly on replay.
            decision = frozen_by[identity]
            if decision["action"] == "translate" and selected is None:
                raise ValueError("Frozen translated placement no longer satisfies spatial constraints")
        else:
            decision = {"group_id": identity, "category": group["category"],
                        "action": "translate" if selected else "withhold",
                        "along_street_translation_m": selected_offset,
                        "before_conservative_rail_conflicts": before_conflicts,
                        "attempts": attempts,
                        "reason": "A bounded street-parallel move passes original scene constraints and railing OBB tolerance" if selected else "No tested street-parallel move within 1m passes all original constraints and conservative railing-bounds tolerance",
                        "does_not_claim_visible_mesh_intersection": True}
            if selected:
                decision["delta_world_xyz_m"] = (np.asarray(selected["position_m"])-group["position_m"]).tolist()
        if selected:
            changed_groups[identity] = selected
            collisions.add(shape(selected["footprint"]), selected["category"], selected["position_m"][:2])
        else:
            withheld.add(identity)
        decisions.append(decision)
    decision_doc = {"schema": "citycode.props_agentic_review_decisions.v1", "inputs": inputs,
                    "authority": "User authorized <=1m along-street adjustment of these three newly generated prop groups, otherwise withhold; preserve buildings, bridge geometry and baseline proposal",
                    "scope": list(TARGETS), "contact_tolerance_m": .2, "decisions": decisions,
                    "review_code_sha256": sha(__file__)}
    final_groups = [changed_groups.get(g["group_id"], g) for g in groups if g["group_id"] not in withheld]
    replacements = {}
    for identity, group in changed_groups.items():
        replacements[identity] = flatten_parts(catalog[group["asset_id"]], identity,
                                               group["position_m"], group["yaw_deg"], group["scale"], group["evidence"])
    rows = []
    for row in baseline["rows"]:
        identity = row["group_id"]
        if identity in withheld:
            continue
        rows.append(replacements[identity][row["part_index"]] if identity in replacements else row)
    result = copy.deepcopy(baseline)
    result["rows"] = rows
    result["counts"]["accepted_groups"] = dict(Counter(g["category"] for g in final_groups))
    result["counts"]["accepted_mesh_rows"] = dict(Counter(r["category"] for r in rows))
    result["agentic_review"] = {"schema": "citycode.props_agentic_review.v1", "baseline_proposal_sha256": inputs["baseline_proposal"]["sha256"],
                                 "railing_proposal_sha256": inputs["rails"]["sha256"],
                                 "decisions_file": "props_review_decisions.json", "decisions": decisions,
                                 "groups_before": len(groups), "groups_after": len(final_groups),
                                 "mesh_rows_before": len(baseline["rows"]), "mesh_rows_after": len(rows),
                                 "translated_groups": list(changed_groups), "withheld_groups": sorted(withheld),
                                 "review_code_sha256": sha(__file__)}
    result["status"] = "cpu_validated_agent_reviewed_proposal"
    validation = validate_groups(final_groups, list(catalog.values()), buildings, roads, land, c)
    validation["reviewed_group_railing_bounds_conflicts_after"] = [
        {"group_id": identity, "conflicts": rail_conflicts(catalog[group["asset_id"]], group["position_m"],
                                                           group["yaw_deg"], group["scale"])}
        for identity, group in changed_groups.items()]
    if any(r["conflicts"] for r in validation["reviewed_group_railing_bounds_conflicts_after"]):
        raise ValueError("Reviewed placement still conflicts with conservative railing bounds")
    if validation["errors"]:
        raise ValueError("Reviewed scene failed full spatial checks")
    json_write(output/"props_reviewed_proposal.json", result)
    json_write(output/"props_reviewed_groups.json", {"schema": "citycode.props_groups.v1", "groups": final_groups})
    json_write(output/"props_review_decisions.json", decision_doc)
    json_write(output/"props_review_validation.json", validation)
    if sha(paths["baseline_proposal"]) != inputs["baseline_proposal"]["sha256"]:
        raise ValueError("Frozen baseline changed unexpectedly")
    print(json.dumps({"groups_after": len(final_groups), "rows_after": len(rows),
                      "translated": list(changed_groups), "withheld": sorted(withheld),
                      "output_sha256": sha(output/"props_reviewed_proposal.json"), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
