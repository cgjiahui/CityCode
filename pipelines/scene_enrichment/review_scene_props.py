"""Create a separate, bounded street-prop repair candidate from frozen inputs.

Vehicle tilt fits the actual terrain to a native-bounds bottom-plane proxy. It
does not claim access to tire contact meshes. All native parts move together.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Point, mapping, shape
from shapely.strtree import STRtree

from props import (CollisionIndex, Config, GroundMaps, PublishedSemantics,
                   evaluate_support, json_write, load_buildings, load_roads,
                   matrix_trs, sha, ue_rotation)
from props_scene_qa import (ActualGround, E, ROOT, Raster, bottom_contact_proxy,
                            deck_overlap, group_geometry, ground_contacts,
                            input_identity, native_obstacles, obb_xy_bounds,
                            read_span_world)
from review_props import obb, obb_penetration

BOUNDED_MOVE_GROUPS = {"PROP_5bec26a23df0aeccc768", "PROP_ea171f6f53c28800453a", "PROP_c0d3d0d15ae4f14ab6e0"}


def tilt_to_ground(group, asset, ground, maximum_tilt_deg=3.):
    before = ground_contacts(group, asset, ground)
    if before["maximum_gap_m"] is None or before["maximum_gap_m"] <= .15:
        return None, {"reason": "local_gap_at_or_below_0_15m", "before": before}
    points, _, _ = bottom_contact_proxy(group, asset)
    heights = ground.sample(points)
    if not np.isfinite(heights).all():
        return None, {"reason": "missing_actual_ground", "before": before}
    fit = np.linalg.lstsq(np.column_stack((points - points.mean(0), np.ones(len(points)))), heights, rcond=None)[0]
    a, b = fit[:2]
    yaw = math.radians(group["yaw_deg"])
    forward, right = a * math.cos(yaw) + b * math.sin(yaw), -a * math.sin(yaw) + b * math.cos(yaw)
    pitch = math.atan(forward)
    roll = math.atan(-right * math.cos(pitch))
    inclination = math.degrees(math.atan(math.hypot(a, b)))
    if inclination > maximum_tilt_deg:
        return None, {"reason": "terrain_plane_exceeds_small_tilt_limit", "inclination_deg": inclination, "before": before}
    candidate = copy.deepcopy(group)
    candidate["rotation_deg"] = [math.degrees(pitch), group["yaw_deg"], math.degrees(roll)]
    candidate["position_m"][2] = 0.
    points, plane_z, _ = bottom_contact_proxy(candidate, asset)
    heights = ground.sample(points)
    if not np.isfinite(heights).all():
        return None, {"reason": "tilted_footprint_has_missing_actual_ground", "before": before}
    candidate["position_m"][2] = float(np.max(heights - plane_z))
    after = ground_contacts(candidate, asset, ground)
    improvement = before["maximum_gap_m"] - after["maximum_gap_m"]
    decision = {"before": before, "after": after, "inclination_deg": inclination,
                "maximum_gap_improvement_m": improvement,
                "contact_interpretation": "native_bounds_bottom_plane_proxy_not_actual_tire_mesh"}
    if improvement < .05 or after["maximum_gap_m"] > before["maximum_gap_m"] * .65:
        decision["reason"] = "local_ground_is_not_planar_enough_for_meaningful_proxy_improvement"
        return None, decision
    decision["reason"] = "small_tilt_meaningfully_improves_bounds_contact_proxy"
    return candidate, decision


def flatten_candidate(group, asset, old_rows):
    matrix = group_geometry(group, asset)[0]
    result = []
    parts = asset.get("parts") or [{"mesh": asset["mesh"], "local_matrix": np.eye(4).tolist()}]
    for old in old_rows:
        row = copy.deepcopy(old)
        part = parts[row["part_index"]]
        if row["mesh_path"] != part["mesh"]:
            raise ValueError("Cannot change the frozen native mesh assembly")
        world = matrix @ np.asarray(part.get("local_matrix", np.eye(4)), float)
        row["position_m"], row["rotation_deg"], row["scale"] = matrix_trs(world)
        row["world_matrix"] = world.tolist()
        row["evidence"] = copy.deepcopy(group["evidence"])
        result.append(row)
    if len(result) != len(parts):
        raise ValueError("Every native body/glass part must be retained")
    return result


class CandidateScene:
    def __init__(self, args, catalog):
        self.catalog, self.args = catalog, args
        self.actual = ActualGround(args.ground)
        self.ground = GroundMaps(args.ground)
        self.semantics = PublishedSemantics(args.semantics)
        land = json.loads(args.land.read_text())
        self.land = shapely.force_2d(shape(land.get("geometry", land)))
        road_root = ROOT / "data/smallcity_p384_gt_roads"
        self.roads = load_roads(road_root / "source/road_sections.geojson", road_root / "source/lane_centerlines.geojson", road_root / "ground_only/road_graph.json")
        self.buildings, self.config = load_buildings(args.binding), Config()
        self.obstacles = native_obstacles(args.supports, args.railings)
        self.obstacle_tree = STRtree([obb_xy_bounds(value) for _, _, value in self.obstacles])
        doc = json.loads(args.spans.read_text())
        if doc.get("uses_ground_truth_for_placement") is not False or doc.get("uses_original_bridge_actor_transforms") is not False:
            raise ValueError("Candidate decks must explicitly reject GT/original bridge transforms")
        self.spans = []
        for row in doc["templates"]:
            _, v, faces = read_span_world(row, args.spans.parent)
            self.spans.append({"id": row["span_id"], "vertices": v, "faces": faces, "bounds": np.array([v.min(0), v.max(0)])})
        self.raster = Raster(Path("/tmp/citycode_props_scene_review_raster"))

    def check(self, candidate, groups):
        asset = self.catalog[candidate["asset_id"]]
        _, poly, bound = group_geometry(candidate, asset)
        index = CollisionIndex(self.buildings, self.roads[0], self.roads[1], self.land, self.config)
        for group in groups:
            if group["group_id"] != candidate["group_id"]:
                p = group_geometry(group, self.catalog[group["asset_id"]])[1]
                index.add(p, group["category"], group["position_m"][:2])
        reason = index.reason(poly, candidate["category"], candidate["position_m"][:2])
        if reason:
            return reason
        points = bottom_contact_proxy(candidate, asset)[0]
        _, reason = evaluate_support(points, candidate["category"], self.ground, self.semantics, self.config)
        if reason:
            return reason
        contact = ground_contacts(candidate, asset, self.actual)
        if contact["missing_actual_ground_samples"] or contact["buried_over_0_05m"] or contact["local_contact_gap_over_0_25m"]:
            return "actual_contact_proxy_failed"
        value = obb([asset["bounds_min_m"], asset["bounds_max_m"]], candidate["position_m"], candidate.get("rotation_deg", [0., candidate["yaw_deg"], 0.]), [candidate["scale"]] * 3)
        for i in self.obstacle_tree.query(poly, predicate="intersects"):
            kind, identity, obstacle = self.obstacles[int(i)]
            if obb_penetration(value, obstacle) > .2:
                return f"native_{kind}_full_bounds_overlap:{identity}"
        for span in self.spans:
            bounds = span["bounds"]
            if not poly.intersects(shapely.box(*bounds[:, :2].ravel())) or min(bound[1][2], bounds[1, 2]) - max(bound[0][2], bounds[0, 2]) <= .2:
                continue
            if "columns" not in span:
                span["columns"] = self.raster.cast(span["vertices"], span["faces"], .25)
            overlap = deck_overlap(poly, bound[0][2], bound[1][2], span["columns"])
            if overlap["maximum_vertical_overlap_m"] > .2 or overlap["uncertain_open_triangle_columns"]:
                return "actual_predicted_deck_vs_full_prop_bounds:" + span["id"]
        return None


def bounded_along_curb_move(group, asset, scene, groups, maximum_m=1.):
    curb = next(c for c in scene.roads[2] if c["id"] == group["evidence"]["curb_edge_id"])
    attempts = []
    distances = [distance for magnitude in (.25, .5, .75, 1., 2., 3., 4., 6.) if magnitude <= maximum_m
                 for distance in (magnitude, -magnitude)]
    for distance in distances:
        trial = copy.deepcopy(group)
        trial["position_m"][:2] = (np.asarray(group["position_m"][:2]) + curb["tangent"] * distance).tolist()
        trial["evidence"]["baseline_curb_along_m"] = group["evidence"]["curb_along_m"]
        trial["evidence"]["curb_along_m"] = group["evidence"]["curb_along_m"] + distance
        trial["position_m"][2] = 0.
        points, plane_z, _ = bottom_contact_proxy(trial, asset)
        ground = scene.actual.sample(points)
        if not np.isfinite(ground).all():
            attempts.append({"along_curb_m": distance, "reason": "missing_actual_ground"})
            continue
        trial["position_m"][2] = float(np.max(ground - plane_z))
        reason = scene.check(trial, groups)
        attempts.append({"along_curb_m": distance, "reason": reason})
        if reason is None:
            return trial, attempts
    return None, attempts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=E / "props_reviewed_proposal.json")
    parser.add_argument("--groups", type=Path, default=E / "props_reviewed_groups.json")
    parser.add_argument("--catalog", type=Path, default=E / "props_asset_catalog.jsonl")
    parser.add_argument("--ground", type=Path, default=E / "props_ground/BACKGROUND_SAMPLING.json")
    parser.add_argument("--binding", type=Path, default=ROOT / "data/smallcity_p384_building_assets_completed/binding_manifest_physical.json")
    parser.add_argument("--land", type=Path, default=ROOT / "data/smallcity_p384_background_coastal/land.geojson")
    parser.add_argument("--semantics", type=Path, default=ROOT / "data/smallcity_p384_background_history/height/completed")
    parser.add_argument("--supports", type=Path, default=E / "bridge_supports_proposal.json")
    parser.add_argument("--railings", type=Path, default=E / "bridge_railings_proposal.json")
    parser.add_argument("--spans", type=Path, default=ROOT / "build/ue_citycode_bridge_assets/spans/span_manifest.json")
    parser.add_argument("--review-lamps", action="store_true", help="Try bounded along-curb moves for the two reviewed lamp groups")
    parser.add_argument("--review-blocked-cars", action="store_true", help="Try bounded along-curb moves for the one reviewed vehicle group")
    parser.add_argument("--withhold-unresolved", action="store_true", help="Withhold only reviewed actual-deck conflicts with no safe tested move; never overwrite the formal inputs")
    parser.add_argument("--maximum-along-curb-m", type=float, choices=[1., 2., 3., 4., 6.], default=1.)
    parser.add_argument("--output-prefix", type=Path, default=E / "props_scene_review_candidate")
    args = parser.parse_args(argv)
    proposal, group_doc = json.loads(args.proposal.read_text()), json.loads(args.groups.read_text())
    groups = copy.deepcopy(group_doc["groups"])
    catalog = {a["asset_id"]: a for a in map(json.loads, args.catalog.read_text().splitlines())}
    scene = CandidateScene(args, catalog)
    decisions, changed, removed = [], set(), set()
    for group_index, group in enumerate(groups):
        asset = catalog[group["asset_id"]]
        candidate, decision = None, None
        if group["category"] == "parked_car":
            candidate, decision = tilt_to_ground(group, asset, scene.actual)
            initial_reason = scene.check(group, groups) if args.review_blocked_cars and group["group_id"] in BOUNDED_MOVE_GROUPS else None
            if initial_reason and initial_reason.startswith(("actual_predicted_deck", "native_railing", "native_support")):
                candidate, attempts = bounded_along_curb_move(group, asset, scene, groups, args.maximum_along_curb_m)
                decision = {"reason": "bounded_along_curb_move" if candidate else "no_safe_move_within_bounded_search", "maximum_search_m": args.maximum_along_curb_m, "original_scene_issue": initial_reason, "attempts": attempts}
        elif args.review_lamps and group["category"] == "street_lamp" and group["group_id"] in BOUNDED_MOVE_GROUPS:
            initial_reason = scene.check(group, groups)
            if initial_reason and initial_reason.startswith(("actual_predicted_deck", "native_railing", "native_support")):
                candidate, attempts = bounded_along_curb_move(group, asset, scene, groups, args.maximum_along_curb_m)
                decision = {"reason": "bounded_along_curb_move" if candidate else "no_safe_move_within_bounded_search", "maximum_search_m": args.maximum_along_curb_m, "original_scene_issue": initial_reason, "attempts": attempts}
        if decision is None:
            continue
        decision["group_id"] = group["group_id"]
        decision["category"] = group["category"]
        decision["original_position_m"] = group["position_m"]
        decision["original_rotation_deg"] = group.get("rotation_deg", [0., group["yaw_deg"], 0.])
        if candidate is not None:
            issue = scene.check(candidate, groups)
            if issue:
                decision["accepted"] = False
                decision["scene_rejection"] = issue
            else:
                decision["accepted"] = True
                decision["old_position_m"] = group["position_m"]
                decision["new_position_m"] = candidate["position_m"]
                decision["old_rotation_deg"] = group.get("rotation_deg", [0., group["yaw_deg"], 0.])
                decision["new_rotation_deg"] = candidate.get("rotation_deg", [0., candidate["yaw_deg"], 0.])
                candidate["footprint"] = mapping(group_geometry(candidate, asset)[1])
                candidate["evidence"]["scene_review"] = {"reason": decision["reason"], "baseline_proposal_sha256": sha(args.proposal), "contact_proxy_not_tire_geometry": True}
                candidate["evidence"]["baseline_support"] = candidate["evidence"].get("support")
                candidate["evidence"]["support"] = {"source": "actual_coastal_mesh_triangles_25cm_with_transformed_native_bottom_plane_proxy",
                                                        "contact": ground_contacts(candidate, asset, scene.actual),
                                                        "rotation_deg": candidate.get("rotation_deg", [0., candidate["yaw_deg"], 0.])}
                groups[group_index] = candidate
                changed.add(group["group_id"])
        else:
            decision["accepted"] = False
            if args.withhold_unresolved and decision["reason"] == "no_safe_move_within_bounded_search":
                removed.add(group["group_id"])
                decision.update(action="withhold", accepted=True, visible_mesh_collision_claimed=False)
        decisions.append(decision)
    by_group = {}
    for row in proposal.get("rows", proposal.get("placements", [])):
        by_group.setdefault(row["group_id"], []).append(row)
    groups = [group for group in groups if group["group_id"] not in removed]
    replacements = {g["group_id"]: flatten_candidate(g, catalog[g["asset_id"]], by_group[g["group_id"]]) for g in groups if g["group_id"] in changed}
    updated_rows = {r["placement_id"]: r for rows in replacements.values() for r in rows}
    candidate_doc = copy.deepcopy(proposal)
    candidate_doc["rows"] = [updated_rows.get(row["placement_id"], row) for row in proposal["rows"] if row["group_id"] not in removed]
    candidate_doc["status"] = "candidate_requires_final_joined_bridge_review"
    candidate_doc["scene_review_candidate"] = {"baseline_proposal_sha256": sha(args.proposal), "baseline_groups_sha256": sha(args.groups),
        "changed_groups": sorted(changed), "withheld_groups": sorted(removed), "formal_baseline_modified": False,
        "decisions_file": Path(str(args.output_prefix) + "_decisions.json").name, "review_code_sha256": sha(__file__),
        "actual_span_manifest_sha256": sha(args.spans), "support_proposal_sha256": sha(args.supports), "railing_proposal_sha256": sha(args.railings),
        "maximum_along_curb_move_m": args.maximum_along_curb_m}
    candidate_doc["counts"]["accepted_groups"] = dict(Counter(group["category"] for group in groups))
    candidate_doc["counts"]["accepted_mesh_rows"] = dict(Counter(row["category"] for row in candidate_doc["rows"]))
    candidate_doc["withheld"] = list(candidate_doc.get("withheld", [])) + [
        {"group_id": identity, "category": by_group[identity][0]["category"],
         "reason": "Conservative actual-deck/full-native-bounds conflict; all tested along-curb moves within %gm violate original scene constraints" % args.maximum_along_curb_m,
         "original_group": next(g for g in group_doc["groups"] if g["group_id"] == identity),
         "removed_mesh_rows": by_group[identity], "does_not_claim_visible_mesh_intersection": True}
        for identity in sorted(removed)]
    candidate_doc["policy"]["candidate_vehicle_contact"] = "Ground plane fit with <=3 degree inclination; native bottom bounds proxy only, not actual tire contacts; all parts share the composed transform"
    out = {kind: Path(str(args.output_prefix) + "_" + kind + ".json") for kind in ("proposal", "groups", "decisions")}
    if args.proposal.resolve() in [p.resolve() for p in out.values()] or args.groups.resolve() in [p.resolve() for p in out.values()]:
        raise ValueError("Candidate outputs must not overwrite formal inputs")
    json_write(out["proposal"], candidate_doc)
    json_write(out["groups"], {**group_doc, "groups": groups})
    paths = {name: getattr(args, name) for name in ("proposal", "groups", "catalog", "ground", "binding", "land", "supports", "railings", "spans")}
    paths.update({"actual_ground_" + k: v for k, v in scene.actual.paths.items()})
    for name in ("base_surface_z_m", "base_confidence", "base_provenance", "base_semantic_id", "pixel_to_world"):
        paths["published_ground_" + name] = args.semantics / (name + (".json" if name == "pixel_to_world" else ".npy"))
    road_root = ROOT / "data/smallcity_p384_gt_roads"
    paths.update(ground_road_sections=road_root / "source/road_sections.geojson", ground_vehicle_lanes=road_root / "source/lane_centerlines.geojson", ground_graph=road_root / "ground_only/road_graph.json")
    for row in json.loads(args.spans.read_text())["templates"]:
        paths["actual_span_" + row["span_id"]] = args.spans.parent / row["template_file"]
    report = {"schema": "citycode.props_scene_review_candidate.v1", "ue_executed": False, "formal_inputs_modified": False,
              "source_code_sha256": sha(__file__), "helper_code_sha256": {name: sha(Path(__file__).parent / name) for name in ("props_scene_qa.py", "props.py", "review_props.py")},
              "inputs": {k: {"path": input_identity(v), "sha256": sha(v)} for k, v in paths.items()},
              "outputs": {k: {"path": input_identity(v), "sha256": sha(v)} for k, v in out.items() if k != "decisions"},
              "counts": {"groups": len(groups), "mesh_rows": len(candidate_doc["rows"]), "changed_groups": len(changed), "removed_groups": len(removed)},
              "limits": {"maximum_vehicle_terrain_inclination_deg": 3., "minimum_local_gap_improvement_m": .05, "maximum_along_curb_move_m": args.maximum_along_curb_m},
              "provenance": "Previously authorized ground-road XY reference; actual published terrain; prediction-derived actual deck triangles. No GT bridge placement or original actor transforms.",
              "limitations": ["Vehicle contact improvement measures native bounds bottom-plane proxy, not actual wheel contact meshes.", "A new joined bridge/ground/support/rail candidate requires a fresh final QA report before publication."], "decisions": decisions}
    json_write(out["decisions"], report)
    print(json.dumps({"outputs": {k: str(v) for k, v in out.items()}, "counts": report["counts"]}))
    return report


if __name__ == "__main__":
    main()
