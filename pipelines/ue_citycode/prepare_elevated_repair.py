"""Prepare a separate elevated-network candidate; never imports assets or runs UE.

Existing enrichment producers are called as functions with explicit inputs.
Final cap contacts and interior-rail suppression use the candidate GLB geometry.
The formal data, baseline manifest and licensed UE project remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines/scene_enrichment"))
import bridge_railings
import bridge_supports
from interchange import accessor, digest_file, read_glb
from prepare import bind_placement


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def height_hits(triangles, xy):
    if not len(triangles):
        return np.empty(0)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, delta = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2], np.asarray(xy)-a[:, :2]
    determinant = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
    valid = abs(determinant) > 1e-10
    denominator = np.where(valid, determinant, 1.)
    u = (delta[:, 0]*ac[:, 1]-delta[:, 1]*ac[:, 0])/denominator
    v = (ab[:, 0]*delta[:, 1]-ab[:, 1]*delta[:, 0])/denominator
    valid &= (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7)
    return a[valid, 2] + u[valid]*(b[valid, 2]-a[valid, 2]) + v[valid]*(c[valid, 2]-a[valid, 2])


class ActualDeckGeometry:
    def __init__(self, manifest_path, joined_manifest_path=None):
        path = Path(manifest_path); manifest = read(path)
        self.bottom, tops = {}, []
        for row in manifest["templates"]:
            if not np.allclose(row.get("scale", [1, 1, 1]), [1, 1, 1]) or not np.allclose(row.get("rotation_deg", [0, 0, 0]), [0, 0, 0]):
                raise ValueError("Candidate span geometry must use the native anchor-only placement contract")
            filename = path.parent / row["template_file"]
            document, binary, sha = read_glb(filename)
            if sha != row["template_sha256"]:
                raise ValueError("Candidate span GLB differs from its manifest")
            lower = []
            for mesh in document["meshes"]:
                for primitive in mesh["primitives"]:
                    positions, _ = accessor(document, binary, primitive["attributes"]["POSITION"])
                    if "indices" in primitive:
                        indices, _ = accessor(document, binary, primitive["indices"])
                        positions = positions[indices.reshape(-1)]
                    # Standard glTF Y-up -> native local UE, reversing winding.
                    triangles = positions[:, [0, 2, 1]].reshape(-1, 3, 3)[:, [0, 2, 1]].astype(float)
                    triangles += np.asarray(row["position_m"])
                    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                    length = np.linalg.norm(normal, axis=1)
                    normal /= np.maximum(length[:, None], 1e-30)
                    lower.extend(triangles[normal[:, 2] < -.2])
                    tops.extend(triangles[normal[:, 2] > .7])
            self.bottom[row["span_id"]] = np.asarray(lower).reshape(-1, 3, 3)
        self.joined_bottom = None
        self.top = np.asarray(tops).reshape(-1, 3, 3)
        if joined_manifest_path is not None:
            joined = ActualDeckGeometry(joined_manifest_path)
            self.top = joined.top
            self.joined_bottom = np.concatenate(list(joined.bottom.values()))
        self.polygons = [Polygon(triangle[:, :2]) for triangle in self.top]
        self.tree = STRtree(self.polygons)

    def interior(self, point, margin=.3, height_tolerance=.75):
        point = np.asarray(point); query = Point(point[:2])
        indices = self.tree.query(query.buffer(margin+.05))
        nearby = []
        for index in indices:
            triangle = self.top[index]
            n = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            z = triangle[0, 2] - np.dot(n[:2], point[:2]-triangle[0, :2])/n[2]
            if abs(z-point[2]) <= height_tolerance:
                nearby.append(self.polygons[index])
        return bool(nearby and unary_union(nearby).buffer(-margin).covers(query))


def filter_rails(proposal, geometry):
    result = copy.deepcopy(proposal); accepted = []; withheld = list(result.get("withheld_placements", []))
    for row in result["placements"]:
        a, b = np.asarray(row["road_edge_start_m"]), np.asarray(row["road_edge_end_m"])
        inside = [geometry.interior(point) for point in (a, (a+b)*.5, b)]
        if inside[1] or sum(inside) >= 2:
            withheld.append({"placement_id": row["placement_id"], "source_span_id": row["source_span_id"],
                "reason": "rail_would_block_actual_same_level_candidate_deck_interior"})
        else:
            accepted.append(row)
    result.update(placements=accepted, placement_count=len(accepted), withheld_placements=withheld, withheld_count=len(withheld))
    result["policy"]["actual_candidate_mesh_boundary_check"] = {"interior_margin_m": .3, "height_tolerance_m": .75,
        "method": "union of actual upward candidate GLB triangles near each rail sample, at matching height"}
    return result


def refit_supports(rows, geometry, assets, maximum_cap_gap=.35):
    accepted, withheld = [], []
    by_path = {asset["mesh_path"]: asset for asset in assets}
    for original in rows:
        row = copy.deepcopy(original)
        triangles = geometry.bottom.get(row["source_segment_id"])
        corners = np.asarray(row["footprint"]["coordinates"][0])[:-1]
        hits = [height_hits(triangles, xy) for xy in corners] if triangles is not None else []
        if hits and all(len(z) for z in hits) and getattr(geometry, "joined_bottom", None) is not None:
            final_hits = []
            for xy, reference in zip(corners, hits):
                values = height_hits(geometry.joined_bottom, xy)
                # Keep the source span's height layer at grade-separated crossings.
                values = values[abs(values-float(np.min(reference))) <= .75]
                final_hits.append(values[np.argmin(abs(values-np.min(reference)))][None] if len(values) else values)
            hits = final_hits
        reason = None
        if not hits or any(not len(z) for z in hits):
            reason = "support_cap_not_fully_under_actual_candidate_span"
        else:
            underside = np.asarray([np.min(z) for z in hits]); gap = float(np.ptp(underside)); cap = float(np.min(underside))
            if gap > maximum_cap_gap:
                reason = "actual_candidate_underside_exceeds_cap_gap_limit"
            else:
                try:
                    center = np.mean(corners, axis=0)
                    position, scale = bridge_supports.fit_support(by_path[row["mesh_path"]], center,
                        row["rotation_deg"][1], row["ground_contact_z_m"], cap)
                except ValueError:
                    reason = "actual_candidate_support_height_outside_limits"
        if reason:
            withheld.append({"placement_id": row["placement_id"], "source_segment_id": row["source_segment_id"], "reason": reason})
            continue
        row.update(position_m=position.tolist(), scale=scale.tolist(), support_cap_z_m=cap,
            support_height_m=cap-row["ground_contact_z_m"], maximum_cap_to_sloped_deck_gap_m=gap,
            actual_candidate_underside_corner_z_m=underside.tolist(),
            cap_fit_method="vertical triangle intersections against final candidate GLB downward faces")
        accepted.append(row)
    return accepted, withheld


def inherited_review_decisions(path, value, root=ROOT):
    """Resolve the first review relative to its verified source proposal.

    A scene review carries the earlier review unchanged into a new directory.
    Its frozen input/output identities locate that earlier proposal without
    guessing a release-specific folder or accepting an unrelated same-name file.
    """
    path = Path(path).resolve()
    scene_review = value.get("scene_review_candidate")
    if scene_review:
        report = read(path.parent / scene_review["decisions_file"])
        if report["outputs"]["proposal"]["sha256"] != digest_file(path):
            raise ValueError("Scene review report does not identify this proposal")
        source = report["inputs"]["proposal"]
        parent = (Path(root) / source["path"]).resolve()
        if (source["sha256"] != scene_review["baseline_proposal_sha256"]
                or digest_file(parent) != source["sha256"]):
            raise ValueError("Scene review source proposal changed")
        original = read(parent)
        if original.get("agentic_review") != value["agentic_review"]:
            raise ValueError("Inherited review differs from its frozen source")
        path = parent
    return path.parent / value["agentic_review"]["decisions_file"]


def candidate_manifest(baseline_path, output, proposal_paths, catalog_path, native_catalog_path):
    baseline_path, output = Path(baseline_path), Path(output)
    baseline = read(baseline_path); result = copy.deepcopy(baseline)
    native = read(native_catalog_path)
    if native.get("status") != "native_bridge_template_imported" or not native.get("ue_executed") or not native.get("bounds_verified") or not native.get("native_materials_verified"):
        raise ValueError("Candidate manifest requires actual verified UE native bridge import results")
    catalog = {row["mesh_path"]: row for row in read(catalog_path)["assets"]}
    catalog.update({row["mesh_path"]: row for row in native["assets"]})
    placements, decisions, sources = [], [], []
    for path in map(Path, proposal_paths):
        value = read(path)
        if value.get("coordinate_system") != {"world": "UE", "units": "m", "up": "Z"}:
            raise ValueError("Candidate proposal coordinate contract differs")
        for row in value.get("placements", value.get("rows", [])):
            bound, decision = bind_placement(row, catalog, {})
            if bound is None:
                raise ValueError("Candidate proposal has an unresolved mesh: " + str(decision))
            placements.append(bound)
        decisions.extend(value.get("withheld_placements", value.get("withheld", [])))
        provenance = {"file": os.path.relpath(path.resolve(), ROOT), "path_base": "release_root", "sha256": digest_file(path), "schema": value.get("schema")}
        if value.get("agentic_review"):
            decision = inherited_review_decisions(path, value)
            provenance.update(agentic_review=value["agentic_review"], review_decisions_sha256=digest_file(decision),
                review_decisions_file=os.path.relpath(decision.resolve(), ROOT))
        if value.get("scene_review_candidate"):
            provenance["scene_review_candidate"] = value["scene_review_candidate"]
            if value["scene_review_candidate"].get("decisions_file"):
                scene_decisions = path.parent/value["scene_review_candidate"]["decisions_file"]
                provenance.update(scene_review_decisions_sha256=digest_file(scene_decisions),
                    scene_review_decisions_file=os.path.relpath(scene_decisions.resolve(), ROOT))
        sources.append(provenance)
    if len({row["id"] for row in placements}) != len(placements):
        raise ValueError("Candidate placement identities overlap")
    scene_id = hashlib.sha256(json.dumps({"baseline": digest_file(baseline_path), "sources": sources, "placements": placements}, sort_keys=True).encode()).hexdigest()[:12]
    # Rebase only file references. Immutable building/background geometry is reused.
    for record in result["meshes"]:
        record["import_file"] = os.path.relpath(baseline_path.parent/record["import_file"], output.parent)
        for part in record.get("import_parts", []):
            part["file"] = os.path.relpath(baseline_path.parent/part["file"], output.parent)
    result["calibration"]["path"] = os.path.relpath(baseline_path.parent/result["calibration"]["path"], output.parent)
    merged_catalog_path = output.with_suffix(".catalog.json")
    write(merged_catalog_path, {"schema": "citycode.ue_local_asset_catalog.v1", "assets": list(catalog.values()),
        "source_catalog_sha256": digest_file(catalog_path), "verified_native_catalog_sha256": digest_file(native_catalog_path)})
    result.update(scene_id=scene_id, new_level_path="/Game/Research/CityCodeOpenSource/Maps/CityCode_"+scene_id,
        placements=placements, proposal_sources=sources, proposal_decisions=decisions, ue_executed=False,
        catalog_sha256=digest_file(merged_catalog_path), candidate_native_catalog_sha256=digest_file(native_catalog_path))
    result["counts"].update(proposal_bound=len(placements), proposal_withheld=len(decisions), proposal_unresolved=0)
    result["proposal_category_counts"] = dict(Counter(row["kind"] for row in placements))
    result["logical_bridge_module_count"] = sum(row.get("logical_module_count", 0) for row in placements)
    result["candidate_baseline_manifest_sha256"] = digest_file(baseline_path)
    old, new = {row["id"]: row for row in baseline["placements"]}, {row["id"]: row for row in placements}
    plan = {"schema": "citycode.candidate_placement_diff.v1", "baseline_manifest_sha256": digest_file(baseline_path),
        "candidate_scene_id": scene_id, "apply_executed": False, "requires_new_map_path": result["new_level_path"],
        "unchanged_generated_inputs": [row["id"] for row in baseline["meshes"]],
        "add": [new[key] for key in sorted(set(new)-set(old))], "remove": [old[key] for key in sorted(set(old)-set(new))],
        "replace": [{"before": old[key], "after": new[key]} for key in sorted(set(old)&set(new)) if old[key] != new[key]],
        "unchanged_placement_ids": [key for key in sorted(set(old)&set(new)) if old[key] == new[key]],
        "map_policy": "current GUI must explicitly execute an apply script; save as a new dedicated map; no background UE or original-map overwrite"}
    write(output, result); plan["candidate_manifest_sha256"] = digest_file(output)
    write(output.with_suffix(".diff.json"), plan)
    return result


def gui_apply_plan(baseline_path, proposal_paths, catalog_path, template_manifest, output):
    """Bind existing native references; bridge binding waits for real GUI import."""
    baseline_path, template_manifest, output = map(Path, (baseline_path, template_manifest, output))
    baseline = read(baseline_path)
    catalog = {row["mesh_path"]: row for row in read(catalog_path)["assets"]}
    placements, sources, withheld = [], [], []
    for path in map(Path, proposal_paths):
        value = read(path)
        if value.get("coordinate_system") != {"world": "UE", "units": "m", "up": "Z"}:
            raise ValueError("GUI candidate coordinate contract differs")
        for row in value.get("placements", value.get("rows", [])):
            if row.get("category") == "predicted_bridge_span":
                raise ValueError("GUI plan must obtain joined bridge paths from actual UE import")
            bound, decision = bind_placement(row, catalog, {})
            if bound is None:
                raise ValueError("GUI candidate has an unresolved native reference: " + str(decision))
            placements.append(bound)
        provenance = {"file": os.path.relpath(path.resolve(), ROOT), "path_base": "release_root",
            "sha256": digest_file(path), "schema": value.get("schema")}
        if value.get("agentic_review"):
            decisions = path.parent/value["agentic_review"]["decisions_file"]
            provenance.update(agentic_review=value["agentic_review"], review_decisions_sha256=digest_file(decisions))
        if value.get("scene_review_candidate"):
            provenance["scene_review_candidate"] = value["scene_review_candidate"]
            if value["scene_review_candidate"].get("decisions_file"):
                scene_decisions = path.parent/value["scene_review_candidate"]["decisions_file"]
                provenance.update(scene_review_decisions_sha256=digest_file(scene_decisions),
                    scene_review_decisions_file=os.path.relpath(scene_decisions.resolve(), ROOT))
        sources.append(provenance)
        withheld.extend(value.get("withheld_placements", value.get("withheld", [])))
    if len({row["id"] for row in placements}) != len(placements):
        raise ValueError("GUI candidate native placement identities overlap")
    templates = read(template_manifest)
    if len(templates["material_paths"]) != 4:
        raise ValueError("Joined deck must retain its four native materials")
    for row in templates["templates"]:
        if digest_file(template_manifest.parent/row["template_file"]) != row["template_sha256"]:
            raise ValueError("Joined deck source hash differs")
        if row.get("rotation_deg", [0, 0, 0]) != [0, 0, 0] or row.get("scale", [1, 1, 1]) != [1, 1, 1]:
            raise ValueError("Joined deck must use anchor-only native placement")
    result = {"schema": "citycode.elevated_repair_gui_plan.v1", "ue_executed": False,
        "baseline_manifest_sha256": digest_file(baseline_path), "baseline_scene_id": baseline["scene_id"],
        "template_manifest_sha256": digest_file(template_manifest), "template_count": len(templates["templates"]),
        "native_catalog_sha256": digest_file(catalog_path), "bound_native_placements": placements,
        "native_catalog_assets": list(catalog.values()),
        "proposal_sources": sources, "proposal_decisions": withheld,
        "bridge_bindings_status": "must_import_and_verify_in_current_GUI_before_manifest_creation",
        "unchanged_generated_input_ids": [row["id"] for row in baseline["meshes"]],
        "map_policy": "save as a new dedicated map before any actor mutation; preserve old map and all generated building/background components"}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-proposal", type=Path, required=True)
    parser.add_argument("--centerlines", type=Path, required=True)
    parser.add_argument("--span-manifest", type=Path, required=True)
    parser.add_argument("--joined-manifest", type=Path, help="Final manifold-union GLBs; source spans still supply rail IDs and support height layers")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--props-groups", type=Path, required=True)
    parser.add_argument("--template", "--template-manifest", dest="template", type=Path,
        default=ROOT/"build/ue_citycode_bridge_assets/template_manifest.json", help="Locally exported/regularized native module manifest")
    parser.add_argument("--binding", type=Path, default=ROOT/"data/smallcity_p384_building_assets_completed/binding_manifest_physical.json",
        help="Physical building binding used for support clearance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-proposal", type=Path)
    parser.add_argument("--native-catalog", type=Path)
    parser.add_argument("--props-proposal", type=Path)
    parser.add_argument("--baseline-manifest", type=Path, default=ROOT/"build/ue_citycode/assembly_manifest.json")
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--apply-plan-output", type=Path, help="One-command current-GUI plan; joined bridge binding occurs only after actual UE import")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if not args.output.resolve().is_relative_to((ROOT/"build").resolve()):
        raise ValueError("Candidate enrichment outputs must stay under build/")
    if args.manifest_output and args.manifest_output.resolve() == args.baseline_manifest.resolve():
        raise ValueError("Candidate output must never overwrite the frozen baseline manifest")
    source = read(args.network_proposal); spans = read(args.span_manifest)
    geometry = ActualDeckGeometry(args.span_manifest, args.joined_manifest)
    rails = bridge_railings.generate(args.network_proposal, args.catalog, args.span_manifest, args.output/"rails_before_actual_boundary_filter.json")
    rails = filter_rails(rails, geometry); write(args.output/"bridge_railings_proposal.json", rails)
    enrichment = ROOT/"data/smallcity_p384_scene_enrichment"
    assets = bridge_supports.selected_assets(args.catalog)
    if len(assets) != 4:
        raise ValueError("All four verified native support variants are required")
    roads, *_ = bridge_supports.load_roads(ROOT/"data/smallcity_p384_gt_roads/source/road_sections.geojson",
        ROOT/"data/smallcity_p384_gt_roads/source/lane_centerlines.geojson", ROOT/"data/smallcity_p384_gt_roads/ground_only/road_graph.json")
    buildings = bridge_supports.load_buildings(args.binding)
    land = shape(read(ROOT/"data/smallcity_p384_background_coastal/land.geojson")["geometry"])
    ground = bridge_supports.GroundMaps(enrichment/"props_ground")
    semantics = bridge_supports.PublishedSemantics(ROOT/"data/smallcity_p384_background_history/height/completed")
    template = read(args.template)
    b = np.asarray(template["expected_native_bounds_m"]); thickness = float(b[1, 2]-b[0, 2])
    rows, withheld, stations = bridge_supports.generate(read(args.centerlines)["features"], source["placements"], assets,
        thickness, ground, semantics, land, roads, buildings, read(args.props_groups)["groups"])
    rows, rejected = refit_supports(rows, geometry, assets); withheld += rejected
    counts = Counter(row["source_segment_id"] for row in rows)
    for station in stations:
        station["support_ids"] = [row["placement_id"] for row in rows if row["source_segment_id"] == station["source_segment_id"]]
        station["placed_supports"] = len(station["support_ids"])
        station["status"] = "visual_supports_partial" if station["placed_supports"] else "pending_no_safe_footing"
    supports = {"schema": "citycode.bridge_support_placement_proposal.v1", "coordinate_system": {"world": "UE", "units": "m", "up": "Z"},
        "rows": rows, "withheld": withheld, "spans": stations, "counts": {"supports": len(rows), "spans_with_supports": len(counts)},
        "source_network_proposal_sha256": digest_file(args.network_proposal), "source_span_manifest_sha256": digest_file(args.span_manifest),
        "native_support_catalog_sha256": digest_file(args.catalog), "uses_ground_truth_bridge_coordinates": False,
        "ground_road_geometry_used_for_clearance_only": True, "maximum_actual_underside_cap_gap_m": .35,
        "ue_executed": False, "status": "cpu_candidate_requires_ue_import_and_fresh_saved_map_validation"}
    write(args.output/"bridge_supports_proposal.json", supports)
    inputs = {key: {"path": str(getattr(args, key)), "sha256": digest_file(getattr(args, key))}
              for key in ("network_proposal", "centerlines", "span_manifest", "catalog", "props_groups", "template", "binding")}
    if args.joined_manifest:
        inputs["joined_manifest"] = {"path": str(args.joined_manifest), "sha256": digest_file(args.joined_manifest)}
    summary = {"schema": "citycode.elevated_repair_preparation.v1", "ue_executed": False, "inputs": inputs,
        "rails": len(rails["placements"]), "supports": len(rows), "code_sha256": digest_file(__file__),
        "actual_glb_boundary_and_underside_used": True, "formal_data_changed": False,
        "candidate_manifest_status": "awaiting_actual_ue_native_catalog_and_full_props_proposal"}
    if args.apply_plan_output:
        if not args.joined_manifest or not args.props_proposal or args.native_proposal or args.native_catalog or args.manifest_output:
            raise ValueError("GUI plan needs --joined-manifest and --props-proposal, without final verified-import arguments")
        plan = gui_apply_plan(args.baseline_manifest,
            [args.props_proposal, args.output/"bridge_railings_proposal.json", args.output/"bridge_supports_proposal.json"],
            args.catalog, args.joined_manifest, args.apply_plan_output)
        summary.update(candidate_manifest_status="GUI_plan_ready_actual_bridge_import_pending",
            apply_plan_sha256=digest_file(args.apply_plan_output), bound_native_count=len(plan["bound_native_placements"]))
    optional = (args.native_proposal, args.native_catalog, args.manifest_output)
    if any(optional) and not all(optional):
        raise ValueError("Final candidate needs --native-proposal --native-catalog --props-proposal --manifest-output together")
    if all(optional):
        if not args.props_proposal:
            raise ValueError("Final candidate needs a complete props proposal")
        span_sha = digest_file(args.joined_manifest or args.span_manifest)
        if read(args.native_proposal).get("template_manifest_sha256") != span_sha or any(
                row.get("template_manifest_sha256") != span_sha for row in read(args.native_catalog)["assets"]):
            raise ValueError("Actual UE catalog/proposal must bind this exact candidate span manifest")
        candidate = candidate_manifest(args.baseline_manifest, args.manifest_output,
            [args.props_proposal, args.native_proposal, args.output/"bridge_railings_proposal.json", args.output/"bridge_supports_proposal.json"],
            args.catalog, args.native_catalog)
        summary.update(candidate_manifest_status="prepared_requires_ue_saved_map_validation", candidate_scene_id=candidate["scene_id"],
            candidate_manifest=str(args.manifest_output), candidate_manifest_sha256=digest_file(args.manifest_output))
    write(args.output/"preparation_result.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
