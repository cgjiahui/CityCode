"""Conservative native support candidates for predicted regularized decks.

Deck coordinates and heights come only from predicted regularized modules.
Published buildings, land and accepted ground-road references are exclusion
constraints. Ground support samples actual published background triangles.
These are visual supports, not a structural-engineering certification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from props import (Config, GroundMaps, PublishedSemantics, ROOT, evaluate_support,
                   footprint, json_write, load_buildings, load_roads, sha,
                   support_points, ue_rotation, unit_hash, digest)


WIDTH_M = 2.25
SPACING_M = 30.


def native_bounds(asset):
    bounds = np.asarray(asset["bounds_cm"], float)/100.
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise ValueError("Native support requires finite nonzero XYZ bounds in centimetres")
    return bounds


def fit_support(asset, center_xy, yaw, ground_z, underside_z, width_m=WIDTH_M):
    bounds = native_bounds(asset)
    height = float(underside_z-ground_z)
    if not .75 <= height <= 24.:
        raise ValueError("Support height outside conservative 0.75..24m visual range")
    scale = np.array([width_m/(bounds[1, 0]-bounds[0, 0]),
                      width_m/(bounds[1, 1]-bounds[0, 1]), height/(bounds[1, 2]-bounds[0, 2])])
    center = (bounds[0]+bounds[1])/2
    pivot_xy = np.asarray(center_xy)-ue_rotation([0, yaw, 0])[:2, :2] @ (center[:2]*scale[:2])
    position = np.array([*pivot_xy, ground_z-bounds[0, 2]*scale[2]])
    return position, scale


def deck_frame(module, target_xy, thickness_m):
    start, end = np.asarray(module["start_m"], float), np.asarray(module["end_m"], float)
    delta = end-start
    forward = delta/np.linalg.norm(delta)
    side = np.array([-forward[1], forward[0], 0.]); side /= np.linalg.norm(side)
    up = np.cross(forward, side)
    t = float(np.clip(np.dot(np.asarray(target_xy)-start[:2], delta[:2])/np.dot(delta[:2], delta[:2]), 0, 1))
    top = start+t*delta
    bottom = top-up*thickness_m
    yaw = math.degrees(math.atan2(delta[1], delta[0]))
    return bottom, side, up, yaw


def poly_for_support(asset, center_xy, yaw):
    bounds = native_bounds(asset)
    # Fit a fixed width, including noncentral native pivots and 800cm variants.
    position, scale = fit_support(asset, center_xy, yaw, 0., 4.)
    # props.footprint accepts a uniform scale; apply the small XY native-bound
    # corrections here before rotation instead.
    scaled = {"bounds_min_m": (bounds[0]*scale).tolist(), "bounds_max_m": (bounds[1]*scale).tolist()}
    return footprint(scaled, position[:2], yaw)


def clearance_reason(poly, land, road_tree, building_tree, prop_tree, existing):
    if not land.buffer(-1.).covers(poly):
        return "water_or_shore"
    if len(road_tree.query(poly, predicate="intersects")):
        return "ground_vehicle_corridor"
    if len(building_tree.query(poly, predicate="intersects")):
        return "building_clearance"
    if len(prop_tree.query(poly, predicate="intersects")):
        return "native_prop_clearance"
    if any(poly.distance(old) < 3. for old in existing):
        return "support_spacing"
    return None


def station_points(coordinates, spacing=SPACING_M):
    coordinates = np.asarray(coordinates, float)
    along = np.r_[0., np.cumsum(np.linalg.norm(np.diff(coordinates[:, :2], axis=0), axis=1))]
    if along[-1] <= 1:
        return []
    count = max(1, math.ceil(along[-1]/spacing)-1)
    points = []
    for i, s in enumerate(np.linspace(0, along[-1], count+2)[1:-1]):
        xy = np.array([np.interp(s, along, coordinates[:, axis]) for axis in (0, 1)])
        points.append((i, float(s), xy))
    return points


def selected_assets(path):
    value = json.loads(Path(path).read_text())
    assets = value.get("assets", value) if isinstance(value, dict) else value
    suffixes = ("SM_Support_Square_Large_400_A_01", "SM_Support_Square_Large_800_A_01",
                "SM_Support_Round_ExLarge_400_A_01", "SM_Support_Round_ExLarge_800_A_01")
    return sorted([a for a in assets if a["mesh_path"].split(".")[-1] in suffixes], key=lambda a: a["mesh_path"])


def generate(features, modules, assets, thickness, ground, semantics, land, roads, buildings, props):
    by_span = defaultdict(list)
    for m in modules:
        by_span[m["source_segment_id"]].append(m)
    road_tree = STRtree([roads.buffer(.5)])
    building_tree = STRtree([poly.buffer(.75) for _, poly in buildings])
    prop_tree = STRtree([shape(g["footprint"]).buffer(.5) for g in props])
    existing = []
    rows, withheld, spans = [], [], []
    # Existing bridge surfaces occlude predicted base observations. Actual
    # published continuous triangles are still concrete geometric evidence for
    # a visual footing; record observed_fraction honestly without requiring it.
    c = Config(maximum_ground_relief_m=.15, minimum_observed_fraction=0.)
    for feature in sorted(features, key=lambda f: f["id"]):
        span = feature["id"]
        source_modules = by_span.get(span, [])
        accepted = []
        pending_stations = []
        stations = station_points(feature["geometry"]["coordinates"])
        for station_index, along, target in stations:
            station_id = f"{span}_SUPPORT_STATION_{station_index:03d}"
            if not source_modules:
                withheld.append({"station_id": station_id, "reason": "no_bound_regularized_deck"})
                pending_stations.append(station_id)
                continue
            def distance(module):
                a, b = np.asarray(module["start_m"])[:2], np.asarray(module["end_m"])[:2]
                t = np.clip(np.dot(target-a, b-a)/np.dot(b-a, b-a), 0, 1)
                return np.linalg.norm(target-(a+t*(b-a)))
            module = min(source_modules, key=distance)
            bottom, side, up, yaw = deck_frame(module, target, thickness)
            width = float(module["width_m"])
            branches = [-1, 1] if width >= 12 else [0]
            placed_at_station = 0
            for branch in branches:
                key = station_id+f"_{branch:+d}"
                shape_kind = "Square" if unit_hash(span+"/support-shape") > .35 else "Round"
                matching = [a for a in assets if shape_kind in a["mesh_path"]]
                if not matching:
                    raise ValueError("Native support shape is missing from the verified catalog")
                offsets = [branch*width*.28, branch*width*.14, 0.] if branch else [0., -width*.2, width*.2]
                last_reason = "no_safe_footing"
                for offset in offsets:
                    if abs(offset)+WIDTH_M*math.sqrt(2)/2+.5 > width/2:
                        last_reason = "support_outside_deck_width"
                        continue
                    xy = (bottom+side*offset)[:2]
                    estimated_height = bottom[2]-.5
                    asset = min(matching, key=lambda a: abs((native_bounds(a)[1, 2]-native_bounds(a)[0, 2])-estimated_height))
                    poly = poly_for_support(asset, xy, yaw)
                    last_reason = clearance_reason(poly, land, road_tree, building_tree, prop_tree, existing)
                    if last_reason:
                        continue
                    points = support_points(poly, "bridge_support", xy)
                    support, last_reason = evaluate_support(points, "bridge_support", ground, semantics, c)
                    if last_reason:
                        continue
                    # Rendered underside plane uses the actual template thickness
                    # along deck local +Z, including the deck pitch.
                    corners = np.asarray(poly.exterior.coords)[:-1]
                    underside_z = bottom[2]-(corners-bottom[:2]) @ up[:2]/up[2]
                    cap_z = float(underside_z.min())
                    cap_gap = float(np.ptp(underside_z))
                    if cap_gap > .35:
                        last_reason = "deck_slope_cap_gap"
                        continue
                    try:
                        position, scale = fit_support(asset, xy, yaw, support["z_m"], cap_z)
                    except ValueError:
                        last_reason = "nonpositive_or_excessive_height"
                        continue
                    identity = "BRIDGE_SUPPORT_"+digest(key)[:20]
                    rows.append({"placement_id": identity, "category": "bridge_support",
                                 "asset_id": asset["asset_id"], "mesh_path": asset["mesh_path"],
                                 "position_m": position.tolist(), "rotation_deg": [0., yaw, 0.], "scale": scale.tolist(),
                                 "placement_source": "predicted_elevated_structure",
                                 "source_segment_id": span, "source_deck_placement_id": module["placement_id"],
                                 "station_along_m": along, "lateral_offset_m": offset,
                                 "ground_contact_z_m": support["z_m"], "support_cap_z_m": cap_z,
                                 "support_height_m": cap_z-support["z_m"], "deck_underside_thickness_m": thickness,
                                 "maximum_cap_to_sloped_deck_gap_m": cap_gap,
                                 "native_bounds_m": native_bounds(asset).tolist(), "footprint": poly.__geo_interface__,
                                 "support_evidence": support})
                    existing.append(poly); accepted.append(identity); placed_at_station += 1
                    break
                else:
                    withheld.append({"station_id": station_id, "source_segment_id": span,
                                     "branch": branch, "reason": last_reason,
                                     "station_along_m": along, "proposed_deck_center_xy_m": target.tolist()})
            if not placed_at_station:
                pending_stations.append(station_id)
        spans.append({"source_segment_id": span, "candidate_stations": len(stations),
                      "placed_supports": len(accepted), "support_ids": accepted,
                      "pending_stations": pending_stations,
                      "status": "visual_supports_partial" if accepted else "pending_no_safe_footing",
                      "structural_certification": False})
    return rows, withheld, spans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    e = ROOT/"data/smallcity_p384_scene_enrichment"
    p.add_argument("--catalog", type=Path, default=Path('/tmp/citycode_ue_asset_catalog.json'))
    p.add_argument("--output", type=Path, default=e/"bridge_supports_proposal.json")
    args = p.parse_args()
    paths = {"centerlines": e/"regularized_centerlines.geojson", "modules": e/"regularized_placement_proposal.json",
             "template": ROOT/"build/ue_citycode_bridge_assets/template_manifest.json",
             "building_binding": ROOT/"data/smallcity_p384_building_assets_completed/binding_manifest_physical.json",
             "land": ROOT/"data/smallcity_p384_background_coastal/land.geojson",
             "road_sections": ROOT/"data/smallcity_p384_gt_roads/source/road_sections.geojson",
             "road_lanes": ROOT/"data/smallcity_p384_gt_roads/source/lane_centerlines.geojson",
             "road_graph": ROOT/"data/smallcity_p384_gt_roads/ground_only/road_graph.json",
             "existing_props": e/"props_groups.json"}
    documents = {k: json.loads(v.read_text()) for k, v in paths.items() if k not in ('building_binding','road_sections','road_lanes','road_graph')}
    template_bounds = np.asarray(documents["template"]["expected_native_bounds_m"])
    thickness = float(template_bounds[1, 2]-template_bounds[0, 2])
    # UE import/fixture status is written back to the live template manifest.
    # Freeze the geometry fields actually consumed by this producer so an
    # execution-status update cannot alter otherwise identical proposals.
    contract = e/"bridge_supports_deck_contract.json"
    json_write(contract, {"schema": "citycode.bridge_support_deck_contract.v1",
                          "source_manifest": str(paths["template"].relative_to(ROOT)),
                          "template_geometry_sha256": documents["template"]["template_sha256"],
                          "expected_native_bounds_m": template_bounds.tolist(),
                          "underside_thickness_m": thickness,
                          "position_semantics": "regularized module coordinates are UE XYZ metres deck TOP"})
    paths["template"] = contract
    assets = selected_assets(args.catalog)
    if len(assets) != 4:
        raise ValueError("Expected all four verified square/round 4m/8m native supports")
    json_write(e/"bridge_supports_asset_catalog.json", assets)
    roads, *_ = load_roads(paths["road_sections"], paths["road_lanes"], paths["road_graph"])
    buildings = load_buildings(paths["building_binding"])
    land = shape(documents["land"]["geometry"])
    ground = GroundMaps(e/"props_ground")
    semantics = PublishedSemantics(ROOT/"data/smallcity_p384_background_history/height/completed")
    rows, withheld, spans = generate(documents["centerlines"]["features"], documents["modules"]["placements"],
                                     assets, thickness, ground, semantics, land, roads, buildings,
                                     documents["existing_props"]["groups"])
    min_road = min((shape(r["footprint"]).distance(roads) for r in rows), default=None)
    validation = {"status": "passed", "supports": len(rows), "minimum_ground_vehicle_road_clearance_m": min_road,
                  "native_pivot_contact_errors_m": [], "input_buildings": len(buildings)}
    for row in rows:
        bounds = np.asarray(row["native_bounds_m"])
        base = row["position_m"][2]+bounds[0, 2]*row["scale"][2]
        cap = row["position_m"][2]+bounds[1, 2]*row["scale"][2]
        if abs(base-row["ground_contact_z_m"]) > 1e-8 or abs(cap-row["support_cap_z_m"]) > 1e-8:
            validation["native_pivot_contact_errors_m"].append(row["placement_id"])
    if validation["native_pivot_contact_errors_m"] or (min_road is not None and min_road < .5-1e-6):
        raise ValueError("Serialized support bounds/contact audit failed")
    proposal = {"schema": "citycode.bridge_support_placement_proposal.v1",
                "status": "cpu_validated_visual_supports_partial", "coordinate_system": {"world": "UE", "units": "m", "up": "Z"},
                "placement_source": "predicted_elevated_structure", "rows": rows, "withheld": withheld,
                "spans": spans, "counts": {"supports": len(rows), "spans_with_supports": sum(bool(s['placed_supports']) for s in spans),
                                          "total_spans": len(spans), "withheld_branches": len(withheld),
                                          "withheld_reasons": dict(Counter(w['reason'] for w in withheld))},
                "policy": {"target_station_spacing_m": SPACING_M, "support_width_m": WIDTH_M,
                           "maximum_terrain_relief_m": .15, "ground_road_clearance_m": .5,
                           "deck_underside_thickness_m": thickness, "maximum_support_height_m": 24.,
                           "minimum_observed_prediction_ground_fraction": 0.,
                           "actual_continuous_triangle_support_required": True},
                "provenance": {"uses_ground_truth_bridge_coordinates": False,
                               "coordinates_source": "Prediction-only regularized deck module geometry",
                               "uses_ground_truth_road_geometry_for_clearance_only": True,
                               "ground_road_reference": "User-accepted GT-derived ground-road annotation; not a prediction",
                               "ground": "Actual published continuous coastal mesh triangles; observed prediction semantic/confidence fraction is reported separately and may be zero under occluders",
                               "land": "Published GT-derived coastal land exclusion extent",
                               "sources": {k: {"path": str(v.relative_to(ROOT)), "sha256": sha(v)} for k, v in paths.items()},
                               "native_support_catalog_sha256": sha(e/"bridge_supports_asset_catalog.json"),
                               "code_sha256": sha(__file__)},
                "limitations": ["Visual supports are not a structural certification; blocked stations remain explicitly pending.",
                                "Columns contact the lowest underside point across their cap; residual slope gap is bounded at 0.35m.",
                                "Strict actual-ground, water, building and vehicle-corridor exclusions can leave spans without supports.",
                                "Actual published ground can exist under a prediction occluder; a footing does not claim that ground was directly predicted or observed."]}
    json_write(args.output, proposal)
    json_write(e/"bridge_supports_validation.json", validation)
    print(json.dumps({**proposal["counts"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
