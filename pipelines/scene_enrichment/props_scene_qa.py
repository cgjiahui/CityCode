"""Replayable CPU review of published native props against scene geometry.

The historical layout and its provenance remain immutable.  The explicitly
authorized ground-road reference is used in XY only; no reference freeway,
bridge geometry, or original actor transform is consumed.  Deck checks use
actual prediction-derived span triangles and native props use full bounds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Point, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from citycode_assets.registration import read_glb_scene_geometry
from citycode_scene.geometry import Raster
from props import Config, MIN_SPACING, PublishedSemantics, json_write, load_roads, sha, support_points, ue_rotation
from review_props import obb, obb_penetration

E = ROOT / "data/smallcity_p384_scene_enrichment"


def corners(bounds):
    bounds = np.asarray(bounds, float)
    return np.asarray(list(itertools.product(*zip(bounds[0], bounds[1]))), float)


def transform(rotation, scale, position):
    matrix = np.eye(4)
    matrix[:3, :3] = ue_rotation(rotation) @ np.diag(np.broadcast_to(scale, 3))
    matrix[:3, 3] = position
    if not np.isfinite(matrix).all() or np.any(np.broadcast_to(scale, 3) <= 0):
        raise ValueError("Native proposal requires finite, positive actor TRS")
    return matrix


def group_geometry(group, asset):
    rotation = group.get("rotation_deg", [0., group["yaw_deg"], 0.])
    matrix = transform(rotation, group["scale"], group["position_m"])
    bounds = np.array([asset["bounds_min_m"], asset["bounds_max_m"]])
    vertices = corners(bounds) @ matrix[:3, :3].T + matrix[:3, 3]
    return matrix, MultiPoint(vertices[:, :2]).convex_hull, [vertices.min(0), vertices.max(0)]


class ActualGround:
    """Sample only actual triangle rasters, without land-mask/solid fallback."""
    def __init__(self, metadata):
        self.path = Path(metadata)
        self.meta = json.loads(self.path.read_text())
        self.origin = np.asarray(self.meta["origin_xy"], float)
        self.cell = float(self.meta["cell_m"])
        self.paths = {k: self.path.parent / v for k, v in self.meta["maps"].items()}
        self.maps = {k: np.load(v, mmap_mode="r") for k, v in self.paths.items()}

    def sample(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        ij = np.floor((xy - self.origin) / self.cell).astype(int)
        h, w = self.maps["continuous_top"].shape
        valid = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < w) & (ij[:, 1] < h)
        values = []
        for key in ("continuous_top", "detail_near_ground_top"):
            z = np.full(len(xy), np.nan)
            z[valid] = self.maps[key][ij[valid, 1], ij[valid, 0]]
            z[~np.isfinite(z)] = np.nan
            values.append(z)
        # fmax permits a real detail triangle where the continuous raster has a
        # hole, without inventing a synthetic land cap.
        return np.fmax(*values)


def ground_contacts(group, asset, ground):
    points, plane_z, bottom_vertices = bottom_contact_proxy(group, asset)
    z = ground.sample(points)
    finite = np.isfinite(z)
    gaps = plane_z[finite] - z[finite]
    return {"sample_count": len(z), "missing_actual_ground_samples": int((~finite).sum()),
            "contact_geometry": "native_bounds_bottom_plane_proxy_not_exported_tire_or_foot_mesh",
            "native_bottom_z_m": float(bottom_vertices[:, 2].min()),
            "ground_min_m": float(z[finite].min()) if finite.any() else None,
            "ground_max_m": float(z[finite].max()) if finite.any() else None,
            "minimum_gap_m": float(gaps.min()) if finite.any() else None,
            "median_gap_m": float(np.median(gaps)) if finite.any() else None,
            "maximum_gap_m": float(gaps.max()) if finite.any() else None,
            "whole_group_floating_over_0_15m": bool(finite.all() and gaps.min() > .15),
            "local_contact_gap_over_0_25m": bool(finite.any() and gaps.max() > .25),
            "buried_over_0_05m": bool(finite.any() and gaps.min() < -.05)}


def bottom_contact_proxy(group, asset):
    matrix = group_geometry(group, asset)[0]
    lo, hi = asset["bounds_min_m"], asset["bounds_max_m"]
    vertices = np.asarray([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    poly = MultiPoint(vertices[:, :2]).convex_hull
    points = support_points(poly, group["category"], group["position_m"][:2])
    normal = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    if normal[2] < .95:
        raise ValueError("Bottom-plane contact proxy is limited to small upright tilts")
    plane_z = vertices[0, 2] - (points - vertices[0, :2]) @ normal[:2] / normal[2]
    return points, plane_z, vertices


def part_check(group, asset, rows):
    parent = group_geometry(group, asset)[0]
    expected = asset.get("parts") or [{"mesh": asset["mesh"], "local_matrix": np.eye(4).tolist()}]
    rows = sorted(rows, key=lambda row: row["part_index"])
    issues, maximum_error = [], 0.
    if len(rows) != len(expected) or [r["part_index"] for r in rows] != list(range(len(expected))):
        issues.append("missing_or_duplicate_native_part")
    for row, part in zip(rows, expected):
        actual = transform(row["rotation_deg"], row["scale"], row["position_m"])
        intended = parent @ np.asarray(part.get("local_matrix", np.eye(4)), float)
        maximum_error = max(maximum_error, float(np.abs(actual - intended).max()))
        if row["mesh_path"] != part["mesh"]:
            issues.append("native_part_mesh_mismatch")
        if row.get("part_count") != len(expected):
            issues.append("native_part_count_metadata_mismatch")
    if maximum_error > 1e-6:
        issues.append("parent_local_native_transform_mismatch")
    return {"part_count": len(rows), "maximum_transform_error": maximum_error, "issues": issues}


def axial_angle_degrees(first, second):
    return abs((first - second + 90.) % 180. - 90.)


def lane_heading(lane, point):
    station = lane.project(point)
    a, b = lane.interpolate(max(0., station - 1.)), lane.interpolate(min(lane.length, station + 1.))
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def load_building_boxes(path):
    doc = json.loads(Path(path).read_text())
    output = []
    for row in doc.get("asset_instances", []) + doc.get("massing_instances", []):
        if not row.get("assembly_enabled", True):
            continue
        lods = row.get("lods", row.get("donor_lods", {}))
        value = lods.get("0.25M", next(iter(lods.values())))
        matrix = np.asarray(row["world_from_local_4x4"], float)
        v = corners(value["bounds_local_accessor"]) @ matrix[:3, :3].T + matrix[:3, 3]
        output.append({"id": row["id"], "poly": MultiPoint(v[:, :2]).convex_hull,
                       "zmin": float(v[:, 2].min()), "zmax": float(v[:, 2].max())})
    return output


def native_obstacles(supports_path, railings_path):
    obstacles = []
    supports = json.loads(Path(supports_path).read_text())
    for row in supports.get("rows", supports.get("placements", [])):
        obstacles.append(("support", row["placement_id"], obb(row["native_bounds_m"], row["position_m"], row["rotation_deg"], row["scale"])))
    rails = json.loads(Path(railings_path).read_text())
    bounds = np.asarray(rails["native_asset"]["bounds_cm"], float) / 100.
    for row in rails["placements"]:
        obstacles.append(("railing", row["placement_id"], obb(bounds, row["position_m"], row["rotation_deg"], row["scale"])))
    return obstacles


def obb_xy_bounds(value):
    radius = np.abs(value["axes"]) @ value["half"]
    lo, hi = value["center"] - radius, value["center"] + radius
    return shapely.box(lo[0], lo[1], hi[0], hi[1])


def input_identity(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_span_world(row, manifest_parent):
    if not np.allclose(row["rotation_deg"], [0., 0., 0.], atol=1e-9) or not np.allclose(row["scale"], [1., 1., 1.], atol=1e-9):
        raise ValueError("Native span contract permits only the documented anchor translation")
    path = Path(manifest_parent) / row["template_file"]
    if sha(path) != row["template_sha256"]:
        raise ValueError("Span geometry does not match its immutable candidate manifest")
    scene = read_glb_scene_geometry(path)
    vertices = scene["vertices"][:, [0, 2, 1]] + np.asarray(row["position_m"])
    if sha(path) != row["template_sha256"]:
        raise ValueError("Span geometry changed during its actual-triangle read")
    return path, vertices, scene["faces"]


def column_intervals(columns, x, y):
    h, w = columns["shape"]
    if not (0 <= x < w and 0 <= y < h):
        return [], False
    index = y * w + x
    a, b = columns["offsets"][index:index + 2]
    hits = columns["hits"][int(a):int(b)]
    if len(hits) % 2:
        return [], True
    return list(zip(hits[::2], hits[1::2])), False


def deck_overlap(poly, zlow, zhigh, columns, tolerance=.2):
    """Actual triangle crossing intervals against a conservative prop prism."""
    cell = float(columns["cell"])
    origin = columns["origin"]
    lo = np.floor((np.array(poly.bounds[:2]) - origin) / cell).astype(int)
    hi = np.floor((np.array(poly.bounds[2:]) - origin) / cell).astype(int)
    height, width = columns["shape"]
    depths, unknown, minimum_clearance = [], 0, None
    for y in range(max(0, lo[1]), min(height - 1, hi[1]) + 1):
        for x in range(max(0, lo[0]), min(width - 1, hi[0]) + 1):
            xy = origin + (np.array([x, y]) + .5) * cell
            if not poly.covers(Point(xy)):
                continue
            intervals, odd = column_intervals(columns, x, y)
            unknown += int(odd)
            for low, high in intervals:
                if low >= zhigh:
                    gap = float(low - zhigh)
                    minimum_clearance = gap if minimum_clearance is None else min(minimum_clearance, gap)
                depth = float(min(high, zhigh) - max(low, zlow))
                if depth > tolerance:
                    depths.append(depth)
    return {"proxy_overlap_sample_area_m2": len(depths) * cell ** 2,
            "proxy_overlap_volume_m3": sum(depths) * cell ** 2,
            "maximum_vertical_overlap_m": max(depths, default=0.),
            "uncertain_open_triangle_columns": unknown,
            "minimum_sampled_overhead_clearance_m": minimum_clearance}


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=E / "props_reviewed_proposal.json")
    parser.add_argument("--groups", type=Path, default=E / "props_reviewed_groups.json")
    parser.add_argument("--catalog", type=Path, default=E / "props_asset_catalog.jsonl")
    parser.add_argument("--ground", type=Path, default=E / "props_ground/BACKGROUND_SAMPLING.json")
    parser.add_argument("--binding", type=Path, default=ROOT / "data/smallcity_p384_building_assets_completed/binding_manifest_physical.json")
    parser.add_argument("--supports", type=Path, default=E / "bridge_supports_proposal.json")
    parser.add_argument("--railings", type=Path, default=E / "bridge_railings_proposal.json")
    parser.add_argument("--spans", type=Path, default=ROOT / "build/ue_citycode_bridge_assets/spans/span_manifest.json")
    parser.add_argument("--land", type=Path, default=ROOT / "data/smallcity_p384_background_coastal/land.geojson")
    parser.add_argument("--semantics", type=Path, default=ROOT / "data/smallcity_p384_background_history/height/completed")
    parser.add_argument("--output", type=Path, default=E / "props_scene_qa_current.json")
    args = parser.parse_args(args)
    proposal, group_doc = json.loads(args.proposal.read_text()), json.loads(args.groups.read_text())
    groups = group_doc["groups"]
    rows = proposal.get("rows", proposal.get("placements", []))
    if len({g["group_id"] for g in groups}) != len(groups) or len({r["placement_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate group/placement identity in immutable input")
    catalog = {a["asset_id"]: a for a in map(json.loads, args.catalog.read_text().splitlines())}
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    if set(by_group) != {g["group_id"] for g in groups}:
        raise ValueError("Serialized groups and flattened rows do not cover the same placements")
    paths = {k: getattr(args, k) for k in ("proposal", "groups", "catalog", "ground", "binding", "supports", "railings", "spans", "land")}
    for name in ("base_surface_z_m", "base_confidence", "base_provenance", "base_semantic_id", "pixel_to_world"):
        paths["published_ground_" + name] = args.semantics / (name + (".json" if name == "pixel_to_world" else ".npy"))
    road_root = ROOT / "data/smallcity_p384_gt_roads"
    paths.update(ground_road_sections=road_root / "source/road_sections.geojson",
                 ground_vehicle_lanes=road_root / "source/lane_centerlines.geojson",
                 ground_graph=road_root / "ground_only/road_graph.json")
    input_start_hashes = {name: sha(path) for name, path in paths.items()}
    ground = ActualGround(args.ground)
    semantics = PublishedSemantics(args.semantics)
    land_doc = json.loads(args.land.read_text())
    land_safe = shapely.force_2d(shape(land_doc.get("geometry", land_doc))).buffer(-1.)
    paths.update({"actual_ground_" + k: v for k, v in ground.paths.items()})
    roads = load_roads(paths["ground_road_sections"], paths["ground_vehicle_lanes"], paths["ground_graph"])
    buildings = load_building_boxes(args.binding)
    building_tree = STRtree([b["poly"] for b in buildings])
    obstacles = native_obstacles(args.supports, args.railings)
    obstacle_tree = STRtree([obb_xy_bounds(value) for _, _, value in obstacles])
    records, errors, advisory = [], [], []
    geometries = []
    for group in groups:
        identity, category = group["group_id"], group["category"]
        asset = catalog[group["asset_id"]]
        _, poly, bound = group_geometry(group, asset)
        geometries.append((poly, bound))
        contacts = ground_contacts(group, asset, ground)
        parts = part_check(group, asset, by_group[identity])
        issues = list(parts["issues"])
        semantic_support = semantics.sample(bottom_contact_proxy(group, asset)[0])
        trusted = (np.isin(semantic_support["base_semantic_id"], [1, 3]) & np.isin(semantic_support["base_provenance"], [1, 2, 3])
                   & (semantic_support["base_confidence"] >= .45) & np.isfinite(semantic_support["base_surface_z_m"]))
        if trusted.mean() < .75:
            issues.append("insufficient_published_observed_ground")
        if not land_safe.covers(poly):
            issues.append("water_or_shore_clearance")
        if contacts["missing_actual_ground_samples"] or contacts["whole_group_floating_over_0_15m"] or contacts["local_contact_gap_over_0_25m"] or contacts["buried_over_0_05m"]:
            issues.append("ground_contact_requires_review")
        building_distance = min(poly.distance(b["poly"]) for b in buildings)
        collision_buildings = [buildings[int(i)]["id"] for i in building_tree.query(poly, predicate="intersects")
                               if min(bound[1][2], buildings[int(i)]["zmax"]) - max(bound[0][2], buildings[int(i)]["zmin"]) > .2]
        if collision_buildings:
            issues.append("building_full_bounds_overlap")
        own_box = obb([asset["bounds_min_m"], asset["bounds_max_m"]], group["position_m"],
                      group.get("rotation_deg", [0., group["yaw_deg"], 0.]), [group["scale"]] * 3)
        obstacle_contacts = []
        for obstacle_index in obstacle_tree.query(poly, predicate="intersects"):
            kind, old_id, value = obstacles[int(obstacle_index)]
            depth = obb_penetration(own_box, value)
            if depth > .2:
                obstacle_contacts.append({"kind": kind, "placement_id": old_id, "minimum_sat_overlap_m": depth})
        if obstacle_contacts:
            issues.append("native_bridge_obstacle_bounds_overlap")
        road_distance = poly.distance(roads[0])
        point = Point(group["position_m"][:2])
        if category == "traffic_signal":
            if point.buffer(.35).intersects(roads[0].buffer(.25)):
                issues.append("signal_pole_in_ground_carriageway")
            if not poly.intersects(roads[0]):
                issues.append("signal_arm_does_not_reach_ground_road")
        elif road_distance < .25 - 1e-6:
            issues.append("ground_carriageway_overlap")
        heading = None
        if category == "parked_car":
            lane = roads[4].geometries[roads[4].nearest(point)]
            heading = axial_angle_degrees(group["yaw_deg"], lane_heading(lane, point))
            if heading > 15.:
                issues.append("parking_direction_not_parallel_to_nearest_ground_lane")
            if road_distance > 1.1 or poly.distance(roads[1]) < 5. - 1e-6:
                issues.append("parking_curb_or_intersection_setback")
            if contacts["maximum_gap_m"] is not None and contacts["maximum_gap_m"] > .15:
                advisory.append({"group_id": identity, "reason": "vehicle_bounds_contact_proxy_is_not_exact_wheel_contact", "maximum_gap_m": contacts["maximum_gap_m"]})
        records.append({"group_id": identity, "category": category, "ground": contacts, "native_parts": parts,
                        "published_trusted_ground_fraction": float(trusted.mean()),
                        "minimum_building_bounds_distance_m": building_distance, "building_bounds_contacts": collision_buildings,
                        "ground_carriageway_distance_m": road_distance, "car_heading_error_deg_modulo_180": heading,
                        "native_bridge_obstacle_contacts": obstacle_contacts, "deck_contacts": [], "issues": issues})
    prop_tree = STRtree([p for p, _ in geometries])
    for first, group in enumerate(groups):
        poly = geometries[first][0]
        for second in prop_tree.query(poly.buffer(.6), predicate="intersects"):
            second = int(second)
            if second <= first:
                continue
            threshold = .6 if "parked_car" in (group["category"], groups[second]["category"]) else .4
            if poly.distance(geometries[second][0]) < threshold - 1e-6:
                records[first]["issues"].append("other_prop_full_bounds_clearance")
                records[second]["issues"].append("other_prop_full_bounds_clearance")
        for second in range(first + 1, len(groups)):
            if group["category"] == groups[second]["category"] and np.linalg.norm(np.asarray(group["position_m"][:2]) - groups[second]["position_m"][:2]) < MIN_SPACING[group["category"]] - 1e-6:
                records[first]["issues"].append("same_category_spacing")
                records[second]["issues"].append("same_category_spacing")
    span_doc = json.loads(args.spans.read_text())
    if span_doc.get("uses_ground_truth_for_placement") is not False or span_doc.get("uses_original_bridge_actor_transforms") is not False:
        raise ValueError("Bridge checks require explicitly prediction-derived span geometry")
    raster = Raster(Path("/tmp/citycode_props_scene_qa_raster"))
    for span in span_doc["templates"]:
        path, v, faces = read_span_world(span, args.spans.parent)
        paths["span_" + span["span_id"]] = path
        # Frozen native span contract: standard glTF numeric X,Z,Y, anchor UE m.
        sb = np.array([v.min(0), v.max(0)])
        candidates = [(i, p, b) for i, (p, b) in enumerate(geometries)
                      if p.intersects(shapely.box(*sb[:, :2].ravel())) and min(b[1][2], sb[1, 2]) - max(b[0][2], sb[0, 2]) > .2]
        if not candidates:
            continue
        columns = raster.cast(v, faces, .25)
        for index, poly, bound in candidates:
            result = deck_overlap(poly, bound[0][2], bound[1][2], columns)
            if result["maximum_vertical_overlap_m"] > .2 or result["uncertain_open_triangle_columns"]:
                result["span_id"] = span["span_id"]
                records[index]["deck_contacts"].append(result)
                records[index]["issues"].append("predicted_deck_vs_full_prop_bounds_requires_review")
    for record in records:
        if record["issues"]:
            errors.append({"group_id": record["group_id"], "issues": record["issues"]})
    inputs = {name: {"path": input_identity(path), "sha256": sha(path)} for name, path in paths.items()}
    if any(inputs[name]["sha256"] != digest for name, digest in input_start_hashes.items()):
        raise ValueError("A scene input changed during QA; wait for its frozen candidate and replay")
    if any(inputs["span_" + row["span_id"]]["sha256"] != row["template_sha256"] for row in span_doc["templates"]):
        raise ValueError("An actual span changed after it was sampled; candidate QA is invalid")
    report = {"schema": "citycode.props_scene_qa.v1", "status": "requires_review" if errors else "passed_with_explicit_contact_limits",
              "coordinate_system": {"world": "UE", "units": "m", "up": "Z"}, "ue_executed": False,
              "formal_inputs_modified": False, "source_code_sha256": sha(__file__), "inputs": inputs,
              "counts": {"groups": len(groups), "mesh_rows": len(rows), "groups_by_category": dict(Counter(g["category"] for g in groups)),
                         "building_obstacles": len(buildings), "bridge_supports": sum(k == "support" for k, _, _ in obstacles),
                         "bridge_railings": sum(k == "railing" for k, _, _ in obstacles), "predicted_spans": len(span_doc["templates"]),
                         "groups_requiring_review": len(errors), "advisory_local_vehicle_contact_groups": len(advisory)},
              "provenance": {"road_reference": "Previously user-authorized GT-derived GROUND City vehicle reference in XY only; never called predicted; freeway/elevated sections excluded",
                             "bridge_source": "Published prediction-derived actual span triangles and native support/rail proposals; no GT bridge placements",
                             "ground_source": "Published actual coastal mesh continuous/near-ground triangle raster, with no land polygon or synthetic solid-cap fallback",
                             "historical_layout_uses_ground_truth_road_reference": proposal.get("provenance", {}).get("uses_ground_truth"),
                             "source_background_sha256_from_sampling": ground.meta["source_sha256"]},
              "thresholds_m": {"gross_entire_object_gap": .15, "local_support_gap": .25, "burial": .05, "bounds_contact_tolerance": .2},
              "summary": {"whole_group_floating": sum(r["ground"]["whole_group_floating_over_0_15m"] for r in records),
                          "buried": sum(r["ground"]["buried_over_0_05m"] for r in records),
                          "missing_actual_ground_groups": sum(bool(r["ground"]["missing_actual_ground_samples"]) for r in records),
                          "maximum_local_support_gap_m": max(r["ground"]["maximum_gap_m"] for r in records if r["ground"]["maximum_gap_m"] is not None),
                          "maximum_car_axial_heading_error_deg": max((r["car_heading_error_deg_modulo_180"] or 0.) for r in records),
                          "minimum_building_bounds_clearance_m": min(r["minimum_building_bounds_distance_m"] for r in records)},
              "limitations": ["Native props, trees, signal arms, rails and supports use conservative full native bounds; a flagged proxy overlap is not a visible mesh collision claim.",
                              "Car base contacts are bounds-derived support samples, not exported tire mesh contact patches; upright vehicles can have local gaps over uneven ground.",
                              "Ground raster is the frozen published coastal mesh sampling; a changed ground candidate must supply a replacement sampling metadata file.",
                              "All bridge/ground candidate hashes must be refreshed together before this becomes post-repair validation."],
              "errors": errors, "advisory": advisory, "records": records}
    json_write(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"], "counts": report["counts"]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
