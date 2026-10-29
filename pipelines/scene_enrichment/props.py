"""Deterministic native props on the published SmallCity scene.

Ground-road references are the explicitly accepted GT-derived annotations,
never predicted roads. They supply only 2D curb/approach geometry here. Actual
published coastal-background triangles supply support heights. No GT elevated
geometry, original prop coordinates, or original actor transforms are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from citycode_scene.ground import GroundMaps, build_background_maps


@dataclass(frozen=True)
class Config:
    seed: str = "smallcity-p384-native-props-v2"
    style_density_fraction: float = .55
    building_clearance_m: float = .75
    water_clearance_m: float = 1.0
    carriageway_clearance_m: float = .25
    minimum_confidence: float = .45
    minimum_observed_fraction: float = .75
    ground_cell_m: float = .25
    maximum_ground_relief_m: float = .22
    maximum_detail_above_continuous_m: float = .35
    maximum_slope_deg: float = 6.


PRIORITY = ("traffic_signal", "street_lamp", "street_tree", "parked_car", "bench", "trash_bin")
MIN_SPACING = {"traffic_signal": 38., "street_lamp": 23., "street_tree": 12.,
               "parked_car": 8., "bench": 24., "trash_bin": 19.}
# Offsets are from the actual drivable polygon boundary, not a centreline.
CURB_OFFSETS = {"traffic_signal": (1.6, 2.1), "street_lamp": (2.8, 1.8, 4.3),
                "street_tree": (5.5, 9., 14., 18.), "parked_car": (1.55, 1.8, 2.1),
                "bench": (6., 11., 17., 22.), "trash_bin": (3., 5., 8.)}


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def unit_hash(key):
    return int(digest(key)[:13], 16) / float(16**13)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def ue_rotation(degrees):
    """UE Rotator [pitch, yaw, roll], acting on column vectors."""
    pitch, yaw, roll = np.radians(degrees)
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, -cr*sp*cy-sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, -cr*sp*sy+sr*cy],
                     [sp, -sr*cp, cr*cp]])


def matrix_trs(matrix):
    """Decompose representable UE actor TRS; reject shear/reflection."""
    m = np.asarray(matrix, float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1]):
        raise ValueError("Part matrix must be finite affine 4x4")
    scale = np.linalg.norm(m[:3, :3], axis=0)
    if np.any(scale <= 1e-8):
        raise ValueError("Part matrix has zero scale")
    r = m[:3, :3] / scale
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1., atol=1e-7):
        raise ValueError("Part matrix has shear or reflection; actor TRS cannot preserve it")
    p = math.asin(float(np.clip(r[2, 0], -1, 1)))
    if abs(math.cos(p)) > 1e-7:
        y, roll = math.atan2(r[1, 0], r[0, 0]), math.atan2(-r[2, 1], r[2, 2])
    else:
        y, roll = math.atan2(-r[0, 1], r[1, 1]), 0.
    degrees = np.degrees([p, y, roll])
    if not np.allclose(ue_rotation(degrees) @ np.diag(scale), m[:3, :3], atol=1e-6):
        raise ValueError("Part matrix failed independent Euler reconstruction")
    return m[:3, 3].tolist(), degrees.tolist(), scale.tolist()


def flatten_parts(asset, group_id, position, yaw, scale=1., source=None):
    """Compose every native mesh part, including transparent car assemblies."""
    parent = np.eye(4)
    parent[:3, :3] = ue_rotation([0, yaw, 0]) * scale
    parent[:3, 3] = position
    parts = asset.get("parts") or [{"mesh": asset["mesh"], "local_matrix": np.eye(4).tolist()}]
    result = []
    for i, part in enumerate(parts):
        local = np.asarray(part.get("local_matrix", np.eye(4)), float)
        world = parent @ local
        pos, rot, scl = matrix_trs(world)
        result.append({"placement_id": f"{group_id}_P{i:02d}", "group_id": group_id,
                       "category": asset["category"], "asset_id": asset["asset_id"],
                       "part_index": i, "part_count": len(parts), "mesh_path": part["mesh"],
                       "position_m": pos, "rotation_deg": rot, "scale": scl,
                       "placement_source": "rule_based_road_semantics",
                       "world_matrix": world.tolist(), "local_matrix": local.tolist(),
                       "evidence": source or {}})
    return result


def footprint(asset, xy, yaw, scale=1.):
    lo, hi = np.asarray(asset["bounds_min_m"]), np.asarray(asset["bounds_max_m"])
    corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) * scale
    r = ue_rotation([0, yaw, 0])[:2, :2]
    return MultiPoint(corners @ r.T + np.asarray(xy)).convex_hull


def load_buildings(binding):
    data = json.loads(Path(binding).read_text())
    records = []
    for row in data.get("asset_instances", []) + data.get("massing_instances", []):
        if row.get("assembly_enabled") is False:
            continue
        lods = row.get("lods", row.get("donor_lods", {}))
        bounds = next((v["bounds_local_accessor"] for v in lods.values() if "bounds_local_accessor" in v), None)
        if bounds is None:
            raise ValueError("Building has no geometric bounds: " + row["id"])
        lo, hi = bounds
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        matrix = np.asarray(row["world_from_local_4x4"], float)
        world = corners @ matrix[:3, :3].T + matrix[:3, 3]
        records.append((row["id"], MultiPoint(world[:, :2]).convex_hull))
    if not records:
        raise ValueError("No active building placements")
    return records


def load_roads(sections_path, lanes_path, graph_path):
    rows = json.loads(Path(sections_path).read_text())["features"]
    # Tags separate the user-approved ground City street network from freeway
    # geometry. Numeric GT heights are deliberately never used for placement.
    ground = [r for r in rows if "City" in r["properties"].get("tags", [])
              and "Vehicle" in r["properties"].get("tags", [])
              and not r["properties"].get("is_freeway", False)]
    roads = unary_union([shapely.force_2d(shape(r["geometry"])).buffer(0) for r in ground])
    intersections = unary_union([shapely.force_2d(shape(r["geometry"])).buffer(0) for r in ground
                                 if r["properties"].get("is_intersection")])
    graph = json.loads(Path(graph_path).read_text())
    nodes = [r for r in graph["nodes"] if r.get("degree", 0) >= 3 and r.get("elevation_class") == "ground"]
    node_tree = STRtree([Point(r["world_xy_m"]) for r in nodes])
    lane_rows = json.loads(Path(lanes_path).read_text())["features"]
    section_ids = {r["id"] for r in ground}
    lanes = [LineString(np.asarray(r["geometry"]["coordinates"])[:, :2]) for r in lane_rows
             if r["properties"].get("road_section_id") in section_ids]
    lane_tree = STRtree(lanes)
    curbs = []
    for section in sorted(ground, key=lambda r: r["id"]):
        if section["properties"].get("is_intersection"):
            continue
        poly = shapely.force_2d(shape(section["geometry"])).buffer(0)
        coords = np.asarray(poly.exterior.coords)
        for i, (a, b) in enumerate(zip(coords[:-1], coords[1:])):
            delta, length = b-a, float(np.linalg.norm(b-a))
            if length < 12.:
                continue
            tangent = delta / length
            normal = np.array([-tangent[1], tangent[0]])
            middle = (a+b) / 2
            if poly.contains(Point(middle+normal*.2)):
                normal = -normal
            # Section boundaries internal to a neighbouring road are not curbs.
            if roads.contains(Point(middle+normal*.4)):
                continue
            curbs.append({"id": f"{section['id']}_E{i:03d}", "section_id": section["id"],
                          "a": a, "b": b, "length": length, "tangent": tangent, "normal": normal})
    road_length = sum(float(e.get("length_m", 0)) for e in graph["edges"])
    return roads, intersections, curbs, node_tree, lane_tree, road_length


class PublishedSemantics:
    def __init__(self, directory):
        directory = Path(directory)
        self.fields = {name: np.load(directory / (name+".npy"), mmap_mode="r") for name in
                       ("base_surface_z_m", "base_confidence", "base_provenance", "base_semantic_id")}
        self.matrix = np.asarray(json.loads((directory/"pixel_to_world.json").read_text())["ue_world_xy_to_pixel_center_index_3x3"])

    def sample(self, xy):
        xy = np.asarray(xy, float)
        uv = xy @ self.matrix[:2, :2].T + self.matrix[:2, 2]
        ix, iy = np.rint(uv).astype(int).T
        h, w = self.fields["base_confidence"].shape
        valid = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
        result = {name: np.zeros(len(xy), float) for name in self.fields}
        for name, field in self.fields.items():
            result[name][valid] = field[iy[valid], ix[valid]]
        return result


def support_points(poly, category, pivot):
    # Elevated arms and tree crowns are collision extents, not contact surfaces.
    if category in ("street_tree", "traffic_signal", "street_lamp", "trash_bin"):
        x, y = pivot
        return np.array([[x+dx, y+dy] for dx in (-.2, 0., .2) for dy in (-.2, 0., .2)])
    coords = np.asarray(poly.exterior.coords)[:-1]
    center = np.array(poly.centroid.coords[0])
    # Include edges to catch a narrow terrain ridge/void under a long vehicle.
    samples = [center]
    for a, b in zip(coords, np.roll(coords, -1, axis=0)):
        for t in np.linspace(0, 1, max(3, math.ceil(np.linalg.norm(b-a)/.5))):
            samples.append(a*(1-t)+b*t)
    return np.asarray(samples)


def evaluate_support(points, category, ground, semantics, c):
    sampled = ground.sample(points)
    z = sampled["scene_ground_z"]
    stable = sampled["continuous_z"]
    if (not np.isfinite(z).all() or not np.isfinite(stable).all()
            or np.any(sampled["solid_fallback"]) or not np.all(sampled["on_land"])):
        return None, "missing_actual_ground"
    if np.any(z-stable > c.maximum_detail_above_continuous_m) or z.min() < -.1 or z.max() > 2.5:
        return None, "ground_layer_ambiguous"
    relief = float(np.ptp(z))
    if relief > c.maximum_ground_relief_m:
        return None, "terrain_relief"
    design = np.column_stack((points-points.mean(0), np.ones(len(points))))
    fit = np.linalg.lstsq(design, z, rcond=None)[0]
    slope = float(math.degrees(math.atan(np.linalg.norm(fit[:2]))))
    if slope > c.maximum_slope_deg:
        return None, "terrain_slope"
    s = semantics.sample(points)
    trusted = (np.isin(s["base_semantic_id"], [1, 3])
               & np.isin(s["base_provenance"], [1, 2, 3])
               & (s["base_confidence"] >= c.minimum_confidence)
               & np.isfinite(s["base_surface_z_m"]))
    if trusted.mean() < c.minimum_observed_fraction:
        return None, "insufficient_observed_ground"
    # Highest contact height keeps native wheels/feet above small undulations.
    return {"z_m": float(np.max(z)), "ground_min_m": float(z.min()), "ground_max_m": float(z.max()),
            "relief_m": relief, "slope_deg": slope, "sample_count": len(points),
            "observed_fraction": float(trusted.mean()),
            "source": "actual_coastal_mesh_triangles_25cm_semantic_role_ground"}, None


def overhead_reason(poly, bottom_z, top_z, ground):
    """Check the full crown/arm/body prism against actual high detail samples."""
    lo_x, lo_y, hi_x, hi_y = poly.bounds
    xs = np.arange(lo_x, hi_x+.001, .5)
    ys = np.arange(lo_y, hi_y+.001, .5)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    points = points[shapely.intersects_xy(poly, points[:, 0], points[:, 1])]
    points = np.concatenate((points, np.asarray(poly.exterior.coords),
                             np.asarray(poly.centroid.coords)))
    result = ground.sample(points)
    high = result["detail_max_z"]
    # Low terrain detail was checked separately as contact support. This check
    # concerns actual high surfaces, without substituting a GT bridge model.
    overlap = (np.isfinite(high) & (high > result["scene_ground_z"]+1.)
               & (high >= bottom_z+.2) & (high <= top_z+.5))
    return "overhead_background_collision" if np.any(overlap) else None


class CollisionIndex:
    def __init__(self, buildings, roads, intersections, land, c):
        self.c = c
        self.building_tree = STRtree([p.buffer(c.building_clearance_m) for _, p in buildings])
        self.roads = roads
        self.road_forbidden = roads.buffer(c.carriageway_clearance_m)
        self.intersections = intersections
        self.land_safe = land.buffer(-c.water_clearance_m)
        self.accepted = []
        self.by_category = defaultdict(list)

    def reason(self, poly, category, xy):
        if not self.land_safe.covers(poly):
            return "water_or_shore"
        if len(self.building_tree.query(poly, predicate="intersects")):
            return "building_clearance"
        if category == "traffic_signal":
            if self.road_forbidden.intersects(Point(xy).buffer(.35)):
                return "signal_pole_in_carriageway"
            if not self.roads.intersects(poly):
                return "signal_arm_does_not_reach_road"
        elif self.road_forbidden.intersects(poly):
            return "carriageway_overlap"
        if category == "parked_car":
            if poly.distance(self.roads) > 1.1:
                return "parking_not_curbside"
            if poly.distance(self.intersections) < 5.:
                return "parking_intersection_setback"
        point = Point(xy)
        for old_xy in self.by_category[category]:
            if point.distance(old_xy) < MIN_SPACING[category]:
                return "category_spacing"
        for old_poly, old_category in self.accepted:
            # Reserve crown/arm projections too; do not reduce tree obstacles to trunks.
            gap = .6 if "parked_car" in (category, old_category) else .4
            if old_poly.distance(poly) < gap:
                return "prop_clearance"
        return None

    def add(self, poly, category, xy):
        self.accepted.append((poly, category))
        self.by_category[category].append(Point(xy))


def weighted_asset(assets, category_profile, key):
    weights = category_profile.get("asset_counts", {})
    ordered = sorted(assets, key=lambda a: a["asset_id"])
    weight = np.array([max(1., float(weights.get(a["asset_id"], 1))) for a in ordered])
    index = int(np.searchsorted(np.cumsum(weight), unit_hash(key)*weight.sum(), side="right"))
    return ordered[min(index, len(ordered)-1)]


def candidate_samples(curbs, category, profile, seed, node_tree, intersections, density_fraction=.55):
    rate = float(profile.get("per_100m_road", 0))
    if rate <= 0:
        return []
    candidates = []
    step = max(MIN_SPACING[category], 200. / (rate*density_fraction))
    for curb in curbs:
        key = f"{seed}/{category}/{curb['id']}"
        length = curb["length"]
        if category == "traffic_signal":
            distances = [min(4., length/3), length-min(4., length/3)]
        else:
            margin = 7. if category == "parked_car" else 3.
            start = margin+unit_hash(key+"/phase")*step
            distances = np.arange(start, length-margin, step)
        for sample, distance in enumerate(distances):
            p = curb["a"] + curb["tangent"] * distance
            if category == "traffic_signal":
                nearest = node_tree.geometries[node_tree.nearest(Point(p))]
                if Point(p).distance(nearest) > 28. or Point(p).distance(intersections) > 6.:
                    continue
                if unit_hash(key+f"/{sample}/retain") > density_fraction:
                    continue
            candidates.append({**curb, "key": key+f"/{sample}", "curb_point": p, "along_m": float(distance)})
    return sorted(candidates, key=lambda q: digest(q["key"]+"/priority"))


def generate(catalog, style, buildings, roads_info, land, ground, semantics, c):
    roads, intersections, curbs, node_tree, lane_tree, road_length = roads_info
    collision = CollisionIndex(buildings, roads, intersections, land, c)
    by_category = defaultdict(list)
    for asset in catalog:
        by_category[asset["category"]].append(asset)
    groups, rows, rejected = [], [], Counter()
    attempts = Counter()
    for category in PRIORITY:
        assets = by_category[category]
        if not assets:
            continue
        category_style = style["categories"].get(category, {})
        candidates = candidate_samples(curbs, category, category_style, c.seed, node_tree, intersections,
                                       c.style_density_fraction)
        upper_count = math.floor(road_length/100 * float(category_style.get("per_100m_road", 0)) * c.style_density_fraction)
        for candidate in candidates:
            if len(collision.by_category[category]) >= upper_count:
                break
            attempts[category] += 1
            key = candidate["key"]
            asset = weighted_asset(assets, category_style, key+"/asset")
            scale = .88 + .16*unit_hash(key+"/scale") if category == "street_tree" else 1.
            tangent, normal = candidate["tangent"], candidate["normal"]
            yaw = math.degrees(math.atan2(tangent[1], tangent[0]))
            if category in ("traffic_signal", "street_lamp"):
                # Native positive-Y arms face inward, toward the carriageway.
                if np.dot(np.array([-tangent[1], tangent[0]]), normal) > 0:
                    yaw += 180
            elif category == "street_tree":
                yaw = unit_hash(key+"/yaw")*360
            elif category == "parked_car" and unit_hash(key+"/facing") > .5:
                yaw += 180
            last_reason = "no_feasible_offset"
            for offset in CURB_OFFSETS[category]:
                xy = candidate["curb_point"] + normal*offset
                poly = footprint(asset, xy, yaw, scale)
                last_reason = collision.reason(poly, category, xy)
                if last_reason:
                    continue
                points = support_points(poly, category, xy)
                support, last_reason = evaluate_support(points, category, ground, semantics, c)
                if last_reason:
                    continue
                z = support["z_m"] - float(asset["bounds_min_m"][2])*scale
                last_reason = overhead_reason(poly, support["z_m"],
                                               z+float(asset["bounds_max_m"][2])*scale, ground)
                if last_reason:
                    continue
                lane_index = lane_tree.nearest(Point(xy))
                lane_distance = float(Point(xy).distance(lane_tree.geometries[lane_index]))
                group_id = "PROP_" + digest(key)[:20]
                source = {"road_section_id": candidate["section_id"], "curb_edge_id": candidate["id"],
                          "curb_offset_m": offset, "curb_along_m": candidate["along_m"],
                          "nearest_ground_vehicle_lane_m": lane_distance,
                          "uses_ground_truth_road_reference": True, "support": support}
                new_rows = flatten_parts(asset, group_id, [*xy, z], yaw, scale, source)
                rows.extend(new_rows)
                groups.append({"group_id": group_id, "category": category, "asset_id": asset["asset_id"],
                               "position_m": [float(xy[0]), float(xy[1]), z], "yaw_deg": yaw,
                               "scale": scale, "part_count": len(new_rows),
                               "footprint": mapping(poly), "evidence": source})
                collision.add(poly, category, xy)
                break
            else:
                rejected[f"{category}:{last_reason}"] += 1
    return rows, groups, {"candidate_groups": dict(attempts), "rejected_groups_by_final_reason": dict(rejected),
                          "accepted_groups": dict(Counter(r["category"] for r in groups)),
                          "accepted_mesh_rows": dict(Counter(r["category"] for r in rows)),
                          "ground_reference_road_length_m": road_length, "eligible_curb_edges": len(curbs)}


def validate_groups(groups, catalog, buildings, roads_info, land, c):
    """Independent final-proposal audit from serialized footprints and pivots."""
    roads, intersections, *_ = roads_info
    safe_land = land.buffer(-c.water_clearance_m)
    building_union = unary_union([p for _, p in buildings])
    polygons = [shape(g["footprint"]) for g in groups]
    errors = []
    seen = set()
    for i, (group, poly) in enumerate(zip(groups, polygons)):
        category, xy = group["category"], group["position_m"][:2]
        if group["group_id"] in seen:
            errors.append([group["group_id"], "duplicate_id"])
        seen.add(group["group_id"])
        if not safe_land.covers(poly):
            errors.append([group["group_id"], "water_extent"])
        if poly.distance(building_union) < c.building_clearance_m-1e-6:
            errors.append([group["group_id"], "building_overlap"])
        if category != "traffic_signal" and poly.distance(roads) < c.carriageway_clearance_m-1e-6:
            errors.append([group["group_id"], "carriageway_overlap"])
        if category == "parked_car" and (poly.distance(roads) > 1.1 or poly.distance(intersections) < 5.-1e-6):
            errors.append([group["group_id"], "parking_setback"])
        for j in range(i):
            other = groups[j]
            if poly.distance(polygons[j]) < (.6 if "parked_car" in (category, other["category"]) else .4)-1e-6:
                errors.append([group["group_id"], "prop_overlap", other["group_id"]])
            if category == other["category"] and np.linalg.norm(np.asarray(xy)-other["position_m"][:2]) < MIN_SPACING[category]-1e-6:
                errors.append([group["group_id"], "category_spacing", other["group_id"]])
    return {"schema": "citycode.props_layout_validation.v1", "status": "passed" if not errors else "failed",
            "groups": len(groups), "mesh_rows": sum(g["part_count"] for g in groups), "errors": errors,
            "building_obstacle_count": len(buildings), "full_tree_crown_and_signal_arm_bounds": True,
            "ground_contact_maximum_relief_m": max((g["evidence"]["support"]["relief_m"] for g in groups), default=None)}


def run(args):
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog = [json.loads(l) for l in args.catalog.read_text().splitlines() if l.strip()]
    style = json.loads(args.style.read_text())
    c = Config()
    buildings = load_buildings(args.binding)
    roads_info = load_roads(args.sections, args.lanes, args.graph)
    land_doc = json.loads(args.land.read_text())
    land = shapely.force_2d(shape(land_doc.get("geometry", land_doc)))
    metadata = build_background_maps(ROOT, args.background, args.ground_cache,
                                     cell_m=c.ground_cell_m, land_path=args.land)
    ground = GroundMaps(metadata)
    semantics = PublishedSemantics(args.semantics)
    rows, groups, counts = generate(catalog, style, buildings, roads_info, land, ground, semantics, c)
    validation = validate_groups(groups, catalog, buildings, roads_info, land, c)
    if validation["errors"]:
        raise ValueError("Final layout audit failed: " + str(validation["errors"][:10]))
    sources = {name: {"path": str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path),
                      "sha256": sha(path)} for name, path in
               [("native_asset_catalog", args.catalog), ("old_style_statistics", args.style),
                ("physical_building_binding", args.binding), ("ground_road_sections", args.sections),
                ("ground_vehicle_lanes", args.lanes), ("accepted_ground_road_graph", args.graph),
                ("published_coastal_land", args.land), ("actual_published_background", args.background),
                ("semantic_height_lineage", args.semantics.parent/"LINEAGE.json"),
                ("published_ground_semantic", args.semantics/"base_semantic_id.npy"),
                ("published_ground_confidence", args.semantics/"base_confidence.npy"),
                ("published_ground_provenance", args.semantics/"base_provenance.npy"),
                ("published_ground_finite_support", args.semantics/"base_surface_z_m.npy"),
                ("published_semantic_world_transform", args.semantics/"pixel_to_world.json"),
                ("layout_implementation", Path(__file__))]}
    provenance = {"uses_ground_truth": True, "uses_original_prop_placement_coordinates": False,
                  "road_reference": "User-accepted GT-derived ground-road reference; not predicted roads",
                  "land_reference": "Published GT-derived visible-land extent, 1m sampled; not a native coastline",
                  "height_reference": "Actual published coastal background mesh triangles; continuous ground plus only near-ground detail",
                  "semantic_support": "Published completed background semantics/confidence, only observed/regularized/small-hole ground support",
                  "elevated_reference_geometry_used": False, "sources": sources}
    proposal = {"schema": "citycode.props_placement_proposal.v1", "status": "cpu_validated_proposal",
                "coordinate_system": {"world": "UE", "units": "m", "up": "Z"},
                "placement_source": "rule_based_road_semantics", "provenance": provenance,
                "policy": {**asdict(c), "category_priority": list(PRIORITY), "minimum_category_spacing_m": MIN_SPACING,
                           "curb_offsets_m": CURB_OFFSETS, "matrix_convention": "column-vector parent @ part.local_matrix, translations in metres"},
                "counts": counts, "rows": rows,
                "limitations": ["Geometric roadside parking candidates; parking legality/painted bays are not reconstructed.",
                                "Native vehicle forward direction follows catalog +X assumption; parking is parallel to the curb.",
                                "Ground heights sample actual triangles at 25cm cell centres; residual within-cell terrain variation remains.",
                                "Full asset bounding-box projections conservatively reject tree-crown and signal-arm collisions.",
                                "Actual background highest-detail surfaces are sampled across full prop footprints; hidden intermediate background layers are not reconstructed.",
                                "Traffic signal poles stay outside carriageways; overhead arms intentionally extend over road space."]}
    json_write(out, proposal)
    json_write(out.with_name("props_groups.json"), {"schema": "citycode.props_groups.v1", "groups": groups})
    json_write(out.with_name("props_validation.json"), validation)
    # A sparse plan view exposes the actual placement pattern and exclusions.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import PathPatch
        fig, ax = plt.subplots(figsize=(12, 10))

        def fill_polygon(poly, color, zorder):
            # A citywide road union has many block-shaped holes. Preserve each
            # ring's winding; filling only its exterior would paint all blocks.
            vertices, codes = [], []
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)
                vertices.extend(xy)
                codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO]*(len(xy)-2) + [PlotPath.CLOSEPOLY])
            ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor="none", zorder=zorder))

        coast_parts = list(land.geoms) if hasattr(land, "geoms") else [land]
        for part in coast_parts:
            fill_polygon(part, "#f4f1e8", 0)
        road_parts = list(roads_info[0].geoms) if hasattr(roads_info[0], "geoms") else [roads_info[0]]
        for part in road_parts:
            fill_polygon(part, "#d3d7d8", 1)
        ax.add_collection(LineCollection([np.asarray(poly.exterior.coords) for _, poly in buildings],
                                        colors="#a2a5a5", linewidths=.5, zorder=2))
        colors = {"traffic_signal": "#a33c65", "street_lamp": "#cc8a00", "street_tree": "#327d48",
                  "parked_car": "#337bb9", "bench": "#803f21", "trash_bin": "#55545e"}
        for category in PRIORITY:
            chosen = [g for g in groups if g["category"] == category]
            if chosen:
                xy = np.array([g["position_m"][:2] for g in chosen])
                ax.scatter(xy[:, 0], xy[:, 1], s=13, c=colors[category],
                           label=f"{category}: {len(chosen)}", zorder=4)
        ax.set_aspect("equal"); ax.set_facecolor("#dceaf0")
        ax.set_xlabel("UE X (m)"); ax.set_ylabel("UE Y (m)")
        ax.set_title("Deterministic native props on the published SmallCity scene\n"
                     "Accepted GT-derived ground-road reference; actual coastal-mesh support")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(out.with_name("props_preview.png"), dpi=160); plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({"proposal": str(out), **counts, "validation": validation["status"]}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    d = ROOT/"data"
    e = d/"smallcity_p384_scene_enrichment"
    p.add_argument("--catalog", type=Path, default=e/"props_asset_catalog.jsonl")
    p.add_argument("--style", type=Path, default=e/"props_style_profile.json")
    p.add_argument("--binding", type=Path, default=d/"smallcity_p384_building_assets_completed/binding_manifest_physical.json")
    p.add_argument("--sections", type=Path, default=d/"smallcity_p384_gt_roads/source/road_sections.geojson")
    p.add_argument("--lanes", type=Path, default=d/"smallcity_p384_gt_roads/source/lane_centerlines.geojson")
    p.add_argument("--graph", type=Path, default=d/"smallcity_p384_gt_roads/ground_only/road_graph.json")
    p.add_argument("--land", type=Path, default=d/"smallcity_p384_background_coastal/land.geojson")
    p.add_argument("--background", type=Path, default=d/"smallcity_p384_background_coastal/background.glb")
    p.add_argument("--semantics", type=Path, default=d/"smallcity_p384_background_history/height/completed")
    p.add_argument("--ground-cache", type=Path, default=e/"props_ground")
    p.add_argument("--output", type=Path, default=e/"props_proposal.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
