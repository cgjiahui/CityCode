"""CPU behavior tests for spatial safety, real ground evidence, and assemblies."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import Point, box

from props import (CollisionIndex, Config, evaluate_support, flatten_parts,
                   footprint, load_buildings, matrix_trs, overhead_reason, ue_rotation, weighted_asset)


def asset(category="parked_car", lo=(-2., -1., 0.), hi=(2., 1., 1.5)):
    return {"asset_id": "native-car", "category": category, "mesh": "/Game/Car.Body",
            "bounds_min_m": list(lo), "bounds_max_m": list(hi),
            "parts": [{"mesh": "/Game/Car.Body", "local_matrix": np.eye(4).tolist()},
                      {"mesh": "/Game/Car.Glass", "local_matrix": np.eye(4).tolist()}]}


class Surface:
    def __init__(self, height=lambda xy: np.full(len(xy), .5), detail_offset=0., fallback=False):
        self.height = height
        self.detail_offset = detail_offset
        self.fallback = fallback

    def sample(self, xy):
        z = self.height(np.asarray(xy))
        return {"scene_ground_z": z+self.detail_offset, "continuous_z": z,
                "solid_fallback": np.full(len(xy), self.fallback, bool),
                "on_land": np.ones(len(xy), bool)}


class Semantics:
    def __init__(self, provenance=1, confidence=.8, semantic=3):
        self.provenance, self.confidence, self.semantic = provenance, confidence, semantic

    def sample(self, xy):
        return {"base_provenance": np.full(len(xy), self.provenance),
                "base_confidence": np.full(len(xy), self.confidence),
                "base_semantic_id": np.full(len(xy), self.semantic),
                "base_surface_z_m": np.full(len(xy), .5)}


class NativeAssemblyTests(unittest.TestCase):
    def test_body_and_glass_are_both_kept_with_composed_local_transform(self):
        car = asset()
        # Glass pivot 1m forward and .5m up, with its own 90-degree rotation.
        local = np.eye(4)
        local[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        local[:3, 3] = [1, 0, .5]
        car["parts"][1]["local_matrix"] = local.tolist()
        rows = flatten_parts(car, "PARKING_1", [10, 20, 2], 90, 2.)
        self.assertEqual([r["mesh_path"] for r in rows], ["/Game/Car.Body", "/Game/Car.Glass"])
        self.assertEqual(len({r["placement_id"] for r in rows}), 2)
        np.testing.assert_allclose(rows[1]["position_m"], [10, 22, 3], atol=1e-10)
        # A point one metre along the glass X axis should land two metres west.
        m = np.array(rows[1]["world_matrix"])
        np.testing.assert_allclose(m @ [1, 0, 0, 1], [8, 22, 3, 1], atol=1e-10)
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])
        self.assertEqual(rows[1]["part_count"], 2)

    def test_identity_car_parts_share_exact_transform_and_all_transparency_parts(self):
        rows = flatten_parts(asset(), "car", [-14, 27, .8], 123)
        self.assertEqual(rows[0]["world_matrix"], rows[1]["world_matrix"])
        self.assertEqual(rows[0]["rotation_deg"], rows[1]["rotation_deg"])

    def test_unrepresentable_actor_shear_and_reflection_are_rejected(self):
        shear = np.eye(4); shear[0, 1] = .3
        reflection = np.diag([-1, 1, 1, 1])
        for matrix in (shear, reflection):
            with self.assertRaises(ValueError):
                matrix_trs(matrix)

    def test_euler_convention_preserves_nonzero_pitch_and_roll(self):
        # A UE +90 pitch rotates forward +X into +Z, and +90 roll +Y into -Z.
        np.testing.assert_allclose(ue_rotation([90, 0, 0]) @ [1, 0, 0], [0, 0, 1], atol=1e-10)
        np.testing.assert_allclose(ue_rotation([0, 0, 90]) @ [0, 1, 0], [0, 0, -1], atol=1e-10)
        for rot in ([25, 77, -32], [90, -45, 0], [-90, 30, 0]):
            m = np.eye(4); m[:3, :3] = ue_rotation(rot) @ np.diag([.8, 1.1, 2])
            _, result, scale = matrix_trs(m)
            np.testing.assert_allclose(ue_rotation(result) @ np.diag(scale), m[:3, :3], atol=1e-8)

    def test_asset_weighting_is_stable_under_catalog_order(self):
        a, b = asset(), {**asset(), "asset_id": "another-car"}
        profile = {"asset_counts": {a["asset_id"]: 2, b["asset_id"]: 7}}
        self.assertEqual([weighted_asset([a, b], profile, str(i))["asset_id"] for i in range(100)],
                         [weighted_asset([b, a], profile, str(i))["asset_id"] for i in range(100)])


class SpatialSafetyTests(unittest.TestCase):
    def setUp(self):
        self.c = Config()
        self.roads = box(-80, -4, 80, 4)
        self.intersections = box(-5, -12, 5, 12)

    def index(self, buildings=(), land=None):
        return CollisionIndex(buildings, self.roads, self.intersections,
                              land if land is not None else box(-100, -100, 100, 100), self.c)

    def test_parking_is_parallel_curbside_and_set_back_from_junctions(self):
        car = asset()
        index = self.index()
        self.assertIsNone(index.reason(footprint(car, [30, 5.5], 0), "parked_car", [30, 5.5]))
        self.assertEqual(index.reason(footprint(car, [30, 0], 0), "parked_car", [30, 0]), "carriageway_overlap")
        self.assertEqual(index.reason(footprint(car, [30, 14], 0), "parked_car", [30, 14]), "parking_not_curbside")
        self.assertEqual(index.reason(footprint(car, [7, 5.5], 0), "parked_car", [7, 5.5]), "parking_intersection_setback")

    def test_tree_crown_rejected_even_when_trunk_is_clear_of_water(self):
        tree = asset("street_tree", (-4, -4, 0), (4, 4, 9))
        land = box(-80, -80, 80, 12)
        index = self.index(land=land)
        self.assertTrue(land.contains(Point(30, 10)))
        self.assertEqual(index.reason(footprint(tree, [30, 10], 0), "street_tree", [30, 10]), "water_or_shore")

    def test_signal_arm_cannot_pass_through_building_with_clear_pole(self):
        signal = asset("traffic_signal", (-.5, -1, 0), (.5, 10, 9))
        index = self.index([("house", box(29, -6, 31, -2))])
        # At yaw 180 its full +Y arm reaches from roadside pole into the road.
        poly = footprint(signal, [30, 6], 180)
        self.assertGreater(Point(30, 6).distance(box(29, -6, 31, -2)), 7)
        self.assertEqual(index.reason(poly, "traffic_signal", [30, 6]), "building_clearance")
        self.assertIsNone(self.index().reason(poly, "traffic_signal", [30, 6]))

    def test_priority_reservation_prevents_duplicate_or_crown_overlap(self):
        tree = asset("street_tree", (-3, -3, 0), (3, 3, 8))
        index = self.index()
        poly = footprint(tree, [30, 12], 0)
        index.add(poly, "street_tree", [30, 12])
        bin_mesh = asset("trash_bin", (-.3, -.3, 0), (.3, .3, 1.2))
        self.assertEqual(index.reason(footprint(bin_mesh, [32, 12], 0), "trash_bin", [32, 12]), "prop_clearance")
        self.assertEqual(index.reason(poly, "street_tree", [30, 12]), "category_spacing")

    def test_building_obstacles_use_physical_transform_including_height_axis_swap(self):
        # Source bounds use glTF Y-up. Published world matrix maps Y into Z.
        matrix = [[0, 0, -1, 20], [1, 0, 0, 30], [0, 1, 0, 0], [0, 0, 0, 1]]
        row = {"id": "physical_house", "world_from_local_4x4": matrix,
               "lods": {"0.1M": {"bounds_local_accessor": [[-2, 0, -4], [2, 12, 4]]}}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"binding.json"
            p.write_text(json.dumps({"asset_instances": [row]}))
            records = load_buildings(p)
        np.testing.assert_allclose(records[0][1].bounds, [16, 28, 24, 32])


class GroundEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[x, y] for x in (-1, 0, 1) for y in (-1, 0, 1)])

    def test_actual_triangle_ground_controls_height_not_predicted_depth(self):
        support, reason = evaluate_support(self.points, "parked_car", Surface(lambda xy: .8+.03*xy[:, 0]), Semantics(), Config())
        self.assertIsNone(reason)
        self.assertAlmostEqual(support["z_m"], .83)
        self.assertAlmostEqual(support["relief_m"], .06)

    def test_plinth_fallback_overhead_layer_and_steep_ground_are_rejected(self):
        for surface, expected in [(Surface(fallback=True), "missing_actual_ground"),
                                  (Surface(detail_offset=5), "ground_layer_ambiguous"),
                                  (Surface(lambda xy: .5+.3*xy[:, 0]), "terrain_relief")]:
            _, reason = evaluate_support(self.points, "bench", surface, Semantics(), Config())
            self.assertEqual(reason, expected)

    def test_unobserved_prior_and_water_semantics_cannot_supply_ground_confidence(self):
        for semantics in (Semantics(provenance=5), Semantics(confidence=.2), Semantics(semantic=2)):
            _, reason = evaluate_support(self.points, "street_tree", Surface(), semantics, Config())
            self.assertEqual(reason, "insufficient_observed_ground")

    def test_high_detail_crossing_full_crown_is_rejected_even_with_clear_trunk(self):
        class Background(Surface):
            def sample(self, xy):
                r = super().sample(xy)
                # A 7m bridge crosses the edge of a 9m-tall tree crown,
                # while leaving the trunk's ground-contact footprint clear.
                r["detail_max_z"] = np.where(np.asarray(xy)[:, 0] > 2., 7., .5)
                return r
        self.assertEqual(overhead_reason(box(-3, -3, 3, 3), .5, 9.5, Background()),
                         "overhead_background_collision")
        self.assertIsNone(overhead_reason(box(-3, -3, 3, 3), .5, 2., Background()))


if __name__ == "__main__":
    unittest.main()
