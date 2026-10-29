import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh
from shapely.geometry import box

from props_scene_qa import ActualGround, axial_angle_degrees, bottom_contact_proxy, deck_overlap, ground_contacts, part_check, read_span_world, sha
from review_scene_props import tilt_to_ground


class ActualPropSceneTest(unittest.TestCase):
    def test_ground_never_fills_a_hole_with_a_synthetic_land_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'continuous.npy', np.array([[0., -np.inf], [.2, 0.]], np.float32))
            np.save(root / 'detail.npy', np.full((2, 2), -np.inf, np.float32))
            meta = {'origin_xy': [0., 0.], 'cell_m': 1., 'maps': {'continuous_top': 'continuous.npy', 'detail_near_ground_top': 'detail.npy'},
                    'solid_cap_m': 0., 'land_geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}}
            (root / 'sampling.json').write_text(json.dumps(meta))
            actual = ActualGround(root / 'sampling.json').sample([[.5, .5], [1.5, .5], [-.01, .5]])
            self.assertEqual(actual[0], 0.)
            self.assertTrue(np.isnan(actual[1:]).all())

    def test_flat_support_distinguishes_local_unevenness_from_entire_floating(self):
        group = {'category': 'street_lamp', 'position_m': [0., 0., .2], 'yaw_deg': 0., 'scale': 1.}
        asset = {'bounds_min_m': [-.1, -.1, 0.], 'bounds_max_m': [.1, .1, 8.]}
        class Ground:
            def sample(self, points):
                return np.where(points[:, 0] < 0, 0., .2)
        result = ground_contacts(group, asset, Ground())
        self.assertEqual(result['minimum_gap_m'], 0.)
        self.assertAlmostEqual(result['maximum_gap_m'], .2)
        self.assertFalse(result['whole_group_floating_over_0_15m'])
        group['position_m'][2] += .3
        self.assertTrue(ground_contacts(group, asset, Ground())['whole_group_floating_over_0_15m'])

    def test_grade_separated_deck_does_not_collide_with_ground_vehicle(self):
        columns = {'cell': .25, 'origin': np.array([0., 0.]), 'shape': [2, 2],
                   'offsets': np.arange(0, 9, 2), 'hits': np.array([10., 11.] * 4)}
        poly = box(0., 0., .5, .5)
        car = deck_overlap(poly, 0., 2., columns)
        self.assertEqual(car['maximum_vertical_overlap_m'], 0.)
        self.assertEqual(car['minimum_sampled_overhead_clearance_m'], 8.)
        pole = deck_overlap(poly, 0., 10.5, columns)
        self.assertAlmostEqual(pole['maximum_vertical_overlap_m'], .5)
        self.assertAlmostEqual(pole['proxy_overlap_sample_area_m2'], .25)

    def test_bridge_column_gap_is_not_filled_between_separate_levels(self):
        columns = {'cell': .25, 'origin': np.array([0., 0.]), 'shape': [1, 1],
                   'offsets': np.array([0, 4]), 'hits': np.array([3., 4., 10., 11.])}
        result = deck_overlap(box(0., 0., .25, .25), 5., 7., columns)
        self.assertEqual(result['maximum_vertical_overlap_m'], 0.)

    def test_glass_must_follow_body_local_transform_and_not_be_omitted(self):
        group = {'position_m': [10., 20., 0.], 'yaw_deg': 0., 'scale': 1.}
        local = np.eye(4); local[:3, 3] = [0., 0., 1.]
        asset = {'mesh': 'body', 'bounds_min_m': [-2., -1., 0.], 'bounds_max_m': [2., 1., 2.],
                 'parts': [{'mesh': 'body', 'local_matrix': np.eye(4).tolist()}, {'mesh': 'glass', 'local_matrix': local.tolist()}]}
        rows = [{'part_index': i, 'part_count': 2, 'mesh_path': mesh, 'position_m': [10., 20., float(i)], 'rotation_deg': [0., 0., 0.], 'scale': [1., 1., 1.]}
                for i, mesh in enumerate(['body', 'glass'])]
        self.assertEqual(part_check(group, asset, rows)['issues'], [])
        self.assertIn('missing_or_duplicate_native_part', part_check(group, asset, rows[:1])['issues'])
        rows[1]['position_m'][2] = 0.
        self.assertIn('parent_local_native_transform_mismatch', part_check(group, asset, rows)['issues'])

    def test_parked_car_axis_allows_opposite_facing_but_not_crosswise(self):
        self.assertAlmostEqual(axial_angle_degrees(181., 1.), 0.)
        self.assertAlmostEqual(axial_angle_degrees(-89., 1.), 90.)

    def test_tilted_contact_proxy_uses_transformed_bottom_plane(self):
        group = {'category': 'parked_car', 'position_m': [0., 0., 0.], 'yaw_deg': 0.,
                 'rotation_deg': [2., 0., -1.], 'scale': 1.}
        asset = {'bounds_min_m': [-2., -1., 0.], 'bounds_max_m': [2., 1., 2.]}
        points, heights, _ = bottom_contact_proxy(group, asset)
        # Independently derived analytic ground plane for UE pitch/roll.
        p, r = np.radians([2., -1.])
        class Ground:
            def sample(self, xy):
                return np.tan(p) * xy[:, 0] - np.tan(r) / np.cos(p) * xy[:, 1]
        np.testing.assert_allclose(heights, Ground().sample(points), atol=1e-12)
        result = ground_contacts(group, asset, Ground())
        self.assertLess(abs(result['minimum_gap_m']), 1e-12)
        self.assertLess(abs(result['maximum_gap_m']), 1e-12)

    def test_vehicle_candidate_reduces_plane_gap_without_moving_parking_xy_or_yaw(self):
        asset = {'bounds_min_m': [-2., -1., 0.], 'bounds_max_m': [2., 1., 2.]}
        group = {'category': 'parked_car', 'position_m': [0., 0., .08], 'yaw_deg': 0., 'scale': 1.}
        class Ground:
            def sample(self, xy):
                return .04 * xy[:, 0]
        candidate, decision = tilt_to_ground(group, asset, Ground())
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate['position_m'][:2], group['position_m'][:2])
        self.assertEqual(candidate['yaw_deg'], group['yaw_deg'])
        self.assertGreater(decision['maximum_gap_improvement_m'], .15)
        self.assertLess(abs(decision['after']['minimum_gap_m']), 1e-12)
        self.assertLess(decision['after']['maximum_gap_m'], 1e-12)
        self.assertEqual(group['position_m'][2], .08)

    def test_vehicle_candidate_refuses_a_steep_plane(self):
        asset = {'bounds_min_m': [-2., -1., 0.], 'bounds_max_m': [2., 1., 2.]}
        group = {'category': 'parked_car', 'position_m': [0., 0., .2], 'yaw_deg': 0., 'scale': 1.}
        class Ground:
            def sample(self, xy):
                return .1 * xy[:, 0]
        candidate, decision = tilt_to_ground(group, asset, Ground())
        self.assertIsNone(candidate)
        self.assertEqual(decision['reason'], 'terrain_plane_exceeds_small_tilt_limit')

    def test_actual_standard_gltf_span_swaps_yz_once_and_adds_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = trimesh.Trimesh(vertices=[[0., 0., 0.], [2., 0., 0.], [0., 0., 3.]], faces=[[0, 1, 2]], process=False)
            (root / 'span.glb').write_bytes(mesh.export(file_type='glb'))
            row = {'template_file': 'span.glb', 'template_sha256': sha(root / 'span.glb'),
                   'position_m': [10., 20., 5.], 'rotation_deg': [0., 0., 0.], 'scale': [1., 1., 1.]}
            _, vertices, faces = read_span_world(row, root)
            np.testing.assert_allclose([vertices.min(0), vertices.max(0)], [[10., 20., 5.], [12., 23., 5.]])
            self.assertEqual(len(faces), 1)
            row['scale'] = [1., 1., 2.]
            with self.assertRaisesRegex(ValueError, 'anchor translation'):
                read_span_world(row, root)


if __name__ == '__main__':
    unittest.main()
