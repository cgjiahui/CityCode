"""Road-network regression tests, using synthetic predictions and scene ground."""
import unittest
import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import numpy as np
from shapely.geometry import box

from repair_elevated_network import (ActualBuildingFootprints, LayerEvidence, alternate_path_length, build_connector, connection_contract,
                                     constrained_height, design_completions, filter_isolated_fragments, generate_approaches, inplace_landings, solve_joints)


def segment(identity, xy, z, width=8.):
    xy = np.asarray(xy, float)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    z = np.broadcast_to(z, len(xy))
    profile = [{'position_m': [*point, float(height)], 's_m': float(along), 'source_profile_index': i,
                'width_m': width, 'uncertainty_reasons': [], 'support_views': 3,
                'height_adjustment_m': 0., 'height_mad_m': .1, 'clearance_proxy_m': float(height)}
               for i, (point, height, along) in enumerate(zip(xy, z, s))]
    return {'segment_id': identity, 'source_segment_id': identity, 'profile': profile, 'length_m': s[-1]}


class FlatGround:
    def sample(self, xy):
        return np.zeros(len(xy))


class Evidence:
    def __init__(self, offset=0., support=True):
        self.offset, self.support = offset, support

    def selection(self, xy, expected):
        if not self.support:
            return None
        return {'height_m': expected+self.offset, 'views': 3, 'layer_index': 0, 'distance_m': 0.}


class NetworkTests(unittest.TestCase):
    def test_shared_node_height_is_exact_and_grade_remains_bounded(self):
        a = segment('a', [[-40, 0], [-30, 0], [-20, 0], [-10, 0], [0, 0]], 9.)
        b = segment('b', [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], 10.)
        self.assertEqual(solve_joints([a, b]), [])
        self.assertAlmostEqual(a['profile'][-1]['position_m'][2], b['profile'][0]['position_m'][2], places=8)
        self.assertAlmostEqual(a['profile'][-1]['position_m'][2], 9.5)
        for item in (a, b):
            self.assertLessEqual(np.max(np.abs(np.diff([n['position_m'][2] for n in item['profile']]))/10.), .10000001)
        self.assertEqual(len(connection_contract([a, b])), 1)

    def test_crossing_height_layers_are_not_joined(self):
        a = segment('a', [[-20, 0], [0, 0]], 9.)
        b = segment('b', [[0, 0], [0, 20]], 16.)
        rejected = solve_joints([a, b])
        self.assertEqual(rejected[0]['reason'], 'distinct_height_layers')
        self.assertEqual(connection_contract([a, b]), [])
        self.assertEqual(a['profile'][-1]['position_m'][2], 9.)

    def test_impossible_grade_is_refused(self):
        with self.assertRaises(ValueError):
            constrained_height([0, 1, 5], [0, 2, 4], {0: 0, 2: 5}, .1)

    def test_layers_choose_continuous_lower_deck_not_primary_upper_deck(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'evidence.npz'
            np.savez(path, origin_xy_m=[0., 0.], height_m=[[16.]],
                     layer_cell_index=[0, 0], layer_height_m=[9., 16.], layer_view_count=[3, 4])
            evidence = LayerEvidence(path, 1.)
            choice = evidence.selection([.5, .5], 9.2)
            self.assertEqual(choice['height_m'], 9.)
            self.assertEqual(choice['views'], 3)
            self.assertIsNone(evidence.selection([.5, .5], 30.))

    def test_connector_keeps_predicted_xy_and_preserves_layer_evidence(self):
        raw = segment('road', [[0, 0], [4, 1], [8, 2], [12, 2]], 9.)
        repaired = build_connector(raw, 0, 3, 9., 9.4, Evidence(), 'connection')
        np.testing.assert_allclose([p['position_m'][:2] for p in repaired['profile']],
                                   [p['position_m'][:2] for p in raw['profile']])
        self.assertTrue(all(p['selected_observed_layer']['views'] == 3 for p in repaired['profile']))
        with self.assertRaisesRegex(ValueError, 'insufficient_continuous_layer_evidence'):
            build_connector(raw, 0, 3, 9., 9.4, Evidence(support=False), 'connection')

    def test_short_junction_shortcut_requires_full_predicted_width_support(self):
        class WideEvidence(Evidence):
            def supported_width(self, xy):
                return np.full(len(xy), 24.)
        raw = segment('road', [[0, 0], [0, 10], [10, 10], [10, 0]], 9.)
        repaired = build_connector(raw, 0, 3, 9., 9., WideEvidence(), 'connection')
        self.assertTrue(repaired['predicted_mask_supported_shortcut'])
        self.assertAlmostEqual(repaired['length_m'], 10.)
        self.assertTrue(all(abs(p['position_m'][1]) < 1e-8 for p in repaired['profile']))

    def test_alternative_path_does_not_use_another_height_layer(self):
        a = segment('a', [[-20, 0], [0, 0]], 9.)
        b = segment('b', [[0, 0], [20, 0]], 16.)
        self.assertTrue(np.isinf(alternate_path_length([a, b], [-20, 0, 9], [20, 0, 16])))
        b = segment('b', [[0, 0], [20, 0]], 9.)
        self.assertAlmostEqual(alternate_path_length([a, b], [-20, 0, 9], [20, 0, 9]), 40.)

    def test_ground_approach_has_buried_horizontal_terminal_and_obeys_grade(self):
        raw = segment('road', [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0], [60, 0]],
                      [6., 5., 4., 3., 2., 1., 0.])
        accepted = segment('road', [[0, 0], [10, 0]], [6., 5.])
        structure = {'segments': [raw], 'junctions_and_endpoints': [{'kind': 'endpoint', 'position_xy_m': [60, 0]}]}
        additions, decisions = generate_approaches(structure, [accepted], FlatGround())
        self.assertEqual(len(additions), 1)
        profile = additions[0]['profile']
        s = np.array([p['s_m'] for p in profile])
        z = np.array([p['position_m'][2] for p in profile])
        self.assertAlmostEqual(z[0], 5.)
        self.assertEqual(z[-1], z[-2])
        self.assertLess(z[-1], 0.)
        self.assertLessEqual(np.max(np.abs(np.diff(z)/np.diff(s))), .10000001)
        self.assertTrue(additions[0]['native_deck_end_buried'])
        additions, decisions = generate_approaches(structure, [accepted], FlatGround(), [('blocking', box(15, -20, 100, 20))])
        self.assertEqual(additions, [])
        self.assertEqual(decisions[0]['reason'], 'approach_would_intersect_existing_building')

    def test_junction_is_never_misinterpreted_as_ground_landing(self):
        a = segment('road', [[0, 0], [20, 0]], 9.)
        structure = {'segments': [a], 'junctions_and_endpoints': [{'kind': 'junction', 'position_xy_m': [20, 0]}]}
        additions, decisions = generate_approaches(structure, [a], FlatGround())
        self.assertEqual(additions, [])
        self.assertEqual(decisions, [])

    def test_prepared_actual_triangles_remove_bounds_false_positive(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'ue_citycode'))
        from interchange import Writer
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            writer = Writer(directory=directory)
            points = np.asarray([[0, 0, 8], [4, 0, 8], [4, 4, 8],
                                 [0, 0, 8], [4, 4, 8], [0, 4, 8]], dtype='<f4')
            writer.doc['meshes'] = [{'primitives': [{'attributes': {'POSITION': writer.array(points)}, 'mode': 4}]}]
            writer.doc['nodes'] = [{'mesh': 0}]
            writer.doc['scenes'] = [{'nodes': [0]}]
            writer.save(directory/'building.glb')
            manifest = {'meshes': [{'id': 'PGB_test', 'import_file': 'building.glb', 'anchor_m': [10., 20., 0.],
                                    'prepared': {'world_bounds_m': [[10, 20, 0], [30, 40, 8]]}}]}
            (directory/'assembly.json').write_text(json.dumps(manifest))
            buildings = ActualBuildingFootprints(directory/'assembly.json')
            self.assertFalse(buildings.intersects(box(20, 30, 25, 35)))
            self.assertTrue(buildings.intersects(box(10, 20, 14, 24)))
            self.assertIn('PGB_test', buildings.geometry_inputs)

    def test_inplace_landing_preserves_xy_and_far_connected_height(self):
        a = segment('road', [[x, 0] for x in range(0, 101, 5)], np.linspace(5., 8., 21))
        xy = [n['position_m'][:2].copy() for n in a['profile']]
        decisions = [{'segment_id': 'road', 'end': 'start', 'status': 'withheld'}]
        inplace_landings([a], decisions, FlatGround(), [])
        self.assertEqual(decisions[0]['status'], 'inplace_landing_candidate')
        self.assertEqual(a['inplace_landing_end'], 'start')
        np.testing.assert_allclose(xy, [n['position_m'][:2] for n in a['profile']])
        self.assertEqual(a['profile'][-1]['position_m'][2], 8.)
        self.assertAlmostEqual(a['profile'][0]['position_m'][2], -.025)
        z = np.array([n['position_m'][2] for n in a['profile']])
        self.assertLessEqual(np.max(abs(np.diff(z))/5.), .10000001)
        b = segment('short', [[0, 0], [10, 0], [20, 0]], [5, 6, 8])
        rejected = [{'segment_id': 'short', 'end': 'start', 'status': 'withheld'}]
        inplace_landings([b], rejected, FlatGround(), [])
        self.assertEqual(rejected[0]['inplace_landing_rejected'], 'insufficient_existing_corridor_length_for_grade')
        self.assertEqual(b['profile'][0]['position_m'][2], 5.)

    def test_isolated_filter_preserves_grounded_or_boundary_components(self):
        airborne = segment('airborne', [[0, 0], [20, 0]], 10.)
        grounded = segment('grounded', [[100, 0], [120, 0]], 1.)
        grounded['ground_transition'] = True
        boundary = segment('boundary', [[-990, 0], [-970, 0]], 10.)
        active, decisions = filter_isolated_fragments([airborne, grounded, boundary],
                            {'origin_xy_m': [-1000, -1000], 'shape_yx': [2000, 2000], 'cell_size_m': 1.})
        self.assertEqual({s['segment_id'] for s in active}, {'grounded', 'boundary'})
        self.assertEqual(sum(not d['assembly_enabled'] for d in decisions), 1)

    def test_design_completion_is_labelled_inference_and_needs_semantic_support(self):
        class RibbonEvidence:
            def __init__(self, width): self.width = width
            def supported_width(self, xy): return np.full(len(xy), self.width)
        raw = segment('road', [[x, 0] for x in range(0, 241, 10)], 18.)
        existing = deepcopy(raw)
        existing['profile'] = existing['profile'][3:-3]
        existing['length_m'] = 180.
        for node in existing['profile']:
            node['position_m'][2] = 9.
            node['s_m'] -= 30.
        left = segment('left', [[-60, 0], [0, 0]], 9.)
        right = segment('right', [[240, 0], [300, 0]], 9.)
        structure = {'segments': [raw, left, right],
                     'grid': {'origin_xy_m': [-1000, -1000], 'shape_yx': [2000, 2000], 'cell_size_m': 1.}}
        additions, decisions = design_completions(structure, [existing, left, right], RibbonEvidence(30.), [])
        self.assertEqual(len(additions), 2)
        self.assertTrue(all(s['inference'] == 'assembly_design_completion' for s in additions))
        self.assertTrue(all(s['observed_height_reconstruction'] is False for s in additions))
        self.assertTrue(all(n['height_anchor'] is False for s in additions for n in s['profile']))
        additions, decisions = design_completions(structure, [existing, left, right], RibbonEvidence(2.), [])
        self.assertEqual(additions, [])
        self.assertTrue(all(d['status'] == 'withheld' for d in decisions))


if __name__ == '__main__':
    unittest.main()
