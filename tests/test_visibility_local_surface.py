"""Synthetic local-query contracts. No project observations or GT are read."""
import copy
import unittest

import numpy as np
from shapely.geometry import box, mapping, shape

from city_pipeline.buildings._visibility.local_surface import local_bands, build_queries, reduce_view, restrict_masks
from city_pipeline.buildings._visibility.associations import classify_hits
from city_pipeline.buildings._visibility.projection import ViewEvidence
from city_pipeline.buildings._visibility.weak_silhouette import refine
from city_pipeline.buildings.visibility_refinement import WEAK_CONFIG


def band(oid='canonical', canonical=True, reliable=True, z=20.):
    return dict(observation_id=oid, canonical=canonical, reliable=reliable,
                point_count=20 if reliable else 19, q10_m=z-1,
                q50_m=z, q90_m=z+1, patch_id='source')


def points(n=1):
    return np.tile(np.array([[[0., 0., 100.]]]), (n, 5, 1))


def view(patch='other', n=1, background=True):
    return {'patch_id': patch, 'source_surface': np.zeros(n, bool),
            'peer_surface': np.zeros(n, bool), 'neighbor_or_unknown': np.zeros(n, bool),
            'all_queries_explicit_background': np.full(n, background, bool)}


def states(n, background=True):
    return {'explicit_background': np.full(n, background, bool),
            'valid_depth_and_roi': np.ones(n, bool), 'occluded': np.zeros(n, bool)}


def hits(n):
    return {k: np.zeros(n, bool) for k in ('source_surface', 'peer_surface', 'neighbor_or_unknown')}


class LocalSurfaceContracts(unittest.TestCase):
    def test_local_band_uses_finite_provenance_one_same_instance_and_radius(self):
        good = np.tile([0., 0., 20.], (20, 1))
        xyz = np.concatenate([good, [[0, 0, 1000], [0, 0, -1000],
                                   [0, 0, np.nan], [2.01, 0, 500]]])
        ids = np.array([7]*20 + [8, 7, 7, 7])
        provenance = np.array([1]*20 + [1, 2, 1, 1])
        records, count = local_bands(xyz, ids, provenance, 7, [[0., 0.]])
        self.assertEqual(count, 21)  # The finite direct point outside radius still exists.
        self.assertEqual(records[0]['point_count'], 20)
        self.assertTrue(records[0]['reliable']); self.assertEqual(records[0]['q50_m'], 20.)

    def test_dense_mixed_layers_are_unknown_and_sparse_coherent_is_unknown(self):
        xyz = np.c_[np.zeros((40, 2)), np.r_[np.full(20, 10.), np.full(20, 30.)]]
        records, _ = local_bands(xyz, np.ones(40), np.ones(40), 1, [[0., 0.]])
        self.assertFalse(records[0]['reliable'])
        records, _ = local_bands(xyz[:19], np.ones(19), np.ones(19), 1, [[0., 0.]])
        self.assertFalse(records[0]['reliable'])

    def test_any_one_canonical_unknown_protects_the_whole_cell_even_with_peer(self):
        for unknown_point in range(5):
            rows = [[band(), band('peer', False)] for _ in range(5)]
            rows[unknown_point][0] = band(reliable=False)
            queries = build_queries(points(), rows)
            self.assertFalse(queries['canonical_all_five_reliable'][0])
            self.assertEqual(len(queries['xyz']), 0)

    def test_all_reliable_bands_and_three_endpoints_are_queried_without_changing_xyz(self):
        xyz = points(); before = xyz.copy()
        rows = [[band(), band('peer', False, z=40.), band('weak_peer', False, False)] for _ in range(5)]
        queries = build_queries(xyz, rows)
        self.assertEqual(len(queries['xyz']), 30)
        self.assertEqual(sorted(set(queries['xyz'][:, 2])), [16., 20., 24., 36., 40., 44.])
        self.assertEqual(set(queries['point_index']), set(range(5)))
        np.testing.assert_array_equal(xyz, before)

    def test_duplicate_or_missing_canonical_band_fails_instead_of_guessing(self):
        for bad in ([band('peer', False)], [band(), band()]):
            with self.assertRaises(ValueError):
                build_queries(points(), [bad, *[[band()] for _ in range(4)]])

    def test_single_surface_or_neighbor_query_vetoes_at_any_point_band_or_height(self):
        query = build_queries(points(), [[band(), band('peer', False, z=40.)] for _ in range(5)])
        for kind in ('source_surface', 'peer_surface', 'neighbor_or_unknown'):
            for position in range(30):
                h = hits(30); h[kind][position] = True
                v = reduce_view(states(30), h, query['cell_index'], 1)
                mask = restrict_masks([True], [True], ['source'], [{'patch_id': 'other', **v}])
                self.assertFalse(mask['local_surface_veto'][0], (kind, position))

    def test_background_requires_all_five_points_all_bands_all_heights(self):
        query = build_queries(points(), [[band(), band('peer', False, z=40.)] for _ in range(5)])
        for position in range(30):
            state = states(30); state['explicit_background'][position] = False
            v = reduce_view(state, hits(30), query['cell_index'], 1)
            mask = restrict_masks([True], [True], ['source'],
                                   [{'patch_id': 'one', **v}, view('two')])
            self.assertTrue(mask['local_surface_veto'][0])
            self.assertFalse(mask['local_surface_background'][0], position)

    def test_source_patch_never_counts_and_physical_patch_duplicates_rejected(self):
        mask = restrict_masks([True], [True], ['source'], [view('source'), view('other')])
        self.assertEqual(mask['other_explicit_background_patch_count'][0], 1)
        self.assertFalse(mask['local_surface_background'][0])
        mask = restrict_masks([True], [True], ['source'], [view('one'), view('two')])
        self.assertTrue(mask['local_surface_background'][0])
        with self.assertRaises(ValueError):
            restrict_masks([True], [True], ['source'], [view('other'), view('other')])

    def test_invalid_depth_roi_occlusion_or_no_query_cannot_make_negative_vote(self):
        for key, value in [('valid_depth_and_roi', False), ('occluded', True)]:
            state = states(15); state[key][7] = value
            reduced = reduce_view(state, hits(15), np.zeros(15, dtype=int), 1)
            self.assertFalse(reduced['all_queries_explicit_background'][0])
        reduced = reduce_view(states(0), hits(0), np.array([], dtype=int), 1)
        self.assertFalse(reduced['all_queries_explicit_background'][0])

    def test_original_top_veto_survives_even_if_every_local_query_says_background(self):
        mask = restrict_masks([False, True], [True, True], ['source']*2,
                               [view('one', 2), view('two', 2)])
        np.testing.assert_array_equal(mask['local_surface_veto'], [False, True])
        np.testing.assert_array_equal(mask['local_surface_background'], [False, True])

    def test_broader_reliable_peer_band_can_protect_when_canonical_queries_do_not(self):
        camera = dict(world_to_camera_4x4_m=np.eye(4).tolist(),
                      intrinsics=dict(K=[[10, 0, 10], [0, 10, 10], [0, 0, 1]]),
                      image_size_px=[21, 21], data_bounds_xy_world_m=[-10, -10, 10, 10])
        view_evidence = ViewEvidence.from_arrays(camera, np.ones((21, 21), dtype=int),
            np.zeros((21, 21)), np.full((21, 21), 40.), np.ones((21, 21), bool), 0, 'v92_explicit')
        query = build_queries(points(), [[band(), band('peer', False, z=40.)] for _ in range(5)])
        state = view_evidence.classify(query['xyz'], [1], 3., 0)
        hit = classify_hits(state, {1: 0}, np.zeros(30, int), np.zeros(30, bool), np.ones(30, int))
        reduced = reduce_view(state, hit, query['cell_index'], 1)
        self.assertTrue(reduced['peer_surface'][0])
        mask = restrict_masks([True], [True], ['source'], [{'patch_id': 'peer', **reduced}])
        self.assertFalse(mask['local_surface_veto'][0])

    def test_visible_unmapped_foreground_is_protected_but_missing_data_is_not_background(self):
        camera = dict(world_to_camera_4x4_m=np.eye(4).tolist(),
                      intrinsics=dict(K=[[10, 0, 10], [0, 10, 10], [0, 0, 1]]),
                      image_size_px=[21, 21], data_bounds_xy_world_m=[-10, -10, 10, 10])
        for instance, depth, valid, neighbor in [(8, 100., True, True), (0, np.nan, True, False),
                                                (0, 1., True, False), (0, 100., False, False)]:
            ve = ViewEvidence.from_arrays(camera, np.full((21, 21), instance), np.zeros((21, 21)),
                np.full((21, 21), depth), np.full((21, 21), valid), 0, 'v92_explicit')
            state = ve.classify([[0., 0., 20.]], [instance], 3., 0)
            h = classify_hits(state, {1: 0}, np.array([0]), np.array([False]), np.array([1]))
            reduced = reduce_view(state, h, np.array([0]), 1)
            self.assertEqual(bool(reduced['neighbor_or_unknown'][0]), neighbor)
            self.assertFalse(reduced['all_queries_explicit_background'][0])

    def test_restriction_is_local_and_does_not_mutate_input_evidence(self):
        legacy = np.array([True, True, False]); ready = np.array([True, False, True])
        views = [view('one', 3), view('two', 3)]; views[0]['source_surface'][0] = True
        before = copy.deepcopy((legacy, ready, views))
        mask = restrict_masks(legacy, ready, ['source']*3, views)
        self.assertFalse(mask['local_surface_veto'].any())
        np.testing.assert_array_equal(legacy, before[0]); np.testing.assert_array_equal(ready, before[1])
        for original, current in zip(before[2], views):
            for key in original:
                np.testing.assert_array_equal(original[key], current[key])

    def test_fewer_cells_can_pass_whole_building_guard_but_baseline_never_regrows(self):
        def feature(geometry):
            return dict(type='Feature', geometry=mapping(geometry), properties=dict(
                predicted_global_building_id='synthetic', canonical_observation_id='synthetic_obs',
                base_z_m=0., top_z_m=10., body_top_z_m=10., height_m=10., roof_layers=[]))
        parent = [feature(box(0, 0, 100, 100))]
        baseline = [feature(box(.5, 0, 100, 100))]
        cells = [{'building_id': 'synthetic', 'proposed_removal': box(99, 0, 100, 100)},
                 {'building_id': 'synthetic', 'proposed_removal': box(.5, 0, 2.5, 100)}]
        all_result, all_report = refine(parent, baseline, cells, [True, True], WEAK_CONFIG)
        subset_result, subset_report = refine(parent, baseline, cells, [True, False], WEAK_CONFIG)
        self.assertEqual(all_result, baseline); self.assertEqual(all_report['rejected'][0]['reason'], 'total_area_limit')
        self.assertEqual(len(subset_report['changes']), 1)
        self.assertEqual(shape(subset_result[0]['geometry']).difference(shape(baseline[0]['geometry'])).area, 0.)
        for key in ('canonical_observation_id', 'base_z_m', 'top_z_m', 'body_top_z_m', 'height_m', 'roof_layers'):
            self.assertEqual(subset_result[0]['properties'][key], parent[0]['properties'][key])



if __name__ == "__main__":
    unittest.main()
