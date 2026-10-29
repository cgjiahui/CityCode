"""Numerical contract tests; no scene, model inference, or reference assets."""
import unittest
import tempfile
from pathlib import Path
import numpy as np

from extract_elevated import (Config, background_filter_mask, clean_mask, fit_bounded_height,
                              cluster_height_layers, compatible_conflict_counts,
                              grouped_median, placement_proposal,
                              recover_ribbons, regularized_proposal, skeleton_chains, thin)


class ElevatedGeometryTests(unittest.TestCase):
    def test_median_gives_each_view_one_vote(self):
        keys = np.asarray([2, 1, 2, 1, 2, 1])
        unique, z = grouped_median(keys, np.asarray([4., 8., 2., 0., 3., 2.]))
        np.testing.assert_equal(unique, [1, 2])
        np.testing.assert_allclose(z, [2, 3])
        _, z = grouped_median(np.asarray([1, 1, 2, 2]), np.asarray([0., 2., 3., 5.]))
        np.testing.assert_allclose(z, [1, 4])

    def test_overlapping_predicted_height_layers_are_not_averaged(self):
        layers = cluster_height_layers(np.asarray([1, 1, 1, 1, 2, 2, 2]),
                                       np.asarray([.5, .8, 9., 9.4, 12., 12.1, 30.]),
                                       np.asarray([0, 1, 2, 3, 0, 1, 2]), 2.5)
        primary = layers["layer_primary_indices"]
        np.testing.assert_allclose(layers["layer_height_m"][primary], [9.2, 12.05], atol=1e-5)
        self.assertEqual(len(layers["layer_height_m"]), 4)
        np.testing.assert_equal(layers["supported_layer_count_per_observed_cell"], [2, 1])

    def test_lower_ground_is_not_semantic_counterevidence_to_upper_bridge(self):
        height = np.asarray([[9., 9., 9.]])
        conflicts = compatible_conflict_counts((1, 3), height, np.asarray([0, 1, 2]),
                                               np.asarray([0, 0, 0]), np.asarray([0, 1, 2]),
                                               np.asarray([.7, 9.2, 9.1]), np.asarray([1, 1, 0]), 3, 2.5)
        # Low ground is another level; high alternate semantic in another
        # view is counterevidence; a same-view coarse boundary is one vote.
        np.testing.assert_equal(conflicts, [[0, 1, 0]])

    def test_lipschitz_height_bound_and_vertical_translation(self):
        rng = np.random.default_rng(9)
        s = np.cumsum(rng.uniform(0.5, 5, 100))
        z = np.sin(s/30)*3 + rng.normal(0, 2, 100)
        fitted = fit_bounded_height(z, s, np.ones(100), 0.08)
        self.assertLessEqual(float(np.max(np.abs(np.diff(fitted)/np.diff(s)))), .080000001)
        shifted = fit_bounded_height(z+27, s, np.ones(100), 0.08)
        np.testing.assert_allclose(shifted, fitted+27, atol=1e-9)
        plane = .03*s + 4
        fit = fit_bounded_height(plane, s, np.ones(100), .08)
        self.assertLess(np.max(np.abs(fit-plane)), .15)

    def test_one_height_spike_does_not_shift_entire_supported_road(self):
        s = np.arange(0., 400., 4.)
        z = 9. + .01*s
        observed = z.copy()
        observed[40] += 80.
        fit = fit_bounded_height(observed, s, np.ones(len(s)), .1)
        self.assertLess(float(np.max(np.abs(fit-z))), .05)
        # Its original evidence discrepancy remains available for rejection.
        self.assertGreater(float(abs(fit[40]-observed[40])), 79.)

    def test_morphology_does_not_bridge_large_missing_span(self):
        mask = np.zeros((50, 140), bool)
        mask[20:30, 5:55] = True
        mask[20:30, 75:130] = True
        c = Config(grid_m=1, min_component_area_m2=10)
        cleaned, _ = clean_mask(mask, c)
        self.assertFalse(cleaned[:, 60:70].any())

    def test_thinning_retains_branched_topology_and_closed_cycle(self):
        mask = np.zeros((80, 80), bool)
        mask[35:45, 5:75] = True
        mask[5:75, 35:45] = True
        skeleton = thin(mask)
        self.assertTrue(np.all(~skeleton | mask))
        chains, nodes = skeleton_chains(skeleton)
        self.assertEqual(sum(n["kind"] == "endpoint" for n in nodes), 4)
        self.assertGreaterEqual(sum(n["kind"] == "junction" for n in nodes), 1)
        self.assertGreaterEqual(len(chains), 4)
        ring = np.zeros((40, 40), bool)
        ring[6:34, 6:34] = True
        ring[11:29, 11:29] = False
        chains, _ = skeleton_chains(thin(ring))
        self.assertTrue(any(np.array_equal(line[0], line[-1]) for line in chains))

    def _ribbon_grid(self, low_height=False):
        shape = (60, 180)
        mask = np.zeros(shape, bool)
        mask[23:37, 5:175] = True
        y, x = np.indices(shape)
        height = np.where(mask, (0.3 if low_height else 9.) + .025*x, np.nan).astype(np.float32)
        keys = np.flatnonzero(mask)
        return {"origin_xy_m": np.array([100., -400.]), "cell_size_m": 1., "shape": shape,
                "height_m": height, "height_mad_m": np.where(mask, .1, np.nan),
                "elevated_view_count": mask.astype(np.uint16)*2,
                "known_view_count": mask.astype(np.uint16)*2,
                "ground_m": np.where(~mask, .5, np.nan),
                "evidence_cell_index": np.repeat(keys, 2),
                "evidence_view_index": np.tile([0, 1], len(keys))}

    def test_predicted_straight_ribbon_recovers_height_width_and_transform(self):
        grid = self._ribbon_grid()
        c = Config(grid_m=1., min_segment_length_m=20.)
        ribbons, cleanup, _ = recover_ribbons(grid, c)
        self.assertEqual(len(ribbons), 1)
        r = ribbons[0]
        self.assertAlmostEqual(r["width_median_m"], 14., delta=2.)
        self.assertGreater(r["length_m"], 145)
        self.assertLess(r["max_abs_grade"], .04)
        for p in r["profile"]:
            self.assertEqual(p["support_sample_indices"], [0, 1])
            self.assertAlmostEqual(p["position_m"][2], 9 + .025*(p["position_m"][0]-100-.5), delta=.15)
        proposal = placement_proposal(ribbons, c)
        self.assertGreater(proposal["placement_count"], 8)
        for p in proposal["placements"]:
            np.testing.assert_allclose(p["position_m"], (np.asarray(p["start_m"])+p["end_m"])/2, atol=1e-6)
            self.assertEqual(p["status"], "requires_asset_binding")
            self.assertEqual(p["placement_source"], "predicted_elevated_structure")

    def test_single_view_and_ground_level_evidence_is_withheld(self):
        grid = self._ribbon_grid(low_height=True)
        grid["elevated_view_count"] //= 2
        c = Config(grid_m=1., min_segment_length_m=20.)
        ribbons, _, _ = recover_ribbons(grid, c)
        proposal = placement_proposal(ribbons, c)
        self.assertEqual(proposal["placement_count"], 0)
        self.assertGreater(proposal["withheld_count"], 0)
        self.assertTrue(all("single_view_support" in p["uncertainty_reasons"] for p in proposal["withheld_placements"]))


    def _regularization_ribbon(self, length=180, height=9.):
        grid = self._ribbon_grid()
        c = Config(grid_m=1., min_segment_length_m=20.)
        ribbons, _, _ = recover_ribbons(grid, c)
        profile = ribbons[0]["profile"]
        for p in profile:
            p["observed_z_m"] = height
            p["position_m"][2] = height
            p["nearby_ground_z_m"] = height-.2
            p["clearance_proxy_m"] = .2
            p["supported_height_layers"] = 1
            p["uncertainty_reasons"] = ["insufficient_predicted_elevation"]
        return ribbons, c

    def test_regularization_uses_explicit_plane_prior_and_preserves_height(self):
        ribbons, c = self._regularization_ribbon()
        proposal, lines = regularized_proposal(ribbons, c, .48)
        self.assertEqual(proposal["continuous_segment_count"], 1)
        self.assertGreater(proposal["placement_count"], 8)
        self.assertEqual(proposal["replaces_not_appends"], "placement_proposal.json")
        self.assertFalse(proposal["uses_ground_truth"])
        for p in proposal["placements"]:
            self.assertAlmostEqual(p["position_m"][2], 9., places=5)
            self.assertEqual(p["inference"], "explicit_rule_regularization")
        self.assertEqual(len(lines["features"]), 1)

    def test_regularization_does_not_lift_ground_level_or_single_view_masks(self):
        ribbons, c = self._regularization_ribbon(height=.6)
        proposal, _ = regularized_proposal(ribbons, c, .48)
        self.assertEqual(proposal["placement_count"], 0)
        ribbons, c = self._regularization_ribbon()
        for p in ribbons[0]["profile"]:
            p["support_views"] = 1
        proposal, _ = regularized_proposal(ribbons, c, .48)
        self.assertEqual(proposal["placement_count"], 0)

    def test_regularization_retains_multiple_layers_as_a_hard_topology_break(self):
        ribbons, c = self._regularization_ribbon()
        profile = ribbons[0]["profile"]
        center = profile[len(profile)//2]["s_m"]
        for p in profile:
            if abs(p["s_m"]-center) < 8:
                p["supported_height_layers"] = 2
        proposal, lines = regularized_proposal(ribbons, c, .48)
        self.assertEqual(len(lines["features"]), 2)
        for line in lines["features"]:
            x = [p[0] for p in line["geometry"]["coordinates"]]
            self.assertTrue(max(x) < profile[len(profile)//2]["position_m"][0] or min(x) > profile[len(profile)//2]["position_m"][0])
        self.assertTrue(all(p["inference"] == "explicit_rule_regularization" for p in proposal["placements"]))

    def test_regularization_does_not_fill_a_long_missing_multiview_span(self):
        ribbons, c = self._regularization_ribbon()
        profile = ribbons[0]["profile"]
        center = profile[len(profile)//2]["s_m"]
        for p in profile:
            if abs(p["s_m"]-center) < 24:
                p["support_views"] = 1
        proposal, lines = regularized_proposal(ribbons, c, .48)
        self.assertEqual(len(lines["features"]), 2)
        self.assertGreater(proposal["placement_count"], 0)

    def test_persistent_height_step_is_split_instead_of_averaged_between_decks(self):
        ribbons, c = self._regularization_ribbon()
        profile = ribbons[0]["profile"]
        for p in profile[len(profile)//2:]:
            p["observed_z_m"] = 20.
            p["position_m"][2] = 20.
        proposal, lines = regularized_proposal(ribbons, c, .48)
        self.assertEqual(len(lines["features"]), 2)
        all_z = [coordinate[2] for feature in lines["features"] for coordinate in feature["geometry"]["coordinates"]]
        self.assertTrue(all(min(abs(z-9), abs(z-20)) < 1e-5 for z in all_z))
        self.assertGreater(proposal["placement_count"], 0)


    def test_background_suppression_mask_keeps_world_grid_and_has_no_empty_source_fill(self):
        source = np.zeros((7, 8), bool); source[3, 4] = True
        grid = {"retained_mask": source, "origin_xy_m": np.asarray([-12., 35.]), "cell_size_m": .75}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            desc = background_filter_mask(grid, path)
            with np.load(path/desc["mask_file"]) as result:
                np.testing.assert_equal(result["origin_xy_m"], [-12., 35.])
                self.assertTrue(result["remove_high_surface_mask"][3, 4])
                self.assertTrue(result["remove_high_surface_mask"][3, 3])
                self.assertFalse(result["remove_high_surface_mask"][2, 3])  # diagonal 1.061m exceeds 1m
            self.assertEqual(desc["coordinate_system"], {"world": "UE", "units": "m", "up": "Z"})
            grid["retained_mask"][:] = False
            desc = background_filter_mask(grid, path)
            with np.load(path/desc["mask_file"]) as result:
                self.assertFalse(result["remove_high_surface_mask"].any())


if __name__ == "__main__":
    unittest.main()
