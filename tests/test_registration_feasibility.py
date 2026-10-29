"""A feasible bounded fit can have lower IoU than the original optimum."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from citycode_assets import registration
from citycode_assets.registration import (
    FitPolicy,
    fit_predicted_asset,
    overlap_metrics,
    raster_triangle_footprint,
    read_glb_scene_geometry,
    sha256_file,
)
from test_asset_delivery_contract import cube_glb


def predicted_building(footprint, height=6.0):
    world_from_local = np.eye(4)
    world_from_local[:2, 3] = footprint.centroid.coords[0]
    return {
        "building_id": "synthetic_feasibility",
        "footprint_world": footprint.__geo_interface__,
        "base_z_m": 0.0,
        "top_z_m": float(height),
        "height_m": float(height),
        "world_from_local_4x4": world_from_local.tolist(),
        "local_from_world_4x4": np.linalg.inv(world_from_local).tolist(),
    }


class RegistrationFeasibility(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name) / "textured_box.glb"
        # Preserve the real scene-node transform and original NORMAL/UV/material
        # attributes: only the registration matrix may change during fitting.
        cube_glb(self.source, nodes=[{
            "mesh": 0, "scale": [10.0, 6.0, 10.0],
            "translation": [-5.0, 0.0, -5.0],
        }])
        self.original_bytes = self.source.read_bytes()
        self.original_sha256 = sha256_file(self.source)
        self.scene = read_glb_scene_geometry(self.source)
        self.prior = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.target = unary_union([
            box(-5.0, -5.0, 5.0, 5.0),
            box(-8.75, -2.0, -5.0, 2.0),
            box(5.0, -2.0, 8.75, 2.0),
        ])
        self.policy = FitPolicy()

    def project(self, matrix):
        footprint, _ = raster_triangle_footprint(
            self.scene["vertices"], self.scene["faces"], matrix, cell_m=0.05,
        )
        return footprint

    def fit(self, target=None, height=6.0, **kwargs):
        record = predicted_building(self.target if target is None else target, height)
        fitted = fit_predicted_asset(
            self.source, record, self.prior,
            expected_source_sha256=self.original_sha256, **kwargs,
        )
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        return fitted, record

    def assert_overlap_passes(self, metrics):
        for name, threshold in (
            ("iou", self.policy.minimum_footprint_iou),
            ("target_coverage", self.policy.minimum_target_coverage),
            ("asset_coverage", self.policy.minimum_asset_coverage),
        ):
            self.assertGreaterEqual(metrics[name] + 1e-9, threshold, (name, metrics))

    def test_lower_iou_feasible_fit_is_selected_and_actual_mesh_is_verified(self):
        initial = overlap_metrics(self.project(self.prior), self.target)
        self.assertAlmostEqual(initial["iou"], 100.0 / 130.0)
        self.assertAlmostEqual(initial["target_coverage"], 100.0 / 130.0)
        self.assertEqual(initial["asset_coverage"], 1.0)
        self.assertLess(initial["target_coverage"], self.policy.minimum_target_coverage)

        # This modest directional change loses some IoU but satisfies all three
        # overlap gates. IoU alone must not discard every feasible candidate.
        feasible = self.prior.copy()
        feasible[0, 0] = 1.14
        known = overlap_metrics(self.project(feasible), self.target)
        self.assertLess(known["iou"], initial["iou"])
        self.assert_overlap_passes(known)

        fitted, record = self.fit()
        self.assertEqual(fitted["status"], "accepted", fitted["reasons"])
        selection = fitted["candidate_selection"]
        self.assertTrue(selection["fallback_used"])
        self.assertLess(selection["original_actual_mesh_overlap"]["target_coverage"],
                        self.policy.minimum_target_coverage)
        self.assertEqual(selection["source_sha256"], sha256_file(registration.__file__))
        checked = selection["actual_fallback_candidates"]
        self.assertGreater(len(checked), 0)
        self.assertLessEqual(len(checked), selection["maximum_actual_fallback_candidates"])
        self.assertTrue(checked[-1]["passed"])
        self.assertEqual(selection["selected_parameters"], checked[-1]["parameters"])
        if "fine_projection_check" in checked[-1]:
            fine = checked[-1]["fine_projection_check"]
            self.assertTrue(fine["same_transform_no_retuning"])
            self.assertTrue(fine["passed"])
            self.assert_overlap_passes(fine["metrics"])
        final = fitted["fit"]
        self.assert_overlap_passes(final["actual_mesh_overlap"])
        # Independently project the source's actual scene triangles at the
        # returned matrix, rather than trusting an optimized polygon proxy.
        actual = self.project(np.array(fitted["world_from_asset_4x4"]))
        self.assert_overlap_passes(overlap_metrics(actual, shape(record["footprint_world"])))
        self.assertLessEqual(abs(final["yaw_change_degrees"]), self.policy.maximum_yaw_change_degrees + 1e-8)
        for scale in final["relative_xy_scales"]:
            self.assertGreaterEqual(scale, self.policy.minimum_xy_scale - 1e-8)
            self.assertLessEqual(scale, self.policy.maximum_xy_scale + 1e-8)
        self.assertLessEqual(final["incremental_anisotropy"], self.policy.maximum_incremental_anisotropy + 1e-8)
        self.assertLessEqual(final["total_xy_anisotropy"], self.policy.maximum_total_anisotropy + 1e-8)
        self.assertLessEqual(np.linalg.norm(final["offset_from_target_centroid_m"]), final["offset_bound_m"] + 1e-8)
        self.assertAlmostEqual(final["relative_z_scale"], 1.0)

    def test_xy_scale_bound_cannot_be_expanded_to_cover_an_impossible_target(self):
        # Even maximum legal XY scales cover at most 1.5625% of this target.
        fitted, _ = self.fit(target=box(-50.0, -50.0, 50.0, 50.0))
        self.assertEqual(fitted["status"], "review_required")
        self.assertIn("actual_mesh_target_coverage_below_threshold", fitted["reasons"])

    def test_feasible_footprint_cannot_hide_out_of_bound_height_change(self):
        fitted, _ = self.fit(height=12.0)
        self.assertEqual(fitted["status"], "review_required")
        self.assertIn("height_scale_change_exceeds_bound", fitted["reasons"])
        self.assertAlmostEqual(fitted["fit"]["relative_z_scale"], 2.0)

    def test_no_feasible_composite_candidate_can_bypass_component_anisotropy(self):
        # A shared incremental anisotropy <=1.15 cannot reduce a component's
        # initial ratio 2 below 2/1.15, which still exceeds the total bound 1.55.
        component = self.prior.copy()
        component[0, 0] = 2.0
        self.assertGreater(2.0 / self.policy.maximum_incremental_anisotropy,
                           self.policy.maximum_total_anisotropy)
        fitted, _ = self.fit(component_prior_world_from_asset=[component.tolist()])
        self.assertEqual(fitted["status"], "review_required")
        self.assertTrue(fitted["component_constraints_in_optimizer"])
        # With no feasible candidate the retained rejected diagnostic may fail
        # either coupled bound; neither may be relaxed to produce acceptance.
        self.assertTrue({"component_total_xy_anisotropy_exceeds_bound:0",
                         "incremental_xy_anisotropy_exceeds_bound"}.intersection(fitted["reasons"]))


if __name__ == "__main__":
    unittest.main()
