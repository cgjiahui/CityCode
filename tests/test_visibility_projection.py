"""Contract tests for visibility, without real scene labels or GT geometry."""
import unittest
import numpy as np
from city_pipeline.buildings._visibility.projection import ViewEvidence, project_points


class ProjectionEvidenceContract(unittest.TestCase):
    def setUp(self):
        self.camera = {'world_to_camera_4x4_m': np.eye(4).tolist(),
                       'intrinsics': {'K': [[2, 0, 4], [0, 2, 4], [0, 0, 1]]},
                       'image_size_px': [8, 8]}
        self.ids = np.zeros((8, 8), np.uint16)
        self.background = np.ones((8, 8), np.uint8)
        self.depth = np.full((8, 8), 10.)
        self.valid = np.ones((8, 8), bool)

    def evidence(self):
        return ViewEvidence.from_arrays(self.camera, self.ids, self.background,
                                        self.depth, self.valid, 0)

    def classify(self, point=(0, 0, 10)):
        return self.evidence().classify([point], [7], depth_tolerance_m=1,
                                        image_margin_px=0)

    def test_explicit_background_can_contradict_visible_candidate(self):
        self.assertTrue(self.classify()['explicit_background'][0])

    def test_occlusion_does_not_become_background(self):
        result = self.classify((0, 0, 15))
        self.assertTrue(result['occluded'][0])
        self.assertFalse(result['explicit_background'][0])
        self.assertTrue(result['unknown'][0])

    def test_missing_depth_is_unknown(self):
        self.depth[:] = np.nan
        self.assertFalse(self.classify()['explicit_background'][0])

    def test_outside_strict_roi_is_unknown(self):
        self.valid[:] = False
        self.assertFalse(self.classify()['explicit_background'][0])

    def test_black_semantics_not_background(self):
        self.background[:] = 0
        self.assertFalse(self.classify()['explicit_background'][0])

    def test_roof_overrides_disagreeing_background(self):
        self.ids[:] = 7
        result = self.classify()
        self.assertTrue(result['positive_surface'][0])
        self.assertFalse(result['explicit_background'][0])

    def test_other_instance_is_distinct_from_background(self):
        self.ids[:] = 8
        result = self.classify()
        self.assertTrue(result['other_visible_instance'][0])
        self.assertFalse(result['explicit_background'][0])

    def test_candidate_outside_image_unknown(self):
        self.assertFalse(self.classify((100, 0, 10))['explicit_background'][0])

    def test_candidate_behind_camera_unknown(self):
        self.assertFalse(self.classify((0, 0, -10))['explicit_background'][0])

    def test_pixel_boundary_floor_convention(self):
        uv, depth, front = project_points([[2.5, 2.5, 10]], self.camera)
        np.testing.assert_allclose(uv, [[4.5, 4.5]])
        np.testing.assert_array_equal(np.floor(uv), [[4, 4]])

    def test_outside_world_prism_unknown(self):
        self.camera['data_bounds_xy_world_m'] = [1, 1, 5, 5]
        self.assertFalse(self.classify()['explicit_background'][0])

    def test_zero_depth_is_unknown(self):
        self.depth[:] = 0
        self.assertTrue(self.classify()['unknown'][0])

    def test_background_core_excludes_boundary(self):
        evidence = ViewEvidence.from_arrays(self.camera, self.ids, self.background,
                                             self.depth, self.valid, 1)
        self.assertFalse(evidence.background_core[0, 0])


if __name__ == '__main__':
    unittest.main()
