"""Native pillar fit and safe-footing behavior, independent of scene files."""
import unittest
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree

from bridge_supports import (clearance_reason, deck_frame, fit_support,
                             native_bounds, poly_for_support, station_points)


def pillar():
    # Deliberately offset native pivot with a 15cm high local bottom.
    return {"bounds_cm": [[-100, -120, 15], [125, 105, 415]]}


class NativePillarTests(unittest.TestCase):
    def test_native_bounds_pivot_correction_contacts_ground_and_deck(self):
        a = pillar()
        position, scale = fit_support(a, [10, 20], 90, .75, 9.25)
        bounds = native_bounds(a)
        self.assertAlmostEqual(position[2]+bounds[0, 2]*scale[2], .75)
        self.assertAlmostEqual(position[2]+bounds[1, 2]*scale[2], 9.25)
        poly = poly_for_support(a, [10, 20], 90)
        np.testing.assert_allclose(poly.centroid.coords[0], [10, 20], atol=1e-10)
        self.assertAlmostEqual(poly.area, 2.25**2)
        self.assertTrue(np.all(scale > 0))

    def test_negative_zero_too_short_and_excessive_columns_are_withheld(self):
        for top in (-1., 0., .5, 25.):
            with self.assertRaises(ValueError):
                fit_support(pillar(), [0, 0], 0, 0., top)

    def test_actual_pitched_deck_underside_uses_local_thickness_not_world_subtraction(self):
        module = {"start_m": [0, 0, 10], "end_m": [20, 0, 12]}
        bottom, side, up, yaw = deck_frame(module, [10, 0], 1.61209137)
        np.testing.assert_allclose(np.dot(up, bottom-np.array([10, 0, 11])), -1.61209137, atol=1e-10)
        self.assertGreater(bottom[0], 10.)
        self.assertAlmostEqual(yaw, 0.)
        np.testing.assert_allclose(np.cross([20/np.sqrt(404),0,2/np.sqrt(404)],side),up)

    def test_pillar_cannot_be_placed_in_ground_vehicle_corridor(self):
        road = box(-30, -4, 30, 4)
        trees = STRtree([])
        reason = clearance_reason(box(-1, -1, 1, 1), box(-100,-100,100,100),
                                  STRtree([road.buffer(.5)]), trees, trees, [])
        self.assertEqual(reason, 'ground_vehicle_corridor')
        self.assertIsNone(clearance_reason(box(-1, 6, 1, 8), box(-100,-100,100,100),
                                          STRtree([road.buffer(.5)]), trees, trees, []))

    def test_pillar_respects_buildings_water_and_preexisting_props(self):
        empty = STRtree([]); land = box(-20,-20,20,20)
        self.assertEqual(clearance_reason(box(18,18,20,20),land,empty,empty,empty,[]),'water_or_shore')
        poly = box(4,4,6,6)
        self.assertEqual(clearance_reason(poly,land,empty,STRtree([box(5,5,10,10)]),empty,[]),'building_clearance')
        self.assertEqual(clearance_reason(poly,land,empty,empty,STRtree([box(5,5,10,10)]),[]),'native_prop_clearance')

    def test_station_spacing_is_stable_uses_xy_metres_and_avoids_endpoints(self):
        line = [[0,0,4],[50,0,9],[100,0,14]]
        points = station_points(line)
        self.assertEqual(len(points),3)
        np.testing.assert_allclose([p[1] for p in points],[25,50,75])
        self.assertEqual([p[1] for p in points],[p[1] for p in station_points(line)])


if __name__ == '__main__':
    unittest.main()
