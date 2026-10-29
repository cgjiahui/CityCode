"""Raster-corner self contacts must be repaired before polygon union."""
import unittest
from unittest.mock import patch
import numpy as np
from citycode_assets.registration import raster_triangle_footprint
class RegistrationProjection(unittest.TestCase):
    def test_invalid_touching_contours_are_repaired_before_union(self):
        contours=[np.array([[0,0],[20,20],[0,20],[20,0]],np.int32).reshape(-1,1,2),np.array([[5,3],[15,3],[15,10],[5,10]],np.int32).reshape(-1,1,2)]
        hierarchy=np.array([[[1,-1,-1,-1],[-1,0,-1,-1]]])
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        with patch('citycode_assets.registration.cv2.findContours',return_value=(contours,hierarchy)):
            result,record=raster_triangle_footprint(vertices,[[0,1,2]],np.eye(4))
        self.assertTrue(result.is_valid);self.assertGreater(result.area,2.)
        self.assertTrue(record['no_hull_or_component_removal'])
if __name__=='__main__':unittest.main()
