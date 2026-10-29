"""Actual textured default-scene geometry and shared scene coordinates in reviews."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from citycode_assets.batch_review import load_render_object, render, review_camera
from test_asset_delivery_contract import cube_glb


class ActualReviewEvidence(unittest.TestCase):
    def test_released_camera_recipe_is_reconstructed_and_checked(self):
        from shapely.geometry import box, mapping
        from citycode_assets.appearance import _shape_yaw
        building={'footprint_world':mapping(box(100,200,120,210))}
        yaw=_shape_yaw(box(100,200,120,210))
        prepared={'provenance':{'shape_yaw_world_deg':yaw,'renderer':{'name':'lod1_v4_sculpted_rough','elevation_deg':34}}}
        camera=review_camera(prepared,building)
        np.testing.assert_allclose(camera.T@camera,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(camera[2,2],np.sin(np.deg2rad(34)))
        prepared['provenance']['shape_yaw_world_deg']+=20
        with self.assertRaisesRegex(ValueError,'differs'):review_camera(prepared,building)

    def test_missing_or_contradictory_camera_does_not_get_an_arbitrary_default(self):
        with self.assertRaisesRegex(ValueError,'Missing recorded'):
            review_camera({'provenance':{'renderer':{'name':'unknown'}}},{})
        with self.assertRaisesRegex(ValueError,'orthonormal'):
            review_camera({'provenance':{'renderer':{'camera_columns_world':[[1,0,0]]*3}}},{})

    def test_default_scene_instances_and_world_placement_share_one_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'two_nodes.glb'
            cube_glb(path,nodes=[{'mesh':0},{'mesh':0,'translation':[3,0,0]}],roots=[0,1])
            matrix=np.eye(4);matrix[:3,3]=[100,200,20]
            obj=load_render_object(path,matrix)
            self.assertEqual(len(obj['faces']),24)
            self.assertEqual(len(obj['uv']),len(obj['world']))
            np.testing.assert_allclose(obj['world'].min(0),[100,200,20])
            np.testing.assert_allclose(obj['world'].max(0),[104,201,21])
            pixels=np.asarray(render([obj],np.eye(3),side=128))
            occupied=(pixels.min(axis=2)<255).any(axis=0)
            runs=np.diff(np.r_[False,occupied,False].astype(int))
            self.assertEqual(int((runs==1).sum()),2)
            self.assertTrue((pixels==255).all(axis=2).any())
