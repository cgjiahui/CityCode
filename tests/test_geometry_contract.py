"""CPU contracts at the predicted-artifact to geometry boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from city_pipeline.geometry.depth_lifting import (calibration_planes,
    class_aware_scale, load_depth_samples, validate_depth_stage)
from city_pipeline.geometry.orchestrator import verify_masks
from city_pipeline.geometry.priors import read_plane_priors
from city_pipeline.buildings.evidence import crop_camera


class GeometryContractTests(unittest.TestCase):
    def test_declared_priors_change_metric_scale(self):
        camera = {'camera_to_world_4x4_m': [[1,0,0,0],[0,-1,0,0],[0,0,-1,100],[0,0,0,1]],
            'intrinsics': {'K': [[100,0,64],[0,100,64],[0,0,1]], 'fx_px':100,
                'fy_px':100,'cx_px':64,'cy_px':64}, 'image_size_px':[128,128],
            'ground_reference_z_m':.48}
        raw = np.full((128,128),200,np.float32)
        cores = {1: np.ones(raw.shape,bool), 2:np.zeros(raw.shape,bool), 3:np.zeros(raw.shape,bool)}
        a, _ = class_aware_scale(raw, camera, cores, 200000,
            calibration_planes(camera,'fixed-smallcity',{1:.73,2:-3.5,3:.73}))
        b, _ = class_aware_scale(raw, camera, cores, 200000,
            calibration_planes(camera,'fixed-smallcity',{1:2.73,2:-3.5,3:2.73}))
        self.assertAlmostEqual(a, .49635, places=6)
        self.assertAlmostEqual(b, .48635, places=6)

    def test_priors_reject_nonfinite_and_wrong_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'priors.json'
            for values in [{'1':float('nan'),'2':-3.5,'3':.73}, {'1':.73,'2':-3.5,'4':.73}]:
                p.write_text(json.dumps({'unit':'metre','predicted_background_semantic_id_to_prior_z_m':values}))
                with self.assertRaises(ValueError): read_plane_priors(p)

    def test_raw_only_manifest_and_path_base(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); np.save(root/'raw.npy',np.ones((2,2),np.float32))
            (root/'manifest.json').write_text(json.dumps({'schema':'citycode.perception.depth.v1',
                'path_resolution':'manifest_directory','depth_stage':'raw_prediction',
                'samples':[{'patch_id':'P','view_id':'tilt_light','outputs':{'depth_raw_zc_m':'raw.npy'}}]}))
            samples, _ = load_depth_samples(root)
            self.assertEqual(samples[0]['outputs']['depth_raw_zc_m'],str(root/'raw.npy'))
            self.assertNotIn('depth_ground_scaled_zc_m',samples[0]['outputs'])
            validate_depth_stage(samples)

    def test_already_calibrated_stage_needs_explicit_smoke_policy(self):
        samples = [{'depth_stage':'already_calibrated_prediction'}]
        with self.assertRaises(ValueError): validate_depth_stage(samples)
        validate_depth_stage(samples, allow_cached_smoke_recalibration=True)

    def test_mixed_depth_superset_checks_only_selected_samples(self):
        samples = [{'patch_id':'P','view_id':'tilt_light','depth_stage':'raw_prediction'},
                   {'patch_id':'Q','view_id':'tilt_light','depth_stage':'already_calibrated_prediction'}]
        validate_depth_stage(samples, sample_ids=['P__tilt_light'])
        with self.assertRaises(ValueError):
            validate_depth_stage(samples, sample_ids=['Q__tilt_light'])
        with self.assertRaises(ValueError):
            validate_depth_stage(samples, sample_ids=['MISSING__tilt_light'])

    def test_mask_superset_accepts_selected_ids_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['structure_v9_2','structure_v7b']:
                dest=root/name;dest.mkdir();(dest/'one.png').write_bytes(b'fixture')
                (dest/'manifest.json').write_text(json.dumps({'samples':[
                    {'id':'P__tilt_light','result':'one.png'},
                    {'id':'Q__tilt_light','result':'unselected_missing.png'}]}))
            verify_masks(root,[{'id':'P__tilt_light'}])
            with self.assertRaises(RuntimeError): verify_masks(root,[{'id':'Q__tilt_light'}])

    def test_crop_camera_preserves_native_projection(self):
        c={'view_id':'tilt_light','intrinsics':{'K':[[100,0,50],[0,100,60],[0,0,1]]},
           'camera_to_world_4x4_m':np.eye(4).tolist(),'world_to_camera_4x4_m':np.eye(4).tolist()}
        cropped = crop_camera(c,[10,20,90,100])
        p=np.array([.3,.5,1.]);original=np.asarray(c['intrinsics']['K'])@p
        observed=np.asarray(cropped['intrinsics']['K'])@p
        np.testing.assert_array_equal(observed,original-[10,20,0])

    def test_dataset_read_guard_allows_priors_and_rejects_gt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'priors.json').write_text('{}');(root/'semantic_ids.png').write_bytes(b'GT')
            env=os.environ.copy();env['CITY_PIPELINE_NO_GT_DATASET']=str(root)
            script="from city_pipeline.runtime import read_json;from pathlib import Path;import os; p=Path(os.environ['CITY_PIPELINE_NO_GT_DATASET']);read_json(p/'priors.json');\ntry:(p/'semantic_ids.png').read_bytes()\nexcept PermissionError:pass\nelse:raise AssertionError('GT read was allowed')"
            subprocess.run([sys.executable,'-c',script],env=env,check=True,capture_output=True)


if __name__ == '__main__': unittest.main()
