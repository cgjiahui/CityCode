"""The replay input package must survive relocation without research files."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from city_pipeline.inputs import bundle_cache, validate_bundle, E09_OUTPUTS
from city_pipeline.runtime import read_json, write_json, sha256
from city_pipeline.buildings.visibility_refinement import build_inventory, collect_input_paths


class CachedInputsContract(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'research'
        self.source = self.data / 'prediction'
        self.output = self.root / 'portable'
        self.sid = 'P01__tilt_light'
        view = self.data / 'views' / self.sid
        view.mkdir(parents=True)
        identity = np.eye(4).tolist()
        camera = {'schema': 'camera', 'view_id': 'tilt_light', 'image_size_px': [8, 8],
                  'intrinsics': {'K': np.eye(3).tolist()},
                  'camera_to_world_4x4_m': identity, 'world_to_camera_4x4_m': identity,
                  'data_bounds_xy_world_m': [0, 0, 8, 8],
                  'frustum_prism_audit': {'owned_building_prism_point_count': 123},
                  'old_source_path': '/absolute/research/camera'}
        write_json(view / 'camera.json', camera)
        for name in ['rgb.png', 'strict_valid_mask.png', 'source_ids.png', 'bev_ids.png', 'v7b.png']:
            Image.new('RGB' if name == 'rgb.png' else 'L', (8, 8), 1).save(view / name)
        np.save(view / 'calibrated_depth.npy', np.ones((8, 8), dtype=np.float32))
        np.savez(view / 'points.npz', xyz_m=np.array([[1., 2., 3.]]),
                 instance_id=np.array([1]), provenance=np.array([1]), rgb=np.array([[1, 2, 3]]))
        props = {'observation_id': 'O1', 'sample_id': self.sid,
                 'patch_id': 'P01', 'view_id': 'tilt_light', 'height_m': 3.,
                 'rgb_path': str((view / 'rgb.png').relative_to(self.data)),
                 'source_instance_ids_path': str((view / 'source_ids.png').relative_to(self.data)),
                 'bev_instance_ids_path': str((view / 'bev_ids.png').relative_to(self.data)),
                 'building_point_cloud_npz_path': str((view / 'points.npz').relative_to(self.data))}
        self.geometry = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
        self.obs = self.source / 'global_fused/building_observation_footprints.geojson'
        write_json(self.obs, {'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': props, 'geometry': self.geometry}]})
        write_json(self.source / 'e00_instance_aware_25d/manifest.json', {
            'samples': [{'id': self.sid, 'input': {'gt_depth_for_audit_only': '/never/copied.exr'},
                         'outputs': {'v7b_class_ids': str((view / 'v7b.png').relative_to(self.data)),
                                     'class_aware_depth_zc_m': str((view / 'calibrated_depth.npy').relative_to(self.data))}}]})
        outputs = {}
        for key in E09_OUTPUTS:
            path = view / (key + ('.npy' if key == 'surface_z' else '.png'))
            if key == 'surface_z':
                np.save(path, np.ones((8, 8), dtype=np.float32))
            else:
                Image.new('RGB' if key == 'bev_rgb' else 'L', (8, 8), 1).save(path)
            outputs[key] = str(path.relative_to(self.data))
        write_json(self.source / 'e09_full_pipeline/manifest.json', {
            'samples': [{'id': self.sid, 'outputs': outputs,
                         'input': {'historical_gt': '/never/copied'},
                         'rasterization': {'pixel_center_index_to_ue_world_xy_3x3': np.eye(3).tolist()}}]})

    def bundle(self):
        return bundle_cache(self.source, self.data, self.output, True, include_visibility=True)

    def rehash(self, path):
        manifest = read_json(self.output / 'INPUTS.json')
        for row in manifest['files']:
            if row['path'] == path.relative_to(self.output).as_posix():
                row.update(sha256=sha256(path), bytes=path.stat().st_size)
        manifest['total_bytes'] = sum(row['bytes'] for row in manifest['files'])
        write_json(self.output / 'INPUTS.json', manifest)

    def test_relocated_bundle_is_complete_and_keeps_numerical_bytes(self):
        source_point = self.data / 'views' / self.sid / 'points.npz'
        expected = sha256(source_point)
        self.bundle()
        destination = self.root / 'moved' / 'standalone'
        destination.parent.mkdir()
        shutil.move(str(self.output), destination)
        shutil.rmtree(self.data)
        report = validate_bundle(destination, True)
        self.assertTrue(report['self_contained'])
        self.assertEqual(report['sample_count'], 1)
        obs = destination / 'source/global_fused/building_observation_footprints.geojson'
        feature = read_json(obs)['features'][0]
        self.assertEqual(feature['geometry'], self.geometry)
        self.assertEqual(sha256(destination / feature['properties']['building_point_cloud_npz_path']), expected)
        inventory = build_inventory(obs, destination / 'source', destination,
                                    destination / 'visibility_inputs.json')
        self.assertEqual(set(inventory), {self.sid})
        self.assertTrue(all(p.is_relative_to(destination) for p in collect_input_paths(
            obs, destination / 'source', destination, destination / 'visibility_inputs.json')))
        for path in destination.rglob('*.json'):
            text = path.read_text()
            self.assertNotIn(str(self.data), text)
            self.assertNotIn('/never/', text)
            self.assertNotIn('owned_building_prism_point_count', text)
        self.assertEqual(read_json(destination / 'visibility_inputs.json')['depth_stage'],
                         'class_aware_calibrated_prediction')
        self.assertTrue(all(not p.is_symlink() and p.stat().st_nlink == 1
                            for p in destination.rglob('*') if p.is_file()))

    def test_missing_surface_fails_before_materialization(self):
        (self.data / 'views' / self.sid / 'surface_z.npy').unlink()
        with self.assertRaises(FileNotFoundError):
            self.bundle()
        self.assertFalse(self.output.exists())

    def test_checksums_reject_mutated_prediction_array(self):
        self.bundle()
        path = next((self.output / 'data/points').iterdir())
        content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, 'Changed bundle input'):
            validate_bundle(self.output, True)

    def test_references_must_stay_inside_bundle_even_with_updated_hash(self):
        self.bundle()
        path = self.output / 'visibility_inputs.json'
        doc = read_json(path); doc['samples'][self.sid]['predicted_depth'] = '../outside.npy'
        write_json(path, doc); self.rehash(path)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            validate_bundle(self.output, False)

    def test_review_and_extra_unlisted_inputs_are_rejected(self):
        self.bundle()
        path = self.output / 'buildings.json'
        config = read_json(path); config['review_file'] = 'review.json'
        write_json(path, config); self.rehash(path)
        with self.assertRaisesRegex(ValueError, 'no review'):
            validate_bundle(self.output, False)
        config['review_file'] = None; write_json(path, config); self.rehash(path)
        (self.output / 'extra_gt.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            validate_bundle(self.output, False)


if __name__ == '__main__':
    unittest.main()
