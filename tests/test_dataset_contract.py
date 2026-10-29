"""Observation-only discovery, path confinement and stage selection."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from citycode_perception.cli import recovery_config
from citycode_perception.dataset import discover_samples, validate_dataset


class DatasetContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'dataset'
        self.dataset.mkdir()
        self.ids = ['SC_P320_R+000_C+003', 'SC_P320_R+000_C+004']
        self.write(self.dataset / 'dataset_manifest.json', {'selected_views': ['tilt_light']})
        self.write(self.dataset / 'patch_index.json', {'patches': [{'patch_id': sid} for sid in self.ids]})
        for sid in self.ids:
            patch = self.dataset / 'patches' / sid
            view = patch / 'views' / 'tilt_light'
            view.mkdir(parents=True)
            Image.new('RGB', (8, 8)).save(view / 'rgb.png')
            Image.new('L', (8, 8), 255).save(view / 'strict_valid_mask.png')
            identity = [[float(i == j) for j in range(4)] for i in range(4)]
            self.write(view / 'camera.json', {'image_size_px': [8, 8],
                       'camera_to_world_4x4_m': identity, 'world_to_camera_4x4_m': identity})
            self.write(patch / 'patch.json', {'data_bounds_xy_world_m': [0, 0, 8, 8],
                       'core_bounds_xy_world_m': [1, 1, 7, 7]})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_observation_only_input_validates_and_unknown_selection_fails(self):
        self.assertEqual(validate_dataset(self.dataset)['view_count'], 2)
        self.assertEqual(discover_samples(self.dataset, [self.ids[1]])[0].stem,
                         self.ids[1] + '__tilt_light')
        with self.assertRaisesRegex(ValueError, 'Unknown patch'):
            discover_samples(self.dataset, ['missing'])

    def test_index_cannot_redirect_observations_outside_dataset(self):
        self.write(self.dataset / 'patch_index.json', {'patches': [
            {'patch_id': self.ids[0], 'directory': '../outside'}]})
        with self.assertRaisesRegex(ValueError, 'escapes dataset'):
            discover_samples(self.dataset)

    def test_camera_size_mismatch_is_rejected(self):
        mask = self.dataset / 'patches' / self.ids[0] / 'views/tilt_light/strict_valid_mask.png'
        Image.new('L', (4, 8), 255).save(mask)
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            validate_dataset(self.dataset)

    def test_recovery_selection_uses_explicit_priors_and_no_historical_review(self):
        depth, masks = self.root / 'depth', self.root / 'masks'
        depth.mkdir()
        (depth / 'prediction.npy').write_bytes(b'preflight-path-fixture')
        sid = self.ids[0] + '__tilt_light'
        manifest = {'samples': [{'patch_id': self.ids[0], 'view_id': 'tilt_light',
                    'depth_stage': 'raw_prediction', 'outputs': {'depth_raw_zc_m': 'prediction.npy'}}]}
        self.write(depth / 'manifest.json', manifest)
        for stage in ('structure_v9_2', 'structure_v7b'):
            folder = masks / stage; folder.mkdir(parents=True)
            (folder / 'mask.png').write_bytes(b'preflight-path-fixture')
            self.write(folder / 'manifest.json', {'samples': [{'id': sid, 'result': 'mask.png'}]})
        priors = self.root / 'priors.json'
        self.write(priors, {'unit': 'metre',
                           'predicted_background_semantic_id_to_prior_z_m': {'1': 2., '2': -1., '3': 2.}})
        config = recovery_config(self.dataset, depth, masks, patches=[self.ids[0]], priors=priors)
        self.assertEqual(config['sample_ids'], [self.ids[0] + '__tilt_light'])
        self.assertEqual(config['priors_path'], str(priors))
        self.assertIsNone(config['review_file'])
        self.assertEqual(config['polish']['name'], 'clean-v2')
        self.assertEqual(config['postprocess']['boundary']['offset_m'], 1.25)
        self.assertTrue(config['roof_retention']['clipped_enabled'])
        self.assertEqual(config['height_consensus']['maximum_adjustment_m'], 3.)
        self.assertEqual(config['selection']['area_weight'], .5)
        self.assertEqual(config['visibility_refinement']['method'], 'local_surface_veto')
        structure = recovery_config(self.dataset, depth, masks, patches=[self.ids[0]], priors=priors,
                                    structure_only=True)
        self.assertFalse(structure['export_generation_evidence'])
        manifest['samples'][0]['depth_stage'] = 'already_calibrated_prediction'
        self.write(depth / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'requires raw model depth'):
            recovery_config(self.dataset, depth, masks, patches=[self.ids[0]], priors=priors)
        manifest['samples'][0]['depth_stage'] = 'raw_prediction'
        self.write(depth / 'manifest.json', manifest)
        self.write(priors, {'unit': 'metre',
                           'predicted_background_semantic_id_to_prior_z_m': {'1': True, '2': -1., '3': 2.}})
        with self.assertRaisesRegex(ValueError, 'finite scalar'):
            recovery_config(self.dataset, depth, masks, patches=[self.ids[0]], priors=priors)


if __name__ == '__main__':
    unittest.main()
