import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from citycode_perception import depth
from citycode_perception.model_manifest import depth_model_requirements


class FakeBackend:
    resource = {'model_version': 'v2', 'test_only': True}
    def __init__(self):
        self.calls = []
    def predict(self, rgb, *, fov_x, resolution_level):
        self.calls.append((rgb.shape, fov_x, resolution_level))
        return np.full(rgb.shape[:2], 7.5, np.float32)


class DepthContracts(unittest.TestCase):
    def make_sample(self, root):
        directory = root / 'patches/P001/views/tilt_light'
        directory.mkdir(parents=True)
        rgb = np.zeros((28, 28, 3), np.uint8)
        rgb[:, :, 0] = 120
        valid = np.ones((28, 28), np.uint8) * 255
        valid[:, 0] = 0
        Image.fromarray(rgb).save(directory / 'rgb.png')
        Image.fromarray(valid).save(directory / 'strict_valid_mask.png')
        camera = {'image_size_px': [28, 28], 'intrinsics': {'horizontal_fov_deg': 67.0}}
        (directory / 'camera.json').write_text(json.dumps(camera))
        return types.SimpleNamespace(patch_id='P001', view_id='tilt_light', stem='P001__tilt_light',
                                     rgb_path=directory / 'rgb.png', camera_path=directory / 'camera.json',
                                     valid_mask_path=directory / 'strict_valid_mask.png', patch_path=root / 'patches/P001/patch.json')

    def discovery(self, sample):
        module = types.ModuleType('citycode_perception.dataset')
        module.discover_samples = lambda *args, **kwargs: [sample]
        return patch.dict(sys.modules, {'citycode_perception.dataset': module})

    def test_native_2560_and_known_fov_without_reference_inputs(self):
        rgb = np.zeros((2560, 2560, 3), np.uint8)
        valid = np.ones((2560, 2560), bool)
        valid[:32] = False
        camera = {'image_size_px': [2560, 2560], 'intrinsics': {'horizontal_fov_deg': 71.0}}
        backend = FakeBackend()
        prediction, metadata = depth.infer_depth(rgb, camera, valid, backend)
        self.assertEqual(backend.calls, [((1022, 1022, 3), 71.0, 9)])
        self.assertEqual(prediction.shape, (2560, 2560))
        self.assertEqual(prediction.dtype, np.float32)
        self.assertTrue(np.isnan(prediction[:32]).all())
        self.assertTrue(np.allclose(prediction[32:], 7.5))
        self.assertEqual(metadata['network_size_px'], [1022, 1022])
        self.assertEqual(metadata['depth_definition'], 'camera-forward Zc')

    def test_fov_can_use_pixel_intrinsics(self):
        self.assertAlmostEqual(depth.horizontal_fov({'intrinsics': {'fx_px': 512.0}}, 1024), 90.0)
        with self.assertRaises(ValueError):
            depth.horizontal_fov({'intrinsics': {'horizontal_fov_deg': float('nan')}}, 1024)

    def test_shape_and_roi_validation(self):
        rgb = np.zeros((28, 28, 3), np.uint8)
        camera = {'image_size_px': [28, 27], 'intrinsics': {'horizontal_fov_deg': 67}}
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            depth.infer_depth(rgb, camera, np.ones((28, 28), bool), FakeBackend())
        camera['image_size_px'] = [28, 28]
        with self.assertRaisesRegex(ValueError, 'Strict-valid'):
            depth.infer_depth(rgb, camera, np.zeros((28, 28), bool), FakeBackend())
        with self.assertRaises(ValueError):
            depth.inference_shape(2560, 2560, 0)

    def test_predict_dataset_manifest_and_single_raw_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample = self.make_sample(root / 'dataset'); output = root / 'depth'
            backend = FakeBackend()
            with self.discovery(sample), patch.object(depth, 'load_full_model', return_value=backend):
                manifest = depth.predict_dataset(root / 'dataset', output, root / 'fake.pt', device='cpu', inference_size=28)
            data = json.loads(manifest.read_text()); row = data['samples'][0]
            self.assertEqual(data['path_resolution'], 'manifest_directory')
            self.assertEqual(row['outputs']['depth_raw_zc_m'], row['outputs']['depth_ground_scaled_zc_m'])
            self.assertFalse(row['scale_calibration']['applied'])
            self.assertEqual(row['scale_calibration']['scale'], 1.0)
            array = np.load(output / row['outputs']['depth_raw_zc_m'])
            self.assertTrue(np.isnan(array[:, 0]).all())
            self.assertEqual(len(list(output.rglob('*.npy'))), 1)
            self.assertEqual(set(row['input']), {'rgb', 'camera', 'strict_valid_mask'})
            self.assertFalse(data['prediction_contract']['uses_reference_depth'])

    def test_real_public_sample_discovery_wiring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset = root / 'dataset'; sample = self.make_sample(dataset)
            sample.patch_path.write_text('{}')
            (dataset / 'dataset_manifest.json').write_text(json.dumps({'selected_views': ['tilt_light']}))
            (dataset / 'patch_index.json').write_text(json.dumps({'patches': [{'patch_id': 'P001'}]}))
            with patch.object(depth, 'load_full_model', return_value=FakeBackend()):
                manifest = depth.predict_dataset(dataset, root / 'predicted', root / 'toy.pt', device='cpu', inference_size=28)
            imported = depth.import_predicted_depth(dataset, manifest.parent, root / 'imported')
            self.assertEqual(json.loads(imported.read_text())['samples'][0]['sample_id'], sample.stem)

    def test_fixture_import_masks_roi_and_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample = self.make_sample(root / 'dataset'); source = root / 'source'; source.mkdir()
            raw = np.full((28, 28), 4.0, np.float64); raw[1, 1] = -3
            path = source / 'raw.npy'; np.save(path, raw)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            (source / 'manifest.json').write_text(json.dumps({'samples': [{'patch_id': 'P001', 'view_id': 'tilt_light', 'outputs': {'depth_raw_zc_m': 'raw.npy'}}]}))
            with self.discovery(sample):
                manifest = depth.import_predicted_depth(root / 'dataset', source, root / 'imported')
            data = json.loads(manifest.read_text()); output = manifest.parent / data['samples'][0]['outputs']['depth_raw_zc_m']
            array = np.load(output)
            self.assertTrue(np.isnan(array[:, 0]).all()); self.assertTrue(np.isnan(array[1, 1]))
            self.assertEqual(array.dtype, np.float32)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            self.assertEqual(data['mode'], 'predicted_depth_import')
            self.assertIsNone(data['model']['checkpoint_identity'])

    def test_cached_calibration_lineage_is_preserved_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset = root / 'dataset'; sample = self.make_sample(dataset)
            source = root / 'source'; source.mkdir(); np.save(source / 'prediction.npy', np.ones((28, 28), np.float32))
            note = 'Smoke input uses cached predicted class-aware depth as input; not original raw checkpoint output. No GT depth.'
            calibration = {'scale': 1.0, 'method': 'already-calibrated prediction used as smoke input'}
            original = {'samples': [{'patch_id': 'P001', 'view_id': 'tilt_light',
                        'outputs': {'depth_raw_zc_m': 'prediction.npy'}, 'scale_calibration': calibration}], 'note': note}
            manifest = source / 'manifest.json'; manifest.write_text(json.dumps(original))
            with self.discovery(sample):
                output = depth.import_predicted_depth(dataset, source, root / 'imported')
            payload = json.loads(output.read_text()); row = payload['samples'][0]
            self.assertEqual(payload['depth_stage'], 'already_calibrated_prediction')
            self.assertEqual(row['depth_stage'], 'already_calibrated_prediction')
            self.assertEqual(row['inference']['mode'], 'imported_already_calibrated_prediction')
            self.assertEqual(row['scale_calibration']['method'], calibration['method'])
            self.assertFalse(row['scale_calibration']['applied_by_import'])
            self.assertTrue(row['scale_calibration']['prior_calibration_preserved'])
            self.assertEqual(row['imported_prediction']['source_note'], note)
            self.assertEqual(row['imported_prediction']['source_scale_calibration'], calibration)
            self.assertEqual(payload['model']['source_manifest_sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertFalse(payload['prediction_contract']['legacy_scaled_output_is_raw_alias'])
            self.assertTrue(payload['prediction_contract']['raw_output_key_is_stage_compatibility_alias'])
            self.assertTrue(row['outputs']['depth_raw_zc_m'].startswith('depth_imported_zc_m/'))

    def test_explicit_import_depth_stage_override(self):
        self.assertEqual(depth._import_depth_stage({}, {}, 'already_calibrated_prediction'), 'already_calibrated_prediction')
        with self.assertRaises(ValueError):
            depth._import_depth_stage({}, {}, 'ground_truth')

    def test_fixture_never_falls_back_to_scaled_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample = self.make_sample(root / 'dataset'); source = root / 'source'; source.mkdir()
            np.save(source / 'scaled.npy', np.ones((28, 28), np.float32))
            (source / 'manifest.json').write_text(json.dumps({'samples': [{'patch_id': 'P001', 'view_id': 'tilt_light', 'outputs': {'depth_ground_scaled_zc_m': 'scaled.npy'}}]}))
            with self.discovery(sample), self.assertRaises(FileNotFoundError):
                depth.import_predicted_depth(root / 'dataset', source, root / 'imported')

    def test_reference_view_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = self.make_sample(Path(directory)); sample.view_id = 'ortho_reference'
            with self.assertRaisesRegex(ValueError, 'Reference'):
                depth._sample_inputs(sample)

    def test_resource_manifest_distinguishes_known_lora_from_unpublished_full(self):
        manifest = depth_model_requirements()
        self.assertEqual(manifest['project_p384_full_checkpoint']['sha256'], '63cf3463631911ea3ce586d576926ec41230f93df547755e594994ded3bbc873')
        self.assertEqual(manifest['project_p384_full_checkpoint']['bytes'], 1323776820)
        self.assertIsNone(manifest['project_p384_full_checkpoint']['download_url'])
        self.assertFalse(manifest['upstream_aerial_lora_checkpoint']['compatible_with_full_adapter'])
        self.assertEqual(len(manifest['upstream_aerial_lora_checkpoint']['sha256']), 64)

    def test_full_loader_uses_v2_strict_state_and_safe_cpu_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'toy.pt'; checkpoint.write_bytes(b'toy-full')
            source = root / 'model.py'; source.write_text('test_only = True\n')
            calls = []
            class FakeModel:
                def __init__(self, **config): calls.append(('config', config))
                def load_state_dict(self, state, strict): calls.append(('state', state, strict))
                def to(self, device): calls.append(('device', device)); return self
                def eval(self): return self
            module = types.ModuleType('moge.model'); module.__file__ = str(source)
            module.import_model_class_by_version = lambda version: (calls.append(('version', version)) or FakeModel)
            torch = types.ModuleType('torch')
            payload = {'model_config': {'toy': True}, 'model': {'weight': 3}, 'ue_finetune': {'global_step': 4800}}
            torch.load = lambda path, **kwargs: (calls.append(('load', kwargs)) or payload)
            with patch.dict(sys.modules, {'torch': torch, 'moge.model': module}):
                backend = depth.load_full_model(checkpoint, device='cpu')
                self.assertFalse(backend.fp16)
                self.assertIn(('version', 'v2'), calls)
                self.assertIn(('state', {'weight': 3}, True), calls)
                self.assertIn(('load', {'map_location': 'cpu', 'weights_only': True}), calls)
                payload['model_version'] = 'v1'
                with self.assertRaisesRegex(ValueError, 'v2'):
                    depth.load_full_model(checkpoint, device='cpu')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    depth.load_full_model(checkpoint, device='cpu', expected_sha256='0' * 64)
                payload.clear(); payload['model'] = {}
                with self.assertRaisesRegex(ValueError, 'Full checkpoint'):
                    depth.load_full_model(checkpoint, device='cpu')


if __name__ == '__main__':
    unittest.main()
