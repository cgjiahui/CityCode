"""Explicit scalar replay changes only image-derived Z and remains frozen."""
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from citycode_assets import batch, batch_gpu, generation
from citycode_assets.appearance import completion_contract, sha256_file, write_json
import test_unified_batch as fixtures


class ImageHeightSpanBatch(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UnifiedBuildingBatch()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_valid_scalar_is_in_plan_and_frozen_record(self):
        rule = self.fixture.rules['sources'][0]
        rule.update(structure_conditioning='image_height_span', structure_height_voxels=27)
        write_json(self.fixture.dispositions, self.fixture.rules)
        root, _ = self.fixture.completed_review()
        row = batch.read(root/'plan.json')['records'][0]
        self.assertEqual(row['structure_height_voxels'], 27)
        batch.freeze(self.fixture.config)
        self.assertEqual(batch_gpu.frozen_selection(self.fixture.config)[0]['structure_height_voxels'], 27)
        changed = copy.deepcopy(self.fixture.rules)
        changed['sources'][0]['structure_height_voxels'] = 26
        write_json(self.fixture.dispositions, changed)
        with self.assertRaisesRegex(ValueError, 'Run inputs changed'):
            batch_gpu.frozen_selection(self.fixture.config)

    def test_missing_noninteger_and_out_of_range_scalars_fail_before_preparation(self):
        original = copy.deepcopy(self.fixture.rules)
        for span in [None, True, False, 0, 33, 4.0, '4']:
            with self.subTest(span=span):
                rules = copy.deepcopy(original)
                rules['sources'][0]['structure_conditioning'] = 'image_height_span'
                if span is not None:
                    rules['sources'][0]['structure_height_voxels'] = span
                write_json(self.fixture.dispositions, rules)
                with patch.object(batch, '_prepare_one') as prepare:
                    with self.assertRaisesRegex(ValueError, 'integer from1 to32'):
                        self.fixture.prepare()
                    prepare.assert_not_called()
                self.assertFalse((Path(self.fixture.config['output'])/'plan.json').exists())

    def test_rank_does_not_leak_scalar_or_mode_to_another_asset(self):
        root = Path(self.fixture.config['output'])
        rows = []
        modes = [('image_height_span', 31), ('height_only', None),
                 ('source_footprint', None), ('image_height_span', 1)]
        for index, (mode, span) in enumerate(modes):
            row = {'asset_id': str(index), 'structure_conditioning': mode,
                   'building': {'path': 'building.json'}, 'appearance_directory': 'appearance'}
            if span is not None:
                row['structure_height_voxels'] = span
            rows.append(row)
        snapshots = []
        class Runtime:
            def __init__(self, config): self.config = config
            def generate(self, building, appearance, output):
                snapshots.append(copy.deepcopy(self.config))
                write_json(output/'raw_manifest.json', {'status': 'complete'})
                return {'inference_seconds': 0}
        with patch.object(batch_gpu, 'frozen_selection', return_value=rows), \
                patch.object(batch_gpu, 'runtime_config', return_value={'structure_height_voxels': 17}), \
                patch.object(generation, 'GenerationRuntime', Runtime):
            self.assertEqual(batch_gpu.infer_rank(self.fixture.config, 0, 1), 0)
        self.assertEqual(snapshots, [
            {'structure_conditioning': 'image_height_span', 'structure_height_voxels': 31}, {},
            {'structure_conditioning': 'source_footprint'},
            {'structure_conditioning': 'image_height_span', 'structure_height_voxels': 1}])
        self.assertTrue((root/'execution/infer_rank0.json').is_file())


class ImageHeightSpanGeneration(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional generation dependency torch is unavailable')
    def test_same_image_stage1_and_explicit_span_determine_raw_cache_identity(self):
        # Run the adapter on CPU tensors and a synthetic pipeline; no weights,
        # CUDA allocation, provider, old occupancy or reference geometry exists.
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            building = root/'building.json'
            write_json(building, {'building_id': 'A'})  # Deliberately no footprint.
            appearance = root/'appearance'
            image = appearance/'generated_main_oblique.png'
            image.parent.mkdir()
            Image.new('RGB', (1024, 1024), 'gray').save(image)
            prepared = {'building_id': 'A', 'model': 'fixture', 'quality': 'fixture',
                'size': '1024x1024', 'prompt_sha256': 'a'*64, 'inputs': [],
                'input_identity_sha256': 'b'*64,
                'provenance': {'building_sha256': sha256_file(building)}}
            write_json(appearance/'appearance_input.json', prepared)
            write_json(appearance/'completion.json', {'status': 'complete',
                'request_identity_sha256': completion_contract(prepared), 'output_sha256': sha256_file(image)})
            original = np.array([(4, 7, 5), (4, 7, 6), (4, 7, 10),
                                 (11, 9, 5), (11, 9, 6), (11, 9, 10)], np.int32)
            sampled = []
            class Pipeline:
                device = 'cpu'
                models = {'shape_slat_flow_model_512': None,
                          'shape_slat_flow_model_1024': None, 'tex_slat_flow_model_1024': None}
                pbr_attr_layout = {'base_color': slice(0, 3), 'alpha': slice(5, 6)}
                def preprocess_image(self, image): return image
                def get_cond(self, *args, **kwargs): return {'cond': torch.zeros((1, 2, 3))}
                def sample_sparse_structure(self, *args, **kwargs):
                    return torch.from_numpy(np.column_stack([np.zeros(len(original), np.int32), original]))
                def sample_shape_slat_cascade(self, *args, **kwargs):
                    sampled.append(args[6].cpu().numpy()[:, 1:])
                    return torch.zeros(1), 1024
                def sample_tex_slat(self, *args, **kwargs): return torch.zeros(1)
                def decode_latent(self, *args, **kwargs):
                    return [SimpleNamespace(vertices=torch.ones((3, 3)),
                        faces=torch.tensor([[0, 1, 2]]*9), attrs=torch.ones((3, 6)),
                        coords=torch.zeros((3, 3), dtype=torch.int32))]
            runtime = generation.GenerationRuntime.__new__(generation.GenerationRuntime)
            runtime.torch, runtime.pipeline = torch, Pipeline()
            runtime.preflight = {'model_manifest_sha256': 'c'*64, 'pipeline_config_sha256': 'd'*64}
            runtime.config = {'structure_conditioning': 'image_height_span', 'structure_height_voxels': 12}
            with patch.object(generation, '_seed'), patch.object(torch.cuda, 'empty_cache'), \
                    patch.object(generation, 'predicted_height_coordinates', side_effect=AssertionError('No footprint fit')):
                first = runtime.generate(building, appearance, root/'first')
                self.assertEqual(runtime.generate(building, appearance, root/'first'), first)
                runtime.config['structure_height_voxels'] = 16
                with self.assertRaisesRegex(ValueError, 'different identity'):
                    runtime.generate(building, appearance, root/'first')
                second = runtime.generate(building, appearance, root/'second')
            self.assertEqual(len(sampled), 2)  # Same identity reused the verified raw cache.
            self.assertNotEqual(first['identity_sha256'], second['identity_sha256'])
            self.assertEqual(first['contract']['structure_conditioning']['mode'], 'image_height_span')
            self.assertEqual(first['contract']['structure_conditioning']['structure_height_voxels'], 12)
            for index, span in enumerate([12, 16]):
                expected, _ = generation.stretch_height(original, span)
                np.testing.assert_array_equal(sampled[index], expected)
                np.testing.assert_array_equal(np.unique(sampled[index][:, :2], axis=0),
                                              np.unique(original[:, :2], axis=0))
            audit = batch.read(root/'first/height_correction.json')
            self.assertFalse(audit['reference_geometry_used'])
            self.assertFalse(audit['footprint_mask_used'])
            self.assertFalse(audit['preexisting_z_gaps_closed'])
            self.assertEqual(audit['height_source'], 'explicit historical scalar')


if __name__ == '__main__':
    unittest.main()
