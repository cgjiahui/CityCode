"""The finite CPU fallback preserves actual source values and the existing pose."""
from dataclasses import asdict
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import numpy as np
from PIL import Image
from shapely.geometry import box, mapping

from citycode_assets.extrema_lod import export_extrema_preserving_lod, extrema_lock_mask, _pack_glb
from citycode_assets.lod import _accessor, _read_glb, sha256
from citycode_assets.registration import FitPolicy
from test_asset_delivery_contract import cube_glb


class ExtremaLodContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source.glb'
        cube_glb(self.source, nodes=[{'children': [1]}, {'mesh': 0, 'translation': [3, 2, -5]}])
        data, document, binary = _read_glb(self.source)
        image = io.BytesIO();Image.new('RGBA', (2048, 2048), (20, 50, 100, 255)).save(image, format='PNG')
        payload = bytes(data[binary:]);offset = len(payload)
        payload += image.getvalue()
        document['bufferViews'][4] = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(image.getvalue())}
        document['buffers'][0]['byteLength'] = len(payload)
        self.source.write_bytes(_pack_glb(document, payload))
        self.matrix = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        self.fit = {'status': 'accepted', 'source_sha256': sha256(self.source), 'reasons': [],
                    'policy': asdict(FitPolicy()), 'world_from_asset_4x4': self.matrix,
                    'predicted_base_z_m': 2., 'predicted_top_z_m': 3.,
                    'target_geometry': mapping(box(3, 4, 4, 5))}

    @unittest.skipUnless(shutil.which('node'), 'Node.js CPU WebAssembly is required')
    def test_real_backend_keeps_node_pose_images_attributes_and_source_bytes(self):
        prior = self.source.read_bytes();fit = copy.deepcopy(self.fit)
        output = self.root / 'candidate.glb'
        report = export_extrema_preserving_lod(self.source, output, fit_record=self.fit, target_faces=12)
        self.assertEqual(report['status'], 'accepted_candidate', report['errors'])
        self.assertEqual(report['backend']['invocation_count'], 1)
        self.assertEqual(report['backend']['flags'], [])
        self.assertEqual(report['low']['faces'], 12)
        self.assertAlmostEqual(report['low']['ground_error_m'], 0)
        self.assertAlmostEqual(report['low']['top_error_m'], 0)
        self.assertTrue(report['same_source_normal_uv_material_images'])
        self.assertTrue(report['source_node_transforms_preserved'])
        self.assertEqual(self.fit, fit);self.assertEqual(self.source.read_bytes(), prior)
        source_data, source_doc, source_binary = _read_glb(self.source)
        low_data, low_doc, low_binary = _read_glb(output)
        source_primitive = source_doc['meshes'][0]['primitives'][0]
        low_primitive = low_doc['meshes'][0]['primitives'][0]
        for name in ['POSITION', 'NORMAL', 'TEXCOORD_0']:
            a = _accessor(source_data, source_doc, source_binary, source_primitive['attributes'][name])[0]
            b = _accessor(low_data, low_doc, low_binary, low_primitive['attributes'][name])[0]
            self.assertTrue(np.array_equal(a, b), name)
        self.assertEqual(low_doc['accessors'][low_primitive['indices']]['componentType'], 5125)

    @unittest.skipUnless(shutil.which('node'), 'Node.js CPU WebAssembly is required')
    def test_impossible_locked_budget_stays_review_without_retry_or_prune(self):
        report = export_extrema_preserving_lod(self.source, self.root / 'too_small.glb', fit_record=self.fit, target_faces=9)
        self.assertEqual(report['status'], 'review_required')
        self.assertIn('low:face_budget', report['errors'])
        self.assertEqual(report['backend']['invocation_count'], 1)
        self.assertEqual(report['backend']['flags'], [])
        self.assertEqual(report['world_from_asset_4x4'], self.matrix)

    @unittest.skipUnless(shutil.which('node'), 'Node.js CPU WebAssembly is required')
    def test_strict_mode_restores_a_protected_face_discarded_by_backend(self):
        real_run = subprocess.run
        def discard_one_face(command, **kwargs):
            result = real_run(command, **kwargs)
            if result.returncode == 0:
                output = Path(command[-1]) / 'simplified_indices.u32'
                faces = np.fromfile(output, dtype='<u4').reshape(-1, 3)
                faces[:-1].tofile(output)
            return result
        with patch('citycode_assets.extrema_lod.subprocess.run', side_effect=discard_one_face):
            report = export_extrema_preserving_lod(self.source, self.root / 'strict.glb',
                fit_record=self.fit, target_faces=12, strict_protected_faces=True)
        self.assertEqual(report['status'], 'accepted_candidate', report['errors'])
        self.assertEqual(report['protected_faces_restored_after_backend'], 1)
        self.assertTrue(report['protected_source_faces_retained'])
        self.assertTrue(report['same_source_normal_uv_material_images'])
        self.assertEqual(report['low']['faces'], 12)

    def test_uv_duplicates_and_small_nonextreme_component_are_locked(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [0, 1, 0], [3, .5, 0], [3.1, .5, 0], [3, .6, .1]], dtype='<f4')
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 4, 3], [1, 3, 4], [5, 6, 7]])
        locks, evidence = extrema_lock_mask(vertices, faces, self.matrix, .05)
        self.assertEqual(int(locks[2]), 1);self.assertEqual(int(locks[4]), 1)
        self.assertTrue(locks[5:8].all())
        self.assertEqual(evidence['physical_component_count'], 2)
        self.assertTrue(any(np.array_equal(row, [5, 6, 7]) for row in evidence['protected_faces']))

    def test_changed_fit_source_policy_and_existing_output_are_rejected(self):
        output = self.root / 'candidate.glb'
        for key, value in [('source_sha256', '0' * 64), ('status', 'review_required'), ('policy', {})]:
            with self.subTest(key=key):
                fit = copy.deepcopy(self.fit);fit[key] = value
                with self.assertRaises(ValueError): export_extrema_preserving_lod(self.source, output, fit_record=fit)
                self.assertFalse(output.exists())
        output.write_bytes(b'preserve existing asset')
        with self.assertRaises(ValueError): export_extrema_preserving_lod(self.source, output, fit_record=self.fit)
        self.assertEqual(output.read_bytes(), b'preserve existing asset')
        with self.assertRaises(ValueError):
            export_extrema_preserving_lod(self.source, self.root / 'oversize.glb', fit_record=self.fit, target_faces=100001)

    def test_invalid_world_matrix_cannot_define_protection(self):
        positions = [[0., 0, 0], [1, 0, 0], [0, 1, 0]];faces = [[0, 1, 2]]
        for matrix in [np.zeros((4, 4)), np.diag([1., 1, 1, 2])]:
            with self.assertRaises(ValueError): extrema_lock_mask(positions, faces, matrix, .05)

    @unittest.skipUnless(shutil.which('node'), 'Node.js CPU WebAssembly is required')
    def test_visual_profile_allows_only_partial_footprint_disagreement(self):
        fit=copy.deepcopy(self.fit);fit.update(status='review_required',reasons=['actual_mesh_target_coverage_below_threshold'],target_geometry=mapping(box(3,4,7,5)))
        report=export_extrema_preserving_lod(self.source,self.root/'visual.glb',fit_record=fit,target_faces=12,validation_profile='visual_scene_completion')
        self.assertEqual(report['status'],'accepted_candidate',report['errors'])
        self.assertEqual(report['validation_profile'],'visual_scene_completion')
        fit['reasons'].append('height_scale_change_exceeds_bound')
        with self.assertRaises(ValueError):export_extrema_preserving_lod(self.source,self.root/'bad.glb',fit_record=fit,target_faces=12,validation_profile='visual_scene_completion')


if __name__ == '__main__': unittest.main()
