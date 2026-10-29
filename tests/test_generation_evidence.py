"""Camera metadata and its declared native ROI remain a portable pair."""
from pathlib import Path
import copy
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
from shapely.geometry import box, mapping

from city_pipeline.buildings.evidence import export_evidence
from city_pipeline.runtime import read_json, sha256, write_json


class GenerationEvidenceContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'input'
        self.view = self.data / 'view'
        self.view.mkdir(parents=True)
        self.source = self.data / 'source'
        self.release = self.data / 'release'
        self.output = self.root / 'evidence'
        self.roi = self.view / 'roi' / 'original_valid.png'
        self.roi.parent.mkdir()
        Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(self.roi)
        camera = {'view_id': 'view', 'image_size_px': [32, 32],
                  'intrinsics': {'K': [[10, 0, 16], [0, 10, 16], [0, 0, 1]]},
                  'world_to_camera_4x4_m': np.eye(4).tolist(),
                  'camera_to_world_4x4_m': np.eye(4).tolist(),
                  'strict_valid_mask_path': 'roi/original_valid.png'}
        write_json(self.view / 'camera.json', camera)
        Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(self.view / 'rgb.png')
        ids = np.zeros((32, 32), dtype=np.uint8)
        ids[8:20, 10:24] = 1
        Image.fromarray(ids).save(self.view / 'ids.png')
        properties = {'observation_id': 'observation', 'sample_id': 'patch__view',
                      'patch_id': 'patch', 'view_id': 'view', 'rgb_instance_id': 1, 'bev_instance_id': 1,
                      'rgb_path': 'view/rgb.png', 'source_instance_ids_path': 'view/ids.png',
                      'bev_instance_ids_path': 'view/ids.png'}
        write_json(self.source / 'global_fused/building_observation_footprints.geojson', {
            'type': 'FeatureCollection', 'features': [{'type': 'Feature',
                'geometry': mapping(box(0, 0, 10, 10)), 'properties': properties}]})
        write_json(self.source / 'e09_full_pipeline/manifest.json', {'samples': [{
            'id': 'patch__view', 'outputs': {'bev_rgb': 'view/rgb.png'},
            'rasterization': {'pixel_center_index_to_ue_world_xy_3x3': np.eye(3).tolist()}}]})
        write_json(self.release / 'global_building_footprints.geojson', {
            'type': 'FeatureCollection', 'features': [{'type': 'Feature',
                'geometry': mapping(box(0, 0, 10, 10)), 'properties': {
                    'predicted_global_building_id': 'building', 'canonical_observation_id': 'observation',
                    'consensus_observation_ids': ['observation'], 'base_z_m': 0.,
                    'top_z_m': 10., 'height_m': 10., 'roof_layers': []}}]})

    def export(self):
        return export_evidence(self.source, self.release, self.output, self.data)

    def test_declared_full_camera_roi_is_copied_hashed_and_survives_relocation(self):
        expected_roi = sha256(self.roi)
        expected_camera = sha256(self.view / 'camera.json')
        self.export()
        destination = self.root / 'moved_evidence'
        shutil.move(self.output, destination)
        shutil.rmtree(self.data)
        manifest = read_json(destination / 'manifest.json')
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['pending'], [])
        building = read_json(destination / 'buildings/building/building.json')
        self.assertEqual(building['status'], 'complete')
        observation_root = destination / 'buildings/building/observations/observation'
        camera = read_json(observation_root / 'camera.json')
        self.assertEqual(camera['strict_valid_mask_path'], 'strict_valid_mask.png')
        roi = observation_root / camera['strict_valid_mask_path']
        self.assertEqual(sha256(roi), expected_roi)
        with Image.open(roi) as picture:
            picture.load()
            self.assertEqual(list(picture.size), camera['image_size_px'])
        with Image.open(observation_root / 'source_mask.png') as picture:
            self.assertNotEqual(list(picture.size), camera['image_size_px'])
        observation = read_json(observation_root / 'observation.json')
        self.assertEqual(observation['source_sha256']['strict_valid'], expected_roi)
        self.assertEqual(observation['source_sha256']['camera'], expected_camera)
        self.assertIn('strict_valid_mask.png', observation['files'])
        for name, digest in observation['files'].items():
            self.assertEqual(sha256(observation_root / name), digest)

    def test_declared_missing_roi_is_pending_instead_of_complete(self):
        self.roi.unlink()
        self.export()
        manifest = read_json(self.output / 'manifest.json')
        self.assertEqual(manifest['status'], 'structure_complete_with_pending_evidence')
        self.assertEqual(manifest['pending'][0]['reason'], 'Declared camera strict-valid mask is missing')
        self.assertEqual(manifest['records'][0]['status'], 'structure_only_evidence_pending')
        self.assertEqual(manifest['records'][0]['observation_count'], 0)

    def test_camera_without_declared_roi_remains_supported(self):
        camera = read_json(self.view / 'camera.json')
        camera.pop('strict_valid_mask_path')
        write_json(self.view / 'camera.json', camera)
        self.roi.unlink()
        self.export()
        self.assertEqual(read_json(self.output / 'manifest.json')['status'], 'complete')
        observation_root = self.output / 'buildings/building/observations/observation'
        self.assertNotIn('strict_valid_mask_path', read_json(observation_root / 'camera.json'))
        self.assertNotIn('strict_valid', read_json(observation_root / 'observation.json')['source_sha256'])

    def test_declared_roi_must_match_full_camera_resolution(self):
        Image.new('L', (12, 14), 255).save(self.roi)
        with self.assertRaisesRegex(ValueError, 'native-size single-channel'):
            self.export()
        self.assertFalse((self.output / 'manifest.json').exists())

    def test_roof_and_repair_sources_export_once_with_canonical_first_and_all_roles(self):
        path = self.source / 'global_fused/building_observation_footprints.geojson'
        observations = read_json(path)
        for oid, instance in [('repair_observation', 2), ('roof_observation', 3)]:
            feature = copy.deepcopy(observations['features'][0])
            feature['properties'].update(observation_id=oid, rgb_instance_id=instance, bev_instance_id=instance)
            observations['features'].append(feature)
        write_json(path, observations)
        with Image.open(self.view / 'ids.png') as image:
            ids = np.asarray(image).copy()
        ids[8:12, 10:24] = 2
        ids[12:16, 10:24] = 3
        Image.fromarray(ids).save(self.view / 'ids.png')
        path = self.release / 'global_building_footprints.geojson'
        document = read_json(path)
        properties = document['features'][0]['properties']
        properties.update(member_observation_ids=['observation'],
                          consensus_observation_ids=['observation', 'observation'],
                          v7_observation_ids=['repair_observation', 'observation', 'repair_observation'],
                          roof_layers=[{'geometry': mapping(box(1, 1, 2, 2)), 'base_z_m': 8., 'top_z_m': 10.,
                                        'observation_ids': ['roof_observation', 'observation', 'roof_observation']}])
        write_json(path, document)
        self.export()
        manifest = read_json(self.output / 'manifest.json')
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['records'][0]['observation_count'], 3)
        building = read_json(self.output / 'buildings/building/building.json')
        records = building['observations']
        self.assertEqual([r['observation_id'] for r in records],
                         ['observation', 'repair_observation', 'roof_observation'])
        self.assertEqual([r['is_canonical'] for r in records], [True, False, False])
        self.assertEqual(records[0]['roles'], ['canonical', 'member', 'consensus', 'boundary_repair', 'roof_support'])
        self.assertEqual(records[1]['roles'], ['boundary_repair'])
        self.assertEqual(records[2]['roles'], ['roof_support'])
        for record in records:
            od = self.output / 'buildings/building' / record['path']
            observation = read_json(od / 'observation.json')
            self.assertEqual(observation['roles'], record['roles'])
            self.assertTrue((od / 'source_primary.png').is_file())
            self.assertTrue((od / 'roof_evidence.png').is_file())
        self.assertEqual(building['roof_layers'], read_json(path)['features'][0]['properties']['roof_layers'])

    def test_missing_declared_support_observation_is_an_explicit_error(self):
        path = self.release / 'global_building_footprints.geojson'
        document = read_json(path)
        document['features'][0]['properties']['v7_observation_ids'] = ['missing_support']
        write_json(path, document)
        with self.assertRaisesRegex(ValueError, 'building references missing observations: missing_support'):
            self.export()
        self.assertFalse((self.output / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
