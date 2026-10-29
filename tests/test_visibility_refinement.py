"""Portable inputs and an actual prediction-only visibility reconstruction."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
from shapely.geometry import box, mapping, shape

from city_pipeline.buildings.visibility_refinement import build_inventory, collect_input_paths, run
from city_pipeline.runtime import read_json, write_json


class VisibilityRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'portable'
        self.root.mkdir()
        self.observations = self.root / 'observations.geojson'
        self.inventory = self.root / 'visibility_inputs.json'
        self.upstream = self.root / 'upstream'
        self.parent = self.root / 'buildings'
        camera = {
            'world_to_camera_4x4_m': [[1, 0, 0, 0], [0, 1, 0, 0],
                                      [0, 0, 1, 1000], [0, 0, 0, 1]],
            'intrinsics': {'K': [[1000, 0, 20], [0, 1000, 20], [0, 0, 1]]},
            'image_size_px': [64, 64], 'data_bounds_xy_world_m': [-5, -5, 15, 15],
        }
        xx, yy = np.meshgrid(np.arange(-2, 12.1, .5), np.arange(-2, 12.1, .5))
        xyz = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 20.)]
        observations, samples, upstream_samples = [], {}, []
        for patch in ('source', 'other1', 'other2'):
            sample = patch + '__view'
            directory = self.root / 'views' / sample
            directory.mkdir(parents=True)
            write_json(directory / 'camera.json', camera)
            Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(directory / 'strict_valid_mask.png')
            Image.fromarray(np.full((64, 64), patch == 'source', dtype=np.uint8)).save(directory / 'ids.png')
            Image.fromarray(np.ones((64, 64), dtype=np.uint8)).save(directory / 'background.png')
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(directory / 'rgb.png')
            # Depth agrees with a local surface ten metres above the modeled
            # roof. Its top-height silhouette has no depth agreement; direct
            # points recover the local surface and veto deletion in the next pass.
            np.save(directory / 'depth.npy', np.full((64, 64), 1020. if patch == 'source' else 2000.))
            np.savez(directory / 'points.npz', xyz_m=xyz, instance_id=np.ones(len(xyz), dtype=np.int32),
                     provenance=np.ones(len(xyz), dtype=np.uint8))
            relative = directory.relative_to(self.root)
            observations.append({'type': 'Feature', 'geometry': mapping(box(0, 0, 10, 10)), 'properties': {
                'observation_id': patch + '_obs', 'sample_id': sample, 'patch_id': patch, 'view_id': 'view',
                'rgb_instance_id': 1, 'top_z_m': 10., 'preliminary_quality': 100.,
                'stable_layer_count': 1, 'boundary_status': 'complete',
                'building_point_cloud_npz_path': str(relative / 'points.npz'),
                'source_instance_ids_path': str(relative / 'ids.png'), 'rgb_path': str(relative / 'rgb.png'),
            }})
            samples[sample] = {key: str(relative / name) for key, name in {
                'camera': 'camera.json', 'strict_valid': 'strict_valid_mask.png', 'source_ids': 'ids.png',
                'background_ids': 'background.png', 'predicted_depth': 'depth.npy'}.items()}
            upstream_samples.append({'id': sample, 'outputs': {
                'v7b_class_ids': str(relative / 'background.png'),
                'class_aware_depth_zc_m': str(relative / 'depth.npy')}})
        write_json(self.observations, {'type': 'FeatureCollection', 'features': observations})
        write_json(self.inventory, {'schema': 'city_pipeline.visibility_inputs.v1',
                                   'uses_ground_truth': False, 'samples': samples})
        write_json(self.upstream / 'e00_instance_aware_25d/manifest.json', {'samples': upstream_samples})
        properties = {'predicted_global_building_id': 'building', 'canonical_observation_id': 'source_obs',
            'canonical_patch_id': 'source', 'canonical_view_id': 'view', 'canonical_instance_id': 1,
            'base_z_m': 0., 'top_z_m': 10., 'body_top_z_m': 10., 'height_m': 10., 'roof_layers': []}
        write_json(self.parent / 'global_building_footprints.geojson', {'type': 'FeatureCollection', 'features': [
            {'type': 'Feature', 'properties': properties, 'geometry': mapping(box(0, 0, 10, 10))}]})
        write_json(self.parent / 'manifest.json', {'status': 'complete'})

    def test_portable_inventory_matches_raw_and_can_move_without_upstream(self):
        raw = build_inventory(self.observations, self.upstream, self.root)
        portable = build_inventory(self.observations, self.upstream, self.root, self.inventory)
        self.assertEqual(raw, portable)
        destination = self.root.parent / 'moved'
        shutil.copytree(self.root, destination)
        shutil.rmtree(destination / 'upstream')
        moved = build_inventory(destination / self.observations.name, destination / 'missing', destination,
                                destination / self.inventory.name)
        self.assertEqual(len(moved), 3)
        self.assertTrue(all(Path(path).is_relative_to(destination) for row in moved.values() for path in row.values()))

    def test_inventory_covers_all_actual_prediction_dependencies(self):
        paths = set(collect_input_paths(self.observations, self.upstream, self.root, self.inventory))
        self.assertEqual(len(paths), 20)  # Observations, inventory, 3 x (5 view inputs + points).
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertNotIn(self.upstream / 'e00_instance_aware_25d/manifest.json', paths)
        raw = set(collect_input_paths(self.observations, self.upstream, self.root))
        self.assertIn(self.upstream / 'e00_instance_aware_25d/manifest.json', raw)
        self.assertEqual(sum(path.name == 'rgb.png' for path in raw), 3)

    def test_inventory_rejects_missing_views_and_duplicate_physical_patches(self):
        document = read_json(self.inventory)
        document['samples'].pop('other2__view')
        write_json(self.inventory, document)
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            build_inventory(self.observations, self.upstream, self.root, self.inventory)
        observations = read_json(self.observations)
        observations['features'][2]['properties']['sample_id'] = 'other1__second'
        write_json(self.observations, observations)
        document['samples']['other1__second'] = document['samples']['other1__view']
        write_json(self.inventory, document)
        with self.assertRaisesRegex(ValueError, 'one view per physical patch'):
            build_inventory(self.observations, self.upstream, self.root, self.inventory)

    def test_rebuild_reaches_local_point_surface_veto_without_research_caches(self):
        output = self.root / 'result'
        with redirect_stdout(StringIO()):
            run(self.parent, self.observations, self.root / 'absent-upstream', self.root, output, self.inventory)
        report = read_json(output / 'changes.json')
        local = report['local_evidence']
        self.assertGreater(local['scoped_cells'], 0)
        self.assertGreater(local['query_count'], 0)
        self.assertEqual(local['source_surface_cells'], local['scoped_cells'])
        self.assertEqual(local['accepted_cells'], 0)
        original = read_json(self.parent / 'global_building_footprints.geojson')['features'][0]
        final = read_json(output / 'global_building_footprints.geojson')['features'][0]
        self.assertTrue(shape(final['geometry']).equals(shape(original['geometry'])))
        self.assertEqual(final['properties'], original['properties'])
        manifest = read_json(output / 'manifest.json')
        self.assertEqual(manifest['status'], 'complete')
        self.assertTrue(manifest['evidence_recomputed_from_original_predictions'])
        self.assertTrue(manifest['source_unchanged'])
        self.assertEqual(manifest['physical_view_count'], 3)


if __name__ == '__main__':
    unittest.main()
