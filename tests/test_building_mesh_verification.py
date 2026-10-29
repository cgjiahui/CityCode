"""Exercise shipped export, movable manifests and independent mesh acceptance."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon, box, mapping

from city_pipeline.buildings.observations import extrude_geometry
from city_pipeline.verify_buildings import cap_audit, source_paths, verify_run, world_mesh


class MeshAcceptance(unittest.TestCase):
    def mesh(self, geometry):
        return extrude_geometry(geometry, 0., 10., [50, 60, 70], {})

    def test_filled_tiny_hole_is_rejected_despite_global_area_tolerance(self):
        footprint = Polygon(box(0, 0, 10, 10).exterior, [box(2, 2, 2.01, 2.01).exterior])
        result = cap_audit(self.mesh(box(0, 0, 10, 10)), footprint, 0., 10.)
        self.assertFalse(result['pass'])
        self.assertFalse(result['checks']['top_cap_matches_geometry'])

    def test_missing_tiny_component_is_rejected_despite_global_area_tolerance(self):
        footprint = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 20.01, .01)])
        result = cap_audit(self.mesh(box(0, 0, 10, 10)), footprint, 0., 10.)
        self.assertFalse(result['pass'])
        self.assertFalse(result['checks']['bottom_cap_matches_geometry'])

    def test_sub_float32_hole_is_reported_under_cap_error_bound(self):
        outer = box(500, 500, 510, 510)
        footprint = Polygon(outer.exterior, [box(502, 502, 502.000001, 502.000001).exterior])
        result = cap_audit(self.mesh(outer), footprint, 0., 10.)
        self.assertTrue(result['pass'])
        self.assertEqual(result['holes_below_xy_resolution'], 1)
        self.assertGreater(result['caps']['top']['filled_area_per_hole_m2'][0], 0)

    def test_scene_graph_transform_and_geometry_alias_are_applied(self):
        scene = trimesh.Scene()
        transform = np.eye(4)
        transform[:3, 3] = [100., -200., 5.]
        scene.add_geometry(self.mesh(box(0, 0, 10, 10)), node_name='building__body',
                           geom_name='shared_mesh', transform=transform)
        result = cap_audit(world_mesh(scene, 'building__body'), box(100, -200, 110, -190), 5., 15.)
        self.assertTrue(result['pass'])

    def test_relative_hash_requires_explicit_base_and_absolute_is_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absolute = str(root / 'source.json')
            self.assertEqual(source_paths({'source_sha256': {absolute: 'hash'}}, root),
                             {Path(absolute): 'hash'})
            with self.assertRaisesRegex(ValueError, 'manifest_directory'):
                source_paths({'source_sha256': {'source.json': 'hash'}}, root)

    def test_export_and_verify_after_moving_entire_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'original'
            source = run / 'work' / '11_visibility'
            source.mkdir(parents=True)
            courtyard = Polygon(box(0, 0, 10, 10).exterior, [box(2, 2, 8, 8).exterior])
            footprint = MultiPolygon([courtyard, box(20, 0, 20.1, .1)])
            properties = {
                'global_building_index': 0, 'predicted_global_building_id': 'building0',
                'canonical_observation_id': 'sample0', 'base_z_m': 0.,
                'body_top_z_m': 10., 'top_z_m': 12., 'height_m': 12.,
                'roof_layers': [
                    {'geometry': mapping(box(.1, .1, .3, .3)), 'top_z_m': 12.},
                    # This geometry only touches the footprint: no positive cap.
                    {'geometry': mapping(box(-1, 0, 0, 1)), 'top_z_m': 11.},
                ],
            }
            (source / 'manifest.json').write_text(json.dumps({
                'status': 'complete', 'prediction_contract': {'uses_ground_truth': False}, 'parameters': {}}))
            (source / 'global_building_footprints.geojson').write_text(json.dumps({
                'type': 'FeatureCollection', 'features': [
                    {'type': 'Feature', 'geometry': mapping(footprint), 'properties': properties}]}))
            environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
                           'CITY_PIPELINE_DATA_ROOT': str(root / 'separate_cache')}
            subprocess.run([sys.executable, '-m', 'city_pipeline.buildings.export',
                            '--trial-root', str(source), '--output-root', str(run / 'release')],
                           env=environment, check=True, capture_output=True, text=True)
            (run / 'RUN_STATUS.json').write_text(json.dumps({'status': 'complete'}))
            destination = root / 'moved' / 'bundle'
            destination.parent.mkdir()
            shutil.move(str(run), destination)
            manifest = json.loads((destination / 'release' / 'manifest.json').read_text())
            self.assertEqual(manifest['source_sha256_path_resolution'], 'manifest_directory')
            self.assertTrue(all(not Path(p).is_absolute() for p in manifest['source_sha256']))
            self.assertEqual(manifest['source_trial'], '../work/11_visibility')
            self.assertTrue(all(not Path(p).is_absolute() for p in manifest['runtime_read_audit']))
            result = verify_run(destination)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['roof_pieces'], 1)
            self.assertEqual(result['closed_meshes'], 2)
            body = result['mesh_piece_results'][0]
            self.assertEqual(body['reference_component_count'], 2)
            self.assertEqual(body['reference_hole_count'], 1)
            # Moving meshes and hashes is not enough if original sources mutate.
            (destination / 'work' / '11_visibility' / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'source SHA changed'):
                verify_run(destination)


if __name__ == '__main__':
    unittest.main()
