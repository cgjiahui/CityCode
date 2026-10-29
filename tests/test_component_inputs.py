"""Component reruns must retain their named meshes, input identities and prior scope."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from shapely.geometry import box, mapping, shape
import trimesh

from citycode_assets.appearance import load_mesh, prepare_building, sha256_file, write_json
from citycode_assets.batch import proof, read
from citycode_assets.component_inputs import build_component_inputs, component_world_mesh


class ComponentInputs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.bundle = self.project/'data/bundle'
        baseline = self.project/'data/baseline'
        baseline.mkdir(parents=True)
        source = self.project/'data/structure/building.json'
        self.support = box(100, 200, 103.5, 202)
        write_json(source, {'building_id': 'P', 'footprint_world': mapping(self.support),
                           'roof_layers': [{'base_z_m': 5, 'top_z_m': 99}], 'observations': []})
        self.scene = trimesh.Scene(base_frame='world')
        for index, height in enumerate([4, 6]):
            mesh = trimesh.creation.box([2, 2, height])
            mesh.apply_translation([1+2*index, 1, height/2])
            self.scene.graph.update(frame_to=f'OLD{index}', matrix=np.eye(4))
            self.scene.add_geometry(mesh, node_name=f'geometry{index}', geom_name=f'body{index}',
                                    parent_node_name=f'OLD{index}')
        model = baseline/'composite.glb'
        model.write_bytes(self.scene.export(file_type='glb'))
        world = np.eye(4)
        world[:3, 3] = [100, 200, 3]
        self.world = world
        self.baseline = baseline/'binding_manifest.json'
        write_json(self.baseline, {'asset_instances': [{'id': 'P', 'status': 'accepted_composite',
            'source_building_ids': ['P'], 'world_from_local_4x4': world.tolist(),
            'lods': {'0.1M': proof(model, baseline)}}],
            'buildings': [{'building_id': 'P', 'structure_path': '../structure/building.json'}]})
        self.inventory, self.references = {'items': []}, []
        for index in range(2):
            old_id = f'OLD{index}'
            image = self.bundle/'units'/old_id/'completed.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (1024, 1024), (index*20, 20, 30)).save(image)
            self.inventory['items'].append({'route': 'component', 'asset_id': 'P',
                'current_source_ids': ['P'], 'old_asset_id': old_id, 'image_sha256': sha256_file(image)})
            reference = {'old_asset_id': old_id, 'current_asset_id': 'P',
                'observation_id': f'original{index}', 'source_camera_crop': {'K_crop_native_resolution': np.eye(3).tolist()},
                'reference_inputs': {}}
            for name in ['source_primary', 'roof_evidence', 'context']:
                path = self.project/'references'/old_id/(name+'.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB' if name == 'context' else 'RGBA', (32, 32), 'white').save(path)
                reference['reference_inputs'][name+'.png'] = {'path': str(path), 'sha256': sha256_file(path)}
            self.references.append(reference)

    def build(self, **kwargs):
        return build_component_inputs(self.project, self.bundle, self.baseline,
            kwargs.get('inventory', self.inventory), reference_records=kwargs.get('references', self.references))

    def test_named_component_extraction_preserves_hierarchy_and_world_pose(self):
        vertices, faces, nodes = component_world_mesh(self.scene, 'OLD1', self.world)
        np.testing.assert_allclose(vertices.min(0), [102, 200, 3])
        np.testing.assert_allclose(vertices.max(0), [104, 202, 9])
        self.assertEqual(len(faces), 12)
        self.assertEqual(nodes, ['geometry1'])
        with self.assertRaisesRegex(ValueError, 'container is missing'):
            component_world_mesh(self.scene, 'MISSING', self.world)

    def test_asset_derived_coarse_inputs_are_readable_without_inventing_new_observations(self):
        original = {path: sha256_file(path) for path in self.project.rglob('*') if path.is_file()}
        result = self.build()
        self.assertEqual(result['component_count'], 2)
        self.assertEqual(result['group_count'], 1)
        manifest = read(self.bundle/'building_generation_inputs/manifest.json')
        self.assertEqual([r['building_id'] for r in manifest['records']], ['P__C00', 'P__C01'])
        for row in manifest['records']:
            path = self.bundle/'building_generation_inputs'/row['path']/'building.json'
            building = read(path)
            footprint = shape(building['footprint_world'])
            self.assertTrue(footprint.is_valid)
            self.assertLess(footprint.difference(self.support).area, 1e-8)
            self.assertEqual(building['roof_layers'], [])
            self.assertEqual(building['derived_prior']['kind'], 'asset_derived_component_prior')
            self.assertFalse(building['derived_prior']['new_observation_extraction'])
            proxy = load_mesh(path.parent/building['paths']['proxy_local_ue_z_up'])
            self.assertAlmostEqual(proxy.bounds[0, 2], 0, places=5)
            self.assertAlmostEqual(proxy.bounds[1, 2], building['height_m'], places=5)
            self.assertAlmostEqual(abs(proxy.volume), footprint.area*building['height_m'], places=4)
            obs = read(path.parent/building['observations'][0]['path']/'observation.json')
            self.assertEqual(obs['evidence_scope'], 'saved_reference_input')
            self.assertFalse(obs['component_segmentation_observed'])
            self.assertEqual(obs['building_id'], row['building_id'])
            prepared = prepare_building(path, self.project/'prepared'/row['building_id'])
            self.assertEqual(len(prepared['inputs']), 4)
            self.assertFalse((self.project/'prepared'/row['building_id']/'completion.json').exists())
        self.assertTrue(all(sha256_file(path) == digest for path, digest in original.items()))
        groups = read(self.bundle/'component_groups.json')
        self.assertEqual(groups['groups'][0]['task_ids'], ['P__C00', 'P__C01'])
        self.assertFalse(groups['automatic_assembly_approval'])
        images = read(self.bundle/'components.images.json')
        self.assertEqual(len(images['records']), 2)

    def test_resume_verifies_existing_files_and_does_not_overwrite_tampering(self):
        self.build()
        files = {path: sha256_file(path) for path in self.bundle.rglob('*') if path.is_file()}
        self.build()
        self.assertTrue(all(sha256_file(path) == digest for path, digest in files.items()))
        path = self.bundle/'building_generation_inputs/buildings/P__C00/geometry/footprints_world.geojson'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'Existing component input differs'):
            self.build()

    def test_changed_images_unknown_containers_and_external_paths_are_rejected(self):
        changed = copy.deepcopy(self.inventory)
        changed['items'][0]['image_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'Completed component image differs'):
            self.build(inventory=changed)
        changed = copy.deepcopy(self.inventory)
        changed['items'][0]['old_asset_id'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'Unsafe component'):
            self.build(inventory=changed)
        with self.assertRaisesRegex(ValueError, 'inside the release data directory'):
            build_component_inputs(self.project, self.project/'outside', self.baseline, self.inventory,
                                   reference_records=self.references)
        refs = copy.deepcopy(self.references)
        refs[0]['reference_inputs']['source_primary.png']['path'] = '/etc/hosts'
        with self.assertRaisesRegex(ValueError, 'inside the selected project'):
            self.build(references=refs)


if __name__ == '__main__':
    unittest.main()
