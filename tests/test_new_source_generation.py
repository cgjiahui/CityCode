"""New inputs must reach generation, preserve image identities and validate every LOD."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from citycode_assets import batch, batch_gpu
from citycode_assets.appearance import complete_building, sha256_file, write_json
from citycode_assets.batch_delivery import prepare_nonbuilding_sources
from citycode_assets.lod import generation_lod_targets, inspect_glb, LOD_SPECS
from citycode_assets.new_inputs import initialize_generation, import_completed_images

import test_unified_batch as batch_fixtures
import test_asset_delivery_contract as asset_fixtures


def prepare_fixture(task):
    building, folder, _, _ = task
    folder = Path(folder)
    record = {'building_id': batch.read(building)['building_id'], 'model': 'fixture',
        'quality': 'fixture', 'size': '1024x1024', 'prompt_sha256': 'a'*64,
        'inputs': [], 'input_identity_sha256': 'b'*64,
        'provenance': {'building_sha256': sha256_file(building)}}
    write_json(folder/'appearance_input.json', record)
    return {'building_id': record['building_id'], 'cached_conditioning': False}


class NewSourceGeneration(unittest.TestCase):
    def setUp(self):
        self.fixture = batch_fixtures.UnifiedBuildingBatch()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project = self.fixture.project

    def initialize(self, *, image=True):
        directory = self.project/'my_completed_images'
        directory.mkdir()
        if image:
            Image.new('RGB', (1024, 1024), 'gray').save(directory/'A.png')
        self.config_path = self.project/'my_generation.json'
        result = initialize_generation(self.config_path, self.fixture.config['input_manifest'],
            self.project/'data/my_generation', project_root=self.project,
            completed_images=directory if image else None)
        value = batch.read(self.config_path)
        value.update(provider_config='provider.json', runtime_config='runtime.json')
        write_json(self.config_path, value)
        return result, batch.load_config(self.config_path)

    def prepare(self, config):
        with patch.object(batch, 'ProcessPoolExecutor', batch_fixtures.ThreadPoolExecutor), \
                patch.object(batch, '_prepare_one', prepare_fixture):
            return batch.prepare(config, workers=1)

    def shared_image_config(self):
        image = self.project/'shared.png'
        Image.new('RGB', (1024, 1024), 'gray').save(image)
        index = self.project/'shared.images.json'
        members = [{'building_id': sid, 'sha256': sha256_file(self.fixture.source/'buildings'/sid/'building.json')}
                   for sid in ['A', 'B']]
        write_json(index, {'schema': 'citycode.external_completed_images.v1', 'records': [{
            'building_id': 'A', 'source_buildings': members, 'image': image.name, 'sha256': sha256_file(image)}]})
        return {**self.fixture.config, 'completed_images_manifest': str(index)}

    def test_own_images_import_without_API_or_automatic_approval(self):
        result, config = self.initialize()
        self.assertEqual(result['lod_targets'], [1000000, 250000, 100000])
        self.assertEqual(config['execution_mode'], 'generate')
        self.assertNotIn('asset_cache', config)
        with patch('citycode_assets.provider.complete_native', side_effect=AssertionError('No API expected')):
            report = self.prepare(config)
            self.assertEqual(report['counts']['image_complete'], 1)
            root = Path(config['output'])
            receipt = complete_building(root/'appearance/A')
        self.assertEqual(receipt['origin'], 'external_completed_image')
        self.assertEqual(receipt['API_calls'], 0)
        self.assertTrue(receipt['image_review_required'])
        self.assertFalse((root/'reviews/appearance.json').exists())
        self.assertEqual(prepare_nonbuilding_sources(config), [])
        from citycode_assets.batch_review import review_appearance
        review_appearance(config)
        self.assertTrue(all(r['decision'] is None for r in batch.read(root/'reviews/appearance.template.json')['records']))

    def test_shared_image_binds_both_original_sources_and_the_actual_derived_building(self):
        config = self.shared_image_config()
        with patch('citycode_assets.provider.complete_native', side_effect=AssertionError('No API expected')):
            report = self.prepare(config)
            self.assertEqual(report['counts']['image_complete'], 1)
            root = Path(config['output'])
            plan = batch.read(root/'plan.json')
            self.assertEqual(plan['physical_generations'], 1)
            row = plan['records'][0]
            receipt = complete_building(root/row['appearance_directory'])
        self.assertEqual(receipt['source_building_sha256'], row['building']['sha256'])
        self.assertEqual(receipt['source_binding_scope'], 'shared_source_members')
        self.assertEqual(receipt['source_buildings'], batch.read(config['completed_images_manifest'])['records'][0]['source_buildings'])
        self.assertNotEqual(receipt['source_building_sha256'], receipt['source_buildings'][0]['sha256'])
        self.assertEqual(receipt['API_calls'], 0)
        self.assertTrue(receipt['image_review_required'])
        self.assertFalse((root/'reviews/appearance.json').exists())

    def test_shared_image_rejects_missing_duplicate_changed_or_ambiguous_members(self):
        config = self.shared_image_config()
        self.prepare(config)
        plan = batch.read(Path(config['output'])/'plan.json')
        index = Path(config['completed_images_manifest'])
        original = batch.read(index)
        cases = ['missing', 'duplicate', 'changed_hash', 'single_and_group_binding']
        for case in cases:
            with self.subTest(case=case):
                changed = copy.deepcopy(original)
                row = changed['records'][0]
                if case == 'missing':
                    row['source_buildings'].pop()
                elif case == 'duplicate':
                    row['source_buildings'][1] = copy.deepcopy(row['source_buildings'][0])
                elif case == 'changed_hash':
                    row['source_buildings'][1]['sha256'] = 'f'*64
                else:
                    row['building_sha256'] = plan['records'][0]['building']['sha256']
                write_json(index, changed)
                with self.assertRaises(ValueError):
                    import_completed_images(config, plan)

    def test_shared_image_rejects_changed_actual_member_or_derived_group(self):
        config = self.shared_image_config()
        self.prepare(config)
        root = Path(config['output'])
        plan = batch.read(root/'plan.json')
        source = self.fixture.source/'buildings/B/building.json'
        before = source.read_bytes()
        source.write_bytes(before+b' ')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            import_completed_images(config, plan)
        source.write_bytes(before)
        derived = root/plan['records'][0]['building']['path']
        value = batch.read(derived)
        value['source_group']['members'] = ['A']
        write_json(derived, value)
        plan['records'][0]['building'] = batch.proof(derived, root)
        with self.assertRaisesRegex(ValueError, 'members differ'):
            import_completed_images(config, plan)

    def test_member_binding_is_not_a_way_to_bypass_single_source_or_image_hash_checks(self):
        _, config = self.initialize()
        index = Path(config['completed_images_manifest'])
        value = batch.read(index)
        row = value['records'][0]
        row['source_buildings'] = [{'building_id': row['building_id'], 'sha256': row.pop('building_sha256')}]
        write_json(index, value)
        with self.assertRaisesRegex(ValueError, 'requires one explicit shared group'):
            self.prepare(config)

    def test_shared_image_bytes_are_still_verified(self):
        config = self.shared_image_config()
        self.prepare(config)
        plan = batch.read(Path(config['output'])/'plan.json')
        Image.new('RGB', (1024, 1024), 'red').save(self.project/'shared.png')
        with self.assertRaisesRegex(ValueError, 'External completed image changed'):
            import_completed_images(config, plan)

    def test_new_sources_can_record_an_exclusion_without_example_fragment_packages(self):
        _, config = self.initialize(image=False)
        rules = batch.read(config['dispositions'])
        rules['sources'][-1].update(action='record_only', exclusion_status='excluded_incomplete', reason='Reviewed insufficient observation')
        write_json(config['dispositions'], rules)
        records = prepare_nonbuilding_sources(config)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'excluded_incomplete')
        self.assertFalse(records[0]['evidence']['reference_assisted_qualification'])

    def test_changed_external_image_or_source_identity_is_rejected(self):
        _, config = self.initialize()
        index = Path(config['completed_images_manifest'])
        data = batch.read(index)
        data['records'][0]['building_sha256'] = 'f'*64
        write_json(index, data)
        with self.assertRaisesRegex(ValueError, 'different building bytes'):
            self.prepare(config)
        self.assertEqual(batch.read(Path(config['output'])/'PREPARATION.json')['status'], 'importing_external_images')
        data['records'][0]['building_sha256'] = sha256_file(self.fixture.source/'buildings/A/building.json')
        write_json(index, data)
        Image.new('RGB', (1024, 1024), 'red').save(self.project/'my_completed_images/A.png')
        # The image-index contract changed, so this needs a new run before it
        # can even reach the image hash check.
        with self.assertRaisesRegex(ValueError, 'Run inputs changed'):
            self.prepare(config)

    def test_generate_mode_refuses_provided_cache_and_frozen_massing_recipe(self):
        _, config = self.initialize(image=False)
        value = batch.read(self.config_path)
        for key, setting in [('asset_cache', 'data/provided_assets'), ('massing_completion_recipe', 'example.json')]:
            changed = {**value, key: setting}
            write_json(self.config_path, changed)
            with self.assertRaisesRegex(ValueError, 'cannot inherit'):
                batch.load_config(self.config_path)
        write_json(self.config_path, {**value, 'execution_mode': 'typo'})
        with self.assertRaisesRegex(ValueError, 'execution_mode'):
            batch.load_config(self.config_path)

    def test_external_image_content_changes_are_caught_before_import_and_before_reuse(self):
        _, config = self.initialize()
        source = self.project/'my_completed_images/A.png'
        original = source.read_bytes()
        Image.new('RGB', (1024, 1024), 'red').save(source)
        with self.assertRaisesRegex(ValueError, 'External completed image changed'):
            self.prepare(config)
        source.write_bytes(original)
        self.prepare(config)
        imported = Path(config['output'])/'appearance/A/generated_main_oblique.png'
        Image.new('RGB', (1024, 1024), 'red').save(imported)
        with patch('citycode_assets.provider.complete_native', side_effect=AssertionError('No API expected')):
            with self.assertRaisesRegex(ValueError, 'Imported completed image changed'):
                complete_building(imported.parent)

    def test_lod_changes_invalidate_a_frozen_generation_plan(self):
        self.fixture.completed_review()
        config = self.fixture.config
        config['lod_targets'] = [1000000, 250000, 100000]
        with self.assertRaisesRegex(ValueError, 'Run inputs changed'):
            batch.freeze(config)
        for targets in [[1000000], [250000, 100000, 100000], [250000, 100000, 42]]:
            with self.assertRaises(ValueError):
                generation_lod_targets({'lod_targets': targets})

    def test_export_passes_three_budgets_and_refuses_a_two_lod_resume(self):
        root = Path(self.fixture.config['output'])
        config = {**self.fixture.config, 'lod_targets': [1000000, 250000, 100000]}
        raw = root/'generation/A/raw_mesh.pt'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'fixture decoded mesh')
        write_json(raw.parent/'raw_manifest.json', {'status': 'complete', 'sha256': sha256_file(raw)})
        def bake(raw_path, output, *, targets, texture_size):
            self.assertEqual(targets, (1000000, 250000, 100000))
            records = {}
            for target in targets:
                label, name = LOD_SPECS[target]
                path = output/name
                asset_fixtures.cube_glb(path)
                records[label] = {**inspect_glb(path), 'path': str(path), 'target_faces': target}
            return {'status': 'completed', 'raw_cache_sha256': sha256_file(raw_path), 'lods': records}
        with patch.object(batch_gpu, 'frozen_selection'), \
                patch.object(batch_gpu, 'runtime_config', return_value={'trellis_root': '/unused'}), \
                patch('citycode_assets.generation.configure_offline_runtime'), \
                patch('citycode_assets.lod.export_decoded_lods', side_effect=bake) as export:
            self.assertEqual(batch_gpu.export_one(config, 'A'), 0)
            self.assertEqual(batch_gpu.export_one(config, 'A'), 0)
            self.assertEqual(export.call_count, 1)
            config['lod_targets'] = [250000, 100000]
            with self.assertRaisesRegex(ValueError, 'different LOD targets'):
                batch_gpu.export_one(config, 'A')

    def test_three_lods_survive_review_and_final_binding_as_one_shared_asset(self):
        import copy
        import numpy as np
        from citycode_assets.batch_delivery import finalize_batch, source_row
        from citycode_assets.cli import validate_assets
        from citycode_assets.scene_completion import placement_identity

        config = self.fixture.config
        config['lod_targets'] = [1000000, 250000, 100000]
        root, _ = self.fixture.completed_review()
        batch.freeze(config)
        planned = batch.read(root/'FROZEN.json')['records'][0]
        building = batch.read(root/planned['building']['path'])
        matrix = np.array([[4,0,0,0],[0,0,-2,1],[0,4,0,0],[0,0,0,1]], dtype=float)
        row = source_row(self.fixture.source/'buildings/A/building.json', root)
        row.update(status='accepted_generated', asset_instance_id='A_B', local_coordinate_system='glTF Y-up',
            world_from_local_4x4=matrix.tolist(), world_from_building_local_4x4=np.eye(4).tolist(),
            building_local_from_asset_4x4=matrix.tolist(), normal_transform_world_3x3=np.linalg.inv(matrix[:3,:3]).T.tolist(),
            target_geometry=building['footprint_world'], lods={})
        for target in config['lod_targets']:
            label, name = LOD_SPECS[target]
            path = root/'assets/A_B'/name
            asset_fixtures.cube_glb(path)
            row['lods'][label] = {**inspect_glb(path), 'path': str(path.relative_to(root)), 'target_faces': target}
        candidate = root/'validation/A_B/candidate_record.json'
        write_json(candidate, row)
        write_json(candidate.parent/'RESULT.json', {'status':'numeric_pass_model_review_pending',
            'candidate_record':batch.proof(candidate, root)})
        write_json(root/'reviews/models.json', {'records':[{'asset_id':'A_B', 'decision':'include',
            'reason':'Current three LODs and shared source placement checked', 'placement_identity':placement_identity(row)}]})
        partials = []
        for sid in ['fragment', 'record_only']:
            partial = source_row(self.fixture.source/'buildings'/sid/'building.json', root)
            partial.update(status='excluded_incomplete', reason='Insufficient evidence in this fixture', evidence={'fixture':True})
            partials.append(partial)
        with patch('citycode_assets.batch_delivery.prepare_nonbuilding_sources', return_value=partials):
            report = finalize_batch(config)
        self.assertTrue(report['complete'], report)
        self.assertEqual(report['physical_assets'], 1)
        result = validate_assets(root/'binding_manifest.json', require_complete=True, geometry=True)
        self.assertEqual(result['status'], 'passed', result['errors'])
        manifest = batch.read(root/'binding_manifest.json')
        self.assertEqual(set(manifest['asset_instances'][0]['lods']), {'1M','0.25M','0.1M'})


class ThreeLODGeometry(unittest.TestCase):
    def setUp(self):
        self.fixture = asset_fixtures.AssetDeliveryContract()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def add_master(self, *, offset=0.):
        path = self.fixture.root/'assets/current_A/model_1m.glb'
        asset_fixtures.cube_glb(path, offset_x=offset)
        self.fixture.record['lods']['1M'] = {**inspect_glb(path), 'path': path.relative_to(self.fixture.root).as_posix(),
                                           'target_faces': 1000000}

    def test_all_three_lods_are_checked_and_optional_1M_is_not_counted_as_an_extra_asset(self):
        self.add_master()
        result = self.fixture.validate(geometry=True)
        self.assertEqual(result['status'], 'passed', result['errors'])
        self.assertEqual(result['validated_assets'], 1)
        self.assertEqual(set(result['records'][0]['lods']), {'0.25M', '0.1M', '1M'})
        self.add_master(offset=.4)
        failed = self.fixture.validate(geometry=True)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('1M mesh footprint fails', failed['errors'][0]['message'])

    def test_master_coordinate_drift_is_checked_even_when_footprint_coverage_passes(self):
        self.add_master(offset=.03)
        failed = self.fixture.validate(geometry=True)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('local frame', failed['errors'][0]['message'])


if __name__ == '__main__':
    unittest.main()
