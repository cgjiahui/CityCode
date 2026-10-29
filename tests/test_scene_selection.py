"""Scene inclusion preserves evidence, never duplicates logical buildings."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from citycode_assets.cli import _validate_building_entities, validate_assets
from citycode_assets.scene_selection import apply_scene_review
import test_asset_delivery_contract as fixtures


class SceneReviewContract(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AssetDeliveryContract()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.path = self.fixture.path
        self.fixture.record['asset_instance_id'] = 'current_A'
        self.fixture.manifest['asset_instances'] = [{
            'id': 'current_A', 'source_building_ids': ['current_A'],
            'status': 'accepted_reuse', 'lods': copy.deepcopy(self.fixture.record['lods']),
            'world_from_local_4x4': copy.deepcopy(self.fixture.record['world_from_local_4x4'])}]
        self.path.write_text(json.dumps(self.fixture.manifest))
        self.review = {'schema': 'citycode.scene_asset_review.v1',
            'input_manifest_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
            'baseline_asset_ids': [], 'reference_assisted_review': True,
            'records': [{'building_id': 'current_A', 'action': 'include', 'reason': 'Reviewed observed building',
                         'evidence': [{'observation_id': 'original_A'}]}]}
        self.review_path = self.root/'review.json'
        self.output = self.root/'reviewed.json'

    def apply(self):
        self.review_path.write_text(json.dumps(self.review))
        return apply_scene_review(self.path, self.review_path, self.output)

    def test_record_only_is_not_active_but_preserves_candidate_mesh_and_source(self):
        original = self.path.read_bytes()
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*.glb')}
        self.review['records'][0]['action'] = 'record_only'
        self.apply()
        result = json.loads(self.output.read_text())
        self.assertEqual(result['asset_instances'], [])
        row = result['buildings'][0]
        self.assertEqual(row['status'], 'excluded_existing_building')
        self.assertNotIn('lods', row)
        self.assertEqual(set(row['retained_candidate_lods']), {'0.25M','0.1M'})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p,h in hashes.items()))
        self.assertEqual(validate_assets(self.output, require_complete=True)['status'], 'passed')

    def test_hold_cannot_be_claimed_complete(self):
        self.review['records'][0]['action'] = 'hold'
        self.apply()
        result = validate_assets(self.output)
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['complete'])
        self.assertEqual(result['pending_ids'], ['current_A'])
        self.assertEqual(validate_assets(self.output, require_complete=True)['status'], 'failed')

    def test_include_retains_valid_mesh_parts(self):
        self.apply()
        result = validate_assets(self.output, require_complete=True, geometry=True)
        self.assertEqual(result['status'], 'passed', result['errors'])
        self.assertEqual(result['physical_asset_instances'], 1)
        self.assertEqual(result['building_entities'], 1)

    def test_review_rejects_stale_inputs_and_incomplete_baseline(self):
        self.review['input_manifest_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'exact input'): self.apply()
        self.review['input_manifest_sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.review['records'] = []
        with self.assertRaisesRegex(ValueError, 'partition'): self.apply()

    def test_one_building_can_have_two_different_mesh_parts(self):
        manifest = {'asset_instances': [
            {'id':'part1','source_building_ids':['s1'],'building_entity_id':'building'},
            {'id':'part2','source_building_ids':['s2'],'building_entity_id':'building'}],
            'building_entities':[{'id':'building','asset_instance_ids':['part1','part2'],
                                  'source_building_ids':['s1','s2']}]}
        self.assertEqual(_validate_building_entities(manifest), 1)
        manifest['building_entities'].append({'id':'another','asset_instance_ids':['part2'],'source_building_ids':['s2']})
        with self.assertRaisesRegex(ValueError, 'multiply owned'): _validate_building_entities(manifest)
        manifest['building_entities'].pop()
        manifest['building_entities'][0]['asset_instance_ids']=['part1']
        manifest['building_entities'][0]['source_building_ids']=['s1']
        with self.assertRaisesRegex(ValueError, 'every active'): _validate_building_entities(manifest)

    def test_entity_cannot_reference_an_inactive_candidate(self):
        manifest = {'asset_instances': [],'building_entities':[{'id':'building','asset_instance_ids':['held'],'source_building_ids':['s']}]}
        with self.assertRaisesRegex(ValueError, 'inactive'): _validate_building_entities(manifest)

if __name__ == '__main__': unittest.main()
