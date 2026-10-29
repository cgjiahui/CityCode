"""Published decisions must reproduce assembly without altering source/assets."""
import copy
import hashlib
import json
import shutil
import unittest

from shapely.geometry import box, mapping
import test_asset_delivery_contract as fixtures
from citycode_assets.completion_replay import replay_scene_completion, SCHEMA
from citycode_assets.scene_completion import digest


class CompletionReplayContract(unittest.TestCase):
    def setUp(self):
        f = fixtures.AssetDeliveryContract(); f.setUp(); self.addCleanup(f.doCleanups)
        self.root = f.root
        after = []
        before = []
        for name, offset in [('current_A', 0), ('current_B', 3)]:
            r = copy.deepcopy(f.record); r.update(building_id=name, asset_instance_id=name,
                status='accepted_generated', reason='Reviewed observed building', structure_path=name+'.json')
            r['world_from_local_4x4'][0][3] = offset
            r['building_local_from_asset_4x4'][0][3] = offset
            r['target_geometry'] = mapping(box(offset, -1, offset+1, 0))
            source = self.root / r['structure_path']
            source.write_text(json.dumps({'building_id': name, 'footprint_world': r['target_geometry']}))
            after.append(r)
            pending = copy.deepcopy(r); pending['retained_candidate_lods'] = pending.pop('lods'); pending['status'] = 'pending_review'
            before.append(pending)
        self.before = copy.deepcopy(f.manifest)
        self.before.update(buildings=before, asset_instances=[], building_entities=[], surface_instances=[],
                           status='in_progress', fine_generation_complete=False, counts={'pending_review': 2})
        self.before['source_structure'].update(instance_count=2, building_ids=['current_A', 'current_B'])
        self.input = self.root/'input.json'; self.input.write_text(json.dumps(self.before))
        evidence = self.root/'evidence.json'; evidence.write_text('{"reviewed": true}')
        self.group = {'id': 'GROUP_A_B', 'asset_instance_ids': ['current_A', 'current_B'],
                      'source_building_ids': ['current_A', 'current_B']}
        self.expected = copy.deepcopy(self.before)
        self.expected.update(buildings=copy.deepcopy(after), status='complete', fine_generation_complete=True,
            counts={'accepted_generated': 2}, updated_utc='2026-10-18T00:00:00+00:00', building_entities=[self.group],
            asset_instances=[{'id': r['building_id'], 'source_building_ids': [r['building_id']], 'status': r['status'],
                              'world_from_local_4x4': r['world_from_local_4x4'], 'lods': r['lods'],
                              'building_entity_id': 'GROUP_A_B'} for r in after])
        self.decisions = {'schema': SCHEMA, 'input_manifest': self.spec(self.input),
            'review_provenance': {'decision_maker': 'Agent-assisted test review', 'reference_data_used_for_review': False},
            'records': [{'building_id': r['building_id'], 'action': 'retain_existing_model',
                         'input_record_digest': digest(pending), 'source_building_sha256': self.spec(self.root/r['structure_path'])['sha256'],
                         'reason': 'Keep reviewed model', 'evidence_files': [self.spec(evidence)], 'building_record': r}
                        for r, pending in zip(after, before)],
            'building_entity_groups': [self.group], 'output_metadata': {'updated_utc': self.expected['updated_utc']}}
        self.review = self.root/'decisions.json'; self.output = self.root/'replayed.json'

    def spec(self, path):
        return {'path': str(path.relative_to(self.root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def apply(self):
        expected = (json.dumps(self.expected, ensure_ascii=False, indent=2)+'\n').encode()
        self.decisions.setdefault('expected_output_manifest_sha256', hashlib.sha256(expected).hexdigest())
        self.review.write_text(json.dumps(self.decisions))
        return replay_scene_completion(self.root, self.review, self.output)

    def test_grouped_replay_is_exact_and_preserves_inputs(self):
        source = self.input.read_bytes()
        models = {p: p.read_bytes() for p in self.root.rglob('*.glb')}
        result = self.apply()
        self.assertEqual(self.output.read_bytes(), (json.dumps(self.expected, ensure_ascii=False, indent=2)+'\n').encode())
        self.assertEqual(result['physical_asset_instances'], 2)
        self.assertEqual(result['building_entities'], 1)
        self.assertFalse(result['reference_data_read_during_replay'])
        self.assertEqual((result['API_calls'], result['GPU_jobs']), (0, 0))
        self.assertEqual(self.input.read_bytes(), source)
        self.assertTrue(all(p.read_bytes() == data for p, data in models.items()))

    def test_stale_input_is_rejected_without_output(self):
        self.input.write_text(self.input.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'): self.apply()
        self.assertFalse(self.output.exists())

    def test_replay_survives_package_relocation(self):
        self.apply()
        relocated = self.root.parent/'relocated_assets'
        shutil.copytree(self.root, relocated)
        output = relocated/'second_replay.json'
        result = replay_scene_completion(relocated, relocated/'decisions.json', output)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(output.read_bytes(), self.output.read_bytes())

    def test_record_change_is_not_silently_applied(self):
        self.decisions['records'][0]['input_record_digest'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source record changed'): self.apply()

    def test_source_geometry_cannot_be_replaced(self):
        self.decisions['records'][0]['building_record']['target_geometry'] = mapping(box(0,0,10,10))
        with self.assertRaisesRegex(ValueError, 'preserve the original predicted footprint'): self.apply()

    def test_evidence_cannot_escape_package(self):
        self.decisions['records'][0]['evidence_files'][0]['path'] = '../outside.json'
        with self.assertRaisesRegex(ValueError, 'inside the published asset directory'): self.apply()

    def test_retained_placement_cannot_change(self):
        self.decisions['records'][0]['building_record']['world_from_local_4x4'][0][3] += 1
        with self.assertRaisesRegex(ValueError, 'Retained model bytes or placement'): self.apply()

    def test_wrong_expected_result_does_not_write_output(self):
        self.decisions['expected_output_manifest_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'frozen reviewed result'): self.apply()
        self.assertFalse(self.output.exists())

    def cached_generation(self):
        decision = self.decisions['records'][0]; decision['action'] = 'regenerate_from_cached_image'
        image = self.root/'completed_image.png'; image.write_bytes(b'fixture image identity')
        generation = self.root/'raw_manifest.json'
        generation.write_text(json.dumps({'status': 'complete', 'building_id': 'current_A',
            'contract': {'building_sha256': decision['source_building_sha256'], 'image_sha256': self.spec(image)['sha256']}}))
        decision.update(completed_image=self.spec(image), generation_record=self.spec(generation))
        decision['building_record']['provenance'] = {'conditioning_image_sha256': self.spec(image)['sha256'],
                                                   'raw_manifest_sha256': self.spec(generation)['sha256']}
        self.expected['buildings'][0] = copy.deepcopy(decision['building_record'])
        return decision

    def test_cached_generation_binds_source_image_and_model(self):
        self.cached_generation()
        self.assertEqual(self.apply()['status'], 'passed')

    def test_cached_generation_cannot_use_unrelated_image(self):
        decision = self.cached_generation()
        decision['building_record']['provenance']['conditioning_image_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'bound to the reviewed source and image'): self.apply()


if __name__ == '__main__': unittest.main()
