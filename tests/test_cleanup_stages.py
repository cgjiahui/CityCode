"""Behavioral contracts for the standard cleanup steps and rejected review input."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from shapely.geometry import MultiPolygon, box, mapping, shape
from city_pipeline.buildings.run import run
from city_pipeline.buildings.stage import apply_boundary_inset, filter_features, parse_args
from city_pipeline.buildings.parameters import stage_parameters


def feature(geometry, gid='building', **properties):
    props = {'predicted_global_building_id': gid, 'consensus_observation_ids': [gid + '-observation'],
             'canonical_boundary_status': 'boundary_risk_review', 'base_z_m': 0.,
             'top_z_m': 10., 'height_m': 10.}
    props.update(properties)
    return {'type': 'Feature', 'properties': props, 'geometry': mapping(geometry)}


class CleanupBehavior(unittest.TestCase):
    def test_inset_keeps_small_component_and_clips_roof_before_recomputing_height(self):
        geometry = MultiPolygon([box(0, 0, 100, 100), box(110, 0, 111, 1)])
        item = feature(geometry, body_top_z_m=8., top_z_m=14., height_m=14., roof_layers=[
            {'geometry': mapping(box(0, 0, .5, .5)), 'top_z_m': 14.},
            {'geometry': mapping(box(10, 10, 20, 20)), 'top_z_m': 11.}])
        result, changes = apply_boundary_inset([item], stage_parameters('boundary', {'offset_m': 1.25}))
        expected = MultiPolygon([box(1.25, 1.25, 98.75, 98.75), box(110, 0, 111, 1)])
        self.assertTrue(shape(result[0]['geometry']).equals(expected))
        self.assertEqual(len(result[0]['properties']['roof_layers']), 1)
        self.assertEqual(result[0]['properties']['top_z_m'], 11.)
        self.assertEqual(result[0]['properties']['height_m'], 11.)
        self.assertEqual(changes, [{'id': 'building', 'action': 'boundary_inset'}])

    def test_well_supported_interior_is_outside_weak_inset_scope(self):
        item = feature(box(0, 0, 100, 100), canonical_boundary_status='interior_keep',
                       consensus_observation_ids=['one', 'two'])
        original = copy.deepcopy(item)
        result, changes = apply_boundary_inset([item], stage_parameters('boundary'))
        self.assertEqual(result, [original])
        self.assertEqual(changes, [])

    def test_prune_retains_supported_roof_but_rejects_unprotected_boundary(self):
        items = [feature(box(0, 0, 10, 10), gid) for gid in ['roof', 'fragment', 'interior']]
        observations = {gid + '-observation': {'boundary_status': status}
                        for gid, status in [('roof', 'boundary_risk_review'),
                                            ('fragment', 'boundary_risk_review'), ('interior', 'interior_keep')]}
        result, changes = filter_features(items, observations, 'prune', stage_parameters('prune'),
                                          {'roof-observation'})
        self.assertEqual([f['properties']['predicted_global_building_id'] for f in result], ['roof', 'interior'])
        self.assertEqual([c['action'] for c in changes], ['retain_with_predicted_roof_support', 'delete'])

    def test_final_filter_keeps_supported_narrow_building_and_rejects_low_height(self):
        items = [feature(box(0, 0, 2, 20), 'narrow', consensus_observation_ids=['a', 'b']),
                 feature(box(0, 0, 20, 20), 'low', height_m=2., consensus_observation_ids=['a'])]
        observations = {'a': {'preliminary_quality': 60.}, 'b': {'preliminary_quality': 60.}}
        result, changes = filter_features(items, observations, 'filter', stage_parameters('filter'))
        self.assertEqual([f['properties']['predicted_global_building_id'] for f in result], ['narrow'])
        self.assertEqual(changes, [{'id': 'low', 'action': 'delete'}])

    def test_legacy_review_configuration_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ['historical-review.json', '']:
                config = root / 'config.json'
                config.write_text(json.dumps({'review_file': value}))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'review configurations'):
                    run(config, root / 'output')
                self.assertFalse((root / 'output').exists())

    def test_removed_review_stage_is_explicitly_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--stage', 'review', '--source', 'source', '--work', 'work'])


if __name__ == '__main__':
    unittest.main()
