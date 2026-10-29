"""Meaningful contracts for bounded, source-only height consensus."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from city_pipeline.buildings.height_consensus import DEFAULT_CONFIG, huber_location, refine_height_consensus


def feature(gid='example', canonical='a', top=10., base=0., roofs=None):
    geometry = {'type': 'Polygon', 'coordinates': [[[0,0], [10,0], [10,10], [0,10], [0,0]]]}
    return {'type': 'Feature', 'geometry': geometry, 'properties': {
        'predicted_global_building_id': gid, 'canonical_observation_id': canonical,
        'top_z_m': top, 'base_z_m': base, 'height_m': top-base, 'roof_layers': roofs or []}}


def observation(oid, patch, top, quality=90., boundary='interior_keep'):
    ob = feature()
    ob['properties'] = {'observation_id': oid, 'patch_id': patch, 'top_z_m': top,
        'preliminary_quality': quality, 'boundary_status': boundary,
        'reliable_core_contains_centroid': True, 'source_edge_distance_px': 10}
    return ob


def stats(observations):
    return {f['properties']['observation_id']: {'q95': f['properties']['top_z_m'],
            'observed_count': 1000} for f in observations}


class HeightConsensusContracts(unittest.TestCase):
    def test_isolated_cli_completes_prediction_hash_and_rejects_source_mismatch(self):
        import hashlib,os,subprocess,sys,tempfile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input';source.mkdir()
            observations=[observation('a','p1',12.),observation('b','p2',14.)]
            obs_path=root/'observations.json';obs_path.write_text(json.dumps({'features':observations}))
            (source/'global_building_footprints.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[feature()]}))
            statistics={'source_observations_sha256':hashlib.sha256(obs_path.read_bytes()).hexdigest(),'observations':stats(observations)}
            stats_path=root/'statistics.json';stats_path.write_text(json.dumps(statistics))
            config=root/'config.json';config.write_text('{}');output=root/'output'
            cmd=[sys.executable,'-m','city_pipeline.buildings.height_consensus','--input-root',str(source),
                 '--observations',str(obs_path),'--point-statistics',str(stats_path),'--config',str(config),'--output-root',str(output)]
            env={**os.environ,'CITY_PIPELINE_DATA_ROOT':str(root)}
            run=subprocess.run(cmd,env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['prediction_sha256'],hashlib.sha256((output/'global_building_footprints.geojson').read_bytes()).hexdigest())
            self.assertNotEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)
            statistics['source_observations_sha256']='different';stats_path.write_text(json.dumps(statistics))
            cmd[-1]=str(root/'mismatched_output')
            self.assertNotEqual(subprocess.run(cmd,env=env,capture_output=True).returncode,0)
            self.assertFalse((root/'mismatched_output/manifest.json').exists())

    def test_single_patch_duplicate_is_not_independent_support(self):
        obs = [observation('a','patchA',20.), observation('b','patchA',40.,quality=80.)]
        source = [feature(top=10.)]
        result, report = refine_height_consensus(source, obs, stats(obs))
        self.assertEqual(result, source)
        self.assertEqual(report['changed_entity_count'], 0)

    def test_huber_rejects_unbounded_influence_of_one_extreme(self):
        self.assertAlmostEqual(huber_location([10.,10.,100.], 4.), 12., places=4)

    def test_bound_and_idempotence_with_severely_biased_prior(self):
        obs = [observation('a','patchA',30.), observation('b','patchB',32.)]
        source = [feature(top=10.)]
        untouched = copy.deepcopy(source)
        result, report = refine_height_consensus(source, obs, stats(obs))
        self.assertEqual(source, untouched)
        self.assertEqual(result[0]['properties']['top_z_m'], 13.)
        again, second_report = refine_height_consensus(result, obs, stats(obs))
        self.assertEqual(again, result)
        self.assertEqual(second_report['changed_entity_count'], 0)

    def test_roof_layer_entity_is_preserved_including_base(self):
        layer = {'top_z_m': 20., 'base_z_m': 10., 'geometry': feature()['geometry']}
        source = [feature(top=20., roofs=[layer])]
        obs = [observation('a','patchA',40.), observation('b','patchB',45.)]
        result, report = refine_height_consensus(source, obs, stats(obs))
        self.assertEqual(result, source)
        self.assertEqual(report['protected_roof_layer_entities'], 1)

    def test_invalid_clipped_and_low_quality_sources_do_not_gain_votes(self):
        obs = [observation('a','patchA',10.), observation('b','patchB',30.,quality=20.),
               observation('c','patchC',30.,boundary='boundary_clipped_exclude'),
               observation('d','patchD',float('nan'))]
        source = [feature(top=10.)]
        result, report = refine_height_consensus(source, obs, stats(obs))
        self.assertEqual(result, source)
        self.assertEqual(report['changed_entity_count'], 0)

    def test_low_point_count_falls_back_to_stable_prediction(self):
        obs = [observation('a','patchA',10.), observation('b','patchB',10.)]
        point_stats = {oid: {'q95': 100., 'observed_count': 10} for oid in ['a','b']}
        source = [feature(top=10.)]
        result, _ = refine_height_consensus(source, obs, point_stats)
        self.assertEqual(result, source)

    def test_height_origin_translation_does_not_change_adjustment(self):
        obs = [observation('a','patchA',14.), observation('b','patchB',16.)]
        result, _ = refine_height_consensus([feature(top=10.)], obs, stats(obs))
        elevated_obs = [observation('a','patchA',114.), observation('b','patchB',116.)]
        elevated, _ = refine_height_consensus([feature(top=110.,base=100.)], elevated_obs, stats(elevated_obs))
        self.assertAlmostEqual(elevated[0]['properties']['top_z_m']-100., result[0]['properties']['top_z_m'])

    def test_minimum_body_cannot_override_change_bound(self):
        obs = [observation('a','patchA',10.), observation('b','patchB',10.)]
        source = [feature(top=.5)]
        result, report = refine_height_consensus(source, obs, stats(obs), {'maximum_adjustment_m':1.})
        self.assertEqual(result, source)
        self.assertEqual(report['infeasible_minimum_height_entities'], 1)

    def test_no_per_object_selector_and_no_boolean_distance(self):
        obs = [observation('a','patchA',10.), observation('b','patchB',10.)]
        for config in [{'object_ids':['example']}, {'maximum_adjustment_m':True}]:
            with self.assertRaises(ValueError):
                refine_height_consensus([feature()], obs, stats(obs), config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
