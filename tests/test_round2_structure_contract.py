"""Global parameter routing, topology protection and direct-roof evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from shapely.geometry import Polygon, box, mapping, shape
from city_pipeline.buildings.parameters import normalize_postprocess, stage_parameters
from city_pipeline.buildings.polish import _apply_geometry, apply_polish, _prediction_path
from city_pipeline.buildings.roof_retention import observed_roof_retention
from citycode_perception.cli import structure_parameters

OP = {'kind': 'bounded_closing', 'radius': 2., 'conditions': {},
      'max_growth': .05, 'max_loss': .01, 'max_drift': .10, 'max_hausdorff_m': 2.}

def feature(geometry, gid='arbitrary', **extra):
    p={'predicted_global_building_id':gid,'consensus_observation_ids':[gid+'-obs'],
       'canonical_observation_id':gid+'-obs','canonical_instance_id':1,
       'canonical_building_point_cloud_npz_path':'cloud.npz',
       'base_z_m':0.,'top_z_m':10.,'height_m':10.}
    p.update(extra)
    return {'type':'Feature','geometry':mapping(geometry),'properties':p}

class GlobalParameters(unittest.TestCase):
    def test_cached_relative_paths_keep_origin_when_absolute_output_boundary_is_wider(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);origin=parent/'inputs';origin.mkdir();output=parent/'generated';output.mkdir()
            self.assertEqual(_prediction_path('points/cloud.npz',origin,parent),origin/'points/cloud.npz')
            self.assertEqual(_prediction_path(str(output/'cloud.npz'),origin,parent),output/'cloud.npz')
            with self.assertRaises(PermissionError):_prediction_path('../generated/cloud.npz',origin,parent)
            with self.assertRaises(PermissionError):_prediction_path(str(output/'cloud.npz'),origin)

    def test_cli_global_overrides_and_single_release_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            import json
            p=Path(folder)/'parameters.json';p.write_text(json.dumps({'postprocess':{'boundary':{'offset_m':1.}}}))
            result=structure_parameters('standard',p)
            self.assertEqual(result['postprocess']['boundary']['offset_m'],1.)
            self.assertEqual(result['height_consensus']['minimum_observation_quality'],40.)
            self.assertEqual(result['postprocess']['replace_repair']['max_shift'],3.)
            self.assertEqual(result['selection']['area_weight'],.5)
            for old in ('clean-v1','clean-v2','none'):
                with self.assertRaises(ValueError):structure_parameters(old)
            p.write_text(json.dumps({'roof_retention':{'retained_ids':['building']}}))
            with self.assertRaises(ValueError):structure_parameters('standard',p)
            p.write_text(json.dumps({'height_consensus':True}))
            with self.assertRaises(ValueError):structure_parameters('standard',p)
            p.write_text(json.dumps({'profile':'none'}))
            with self.assertRaises(ValueError):structure_parameters('standard',p)
            with self.assertRaises(ValueError):structure_parameters('none',p)

    def test_absent_overrides_preserve_legacy_and_do_not_mutate_input(self):
        self.assertEqual(normalize_postprocess(),{})
        self.assertEqual(stage_parameters('boundary')['offset_m'],.75)
        self.assertEqual(stage_parameters('replace_repair')['max_shift'],2.)
        supplied={'boundary':{'offset_m':1.25},'replace_repair':{'max_shift':3.}}
        before=copy.deepcopy(supplied);result=normalize_postprocess(supplied)
        self.assertEqual(supplied,before)
        self.assertEqual(result['boundary']['minimum_component_area_retention'],.85)
        self.assertEqual(result['replace_repair']['scope'],'weak')

    def test_invalid_and_individual_parameters_fail_before_execution(self):
        for supplied in [{'boundary':{'building_ids':['object']}}, {'filter':{'height_le_m':True}},
                         {'boundary':{'offset_m':float('nan')}}, {'local_repair':{'iou':.1}},
                         {'local_repair':{'vote':0}}, {'replace_repair':{'method':'recover'}},
                         {'review':{'path':'ground_truth.json'}}]:
            with self.subTest(supplied=supplied),self.assertRaises(ValueError):
                normalize_postprocess(supplied)

class BoundedTopology(unittest.TestCase):
    def test_courtyard_survives_closing_that_would_fill_it(self):
        g=Polygon(box(0,0,20,20).exterior.coords,[box(9.5,1,10.5,19).exterior.coords])
        result,report=_apply_geometry([feature(g)],OP)
        self.assertTrue(shape(result[0]['geometry']).equals(g))
        self.assertEqual(report['changes'],[])

    def test_conditions_protect_supported_and_out_of_scope_structures(self):
        g=Polygon([(0,0),(20,0),(20,20),(10.5,20),(10.5,19),(9.5,19),(9.5,20),(0,20)])
        for condition in [{'support_max':1},{'area_max':100}]:
            f=feature(g,consensus_observation_ids=['one','two'])
            result,report=_apply_geometry([f],{**OP,'conditions':condition})
            self.assertTrue(shape(result[0]['geometry']).equals(g))
            self.assertEqual(report['changes'],[])
        with self.assertRaises(ValueError):
            apply_polish([feature(g)],{},'.',{'operations':[{**OP,'conditions':{'building_id':'one'}}]})

    def test_actual_small_gap_is_closed_without_changing_height_or_input(self):
        g=Polygon([(0,0),(20,0),(20,20),(10.5,20),(10.5,19),(9.5,19),(9.5,20),(0,20)])
        original=[feature(g)];before=copy.deepcopy(original)
        result,report=_apply_geometry(original,OP)
        self.assertGreater(shape(result[0]['geometry']).area,g.area)
        self.assertEqual(result[0]['properties']['top_z_m'],10.)
        self.assertEqual(original,before)
        self.assertEqual(len(report['changes']),1)

class RawRoofEvidence(unittest.TestCase):
    def scene(self,root,kind,status='boundary_risk_review'):
        fs=[feature(box(0,0,20,10))]
        obs={'arbitrary-obs':{'observation_id':'arbitrary-obs','boundary_status':status,
              'stable_layer_count':1,'building_point_cloud_npz_path':'cloud.npz','rgb_instance_id':1}}
        x,y=np.meshgrid(np.linspace(0,20,30),np.linspace(0,10,30))
        if kind=='vertical':xyz=np.column_stack([x.ravel(),np.zeros(x.size),y.ravel()])
        else:xyz=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,10.)])
        if kind!='missing':np.savez(root/'cloud.npz',xyz_m=xyz.astype(np.float32),instance_id=np.ones(x.size,np.uint16),provenance=np.full(x.size,4 if kind=='filled' else 1,np.uint8))
        return fs,obs

    def test_only_direct_horizontal_stable_roof_retains_boundary_candidate(self):
        for kind in ['horizontal','vertical','filled','missing']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                fs,obs=self.scene(Path(folder),kind);before=copy.deepcopy(fs)
                kept,report=observed_roof_retention(fs,obs,folder)
                self.assertEqual(kept,['arbitrary-obs'] if kind=='horizontal' else [])
                self.assertFalse(report['uses_ground_truth']);self.assertFalse(report['feature_cache_read'])
                self.assertEqual(fs,before)

    def test_clipped_is_opt_in_and_still_requires_roof_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            fs,obs=self.scene(Path(folder),'horizontal','boundary_clipped_exclude')
            self.assertEqual(observed_roof_retention(fs,obs,folder)[0],[])
            self.assertEqual(observed_roof_retention(fs,obs,folder,{'clipped_enabled':True})[0],['arbitrary-obs'])
            fs[0]['properties']['height_m']=40.
            self.assertEqual(observed_roof_retention(fs,obs,folder,{'clipped_enabled':True})[0],[])

    def test_reference_path_and_individual_override_cannot_supply_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            fs,obs=self.scene(Path(folder),'horizontal')
            fs[0]['properties']['canonical_building_point_cloud_npz_path']='global_vectors/cloud.npz'
            self.assertEqual(observed_roof_retention(fs,obs,folder)[0],[])
            with self.assertRaises(ValueError):
                observed_roof_retention(fs,obs,folder,{'retained_ids':['arbitrary-obs']})

if __name__=='__main__':unittest.main()
