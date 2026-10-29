"""Inferred components preserve the baseline and require actual scene evidence."""
import copy
from pathlib import Path
import unittest
from shapely.geometry import box,mapping

from citycode_assets.appearance import write_json,sha256_file
from citycode_assets.batch import proof,read
from citycode_assets.massing_completion import construct_manifest,validate_massing,_measure,write_coverage,prune_recipe,part_review_identity
from citycode_assets.scene_completion import scene_identity
import test_cache_batch


class MassingCompletion(unittest.TestCase):
    def setUp(self):
        fixture=test_cache_batch.FrozenAssetCache('test_restore_is_identical_and_preserves_reuse_without_network_or_GPU')
        fixture.setUp();self.addCleanup(fixture.doCleanups);self.root=fixture.cache
        self.baseline=read(self.root/'binding_manifest.json');self.geom=mapping(box(2,-1,3,0))
        self.baseline['buildings'].append({'building_id':'B','status':'excluded_incomplete','reason':'Partial wing',
            'target_geometry':self.geom})
        self.baseline['source_structure'].update(instance_count=2,building_ids=['A','B'])
        source=self.root.parent/'structure/B.json';write_json(source,{'building_id':'B','footprint_world':self.geom})
        baseline_file=self.root/'massing_completion/input_binding_manifest.json';write_json(baseline_file,self.baseline)
        matrix=copy.deepcopy(self.baseline['asset_instances'][0]['world_from_local_4x4']);matrix[0][3]=2
        part={'id':'MASSING_B_01','source_building_ids':['B'],'donor_asset_id':'A',
            'donor_lods':self.baseline['asset_instances'][0]['lods'],'world_from_local_4x4':matrix,
            'target_geometry':self.geom,'target_base_z_m':0,'target_top_z_m':1}
        evidence=self.root/'massing_completion/review/evidence.txt';evidence.parent.mkdir(parents=True);evidence.write_text('Observed and actual matched-camera context review fixture')
        self.recipe={'baseline_binding_sha256':sha256_file(baseline_file),'parts':[part],
            'source_decisions':[{'source_id':'B','parts':[part['id']],'source_geometry':self.geom,
                'source_building':proof(source,self.root)}],
            'review':{'approved':True,'reviewer':'test-reviewer','baseline_visual_context_proofs':[],
                'proof_files':[{**proof(evidence,self.root),'source_id':'B','role':role} for role in
                    ['observed_source','actual_scene_front','actual_scene_back']]}}
        self.manifest=construct_manifest(self.baseline,self.recipe)
        a=self.baseline['asset_instances'][0]
        self.features=[{'type':'Feature','geometry':mapping(box(0,-1,1,0)),'properties':{'asset_id':'A',
            'sha256':a['lods']['0.25M']['sha256'],'world_from_local_4x4':a['world_from_local_4x4']}}]
        write_json(self.root/'massing_completion/baseline_footprints.geojson',{'type':'FeatureCollection','features':self.features})
        measured=_measure((str(self.root),self.manifest['massing_instances'][0],self.geom,self.features))
        write_json(self.root/'massing_completion/geometry_validation.json',{'status':'passed','records':[measured],'errors':[]})
        self.seal()

    def seal(self):
        path=self.root/'massing_completion/recipe.json';write_json(path,self.recipe)
        self.manifest['massing_completion'].update(recipe=proof(path,self.root),
            input_binding=proof(self.root/'massing_completion/input_binding_manifest.json',self.root),
            geometry_validation=proof(self.root/'massing_completion/geometry_validation.json',self.root),
            baseline_footprints=proof(self.root/'massing_completion/baseline_footprints.geojson',self.root))

    def test_reuse_preserves_baseline_and_does_not_add_an_independent_building(self):
        result=validate_massing(self.root,self.manifest)
        self.assertEqual(result['massing_parts'],1);self.assertEqual(result['independent_buildings_added'],0)
        self.assertEqual(self.baseline['asset_instances'],self.manifest['asset_instances'])
        self.assertEqual(self.baseline['building_entities'],self.manifest['building_entities'])
        self.assertEqual(self.manifest['massing_instances'][0]['lods'],self.baseline['asset_instances'][0]['lods'])
        self.assertNotEqual(scene_identity(self.baseline),scene_identity(self.manifest))

    def test_coverage_measures_the_qualified_patch_without_counting_a_new_building(self):
        summary=write_coverage(self.root,self.manifest,self.recipe)
        self.assertEqual(summary['area_weighted_before'],0)
        self.assertGreater(summary['area_weighted_after'],.65)
        self.assertEqual(summary['part_count'],1)
        self.assertEqual(summary['independent_buildings_added'],0)

    def test_changed_original_pose_or_source_state_is_rejected(self):
        self.manifest['asset_instances'][0]['world_from_local_4x4'][0][3]=.5
        with self.assertRaisesRegex(ValueError,'protected baseline'):validate_massing(self.root,self.manifest)

    def test_empty_review_evidence_cannot_approve_candidates(self):
        self.recipe['review']['proof_files']=[];self.seal()
        with self.assertRaisesRegex(ValueError,'lacks observed'):validate_massing(self.root,self.manifest)

    def test_changed_actual_footprint_identity_is_rejected(self):
        f=copy.deepcopy(self.features);f[0]['properties']['sha256']='0'*64
        path=self.root/'massing_completion/baseline_footprints.geojson';write_json(path,{'features':f})
        self.manifest['massing_completion']['baseline_footprints']=proof(path,self.root)
        with self.assertRaisesRegex(ValueError,'different model bytes'):validate_massing(self.root,self.manifest)

    def test_new_pose_or_donor_bytes_cannot_inherit_review(self):
        self.manifest['massing_instances'][0]['world_from_local_4x4'][0][3]+=1
        with self.assertRaisesRegex(ValueError,'reviewed recipe'):validate_massing(self.root,self.manifest)

    def test_mirrored_or_changed_donor_placement_is_rejected_at_construction(self):
        recipe=copy.deepcopy(self.recipe);recipe['parts'][0]['world_from_local_4x4'][0][0]=-1
        with self.assertRaisesRegex(ValueError,'orientation'):construct_manifest(self.baseline,recipe)
        recipe=copy.deepcopy(self.recipe);recipe['parts'][0]['donor_lods']['0.25M']['sha256']='0'*64
        # The candidate recipe is independent of the immutable baseline.
        original=read(self.root/'massing_completion/input_binding_manifest.json')
        with self.assertRaisesRegex(ValueError,'donor mesh identity'):construct_manifest(original,recipe)

    def test_changed_source_polygon_cannot_be_used_as_observation_support(self):
        self.recipe['source_decisions'][0]['source_geometry']=mapping(box(10,10,11,11));self.seal()
        with self.assertRaisesRegex(ValueError,'retained current source'):validate_massing(self.root,self.manifest)

    def test_actual_mesh_rejects_existing_core_collision(self):
        features=copy.deepcopy(self.features);features[0]['geometry']=self.geom
        with self.assertRaisesRegex(ValueError,'existing model cores'):
            _measure((str(self.root),self.manifest['massing_instances'][0],self.geom,features))

    def test_actual_mesh_rejects_unsupported_patch_and_wrong_height(self):
        row=self.manifest['massing_instances'][0]
        with self.assertRaisesRegex(ValueError,'observed missing patch'):
            _measure((str(self.root),row,mapping(box(10,10,11,11)),self.features))
        row=copy.deepcopy(row);row['target_top_z_m']=5
        with self.assertRaisesRegex(ValueError,'declared height'):_measure((str(self.root),row,self.geom,self.features))

    def test_pruning_honors_user_exclusions_and_never_inherits_old_scene_approval(self):
        policy={'excluded_source_ids':['B'],'minimum_width_m':.5,'minimum_area_m2':.5,
            'tall_height_m':70.,'tall_minimum_area_m2':250.}
        candidate=prune_recipe(self.recipe,policy)
        self.assertEqual(candidate['parts'],[])
        self.assertEqual(candidate['pruning']['removed_parts'],1)
        self.assertEqual(candidate['source_decisions'][0]['parts'],[])
        self.assertFalse(candidate['review']['approved'])
        self.assertEqual(candidate['review']['proof_files'],[])
        self.assertEqual(len(self.recipe['parts']),1)

    def test_narrow_small_and_tall_small_parts_are_omitted_while_larger_bodies_remain(self):
        recipe=copy.deepcopy(self.recipe);original=recipe['parts'][0]
        recipe['parts']=[]
        for name,w,length,height in [('narrow',3,60,20),('small',8,10,20),('needle',12,12,100),('body',12,30,100)]:
            recipe['parts'].append({**copy.deepcopy(original),'id':name,'target_geometry':mapping(box(0,0,w,length)),
                'target_base_z_m':0,'target_top_z_m':height})
        recipe['source_decisions'][0]['parts']=[p['id'] for p in recipe['parts']]
        policy={'excluded_source_ids':[],'minimum_width_m':8.,'minimum_area_m2':120.,
            'tall_height_m':70.,'tall_minimum_area_m2':250.}
        candidate=prune_recipe(recipe,policy)
        self.assertEqual([p['id'] for p in candidate['parts']],['body'])
        self.assertEqual(candidate['pruning']['removed_parts'],3)

    def test_omission_does_not_reenable_old_observation_surfaces(self):
        baseline=copy.deepcopy(self.baseline)
        baseline['surface_instances']=[{'id':'surface_B','source_building_ids':['B'],'assembly_enabled':True}]
        baseline['buildings'][-1]['postprocess']={'surface':{'assembly_enabled':True}}
        recipe=copy.deepcopy(self.recipe);recipe['parts']=[];recipe['all_observation_surfaces_reference_only']=True
        result=construct_manifest(baseline,recipe)
        self.assertFalse(result['surface_instances'][0]['assembly_enabled'])
        self.assertFalse(result['buildings'][-1]['postprocess']['surface']['assembly_enabled'])

    def test_individual_review_keeps_a_group_body_and_only_removes_its_small_patch(self):
        recipe=copy.deepcopy(self.recipe);body=recipe['parts'][0]
        body.update(id='body',target_geometry=mapping(box(0,0,6,18.5)),target_top_z_m=19.)
        patch={**copy.deepcopy(body),'id':'patch','target_geometry':mapping(box(0,0,3.5,4.5))}
        recipe['parts']=[body,patch];recipe['source_decisions'][0]['parts']=['body','patch']
        policy={'excluded_source_ids':[],'minimum_width_m':8.,'minimum_area_m2':120.,
            'tall_height_m':70.,'tall_minimum_area_m2':250.,'part_reviews':{
                'body':{'retain':True,'reason':'Coherent low main body; dimensions alone do not warrant removal.',
                        'reviewer':'test-reviewer','part_identity':part_review_identity(body)}}}
        candidate=prune_recipe(recipe,policy)
        self.assertEqual([p['id'] for p in candidate['parts']],['body'])
        self.assertEqual(candidate['source_decisions'][0]['removed_part_ids'],['patch'])
        self.assertEqual(candidate['pruning']['decisions'][0]['decision_basis'],'individual_part_review')
        self.assertIsNotNone(candidate['pruning']['decisions'][0]['size_screen_reason'])
        self.assertFalse(candidate['review']['approved']);self.assertFalse(candidate['review']['scene_approval'])
        policy['excluded_source_ids']=['B']
        self.assertEqual(prune_recipe(recipe,policy)['parts'],[])

    def test_individual_review_cannot_survive_a_changed_candidate(self):
        policy={'excluded_source_ids':[],'minimum_width_m':8.,'minimum_area_m2':120.,
            'tall_height_m':70.,'tall_minimum_area_m2':250.,'part_reviews':{
                'MASSING_B_01':{'retain':True,'reason':'Reviewed in context','reviewer':'test-reviewer',
                    'part_identity':part_review_identity(self.recipe['parts'][0])}}}
        recipe=copy.deepcopy(self.recipe);recipe['parts'][0]['target_top_z_m']+=1
        with self.assertRaisesRegex(ValueError,'different candidate'):prune_recipe(recipe,policy)
