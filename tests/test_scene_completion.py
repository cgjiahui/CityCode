import copy,hashlib,json,unittest
from pathlib import Path
from shapely.geometry import box,mapping
import test_asset_delivery_contract as fixtures
from citycode_assets.cli import _validate_accepted,_policy
from citycode_assets.scene_completion import (placement_identity,scene_identity,
    validate_scene_completion_context,validate_scene_completion_review)

class VisualSceneCompletionContract(unittest.TestCase):
    def setUp(self):
        f=fixtures.AssetDeliveryContract();f.setUp();self.addCleanup(f.doCleanups)
        self.root=f.root;self.record=copy.deepcopy(f.record);self.manifest=f.manifest
        r=self.record;r['status']='accepted_scene_completion';r['structure_path']='source.json'
        source={'building_id':r['building_id'],'footprint_world':r['target_geometry']}
        self.source=self.root/'source.json';self.source.write_text(json.dumps(source))
        files=[]
        for role in ['actual_front','actual_back','scene_intersections']:
            p=self.root/(role+'.json');p.write_text('{}');files.append({'role':role,'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        r['scene_completion_review']={'schema':'citycode.visual_scene_completion.v1','placement_identity':placement_identity(r),
            'source_building_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'proof_files':files,
            'real_building_confirmed':True,'appearance_coherent':True,'no_major_scene_conflict':True,'reason':'Explicit source and actual model visual assessment'}

    def test_review_binds_actual_mesh_and_pose(self):
        validate_scene_completion_review(self.root,self.record)
        self.record['world_from_local_4x4'][0][3]+=1
        with self.assertRaisesRegex(ValueError,'different mesh/placement'):validate_scene_completion_review(self.root,self.record)

    def test_source_geometry_and_proof_files_cannot_change(self):
        self.source.write_text('{}')
        with self.assertRaisesRegex(ValueError,'source evidence changed'):validate_scene_completion_review(self.root,self.record)

    def test_incomplete_assessment_cannot_pass(self):
        self.record['scene_completion_review']['appearance_coherent']=False
        with self.assertRaisesRegex(ValueError,'assessments'):validate_scene_completion_review(self.root,self.record)

    def test_rendered_evidence_bytes_are_bound(self):
        (self.root/'actual_front.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'proof bytes changed'):validate_scene_completion_review(self.root,self.record)

    def test_scene_change_requires_repeat_review(self):
        r=self.record;inst={'id':r['building_id'],'lods':r['lods'],'world_from_local_4x4':r['world_from_local_4x4']}
        m={'buildings':[r],'asset_instances':[inst]};r['scene_completion_review']['scene_identity']=scene_identity(m)
        validate_scene_completion_context(m)
        m['asset_instances'].append({**copy.deepcopy(inst),'id':'new_neighbour'})
        with self.assertRaisesRegex(ValueError,'scene changed'):validate_scene_completion_context(m)

    def test_reenabling_surface_invalidates_scene_review(self):
        r=self.record;inst={'id':r['building_id'],'lods':r['lods'],'world_from_local_4x4':r['world_from_local_4x4']}
        surface={**copy.deepcopy(inst),'id':'roof_fragment','assembly_enabled':False}
        m={'buildings':[r],'asset_instances':[inst],'surface_instances':[surface]};r['scene_completion_review']['scene_identity']=scene_identity(m)
        validate_scene_completion_context(m);surface['assembly_enabled']=True
        with self.assertRaisesRegex(ValueError,'scene changed'):validate_scene_completion_context(m)

    def test_standard_geometry_gate_is_preserved(self):
        r=self.record;r['target_geometry']=mapping(box(0,-1,4,0));s=json.loads(self.source.read_text());s['footprint_world']=r['target_geometry'];self.source.write_text(json.dumps(s));r['scene_completion_review']['source_building_sha256']=hashlib.sha256(self.source.read_bytes()).hexdigest()
        # The reviewed model is supported by one part of a larger incomplete
        # observation; it is explicitly not a strict footprint reconstruction.
        _validate_accepted(self.root,r,_policy(self.manifest),geometry=True)
        r['status']='accepted_generated'
        with self.assertRaisesRegex(ValueError,'footprint fails'):_validate_accepted(self.root,r,_policy(self.manifest),geometry=True)

    def test_missing_or_escaped_evidence_rejected(self):
        self.record['scene_completion_review']['proof_files'][0]['path']='../escape.png'
        with self.assertRaisesRegex(ValueError,'inside the asset package'):validate_scene_completion_review(self.root,self.record)

    def test_unrelated_model_has_insufficient_source_support(self):
        r=self.record;r['target_geometry']=mapping(box(0,-1,10,0))
        self.source.write_text(json.dumps({'building_id':r['building_id'],'footprint_world':r['target_geometry']}));r['scene_completion_review']['source_building_sha256']=hashlib.sha256(self.source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'footprint fails'):_validate_accepted(self.root,r,_policy(self.manifest),geometry=True)

if __name__=='__main__':unittest.main()
