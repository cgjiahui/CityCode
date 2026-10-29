"""Whole-source accounting, portable resources and immutable phase contracts."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh
from shapely.geometry import box, mapping, shape

from citycode_assets import batch
from citycode_assets.appearance import completion_contract, sha256_file, write_json
from citycode_assets.batch_gpu import frozen_selection, export_one
from citycode_assets.batch_resources import resource_contract, verify_runtime_contract
from citycode_assets.generation import pipeline_specification


def fake_prepare(task):
    building, folder, provider, cached=task;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    row={'building_id':batch.read(building)['building_id']}
    write_json(folder/'appearance_input.json',row)
    return {'building_id':row['building_id'],'cached_conditioning':False}


class UnifiedBuildingBatch(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.project=Path(self.temp.name)
        self.source=self.project/'data/structure/building_generation_inputs';self.source.mkdir(parents=True)
        records=[]
        for index,sid in enumerate(['A','B','fragment','record_only']):
            folder=self.source/'buildings'/sid;(folder/'geometry').mkdir(parents=True)
            world=np.eye(4);world[0,3]=index*2
            building={'schema':'citycode.building_generation_input.v1','status':'complete','building_id':sid,
                'footprint_world':mapping(box(index*2,-1,index*2+2,1)), 'base_z_m':0.,'top_z_m':4.,'height_m':4.,
                'world_from_local_4x4':world.tolist(),'local_from_world_4x4':np.linalg.inv(world).tolist(),
                'roof_layers':[],'observations':[],'paths':{'proxy_local_ue_z_up':'geometry/proxy.glb'}}
            mesh=trimesh.creation.box([2,2,4]);mesh.apply_translation([1,0,2]);mesh.export(folder/'geometry/proxy.glb')
            write_json(folder/'building.json',building);records.append({'building_id':sid,'path':'buildings/'+sid,'status':'complete'})
        manifest=self.source/'manifest.json';write_json(manifest,{'building_count':4,'records':records})
        self.dispositions=self.project/'decisions.json'
        self.rules={'input_manifest_sha256':sha256_file(manifest),'reference_assisted_qualification':False,'sources':[
            {'building_id':'A','action':'generate','reason':'complete'}, {'building_id':'B','action':'generate','reason':'same building'},
            {'building_id':'fragment','action':'observed_surface','reason':'partial'},
            {'building_id':'record_only','action':'record_only','exclusion_status':'excluded_incomplete','reason':'insufficient evidence'}],
            'shared_physical_groups':[{'asset_id':'A_B','source_ids':['A','B']}]}
        write_json(self.dispositions,self.rules);provider=self.project/'provider.json';write_json(provider,{})
        self.config={'output':str(self.project/'data/run'),'_project_root':str(self.project),
            'input_manifest':str(manifest),'dispositions':str(self.dispositions),'provider_config':str(provider)}
        runtime=self.project/'runtime.json';pipeline=self.project/'pipeline.json';write_json(pipeline,{})
        self.weight=self.project/'weight.pt';self.weight.write_bytes(b'weights')
        write_json(self.project/'selected.json',{'models':[]})
        write_json(self.project/'base.json',{'files':[batch.proof(self.weight,self.project)]})
        write_json(self.project/'wheels.json',{'files':[]})
        write_json(runtime,{'model_manifest':'selected.json','base_resources_manifest':'base.json',
            'gpu_wheels_manifest':'wheels.json','pipeline_config':'pipeline.json'})
        self.config['runtime_config']=str(runtime)

    def prepare(self):
        with patch.object(batch,'ProcessPoolExecutor',ThreadPoolExecutor),patch.object(batch,'_prepare_one',fake_prepare):
            return batch.prepare(self.config,workers=1)

    def test_all_sources_accounted_and_shared_asset_generated_once(self):
        before={p:sha256_file(p) for p in self.source.rglob('*') if p.is_file()}
        self.prepare();plan=batch.read(Path(self.config['output'])/'plan.json')
        self.assertEqual(plan['source_count'],4);self.assertEqual(plan['physical_generations'],1)
        self.assertEqual(plan['source_actions'],{'generate':2,'observed_surface':1,'record_only':1})
        self.assertEqual(plan['historical_model_inputs'],0)
        derived=batch.read(Path(self.config['output'])/plan['records'][0]['building']['path'])
        self.assertEqual(shape(derived['footprint_world']).area,8)
        self.assertEqual(derived['source_group']['members'],['A','B'])
        self.assertTrue(all(sha256_file(p)==value for p,value in before.items()))

    def test_missing_source_or_duplicate_shared_members_are_rejected(self):
        broken=copy.deepcopy(self.rules);broken['sources'].pop();write_json(self.dispositions,broken)
        with self.assertRaisesRegex(ValueError,'cover every source'):self.prepare()
        broken=copy.deepcopy(self.rules);broken['shared_physical_groups'][0]['source_ids']=['A','A'];write_json(self.dispositions,broken)
        with self.assertRaisesRegex(ValueError,'unique'):self.prepare()

    def test_resume_rejects_changed_dispositions(self):
        self.prepare();changed=copy.deepcopy(self.rules);changed['sources'][0]['reason']='changed review'
        write_json(self.dispositions,changed)
        with self.assertRaisesRegex(ValueError,'Run inputs changed'):self.prepare()

    def completed_review(self):
        self.prepare();root=Path(self.config['output']);record=batch.read(root/'plan.json')['records'][0]
        folder=root/record['appearance_directory'];image=folder/'generated_main_oblique.png';image.write_bytes(b'fixture image identity')
        prepared={'building_id':'A','model':'fixed','quality':'fixed','size':'1024x1024','prompt_sha256':'a'*64,
            'inputs':[{'sha256':'b'*64}], 'provenance':{'building_sha256':record['building']['sha256']}}
        write_json(folder/'appearance_input.json',prepared)
        write_json(folder/'completion.json',{'status':'complete','output_sha256':sha256_file(image),'request_identity_sha256':completion_contract(prepared)})
        review={'records':[{'asset_id':'A_B','decision':'generate','reason':'reviewed current image','image_sha256':sha256_file(image)}]}
        write_json(root/'reviews/appearance.json',review)
        return root,image

    def test_freeze_and_gpu_selection_reject_changed_source_and_image(self):
        root,image=self.completed_review();batch.freeze(self.config)
        self.assertEqual(len(frozen_selection(self.config)),1)
        image.write_bytes(b'changed image')
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):frozen_selection(self.config)

    def test_group_secondary_source_is_also_frozen(self):
        root,image=self.completed_review();batch.freeze(self.config)
        source=self.source/'buildings/B/building.json';source.write_text(source.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):frozen_selection(self.config)

    def test_same_size_weight_tampering_and_changed_runtime_are_rejected(self):
        root,image=self.completed_review();batch.freeze(self.config)
        frozen=batch.read(root/'FROZEN.json')
        self.weight.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):
            verify_runtime_contract(self.config,frozen,full_hash=True)
        self.weight.write_bytes(b'weights')
        recipe=Path(self.config['runtime_config']);recipe.write_text(recipe.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Frozen runtime'):frozen_selection(self.config)

    def test_unselected_or_traversing_export_id_is_rejected_before_GPU_import(self):
        root,image=self.completed_review();batch.freeze(self.config)
        with self.assertRaisesRegex(ValueError,'not in the frozen'):export_one(self.config,'../../other')

    def test_run_does_not_skip_changed_input_contract(self):
        self.prepare();changed=copy.deepcopy(self.rules);changed['sources'][0]['reason']='new decision'
        write_json(self.dispositions,changed)
        with self.assertRaisesRegex(ValueError,'Run inputs changed'):batch.run(self.config)

    def test_provider_dispatch_stops_on_auth_failure_without_retrying_cached_images(self):
        self.prepare();root=Path(self.config['output']);plan=batch.read(root/'plan.json')
        # Test the scheduler with independent prepared assets; no network is used.
        plan['records']=[{**plan['records'][0],'asset_id':str(i),'appearance_directory':'appearance/'+str(i)} for i in range(10)]
        write_json(root/'plan.json',plan)
        def rejected(folder,**kwargs):
            write_json(folder/'provider_response.json',{'status_code':401})
            write_json(folder/'completion.json',{'status':'failed'})
            raise RuntimeError('service authentication failed')
        with patch('citycode_assets.appearance.complete_building',side_effect=rejected) as call:
            batch.complete_images(self.config,workers=1)
            self.assertEqual(call.call_count,1)
            batch.complete_images(self.config,workers=1)
            # The failed item is retained; a subsequent explicit batch invocation
            # can reach the next item, but never automatically retries the first.
            self.assertEqual(call.call_count,2)
        self.assertTrue(batch.read(root/'IMAGE_EXECUTION.json')['dispatch_stopped'])

    def test_cpu_preparation_does_not_accept_an_unrelated_conditioning_image(self):
        root,image=self.completed_review();folder=image.parent
        prepared=batch.read(folder/'appearance_input.json');prepared['building_id']='another_source';write_json(folder/'appearance_input.json',prepared)
        building=root/batch.read(root/'plan.json')['records'][0]['building']['path']
        with self.assertRaisesRegex(ValueError,'not bound'):batch._copy_conditioning(folder,root/'copied',building)

    def test_resource_paths_relocate_without_rewriting_recipe_bytes(self):
        recipe={'args':{'models':{'decoder':'models/base/decoder'},'image_cond_model':{'args':{'model_name':'models/base/dino'}},
                        'rembg_model':{'args':{'model_name':'models/base/rembg'}}}}
        source=self.project/'pipeline.json';write_json(source,recipe);before=sha256_file(source)
        a=pipeline_specification({'pipeline_config':str(source),'model_package_root':str(self.project/'A')})
        b=pipeline_specification({'pipeline_config':str(source),'model_package_root':str(self.project/'B')})
        self.assertEqual(a['models']['decoder'],str(self.project/'A/models/base/decoder'))
        self.assertEqual(b['models']['decoder'],str(self.project/'B/models/base/decoder'))
        self.assertEqual(before,sha256_file(source))

    def test_final_binding_preserves_shared_source_frames_and_requires_current_review(self):
        from citycode_assets.batch_delivery import finalize_batch, source_row
        from citycode_assets.cli import validate_assets
        from citycode_assets.lod import inspect_glb
        from citycode_assets.scene_completion import placement_identity
        from test_asset_delivery_contract import cube_glb
        root,image=self.completed_review();batch.freeze(self.config)
        matrix=np.array([[4,0,0,0],[0,0,-2,1],[0,4,0,0],[0,0,0,1]],dtype=float)
        effective=batch.read(root/'FROZEN.json')['records'][0]
        building=batch.read(root/effective['building']['path'])
        candidate=source_row(self.source/'buildings/A/building.json',root)
        candidate.update(status='accepted_generated',asset_instance_id='A_B',
            local_coordinate_system='glTF Y-up',world_from_local_4x4=matrix.tolist(),
            world_from_building_local_4x4=np.eye(4).tolist(),building_local_from_asset_4x4=matrix.tolist(),
            normal_transform_world_3x3=np.linalg.inv(matrix[:3,:3]).T.tolist(),
            target_geometry=building['footprint_world'],lods={})
        for label,filename in [('0.25M','high.glb'),('0.1M','low.glb')]:
            path=root/'assets/A_B'/filename;cube_glb(path)
            candidate['lods'][label]={**inspect_glb(path),'path':str(path.relative_to(root)),'faces':12}
        folder=root/'validation/A_B';write_json(folder/'candidate_record.json',candidate)
        write_json(folder/'RESULT.json',{'status':'numeric_pass_model_review_pending',
            'candidate_record':batch.proof(folder/'candidate_record.json',root)})
        partials=[]
        for sid in ['fragment','record_only']:
            row=source_row(self.source/'buildings'/sid/'building.json',root)
            row.update(status='excluded_incomplete',reason='Fixture has insufficient observation',evidence={'fixture':True})
            partials.append(row)
        self.config['_config_path']=str(self.project/'configuration.json')
        with patch('citycode_assets.batch_delivery.prepare_nonbuilding_sources',return_value=partials):
            report=finalize_batch(self.config)
            self.assertFalse(report['complete']);self.assertFalse((root/'binding_manifest.json').exists())
            write_json(root/'reviews/models.json',{'records':[{'asset_id':'A_B','decision':'include',
                'reason':'Actual shared fixture geometry and placement reviewed',
                'placement_identity':placement_identity(candidate)}]})
            report=finalize_batch(self.config)
        self.assertTrue(report['complete'],report)
        self.assertEqual(report['physical_assets'],1);self.assertEqual(report['source_instances'],4)
        manifest=batch.read(root/'binding_manifest.json')
        second=next(r for r in manifest['buildings'] if r['building_id']=='B')
        np.testing.assert_allclose(np.asarray(second['world_from_building_local_4x4'])@
            second['building_local_from_asset_4x4'],matrix)
        self.assertEqual(validate_assets(root/'binding_manifest.json',require_complete=True,geometry=True)['status'],'passed')


if __name__=='__main__':unittest.main()
