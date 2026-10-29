"""A cache hit preserves real provenance and requires identical source inputs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from citycode_assets.appearance import write_json,sha256_file
from citycode_assets.batch import proof,read
from citycode_assets.cache_batch import seal_cache,restore_cache,cache_inputs
from citycode_assets.lod import inspect_glb
from test_asset_delivery_contract import cube_glb


class FrozenAssetCache(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);project=Path(self.temp.name)
        self.cache=project/'data/cache';self.cache.mkdir(parents=True)
        self.inputs=project/'data/structure/building_generation_inputs';folder=self.inputs/'buildings/A';folder.mkdir(parents=True)
        geometry={'type':'Polygon','coordinates':[[[0,-1],[1,-1],[1,0],[0,0],[0,-1]]]}
        write_json(folder/'building.json',{'building_id':'A','footprint_world':geometry})
        write_json(self.inputs/'manifest.json',{'building_count':1,'records':[{'building_id':'A','path':'buildings/A'}]})
        matrix=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float)
        row={'building_id':'A','asset_instance_id':'A','status':'accepted_reuse','reason':'Reviewed historical model reuse',
            'target_geometry':geometry,'base_z_m':0,'top_z_m':1,'local_coordinate_system':'glTF Y-up',
            'world_from_local_4x4':matrix.tolist(),'building_local_from_asset_4x4':matrix.tolist(),
            'world_from_building_local_4x4':np.eye(4).tolist(),'normal_transform_world_3x3':matrix[:3,:3].tolist(),
            'provenance':{'kind':'historical_model_reuse'},'lods':{}}
        for label,name in [('0.25M','high.glb'),('0.1M','low.glb')]:
            path=self.cache/'assets/A'/name;cube_glb(path)
            row['lods'][label]={**inspect_glb(path),'path':str(path.relative_to(self.cache)),'faces':12}
        binding={'schema':'citycode.building_asset_bindings.v1','path_base':'manifest_directory',
            'source_structure':{'instance_count':1,'building_ids':['A'],'manifest_sha256':sha256_file(self.inputs/'manifest.json')},
            'buildings':[row],'asset_instances':[{'id':'A','source_building_ids':['A'],'status':'accepted_reuse',
                'world_from_local_4x4':matrix.tolist(),'lods':row['lods'],'building_entity_id':'A'}],
            'building_entities':[{'id':'A','asset_instance_ids':['A'],'source_building_ids':['A']}],'surface_instances':[]}
        write_json(self.cache/'binding_manifest.json',binding)
        write_json(self.cache/'reproduction/manifest.json',{'binding_manifest':proof(self.cache/'binding_manifest.json',self.cache),
            'source_manifest':proof(self.inputs/'manifest.json',self.cache),
            'source_buildings':[{'building_id':'A',**proof(folder/'building.json',self.cache)}]})
        self.manifest=project/'frozen_cache.json';seal_cache(self.cache,self.inputs/'manifest.json',self.manifest)
        self.config={'_project_root':str(project),'asset_cache':str(self.cache),'asset_cache_manifest':str(self.manifest),
            'input_manifest':str(self.inputs/'manifest.json'),'output':str(project/'data/restored')}

    def test_restore_is_identical_and_preserves_reuse_without_network_or_GPU(self):
        with patch('citycode_assets.appearance.complete_building') as api,patch('citycode_assets.batch_gpu.run_gpu_batch') as gpu:
            result=restore_cache(self.config,replay_special=False)
            api.assert_not_called();gpu.assert_not_called()
        root=Path(self.config['output'])
        self.assertEqual(result['fresh_neural_models'],0);self.assertTrue(result['binding_identical_to_frozen_example'])
        self.assertEqual(sha256_file(root/'binding_manifest.json'),sha256_file(self.cache/'binding_manifest.json'))
        self.assertEqual(read(root/'binding_manifest.json')['buildings'][0]['provenance']['kind'],'historical_model_reuse')
        self.assertEqual(restore_cache(self.config,replay_special=False)['status'],'complete_verified_asset_cache')

    def test_changed_source_or_cached_model_is_rejected_before_delivery(self):
        source=self.inputs/'buildings/A/building.json';source.write_text(source.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):cache_inputs(self.config)
        source.write_text(source.read_text()[:-1])
        model=self.cache/'assets/A/high.glb';raw=model.read_bytes();model.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        with self.assertRaisesRegex(ValueError,'cache file missing or changed'):restore_cache(self.config,replay_special=False)
        self.assertFalse((Path(self.config['output'])/'binding_manifest.json').exists())
