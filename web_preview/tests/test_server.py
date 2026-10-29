"""The local preview must expose only explicitly published files."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from io import BytesIO

SOURCE=Path(__file__).resolve().parents[1]/'serve.py'
spec=importlib.util.spec_from_file_location('citycode_preview_server',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ServerTests(TestCase):
    def test_background_only_exposes_manifest_files_and_rejects_tampering(self):
        import json, hashlib
        with TemporaryDirectory() as temporary:
            root=Path(temporary); scene={}; allowed={}
            for name in ['map.jpg','surfaces.glb','pixal3d.glb','coastal.glb','map_solid.glb']: (root/name).write_bytes(b'preview')
            manifest={'schema':'citycode.background_web_preview.v1','files':{
                role:{'path':name,'sha256':hashlib.sha256(b'preview').hexdigest()}
                for role,name in [('map','map.jpg'),('surfaces','surfaces.glb'),('neural','pixal3d.glb'),('coastal','coastal.glb'),('map_solid','map_solid.glb')]}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            module.add_background(scene,allowed,root)
            self.assertEqual(set(allowed),{'background/map.jpg','background/surfaces.glb','background/pixal3d.glb','background/coastal.glb','background/map_solid.glb'})
            self.assertEqual(scene['background']['files']['map']['url'],'/data/background/map.jpg')
            (root/'coastal.glb').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):module.add_background({}, {},root)
            (root/'coastal.glb').write_bytes(b'preview')
            (root/'map_solid.glb').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):module.add_background({}, {},root)

    def request(self, path, *, headers=None, head=False):
        with TemporaryDirectory() as temporary:
            file=Path(temporary)/'model.glb';file.write_bytes(b'0123456789')
            Handler=module.make_handler({'assets':[]},{'model.glb':file})
            handler=object.__new__(Handler)
            handler.path=path;handler.headers=headers or {};handler.wfile=BytesIO()
            handler.status=None;handler.response_headers={}
            handler.send_response=lambda status:setattr(handler,'status',status)
            handler.send_header=lambda name,value:handler.response_headers.update({name:value})
            handler.end_headers=lambda:None
            handler.send_error=lambda status:setattr(handler,'status',status)
            handler.serve(head)
            return handler.status,handler.response_headers,handler.wfile.getvalue()

    def test_published_model_range(self):
        status,headers,data=self.request('/data/model.glb',headers={'Range':'bytes=2-5'})
        self.assertEqual(status,206);self.assertEqual(data,b'2345')
        self.assertEqual(headers['Content-Range'],'bytes 2-5/10')

    def test_tail_range(self):
        self.assertEqual(self.request('/data/model.glb',headers={'Range':'bytes=-3'})[2],b'789')

    def test_invalid_range(self):
        self.assertEqual(self.request('/data/model.glb',headers={'Range':'bytes=20-30'})[0],416)

    def test_head_has_no_body(self):
        status,headers,data=self.request('/data/model.glb',head=True)
        self.assertEqual(status,200);self.assertEqual(headers['Content-Length'],'10');self.assertEqual(data,b'')

    def test_unknown_and_traversal_paths_denied(self):
        for path in ['/data/binding_manifest.json','/data/../../serve.py','/%2e%2e/serve.py','/../../README.md','/api/../serve.py']:
            self.assertEqual(self.request(path)[0],404,path)

    def test_completed_scene_counts_and_massing_height_do_not_reclassify_sources(self):
        import json
        with TemporaryDirectory() as temporary:
            root=Path(temporary)/'data/completed';root.mkdir(parents=True)
            (root/'model.glb').write_bytes(b'cached-model')
            lod={'path':'model.glb','bytes':12,'faces':12,'bounds_local_accessor':[[0,0,0],[1,1,1]]}
            geometry={'type':'Polygon','coordinates':[[[2,0],[3,0],[3,1],[2,1],[2,0]]]}
            matrix=[[1,0,0,2],[0,0,-1,1],[0,16.02,0,.48],[0,0,0,1]]
            base={'id':'PGB_SC_000001','source_building_ids':['PGB_SC_000001'],'status':'accepted_reuse',
                'world_from_local_4x4':matrix,'lods':{'0.25M':lod,'0.1M':lod}}
            part={**base,'id':'MASSING_PGB_SC_000002_01','source_building_ids':['PGB_SC_000002'],
                'status':'accepted_massing_completion','target_geometry':geometry,'target_base_z_m':.48,'target_top_z_m':16.5}
            part2={**part,'id':'MASSING_PGB_SC_000002_02'}
            source_geometry={'type':'Polygon','coordinates':[[[2,0],[5,0],[5,1],[3,1],[3,3],[2,3],[2,0]]]}
            binding={'buildings':[{'building_id':'PGB_SC_000001'},
                {'building_id':'PGB_SC_000002','base_z_m':74,'top_z_m':90,'status':'resolved_fragment','target_geometry':source_geometry}],
                'asset_instances':[base],'massing_instances':[part,part2],
                'building_entities':[{'asset_instance_ids':[base['id']]}]}
            (root/'binding_manifest.json').write_text(json.dumps(binding))
            scene,allowed=module.build_scene(root)
            self.assertEqual(scene['counts'],{'assets':3,'base_assets':1,'massing_parts':2,'entities':1,'source_instances':2,'surfaces':0})
            displayed=next(a for a in scene['assets'] if a['massing'])
            self.assertEqual(displayed['display_number'],'2·1')
            self.assertEqual(displayed['base'],.48);self.assertEqual(displayed['top'],16.5)
            self.assertEqual(displayed['group'],[part['id'],part2['id']])
            self.assertEqual(len(scene['massing_sources']),1)
            source=scene['massing_sources'][0]
            self.assertEqual(source['part_ids'],[part['id'],part2['id']])
            self.assertEqual(source['polygons'],[source_geometry['coordinates']])
            self.assertEqual(source['outline_z'],17.2)
            self.assertEqual(allowed,{'model.glb':root/'model.glb'})
