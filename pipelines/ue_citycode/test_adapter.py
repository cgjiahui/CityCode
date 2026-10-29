"""Focused CPU regressions for geometry and bridge coordinate contracts."""
import copy
import ast
import contextlib
import io
import json
import locale
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from interchange import Writer, accessor, bake, read_glb
from prepare import bind_placement, select_lod_sources, source_reference, main as prepare_main
from split_background import split
from prepare_elevated_repair import ActualDeckGeometry, candidate_manifest, refit_supports
from audit_elevated_geometry import TriangleBoxes, triangle_intersections
from apply_elevated_repair import digest, equivalent_placement, finalize_manifest, same_state, save_candidate_copy
from audit_elevated_landings import top_triangle_hits


class AdapterTests(unittest.TestCase):
    def test_custom_binding_1m_cli_records_external_sources_with_utf8_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);release=base/'release';config=release/'workflow/04_citycode_assembly/standard.json'
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({'inputs':{'building_binding':'不会读取.json'},'default_scene':{'building_lod':'0.25M'}},ensure_ascii=False),encoding='utf-8')
            high=base/'一百万面.glb';high.write_bytes(b'new-1M-export')
            binding=base/'物理绑定.json'
            binding.write_text(json.dumps({'asset_instances':[{'id':'building','building_entity_id':'entity',
                'world_from_local_4x4':np.eye(4).tolist(),'lods':{'1M':{'path':high.name,'sha256':digest(high)}}}]},ensure_ascii=False),encoding='utf-8')
            output=release/'build/new-1m';old_locale=locale.setlocale(locale.LC_CTYPE)
            try:
                locale.setlocale(locale.LC_CTYPE,'C')
                with patch('sys.argv',['prepare.py','--root',str(release),'--binding',str(binding),'--lod','1M',
                        '--output',str(output),'--without-background']),contextlib.redirect_stdout(io.StringIO()):
                    prepare_main()
            finally:locale.setlocale(locale.LC_CTYPE,old_locale)
            result=json.loads((output/'assembly_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(result['building_lod'],'1M');self.assertEqual(result['source_binding'],str(binding))
            self.assertEqual(result['source_binding_path_base'],'absolute_external_input')
            self.assertEqual(result['meshes'][0]['source_file'],str(high));self.assertEqual(result['meshes'][0]['source_sha256'],digest(high))
            self.assertEqual(result['status'],'preparation_incomplete');self.assertEqual(result['pending_interchange_ids'],['building'])

    def test_explicit_1m_selects_every_asset_and_donor_without_mutating_the_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);high=root/'high.glb';low=root/'low.glb'
            high.write_bytes(b'high-resolution');low.write_bytes(b'low-resolution')
            high_detail={'path':'high.glb','sha256':digest(high)}
            binding={'asset_instances':[{'id':'building','lods':{'0.25M':{'path':'low.glb','sha256':digest(low)},'1M':high_detail}},
                {'id':'disabled','assembly_enabled':False,'lods':{}}],
                'massing_instances':[{'id':'donor','donor_lods':{'1M':high_detail}}]}
            before=copy.deepcopy(binding);selected=select_lod_sources(binding,root/'custom_binding.json','1M')
            self.assertEqual([row[0]for row in selected],['building','massing'])
            self.assertEqual([row[3]for row in selected],[high,high]);self.assertEqual(binding,before)
            name,base=source_reference(high,root/'release')
            self.assertEqual(name,str(high));self.assertEqual(base,'absolute_external_input')

    def test_incomplete_1m_reports_all_missing_rows_and_never_uses_025m(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);low=root/'low.glb';low.write_bytes(b'low')
            binding={'asset_instances':[{'id':'building-no-1m','lods':{'0.25M':{'path':'low.glb','sha256':digest(low)}}}],
                'massing_instances':[{'id':'donor-missing-file','donor_lods':{'1M':{'path':'missing.glb','sha256':'a'*64}}}]}
            with self.assertRaises(ValueError) as caught:select_lod_sources(binding,root/'binding.json','1M')
            message=str(caught.exception)
            for text in ('2 enabled asset/donor rows','no fallback','building-no-1m','donor-missing-file','missing.glb'):
                self.assertIn(text,message)
            self.assertEqual(low.read_bytes(),b'low')

    def test_repair_entry_leaves_shutdown_to_execute_python_script(self):
        tree=ast.parse(Path(__file__).with_name('apply_elevated_repair.py').read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and node.func.attr=='quit_editor']
        self.assertEqual(calls,[])

    def test_save_map_copy_retains_original_world_then_loads_verified_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            content=Path(temporary)/"Content";old_file=content/"Research/CityCodeOpenSource/Maps/Old.umap"
            old_file.parent.mkdir(parents=True);old_file.write_bytes(b"original")
            old="/Game/Research/CityCodeOpenSource/Maps/Old";new="/Game/Research/CityCodeOpenSource/Maps/New"
            state={"level":old};loads=[]
            snapshot={old+".Old:PersistentLevel.Child":{"mesh_path":"/Game/M.M","position_cm":[1,2,3],
                "scale":[1,1,1],"rotation_quat_xyzw":[0,0,0,1],"parent_actor_path":None,"tags":["CityCodeInput=A"]}}
            def save_map(world,level):
                (content/(level[len('/Game/'):]+'.umap')).write_bytes(b"copy")
                return True
            def load_level(level):
                state['level']=level;loads.append(level);return True
            unreal=SimpleNamespace(EditorLoadingAndSavingUtils=SimpleNamespace(save_map=save_map),
                EditorLevelLibrary=SimpleNamespace(get_editor_world=lambda:SimpleNamespace(get_path_name=lambda:state['level']+'.World'),load_level=load_level),
                Paths=SimpleNamespace(project_content_dir=lambda:str(content)))
            entry=SimpleNamespace(tagged_geometry_snapshot=lambda:snapshot)
            proof=save_candidate_copy(unreal,entry,old,new,old_file,digest(old_file),snapshot)
            self.assertEqual(loads,[new]);self.assertTrue(proof['explicit_candidate_load']);self.assertEqual(old_file.read_bytes(),b'original')
            state['level']=old
            def corrupt_save(world,level):
                old_file.write_bytes(b"changed");return True
            unreal.EditorLoadingAndSavingUtils.save_map=corrupt_save
            with self.assertRaisesRegex(RuntimeError,'changed baseline bytes'):
                save_candidate_copy(unreal,entry,old,new,old_file,digest(old_file),snapshot)
            self.assertEqual(loads,[new])
            old_file.write_bytes(b"original");state['level']=old
            unreal.EditorLoadingAndSavingUtils.save_map=save_map
            changed=copy.deepcopy(snapshot);changed[next(iter(changed))]['position_cm'][2]+=1.
            entry.tagged_geometry_snapshot=lambda:changed
            with self.assertRaisesRegex(ValueError,'changed tagged geometry before any actor edits'):
                save_candidate_copy(unreal,entry,old,new,old_file,digest(old_file),snapshot)
            self.assertEqual(loads,[new,new]);self.assertEqual(old_file.read_bytes(),b'original')

    def test_landing_query_uses_actual_top_face_above_an_underside_only_center_vertex(self):
        triangles=np.array([[[0,0,1],[2,0,1],[2,2,1]],[[0,0,1],[2,2,1],[0,2,1]],
            [[0,0,-1],[2,0,-1],[1,1,-1]]],float)
        hits,top=top_triangle_hits(triangles,[1,1])
        self.assertAlmostEqual(top,1.)
        self.assertAlmostEqual(hits.min(),-1.)
        with self.assertRaisesRegex(ValueError,'No actual deck triangle'):
            top_triangle_hits(triangles,[10,10])

    def test_gui_repair_save_as_guard_compares_geometry_without_map_path_or_scene_tag(self):
        state={"mesh_path":"/Game/M.M","position_cm":[1.,2.,3.],"rotation_quat_xyzw":[0,0,0,1],
            "scale":[1,1,-1],"parent_actor_path":"/Game/Old.Old:PersistentLevel.Parent","tags":["CityCodeInput=A","CityCodeScene=old"]}
        before={"/Game/Old.Old:PersistentLevel.Child":state}
        after=copy.deepcopy(before);value=after.pop(next(iter(after)))
        value["parent_actor_path"]="/Game/New.New:PersistentLevel.Parent";value["tags"][-1]="CityCodeScene=new"
        after["/Game/New.New:PersistentLevel.Child"]=value
        self.assertTrue(same_state(before,after,ignore_scene=True));self.assertFalse(same_state(before,after))
        value["position_cm"][2]+=.01;self.assertFalse(same_state(before,after,ignore_scene=True))

    def test_gui_manifest_uses_actual_import_paths_without_rebinding_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);base=root/"base";base.mkdir();output=root/"candidate/assembly_manifest.json"
            baseline=base/"manifest.json"
            original={"scene_id":"old","placements":[],"meshes":[{"id":"building","import_file":"interchange/b.glb"}],
                "calibration":{"path":"interchange/axis.glb"},"counts":{}}
            baseline.write_text(json.dumps(original));before=baseline.read_bytes()
            path="/Game/Research/CityCodeOpenSource/Imported/NEW/M.M"
            bridge={"placement_id":"joined","category":"predicted_bridge_span","mesh_path":path,"position_m":[1,2,3],
                "rotation_deg":[0,0,0],"scale":[1,1,1],"placement_source":"predicted_elevated_structure","prepared_world_geometry":True,
                "logical_module_count":306,"expected_native_bounds_m":[[-1,-1,-1],[1,1,0]],"native_material_paths":["A","B","C","D"],"template_sha256":"a"*64}
            plan={"bound_native_placements":[],"proposal_sources":[],"proposal_decisions":[],"template_manifest_sha256":"b"*64}
            result=finalize_manifest(baseline,output,plan,"c"*64,{"placement_proposal":{"placements":[bridge],"schema":"native"}})
            self.assertEqual(baseline.read_bytes(),before);self.assertEqual(result["placements"][0]["mesh_path"],path)
            self.assertEqual(result["placements"][0]["position_cm"],[100.,200.,300.]);self.assertEqual(result["logical_bridge_module_count"],306)
            self.assertEqual((output.parent/result["meshes"][0]["import_file"]).resolve(),(base/"interchange/b.glb").resolve())
            self.assertFalse(result["ue_executed"]);self.assertNotEqual(result["scene_id"],"old")
            same=copy.deepcopy(result["placements"][0]);same["source_segment_id"]="metadata-only"
            self.assertTrue(equivalent_placement(same,result["placements"][0]))
            same["position_cm"][0]+=.1;self.assertFalse(equivalent_placement(same,result["placements"][0]))

    def test_actual_triangle_audit_rejects_bbox_only_and_degenerate_contacts(self):
        reference=np.array([[0,0,0],[2,0,0],[0,2,0]],dtype=float)
        candidates=np.array([[[1.5,1.5,0],[2,1.5,0],[1.5,2,0]],
            [[.2,.2,-1],[.2,.2,1],[1,.2,0]], [[.2,.2,0],[.8,.2,0],[.2,.8,0]],
            [[.2,.2,1],[.8,.2,1],[.2,.8,1]], [[.2,.2,0],[.2,.2,0],[.2,.2,0]]])
        np.testing.assert_array_equal(triangle_intersections(reference,candidates),[False,True,True,False,False])
        np.testing.assert_array_equal(triangle_intersections(candidates[-1],candidates),[False]*5)
        spatial=TriangleBoxes(candidates,cell_size=.5)
        queried=spatial.query(np.array([[0,0,-.1],[1,1,.1]]))
        np.testing.assert_array_equal(queried,[1,2,4])

    def test_candidate_manifest_reuses_geometry_rebases_fixture_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); base=root/"base";base.mkdir(); output=root/"candidate"/"manifest.json"
            baseline=base/"manifest.json"
            original={"placements":[{"id":"old-native"}],"meshes":[{"id":"building","import_file":"interchange/building.glb","prepared":{"mesh_count":1}}],
                "calibration":{"path":"interchange/axis.glb"},"counts":{"proposal_bound":1}}
            baseline.write_text(json.dumps(original)); original_bytes=baseline.read_bytes()
            path="/Game/Research/CityCodeOpenSource/Imported/Span/Mesh.Mesh"
            row={"placement_id":"new-span","asset_id":"new-span","mesh_path":path,"category":"predicted_bridge_span",
                "position_m":[10,20,3],"scale":[1,1,1],"rotation_deg":[0,0,0],"placement_source":"predicted_elevated_structure",
                "prepared_world_geometry":True,"logical_module_count":1,"template_sha256":"a"*64,"expected_native_bounds_m":[[-1,-1,-1],[1,1,0]],
                "native_material_paths":["/Game/A.A","/Game/B.B","/Game/C.C","/Game/D.D"]}
            proposal=root/"proposal.json";proposal.write_text(json.dumps({"coordinate_system":{"world":"UE","units":"m","up":"Z"},"placements":[row]}))
            catalog=root/"catalog.json";catalog.write_text(json.dumps({"assets":[]}))
            native=root/"native.json"; document={"status":"native_bridge_template_imported","ue_executed":True,"bounds_verified":True,
                "native_materials_verified":True,"assets":[{"mesh_path":path}]};native.write_text(json.dumps(document))
            candidate=candidate_manifest(baseline,output,[proposal],catalog,native)
            self.assertEqual(baseline.read_bytes(),original_bytes)
            self.assertEqual((output.parent/candidate["meshes"][0]["import_file"]).resolve(),(base/"interchange/building.glb").resolve())
            self.assertEqual((output.parent/candidate["calibration"]["path"]).resolve(),(base/"interchange/axis.glb").resolve())
            diff=json.loads(output.with_suffix('.diff.json').read_text())
            self.assertEqual([r['id'] for r in diff['add']],['new-span']);self.assertEqual(diff['remove'],original['placements'])
            self.assertFalse(diff['apply_executed']);self.assertFalse(candidate['ue_executed'])
            document['ue_executed']=False;native.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,'actual verified UE'):candidate_manifest(baseline,output,[proposal],catalog,native)

    def test_candidate_support_cap_fits_actual_sloped_mesh_instead_of_module_proxy(self):
        class Geometry:
            bottom={"span":np.array([[[-3,-3,5],[3,-3,5.6],[3,3,5.6]],[[-3,-3,5],[3,3,5.6],[-3,3,5]]])}
        row={"placement_id":"pillar","source_segment_id":"span","mesh_path":"/Game/Support.Support",
             "footprint":{"coordinates":[[[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]]]},
             "rotation_deg":[0,0,0],"ground_contact_z_m":.5}
        asset={"mesh_path":row["mesh_path"],"bounds_cm":[[-100,-100,0],[100,100,400]]}
        rows,withheld=refit_supports([row],Geometry(),[asset])
        self.assertEqual(withheld,[]);self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]["support_cap_z_m"],5.2)
        self.assertAlmostEqual(rows[0]["position_m"][2]+4*rows[0]["scale"][2],5.2)
        self.assertAlmostEqual(rows[0]["maximum_cap_to_sloped_deck_gap_m"],.2)
        row["footprint"]["coordinates"][0]=[[10,10],[12,10],[12,12],[10,12],[10,10]]
        rows,withheld=refit_supports([row],Geometry(),[asset])
        self.assertEqual(rows,[]);self.assertEqual(withheld[0]["reason"],"support_cap_not_fully_under_actual_candidate_span")

    def test_candidate_rail_interior_respects_actual_geometry_height_layers(self):
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree
        geometry=ActualDeckGeometry.__new__(ActualDeckGeometry)
        geometry.top=np.array([[[-10,-10,10],[10,-10,10],[10,10,10]],[[-10,-10,10],[10,10,10],[-10,10,10]]])
        geometry.polygons=[Polygon(t[:,:2]) for t in geometry.top];geometry.tree=STRtree(geometry.polygons)
        self.assertTrue(geometry.interior([0,0,10]))
        self.assertFalse(geometry.interior([0,0,5]))
        self.assertFalse(geometry.interior([10,0,10]))

    def test_node_then_binding_reflection_and_lossless_webp_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary)/"source.glb", Path(temporary)/"prepared.glb"
            writer = Writer()
            p = writer.array(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4"))
            n = writer.array(np.array([[0, 0, 1]]*3, dtype="<f4"))
            i = writer.array(np.array([[0], [1], [2]], dtype="<u4"), {"componentType": 5125, "count": 3, "type": "SCALAR"}, 34963)
            pixels = np.array([[[255, 10, 30], [20, 200, 30]], [[50, 30, 200], [100, 90, 80]]], dtype="u1")
            stream = io.BytesIO(); Image.fromarray(pixels).save(stream, "WEBP", lossless=True)
            writer.doc["images"] = [{"bufferView": writer.blob(stream.getvalue()), "mimeType": "image/webp"}]
            writer.doc["textures"] = [{"extensions": {"EXT_texture_webp": {"source": 0}}}]
            writer.doc["extensionsRequired"] = ["EXT_texture_webp"]
            writer.doc["meshes"] = [{"primitives": [{"attributes": {"POSITION": p, "NORMAL": n}, "indices": i}]}]
            writer.doc["nodes"] = [{"translation": [2, 3, 4], "children": [1]}, {"scale": [2, 3, 4], "mesh": 0}]
            writer.doc["scenes"][0]["nodes"] = [0]; writer.save(source)
            original = source.read_bytes()
            proof = bake(source, output, [[1,0,0,10], [0,0,1,20], [0,1,0,30], [0,0,0,1]], [10,20,30])
            doc, binary, _ = read_glb(output)
            primitive = doc["meshes"][0]["primitives"][0]
            positions, _ = accessor(doc, binary, primitive["attributes"]["POSITION"])
            np.testing.assert_allclose(positions, [[2,4,3], [4,4,3], [2,4,6]])
            normals, _ = accessor(doc, binary, primitive["attributes"]["NORMAL"])
            indices, _ = accessor(doc, binary, primitive["indices"])
            triangle = positions[indices.reshape(3)]
            self.assertGreater(np.dot(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0]), normals[0]), 0)
            self.assertNotIn("extensionsRequired", doc)
            self.assertEqual(doc["textures"][0]["source"], 0)
            image = doc["images"][0]; view = doc["bufferViews"][image["bufferView"]]
            decoded = Image.open(io.BytesIO(bytes(binary[view["byteOffset"]:view["byteOffset"]+view["byteLength"]])))
            np.testing.assert_array_equal(np.array(decoded), pixels)
            self.assertEqual(proof["world_bounds_m"], [[12.,24.,33.], [14.,24.,36.]])
            self.assertEqual(source.read_bytes(), original)

    def test_bridge_binds_predicted_top_to_local_pivot_and_direction(self):
        path = "/Game/Test/Deck.Deck"
        catalog = {path: {"mesh_path": path, "bounds_cm": [[-100,-200,-50], [900,200,50]],
                          "local_axes": {"length_axis": "X"}, "orientation_confirmed": True, "deck_top_local_z_cm": 50}}
        row = {"placement_id": "bridge1", "category": "bridge_deck", "asset_id": "bridge_deck_default",
               "position_m": [50,60,20], "start_m": [50,50,20], "end_m": [50,70,20], "width_m": 8,
               "placement_source": "predicted_elevated_structure"}
        bindings = {"bridge_deck_default": {"mesh_path": path, "maximum_width_scale": 2.}}
        bound, decision = bind_placement(row, catalog, bindings)
        self.assertIsNone(decision)
        np.testing.assert_allclose(bound["scale"], [2,2,1])
        np.testing.assert_allclose(bound["position_cm"], [5000,5200,1950], atol=1e-8)
        np.testing.assert_allclose(np.abs(bound["rotation_quat_xyzw"]), [0,0,np.sqrt(.5),np.sqrt(.5)])
        denied = copy.deepcopy(row); denied["source_actor_world_transform"] = [[1]]
        with self.assertRaisesRegex(ValueError, "prohibited"):
            bind_placement(denied, catalog, bindings)

    def test_unknown_mesh_orientation_is_not_silently_used(self):
        path = "/Game/Test/Deck.Deck"
        catalog = {path: {"mesh_path": path, "bounds_cm": [[0,0,0],[100,100,10]], "local_axes": {"length_axis":"X"}}}
        row = {"placement_id":"bridge1", "category":"bridge_deck", "mesh_path":path,
               "position_m":[0,0,10], "start_m":[0,0,10], "end_m":[10,0,10], "width_m":10,
               "placement_source":"predicted_elevated_structure"}
        bound, decision = bind_placement(row, catalog, {})
        self.assertIsNone(bound)
        self.assertEqual(decision["status"], "bridge_orientation_requires_confirmation")
        row["placement_source"] = "old_gt_bridge_actor"
        with self.assertRaisesRegex(ValueError, "predicted_elevated_structure"):
            bind_placement(row, catalog, {})

    def test_deformed_native_span_keeps_producer_anchor_without_deck_pivot_fitting(self):
        path = "/Game/Research/CityCodeOpenSource/Span.Span"
        row = {"placement_id":"span1", "asset_id":"span1", "mesh_path":path,
               "category":"predicted_bridge_span", "position_m":[100,200,3], "rotation_deg":[0,0,0], "scale":[1,1,1],
               "placement_source":"predicted_elevated_structure", "prepared_world_geometry":True,
               "logical_module_count":12, "expected_native_bounds_m":[[-50,-30,-1.6],[50,30,15]],
               "native_material_paths":["/Game/M/A.A","/Game/M/B.B","/Game/M/C.C","/Game/M/D.D"],
               "template_sha256":"a"*64}
        bound, decision = bind_placement(row, {path:{"mesh_path":path}}, {})
        self.assertIsNone(decision)
        self.assertEqual(bound["position_cm"],[10000.,20000.,300.])
        self.assertEqual(bound["scale"],[1.,1.,1.])
        self.assertEqual(bound["rotation_quat_xyzw"],[0.,0.,0.,1.])
        self.assertEqual(bound["logical_module_count"],12)

    def test_split_removes_only_unreferenced_vertices_and_keeps_texture_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); writer = Writer()
            stream = io.BytesIO(); Image.new("RGBA", (4, 4), (80, 90, 100, 255)).save(stream, "PNG")
            texture_bytes = stream.getvalue()
            writer.doc["images"] = [{"bufferView": writer.blob(texture_bytes), "mimeType": "image/png"}]
            writer.doc["textures"] = [{"source": 0}]
            writer.doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
            for offset in (0, 10):
                positions = np.array([[offset,0,0],[offset+1,0,0],[offset,1,0],[999,999,999]], dtype="<f4")
                p = writer.array(positions)
                indices = writer.array(np.array([[0],[1],[2]], dtype="<u4"), {"componentType":5125,"type":"SCALAR","count":3}, 34963)
                writer.doc["meshes"].append({"primitives": [{"attributes":{"POSITION":p},"indices":indices,"material":0}]})
                writer.doc["nodes"].append({"mesh":len(writer.doc["meshes"])-1})
                writer.doc["scenes"][0]["nodes"].append(len(writer.doc["nodes"])-1)
            source = folder/"source.glb"; writer.save(source)
            result = split(source, folder/"parts", folder/"parts.json", max_bytes=100)
            self.assertEqual(result["mesh_count"], 2); self.assertEqual(result["face_count"], 2)
            self.assertEqual(result["world_bounds_m"], [[0.,0.,0.],[11.,1.,0.]])
            self.assertEqual(result["part_count"], 2)
            for part in result["import_parts"]:
                doc, binary, _ = read_glb(folder/part["file"])
                primitive = doc["meshes"][0]["primitives"][0]
                positions, _ = accessor(doc,binary,primitive["attributes"]["POSITION"])
                self.assertEqual(len(positions),3)
                view=doc["bufferViews"][doc["images"][0]["bufferView"]]; start=view["byteOffset"]
                self.assertEqual(bytes(binary[start:start+view["byteLength"]]),texture_bytes)


if __name__ == "__main__":
    unittest.main()
