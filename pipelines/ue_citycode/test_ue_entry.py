"""Regression for UE 5.0's observed primary-only AssetImportTask result.

No UE installation is required. The stub reproduces the actual Windows failure:
the factory creates two meshes while the task lists only the first one.
"""
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class ObjectProperties:
    def __init__(self):
        self.properties = {}

    def set_editor_property(self, key, value):
        self.properties[key] = value

    def get_editor_property(self, key):
        return self.properties[key]


class Mesh:
    def __init__(self, name):
        self.path = "/Game/Research/CityCodeOpenSource/Imported/MultiHash/" + name

    def get_path_name(self):
        return self.path


class ImportTests(unittest.TestCase):
    def load_entry(self, fake):
        spec = importlib.util.spec_from_file_location("ue_entry_regression", Path(__file__).with_name("ue_entry.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"unreal": fake}):
            spec.loader.exec_module(module)
        return module

    def test_primary_task_plus_full_destination_preserves_two_mesh_instances(self):
        first, second = Mesh("0_node_0001_mesh_0000"), Mesh("1_node_0004_mesh_0001")
        objects = {"primary": first, "secondary": second, "material": object()}
        state = {"imported": False, "factory_calls": 0, "metadata": {}}
        def import_tasks(tasks):
            state["imported"] = True; state["factory_calls"] += 1
            tasks[0].properties["imported_object_paths"] = ["primary"]
        library = types.SimpleNamespace(
            list_assets=lambda *args, **kwargs: ["primary", "secondary", "material"] if state["imported"] else [],
            make_directory=lambda *args: True,
            set_metadata_tag=lambda mesh, name, value: state["metadata"].update({mesh.path: value}),
            get_metadata_tag=lambda mesh, name: state["metadata"].get(mesh.path))
        fake = types.SimpleNamespace(StaticMesh=Mesh, EditorAssetLibrary=library, load_asset=objects.get,
            AssetImportTask=ObjectProperties, GLTFImportFactory=ObjectProperties, GLTFImportOptions=ObjectProperties,
            AssetToolsHelpers=types.SimpleNamespace(get_asset_tools=lambda: types.SimpleNamespace(import_asset_tasks=import_tasks)))
        module = self.load_entry(fake)
        destination = "/Game/Research/CityCodeOpenSource/Imported/MultiHash"
        result = module.import_meshes("not-read-after-prevalidated-hash", destination, "a"*64, hash_verified=True)
        self.assertEqual({mesh.path for mesh in result}, {first.path, second.path})
        self.assertEqual(state["metadata"], {first.path:"a"*64, second.path:"a"*64})
        # A subsequent pass must reuse both verified meshes without a new import.
        reused = module.import_meshes("not-read-after-prevalidated-hash", destination, "a"*64, hash_verified=True)
        self.assertEqual({mesh.path for mesh in reused}, {first.path, second.path})
        self.assertEqual(state["factory_calls"], 1)

    def test_partial_upload_cannot_be_consumed_as_a_complete_asset(self):
        module = self.load_entry(types.SimpleNamespace())
        complete = b"complete-upload-bytes"
        part = {"bytes":len(complete), "sha256":hashlib.sha256(complete).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"incoming.glb"
            self.assertFalse(module.file_ready(path, part)[0])
            path.write_bytes(complete[:6]); self.assertFalse(module.file_ready(path, part)[0])
            path.write_bytes(b"x"*len(complete)); self.assertFalse(module.file_ready(path, part)[0])
            path.write_bytes(complete); self.assertEqual(module.file_ready(path, part), (True, None))

    def test_checkpoint_resume_skips_saved_mesh_loading_and_imports_only_missing_part(self):
        loaded, imported = [], []
        def forbidden_load(path):
            loaded.append(path)
            raise AssertionError("Resume must not load saved UObjects")
        fake = types.SimpleNamespace(load_asset=forbidden_load, StaticMeshEditorSubsystem=object,
            get_editor_subsystem=lambda cls: object(),
            SystemLibrary=types.SimpleNamespace(collect_garbage=lambda: None),
            EditorAssetLibrary=types.SimpleNamespace(save_directory=lambda *args, **kwargs: True))
        module = self.load_entry(fake)
        rows = [{"id": identity, "kind": "building", "nanite": True, "import_file": identity+".glb",
                 "prepared": {"interchange_sha256": value*64, "interchange_bytes": 20, "mesh_count": 1}}
                for identity, value in (("SAVED", "a"), ("MISSING", "b"))]
        saved_path = module.shared_destination("SAVED", "a"*64) + "/Mesh.Mesh"
        fake.EditorAssetLibrary.does_asset_exist = lambda path: path == saved_path
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"schema":"citycode.ue_citycode_assembly.v1", "meshes":rows}))
            prior = {"schema":"citycode.ue_available_import_checkpoint.v1", "status":"assets_imported_partial",
                "assembly_manifest_sha256":module.digest_file(manifest), "imports":[{
                    "input_id":"SAVED", "file":"SAVED.glb", "sha256":"a"*64,
                    "mesh_paths":[saved_path], "nanite_requested":True}]}
            backup = root/"read_only_backup.json"; backup.write_text(json.dumps(prior))
            original_bytes = backup.read_bytes()
            output = root/"new_checkpoint.json"
            def import_missing(filename, destination, *args, **kwargs):
                imported.append(Path(filename).name)
                self.assertEqual(Path(filename).name, "MISSING.glb")
                self.assertTrue(kwargs["require_materials"])
                return [types.SimpleNamespace(get_path_name=lambda: destination+"/Mesh.Mesh")]
            with patch.object(module, "preflight", return_value={"ready":True, "runtime_generate_mesh_distance_fields":0}), \
                 patch.object(module, "file_ready", return_value=(True,None)), \
                 patch.object(module, "configure_nanite"), patch.object(module, "import_meshes", side_effect=import_missing):
                result = module.import_available({"assembly_manifest":str(manifest), "output_json":str(output),
                    "resume_checkpoint_file":str(backup), "require_distance_fields_disabled":True})
            self.assertEqual(result["status"], "assets_imported_complete")
            self.assertEqual(imported, ["MISSING.glb"]); self.assertEqual(loaded, [])
            self.assertEqual(result["resume_checkpoint"]["accepted_parts"], 1)
            self.assertEqual(result["resume_checkpoint"]["rejected_parts"], 0)
            self.assertEqual(result["imports"][0]["mesh_count"], 1) # Legacy checkpoint path count.
            self.assertTrue(result["imports"][0]["resumed_from_checkpoint"])
            self.assertEqual(backup.read_bytes(), original_bytes)
            self.assertEqual(json.loads(output.read_text())["imported_parts"], 2)

    def test_resume_rejects_changed_manifest_and_same_output_before_registry_access(self):
        fake = types.SimpleNamespace(EditorAssetLibrary=types.SimpleNamespace(
            does_asset_exist=lambda path: self.fail("Invalid recovery source must not access assets")))
        module = self.load_entry(fake)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/"backup.json"
            source.write_text(json.dumps({"schema":"citycode.ue_available_import_checkpoint.v1",
                "status":"assets_imported_partial", "assembly_manifest_sha256":"old", "imports":[]}))
            with self.assertRaisesRegex(ValueError, "different immutable"):
                module.resume_completed({"resume_checkpoint_file":str(source), "output_json":str(source.with_name("new.json"))}, [], "new")
            with self.assertRaisesRegex(ValueError, "separate read-only backup"):
                module.resume_completed({"resume_checkpoint_file":str(source), "output_json":str(source)}, [], "old")

    def test_resume_accepts_only_matching_assigned_counts_paths_and_existing_assets(self):
        checked = []
        fake = types.SimpleNamespace(EditorAssetLibrary=types.SimpleNamespace(
            does_asset_exist=lambda path: checked.append(path) is None and "absent" not in path))
        module = self.load_entry(fake)
        rows, prior_rows = [], []
        for identity in ("VALID", "FILE", "COUNT", "PREFIX", "ABSENT", "SHA", "DUPPATH"):
            part_sha = hashlib.sha256(identity.encode()).hexdigest()
            rows.append({"id":identity, "kind":"building", "import_file":identity+".glb",
                "prepared":{"interchange_sha256":part_sha, "interchange_bytes":1, "mesh_count":2}})
            prefix = module.shared_destination(identity, part_sha)
            prior_rows.append({"input_id":identity, "file":identity+".glb", "sha256":part_sha,
                "mesh_count":2, "mesh_paths":[prefix+"/first.first",prefix+"/second.second"], "nanite_requested":False})
        prior_rows[1]["file"] = "wrong.glb"
        prior_rows[2]["mesh_count"] = 1
        prior_rows[3]["mesh_paths"][0] = "/Game/Other/first.first"
        prior_rows[4]["mesh_paths"][0] += "_absent"
        prior_rows[5]["sha256"] = "wrong"
        prior_rows[6]["mesh_paths"][1] = prior_rows[6]["mesh_paths"][0]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/"backup.json"
            source.write_text(json.dumps({"schema":"citycode.ue_available_import_checkpoint.v1",
                "status":"assets_imported_partial", "assembly_manifest_sha256":"frozen", "imports":prior_rows}))
            completed, report = module.resume_completed({"resume_checkpoint_file":str(source),
                "output_json":str(source.with_name("new.json"))}, rows, "frozen")
        self.assertEqual(len(completed), 1)
        self.assertEqual(next(iter(completed.values()))["input_id"], "VALID")
        self.assertEqual(report["rejected_parts"], 6)
        self.assertEqual({row["reason"] for row in report["rejected"]}, {
            "interchange_file_mismatch", "mesh_count_mismatch", "mesh_outside_shared_digest_destination",
            "saved_asset_missing_from_registry", "input_or_sha_not_assigned"})
        self.assertEqual(len(checked), 3) # Two valid meshes and the first absent one.

    def test_lighting_saves_only_when_all_tagged_geometry_is_unchanged(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                module = self.load_entry(types.SimpleNamespace())
                manifest = root/"manifest.json"; assembled = root/"assembled.json"; audited = root/"audited.json"
                manifest.write_text(json.dumps({"schema":"citycode.ue_citycode_assembly.v1", "scene_id":"scene"}))
                level = module.CONTENT_ROOT+"/Maps/Verified"
                assembled.write_text(json.dumps({"status":"assembled_pending_independent_readback", "ue_executed":True,
                    "scene_id":"scene", "new_level_path":level, "assembly_manifest_sha256":module.digest_file(manifest)}))
                audited.write_text(json.dumps({"status":"passed", "errors":[], "ue_executed":True,
                    "independent_saved_map_readback":True, "scene_id":"scene", "new_level_path":level,
                    "assembly_manifest_sha256":module.digest_file(manifest), "assembly_result_sha256":module.digest_file(assembled),
                    "actual_tagged_actor_count":1, "building_entity_root_count":1}))
                fake = types.SimpleNamespace(EditorLevelLibrary=types.SimpleNamespace(
                    load_level=lambda path: path == level, save_current_level=lambda: True))
                module.unreal = fake
                before = {"actor":{"mesh_path":"mesh", "position_cm":[1,2,3]}, "root":{"mesh_path":None}}
                after = {"actor":{"mesh_path":"mesh", "position_cm":[1,2,4]}, "root":{"mesh_path":None}} if changed else before
                with patch.object(module, "tagged_geometry_snapshot", side_effect=[before,after]), \
                     patch.object(module, "inspection_daylight", return_value={}), \
                     patch.object(fake.EditorLevelLibrary, "save_current_level", return_value=True) as saved:
                    job = {"assembly_manifest":str(manifest), "assembly_result":str(assembled), "readback_result":str(audited)}
                    if changed:
                        with self.assertRaisesRegex(ValueError, "geometry; the map was not saved"):
                            module.configure_inspection_lighting(job)
                        saved.assert_not_called()
                    else:
                        result = module.configure_inspection_lighting(job)
                        saved.assert_called_once()
                        self.assertTrue(result["geometry_unchanged"])
                        self.assertEqual(result["geometry_snapshot_sha256_before"], result["geometry_snapshot_sha256_after"])
                        self.assertEqual(result["status"], "configured_pending_independent_readback")

    def test_formal_cache_rejects_missing_materials_but_fixture_can_opt_out(self):
        mesh = Mesh("broken")
        slot = ObjectProperties(); slot.properties["material_interface"] = None
        mesh.get_editor_property = lambda name: [slot]
        fake = types.SimpleNamespace(StaticMesh=Mesh, load_asset=lambda path: mesh,
            EditorAssetLibrary=types.SimpleNamespace(list_assets=lambda *args, **kwargs: [mesh.path],
                get_metadata_tag=lambda *args: "a"*64))
        module = self.load_entry(fake)
        with self.assertRaisesRegex(ValueError, "missing material slots"):
            module.import_meshes("unused", "/Game/Research/CityCodeOpenSource/Imported/MultiHash", "a"*64,
                                 hash_verified=True, require_materials=True)
        self.assertEqual(module.import_meshes("unused", "/Game/Research/CityCodeOpenSource/Imported/MultiHash",
                                             "a"*64, hash_verified=True), [mesh])

    def test_nanite_usage_persistence_is_scoped_deduplicated_and_idempotent(self):
        changed, saved = [], []
        class Material:
            def __init__(self, path, usage=False): self.path, self.usage = path, usage
            def get_path_name(self): return self.path
        class MaterialMesh:
            def __init__(self, materials): self.materials = materials
            def get_editor_property(self, name):
                result = []
                for material in self.materials:
                    slot = ObjectProperties(); slot.properties["material_interface"] = material; result.append(slot)
                return result
        pending = Material("/Game/Research/CityCodeOpenSource/Imported/Asset/Material.Material")
        existing = Material("/Game/Research/CityCodeOpenSource/MaterialRecovery/Asset/Other.Other", True)
        epic = Material("/Game/Road/Native.Native")
        def set_usage(material, usage): changed.append(material.path); material.usage = True
        def save_material(material, **kwargs):
            self.assertFalse(kwargs["only_if_is_dirty"]); saved.append(material.path); return True
        fake = types.SimpleNamespace(Material=Material, MaterialUsage=types.SimpleNamespace(MATUSAGE_NANITE="nanite"),
            MaterialEditingLibrary=types.SimpleNamespace(has_material_usage=lambda material, usage: material.usage,
                set_material_usage=set_usage), EditorAssetLibrary=types.SimpleNamespace(save_loaded_asset=save_material))
        module = self.load_entry(fake)
        meshes = [MaterialMesh([pending,existing,epic]), MaterialMesh([pending])]
        settings = ObjectProperties(); settings.properties["enabled"] = True
        subsystem = types.SimpleNamespace(get_nanite_settings=lambda mesh: settings,
            set_nanite_settings=lambda *args: self.fail("Cache repair must not rebuild enabled Nanite meshes"))
        first = module.configure_nanite(meshes, True, subsystem)
        self.assertEqual(first["scoped_material_count"], 2)
        self.assertEqual(first["modified_count"], 1); self.assertEqual(first["already_present_count"], 1)
        self.assertEqual(first["excluded_count"], 1)
        self.assertEqual(changed, [pending.path]); self.assertEqual(saved, [pending.path])
        second = module.persist_mesh_nanite_material_usage(meshes)
        self.assertEqual(second["modified_count"], 0); self.assertEqual(second["already_present_count"], 2)
        self.assertEqual(changed, [pending.path]); self.assertEqual(saved, [pending.path])
        self.assertFalse(epic.usage)
        # Loading can auto-set an in-memory flag without dirtying its package.
        # Maintenance must persist even these already-present scoped flags.
        forced = module.persist_mesh_nanite_material_usage(meshes, force_save_existing=True)
        self.assertEqual(forced["modified_count"], 0)
        self.assertEqual(forced["saved_material_count"], 2)
        self.assertTrue(all(row["material_package_saved"] for row in forced["records"]))
        self.assertEqual(changed, [pending.path])
        self.assertEqual(set(saved[1:]), {pending.path, existing.path})

    def test_nanite_usage_maintenance_preserves_tagged_map_and_rejects_unbound_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); module = self.load_entry(types.SimpleNamespace())
            manifest = root/"manifest.json"; assembled = root/"assembled.json"; audited = root/"audited.json"
            manifest.write_text(json.dumps({"schema":"citycode.ue_citycode_assembly.v1", "scene_id":"scene",
                "meshes":[{"id":"BUILDING", "kind":"building"}, {"id":"BACKGROUND", "kind":"background"}]}))
            level = module.CONTENT_ROOT+"/Maps/Verified"
            assembled.write_text(json.dumps({"status":"assembled_pending_independent_readback", "ue_executed":True,
                "scene_id":"scene", "new_level_path":level, "assembly_manifest_sha256":module.digest_file(manifest)}))
            audit = {"status":"passed", "errors":[], "ue_executed":True, "independent_saved_map_readback":True,
                "scene_id":"scene", "new_level_path":level, "assembly_manifest_sha256":module.digest_file(manifest),
                "assembly_result_sha256":module.digest_file(assembled), "actual_tagged_actor_count":2,"building_entity_root_count":1}
            audited.write_text(json.dumps(audit))
            mesh = Mesh("geometry")
            settings = ObjectProperties(); settings.properties["enabled"] = True
            loaded = []
            def load_mesh(path): loaded.append(path); self.assertEqual(path, mesh.path); return mesh
            fake = types.SimpleNamespace(StaticMesh=Mesh, StaticMeshEditorSubsystem=object, load_asset=load_mesh,
                get_editor_subsystem=lambda cls: types.SimpleNamespace(get_nanite_settings=lambda value: settings),
                EditorLevelLibrary=types.SimpleNamespace(load_level=lambda path: path == level,
                    save_current_level=lambda: self.fail("Material maintenance must not save the map")))
            module.unreal = fake
            before = {"actor":{"mesh_path":mesh.path,"tags":["CityCodeInput=BUILDING","CityCodeScene=scene"]},
                "background":{"mesh_path":"background","tags":["CityCodeInput=BACKGROUND","CityCodeScene=scene"]},
                "root":{"mesh_path":None,"tags":["CityCodeEntityRoot=entity"]}}
            usage = {"scoped_material_count":1,"modified_count":1,"already_present_count":0,"excluded_count":0,"excluded":[],"records":[]}
            job = {"assembly_manifest":str(manifest),"assembly_result":str(assembled),"readback_result":str(audited)}
            with patch.object(module,"tagged_geometry_snapshot",side_effect=[before,before]), \
                 patch.object(module,"mesh_repair_snapshot",return_value={"unchanged":True}), \
                 patch.object(module,"persist_mesh_nanite_material_usage",return_value=usage) as persist:
                result = module.persist_nanite_material_usage(job)
                persist.assert_called_once_with([mesh], force_save_existing=True)
            self.assertEqual(loaded, [mesh.path]); self.assertTrue(result["geometry_unchanged"])
            self.assertTrue(result["nanite_settings_unchanged"]); self.assertFalse(result["map_modified"])
            self.assertEqual(result["eligible_nanite_mesh_count"], 1)
            audit["assembly_manifest_sha256"] = "other"; audited.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,"exact verified"):
                module.persist_nanite_material_usage(job)
            self.assertEqual(loaded,[mesh.path])


class NormalPolicyTests(unittest.TestCase):
    def source_glb(self, path, normals):
        meshes = [{"name":"node_%04d_mesh_%04d" % (i,i), "primitives":[
            {"attributes":{"POSITION":0, **({"NORMAL":1} if present else {})}} for present in primitive_normals]}
            for i,primitive_normals in enumerate(normals)]
        data = json.dumps({"meshes":meshes}).encode(); data += b" " * (-len(data) % 4)
        path.write_bytes(struct.pack("<4sIII4s",b"glTF",2,28+len(data),len(data),b"JSON")+data+struct.pack("<I4s",0,b"BIN\0"))
        return meshes

    def setup_normal_meshes(self, module, prefix, count):
        states, events = {}, []
        meshes = []
        for i in range(count):
            mesh = Mesh("%d_node_%04d_mesh_%04d" % (i,i,i)); mesh.path = prefix+"/"+mesh.path.rsplit("/",1)[-1]
            meshes.append(mesh); states[mesh.path] = {"recompute_normals":False,"recompute_tangents":False,"use_full_precision_u_vs":False}
        def get_options(mesh, index):
            self.assertEqual(index,0); options = ObjectProperties(); options.properties = dict(states[mesh.path]); return options
        def set_options(mesh, index, options):
            self.assertEqual(index,0); events.append(mesh.path); states[mesh.path] = dict(options.properties)
        subsystem = types.SimpleNamespace(get_lod_build_settings=get_options,set_lod_build_settings=set_options)
        return meshes, states, events, subsystem

    def test_missing_normal_policy_preserves_explicit_normals_and_other_build_options(self):
        module = ImportTests().load_entry(types.SimpleNamespace())
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/"input.glb"; self.source_glb(source, [[True],[False],[True,False]])
            meshes,states,events,subsystem = self.setup_normal_meshes(module,module.CONTENT_ROOT+"/Imported/Input",3)
            with patch.object(module,"mesh_repair_snapshot",return_value={"unchanged":True}), \
                 patch.object(module,"material_paths",return_value=["same-material"]):
                rows = module.configure_source_normal_policy(source,meshes,subsystem)
                self.assertEqual(events,[meshes[1].path,meshes[2].path])
                self.assertFalse(states[meshes[0].path]["recompute_normals"])
                self.assertTrue(states[meshes[1].path]["recompute_normals"])
                self.assertTrue(states[meshes[2].path]["recompute_normals"])
                self.assertTrue(all(not state["recompute_tangents"] and not state["use_full_precision_u_vs"] for state in states.values()))
                self.assertEqual(rows[2]["missing_normal_primitives"],[1])
                module.configure_source_normal_policy(source,meshes,subsystem)
                self.assertEqual(len(events),2) # Idempotent cache policy does not rebuild again.
                meshes[0].path = "/Game/Road/" + meshes[0].path.rsplit("/",1)[-1]
                with self.assertRaisesRegex(ValueError,"cannot be matched|own generated"):
                    module.configure_source_normal_policy(source,meshes,subsystem)
                self.assertEqual(len(events),2)

    def test_verified_cache_applies_source_normal_policy_without_reimport(self):
        mesh = Mesh("0_node_0000_mesh_0000")
        module = ImportTests().load_entry(types.SimpleNamespace(StaticMesh=Mesh, load_asset=lambda path:mesh,
            EditorAssetLibrary=types.SimpleNamespace(list_assets=lambda *args,**kwargs:[mesh.path],get_metadata_tag=lambda *args:"a"*64)))
        with patch.object(module,"configure_source_normal_policy") as applied:
            actual = module.import_meshes("input.glb", module.CONTENT_ROOT+"/Imported/Input", "a"*64,
                                         hash_verified=True, configure_missing_normals=True)
        self.assertEqual(actual,[mesh]); applied.assert_called_once_with("input.glb",[mesh])

    def test_background_maintenance_only_saves_own_missing_normal_mesh_and_preserves_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); module = ImportTests().load_entry(types.SimpleNamespace())
            source = root/"background.glb"; self.source_glb(source,[[False]])
            sha = module.digest_file(source); destination = module.shared_destination("BACKGROUND",sha)
            meshes,states,events,subsystem = self.setup_normal_meshes(module,destination,1); mesh = meshes[0]
            saved=[]
            fake=types.SimpleNamespace(StaticMesh=Mesh,StaticMeshEditorSubsystem=object,load_asset=lambda path:mesh,
                get_editor_subsystem=lambda cls:subsystem,
                EditorLevelLibrary=types.SimpleNamespace(load_level=lambda path:True,
                    save_current_level=lambda:self.fail("Normal maintenance must not save a map")),
                EditorAssetLibrary=types.SimpleNamespace(get_metadata_tag=lambda *args:sha,
                    save_loaded_asset=lambda asset,**kwargs:saved.append((asset.path,kwargs)) is None),
                log=lambda message:None)
            module.unreal=fake
            manifest=root/"manifest.json"; assembly=root/"assembly.json"; audit=root/"audit.json"
            manifest.write_text(json.dumps({"schema":"citycode.ue_citycode_assembly.v1","scene_id":"scene","meshes":[
                {"id":"BACKGROUND","kind":"background","import_file":source.name,
                 "prepared":{"mesh_count":1,"interchange_sha256":sha,"interchange_bytes":source.stat().st_size}},
                {"id":"BUILDING","kind":"building"}]}))
            level=module.CONTENT_ROOT+"/Maps/Verified"
            assembly.write_text(json.dumps({"status":"assembled_pending_independent_readback","ue_executed":True,
                "scene_id":"scene","new_level_path":level,"assembly_manifest_sha256":module.digest_file(manifest)}))
            audit.write_text(json.dumps({"status":"passed","errors":[],"ue_executed":True,"independent_saved_map_readback":True,
                "scene_id":"scene","new_level_path":level,"assembly_manifest_sha256":module.digest_file(manifest),
                "assembly_result_sha256":module.digest_file(assembly),"actual_tagged_actor_count":2,"building_entity_root_count":0}))
            before={"bg":{"mesh_path":mesh.path,"tags":["CityCodeInput=BACKGROUND","CityCodeScene=scene"]},
                "building":{"mesh_path":"never-loaded","tags":["CityCodeInput=BUILDING","CityCodeScene=scene"]}}
            job={"assembly_manifest":str(manifest),"assembly_result":str(assembly),"readback_result":str(audit),"output_json":str(root/"repair.json")}
            original_manifest=manifest.read_bytes(); original_source=source.read_bytes()
            with patch.object(module,"tagged_geometry_snapshot",return_value=before), \
                 patch.object(module,"mesh_repair_snapshot",return_value={"unchanged":True}), \
                 patch.object(module,"material_paths",return_value=["same-material"]):
                result=module.repair_background_normals(job)
            self.assertEqual(result["status"],"background_normals_repaired_pending_independent_readback")
            self.assertEqual(result["saved_mesh_count"],1);self.assertEqual(result["modified_count"],1)
            self.assertEqual(saved,[(mesh.path,{"only_if_is_dirty":False})]);self.assertEqual(events,[mesh.path])
            self.assertTrue(result["geometry_unchanged"]);self.assertFalse(result["map_modified"])
            self.assertEqual(manifest.read_bytes(),original_manifest);self.assertEqual(source.read_bytes(),original_source)
            # A changed source must fail before any second rebuild or package save.
            source.write_bytes(b"changed")
            with patch.object(module,"tagged_geometry_snapshot",return_value=before):
                with self.assertRaisesRegex(ValueError,"source is missing or changed"): module.repair_background_normals(job)
            self.assertEqual(len(saved),1)


class MaterialRepairTests(unittest.TestCase):
    def setup_repair(self, root):
        events = []
        module = ImportTests().load_entry(types.SimpleNamespace())
        source = root/"source.glb"; source.write_bytes(b"immutable-material-and-geometry-input")
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        identity = "BUILDING_1"; destination = module.shared_destination(identity, sha)
        recovery = module.CONTENT_ROOT+"/MaterialRecovery/"+identity+"_"+sha[:12]
        class Material:
            def __init__(self, path): self.path = path; self.nanite_usage = False
            def get_path_name(self): return self.path
        class RepairMesh:
            def __init__(self, path, material):
                self.path = path; self.material = material
            def get_path_name(self): return self.path
            def get_name(self): return "0_node_mesh"
            def get_editor_property(self, name):
                slot = ObjectProperties(); slot.properties["material_interface"] = self.material
                return [slot]
            def set_material(self, index, material):
                if "saved_dependencies" not in events: raise AssertionError("Must save dependencies before assigning references")
                events.append("assign_original_material"); self.material = material
        original = RepairMesh(destination+"/0_node_mesh.0_node_mesh", None)
        fresh = RepairMesh(recovery+"/0_node_mesh.0_node_mesh", Material(recovery+"/Materials/Recovered.Recovered"))
        settings = ObjectProperties(); settings.properties["enabled"] = True
        def save_directory(path, **kwargs):
            self.assertEqual(path, recovery); self.assertTrue(kwargs["recursive"])
            self.assertFalse(kwargs["only_if_is_dirty"])
            events.append("saved_dependencies"); return True
        def save_original(mesh, **kwargs):
            if isinstance(mesh, Material):
                self.assertIs(mesh, fresh.material); self.assertTrue(mesh.nanite_usage)
                self.assertFalse(kwargs["only_if_is_dirty"])
                events.append("saved_material_usage"); return True
            self.assertIs(mesh, original); events.append("saved_original"); return True
        fake = types.SimpleNamespace(StaticMesh=RepairMesh, StaticMeshEditorSubsystem=object, Material=Material,
            MaterialUsage=types.SimpleNamespace(MATUSAGE_NANITE="nanite"),
            MaterialEditingLibrary=types.SimpleNamespace(has_material_usage=lambda mat, usage: mat.nanite_usage,
                set_material_usage=lambda mat, usage: setattr(mat, "nanite_usage", True)),
            load_asset=lambda path: original if path == original.path else None,
            get_editor_subsystem=lambda cls: types.SimpleNamespace(get_nanite_settings=lambda mesh: settings),
            SystemLibrary=types.SimpleNamespace(collect_garbage=lambda: None),
            EditorAssetLibrary=types.SimpleNamespace(get_metadata_tag=lambda *args: sha,
                save_directory=save_directory, save_loaded_asset=save_original))
        module.unreal = fake
        row = {"id":identity, "kind":"building", "nanite":True, "import_file":source.name,
               "prepared":{"interchange_sha256":sha, "interchange_bytes":source.stat().st_size, "mesh_count":1}}
        manifest = root/"manifest.json"
        manifest.write_text(json.dumps({"schema":"citycode.ue_citycode_assembly.v1", "scene_id":"scene", "meshes":[row]}))
        audit = {"schema":"citycode.ue_independent_readback.v1", "independent_saved_map_readback":True,
            "ue_executed":True, "status":"failed", "assembly_manifest_sha256":module.digest_file(manifest), "scene_id":"scene",
            "new_level_path":module.CONTENT_ROOT+"/Maps/Existing", "errors":[{"input_id":identity,"issues":["missing_material_slot"]}],
            "records":[{"input_id":identity,"mesh_paths":[original.path]}]}
        audit_path = root/"readback.json"; audit_path.write_text(json.dumps(audit))
        job = {"assembly_manifest":str(manifest), "readback_result":str(audit_path), "output_json":str(root/"repair.json")}
        return module, job, audit_path, audit, original, fresh, events

    def test_repair_saves_fresh_dependencies_then_only_changes_original_materials(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, job, audit_path, audit, original, fresh, events = self.setup_repair(Path(temporary))
            with patch.object(module, "import_meshes", return_value=[fresh]) as imported, \
                 patch.object(module, "bounds", return_value=[[0.,0.,0.],[100.,200.,300.]]):
                result = module.repair_missing_materials(job)
            self.assertEqual(result["status"], "materials_repaired_pending_independent_readback")
            self.assertEqual(events, ["saved_dependencies", "assign_original_material", "saved_material_usage", "saved_original"])
            self.assertIs(original.material, fresh.material)
            self.assertEqual(result["records"][0]["before"], result["records"][0]["after"])
            self.assertTrue(imported.call_args.kwargs["require_materials"])
            self.assertFalse(result["map_modified"]); self.assertFalse(result["geometry_replaced"])
            self.assertEqual(json.loads(audit_path.read_text()), audit)

    def test_repair_rejects_changed_source_and_wrong_identity_before_any_asset_mutation(self):
        for mutation, message in (("manifest", "exact frozen"), ("identity", "identity/count"), ("source", "missing or changed")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                module, job, audit_path, audit, original, fresh, events = self.setup_repair(Path(temporary))
                if mutation == "manifest": audit["assembly_manifest_sha256"] = "other"
                elif mutation == "identity": audit["records"][0]["mesh_paths"] = ["/Game/Other/Unexpected.Unexpected"]
                else: (Path(temporary)/"source.glb").write_bytes(b"changed")
                audit_path.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, message): module.repair_missing_materials(job)
                self.assertEqual(events, [])

    def test_repair_refuses_geometry_or_transform_readback_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, job, audit_path, audit, original, fresh, events = self.setup_repair(Path(temporary))
            audit["errors"][0]["issues"].append("world_bounds_error_over_tolerance")
            audit_path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "other than missing_material_slot"):
                module.repair_missing_materials(job)
            self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
