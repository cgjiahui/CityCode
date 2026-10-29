"""Conversion contracts using tiny synthetic files; no GPU, API or UE."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from citycode_core.scene import (compile_scene, export_ue, inside, load_scene,
    matrix, read, rebase_provenance, sha, validate, validate_layers, write)


class CityCodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'release'; self.root.mkdir()
        self.out = self.root/'data/citycode'
        self.config = self.fixture()

    def fixture(self):
        r = self.root; (r/'data/assets').mkdir(parents=True); (r/'build/interchange').mkdir(parents=True)
        for path in ('data/assets/model.glb', 'build/interchange/model.glb', 'build/interchange/axis.glb'):
            (r/path).write_bytes(path.encode())
        structures = {'features': [{'properties': dict(predicted_global_building_id=i, base_z_m=2, top_z_m=12,
            height_m=10, height_method='observed', member_observation_ids=['view1']),
            'geometry': {'type': 'Polygon', 'coordinates': [[[0,0],[2,0],[2,2],[0,0]]]}} for i in ('S1','S2','S3')]}
        write(r/'data/structures.json', structures)
        # Full affine matrix with shear, reflection and independent axis scales.
        transform = [[2,0.2,0,11],[0,3,0,12],[0,0,-4,13],[0,0,0,1]]
        lod = {'path': 'assets/model.glb', 'sha256': sha(r/'data/assets/model.glb'), 'bytes': (r/'data/assets/model.glb').stat().st_size, 'faces': 12}
        rows = [dict(id=i, source_building_ids=[source], world_from_local_4x4=transform,
            lods={'0.25M': lod, '0.1M': lod}, building_entity_id='E1')
            for i, source in [('P1','S1'),('P2','S1'),('MASSING_P3','S2'),('MASSING_P4','S2')]]
        binding = dict(source_structure=dict(instance_count=3, footprints_sha256=sha(r/'data/structures.json')),
            asset_instances=rows[:2], massing_instances=rows[2:],
            building_entities=[dict(id='E1', asset_instance_ids=['P1','P2'], source_building_ids=['S1'])])
        write(r/'data/binding.json', binding)
        original = deepcopy(binding); original['asset_instances'][0]['world_from_local_4x4'] = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        write(r/'data/original.json', original)
        native = [dict(id='PROP1', kind='parked_car', mesh_path='/Game/Cars/Car.Car',
            position_cm=[123.5,-255,17], rotation_quat_xyzw=[0,0,0,1], scale=[1.2,.8,1], group_id='G1')]
        catalog = {'assets': [{'mesh_path': native[0]['mesh_path']}]}; write(r/'data/catalog.json', catalog)
        meshes = [dict(id=row['id'], kind='massing' if row['id'].startswith('MASSING') else 'building',
            source_file='data/assets/model.glb', source_sha256=lod['sha256'], source_bytes=lod['bytes'],
            world_from_scene_4x4=transform, anchor_m=[11,12,13], import_file='interchange/model.glb',
            prepared={'interchange_sha256': sha(r/'build/interchange/model.glb'), 'interchange_bytes': (r/'build/interchange/model.glb').stat().st_size}) for row in rows]
        assembly = dict(schema='citycode.ue_citycode_assembly.v1', building_lod='0.25M', scene_id='tiny',
            source_binding='data/binding.json', source_binding_sha256=sha(r/'data/binding.json'), catalog_sha256=sha(r/'data/catalog.json'),
            meshes=meshes, placements=native, calibration={'path': 'interchange/axis.glb', 'sha256': sha(r/'build/interchange/axis.glb')})
        write(r/'build/assembly.json', assembly)
        documents = {'background': {}, 'ground_roads': {'nodes':[{'node_id':'N1'},{'node_id':'N2'}], 'edges':[{'edge_id':'R1','source':'N1','target':'N2','height_source':'display_reference_plane_not_measurement'}]},
            'elevated_centerlines': {'features':[{'properties':{'segment_id':'SPAN1'},'geometry':{'type':'LineString','coordinates':[[0,0,6],[2,0,6]]}}]},
            'elevated_proposal': {}, 'network': {}, 'props_groups': {'groups':[{'group_id':'G1','part_count':1}]}}
        config = dict(release_id='test', binding='data/binding.json', original_binding='data/original.json', structures='data/structures.json',
                      assembly='build/assembly.json', catalog='data/catalog.json')
        for key, value in documents.items():
            config[key] = 'data/'+key+'.json'; write(r/config[key], value)
        return config

    def compile(self):
        return compile_scene(self.root, self.config, self.out)

    def test_many_to_many_unbound_and_original_transform(self):
        m = self.compile(); _, layers = load_scene(self.out)
        self.assertEqual(m['counts']['structure_sources'], 3)
        self.assertEqual(m['counts']['total_building_placements'], 4)
        self.assertEqual(m['counts']['unbound_structure_sources'], 1)
        self.assertEqual(layers['relations']['source_to_instances']['S2'], ['MASSING_P3','MASSING_P4'])
        self.assertTrue(layers['assembly']['buildings'][0]['transform_changed'])
        self.assertEqual(layers['roads']['ground']['edges'][0]['height_source'], 'display_reference_plane_not_measurement')

    def test_roundtrip_preserves_all_geometry_transforms_and_native_trs(self):
        self.compile(); output = self.root/'build/new/assembly.json'; export_ue(self.out, self.root, output)
        actual = read(output); original = read(self.root/'build/assembly.json')
        self.assertEqual(actual['placements'], original['placements'])
        for got, expected in zip(actual['meshes'], original['meshes']):
            self.assertEqual((output.parent/got.pop('import_file')).resolve(), (self.root/'build'/expected.pop('import_file')).resolve())
            self.assertEqual(got, expected)

    def test_portable_after_moving_release(self):
        self.compile(); moved = Path(self.temp.name)/'moved'; self.root.rename(moved)
        result = validate(moved/'data/citycode', moved, verify_files=True)
        self.assertEqual(result['status'], 'passed')
        export_ue(moved/'data/citycode', moved, moved/'build/new/assembly.json')

    def test_changed_same_size_model_is_detected(self):
        self.compile(); p = self.root/'data/assets/model.glb'; p.write_bytes(b'x'*p.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate(self.out, self.root, verify_files=True)

    def test_mismatching_frozen_input_stops_compilation(self):
        p = self.root/'data/binding.json'; b = read(p); b['asset_instances'].pop(); write(p,b)
        with self.assertRaisesRegex(ValueError, 'Assembly/binding mismatch'): self.compile()
        self.assertFalse(self.out.exists())

    def test_incomplete_selected_lod_stops_compilation(self):
        p = self.root/'build/assembly.json'; m = read(p); m['building_lod'] = '1M'; write(p,m)
        with self.assertRaisesRegex(ValueError, 'Missing selected LOD'): self.compile()

    def test_immutable_release_and_export(self):
        self.compile()
        with self.assertRaisesRegex(ValueError, 'overwrite'): self.compile()
        output = self.root/'build/new/assembly.json'; export_ue(self.out, self.root, output)
        with self.assertRaisesRegex(ValueError, 'overwrite'): export_ue(self.out, self.root, output)

    def test_changed_layer_is_detected(self):
        self.compile(); p = self.out/'assembly.json'; value = read(p); value['native'] = []; write(p,value)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'): load_scene(self.out)

    def test_paths_cannot_escape_even_via_symlink(self):
        (self.root/'outside').symlink_to(self.temp.name, target_is_directory=True)
        for path in ('../secret', '/tmp/file', 'C:\\private\\file', 'outside/secret'):
            with self.subTest(path=path), self.assertRaises(ValueError): inside(self.root,path)

    def test_references_native_transform_and_road_topology_are_validated(self):
        self.compile(); _, layers = load_scene(self.out)
        for mutate in (
            lambda d: d['assembly']['native'][0]['world_from_asset_m'][0].__setitem__(3, 999),
            lambda d: d['roads']['ground']['edges'][0].__setitem__('target','missing'),
            lambda d: d['buildings']['entities'][0]['asset_instance_ids'].append('missing'),
            lambda d: d['relations']['prop_groups'][0].__setitem__('part_count',2)):
            bad = deepcopy(layers); mutate(bad)
            with self.assertRaises(ValueError): validate_layers(bad)

    def test_singular_or_non_finite_transform_is_rejected(self):
        for m in ([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]], [[float('nan'),0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]):
            with self.assertRaises(ValueError): matrix(m)

    def test_explicit_windows_interchange_relocation(self):
        p = self.root/'build/assembly.json'; m = read(p)
        for row in m['meshes']: row['import_file'] = '..\\prepared\\interchange\\model.glb'
        m['calibration']['path'] = '..\\prepared\\interchange\\axis.glb'; write(p,m)
        compile_scene(self.root, self.config, self.out, self.root/'build/interchange')
        self.assertEqual(validate(self.out,self.root,True)['status'], 'passed')

    def test_filtered_background_uses_explicit_verified_derivative(self):
        r = self.root; p = r/'build/assembly.json'; m = read(p)
        (r/'build/interchange/filtered.glb').write_bytes(b'filtered background')
        f = r/'build/interchange/filtered.glb'
        background = deepcopy(m['meshes'][0])
        background.update(id='BACKGROUND', kind='background', import_file='interchange/old_background.glb',
            import_parts=[dict(file='interchange/filtered.glb',sha256=sha(f),bytes=f.stat().st_size)],
            background_derivative=dict(file='build/interchange/filtered.glb',sha256=sha(f)))
        background['prepared'].update(interchange_sha256=sha(f),interchange_bytes=f.stat().st_size)
        m['meshes'].append(background); write(p,m)
        self.compile(); output=r/'build/export/assembly.json'; export_ue(self.out,r,output)
        row=read(output)['meshes'][-1]
        self.assertEqual((output.parent/row['import_file']).resolve(),f)
        self.assertEqual((output.parent/row['import_parts'][0]['file']).resolve(),f)
        self.assertEqual(validate(self.out,r,True)['status'],'passed')


if __name__ == '__main__':
    unittest.main()
