"""Contact overlays must identify current 3D evidence without exposing audit files."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

HERE = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

server = load('collision_preview_server', HERE/'serve.py')
exporter = load('collision_preview_exporter', HERE/'prepare_collisions.py')


def columns(heights):
    return {'origin': np.array([0., 0.]), 'cell': np.array(.25), 'shape': np.array([1, 1]),
            'zmin': np.array([[min(heights)]]), 'zmax': np.array([[max(heights)]]),
            'offsets': np.array([0, len(heights)]), 'hits': np.array(heights, float)}


def record(aid, heights, components=None):
    out = {'id': aid, 'kind': 'base', 'source_ids': [aid], 'entity_id': aid,
           'world_bounds': [[0, 0, 0], [.25, .25, 11]], '_columns': exporter.pairs.prepare_columns(columns(heights))}
    if components:
        out['components'] = components
    return out


class ContactGeometryTests(TestCase):
    def test_projected_overlap_with_disjoint_heights_has_no_patch(self):
        a, b = record('A', [0, 1, 10, 11]), record('B', [4, 6])
        self.assertEqual(exporter.building_patches(a, b), [])

    def test_physical_component_union_and_actual_height_are_preserved(self):
        parts = [record('A1', [0, 3]), record('A2', [1, 4])]
        a = record('A', [0, 1, 3, 4], parts)
        b = record('B', [2, 3])
        patches = exporter.building_patches(a, b)
        self.assertEqual(patches, [{'center': [.125, .125, 2.5], 'size': [.25, .25, 1.]}])
        evidence = exporter.geometry_fields(patches, [[0, 0, 0], [100, 100, 100]], 1000)
        self.assertEqual(evidence['focus_position'], [.125, .125, 2.5])

    def test_patch_limit_samples_without_inventing_a_larger_contact_area(self):
        patches = [exporter.patch(i*.25, 0, 10, 11, .25) for i in range(1200)]
        result = exporter.geometry_fields(patches, [[0, 0, 0], [1, 1, 1]], 1000)
        self.assertTrue(result['patches_truncated'])
        self.assertEqual(len(result['patches']), 1000)
        self.assertTrue(all(p['size'] == [.25, .25, 1.] for p in result['patches']))


class PublishedOverlayTests(TestCase):
    def payload(self):
        return {'schema': exporter.SCHEMA, 'binding_sha256': 'current', 'source_audit_path': '/private/audit',
                'cases': [{'id': 'building:A__B', 'kind': 'building', 'asset_ids': ['A', 'B'], 'label': 'Contact',
                           'severity': 'minor', 'review_status': 'minor_contact', 'default_visible': False,
                           'focus_position': [1, 2, 10], 'bounds': [[0, 1, 9], [2, 3, 11]],
                           'metrics': {'overlap_area_m2': .25, 'private_source': '/private/mesh'},
                           'patches': [{'center': [1, 2, 10], 'size': [.25, .25, .1]}], 'source_file': '/private/mesh'}]}

    def run_payload(self, root, payload):
        folder = root/'physical_postprocess'
        folder.mkdir(exist_ok=True)
        (folder/'collision_preview.json').write_text(json.dumps(payload))
        scene = {'binding_sha256': 'current', 'assets': [{'id': 'A'}, {'id': 'B'}]}
        allowed = {}
        server.add_collision_audit(scene, allowed, root)
        self.assertEqual(allowed, {})
        return scene

    def test_missing_or_stale_overlay_never_draws_old_geometry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scene = {'binding_sha256': 'current', 'assets': []}
            server.add_collision_audit(scene, {}, root)
            self.assertIsNone(scene['collision_audit'])
            value = self.payload()
            value['binding_sha256'] = 'old-placement'
            scene = self.run_payload(root, value)
            self.assertIsNone(scene['collision_audit'])
            self.assertEqual(scene['collision_audit_status'], 'binding_mismatch')

    def test_only_whitelisted_evidence_is_embedded(self):
        with TemporaryDirectory() as directory:
            scene = self.run_payload(Path(directory), self.payload())
            value = scene['collision_audit']
            self.assertEqual(value['cases'][0]['focus_position'], [1, 2, 10])
            self.assertEqual(value['counts']['building_contacts'], 1)
            self.assertNotIn('/private/', json.dumps(value))

    def test_boundary_review_is_neutral_and_is_not_reclassified_as_minor(self):
        with TemporaryDirectory() as directory:
            value = self.payload()
            value['cases'][0].update(severity='review', review_status='boundary_review', label='建筑A / B · 边界接触')
            case = self.run_payload(Path(directory), value)['collision_audit']['cases'][0]
            self.assertEqual(case['severity'], 'review')
            self.assertEqual(case['review_status'], 'boundary_review')
            self.assertFalse(case['default_visible'])

    def test_invalid_asset_or_patch_is_rejected(self):
        with TemporaryDirectory() as directory:
            value = self.payload()
            value['cases'][0]['asset_ids'] = ['not-published']
            with self.assertRaisesRegex(ValueError, 'unknown'):
                self.run_payload(Path(directory), value)
            value = self.payload()
            value['cases'][0]['patches'][0]['size'][2] = -1
            with self.assertRaisesRegex(ValueError, 'patch size'):
                self.run_payload(Path(directory), value)
            value = self.payload()
            value['cases'][0]['focus_position'][2] = float('nan')
            with self.assertRaisesRegex(ValueError, 'coordinates'):
                self.run_payload(Path(directory), value)


class PortableExportTests(TestCase):
    def test_arbitrary_release_has_only_real_current_independent_contact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets, audit = root/'assets', root/'audit'
            assets.mkdir()
            matrix = np.eye(4).tolist()
            binding = {'asset_instances': [{'id': aid, 'world_from_local_4x4': matrix, 'lods': {'0.1M': {'sha256': aid}}} for aid in ['A', 'B']], 'massing_instances': []}
            binding_file = assets/'binding_manifest_physical.json'
            binding_file.write_text(json.dumps(binding))
            identity = exporter.sha(binding_file)
            records = []
            for aid, heights in [('A', [0, 1, 10, 11]), ('B', [.5, 1.5])]:
                folder = audit/'cache/instances'/aid
                folder.mkdir(parents=True)
                np.savez(folder/'columns.npz', **columns(heights))
                r = record(aid, heights)
                r.pop('_columns')
                r.update(columns=f'instances/{aid}/columns.npz', world_from_local_4x4=matrix, source_model_sha256=aid)
                records.append(r)
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write(audit/'cache/INDEX.json', {'binding_sha256': identity, 'cell_m': .25, 'records': records})
            write(audit/'SUMMARY.json', {'binding_sha256': identity})
            write(audit/'ground/GROUND_CONTACTS.json', {'binding_sha256': identity, 'records': []})
            write(audit/'buildings/SUMMARY.json', {'binding_sha256': identity})
            contact = {'asset_a': 'A', 'asset_b': 'B', 'same_building_family': False, 'known_solid_overlap_area_m2': .0625,
                       'known_overlap_volume_m3': .03125, 'max_common_vertical_occupied_length_m': .5,
                       'max_horizontal_interior_radius_m': .125, 'classification': 'minor_contact'}
            write(audit/'buildings/PAIRS.json', {'pairs': [contact, {**contact, 'same_building_family': True}, {**contact, 'known_solid_overlap_area_m2': 0, 'classification': 'XY_only_no_common_solid'}]})
            lane = root/'lanes.geojson'
            write(lane, {'type': 'FeatureCollection', 'features': []})
            write(audit/'roads/car_2m/SUMMARY.json', {'source_cache_index_sha256': exporter.sha(audit/'cache/INDEX.json'),
                  'lane_geojson_sha256': exporter.sha(lane), 'clearance_m': 2., 'thresholds': {'one_lane_centerline_block_m': 1.}})
            write(audit/'roads/car_2m/DETAILS.json', {'instances': []})
            before = root/'before.json'
            write(before, {'pairs': [{**contact, 'classification': 'clear_overlap', 'known_solid_overlap_area_m2': 10}]})
            result = exporter.prepare_collision_preview(assets, audit, before_report=before, lane_geojson=lane)
            self.assertEqual(result['counts']['building_contacts'], 1)
            self.assertEqual(result['counts']['default_visible_cases'], 1)
            case = result['cases'][0]
            self.assertEqual(case['id'], 'building:A__B')
            self.assertEqual(case['focus_position'], [.125, .125, .75])
            self.assertEqual(case['patches'][0]['size'], [.25, .25, .5])
            # A geometric review classification does not establish manual
            # approval, and must not be relabelled as a minor contact.
            write(audit/'buildings/PAIRS.json', {'pairs': [{**contact, 'classification': 'review_overlap'}]})
            boundary = exporter.prepare_collision_preview(assets, audit, lane_geojson=lane)['cases'][0]
            self.assertEqual(boundary['severity'], 'review')
            self.assertEqual(boundary['review_status'], 'boundary_review')
            self.assertEqual(boundary['label'], '建筑A / B · 边界接触')
            self.assertFalse(boundary['default_visible'])
            self.assertEqual(boundary['patches'], case['patches'])
            binding_file.write_text(json.dumps({**binding, 'changed': True}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                exporter.prepare_collision_preview(assets, audit, lane_geojson=lane)
