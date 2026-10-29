"""Distribution must preserve paths, reject changed bytes and omit build caches."""
import importlib.util
import json
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


FILE = Path(__file__).resolve().parents[1]/'release/bundles.py'
spec = importlib.util.spec_from_file_location('release_bundles', FILE)
bundles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundles)


class ReleaseBundles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.catalog = self.root/'source', self.root/'catalog'
        self.target, self.cache = self.root/'target', self.root/'cache'
        self.source.mkdir()
        self.rows = {}
        for name, files in {
            'observations': [('data/input/a.bin', b'abc'*700)],
            'structure': [('data/structure/same.bin', b'abc'*700),
                          ('data/structure/other.bin', b'other data')],
        }.items():
            rows = []
            for path, payload in files:
                destination = self.source/path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                rows.append({'path': path, 'bytes': len(payload), 'sha256': bundles.sha(destination),
                    'object_package': 'observations' if payload.startswith(b'abc') else name, 'mode': 0o644})
            self.rows[name] = rows
            bundles.write(self.catalog/'manifests'/f'{name}.json', {'id': name, 'files': rows})
        self.cat = {'profiles': {'scene': ['structure']}, 'packages': {
            name: {'manifest': f'manifests/{name}.json',
                   'manifest_sha256': bundles.sha(self.catalog/'manifests'/f'{name}.json'),
                   'dependencies': ['observations'] if name == 'structure' else []}
            for name in self.rows}}
        bundles.write(self.catalog/'catalog.json', self.cat)
        self.archive = self.root/'archives'

    def pack(self):
        return bundles.pack(self.source, self.catalog, self.archive, part_bytes=1024)

    def install(self):
        return bundles.install(self.catalog, self.archive/'downloads.json', self.archive,
                               self.target, self.cache)

    def test_deduplicated_multi_part_roundtrip_and_idempotent_install(self):
        self.pack()
        result = self.install()
        self.assertEqual(result['files_restored'], 3)
        self.assertEqual(len(list((self.cache/'objects').iterdir())), 2)
        self.assertTrue(all(p.stat().st_size <= 1024 for p in self.archive.rglob('*.tarpart')))
        for rows in self.rows.values():
            for row in rows:
                self.assertEqual(bundles.sha(self.target/row['path']), row['sha256'])
        self.assertEqual(self.install()['files_restored'], 0)

    def test_corrupted_archive_never_installs_unverified_data(self):
        self.pack()
        part = next(self.archive.rglob('*.tarpart'))
        part.write_bytes(b'x'*part.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'verification'):
            self.install()
        self.assertFalse((self.target/'data/input/a.bin').exists())

    def test_changed_source_is_not_packaged_under_old_proof(self):
        (self.source/'data/input/a.bin').write_bytes(b'x'*2100)
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            self.pack()
        self.assertFalse(self.archive.exists())

    def test_unchanged_archive_parts_can_be_reused_with_verified_content(self):
        self.pack()
        next_archive = self.root/'next-archives'
        bundles.pack(self.source, self.catalog, next_archive, reuse=self.archive, part_bytes=1024)
        old, new = bundles.read(self.archive/'downloads.json'), bundles.read(next_archive/'downloads.json')
        self.assertEqual(old, new)
        self.archive = next_archive
        self.assertEqual(self.install()['files_restored'], 3)

    def test_corrupt_reusable_parts_are_not_propagated(self):
        self.pack()
        part = next(self.archive.rglob('*.tarpart'))
        part.write_bytes(b'x'*part.stat().st_size)
        output = self.root/'next-archives'
        with self.assertRaisesRegex(ValueError, 'Reusable archive failed'):
            bundles.pack(self.source, self.catalog, output, reuse=self.archive)
        self.assertFalse(output.exists())

    def test_changed_installed_file_is_not_overwritten(self):
        self.pack()
        self.install()
        path = self.target/'data/input/a.bin'
        path.write_bytes(b'manual edit')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.install()
        self.assertEqual(path.read_bytes(), b'manual edit')

    def test_catalog_and_manifest_tampering_are_rejected(self):
        self.pack()
        path = self.catalog/'manifests/structure.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'Changed package manifest'):
            self.install()

    def test_http_download_resumes_at_verified_range(self):
        payload = b'0123456789'*300
        source = self.root/'reference'
        source.write_bytes(payload)
        record = {'path': 'package/part.tarpart', 'bytes': len(payload), 'sha256': bundles.sha(source)}
        target = self.root/'download'
        target.with_name('download.partial').write_bytes(payload[:73])
        def response(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=73-')
            stream = io.BytesIO(payload[73:])
            stream.status, stream.headers = 206, {'Content-Range': 'bytes 73-2999/3000'}
            return stream
        with patch.object(bundles.urllib.request, 'urlopen', response):
            bundles.fetch('https://example.invalid/release', record, target)
        self.assertEqual(target.read_bytes(), payload)

    def test_http_without_range_support_restarts_partial_file(self):
        payload = b'restart full verified payload'
        source = self.root/'reference'
        source.write_bytes(payload)
        record = {'path': 'package/part.tarpart', 'bytes': len(payload), 'sha256': bundles.sha(source)}
        target = self.root/'download'
        target.with_name('download.partial').write_bytes(payload[:3])
        def response(request, timeout):
            stream = io.BytesIO(payload)
            stream.status, stream.headers = 200, {}
            return stream
        with patch.object(bundles.urllib.request, 'urlopen', response):
            bundles.fetch('https://example.invalid/release', record, target)
        self.assertEqual(target.read_bytes(), payload)

    def test_escaping_and_protected_paths_are_rejected(self):
        for value in ['../outside', '/etc/passwd', 'data/../../outside', '.git/config',
                      'data/link/../../outside', 'C:\\private', '.env/../x']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                bundles.inside(self.target, value)
        self.target.mkdir()
        (self.target/'outside').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            bundles.inside(self.target, 'outside/file')

    def test_missing_and_cyclic_dependencies_fail_before_packaging(self):
        bad = json.loads(json.dumps(self.cat))
        bad['packages']['observations']['dependencies'] = ['structure']
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            bundles.selected(bad)
        with self.assertRaisesRegex(ValueError, 'Unknown package'):
            bundles.selected(self.cat, packages=['missing'])

    def test_source_archive_contains_code_and_docs_without_large_resources(self):
        for name in ['README.md', 'src/example.py', 'models/depth/README.md',
                     'models/depth/model.pt', 'data/README.md', 'build/old.glb',
                     'environment/wheels/old.whl', '.env', 'dist/old-package.whl',
                     'web_preview/dist/index.html', 'web_preview/dist/assets/viewer.js']:
            p = self.source/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('example')
        output = self.root/'source.tar.gz'
        bundles.source_archive(self.source, output)
        with tarfile.open(output) as archive:
            names = archive.getnames()
        self.assertIn('CityCode/src/example.py', names)
        self.assertIn('CityCode/data/README.md', names)
        self.assertIn('CityCode/web_preview/dist/index.html', names)
        self.assertIn('CityCode/web_preview/dist/assets/viewer.js', names)
        self.assertFalse(any(n.endswith(('.pt', '.whl', '.bin', '.env', '.glb')) for n in names))


if __name__ == '__main__':
    unittest.main()
