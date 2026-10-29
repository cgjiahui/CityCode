#!/usr/bin/env python3
"""Index, deduplicate, shard, download and restore the separate CityCode data.

Python standard library only. No upload or model/API execution is performed.
Archives store each SHA256 object once; manifests restore the original relative
paths so existing geometry, evidence and model references continue to work.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    'observations', 'structure', 'buildings', 'maps', 'scene-layout', 'preview',
    'prediction-cache', 'building-regeneration', 'building-replay', 'background-replay',
    'pixal3d-inputs', 'depth-weights',
    'generation-weights', 'offline-environment',
]
DEPENDENCIES = {
    'scene-layout': ['structure', 'buildings', 'maps'],
    'preview': ['buildings', 'maps'], 'prediction-cache': ['observations'],
    'building-replay': ['buildings', 'structure'], 'background-replay': ['maps'],
    'building-regeneration': ['structure'],
    'pixal3d-inputs': ['background-replay'],
}
PROFILES = {
    'scene': ['structure', 'buildings', 'maps', 'scene-layout', 'preview'],
    'replay': ['observations', 'structure', 'buildings', 'maps', 'scene-layout',
               'preview', 'prediction-cache', 'building-replay', 'background-replay'],
    'regeneration-inputs': ['building-regeneration'],
    'generate': ['building-regeneration', 'generation-weights', 'offline-environment'],
    'depth': ['observations', 'depth-weights'], 'all': PACKAGES,
}
SKIP = {'__pycache__', 'node_modules', '.git', '.agents', '.codex', '.aws',
        '.venv', '.venv-generation', '.venv-background', '.cache'}
HEX = re.compile(r'^[a-f0-9]{64}$')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Expected a portable relative path')
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith('/') or '..' in parts or any(p in SKIP for p in parts):
        raise ValueError('Unsafe relative path: ' + relative)
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes release: ' + relative)
    return path


def files_under(root):
    for directory, names, files in os.walk(root):
        names[:] = sorted(n for n in names if n not in SKIP and not n.endswith('.egg-info'))
        for name in sorted(files):
            if name.endswith(('.pyc', '.partial', '.writing')):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError('Data must be materialized, not a symlink: ' + str(path))
            yield path


def building_files(root):
    """Only authoritative bindings, used meshes and display decisions are core."""
    directory = root / 'data/smallcity_p384_building_assets_completed'
    required = {'binding_manifest.json', 'binding_manifest_physical.json', 'README.md',
                'massing_completion/recipe.json'}
    for file in directory.glob('physical_postprocess/*.json'):
        required.add(file.relative_to(directory).as_posix())
    binding = read(directory / 'binding_manifest_physical.json')
    for row in binding['asset_instances'] + binding.get('massing_instances', []) + binding.get('surface_instances', []):
        for model in row.get('lods', row.get('donor_lods', {})).values():
            file = (directory / model['path']).resolve()
            if not file.is_relative_to(root.resolve()):
                raise ValueError('Model escapes data release')
            required.add(os.path.relpath(file, directory))
    return {(directory / value).resolve() for value in required if (directory / value).is_file()}


def regeneration_files(root):
    """Saved images/conditions needed for regeneration, excluding old meshes."""
    path = root/'data/smallcity_p384_reused_generation_inputs/manifest.json'
    doc = read(path)
    return {(path.parent/row['path']).resolve() for row in doc['required_files']
            if row['path'].startswith('../smallcity_p384_building_assets/')}


def classify(relative, path, core_buildings, generation_inputs):
    parts = PurePosixPath(relative).parts
    if parts[0] == 'models':
        return 'depth-weights' if parts[1] == 'depth' else 'generation-weights'
    if parts[0] == 'environment':
        return 'offline-environment'
    name = parts[1]
    if name == 'smallcity_p384_building_assets' and path.resolve() in generation_inputs:
        return 'building-regeneration'
    direct = {
        'smallcity_p384': 'observations', 'smallcity_p384_structure': 'structure',
        'smallcity_p384_predictions': 'prediction-cache',
        'smallcity_p384_recovery_cache': 'prediction-cache',
        'smallcity_p384_background_coastal': 'maps',
        'smallcity_p384_background_completed': 'maps',
        'smallcity_p384_scene_enrichment': 'scene-layout',
        'smallcity_p384_gt_roads': 'scene-layout',
        'smallcity_p384_building_assets_completed_web_preview': 'preview',
        'smallcity_p384_background_web_preview': 'preview',
        'smallcity_p384_building_assets': 'building-replay',
        'smallcity_p384_massing_completion_inputs': 'building-replay',
        'smallcity_p384_background_compatibility': 'background-replay',
        'smallcity_p384_reused_generation_inputs': 'building-regeneration',
        'smallcity_p384_background_pixal3d_inputs': 'pixal3d-inputs',
    }
    if name == 'smallcity_p384_building_assets_completed':
        return 'buildings' if path.resolve() in core_buildings else 'building-replay'
    if name == 'smallcity_p384_background_history':
        return 'maps' if relative.startswith('data/'+name+'/height/completed/') else 'background-replay'
    if name in {'smallcity_p384_citycode', 'smallcity_p384_web_preview', 'README.md'}:
        return None
    if name not in direct:
        raise ValueError('Unclassified data; update release policy before packaging: ' + relative)
    return direct[name]


def index(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    core = building_files(root)
    generation_inputs = regeneration_files(root)
    rows = {name: [] for name in PACKAGES}
    excluded, directories = [], Counter()
    roots = [root/'data', root/'models', root/'environment/wheels', root/'environment/background_wheels']
    for base in roots:
        for path in files_under(base):
            relative = path.relative_to(root).as_posix()
            package = classify(relative, path, core, generation_inputs)
            size = path.stat().st_size
            directories['/'.join(relative.split('/')[:2])] += size
            if package is None:
                excluded.append({'path': relative, 'bytes': size})
                continue
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Input changed while indexing: ' + relative)
            rows[package].append({'path': relative, 'bytes': size, 'sha256': digest,
                                  'mode': before.st_mode & 0o777})
    owners, packages = {}, {}
    for name in PACKAGES:
        deps = set(DEPENDENCIES.get(name, []))
        objects = {}
        for row in rows[name]:
            digest = row['sha256']
            # Avoid downloading a whole unrelated package for one shared tiny
            # JSON/text file. Large shared resources have one download owner.
            key = digest if row['bytes'] >= 1 << 20 else (name, digest)
            # Eight small component-prior meshes must be available without
            # making an input-only download depend on the full building set.
            if name == 'building-regeneration' and owners.get(digest) == 'buildings':
                key = (name, digest)
            owner = owners.setdefault(key, name)
            row['object_package'] = owner
            if owner == name:
                objects.setdefault(digest, row)
            else:
                deps.add(owner)
        manifest = {'schema': 'citycode.data_package.v1', 'id': name,
                    'dependencies': sorted(deps, key=PACKAGES.index), 'files': rows[name]}
        file = output/'manifests'/(name+'.json')
        write(file, manifest)
        packages[name] = {'manifest': 'manifests/'+name+'.json', 'manifest_sha256': sha(file),
            'files': len(rows[name]), 'logical_bytes': sum(x['bytes'] for x in rows[name]),
            'unique_objects': len(objects), 'download_object_bytes': sum(x['bytes'] for x in objects.values()),
            'dependencies': manifest['dependencies']}
    catalog = {'schema': 'citycode.data_catalog.v1', 'release_id': 'smallcity-20261028',
        'status': 'local_candidate_not_uploaded', 'packages': packages, 'profiles': PROFILES,
        'excluded': {'local_citycode_snapshot': 'Requires locally generated UE interchange; regenerate after prepare/finalize.',
                     'legacy_preview': 'Replaced by the completed building preview.',
                     'build': 'CPU conversion copies, native Epic derivatives and temporary checks are not data downloads.',
                     'files': len(excluded), 'bytes': sum(x['bytes'] for x in excluded)},
        'totals': {'logical_bytes': sum(p['logical_bytes'] for p in packages.values()),
                   'unique_object_bytes': sum(p['download_object_bytes'] for p in packages.values())},
        'directories': dict(sorted(directories.items()))}
    catalog['totals']['duplicate_bytes_avoided'] = catalog['totals']['logical_bytes']-catalog['totals']['unique_object_bytes']
    write(output/'catalog.json', catalog)
    return catalog


def selected(catalog, profile=None, packages=()):
    wanted = set(packages or catalog['profiles'][profile or 'scene'])
    result, visiting = [], set()
    def visit(name):
        if name in result:
            return
        if name not in catalog['packages'] or name in visiting:
            raise ValueError('Unknown/cyclic package dependency: ' + name)
        visiting.add(name)
        for dependency in catalog['packages'][name]['dependencies']:
            visit(dependency)
        visiting.remove(name)
        result.append(name)
    for name in catalog['packages']:
        if name in wanted:
            visit(name)
    if wanted-set(result):
        raise ValueError('Unknown package selection: ' + str(wanted-set(result)))
    return result


def manifest(catalog_root, catalog, name):
    record = catalog['packages'][name]
    path = inside(catalog_root, record['manifest'])
    if sha(path) != record['manifest_sha256']:
        raise ValueError('Changed package manifest: ' + name)
    result = read(path)
    if result.get('id') != name:
        raise ValueError('Package identity mismatch')
    for row in result['files']:
        inside(catalog_root, row['path'])  # Reject protected/escaping destination names.
        if not HEX.fullmatch(row['sha256']) or type(row['bytes']) is not int or row['bytes'] < 0:
            raise ValueError('Invalid object proof')
    return result


class PartWriter:
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.stream, self.size, self.digest, self.parts = None, 0, None, []

    def write(self, data):
        total = len(data)
        while data:
            if self.stream is None:
                self.path = self.directory/f'part-{len(self.parts):05d}.tarpart'
                self.stream = self.path.open('xb')
                self.size, self.digest = 0, hashlib.sha256()
            piece = data[:self.limit-self.size]
            self.stream.write(piece)
            self.digest.update(piece)
            self.size += len(piece)
            data = data[len(piece):]
            if self.size == self.limit:
                self.finish_part()
        return total

    def finish_part(self):
        if self.stream is not None:
            self.stream.close()
            self.parts.append({'path': self.path.name, 'bytes': self.size, 'sha256': self.digest.hexdigest()})
            self.stream = None

    def flush(self):
        if self.stream:
            self.stream.flush()


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def pack(root, catalog_root, output, *, profile='scene', packages=(), part_bytes=1 << 30, reuse=None):
    root, catalog_root, output = Path(root).resolve(), Path(catalog_root).resolve(), Path(output).resolve()
    if output.exists() or part_bytes < 1024:
        raise ValueError('Use a new archive directory and a part size of at least 1024 bytes')
    catalog = read(catalog_root/'catalog.json')
    names = selected(catalog, profile, packages)
    previous = read(Path(reuse)/'downloads.json')['packages'] if reuse else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.citycode-pack-', dir=output.parent))
    try:
        delivery = {'schema': 'citycode.data_downloads.v1', 'catalog_sha256': sha(catalog_root/'catalog.json'), 'packages': {}}
        for name in names:
            doc = manifest(catalog_root, catalog, name)
            folder = staging/name
            folder.mkdir()
            object_index = sorted({(row['sha256'], row['bytes']) for row in doc['files']
                                   if row['object_package'] == name})
            identity = hashlib.sha256(json.dumps(object_index, separators=(',', ':')).encode()).hexdigest()
            old = previous.get(name, {})
            if old.get('object_index_sha256') == identity:
                for part in old['parts']:
                    source = inside(Path(reuse), part['path'])
                    if source.stat().st_size != part['bytes'] or sha(source) != part['sha256']:
                        raise ValueError('Reusable archive failed verification: '+part['path'])
                    destination = inside(staging, part['path'])
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copyfile(source, destination)
                delivery['packages'][name] = old
                print(json.dumps({'package': name, 'reused_verified_parts': len(old['parts'])}), flush=True)
                continue
            writer, seen = PartWriter(folder, part_bytes), set()
            with tarfile.open(fileobj=writer, mode='w|') as archive:
                for row in doc['files']:
                    if row['object_package'] != name or row['sha256'] in seen:
                        continue
                    seen.add(row['sha256'])
                    path = inside(root, row['path'])
                    if path.stat().st_size != row['bytes']:
                        raise ValueError('Source size changed: ' + row['path'])
                    info = tarfile.TarInfo('objects/'+row['sha256'])
                    info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
                    with path.open('rb') as f:
                        reader = HashReader(f)
                        archive.addfile(info, reader)
                        if reader.digest.hexdigest() != row['sha256']:
                            raise ValueError('Source checksum changed: ' + row['path'])
            writer.finish_part()
            delivery['packages'][name] = {'object_index_sha256': identity,
                'parts': [{**p, 'path': name+'/'+p['path']} for p in writer.parts]}
            print(json.dumps({'package': name, 'parts': len(writer.parts), 'objects': len(seen)}), flush=True)
        write(staging/'downloads.json', delivery)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'status': 'archives_ready', 'packages': names, 'output': str(output), 'uploaded': False}


class JoinedReader:
    def __init__(self, files):
        self.files, self.stream = iter(files), None

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Streaming reads must have a bounded size')
        result = bytearray()
        while len(result) < size:
            if self.stream is None:
                try:
                    self.stream = next(self.files).open('rb')
                except StopIteration:
                    break
            block = self.stream.read(size-len(result))
            if block:
                result.extend(block)
            else:
                self.stream.close()
                self.stream = None
        return bytes(result)

    def close(self):
        if self.stream:
            self.stream.close()


def fetch(base, record, target):
    """Resume HTTP byte ranges, but accept only the pinned full-file hash."""
    if target.exists() and target.stat().st_size == record['bytes'] and sha(target) == record['sha256']:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.partial')
    if urllib.parse.urlparse(str(base)).scheme in {'https', 'http'}:
        url = str(base).rstrip('/')+'/'+urllib.parse.quote(record['path'], safe='/')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= record['bytes']:
            partial.unlink()
            offset = 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
            append = offset and response.status == 206
            if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise ValueError('Server returned a different download range')
            with partial.open('ab' if append else 'wb') as out:
                shutil.copyfileobj(response, out, 8 << 20)
    else:
        source = inside(Path(base), record['path'])
        shutil.copyfile(source, partial)
    if partial.stat().st_size != record['bytes'] or sha(partial) != record['sha256']:
        partial.unlink()
        raise ValueError('Downloaded part failed verification: ' + record['path'])
    partial.replace(target)


def install(catalog_root, downloads, base, target, cache, *, profile='scene', packages=()):
    catalog_root, target, cache = map(lambda p: Path(p).resolve(), (catalog_root, target, cache))
    catalog = read(catalog_root/'catalog.json')
    delivery = read(downloads)
    if delivery['catalog_sha256'] != sha(catalog_root/'catalog.json'):
        raise ValueError('Download index belongs to another catalog')
    names = selected(catalog, profile, packages)
    # Preflight every target, catalog and available package before downloading/writing data.
    docs = {name: manifest(catalog_root, catalog, name) for name in names}
    for name, doc in docs.items():
        if name not in delivery['packages']:
            raise ValueError('Archives not built/hosted for package: ' + name)
        for row in doc['files']:
            path = inside(target, row['path'])
            if path.exists() and (not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']):
                raise ValueError('Refusing to overwrite different installed file: ' + row['path'])
    objects = cache/'objects'
    objects.mkdir(parents=True, exist_ok=True)
    for name in names:
        doc = docs[name]
        expected = {r['sha256']: r['bytes'] for r in doc['files'] if r['object_package'] == name}
        missing = {digest for digest, size in expected.items()
                   if not (objects/digest).is_file() or (objects/digest).stat().st_size != size or sha(objects/digest) != digest}
        if not missing:
            continue
        parts = []
        for row in delivery['packages'][name]['parts']:
            part = inside(cache/'parts', row['path'])
            fetch(base, row, part)
            parts.append(part)
        joined = JoinedReader(parts)
        found = set()
        try:
            with tarfile.open(fileobj=joined, mode='r|') as archive:
                for member in archive:
                    digest = member.name.removeprefix('objects/')
                    if (member.name != 'objects/'+digest or not HEX.fullmatch(digest)
                            or not member.isfile() or digest not in expected or digest in found
                            or member.size != expected[digest]):
                        raise ValueError('Unexpected archive member: ' + member.name)
                    found.add(digest)
                    if digest not in missing:
                        continue
                    temporary = objects/(digest+'.partial')
                    with archive.extractfile(member) as f, temporary.open('wb') as out:
                        shutil.copyfileobj(f, out, 8 << 20)
                    if temporary.stat().st_size != expected[digest] or sha(temporary) != digest:
                        temporary.unlink()
                        raise ValueError('Archive object checksum mismatch')
                    temporary.replace(objects/digest)
        finally:
            joined.close()
        if found != set(expected):
            raise ValueError('Archive is incomplete: ' + name)
    restored = 0
    for doc in docs.values():
        for row in doc['files']:
            source = objects/row['sha256']
            if not source.is_file() or source.stat().st_size != row['bytes']:
                raise ValueError('Missing object dependency: ' + row['path'])
            destination = inside(target, row['path'])
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name+'.partial')
            shutil.copyfile(source, temporary)
            if sha(temporary) != row['sha256']:
                raise ValueError('Installed file failed verification')
            temporary.chmod(row.get('mode', 0o644) & 0o777)
            temporary.replace(destination)
            restored += 1
    return {'status': 'installed', 'packages': names, 'files_restored': restored,
            'verified': True, 'API_calls': 0, 'GPU_jobs': 0}


def source_archive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Source archive already exists')
    allowed = {'src', 'pipelines', 'workflow', 'tests', 'third_party', 'web_preview', 'release', 'environment', 'models'}
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz', dereference=True) as archive:
        for directory, names, files in os.walk(root):
            relative = Path(directory).relative_to(root)
            names[:] = sorted(n for n in names if n not in SKIP and not n.endswith('.egg-info')
                and n not in {'wheels', 'background_wheels', 'build', 'runs', 'artifacts'}
                and (n != 'dist' or relative == Path('web_preview')))
            if relative.parts and relative.parts[0] not in allowed:
                names[:] = []
                continue
            for name in sorted(files):
                path = Path(directory)/name
                rel = path.relative_to(root)
                if name == '.env' or name.endswith(('.pyc', '.pt', '.pth', '.safetensors', '.whl', '.partial', '.log', '.tar', '.gz', '.tarpart')):
                    continue
                if path.stat().st_size > 50 << 20:
                    raise ValueError('Unexpected large source file: '+str(rel))
                if path.is_symlink() and not path.resolve().is_relative_to(root):
                    raise ValueError('External source symlink: '+str(rel))
                archive.add(path, arcname='CityCode/'+rel.as_posix(), recursive=False)
                count += 1
        data_readme = root/'data/README.md'
        if data_readme.is_file():
            archive.add(data_readme, arcname='CityCode/data/README.md', recursive=False)
            count += 1
    return {'status': 'source_archive_ready', 'files': count, 'bytes': output.stat().st_size,
            'sha256': sha(output), 'contains_data_or_weight_binaries': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['index', 'plan', 'pack', 'install', 'source']:
        command = commands.add_parser(name)
        command.add_argument('--catalog', type=Path, default=ROOT/'release')
        if name in {'plan', 'pack', 'install'}:
            command.add_argument('--profile', choices=PROFILES, default='scene')
            command.add_argument('--package', action='append', default=[])
        if name in {'index', 'pack', 'source'}:
            command.add_argument('--output', type=Path, required=True)
        if name == 'pack':
            command.add_argument('--part-mib', type=int, default=1024)
            command.add_argument('--reuse', type=Path, help='Reuse checksum-verified unchanged package parts')
        if name == 'install':
            command.add_argument('--downloads', type=Path, required=True)
            command.add_argument('--base-url', required=True, help='HTTPS archive base or local archive directory')
            command.add_argument('--target', type=Path, default=ROOT)
            command.add_argument('--cache', type=Path, default=ROOT/'.cache/data-downloads')
    args = parser.parse_args()
    if args.command == 'index':
        result = index(args.root, args.output)
        result = {k: result[k] for k in ['status', 'totals', 'excluded']}
    elif args.command == 'source':
        result = source_archive(args.root, args.output)
    elif args.command == 'plan':
        catalog = read(args.catalog/'catalog.json')
        names = selected(catalog, args.profile, args.package)
        result = {'packages': names, 'logical_bytes': sum(catalog['packages'][n]['logical_bytes'] for n in names),
                  'download_object_bytes': sum(catalog['packages'][n]['download_object_bytes'] for n in names)}
    elif args.command == 'pack':
        result = pack(args.root, args.catalog, args.output, profile=args.profile,
                      packages=args.package, part_bytes=args.part_mib << 20, reuse=args.reuse)
    else:
        result = install(args.catalog, args.downloads, args.base_url, args.target, args.cache,
                         profile=args.profile, packages=args.package)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
