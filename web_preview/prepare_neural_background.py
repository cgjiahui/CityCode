#!/usr/bin/env python3
"""Derive a small textured viewer model from the published Pixal3D GLB on CPU.

Reads one mesh at a time. Keeps all generated/underlay mesh parts, original UV
orientation and world positions; simplifies geometry and base-color textures.
No inference, image API, historical experiment path or source mutation.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

from prepare_city import convert

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_glb(path, doc, binary):
    doc['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\0' * (-len(binary) % 4)
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary))
                     + struct.pack('<I4s', len(encoded), b'JSON') + encoded
                     + struct.pack('<I4s', len(binary), b'BIN\0') + binary)


def extract_mesh(stream, source, binary_offset, mesh_index, path):
    mesh = source['meshes'][mesh_index]
    if len(mesh['primitives']) != 1:
        raise ValueError('Expected one primitive per historical mesh')
    primitive = copy.deepcopy(mesh['primitives'][0])
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'mesh': 0, 'name': mesh['name']}],
           'meshes': [{'name': mesh['name'], 'primitives': [primitive]}],
           'bufferViews': [], 'accessors': []}
    binary = bytearray()
    def view(index):
        v = copy.deepcopy(source['bufferViews'][index])
        stream.seek(binary_offset + v.get('byteOffset', 0))
        binary.extend(b'\0' * (-len(binary) % 4))
        offset = len(binary)
        binary.extend(stream.read(v['byteLength']))
        v.update(buffer=0, byteOffset=offset)
        doc['bufferViews'].append(v)
        return len(doc['bufferViews']) - 1
    def accessor(index):
        a = copy.deepcopy(source['accessors'][index])
        if a.get('sparse'):
            raise ValueError('Sparse source accessor not supported')
        a['bufferView'] = view(a['bufferView'])
        doc['accessors'].append(a)
        return len(doc['accessors']) - 1
    primitive['attributes'] = {k: accessor(v) for k, v in primitive['attributes'].items()}
    primitive['indices'] = accessor(primitive['indices'])
    material = source['materials'][primitive['material']]
    pbr = copy.deepcopy(material['pbrMetallicRoughness'])
    pbr.pop('metallicRoughnessTexture', None)
    pbr.update(metallicFactor=0, roughnessFactor=.9)
    tex = source['textures'][pbr['baseColorTexture']['index']]
    image_index = tex.get('source', tex.get('extensions', {}).get('EXT_texture_webp', {}).get('source'))
    im = copy.deepcopy(source['images'][image_index])
    im['bufferView'] = view(im['bufferView'])
    doc['images'] = [im]
    doc['textures'] = [{'source': 0}]
    pbr['baseColorTexture']['index'] = 0
    doc['materials'] = [{'pbrMetallicRoughness': pbr, 'doubleSided': True}]
    primitive['material'] = 0
    write_glb(path, doc, binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'data/smallcity_p384_background_history')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data/smallcity_p384_background_web_preview')
    args = parser.parse_args()
    source = args.bundle.resolve() / 'reference_mesh/pixal3d_background.glb'
    output = args.output_root.resolve()
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    destination = output / 'pixal3d.glb'
    if destination.exists():
        raise ValueError('Refusing to overwrite an existing neural preview')
    expected = next(r['sha256'] for r in json.loads((args.bundle / 'CHECKSUMS.json').read_text())['files']
                    if r['path'] == 'reference_mesh/pixal3d_background.glb')
    if sha(source) != expected:
        raise ValueError('Published Pixal3D source checksum mismatch')
    module_path = ROOT / 'pipelines/background_pixal3d/experiments/pixal3d_p384_v5/merge_fused_tiles_glb.py'
    spec = importlib.util.spec_from_file_location('glb_merge', module_path)
    merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merger)
    doc, offset, _ = merger.parse_glb(source)
    names = [m['name'] for m in doc['meshes']]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Expected unique, filename-safe mesh part names')
    if any(any(k in n for k in ('matrix', 'translation', 'rotation', 'scale')) for n in doc['nodes']):
        raise ValueError('This historical source must already have world-space mesh coordinates')
    records = []
    with tempfile.TemporaryDirectory(prefix='.pixal3d-preparation-', dir=output) as temp:
        work = Path(temp)
        lods = work / 'lods'
        lods.mkdir()
        with source.open('rb') as stream:
            for i, mesh in enumerate(doc['meshes']):
                part = work / 'part.glb'
                extract_mesh(stream, doc, offset, i, part)
                item = {'id': mesh['name'], 'lods': {'0.1M': {'path': part.name, 'sha256': sha(part)}}}
                record = convert((work, lods, item))
                old = doc['accessors'][mesh['primitives'][0]['attributes']['POSITION']]
                new_doc, _, _ = merger.parse_glb(lods / record['path'])
                new = new_doc['accessors'][new_doc['meshes'][0]['primitives'][0]['attributes']['POSITION']]
                if any(new['min'][j] < old['min'][j] - .001 or new['max'][j] > old['max'][j] + .001 for j in range(3)):
                    raise ValueError(f'Preview escaped original bounds: {mesh["name"]}')
                record['source_bounds'] = [old['min'], old['max']]
                record['preview_bounds'] = [new['min'], new['max']]
                record['source_faces'] = doc['accessors'][mesh['primitives'][0]['indices']]['count'] // 3
                records.append(record)
                if (i + 1) % 20 == 0 or i + 1 == len(doc['meshes']):
                    print(f'Prepared {i + 1}/{len(doc["meshes"])} mesh parts', flush=True)
        sources = [(lods / r['path'], *merger.parse_glb(lods / r['path'])) for r in records]
        faces = sum(r['faces'] for r in records)
        combined, bases = merger.merge_documents(sources, faces)
        merger.write_glb(destination, combined, sources, bases)
    if sha(source) != expected:
        raise ValueError('Source changed during preparation')
    proof = {'view_only': True, 'source_sha256': expected, 'source_faces': sum(r['source_faces'] for r in records),
             'faces': faces, 'mesh_parts': len(records), 'texture_max_px': 256,
             'source_coordinates': 'glTF (UE X, UE Z-height, UE Y)',
             'world_from_source_4x4': [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
             'all_mesh_parts_preserved': True, 'per_mesh_bounds_within_source': True,
             'geometry_simplified': True, 'materials': 'base color only; no metallic/roughness maps',
             'api_calls': 0, 'gpu_inference': False, 'parts': records}
    (output / 'pixal3d.json').write_text(json.dumps(proof, indent=2) + '\n')
    manifest['files']['neural'] = {'path': destination.name, 'sha256': sha(destination), 'bytes': destination.stat().st_size}
    manifest['neural'] = {k: v for k, v in proof.items() if k != 'parts'}
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    print(json.dumps({'status': 'complete', 'faces': faces, 'bytes': destination.stat().st_size}))


if __name__ == '__main__':
    main()
