#!/usr/bin/env python3
"""Create expendable textured city preview LODs on CPU; never edit release assets.

Requires Pillow and Node.js. Geometry uses the release's pinned MIT meshoptimizer.
Approximately 2,500 triangles and 256px base-color textures per building. These
view-only copies are never used for reconstruction evaluation or stage04 delivery.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from PIL import Image

HERE = Path(__file__).resolve().parent


def convert(job):
    root, output, item = job
    source = root/item['lods']['0.1M']['path']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != item['lods']['0.1M']['sha256']: raise ValueError(f'Changed source: {source}')
    destination = output/f'{item["id"]}.glb'
    proof = destination.with_suffix('.json')
    if proof.is_file() and destination.is_file():
        record = json.loads(proof.read_text())
        if record.get('source_sha256') == digest and record.get('preview_sha256') == hashlib.sha256(destination.read_bytes()).hexdigest(): return record
    data = source.read_bytes(); json_length = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20+json_length]); binary_offset = 28+json_length
    output_doc = copy.deepcopy(doc); output_doc['bufferViews'] = []; output_doc['accessors'] = []
    binary = bytearray()
    def blob(value, target=None):
        binary.extend(b'\0'*(-len(binary)%4))
        view = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(value)}
        if target: view['target'] = target
        n = len(output_doc['bufferViews']); output_doc['bufferViews'].append(view); binary.extend(value); return n
    with tempfile.TemporaryDirectory(prefix='citycode-web-lod-') as temporary:
        prefix = Path(temporary)/'geometry'
        subprocess.run(['node', str(HERE/'preview_geometry.cjs'), str(source), str(prefix)], check=True, capture_output=True, timeout=120)
        records = json.loads(prefix.with_suffix('.json').read_text()); payload = prefix.with_suffix('.bin').read_bytes()
    faces = 0
    for r in records:
        primitive = output_doc['meshes'][r['mesh']]['primitives'][r['primitive']]; primitive['attributes'] = {}
        for name, a in r['attributes'].items():
            width = a['width']; index = len(output_doc['accessors'])
            output_doc['accessors'].append({'bufferView': blob(payload[a['offset']:a['offset']+a['bytes']],34962),
                'componentType':5126,'count':a['count'],'type':f'VEC{width}' if width>1 else 'SCALAR','min':a['min'],'max':a['max']})
            primitive['attributes'][name] = index
        a = r['index']; primitive['indices'] = len(output_doc['accessors'])
        output_doc['accessors'].append({'bufferView': blob(payload[a['offset']:a['offset']+a['bytes']],34963),
            'componentType':5125,'count':a['count'],'type':'SCALAR'})
        faces += r['faces']
    output_doc['images'] = []; output_doc['textures'] = []; texture_map = {}
    for material in output_doc.get('materials',[]):
        for key in ['normalTexture','occlusionTexture','emissiveTexture','extensions']:
            material.pop(key,None)
        material['emissiveFactor']=[0,0,0]
        pbr=material.setdefault('pbrMetallicRoughness',{});pbr.pop('metallicRoughnessTexture',None)
        pbr.update(metallicFactor=0,roughnessFactor=.9)
        reference=pbr.get('baseColorTexture')
        if not reference: continue
        old_index=reference['index']
        if old_index not in texture_map:
            texture=doc['textures'][old_index]
            source_index=texture.get('source',texture.get('extensions',{}).get('EXT_texture_webp',{}).get('source'))
            if source_index is None:raise ValueError('Unsupported embedded texture extension')
            image=doc['images'][source_index]
            view=doc['bufferViews'][image['bufferView']];offset=binary_offset+view.get('byteOffset',0)
            with Image.open(io.BytesIO(data[offset:offset+view['byteLength']])) as image:
                image.thumbnail((256,256),Image.Resampling.LANCZOS)
                has_alpha=image.mode=='RGBA' and image.getextrema()[3][0]<255
                stream=io.BytesIO();image.convert('RGBA' if has_alpha else 'RGB').save(stream,format='PNG' if has_alpha else 'JPEG',quality=75,optimize=True)
            image_index=len(output_doc['images']);output_doc['images'].append({'bufferView':blob(stream.getvalue()),'mimeType':'image/png' if has_alpha else 'image/jpeg'})
            new_texture={'source':image_index}
            if 'sampler' in texture:new_texture['sampler']=texture['sampler']
            texture_map[old_index]=len(output_doc['textures']);output_doc['textures'].append(new_texture)
        reference['index']=texture_map[old_index]
    for name in ['extensionsUsed','extensionsRequired']:
        if name in output_doc:
            output_doc[name]=[e for e in output_doc[name] if e!='EXT_texture_webp']
            if not output_doc[name]:output_doc.pop(name)
    output_doc['buffers']=[{'byteLength':len(binary)}]
    output_doc['asset'].setdefault('extras',{}).update({'view_only':True,'source_sha256':digest,'texture_max_px':256})
    encoded=json.dumps(output_doc,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%4);binary+=b'\0'*(-len(binary)%4)
    body=struct.pack('<I4s',len(encoded),b'JSON')+encoded+struct.pack('<I4s',len(binary),b'BIN\0')+binary
    result=struct.pack('<4sII',b'glTF',2,len(body)+12)+body
    destination.write_bytes(result)
    record={'id':item['id'],'path':destination.name,'bytes':len(result),'faces':faces,
        'source_sha256':digest,'preview_sha256':hashlib.sha256(result).hexdigest(),
        'view_only':True,'node_transforms_preserved':True,'texture_max_px':256}
    proof.write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root',type=Path,default=HERE.parent/'data/smallcity_p384_building_assets_completed')
    parser.add_argument('--output',type=Path,default=HERE.parent/'data/smallcity_p384_building_assets_completed_web_preview')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();root=args.assets_root.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    binding_path=root/'binding_manifest_physical.json'
    if not binding_path.is_file():binding_path=root/'binding_manifest.json'
    binding_bytes=binding_path.read_bytes();binding=json.loads(binding_bytes)
    items=[item for item in binding['asset_instances']+binding.get('massing_instances',[]) if item.get('assembly_enabled',True)]
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(convert,(root,output,item)) for item in items]
        for future in as_completed(tasks):
            records.append(future.result())
            if len(records)%25==0 or len(records)==len(items):print(f'{len(records)}/{len(items)} preview copies prepared',flush=True)
    if hashlib.sha256(binding_path.read_bytes()).hexdigest()!=hashlib.sha256(binding_bytes).hexdigest():raise ValueError('Binding changed during preparation')
    manifest={'schema':'citycode.web_preview_lods.v1','view_only':True,'binding_sha256':hashlib.sha256(binding_bytes).hexdigest(),
        'count':len(records),'bytes':sum(r['bytes'] for r in records),'faces':sum(r['faces'] for r in records),
        'elapsed_seconds':round(time.monotonic()-start,3),'assets':sorted(records,key=lambda r:r['id'])}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='assets'}),flush=True)


if __name__=='__main__':main()
