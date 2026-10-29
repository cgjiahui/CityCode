"""Explicit visual scene acceptance for plausible image-led reconstructions.

This profile does not claim exact agreement with a partial source footprint.
It binds the Agent or human visual assessment to actual mesh, pose, source and scene
identities. Existing reconstruction acceptance thresholds remain unchanged.
"""
import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def placement_identity(record):
    return digest({'lod_sha256':{k:v['sha256'] for k,v in record['lods'].items()},
                   'world_from_local_4x4':record['world_from_local_4x4']})

def scene_identity(manifest):
    assets=sorted([{'id':r['id'],'placement_identity':placement_identity(r)}
                   for r in manifest['asset_instances']],key=lambda r:r['id'])
    surfaces=sorted([{'id':r['id'],'placement_identity':placement_identity(r)}
                     for r in manifest.get('surface_instances',[]) if r.get('assembly_enabled',True)],key=lambda r:r['id'])
    identity={'assets':assets,'default_surfaces':surfaces}
    if manifest.get('massing_instances'):
        identity['massing']=sorted([{'id':r['id'],'placement_identity':placement_identity(r)}
            for r in manifest['massing_instances']],key=lambda r:r['id'])
    return digest(identity)

def validate_scene_completion_review(root,record):
    from shapely.geometry import shape
    root=Path(root).resolve();review=record.get('scene_completion_review',{})
    if (review.get('schema')!='citycode.visual_scene_completion.v1'
            or review.get('placement_identity')!=placement_identity(record)):
        raise ValueError('Visual scene review is missing or belongs to different mesh/placement')
    if not review.get('reason') or not all(review.get(k) is True for k in
            ['real_building_confirmed','appearance_coherent','no_major_scene_conflict']):
        raise ValueError('Visual completion requires explicit observed-building, appearance and scene assessments')
    source=(root/record['structure_path']).resolve()
    if not source.is_relative_to(root.parent):
        raise ValueError('Visual completion source must remain in the packaged data tree')
    payload=source.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=review.get('source_building_sha256'):
        raise ValueError('Visual completion source evidence changed')
    building=json.loads(payload)
    if building['building_id']!=record['building_id'] or not shape(building['footprint_world']).equals(shape(record['target_geometry'])):
        raise ValueError('Visual completion must retain its original source footprint for diagnostics')
    files=review.get('proof_files',[])
    roles={p.get('role') for p in files}
    if not {'actual_front','actual_back','scene_intersections'}<=roles:
        raise ValueError('Visual completion needs actual front/back and scene intersection evidence')
    for item in files:
        relative=Path(item['path']);path=(root/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
            raise ValueError('Scene completion evidence must remain inside the asset package')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Visual completion proof bytes changed')
    return review

def validate_scene_completion_context(manifest):
    expected=scene_identity(manifest)
    for row in manifest['buildings']:
        if row.get('status')=='accepted_scene_completion':
            if row.get('scene_completion_review',{}).get('scene_identity')!=expected:
                raise ValueError('Visual completion scene changed; repeat placement/conflict review')
