"""Restore a frozen release cache without claiming a new neural generation."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

from .appearance import sha256_file,write_json
from .batch import read,proof,verify


def seal_cache(cache,input_manifest,output):
    """Packaging operation: freeze an already reviewed cache and all its inputs."""
    cache=Path(cache).resolve();inputs=Path(input_manifest).resolve();output=Path(output)
    binding=read(cache/'binding_manifest.json');frozen=read(cache/'reproduction/manifest.json')
    if frozen['binding_manifest']['sha256']!=sha256_file(cache/'binding_manifest.json'):
        raise ValueError('Cache binding differs from its reviewed reproduction inputs')
    if frozen['source_manifest']['sha256']!=sha256_file(inputs):
        raise ValueError('Cache belongs to a different upstream input')
    sources=[]
    for row in frozen['source_buildings']:
        path=verify(cache,row,cache.parent)
        sources.append({'building_id':row['building_id'],**proof(path,inputs.parent)})
    files=[proof(path,cache) for path in sorted(cache.rglob('*')) if path.is_file()]
    if any(Path(item['path']).is_absolute() or '..' in Path(item['path']).parts for item in files):
        raise ValueError('Cache inventory contains an escaping file')
    manifest={'schema':'citycode.frozen_asset_cache.v1','source_manifest_sha256':sha256_file(inputs),
        'source_buildings':sources,'source_count':len(sources),'files':files,
        'binding':proof(cache/'binding_manifest.json',cache),'physical_assets':len(binding['asset_instances']),
        'logical_entities':len(binding['building_entities']),'source_statuses':dict(Counter(r['status'] for r in binding['buildings'])),
        'original_model_provenance_preserved':True,'new_neural_generation_claimed':False}
    if output.exists() and read(output)!=manifest:raise ValueError('Frozen cache inventory differs; choose a new manifest')
    write_json(output,manifest)
    return {'status':'cache_sealed','files':len(files),'bytes':sum(r['bytes'] for r in files),
        'source_instances':len(sources),'physical_assets':manifest['physical_assets']}


def cache_inputs(config):
    project=Path(config['_project_root']);cache=Path(config['asset_cache']);root=Path(config['output'])
    frozen=read(config['asset_cache_manifest']);inputs=Path(config['input_manifest'])
    if cache==root or root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError('Restored output must be separate from the frozen cache')
    if frozen['schema']!='citycode.frozen_asset_cache.v1' or frozen['source_manifest_sha256']!=sha256_file(inputs):
        raise ValueError('Frozen cache requires the identical upstream structure input')
    upstream=read(inputs);source_ids=[r['building_id'] for r in frozen['source_buildings']]
    if (len(set(source_ids))!=frozen['source_count'] or set(source_ids)!={r['building_id'] for r in upstream['records']}):
        raise ValueError('Cache source inventory must cover every upstream source exactly once')
    for row in frozen['source_buildings']:verify(inputs.parent,row,project)
    identifiers=[r['path'] for r in frozen['files']]
    if len(set(identifiers))!=len(identifiers):raise ValueError('Duplicate cache inventory entries')
    for relative in identifiers:
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not (cache/path).resolve().is_relative_to(cache):
            raise ValueError('Cache file escaped the selected package')
    if frozen['binding']['path']!='binding_manifest.json' or frozen['binding'] not in frozen['files']:
        raise ValueError('Cache inventory does not include its canonical binding')
    verify(cache,frozen['binding'])
    return frozen


def cache_status(config):
    frozen=read(config['asset_cache_manifest']);root=Path(config['output']);complete=root/'CACHE_RESULT.json'
    if config.get('massing_completion_recipe') and (root/'massing_completion/RESULT.json').is_file():
        return read(root/'massing_completion/RESULT.json')
    result={'status':'cache_available_restore_pending','sources':frozen['source_count'],
        'physical_assets':frozen['physical_assets'],'logical_entities':frozen['logical_entities'],
        'source_statuses':frozen['source_statuses'],'API_calls':0,'GPU_jobs':0,'fresh_neural_models':0}
    if complete.exists():result.update(read(complete))
    elif (root/'RESTORE_PROGRESS.json').exists():result.update(read(root/'RESTORE_PROGRESS.json'))
    return result


def _copy_verified(source,destination,item):
    if destination.exists():
        if destination.stat().st_size!=item['bytes'] or sha256_file(destination)!=item['sha256']:
            raise ValueError('Restored output changed; choose a new output directory: '+item['path'])
        return
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.restore-partial')
    digest=hashlib.sha256();size=0
    with source.open('rb') as reader,temporary.open('wb') as writer:
        for block in iter(lambda:reader.read(8<<20),b''):
            writer.write(block);digest.update(block);size+=len(block)
    if size!=item['bytes'] or digest.hexdigest()!=item['sha256']:
        raise ValueError('Frozen cache file missing or changed: '+item['path'])
    temporary.replace(destination)


def restore_cache(config,*,replay_special=True):
    from .cli import validate_assets
    frozen=cache_inputs(config);cache=Path(config['asset_cache']);root=Path(config['output'])
    root.mkdir(parents=True,exist_ok=True)
    contract={'cache_manifest_sha256':sha256_file(config['asset_cache_manifest']),
        'source_manifest_sha256':sha256_file(config['input_manifest'])}
    contract_path=root/'CACHE_INPUTS.json'
    if contract_path.exists() and read(contract_path)!=contract:raise ValueError('Cache restoration input changed')
    write_json(contract_path,contract)
    binding=frozen['binding'];completed=0
    for item in frozen['files']:
        if item['path']=='binding_manifest.json':continue
        source=cache/item['path'];destination=root/item['path']
        if not source.is_file() or source.stat().st_size!=item['bytes']:raise ValueError('Frozen cache missing: '+item['path'])
        _copy_verified(source,destination,item);completed+=1
        if completed%100==0:
            progress={'status':'restoring_verified_cache','files_copied_or_verified':completed,'expected_files':len(frozen['files'])}
            write_json(root/'RESTORE_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    candidate=root/'binding_manifest.cache_candidate.json';_copy_verified(cache/binding['path'],candidate,binding)
    validation=validate_assets(candidate,require_complete=True,geometry=False)
    write_json(root/'audits/cache_restore/asset_validation.json',validation)
    if validation['status']!='passed' or not validation['complete']:
        raise ValueError('Restored binding failed full-source asset validation')
    replay=None
    if replay_special:
        # Replay uses the canonical filename, so expose it only in a temporary
        # complete tree after all bytes and metadata have been verified. Remove
        # it if any proof replay fails; no incomplete delivery is promoted.
        target=root/'binding_manifest.json';created=not target.exists();_copy_verified(cache/binding['path'],target,binding)
        try:
            script=Path(config['_project_root'])/'workflow/02_building_assets/audit_example.py'
            specification=importlib.util.spec_from_file_location('_citycode_cache_audit',script)
            module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
            replay,inventory=module.audit(root,verify_models=True,replay=True,orientation=True,cpu_repairs=True)
            write_json(root/'audits/cache_restore/special_replay.json',replay)
        except Exception:
            if created:target.unlink(missing_ok=True)
            raise
    else:_copy_verified(cache/binding['path'],root/'binding_manifest.json',binding)
    result={'schema':'citycode.verified_cache_execution.v1','status':'complete_verified_asset_cache',
        'source_instances':frozen['source_count'],'physical_assets':frozen['physical_assets'],
        'logical_entities':frozen['logical_entities'],'source_statuses':frozen['source_statuses'],
        'cached_physical_assets':frozen['physical_assets'],'fresh_neural_models':0,'API_calls':0,'GPU_jobs':0,
        'cache_files_copied_or_verified':len(frozen['files']),'binding_sha256':sha256_file(root/'binding_manifest.json'),
        'binding_identical_to_frozen_example':sha256_file(root/'binding_manifest.json')==binding['sha256'],
        'original_model_provenance_preserved':True,'special_CPU_replay_executed':replay is not None,
        'scope':'Frozen example asset restoration and reviewed CPU replay; no new image or neural mesh generation.'}
    write_json(root/'CACHE_RESULT.json',result);write_json(root/'RESTORE_PROGRESS.json',{'status':'complete','expected_files':len(frozen['files']),
        'files_copied_or_verified':len(frozen['files'])})
    return result


def execute_cache_phase(config,phase):
    if config.get('massing_completion_recipe') and phase not in {'prepare','images','freeze','gpu','review-images','review-models','status'}:
        from .massing_completion import apply_recipe
        base={**config,'output':config['asset_cache_restore_output']}
        restore_cache(base,replay_special=True)
        return apply_recipe(config,base['output'])
    if phase=='status':return cache_status(config)
    if phase=='prepare':
        frozen=cache_inputs(config)
        return {'status':'cache_inputs_verified','source_instances':frozen['source_count'],'physical_assets':frozen['physical_assets'],
            'API_calls':0,'GPU_jobs':0}
    if phase in {'images','freeze','gpu','review-images','review-models'}:
        return {'status':'not_required_for_verified_cache','phase':phase,'API_calls':0,'GPU_jobs':0,
            'reason':'Frozen images, model bytes, placements and reviews are restored together.'}
    return restore_cache(config,replay_special=True)
