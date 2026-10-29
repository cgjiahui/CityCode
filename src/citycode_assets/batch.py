"""One source-complete, resumable building-generation workflow.

Result recovery and new-source generation are explicit modes. GPU and image
service operations are separate phases; neither is started by prepare.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shutil

from .appearance import completion_contract, prepare_building, sha256_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def proof(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return {'path': os.path.relpath(path, root), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def verify(root, item, allowed_root=None):
    root = Path(root).resolve(); allowed = Path(allowed_root or root).resolve()
    relative = Path(item['path']); path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError('File is outside the selected package: ' + str(relative))
    if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
        raise ValueError('Frozen file changed: ' + str(relative))
    return path


def load_config(path):
    path = Path(path).resolve(); config = read(path)
    project = (path.parent/config['project_root']).resolve()
    result = {**config, '_config_path': str(path), '_project_root': str(project)}
    result['execution_mode'] = config.get('execution_mode', 'verified_asset_cache' if config.get('asset_cache') else 'generate')
    if result['execution_mode'] not in {'verified_asset_cache', 'generate'}:
        raise ValueError('execution_mode must be verified_asset_cache or generate')
    if result['execution_mode'] == 'verified_asset_cache' and not config.get('asset_cache'):
        raise ValueError('verified_asset_cache requires the provided asset_cache')
    if result['execution_mode'] == 'generate':
        if config.get('asset_cache') or config.get('massing_completion_recipe'):
            raise ValueError('New generation cannot inherit an asset cache or the example massing recipe; initialize a new config')
        from .lod import generation_lod_targets
        generation_lod_targets(config)
    for key in ['input_manifest', 'dispositions', 'provider_config', 'output', 'runtime_config']:
        result[key] = str((project/config[key]).resolve())
    for key in ['asset_cache','asset_cache_manifest','asset_cache_restore_output','completed_images_manifest']:
        if config.get(key):result[key]=str((project/config[key]).resolve())
    # Asset validators allow evidence in sibling data packages. Keep this
    # relationship explicit instead of depending on a research working folder.
    if Path(result['output']).parent != Path(result['input_manifest']).parents[2]:
        raise ValueError('Output must be a sibling of the upstream structure package')
    return result


def input_contract(config):
    from .lod import generation_lod_targets
    result = {'schema':'citycode.unified_batch_contract.v1',
        'input_manifest_sha256':sha256_file(config['input_manifest']),
        'dispositions_sha256':sha256_file(config['dispositions']),
        'provider_config_sha256':sha256_file(config['provider_config']),
        'execution_mode':config.get('execution_mode','generate'),
        'lod_targets':list(generation_lod_targets(config))}
    if config.get('completed_images_manifest'):
        result['completed_images_manifest_sha256']=sha256_file(config['completed_images_manifest'])
    return result


def check_plan(config):
    plan=read(Path(config['output'])/'plan.json')
    if plan['contract']!=input_contract(config):
        raise ValueError('Run inputs changed; choose a new output directory')
    identifiers=[r['asset_id'] for r in plan['records']]
    if len(set(identifiers))!=len(identifiers) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in identifiers):
        raise ValueError('Physical asset IDs must be unique safe directory names')
    return plan


def _group_building(source_paths, destination):
    """Create a derived group input, retaining every upstream file unchanged."""
    import numpy as np
    import trimesh
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from .appearance import load_mesh

    buildings = [read(p) for p in source_paths]
    shutil.copytree(source_paths[0].parent, destination)
    merged = copy.deepcopy(buildings[0])
    geometry = unary_union([shape(b['footprint_world']) for b in buildings])
    merged.update(footprint_world=mapping(geometry), base_z_m=min(b['base_z_m'] for b in buildings),
                  top_z_m=max(b['top_z_m'] for b in buildings),
                  roof_layers=[layer for b in buildings for layer in b.get('roof_layers', [])])
    merged['height_m'] = merged['top_z_m']-merged['base_z_m']
    meshes = []
    inverse = np.asarray(merged['local_from_world_4x4'])
    for path, building in zip(source_paths, buildings):
        mesh = load_mesh(path.parent/building['paths']['proxy_local_ue_z_up'])
        mesh.apply_transform(inverse @ np.asarray(building['world_from_local_4x4']))
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    combined.export(destination/merged['paths']['proxy_local_ue_z_up'])
    merged['source_group'] = {'members': [b['building_id'] for b in buildings],
        'qualification': 'reviewed single physical building; source footprints unioned for one generation'}
    write_json(destination/'building.json', merged)
    return destination/'building.json'


def _copy_conditioning(source, destination, building_path):
    prepared, receipt = read(source/'appearance_input.json'), read(source/'completion.json')
    if (prepared['building_id'] != read(building_path)['building_id']
            or prepared['provenance']['building_sha256'] != sha256_file(building_path)
            or receipt['status'] != 'complete' or receipt['request_identity_sha256'] != completion_contract(prepared)
            or receipt['output_sha256'] != sha256_file(source/'generated_main_oblique.png')):
        raise ValueError('Conditioning cache is not bound to the current source')
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if (not path.is_file() or len(relative.parts)>2
                or (len(relative.parts)==2 and relative.parts[0]!='inputs')
                or path.suffix.lower() not in {'.json', '.txt', '.png', '.jpg', '.jpeg'}):
            continue
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target)!=sha256_file(path):
            raise ValueError('Conditioning output changed during resume')
        if not target.exists(): shutil.copyfile(path, target)
    return prepared


def _prepare_one(task):
    building, folder, provider, cached = task
    building, folder = Path(building), Path(folder)
    if cached:
        prepared = _copy_conditioning(Path(cached), folder, building)
    else:
        prepared = prepare_building(building, folder, provider_config=provider)
    return {'building_id': prepared['building_id'], 'prepared_input': str(folder/'appearance_input.json'),
            'request_identity_sha256': completion_contract(prepared), 'cached_conditioning': bool(cached)}


def prepare(config, *, workers=4):
    root = Path(config['output']); project = Path(config['_project_root'])
    upstream = read(config['input_manifest']); dispositions = read(config['dispositions'])
    if dispositions.get('input_manifest_sha256') != sha256_file(config['input_manifest']):
        raise ValueError('Disposition decisions belong to a different upstream input')
    sources = {r['building_id']: r for r in upstream['records']}
    decisions = {r['building_id']: r for r in dispositions['sources']}
    if (len(sources)!=upstream['building_count'] or set(sources)!=set(decisions)
            or len(decisions)!=len(dispositions['sources'])):
        raise ValueError('Disposition decisions must cover every source exactly once')
    if any(r['action'] not in {'generate', 'observed_surface', 'record_only'} for r in decisions.values()):
        raise ValueError('Unknown source action')
    from .generation import image_height_span
    for rule in decisions.values():
        if rule['action']=='generate':image_height_span(rule)
    contract = input_contract(config)
    root.mkdir(parents=True, exist_ok=True)
    plan_path=root/'plan.json'
    if plan_path.exists() and read(plan_path)['contract']!=contract:
        raise ValueError('Run inputs changed; choose a new output directory')
    groups={}; owners={}
    for group in dispositions.get('shared_physical_groups', []):
        members=group['source_ids']
        if group['asset_id'] in groups or group['asset_id'] in sources:
            raise ValueError('Shared asset ID must be distinct from source and other group IDs')
        if len(members)<2 or len(set(members))!=len(members) or any(x in owners or x not in sources for x in members):
            raise ValueError('Shared groups require unique, existing source members')
        if any(decisions[x]['action']!='generate' for x in members):
            raise ValueError('Only complete building sources can share a generated asset')
        groups[group['asset_id']]=group
        owners.update({sid:group['asset_id'] for sid in members})
    records=[]; processed=set(); tasks=[]; provider=read(config['provider_config'])
    for sid in sources:
        if decisions[sid]['action']!='generate':continue
        aid=owners.get(sid,sid)
        if not re.fullmatch(r'[A-Za-z0-9_-]+',aid):raise ValueError('Unsafe physical asset ID')
        if aid in processed:continue
        processed.add(aid); members=groups.get(aid,{}).get('source_ids',[sid]); representative=members[0]
        paths=[Path(config['input_manifest']).parent/sources[x]['path']/'building.json' for x in members]
        if any(read(p)['building_id']!=member for member,p in zip(members,paths)):
            raise ValueError('Upstream file and source manifest IDs differ')
        if len(paths)>1:
            target=root/'conditioning_sources'/aid
            building=target/'building.json' if target.exists() else _group_building(paths,target)
        else:building=paths[0]
        rule=decisions[representative]; cached=rule.get('conditioning_cache')
        if len(members)>1:cached=None
        cached=str(project/cached) if cached else None
        folder=root/'appearance'/aid
        record={'asset_id':aid,'building_id':representative,'source_ids':members,
            'building':proof(building,root),'source_buildings':[{'building_id':x,**proof(p,root)} for x,p in zip(members,paths)],
            'appearance_directory':str(folder.relative_to(root)),
            'structure_conditioning':rule.get('structure_conditioning','source_footprint'),
            'validation_profile':rule.get('validation_profile','source_reconstruction'),
            'postprocess':rule.get('postprocess',{}),'conditioning_cached':bool(cached)}
        if record['structure_conditioning']=='image_height_span':
            record['structure_height_voxels']=image_height_span(rule)
        records.append(record);tasks.append((str(building),str(folder),provider,cached))
    plan={'schema':'citycode.unified_building_batch.v1','contract':contract,'source_count':len(sources),
          'source_manifest':proof(config['input_manifest'],root),'dispositions':proof(config['dispositions'],root),
          'records':records,'source_actions':dict(Counter(r['action'] for r in decisions.values())),
          'physical_generations':len(records),'historical_model_inputs':0,
          'all_source_records_preserved':True,'API_calls_during_prepare':0,'GPU_jobs_during_prepare':0}
    write_json(plan_path,plan)
    completed=[]; errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_prepare_one,t):r for t,r in zip(tasks,records)}
        for future in as_completed(futures):
            record=futures[future]
            try:
                value=future.result(); completed.append(value)
                print(json.dumps({'asset_id':record['asset_id'],'phase':'prepare','status':'complete'}),flush=True)
            except Exception as exc:
                errors.append({'asset_id':record['asset_id'],'error':str(exc)})
                print(json.dumps({'asset_id':record['asset_id'],'phase':'prepare','error':str(exc)}),flush=True)
            write_json(root/'PREPARATION.json',{'status':'complete' if len(completed)+len(errors)==len(records) and not errors else 'in_progress',
                'expected':len(records),'complete':len(completed),'errors':errors})
    if errors:raise ValueError('Input preparation failed for '+str(len(errors))+' assets')
    if config.get('completed_images_manifest'):
        from .new_inputs import import_completed_images
        write_json(root/'PREPARATION.json',{'status':'importing_external_images',
            'expected':len(records),'complete':len(completed),'errors':[]})
        import_completed_images(config,plan)
    write_json(root/'PREPARATION.json',{'status':'complete','expected':len(records),
        'complete':len(completed),'errors':[]})
    return status(config)


def status(config):
    root=Path(config['output']);plan=read(root/'plan.json');counts=Counter();records=[]
    failures={}
    for phase in ['infer','export']:
        for path in (root/'execution').glob(phase+'_rank*.json'):
            for value in read(path).get('records',[]):
                if value.get('status')!='complete':failures[value['asset_id']]='generation_failed' if phase=='infer' else 'export_failed'
    for row in plan['records']:
        folder=root/row['appearance_directory'];state='unprepared';completion=folder/'completion.json'
        if (folder/'appearance_input.json').exists():state='prepared'
        if completion.exists():state='image_complete' if read(completion).get('status')=='complete' else 'image_failed'
        raw=root/'generation'/row['asset_id']/'raw_manifest.json'
        if raw.exists():state='generated' if read(raw).get('status')=='complete' else 'generation_failed'
        if (root/'assets'/row['asset_id']/'export_manifest.json').exists():state='exported'
        if row['asset_id'] in failures:state=failures[row['asset_id']]
        result=root/'validation'/row['asset_id']/'RESULT.json'
        if result.exists():state=read(result)['status']
        counts[state]+=1;records.append({'asset_id':row['asset_id'],'status':state})
    report={'schema':'citycode.unified_batch_status.v1','sources':plan['source_count'],
        'physical_generations':plan['physical_generations'],'source_actions':plan['source_actions'],
        'historical_model_inputs':0,'counts':dict(counts),'records':records}
    partials=root/'nonbuilding_sources.json'
    if partials.exists():report['nonbuilding_counts']=dict(Counter(r['status'] for r in read(partials)['source_records']))
    write_json(root/'STATUS.json',report)
    return report


def complete_images(config, *, workers=4):
    from .appearance import complete_building
    root=Path(config['output']);plan=check_plan(config);provider=read(config['provider_config'])
    workers=min(workers,config.get('provider_maximum_parallel',4))
    outbound_path=root/'OUTBOUND_IMAGE_REQUESTS.json'
    if outbound_path.exists():
        outbound=read(outbound_path);requests={r['asset_id']:r for r in outbound['records']}
        expected={r['asset_id'] for r in plan['records'] if not r['conditioning_cached']}
        if set(requests)!=expected or len(requests)!=outbound['maximum_new_requests']:
            raise ValueError('Prepared outbound request set changed')
        if outbound['provider_config_sha256']!=sha256_file(config['provider_config']):
            raise ValueError('Prepared outbound provider configuration changed')
        for row in plan['records']:
            if row['asset_id'] not in requests:continue
            prepared=read(root/row['appearance_directory']/'appearance_input.json')
            if completion_contract(prepared)!=requests[row['asset_id']]['request_identity_sha256']:
                raise ValueError('Prepared outbound request identity changed')
            for reference in requests[row['asset_id']]['references']:verify(root,reference)
    def one(row):
        folder=root/row['appearance_directory']
        prior=folder/'completion.json'
        if prior.exists():
            if read(prior).get('status')!='complete':
                return {'asset_id':row['asset_id'],'status':'previous_failure_requires_explicit_retry'}
            receipt=read(prior);prepared=read(folder/'appearance_input.json')
            if (receipt.get('request_identity_sha256')!=completion_contract(prepared)
                    or sha256_file(folder/'generated_main_oblique.png')!=receipt.get('output_sha256')):
                return {'asset_id':row['asset_id'],'status':'failed','error':'Cached image identity changed'}
            return {'asset_id':row['asset_id'],'status':'cached'}
        if (folder/'request.json').exists():
            return {'asset_id':row['asset_id'],'status':'previous_request_unresolved',
                'error':'Interrupted request has no receipt; check provider outcome before explicit retry'}
        try:
            result=complete_building(folder,provider_config=provider)
            return {'asset_id':row['asset_id'],'status':result['status'],'image_sha256':result['output_sha256']}
        except Exception as exc:
            response=folder/'provider_response.json'
            return {'asset_id':row['asset_id'],'status':'failed','error':str(exc),
                'http_status':read(response).get('status_code') if response.exists() else None}
    results=[];queue=iter(plan['records']);consecutive_failures=0;stopped=False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active={pool.submit(one,row) for row in [next(queue,None) for _ in range(workers)] if row is not None}
        while active:
            ready,active=wait(active,return_when=FIRST_COMPLETED)
            for future in ready:
                value=future.result();results.append(value)
                if value['status']=='complete':consecutive_failures=0
                elif value['status']=='failed':consecutive_failures+=1
                if consecutive_failures>=3 or value.get('http_status') in {401,403,404}:
                    stopped=True
                print(json.dumps(value),flush=True)
            if not stopped:
                for _ in range(workers-len(active)):
                    row=next(queue,None)
                    if row is not None:active.add(pool.submit(one,row))
            write_json(root/'IMAGE_EXECUTION.json',{'records':results,'dispatch_stopped':stopped,
                'automatic_retries':0,'maximum_new_requests':sum(not r['conditioning_cached'] for r in plan['records'])})
    return status(config)


def freeze(config):
    from .batch_resources import resource_contract
    root=Path(config['output']);plan=check_plan(config);review=read(root/'reviews/appearance.json')
    decisions={r['asset_id']:r for r in review['records']}
    if len(decisions)!=len(review['records']) or set(decisions)!= {r['asset_id'] for r in plan['records']}:
        raise ValueError('Appearance review must cover the entire physical batch')
    for row in plan['records']:
        folder=root/row['appearance_directory'];prepared=read(folder/'appearance_input.json');receipt=read(folder/'completion.json')
        item=decisions[row['asset_id']];image=folder/'generated_main_oblique.png'
        if (item['decision'] not in {'generate','hold'} or not item.get('reason')
                or receipt.get('status')!='complete'
                or item['image_sha256']!=sha256_file(image)
                or receipt['output_sha256']!=item['image_sha256']
                or receipt['request_identity_sha256']!=completion_contract(prepared)):
            raise ValueError('Image review or completed request identity changed: '+row['asset_id'])
        row['image']=proof(image,root);row['prepared_input']=proof(folder/'appearance_input.json',root)
        row['completion_receipt']=proof(folder/'completion.json',root);row['image_decision']=item['decision']
    policy=read(config['dispositions']);project=Path(config['_project_root'])
    auxiliary=[project/policy['fragment_recipe_package']] if policy.get('fragment_recipe_package') else []
    for rule in policy['sources']:
        for key in ['reviewed_source_orientation','reviewed_cross_lod_recipe']:
            if rule.get('postprocess',{}).get(key):auxiliary.append(project/rule['postprocess'][key])
    auxiliary+=list(p for p in (root/'postprocess_support').rglob('*') if p.is_file())
    inputs=Path(config['input_manifest'])
    auxiliary += [inputs.parent/r['path']/'building.json' for r in read(inputs)['records']]
    frozen={**plan,'plan_sha256':sha256_file(root/'plan.json'),'review':proof(root/'reviews/appearance.json',root),
            'auxiliary_inputs':[proof(p,root) for p in sorted(set(auxiliary))],
            'records':plan['records'],'execution_code_sha256':sha256_file(Path(__file__)),
            'gpu_execution_code_sha256':sha256_file(Path(__file__).with_name('batch_gpu.py')),
            'generation_code_sha256':sha256_file(Path(__file__).with_name('generation.py')),
            'runtime_resources':resource_contract(config,full_hash=True),
            'status':'reviewed_ready','selected_generations':sum(r['image_decision']=='generate' for r in plan['records'])}
    path=root/'FROZEN.json'
    if path.exists() and read(path)!=frozen:raise ValueError('Frozen selection changed; choose a new run')
    write_json(path,frozen);return {'status':'reviewed_ready','generations':frozen['selected_generations'],'held':len(plan['records'])-frozen['selected_generations']}


def run(config,*,workers=4,gpu_count=4,deadline_seconds=None):
    """Resume until a real image/review/CUDA requirement is reached.

    Sending new images is always the explicit images phase. The GPU phase
    exits after exports, allowing cloud allocations to be released before CPU QA.
    Frozen example inputs/reviews can be supplied in the config for offline runs.
    """
    root=Path(config['output']);project=Path(config['_project_root'])
    prepared=root/'PREPARATION.json'
    if not prepared.exists() or read(prepared)['status']!='complete':prepare(config,workers=workers)
    plan=check_plan(config)
    missing=[r['asset_id'] for r in plan['records'] if not (root/r['appearance_directory']/'completion.json').exists()
             or read(root/r['appearance_directory']/'completion.json').get('status')!='complete']
    if missing:return {'status':'awaiting_explicit_image_completion','remaining_images':len(missing),'next_phase':'images','GPU_allocated':False}
    for name,key in [('appearance','appearance_review'),('models','model_review')]:
        target=root/'reviews'/(name+'.json');source=config.get(key)
        if source and not target.exists() and (project/source).is_file():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/source,target)
    if not (root/'reviews/appearance.json').exists():
        return {'status':'awaiting_appearance_review','next_phase':'freeze','physical_assets':len(plan['records'])}
    if not (root/'FROZEN.json').exists():freeze(config)
    from .batch_gpu import frozen_selection
    selected=frozen_selection(config)
    exported=sum((root/'assets'/r['asset_id']/'export_manifest.json').exists() for r in selected)
    if exported<len(selected):
        try:
            import torch
            cuda=torch.cuda.is_available()
        except ImportError:cuda=False
        if not cuda:return {'status':'awaiting_allocated_CUDA_worker','remaining_exports':len(selected)-exported,'next_phase':'gpu'}
        from .batch_gpu import run_gpu_batch
        return run_gpu_batch(config,gpu_count=gpu_count,deadline_seconds=deadline_seconds)
    from .batch_delivery import validate_batch,finalize_batch
    if not (root/'VALIDATION_PROGRESS.json').exists() or read(root/'VALIDATION_PROGRESS.json')['complete']<len(selected):
        validate_batch(config,workers=workers)
    if not (root/'reviews/models.json').exists():
        return {'status':'awaiting_current_model_review','next_phase':'finalize','numeric_states':read(root/'VALIDATION_PROGRESS.json')['counts']}
    return finalize_batch(config)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['init','run','prepare','review-images','images','freeze','gpu','validate','review-models','finalize','status'],default='run')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--gpu-count',type=int,default=4)
    parser.add_argument('--deadline-seconds',type=int,default=None, help='Optional explicit GPU runtime limit; omitted means finish the task without a fixed cutoff')
    parser.add_argument('--input-manifest',type=Path,help='init: manifest exported by structure recovery')
    parser.add_argument('--project-root',type=Path,help='init: release root, default current working directory')
    parser.add_argument('--output',type=Path,help='init: new output beside the structure package')
    images=parser.add_mutually_exclusive_group()
    images.add_argument('--completed-images',type=Path,help='init: directory of <building_id>.png images completed by your own service')
    images.add_argument('--completed-images-manifest',type=Path,help='init: explicit source/image identity mapping')
    parser.add_argument('--lod-targets',type=int,nargs='+',choices=[100000,250000,1000000],
        help='init: triangle budgets, default 1000000 250000 100000')
    args=parser.parse_args(argv)
    if not 1<=args.workers<=16:raise ValueError('Worker count must be 1 to16')
    if args.phase=='init':
        if not args.input_manifest or not args.output:parser.error('init requires --input-manifest and --output')
        from .new_inputs import initialize_generation
        result=initialize_generation(args.config,args.input_manifest,args.output,
            project_root=args.project_root or Path.cwd(),completed_images=args.completed_images,
            completed_images_manifest=args.completed_images_manifest,
            lod_targets=args.lod_targets if args.lod_targets is not None else [1000000,250000,100000])
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if any(value is not None for value in [args.input_manifest,args.project_root,args.output,args.completed_images,args.completed_images_manifest,args.lod_targets]):
        parser.error('Input initialization options are only used with --phase init; subsequent phases read the saved config')
    config=load_config(args.config)
    if config['execution_mode']=='verified_asset_cache':
        from .cache_batch import execute_cache_phase
        result=execute_cache_phase(config,args.phase)
    elif args.phase=='run':result=run(config,workers=args.workers,gpu_count=args.gpu_count,deadline_seconds=args.deadline_seconds)
    elif args.phase=='prepare':result=prepare(config,workers=args.workers)
    elif args.phase=='images':result=complete_images(config,workers=args.workers)
    elif args.phase=='freeze':result=freeze(config)
    elif args.phase=='status':result=status(config)
    elif args.phase in {'review-images','review-models'}:
        from .batch_review import review_appearance,review_models
        result=review_appearance(config) if args.phase=='review-images' else review_models(config,workers=args.workers)
    elif args.phase=='gpu':
        from .batch_gpu import run_gpu_batch
        result=run_gpu_batch(config,gpu_count=args.gpu_count,deadline_seconds=args.deadline_seconds)
    else:
        from .batch_delivery import validate_batch,finalize_batch
        result=validate_batch(config,workers=args.workers) if args.phase=='validate' else finalize_batch(config)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
    write_json(Path(config['output'])/'phase_results'/(args.phase+'.json'),result)
    return 2 if result.get('status') in {'failed','incomplete'} else 0


if __name__=='__main__':raise SystemExit(main())
