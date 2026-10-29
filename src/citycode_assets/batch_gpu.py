"""Task-scoped GPU execution for a frozen unified batch; no API or allocation code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .batch import load_config, read, verify, check_plan
from .appearance import write_json, sha256_file

WORKER_MODULE = 'citycode_assets.batch_gpu'


def runtime_config(config):
    project=Path(config['_project_root']); runtime=read(config['runtime_config'])
    for key in ['trellis_root','pipeline_config','model_package_root','model_manifest']:
        runtime[key]=str((project/runtime[key]).resolve())
    return runtime


def frozen_selection(config, *, asset_id=None, check_resources=True):
    root=Path(config['output']); frozen=read(root/'FROZEN.json')
    check_plan(config)
    from . import batch
    if frozen['status']!='reviewed_ready' or frozen['execution_code_sha256']!=sha256_file(Path(batch.__file__)):
        raise ValueError('Frozen execution code changed')
    if frozen['plan_sha256']!=sha256_file(root/'plan.json'):
        raise ValueError('Frozen plan changed')
    from . import generation
    if (frozen['gpu_execution_code_sha256']!=sha256_file(Path(__file__))
            or frozen['generation_code_sha256']!=sha256_file(Path(generation.__file__))):
        raise ValueError('Frozen GPU or generation implementation changed')
    verify(root,frozen['review'])
    from .batch_resources import verify_runtime_contract
    verify_runtime_contract(config,frozen,check_sizes=check_resources)
    selected=[r for r in frozen['records'] if r['image_decision']=='generate']
    if asset_id is not None:
        selected=[r for r in selected if r['asset_id']==asset_id]
        if len(selected)!=1:raise ValueError('Asset ID is not in the frozen generation selection')
    if asset_id is None:
        for item in frozen.get('auxiliary_inputs',[]):verify(root,item,Path(config['_project_root']))
    for row in selected:
        verify(root,row['building'],Path(config['_project_root']))
        for item in row['source_buildings']:verify(root,item,Path(config['_project_root']))
        for key in ['image','prepared_input','completion_receipt']:verify(root,row[key])
    return selected


def infer_rank(config,rank,count):
    from .generation import GenerationRuntime, image_height_span
    root=Path(config['output']); selected=frozen_selection(config); runtime=GenerationRuntime(runtime_config(config))
    results=[]
    for row in selected[rank::count]:
        start=time.monotonic(); aid=row['asset_id']
        try:
            runtime.config.pop('structure_height_voxels',None)
            if row['structure_conditioning']=='source_footprint':runtime.config['structure_conditioning']='source_footprint'
            elif row['structure_conditioning']=='image_height_span':
                runtime.config['structure_conditioning']='image_height_span'
                runtime.config['structure_height_voxels']=image_height_span(row)
            else:runtime.config.pop('structure_conditioning',None)
            result=runtime.generate(root/row['building']['path'],root/row['appearance_directory'],root/'generation'/aid)
            value={'asset_id':aid,'status':'complete','raw_manifest_sha256':sha256_file(root/'generation'/aid/'raw_manifest.json'),
                   'seconds':time.monotonic()-start,'inference_seconds':result.get('inference_seconds')}
        except Exception:
            value={'asset_id':aid,'status':'failed','traceback':traceback.format_exc(),'seconds':time.monotonic()-start}
        results.append(value);write_json(root/'execution'/f'infer_rank{rank}.json',{'records':results});print(json.dumps(value),flush=True)
    return 0 if all(r['status']=='complete' for r in results) else 1


def export_one(config,aid):
    frozen_selection(config,asset_id=aid,check_resources=False)
    from .generation import configure_offline_runtime
    from .lod import export_decoded_lods, generation_lod_targets, LOD_SPECS
    configure_offline_runtime(Path(runtime_config(config)['trellis_root']))
    root=Path(config['output']);raw=root/'generation'/aid/'raw_mesh.pt';metadata=read(raw.parent/'raw_manifest.json')
    if metadata['status']!='complete' or metadata['sha256']!=sha256_file(raw):
        raise ValueError('Raw generation bytes changed')
    output=root/'assets'/aid
    saved=output/'export_manifest.json'
    targets=generation_lod_targets(config)
    labels={LOD_SPECS[target][0] for target in targets}
    if saved.exists():
        previous=read(saved)
        if previous['raw_cache_sha256']!=metadata['sha256']:raise ValueError('Export cache belongs to different generation')
        if set(previous['lods'])!=labels:raise ValueError('Export cache has different LOD targets; choose a new output')
        for row in previous['lods'].values():
            verify(root,row)
        return 0
    result=export_decoded_lods(raw,output,targets=targets,texture_size=2048)
    for row in result['lods'].values():row['path']=str(Path(row['path']).relative_to(root))
    result.update(asset_id=aid,raw_manifest_sha256=sha256_file(raw.parent/'raw_manifest.json'))
    write_json(saved,result)
    print(json.dumps({'asset_id':aid,'status':result['status']}),flush=True)
    return 0


def export_rank(config,rank,count,config_path):
    root=Path(config['output']);results=[]
    for row in frozen_selection(config)[rank::count]:
        aid=row['asset_id'];metadata=root/'generation'/aid/'raw_manifest.json'
        if not metadata.exists():
            results.append({'asset_id':aid,'status':'no_completed_raw','exit_code':1});continue
        log=root/'execution/logs'/f'export_{aid}.log'
        with log.open('a') as stream:
            code=subprocess.run([sys.executable,'-u','-m',WORKER_MODULE,'--config',str(config_path),
                '--mode','export-one','--asset-id',aid],stdout=stream,stderr=subprocess.STDOUT).returncode
        value={'asset_id':aid,'exit_code':code,'status':'complete' if code==0 else 'failed'}
        results.append(value);write_json(root/'execution'/f'export_rank{rank}.json',{'records':results});print(json.dumps(value),flush=True)
    return 0 if all(r['exit_code']==0 for r in results) else 1


def run_gpu_batch(config,*,gpu_count=4,deadline_seconds=None):
    import torch
    if gpu_count not in [1,2,4,8] or (deadline_seconds is not None and deadline_seconds <= 0):
        raise ValueError('Use 1/2/4/8 GPUs; an explicitly requested deadline must be positive')
    if not torch.cuda.is_available() or torch.cuda.device_count()<gpu_count:
        raise RuntimeError('Run this phase only on an allocated CUDA worker')
    selected=frozen_selection(config);root=Path(config['output']);run=root/'execution';(run/'logs').mkdir(parents=True,exist_ok=True)
    from .batch_resources import verify_runtime_contract
    verify_runtime_contract(config,read(root/'FROZEN.json'),full_hash=True)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)<gpu_count or not devices[0]:devices=[str(i) for i in range(gpu_count)]
    started=time.monotonic();deadline=None if deadline_seconds is None else started+deadline_seconds;active=[];stopped=False;phases=[]
    def stop(signum=None,frame=None):
        nonlocal stopped
        stopped=True
        for child,stream in active:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
    old_term,old_int=signal.signal(signal.SIGTERM,stop),signal.signal(signal.SIGINT,stop)
    try:
        for mode in ['infer','export']:
            active=[]
            for rank in range(gpu_count):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[rank],FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(run/f'flex_rank{rank}.json'))
                stream=(run/f'{mode}_rank{rank}.log').open('a')
                child=subprocess.Popen([sys.executable,'-u','-m',WORKER_MODULE,'--config',config['_config_path'],
                    '--mode',mode,'--rank',str(rank),'--gpu-count',str(gpu_count)],env=env,stdout=stream,
                    stderr=subprocess.STDOUT,start_new_session=True);active.append((child,stream))
            while any(c.poll() is None for c,s in active) and not stopped:
                if deadline is not None and time.monotonic()>deadline:stop();break
                time.sleep(1)
            codes=[]
            for child,stream in active:
                try:codes.append(child.wait(timeout=30))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);codes.append(child.wait(timeout=10))
                stream.close()
            phases.append({'phase':mode,'exit_codes':codes})
            write_json(run/'PROGRESS.json',{'phases':phases,'seconds':time.monotonic()-started,'stopped':stopped})
            if stopped:break
    finally:
        stop();signal.signal(signal.SIGTERM,old_term);signal.signal(signal.SIGINT,old_int)
    complete=sum((root/'assets'/r['asset_id']/'export_manifest.json').is_file() for r in selected)
    result={'status':'GPU_outputs_complete_CPU_validation_pending' if complete==len(selected) else 'incomplete',
        'selected':len(selected),'exported_assets':complete,'phases':phases,'elapsed_seconds':time.monotonic()-started,
        'allocation_should_terminate_now':True,'GPU_allocation_requested_by_workflow':False}
    write_json(run/'GPU_RESULT.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=['infer','export','export-one'],required=True)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--gpu-count',type=int,default=4);parser.add_argument('--asset-id')
    args=parser.parse_args(argv);config=load_config(args.config)
    if args.mode=='infer':return infer_rank(config,args.rank,args.gpu_count)
    if args.mode=='export':return export_rank(config,args.rank,args.gpu_count,args.config)
    return export_one(config,args.asset_id)


if __name__=='__main__':raise SystemExit(main())
