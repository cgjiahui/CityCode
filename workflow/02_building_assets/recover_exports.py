"""Recover only a failed 0.1M export from this batch's reviewed 0.25M bytes.

Run after releasing the GPU. A missing review emits a template and changes no
mesh. Recovery creates a candidate; the normal validate/review/finalize phases
still decide whether it can be placed. Historical model inputs are never read.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import numpy as np
from citycode_assets.appearance import sha256_file, write_json
from citycode_assets.batch import load_config, read, proof, verify
from citycode_assets.batch_gpu import frozen_selection
from citycode_assets.lod import generation_lod_targets, LOD_SPECS, inspect_glb


def recovery_inputs(config, row):
    root=Path(config['output']);aid=row['asset_id'];folder=root/'assets'/aid
    paths={LOD_SPECS[t][0]:folder/LOD_SPECS[t][1] for t in generation_lod_targets(config)}
    if '0.1M' not in paths or '0.25M' not in paths:
        raise ValueError('Recovery requires configured 0.25M and 0.1M targets')
    if [label for label,path in paths.items() if not path.is_file()]!=['0.1M']:
        raise ValueError('Recovery requires exactly one missing export: 0.1M')
    raw=root/'generation'/aid/'raw_manifest.json';metadata=read(raw)
    if metadata.get('status')!='complete' or metadata.get('contract',{}).get('building_sha256')!=row['building']['sha256']:
        raise ValueError('Current complete generation/source identity required')
    for label,path in paths.items():
        if label!='0.1M':inspect_glb(path)
    return {'asset_id':aid,'source_building_sha256':row['building']['sha256'],
        'raw_manifest_sha256':sha256_file(raw),'raw_cache_sha256':metadata['sha256'],
        'input_lods':{label:proof(path,root) for label,path in paths.items() if label!='0.1M'}},paths


def verify_review(expected, review):
    if (any(review.get(key)!=value for key,value in expected.items())
            or review.get('decision')!='recover_0.1M_from_0.25M' or not review.get('reason')):
        raise ValueError('Export recovery needs a review bound to the current source, raw mesh and higher LOD bytes')
    if review.get('target_relative_error',.01) not in (.01,.02):
        raise ValueError('Reviewed CPU error budget must be1 or2 percent')


def recover_exports(config):
    from citycode_assets.placement import load_placement_inputs, fit_generated_asset
    from citycode_assets.extrema_lod import export_extrema_preserving_lod
    root=Path(config['output']);selected=frozen_selection(config)
    review_path=root/'reviews/export_recovery.json'
    reviews=read(review_path)['records'] if review_path.exists() else []
    decisions={row['asset_id']:row for row in reviews}
    if len(decisions)!=len(reviews):raise ValueError('Duplicate export recovery reviews')
    proposals=[];results=[]
    for row in selected:
        aid=row['asset_id'];folder=root/'assets'/aid;saved=folder/'export_manifest.json'
        if saved.exists():
            previous=read(saved)
            raw_path=root/'generation'/aid/'raw_manifest.json';raw=read(raw_path)
            labels={LOD_SPECS[target][0] for target in generation_lod_targets(config)}
            if (previous.get('raw_cache_sha256')!=raw.get('sha256')
                    or previous.get('raw_manifest_sha256')!=sha256_file(raw_path)
                    or set(previous['lods'])!=labels):
                raise ValueError('Existing export manifest belongs to another raw generation or LOD set')
            for item in previous['lods'].values():verify(root,item)
            results.append({'asset_id':aid,'status':'already_exported'});continue
        identity,paths=recovery_inputs(config,row)
        proposals.append({**identity,'target_relative_error':.01,'decision':None,'reason':''})
        if aid not in decisions:
            results.append({'asset_id':aid,'status':'awaiting_current_export_recovery_review'});continue
        verify_review(identity,decisions[aid])
        building_path=verify(root,row['building'],Path(config['_project_root']))
        building,audit=load_placement_inputs(building_path,root/'generation'/aid)
        fit=fit_generated_asset(paths['0.25M'],building,audit,expected_source_sha256=identity['input_lods']['0.25M']['sha256'])
        attempt=folder/'export_recovery'/sha256_file(review_path)[:16]
        attempt.mkdir(parents=True,exist_ok=True);candidate=attempt/'model_01m.glb'
        repair_path=attempt/'proof.json'
        if candidate.exists():raise ValueError('This reviewed recovery attempt already exists; inspect its proof before a new decision')
        repair=export_extrema_preserving_lod(paths['0.25M'],candidate,fit_record=fit,
            target_relative_error=decisions[aid].get('target_relative_error',.01),
            validation_profile=row['validation_profile'])
        write_json(repair_path,{**repair,'review':proof(review_path,root),'input_identity':identity,
            'recovery_entry_sha256':sha256_file(Path(__file__))})
        if repair['status']!='accepted_candidate':
            raise ValueError('Recovered low LOD failed independent simplification checks; inspect its proof')
        paths['0.1M']=candidate
        lods={}
        for target in generation_lod_targets(config):
            label,name=LOD_SPECS[target];path=paths[label];inspection=inspect_glb(path)
            if inspection['faces']>target*1.02:raise ValueError('Recovered export exceeds face budget')
            lods[label]={**inspection,'path':str(path.relative_to(root)),'target_faces':target,'texture_size':2048}
        master=max(lods.values(),key=lambda value:value['target_faces'])
        span=np.maximum(np.ptp(np.asarray(master['bounds_local_accessor']),axis=0),1e-6)
        if any(np.max(np.abs(np.asarray(lod['bounds_local_accessor'])-master['bounds_local_accessor'])/span)>.025 for lod in lods.values()):
            raise ValueError('Recovered LOD frame differs by more than2.5 percent')
        write_json(saved,{'status':'completed','asset_id':aid,'raw_cache_sha256':identity['raw_cache_sha256'],
            'raw_manifest_sha256':identity['raw_manifest_sha256'],'lods':lods,
            'shared_local_coordinate_frame':True,'positions_not_world_baked':True,
            'cpu_low_lod_recovery':proof(repair_path,root),'independent_model_review_required':True})
        results.append({'asset_id':aid,'status':'recovered_candidate_validation_pending'})
    if proposals:write_json(root/'reviews/export_recovery.template.json',{'schema':'citycode.export_recovery_review.v1','records':proposals})
    pending=any(row['status'].startswith('awaiting') for row in results)
    result={'status':'awaiting_export_recovery_review' if pending else 'exports_ready_validation_pending','records':results,'GPU_required':False}
    write_json(root/'EXPORT_RECOVERY.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv);result=recover_exports(load_config(args.config))
    import json
    print(json.dumps(result,indent=2))
    return 2 if result['status']=='awaiting_export_recovery_review' else 0


if __name__=='__main__':raise SystemExit(main())
