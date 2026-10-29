"""CPU placement, reviewed repairs, partial surfaces and complete-source binding."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from .appearance import write_json, sha256_file
from .batch import read, proof, verify


def source_row(building_path, root):
    building=read(building_path)
    return {'building_id':building['building_id'],'status':'pending_generation',
        'structure_path':os.path.relpath(building_path,root),'target_geometry':building['footprint_world'],
        'base_z_m':building['base_z_m'],'top_z_m':building['top_z_m'],
        'reason':'Complete observed building selected for the unified generation workflow; no historical mesh input.'}


def prepare_nonbuilding_sources(config):
    from .postprocess import build_fragment_record
    root=Path(config['output']);project=Path(config['_project_root']);policy=read(config['dispositions'])
    inputs=Path(config['input_manifest']);records={r['building_id']:r for r in read(inputs)['records']}
    decisions={}
    if any(rule['action']=='observed_surface' for rule in policy['sources']):
        if not policy.get('fragment_support_package') or not policy.get('fragment_recipe_package'):
            raise ValueError('Observed surfaces require a newly reviewed support package and recipe for these sources')
        package=project/policy['fragment_support_package']
        decisions={d['building_id']:copy.deepcopy(d) for d in read(project/policy['fragment_recipe_package'])['records']}
    results=[]
    for rule in policy['sources']:
        if rule['action']=='generate':continue
        sid=rule['building_id'];building_path=inputs.parent/records[sid]['path']/'building.json'
        row=source_row(building_path,root)
        if rule['action']=='record_only':
            excluded=rule.get('exclusion_status')
            if excluded not in {'excluded_existing_building','excluded_incomplete','excluded_noise'}:
                raise ValueError('Record-only dispositions need an explicit exclusion_status')
            row.update(status=excluded,
                reason=rule['reason'],evidence={'qualification':'frozen reviewed source disposition',
                    'source_building':proof(building_path,root),'reference_assisted_qualification':policy.get('reference_assisted_qualification',False)})
        else:
            decision=decisions[sid];decision['assembly_enabled']=sid in policy['default_surface_source_ids']
            # Copy only source-supported observations, never an old building GLB.
            source=package/'postprocess_support/fragments'/sid;target=root/'postprocess_support/fragments'/sid
            shutil.copytree(source,target,dirs_exist_ok=True)
            folder=root/'observed_surfaces'/sid
            record_path=folder/'source_record.json'
            if record_path.exists():row=read(record_path)
            else:
                folder.mkdir(parents=True,exist_ok=True)
                row=build_fragment_record(row,building_path,decision,root,folder,root)
                write_json(record_path,row)
        results.append(row)
    write_json(root/'nonbuilding_sources.json',{'source_records':results,'historical_model_inputs':0})
    return results


def _base_generated_row(config,plan,building,fit,paths):
    from .lod import inspect_glb
    from .cli import SUPPORTED_LOD_TARGETS
    from .registration import read_glb_scene_geometry
    root=Path(config['output']);source=root/plan['source_buildings'][0]['path']
    row=source_row(source,root);matrix=np.asarray(fit['world_from_asset_4x4'])
    row.update(status='accepted_generated',asset_instance_id=plan['asset_id'],
        target_geometry=building['footprint_world'],base_z_m=building['base_z_m'],top_z_m=building['top_z_m'],
        local_coordinate_system='glTF Y-up',world_coordinate_system='UE XYZ metres Z-up',
        world_from_local_4x4=matrix.tolist(),building_local_from_asset_4x4=fit['building_local_from_asset_4x4'],
        world_from_building_local_4x4=fit['world_from_building_local_4x4'],
        normal_transform_world_3x3=np.linalg.inv(matrix[:3,:3]).T.tolist(),
        geometry_fit_metrics=fit['fit']['actual_mesh_overlap'],lods={})
    for label,path in paths.items():
        scene=read_glb_scene_geometry(path)
        row['lods'][label]={**inspect_glb(path),'path':str(path.relative_to(root)),
            'faces':len(scene['faces']),'target_faces':SUPPORTED_LOD_TARGETS[label],'texture_size':2048}
    raw=root/'generation'/plan['asset_id']/'raw_manifest.json';meta=read(raw)
    row['provenance']={'kind':'unified_current_source_generation',
        'source_buildings':plan['source_buildings'],'effective_building_sha256':plan['building']['sha256'],
        'conditioning_image_sha256':plan['image']['sha256'],'generation_contract':meta['contract'],
        'raw_manifest_sha256':sha256_file(raw),'raw_cache_sha256':meta['sha256'],
        'historical_asset_or_GT_geometry_read':False,'structure_conditioning':plan['structure_conditioning']}
    return row


def _postprocess(config,plan,building,fit,paths,output):
    """Apply only recorded, source-bound operations to new candidate files."""
    from .extrema_lod import export_extrema_preserving_lod
    root=Path(config['output']);project=Path(config['_project_root']);rules=plan.get('postprocess',{})
    modified=dict(paths);proofs={}
    # New byte identities can require new reviewed operations. These are CPU
    # decisions made after inference, not changes to the frozen GPU inputs.
    fresh_path=root/'reviews/repairs.json'
    if fresh_path.exists():
        decisions=read(fresh_path)['records'];current=[r for r in decisions if r['asset_id']==plan['asset_id']]
        if len(current)>1:raise ValueError('Duplicate current repair decision')
        if current:
            current=current[0]
            if (current['source_building_sha256']!=plan['building']['sha256']
                    or current['input_lod_sha256']!={label:sha256_file(path) for label,path in paths.items()}
                    or not current.get('reason')):
                raise ValueError('New repair decision belongs to different input bytes')
            rules={**rules,**{k:v for k,v in current.items() if k in
                {'observed_roof_projection','extrema_lod_if_needed'}}}
            if current.get('source_orientation_approval'):
                approval_path=verify(root,current['source_orientation_approval'])
                approval=read(approval_path)
                if approval.get('schema')!='citycode.portable_source_orientation.v1':
                    raise ValueError('Current source orientation must be a portable directed-role approval')
                if not np.allclose(approval['original_prior_world_from_asset_4x4'],fit['prior_world_from_asset_4x4'],atol=1e-8,rtol=0):
                    raise ValueError('Orientation approval belongs to another initial pose')
                if ((root/approval['cue']['actual_model']['path']).resolve()!=paths['0.25M'].resolve()
                        or (root/approval['cue']['current_building']['path']).resolve()!=(root/plan['building']['path']).resolve()):
                    raise ValueError('Current orientation proof must refer to current input files')
                def rebase(value):
                    if isinstance(value,dict):
                        if set(value)=={'path','sha256','bytes'}:
                            return {**value,'path':str(verify(root,value,project))}
                        return {k:rebase(v) for k,v in value.items()}
                    return [rebase(v) for v in value] if isinstance(value,list) else value
                cue=rebase(approval['cue']);cue_path=output/'source_orientation.materialized.json';write_json(cue_path,cue)
                if cue['actual_model']['sha256']!=sha256_file(paths['0.25M']):
                    raise ValueError('Source-role evidence refers to a different actual mesh')
                from .orientation import source_supported_yaw_initialization
                from .registration import fit_predicted_asset
                initialization=source_supported_yaw_initialization(approval['original_prior_world_from_asset_4x4'],
                    building,cue_path,approved_cue_sha256=sha256_file(cue_path),evidence_root=project)
                fit=fit_predicted_asset(paths['0.25M'],building,initialization['source_supported_prior_world_from_asset_4x4'],
                    expected_source_sha256=sha256_file(paths['0.25M']))
                proofs['fresh_orientation']={'approval':proof(approval_path,root),'source_role_initializer_reexecuted':True}
                rules={**rules,'reviewed_source_orientation':None}
            if current.get('cross_lod_recipe'):
                from .mesh_repair import repair_lods
                recipe=verify(root,current['cross_lod_recipe']);folder=output/'fresh_cross_lod'
                if not folder.exists():repair_lods(paths['0.25M'],paths['0.1M'],fit['world_from_asset_4x4'],folder,recipe,
                    expected_recipe_sha256=sha256_file(recipe))
                modified.update({'0.25M':folder/'model_025m.glb','0.1M':folder/'model_01m.glb'})
                proofs['fresh_cross_lod_recipe']=proof(recipe,root)
                rules={**rules,'reviewed_cross_lod_recipe':None}
            proofs['current_repair_decisions']=proof(fresh_path,root)
    orientation=rules.get('reviewed_source_orientation')
    # Old role anchors are valid only for the exact same generated high mesh.
    # Different meshes must obtain a fresh review rather than inherit old cues.
    if orientation:
        approval=read(project/orientation)
        if sha256_file(paths['0.25M'])==approval['cue']['actual_model']['sha256']:
            from .registration import fit_predicted_asset
            fit=fit_predicted_asset(paths['0.25M'],building,
                approval['expected_source_supported_prior_world_from_asset_4x4'],
                expected_source_sha256=sha256_file(paths['0.25M']))
            proofs['orientation']={'method':'exact_mesh_bound_frozen_source_role_initialization',
                'approval_sha256':sha256_file(project/orientation),'new_model_roles_reused_only_if_mesh_identical':True}
        else:proofs['orientation']={'status':'fresh_correspondence_review_required_if_numeric_fit_fails'}
    recipe=rules.get('reviewed_cross_lod_recipe')
    if recipe:
        from .mesh_repair import repair_lods
        recipe=project/recipe;spec=read(recipe)
        if all(sha256_file(paths[label])==spec['inputs'][label]['sha256'] for label in ('0.25M','0.1M')):
            folder=output/'cross_lod'
            if not folder.exists():repair_lods(paths['0.25M'],paths['0.1M'],fit['world_from_asset_4x4'],folder,recipe,
                expected_recipe_sha256=sha256_file(recipe))
            modified.update({'0.25M':folder/'model_025m.glb','0.1M':folder/'model_01m.glb'})
            proofs['cross_lod_recipe_sha256']=sha256_file(recipe)
        else:proofs['cross_lod']={'status':'new_triangle_selection_required_if_holes_remain'}
    if rules.get('observed_roof_projection') and fit['status']=='accepted':
        from .roof_reconciliation import reconcile_roof
        source_path=root/plan['building']['path']
        observation=next(o for o in building['observations'] if o['observation_id']==building['canonical_observation_id'])
        folder=source_path.parent/observation['path'];metadata=read(folder/'observation.json');roof=folder/metadata['paths']['roof_evidence']
        for label,path in list(modified.items()):
            destination=output/'roof'/path.name;destination.parent.mkdir(parents=True,exist_ok=True)
            saved=destination.with_suffix('.proof.json')
            if not saved.exists():
                result=reconcile_roof(path,destination,building=building,world_from_asset=fit['world_from_asset_4x4'],
                    observation=metadata,evidence_image=roof)
                write_json(saved,result)
            result=read(saved)
            if result['source']['sha256']!=sha256_file(path) or result['result']['sha256']!=sha256_file(destination):
                raise ValueError('Roof postprocessing cache changed')
            modified[label]=destination
            proofs.setdefault('observed_roof',{})[label]=proof(saved,root)
    return fit,modified,proofs


def validate_one(task):
    from .placement import load_placement_inputs, fit_generated_asset
    from .cli import _policy, _validate_candidate_lods
    from .extrema_lod import export_extrema_preserving_lod
    config,plan=task;root=Path(config['output']);aid=plan['asset_id'];output=root/'validation'/aid;output.mkdir(parents=True,exist_ok=True)
    result={'asset_id':aid,'source_ids':plan['source_ids'],'status':'pending_generation','errors':[]}
    try:
        building_path=verify(root,plan['building'],Path(config['_project_root']))
        building,audit=load_placement_inputs(building_path,root/'generation'/aid)
        exported=read(root/'assets'/aid/'export_manifest.json')
        from .lod import generation_lod_targets, LOD_SPECS
        if set(exported['lods'])!={LOD_SPECS[target][0] for target in generation_lod_targets(config)}:
            raise ValueError('Exported LODs do not match the configured generation targets')
        paths={label:verify(root,record) for label,record in exported['lods'].items()}
        fit=fit_generated_asset(paths['0.25M'],building,audit,expected_source_sha256=sha256_file(paths['0.25M']))
        fit,paths,operations=_postprocess(config,plan,building,fit,paths,output)
        write_json(output/'fit.json',fit);result.update(fit_status=fit['status'],fit_reasons=fit['reasons'],postprocess=operations)
        row=_base_generated_row(config,plan,building,fit,paths)
        # Visual completion never inherits an approval from the previous scene.
        # Keep the actual candidate and its pose for a fresh scene review.
        visual=fit['status']!='accepted' and plan['validation_profile']=='visual_scene_completion'
        if fit['status']!='accepted' and not visual:
            result['status']='pending_registration'
            write_json(output/'candidate_record.json',row)
        else:
            try:checks=_validate_candidate_lods(root,row,_policy({}),geometry=True,visual_completion=visual)
            except ValueError as exc:
                if ('base/top' not in str(exc) and 'local frame' not in str(exc)):
                    raise
                fresh_extrema=read(root/'reviews/repairs.json')['records'] if (root/'reviews/repairs.json').exists() else []
                if not (plan['postprocess'].get('extrema_lod_if_needed') or any(r['asset_id']==aid and
                        r.get('extrema_lod_if_needed') for r in fresh_extrema)):
                    raise
                repaired=output/'extrema/model_01m.glb';repaired.parent.mkdir(parents=True,exist_ok=True)
                saved=repaired.with_suffix('.proof.json')
                if not saved.exists():
                    repair=export_extrema_preserving_lod(paths['0.25M'],repaired,fit_record=fit,
                        validation_profile=plan['validation_profile']);write_json(saved,repair)
                repair=read(saved)
                if repair['status']!='accepted_candidate' or repair['low']['sha256']!=sha256_file(repaired):
                    raise ValueError('Extrema repair candidate failed')
                paths['0.1M']=repaired;row=_base_generated_row(config,plan,building,fit,paths)
                checks=_validate_candidate_lods(root,row,_policy({}),geometry=True,visual_completion=visual)
                result['postprocess']['extrema_lod']=proof(saved,root)
            row['validation']={'status':'actual_dual_lod_numeric_pass','lods':checks,
                'profile':'visual_scene_completion_candidate' if visual else 'source_reconstruction',
                'source_and_model_identity_preserved':True,'semantic_and_scene_review_separate':True}
            write_json(output/'candidate_record.json',row)
            result.update(status='numeric_pass_scene_review_pending' if visual else 'numeric_pass_model_review_pending',checks=checks,
                placement_identity=__import__('citycode_assets.scene_completion',fromlist=['placement_identity']).placement_identity(row))
        result['candidate_record']=proof(output/'candidate_record.json',root)
    except Exception as exc:
        result.update(status='pending_repair',errors=[str(exc)],traceback=traceback.format_exc())
    write_json(output/'RESULT.json',result);return result


def validate_batch(config,*,workers=4):
    from .batch_gpu import frozen_selection
    root=Path(config['output']);prepare_nonbuilding_sources(config)
    selected=frozen_selection(config);results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(validate_one,(config,row)) for row in selected]):
            value=future.result();results.append(value)
            write_json(root/'VALIDATION_PROGRESS.json',{'selected':len(selected),'complete':len(results),
                'counts':dict(Counter(r['status'] for r in results)),'records':results})
            print(json.dumps({k:value[k] for k in ['asset_id','status','errors']}),flush=True)
    return {'status':'CPU_checks_complete_review_pending','selected':len(selected),
        'counts':dict(Counter(r['status'] for r in results)),'records':results}


def finalize_batch(config):
    """Bind explicit model reviews to actual new bytes; preserve all source IDs."""
    from .cli import validate_assets
    from .scene_completion import placement_identity, scene_identity
    from .batch_gpu import frozen_selection
    from .lod import generation_lod_targets, LOD_SPECS
    frozen_selection(config)
    root=Path(config['output']);plan=read(root/'FROZEN.json');policy=read(config['dispositions'])
    inputs=Path(config['input_manifest']);upstream=read(inputs)
    rows={r['building_id']:source_row(inputs.parent/r['path']/'building.json',root) for r in upstream['records']}
    partials=prepare_nonbuilding_sources(config)
    rows.update({r['building_id']:r for r in partials})
    review_path=root/'reviews/models.json'
    reviews={r['asset_id']:r for r in read(review_path)['records']} if review_path.exists() else {}
    physical=[];shared=[]
    for record in plan['records']:
        aid=record['asset_id'];folder=root/'validation'/aid
        if not (folder/'candidate_record.json').exists():continue
        candidate=read(folder/'candidate_record.json');numeric=read(folder/'RESULT.json');review=reviews.get(aid)
        if set(candidate['lods'])!={LOD_SPECS[target][0] for target in generation_lod_targets(config)}:
            raise ValueError('Candidate LODs do not match the configured generation targets')
        if not numeric.get('candidate_record'):continue
        verify(root,numeric['candidate_record'])
        if (not review or review['decision']!='include' or not review.get('reason')
                or review.get('placement_identity')!=placement_identity(candidate)):
            for sid in record['source_ids']:rows[sid]['reason']='New model awaits explicit review of its current bytes and scene placement.'
            continue
        if numeric['status']!='numeric_pass_model_review_pending':
            if (numeric['status']!='numeric_pass_scene_review_pending'
                    or record['validation_profile']!='visual_scene_completion' or not review.get('scene_completion_review')):
                continue
            candidate['status']='accepted_scene_completion';candidate['scene_completion_review']=review['scene_completion_review']
        candidate['reason']=review['reason'];candidate['semantic_review']={**review,'review_file':proof(review_path,root)}
        for original in record['source_buildings']:
            sid=original['building_id'];source=read(root/original['path']);row=copy.deepcopy(candidate)
            row.update(building_id=sid,structure_path=original['path'])
            if len(record['source_ids'])>1:
                row.update(asset_instance_group_id=aid,source_target_geometry=source['footprint_world'])
                source_world=np.asarray(source['world_from_local_4x4'])
                row['world_from_building_local_4x4']=source_world.tolist()
                row['building_local_from_asset_4x4']=(np.linalg.inv(source_world) @
                    np.asarray(candidate['world_from_local_4x4'])).tolist()
            rows[sid]=row
        physical.append({'id':aid,'source_building_ids':record['source_ids'],'status':candidate['status'],
            'world_from_local_4x4':candidate['world_from_local_4x4'],'lods':candidate['lods']})
        if len(record['source_ids'])>1:
            shared.append({'id':aid,'building_ids':record['source_ids'],'one_physical_instance':True,
                'world_from_local_4x4':candidate['world_from_local_4x4'],'lods':candidate['lods']})
    active={r['id']:r for r in physical};entities=[];grouped=set()
    for entity in policy.get('building_entity_groups',[]):
        if set(entity['asset_instance_ids'])<=set(active):entities.append(entity);grouped.update(entity['asset_instance_ids'])
    entities += [{'id':aid,'asset_instance_ids':[aid],'source_building_ids':a['source_building_ids']}
                 for aid,a in active.items() if aid not in grouped]
    for entity in entities:
        for aid in entity['asset_instance_ids']:active[aid]['building_entity_id']=entity['id']
    surfaces=[{'id':r['surface_instance_id'],'source_building_ids':[r['building_id']],
        'status':r['status'],'world_from_local_4x4':r['world_from_local_4x4'],'lods':r['lods'],
        'parent_building_id':r['postprocess'].get('parent_building_id'),
        'assembly_enabled':r['postprocess']['surface']['assembly_enabled']} for r in partials if r['status']=='resolved_fragment']
    pending=any(r['status'].startswith('pending') for r in rows.values())
    manifest={'schema':'citycode.building_asset_bindings.v1','path_base':'manifest_directory',
        'status':'in_progress' if pending else 'complete','fine_generation_complete':not pending,
        'source_structure':{'instance_count':len(rows),'manifest_path':os.path.relpath(inputs,root),
            'manifest_sha256':sha256_file(inputs),'building_ids':list(rows)},
        'validation_policy':{},'buildings':list(rows.values()),'asset_instances':physical,
        'asset_instance_groups':shared,'building_entities':entities,'surface_instances':surfaces,
        'counts':dict(Counter(r['status'] for r in rows.values())),
        'generation_workflow':{'historical_model_inputs':0,'all_source_instances_preserved':True,
            'frozen_plan_sha256':sha256_file(root/'plan.json'),'qualification_reference_assisted':policy.get('reference_assisted_qualification',False),
            'lod_targets':list(generation_lod_targets(config)),'execution_mode':'generate'}}
    context=scene_identity(manifest)
    for row in manifest['buildings']:
        if row['status']=='accepted_scene_completion':
            if row['scene_completion_review'].get('scene_identity')!=context:
                raise ValueError('Visual approval does not match the new complete scene')
    candidate=root/'binding_manifest.candidate.json';write_json(candidate,manifest)
    report=validate_assets(candidate,require_complete=True,geometry=False)
    write_json(root/'delivery_validation.json',report)
    if report['status']=='passed' and report['complete']:
        target=root/'binding_manifest.json'
        if target.exists() and sha256_file(target)!=sha256_file(candidate):
            raise ValueError('Existing delivery differs; retain it and use a new run directory')
        if not target.exists():shutil.copyfile(candidate,target)
    return {'status':report['status'],'complete':report['complete'],'source_instances':len(rows),
        'physical_assets':len(physical),'logical_entities':len(entities),'pending_sources':sum(r['status'].startswith('pending') for r in rows.values()),
        'errors':report['errors'],'historical_model_inputs':0}
