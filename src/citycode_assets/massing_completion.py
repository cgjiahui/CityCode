"""Reviewed reuse of existing textured assets for missing observed building massing.

These are inferred wings/podia, separate from independently accepted buildings.
Original source IDs and baseline mesh bytes/placements are never replaced.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from shapely.geometry import shape,mapping
from .appearance import write_json,sha256_file
from .batch import read,proof,verify,digest
from .scene_completion import placement_identity,scene_identity


def construct_manifest(baseline,recipe):
    result=copy.deepcopy(baseline);donors={a['id']:a for a in baseline['asset_instances']}
    sources={r['building_id']:r for r in result['buildings']};parts=[]
    known=set(sources);seen=set()
    for decision in recipe['parts']:
        aid=decision['id'];ids=decision['source_building_ids']
        if aid in seen or aid in donors or not ids or set(ids)-known:
            raise ValueError('Massing IDs and source identities must be explicit and unique')
        seen.add(aid);donor=donors[decision['donor_asset_id']]
        if decision['donor_lods']!=donor['lods']:
            raise ValueError('Massing donor mesh identity differs from the frozen baseline')
        matrix=np.asarray(decision['world_from_local_4x4'],float)
        if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3],[0,0,0,1]) or np.linalg.det(matrix[:3,:3])<=0):
            raise ValueError('Massing placement must preserve orientation and be invertible')
        part={**copy.deepcopy(decision),'status':'accepted_massing_completion',
            'lods':copy.deepcopy(donor['lods']),'normal_transform_world_3x3':np.linalg.inv(matrix[:3,:3]).T.tolist(),
            'independent_building':False,'appearance_and_body_inferred':True,'geometry_bytes_reused':True,
            'original_donor_placement':donor['world_from_local_4x4']}
        parts.append(part)
    active_sources={sid for p in parts for sid in p['source_building_ids']}
    for sid in active_sources:
        sources[sid]['massing_completion']={'method':'reviewed_existing_asset_reuse',
            'part_ids':[p['id'] for p in parts if sid in p['source_building_ids']],
            'independent_building_claimed':False,'original_source_state_preserved':True}
    for item in result.get('surface_instances',[]):
        if recipe.get('all_observation_surfaces_reference_only') or set(item['source_building_ids'])&active_sources:
            item['assembly_enabled']=False
            for sid in item['source_building_ids']:
                sources[sid]['postprocess']['surface']['assembly_enabled']=False
    for decision in recipe.get('source_decisions',[]):
        if decision.get('removed_part_ids') and not decision.get('parts'):
            sources[decision['source_id']]['massing_completion']={
                'method':'reviewed_massing_omission','part_ids':[],
                'reason':decision['reason'],'source_record_preserved':True}
    result['massing_instances']=parts
    result['massing_completion']={'schema':'citycode.massing_completion.v1','status':'candidate_actual_review_pending',
        'baseline_assets':len(baseline['asset_instances']),'massing_parts':len(parts),
        'source_instances_completed':len(active_sources),'independent_buildings_added':0,
        'source_count_preserved':len(sources),'API_calls':0,'GPU_jobs':0,'fresh_neural_models':0,
        'role':'Plausible missing building wings and podium bodies inferred using existing textured assets.',
        'input_binding':{'path':'massing_completion/input_binding_manifest.json',
            'sha256':recipe['baseline_binding_sha256']}}
    if recipe.get('pruning'):
        result['massing_completion']['pruning']=copy.deepcopy(recipe['pruning'])
    return result


def part_dimensions(part):
    geometry=shape(part['target_geometry'])
    if geometry.is_empty or not geometry.is_valid or geometry.area<=0:
        raise ValueError('Massing part needs a valid positive footprint')
    corners=np.asarray(geometry.minimum_rotated_rectangle.exterior.coords,float)
    sides=np.linalg.norm(np.diff(corners,axis=0),axis=1)
    return {'width_m':float(sides.min()),'length_m':float(sides.max()),
        'footprint_area_m2':float(geometry.area),
        'height_m':float(part['target_top_z_m']-part['target_base_z_m'])}


def part_review_identity(part):
    """Bind individual decisions to the actual candidate, including its observed target."""
    return digest({key:part[key] for key in ['id','source_building_ids','donor_asset_id','donor_lods',
        'world_from_local_4x4','target_geometry','target_base_z_m','target_top_z_m']})


def part_omission_reason(part,policy):
    if set(part['source_building_ids'])&set(policy['excluded_source_ids']):
        return '用户复核取消该来源的全部补入体量。'
    review=policy.get('part_reviews',{}).get(part['id'])
    if review is not None:
        if (type(review.get('retain')) is not bool or not review.get('reason') or not review.get('reviewer') or
                review.get('part_identity')!=part_review_identity(part)):
            raise ValueError('Individual part review is incomplete or belongs to a different candidate: '+part['id'])
        return None if review['retain'] else review['reason']
    dims=part_dimensions(part);epsilon=1e-5
    if dims['width_m']+epsilon<policy['minimum_width_m']:
        return '补入模型过窄；保留空缺，不再按边带填充。'
    if dims['footprint_area_m2']+epsilon<policy['minimum_area_m2']:
        return '补入模型占地过小；删除用于贴合轮廓的小补丁。'
    if (dims['height_m']>=policy['tall_height_m'] and
            dims['footprint_area_m2']+epsilon<policy['tall_minimum_area_m2']):
        return '高而小的补入模型形态不自然；不保留细长塔状补丁。'
    return None


def prune_recipe(recipe,policy):
    """Produce an unapproved candidate; new scene evidence must be supplied before replay."""
    for key in ['minimum_width_m','minimum_area_m2','tall_height_m','tall_minimum_area_m2']:
        if not np.isfinite(policy[key]) or policy[key]<=0:
            raise ValueError('Pruning thresholds must be finite and positive')
    result=copy.deepcopy(recipe);decisions=[];kept=[]
    if set(policy.get('part_reviews',{}))-{part['id'] for part in recipe['parts']}:
        raise ValueError('Individual reviews reference unknown candidate parts')
    for part in recipe['parts']:
        reason=part_omission_reason(part,policy)
        explicit=bool(set(part['source_building_ids'])&set(policy['excluded_source_ids']))
        review=None if explicit else policy.get('part_reviews',{}).get(part['id'])
        decisions.append({'part_id':part['id'],'source_ids':part['source_building_ids'],
            'retain':reason is None,'reason':reason or (review['reason'] if review else '保留较完整、尺度足够的主体块。'),
            'decision_basis':'explicit_source_exclusion' if explicit else 'individual_part_review' if review else 'size_screen',
            'size_screen_reason':part_omission_reason(part,{k:v for k,v in policy.items() if k!='part_reviews'}),
            'dimensions':part_dimensions(part)})
        if reason is None:kept.append(copy.deepcopy(part))
    retained={p['id'] for p in kept};old_active={sid for p in recipe['parts'] for sid in p['source_building_ids']}
    active={sid for p in kept for sid in p['source_building_ids']}
    for source in result['source_decisions']:
        old=source.get('parts',[])
        source['parts']=[pid for pid in old if pid in retained]
        removed=[pid for pid in old if pid not in retained]
        if removed:
            source['removed_part_ids']=removed
            source['action']='reuse_assets_for_missing_massing' if source['parts'] else 'retain_without_new_massing'
            source['reason']=('逐个部件复核后保留合理体块，只删去窄小或过细长的补丁，允许原提取范围留下空缺。' if source['parts'] else
                '用户指定该来源取消补入，保留原结构和观测记录。' if source['source_id'] in policy['excluded_source_ids'] else
                '逐个部件检查后，该来源没有可保留的体块；各部件删除理由分别记录，原结构和观测保留。')
    result['parts']=kept
    result['all_observation_surfaces_reference_only']=True
    result['pruning']={'schema':'citycode.reviewed_massing_pruning.v1','policy':copy.deepcopy(policy),
        'previous_parts':len(recipe['parts']),'retained_parts':len(kept),'removed_parts':len(recipe['parts'])-len(kept),
        'previous_sources':len(old_active),'retained_sources':len(active),'removed_source_ids':sorted(old_active-active),
        'decisions':decisions,'goal':'Prefer plausible larger bodies; source-boundary coverage may decrease.'}
    result['review'].update(approved=False,scene_approval=False,proof_files=[],baseline_visual_context_proofs=[])
    result['status']='pruned_candidate_scene_review_required'
    result.pop('support_files',None)
    return result


def _render_review(task):
    from .batch_review import load_render_object,render
    root,destination,sid,before,added,own,observed=task
    root,destination=Path(root),Path(destination);folder=destination/'review'/sid;folder.mkdir(parents=True,exist_ok=True)
    def load(rows):
        return [load_render_object(verify(root,row['lods']['0.1M']),row['world_from_local_4x4']) for row in rows]
    original=load(before);extra=load(added);parts=load(own);framing=original+extra
    if not framing:raise ValueError('Scene review needs actual contextual models')
    def camera(direction):
        depth=np.asarray(direction,float);depth/=np.linalg.norm(depth)
        right=np.cross([0,0,1],depth);right/=np.linalg.norm(right)
        return np.stack([right,np.cross(depth,right),depth],axis=1)
    views={'front':camera([.72,-1,.85]),'back':camera([-.72,1,.85])}
    for name,view in views.items():
        render(original,view,side=640,framing_objects=framing).save(folder/f'before_{name}.png')
        render(original+extra,view,side=640,framing_objects=framing).save(folder/f'after_{name}.png')
        if parts:render(parts,view,side=480).save(folder/f'parts_{name}.png')
    if parts:render(parts,np.eye(3),side=480).save(folder/'parts_roof.png')
    if observed:shutil.copyfile(observed,folder/'observed.png')
    identities={row['id']:{'id':row['id'],'placement_identity':placement_identity(row),
        'lod_sha256':{label:item['sha256'] for label,item in row['lods'].items()}} for row in before+added}
    files=[{**proof(path,destination.parent),'path':'massing_completion/'+path.relative_to(destination).as_posix()}
           for path in sorted(folder.glob('*.png'))]
    record={'source_id':sid,'before_and_after_share_camera_and_frame':True,
        'baseline_neighbours':[row['id'] for row in before],'new_parts':[row['id'] for row in added],
        'own_parts':[row['id'] for row in own],'file_proofs':files,'models':list(identities.values()),
        'camera_columns_world':{name:view.tolist() for name,view in views.items()},
        'scene_review_required':True}
    write_json(folder/'PROOF.json',record)
    return {'source_id':sid,'files':files}


def render_review(baseline_root,recipe,support_root,*,workers=4):
    """Render the current candidate and its actual neighbours; this never approves it."""
    baseline_root,support_root=map(Path,(baseline_root,support_root))
    baseline=read(baseline_root/'binding_manifest.json');parts=recipe['parts']
    donors=baseline['asset_instances'];rows={r['building_id']:r for r in baseline['buildings']}
    decisions={d['source_id']:d for d in recipe['source_decisions']}
    features=read(support_root/'baseline_footprints.geojson')['features']
    geometry={f['properties']['asset_id']:shape(f['geometry']) for f in features}
    visual={r['building_id'] for r in baseline['buildings'] if r['status']=='accepted_scene_completion'}
    active={sid for part in parts for sid in part['source_building_ids']};tasks=[]
    for sid in sorted(active|visual):
        g=shape(decisions[sid]['source_geometry'] if sid in decisions else rows[sid]['target_geometry'])
        parent=decisions.get(sid,{}).get('parent_building_id')
        neighbours=sorted(donors,key=lambda a:(not(sid in a['source_building_ids'] or parent in a['source_building_ids']),
            geometry[a['id']].distance(g)))[:5]
        own=[p for p in parts if sid in p['source_building_ids']]
        nearby=[p for p in parts if shape(p['target_geometry']).distance(g)<80]
        # Candidate part paths are the unchanged donor files in the baseline.
        added=[{**p,'lods':p['donor_lods']} for p in nearby]
        own=[{**p,'lods':p['donor_lods']} for p in own]
        source=verify(baseline_root,decisions[sid]['source_building'],baseline_root.parent) if sid in decisions else (baseline_root/rows[sid]['structure_path']).resolve()
        building=read(source);observations=building.get('observations',[])
        canonical=next((o for o in observations if o['observation_id']==building.get('canonical_observation_id')),None)
        observed=source.parent/canonical['path']/'source_primary.png' if canonical else None
        tasks.append((str(baseline_root),str(support_root),sid,neighbours,added,own,str(observed) if observed and observed.is_file() else None))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_render_review,task) for task in tasks]):
            result=future.result();results.append(result);print(json.dumps({'review_rendered':result['source_id']}),flush=True)
    qualified=[];contexts=[]
    for result in results:
        sid=result['source_id'];files={Path(p['path']).name:p for p in result['files']}
        roles=[('after_front.png','actual_scene_front'),('after_back.png','actual_scene_back')]
        if sid in active:roles.append(('observed.png','observed_source'))
        output=contexts if sid in visual else qualified
        for name,role in roles:
            if name not in files:raise ValueError('Current review lacks '+name+' for '+sid)
            output.append({**files[name],'source_id':sid,'role':role})
    return {'proof_files':sorted(qualified,key=lambda p:(p['source_id'],p['role'])),
        'baseline_visual_context_proofs':sorted(contexts,key=lambda p:(p['source_id'],p['role'])),
        'scene_approval':False}


def _measure(task):
    root,row,source_geometry,baseline_features=task
    from .registration import read_glb_scene_geometry,raster_triangle_footprint,overlap_metrics
    from .lod import inspect_glb
    root=Path(root);matrix=np.asarray(row['world_from_local_4x4']);target=shape(row['target_geometry']);source=shape(source_geometry)
    checks={};bounds=[];high=None
    for label,item in row['lods'].items():
        path=verify(root,item);inspection=inspect_glb(path);scene=read_glb_scene_geometry(path)
        world=scene['vertices']@matrix[:3,:3].T+matrix[:3,3];bounds.append([world.min(0),world.max(0)])
        geom,profile=raster_triangle_footprint(scene['vertices'],scene['faces'],matrix,cell_m=.10,max_dimension=4096)
        metrics=overlap_metrics(geom,target);support=geom.intersection(source).area/geom.area
        if metrics['target_coverage']<.65 or metrics['asset_coverage']<.94 or support<.94:
            raise ValueError('Actual massing mesh does not remain in the observed missing patch: '+row['id'])
        tolerance=.75
        if (abs(float(world[:,2].min())-row['target_base_z_m'])>tolerance or
                abs(float(world[:,2].max())-row['target_top_z_m'])>tolerance):
            raise ValueError('Massing placement is not grounded at its declared height')
        checks[label]={'mesh_sha256':inspection['sha256'],'faces':len(scene['faces']),
            'actual_source_support':support,'target_overlap':metrics,'projection':profile,
            'world_z':[float(world[:,2].min()),float(world[:,2].max())]}
        if label=='0.25M':high=geom
    if np.max(abs(np.asarray(bounds)[0]-np.asarray(bounds)[1]))>max(.75,row['target_top_z_m']*.025):
        raise ValueError('Massing LOD frames disagree')
    hits=[]
    for feature in baseline_features:
        other=shape(feature['geometry'])
        if not high.intersects(other):continue
        area=high.intersection(other).area
        if area>.01:hits.append({'other':feature['properties']['asset_id'],'area_m2':area,'fraction':area/high.area,
            'inset_05m_area_m2':high.buffer(-.5).intersection(other.buffer(-.5)).area})
    overlap=sum(h['area_m2'] for h in hits)/high.area
    if overlap>.03 or any(h['inset_05m_area_m2']>.25 for h in hits):
        raise ValueError('Massing intersects existing model cores: '+row['id'])
    return {'id':row['id'],'status':'passed','placement_identity':placement_identity(row),'lods':checks,
        'actual_footprint':mapping(high),'baseline_intersections':hits,'total_intersection_fraction':overlap}


def measure_manifest(root,manifest,recipe,baseline_features,*,workers=4):
    root=Path(root);sources={r['source_id']:r for r in recipe['source_decisions']}
    tasks=[]
    for row in manifest['massing_instances']:
        near=shape(row['target_geometry']).buffer(2)
        features=[f for f in baseline_features if shape(f['geometry']).intersects(near)]
        tasks.append((str(root),row,sources[row['source_building_ids'][0]]['source_geometry'],features))
    records=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_measure,t):t[1]['id'] for t in tasks}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as exc:errors.append({'id':futures[future],'error':str(exc)})
            print(json.dumps({'massing_checked':len(records)+len(errors),'expected':len(tasks),'errors':len(errors)}),flush=True)
    pairs=[]
    for i,a in enumerate(records):
        g=shape(a['actual_footprint'])
        for b in records[i+1:]:
            t=shape(b['actual_footprint']);area=g.intersection(t).area
            if area>.01:
                item={'a':a['id'],'b':b['id'],'intersection_m2':area,
                    'fraction_a':area/g.area,'fraction_b':area/t.area,
                    'inset_05m_area_m2':g.buffer(-.5).intersection(t.buffer(-.5)).area};pairs.append(item)
                if max(item['fraction_a'],item['fraction_b'])>.03 or item['inset_05m_area_m2']>.25:
                    errors.append({'id':a['id'],'error':'New massing parts have a core intersection','other':b['id']})
    return {'schema':'citycode.actual_massing_geometry.v1','status':'passed' if not errors else 'failed',
        'records':sorted(records,key=lambda r:r['id']),'errors':errors,'new_part_intersections':pairs,
        'all_actual_triangles_used':True,'no_GT_used':True,'scope':'Actual high/low mesh projections, heights and XY core intersections; not closed-solid collision.'}


def validate_massing(root,manifest,*,geometry=False):
    root=Path(root);parts=manifest.get('massing_instances',[])
    if not parts:return {'status':'passed','massing_parts':0}
    meta=manifest['massing_completion'];recipe_path=verify(root,meta['recipe']);recipe=read(recipe_path)
    if recipe.get('review',{}).get('approved') is not True or not recipe['review'].get('reviewer'):
        raise ValueError('Massing candidates need an explicit current-model and scene review')
    source_ids={r['building_id'] for r in manifest['buildings']};donors={r['id']:r for r in manifest['asset_instances']}
    selected={r['id']:r for r in recipe['parts']}
    if len(selected)!=len(parts) or set(selected)!={r['id'] for r in parts}:raise ValueError('Massing selection changed')
    if recipe.get('pruning'):
        selection=recipe['pruning'];audit=selection['decisions']
        if ({a['part_id'] for a in audit if a['retain']}!=set(selected) or
                len({a['part_id'] for a in audit})!=selection['previous_parts'] or
                selection['retained_parts']!=len(parts) or selection['removed_parts']+len(parts)!=len(audit)):
            raise ValueError('Pruning decisions do not match the retained part inventory')
        for part in parts:
            if part_omission_reason(part,selection['policy']):
                raise ValueError('Retained part fails its reviewed size policy: '+part['id'])
    baseline=read(verify(root,meta['input_binding']));expected=construct_manifest(baseline,recipe)
    if sha256_file(verify(root,meta['input_binding']))!=recipe['baseline_binding_sha256']:
        raise ValueError('Massing baseline differs from the reviewed input')
    _attach_context_review(expected,recipe)
    for key in ['source_structure','buildings','asset_instances','building_entities','surface_instances']:
        if digest(manifest.get(key))!=digest(expected.get(key)):
            raise ValueError('Massing completion changed protected baseline records: '+key)
    if digest(parts)!=digest(expected['massing_instances']):raise ValueError('Massing mesh or pose differs from its reviewed recipe')
    for part in parts:
        if set(part['source_building_ids'])-source_ids or part['independent_building']:
            raise ValueError('Massing must retain its qualified partial-source relationship')
        for label,item in part['lods'].items():
            path=verify(root,item)
            if item!=donors[part['donor_asset_id']]['lods'][label]:raise ValueError('Massing donor identity changed')
    decisions=recipe['source_decisions']
    if len({d['source_id'] for d in decisions})!=len(decisions):raise ValueError('Duplicate massing source decision')
    for decision in decisions:
        source=read(verify(root,decision['source_building'],root.parent))
        if (source['building_id']!=decision['source_id'] or
                (decision.get('parts') and source['footprint_world']!=decision['source_geometry'])):
            raise ValueError('Massing support polygon is not the retained current source')
        declared={p['id'] for p in parts if decision['source_id'] in p['source_building_ids']}
        if declared!=set(decision.get('parts',[])):raise ValueError('Massing source decision does not match its parts')
    proofs=recipe['review']['proof_files']
    active={sid for part in parts for sid in part['source_building_ids']}
    for sid in active:
        roles={p.get('role') for p in proofs if p.get('source_id')==sid}
        if not {'observed_source','actual_scene_front','actual_scene_back'}<=roles:
            raise ValueError('Massing source lacks observed and actual scene review evidence: '+sid)
    for item in proofs:verify(root,item)
    context_proofs=recipe['review'].get('baseline_visual_context_proofs',[])
    for row in baseline['buildings']:
        if row['status']!='accepted_scene_completion':continue
        roles={p.get('role') for p in context_proofs if p.get('source_id')==row['building_id']}
        if not {'actual_scene_front','actual_scene_back'}<=roles:
            raise ValueError('Changed scene needs current context evidence for '+row['building_id'])
    for item in context_proofs:verify(root,item)
    features=read(verify(root,meta['baseline_footprints']))['features']
    if len(features)!=len(donors) or {f['properties']['asset_id'] for f in features}!=set(donors):
        raise ValueError('Actual baseline footprint inventory is incomplete')
    for feature in features:
        p=feature['properties'];donor=donors[p['asset_id']]
        if (p['sha256']!=donor['lods']['0.25M']['sha256'] or
                p['world_from_local_4x4']!=donor['world_from_local_4x4']):
            raise ValueError('Actual baseline footprint belongs to different model bytes or placement')
    measured=read(verify(root,meta['geometry_validation']))
    if measured['status']!='passed' or {r['id']:r['placement_identity'] for r in measured['records']}!={r['id']:placement_identity(r) for r in parts}:
        raise ValueError('Massing geometry checks belong to different bytes or placements')
    if geometry:
        measured=measure_manifest(root,manifest,recipe,features)
        if measured['status']!='passed':raise ValueError('Actual massing geometry failed: '+str(measured['errors']))
    return {'status':'passed','massing_parts':len(parts),'completed_source_instances':len({s for p in parts for s in p['source_building_ids']}),
        'independent_buildings_added':0,'actual_geometry_recomputed':bool(geometry)}


def _attach_context_review(result,recipe):
    context=scene_identity(result)
    for row in result['buildings']:
        if row['status']=='accepted_scene_completion':
            review=row['scene_completion_review'];review['previous_scene_identity']=review['scene_identity'];review['scene_identity']=context
            review['augmentation_context_review']={'reviewer':recipe['review']['reviewer'],
                'reason':'Unchanged building bytes and placement; current retained inferred placements reviewed in actual scene context.',
                'proof_files':recipe['review']['baseline_visual_context_proofs']}


def write_coverage(root,manifest,recipe):
    """Report coverage of qualified partial sources, never a GT accuracy score."""
    from shapely.strtree import STRtree
    from shapely.ops import unary_union
    root=Path(root);meta=manifest['massing_completion']
    baseline=[shape(f['geometry']) for f in read(verify(root,meta['baseline_footprints']))['features']]
    tree=STRtree(baseline)
    measured={r['id']:shape(r['actual_footprint']) for r in read(verify(root,meta['geometry_validation']))['records']}
    parts={p['id']:p for p in manifest['massing_instances']};records=[]
    for decision in recipe['source_decisions']:
        if not decision.get('parts'):continue
        source=shape(decision['source_geometry'])
        old=unary_union([baseline[i].intersection(source) for i in tree.query(source,predicate='intersects')])
        new=unary_union([measured[pid] for pid in decision['parts']]).intersection(source)
        before=old.area/source.area;added=new.difference(old).area/source.area
        records.append({'source_id':decision['source_id'],'parts':len(decision['parts']),'source_area_m2':source.area,
            'before_actual_model_xy_coverage':before,'added_actual_model_xy_coverage':added,
            'after_actual_model_xy_coverage':min(1,before+added),
            'target_roof_z_m':parts[decision['parts'][0]]['target_top_z_m']})
    area=sum(r['source_area_m2'] for r in records)
    summary={'scope':'Only qualified partial-source polygons; actual-mesh XY coverage, not GT accuracy or building recall.',
        'records':records,'qualified_source_count':len(records),'part_count':len(parts),
        'area_weighted_before':sum(r['before_actual_model_xy_coverage']*r['source_area_m2'] for r in records)/area if area else 0,
        'area_weighted_after':sum(r['after_actual_model_xy_coverage']*r['source_area_m2'] for r in records)/area if area else 0,
        'area_weighted_added':sum(r['added_actual_model_xy_coverage']*r['source_area_m2'] for r in records)/area if area else 0,
        'independent_buildings_added':0,'no_GT_used':True}
    write_json(root/'massing_completion/coverage_summary.json',summary)
    with (root/'massing_completion/coverage_summary.csv').open('w') as stream:
        fields=list(records[0]) if records else ['source_id','parts','source_area_m2','before_actual_model_xy_coverage','added_actual_model_xy_coverage','after_actual_model_xy_coverage','target_roof_z_m']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
    return summary


def write_output_readme(project,root,baseline_root):
    """Keep the cached baseline guide separately from the current output guide."""
    from .cache_batch import _copy_verified
    project,root,baseline_root=map(Path,(project,root,baseline_root))
    baseline_guide=baseline_root/'README.md'
    if baseline_guide.exists():
        _copy_verified(baseline_guide,root/'massing_completion/BASELINE_README.md',proof(baseline_guide,baseline_root))
    template=project/'workflow/02_building_assets/COMPLETED_OUTPUT_README.md';target=root/'README.md'
    allowed={sha256_file(template)}
    if baseline_guide.exists():allowed.add(sha256_file(baseline_guide))
    if target.exists() and sha256_file(target) not in allowed:raise ValueError('Completed output guide changed')
    shutil.copyfile(template,target)


def apply_recipe(config,baseline_root,*,workers=4):
    from .cache_batch import _copy_verified
    from .cli import validate_assets
    project=Path(config['_project_root']);root=Path(config['output']);baseline_root=Path(baseline_root)
    recipe_path=project/config['massing_completion_recipe'];recipe=read(recipe_path)
    if sha256_file(baseline_root/'binding_manifest.json')!=recipe['baseline_binding_sha256']:
        raise ValueError('Massing decisions require the unchanged reviewed baseline')
    root.mkdir(parents=True,exist_ok=True)
    for item in read(config['asset_cache_manifest'])['files']:
        if item['path']=='README.md':continue
        destination=root/item['path'] if item['path']!='binding_manifest.json' else root/'massing_completion/input_binding_manifest.json'
        _copy_verified(baseline_root/item['path'],destination,item)
    support=project/config['massing_completion_support']
    for item in recipe['support_files']:
        src=verify(support,item);dest=root/'massing_completion'/item['path'];_copy_verified(src,dest,item)
    dest=root/'massing_completion/recipe.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and sha256_file(dest)!=sha256_file(recipe_path):raise ValueError('Existing massing selection differs; use a new output')
    if not dest.exists():shutil.copyfile(recipe_path,dest)
    baseline=read(baseline_root/'binding_manifest.json');result=construct_manifest(baseline,recipe)
    meta=result['massing_completion'];meta.update(status='reviewed_complete',recipe=proof(dest,root),
        input_binding=proof(root/'massing_completion/input_binding_manifest.json',root),
        baseline_footprints=proof(root/'massing_completion/baseline_footprints.geojson',root),
        geometry_validation=proof(root/'massing_completion/geometry_validation.json',root))
    _attach_context_review(result,recipe)
    result['web_preview_directory']='../smallcity_p384_building_assets_completed_web_preview'
    candidate=root/'binding_manifest.massing_candidate.json';write_json(candidate,result)
    validation=validate_assets(candidate,require_complete=True,geometry=False)
    write_json(root/'massing_completion/final_validation.json',validation)
    if validation['status']!='passed' or not validation['complete']:raise ValueError('Completed massing binding failed: '+str(validation['errors']))
    target=root/'binding_manifest.json'
    if target.exists() and sha256_file(target)!=sha256_file(candidate):raise ValueError('Existing completed binding differs')
    if not target.exists():shutil.copyfile(candidate,target)
    write_coverage(root,result,recipe)
    write_output_readme(project,root,baseline_root)
    report={'status':'complete_reviewed_massing','source_instances':len(result['buildings']),
        'baseline_physical_assets':len(result['asset_instances']),'massing_parts_added':len(result['massing_instances']),
        'rendered_model_instances':len(result['asset_instances'])+len(result['massing_instances']),
        'logical_building_entities':len(result['building_entities']),'independent_buildings_added':0,
        'sources_completed':meta['source_instances_completed'],'API_calls':0,'GPU_jobs':0,'fresh_neural_models':0,
        'baseline_models_and_placements_unchanged':True,'binding_manifest':str(target),
        'recipe_sha256':sha256_file(recipe_path),'complete':True}
    write_json(root/'massing_completion/RESULT.json',report);return report
