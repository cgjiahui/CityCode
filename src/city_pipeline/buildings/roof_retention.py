"""Keep otherwise-pruned boundary buildings with directly observed stable roofs.

Only predicted footprint/observation metadata and referenced provenance=1 XYZ
are inputs. No cache, evaluation table, reference count or per-object config.
Run this at the prune stage, before local/replacement boundary repair.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import math
from pathlib import Path

import numpy as np
from shapely.geometry import shape
from city_pipeline.buildings.polish import normal_statistics, _prediction_path
from city_pipeline.runtime import sha256

DEFAULT_CONFIG = {
    "name": "conservative_observed_roof_retention",
    "prune_maximum_area_m2": 1200.,
    "minimum_area_m2": 100., "minimum_width_m": 8., "minimum_height_m": 2.,
    "minimum_stable_layers": 1,
    "compact_enabled": True, "compact_minimum_rectangle_fill": .8,
    "compact_maximum_aspect": 2.,
    "clipped_enabled": False, "clipped_maximum_height_m": 30.,
    "clipped_maximum_aspect": 3.,
    "minimum_direct_points": 500, "minimum_planar_samples": 200,
    "minimum_horizontal_fraction": .6, "maximum_vertical_fraction": .1,
    "top_band_half_width_m": 3., "minimum_top_band_fraction": .9,
    "maximum_q95_minus_q90_m": 1., "maximum_q99_minus_q95_m": 1.,
}

def _lookup(observations):
    if isinstance(observations,dict) and 'features' in observations:
        observations=observations['features']
    if isinstance(observations,dict):
        return {str(k):v.get('properties',v) for k,v in observations.items()}
    return {str(item.get('properties',item)['observation_id']):item.get('properties',item) for item in observations}

def _configuration(config):
    unexpected=set(config or {})-set(DEFAULT_CONFIG)
    if unexpected:raise ValueError('Unknown roof retention parameters: '+','.join(sorted(unexpected)))
    cfg={**DEFAULT_CONFIG,**copy.deepcopy(config or {})}
    for key,value in cfg.items():
        if key=='name':continue
        if key.endswith('_enabled'):
            if not isinstance(value,bool):raise ValueError('Expected boolean '+key)
        elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Expected finite nonnegative parameter '+key)
    for key in ['compact_minimum_rectangle_fill','minimum_horizontal_fraction','maximum_vertical_fraction','minimum_top_band_fraction']:
        if cfg[key]>1:raise ValueError('Expected fraction '+key)
    return cfg

def observed_roof_retention(features,observations,data_root,config=None,point_boundary_root=None):
    """Return observation IDs to retain and complete predicted-evidence audit.

Any missing/invalid metadata, point file or unreliable normal leaves the old
prune decision unchanged. IDs are audit outputs; thresholds are global.
    """
    cfg=_configuration(config);lookup=_lookup(observations);checks=[];grouped=defaultdict(list)
    for f in features:
        p=f['properties'];ids=p.get('consensus_observation_ids',[])
        if len(ids)!=1:continue
        oid=str(ids[0]);o=lookup.get(oid)
        if o is None:continue
        status=o.get('boundary_status')
        if status=='interior_keep':continue
        g=shape(f['geometry'])
        if g.is_empty or not g.is_valid or g.geom_type not in ['Polygon','MultiPolygon'] or g.area>cfg['prune_maximum_area_m2']:continue
        check={'observation_id':oid,'building_id':p.get('predicted_global_building_id'),'decision':'keep_old_prune','boundary_status':status,'area_m2':float(g.area)};checks.append(check)
        rect=g.minimum_rotated_rectangle
        if rect.geom_type!='Polygon' or rect.area<=0:check['reason']='invalid_rectangle';continue
        lengths=np.linalg.norm(np.diff(np.asarray(rect.exterior.coords,float),axis=0),axis=1);positive=lengths[lengths>1e-9]
        if not len(positive):check['reason']='invalid_width';continue
        width=float(positive.min());aspect=float(positive.max()/width);fill=float(g.area/rect.area)
        top=p.get('top_z_m');height=p.get('height_m')
        if not isinstance(top,(int,float)) or not isinstance(height,(int,float)) or not math.isfinite(top) or not math.isfinite(height):check['reason']='missing_height';continue
        check.update(width_m=width,aspect=aspect,rectangle_fill=fill,height_m=float(height),top_z_m=float(top))
        if g.area<cfg['minimum_area_m2'] or width<cfg['minimum_width_m'] or height<=cfg['minimum_height_m'] or o.get('stable_layer_count',0)<cfg['minimum_stable_layers']:
            check['reason']='geometry_or_stable_height_gate';continue
        compact=(cfg['compact_enabled'] and status=='boundary_risk_review' and fill>=cfg['compact_minimum_rectangle_fill'] and aspect<=cfg['compact_maximum_aspect'])
        clipped=(cfg['clipped_enabled'] and status=='boundary_clipped_exclude' and height<=cfg['clipped_maximum_height_m'] and aspect<=cfg['clipped_maximum_aspect'])
        if not(compact or clipped):check['reason']='roof_shape_gate';continue
        check['roof_shape_branch']='compact' if compact else 'clipped_lowrise'
        value=p.get('canonical_building_point_cloud_npz_path') or o.get('building_point_cloud_npz_path');instance=p.get('canonical_instance_id',o.get('rgb_instance_id'))
        if not value or instance is None:check['reason']='missing_point_reference';continue
        try:path=_prediction_path(value,data_root,point_boundary_root);instance=int(instance)
        except (PermissionError,ValueError,TypeError):check['reason']='invalid_point_reference';continue
        grouped[path].append((check,instance))
    retained=[];point_hashes={};reads=[]
    for path,items in sorted(grouped.items(),key=lambda item:str(item[0])):
        try:
            with np.load(path,allow_pickle=False) as data:
                xyz,ids,prov=data['xyz_m'],data['instance_id'],data['provenance']
                if xyz.ndim!=2 or xyz.shape[1]!=3 or ids.shape!=(len(xyz),) or prov.shape!=(len(xyz),):raise ValueError('Inconsistent XYZ arrays')
            point_hashes[str(path)]=sha256(path);reads.append(str(path))
        except (OSError,KeyError,ValueError):
            for check,_ in items:check['reason']='unavailable_or_invalid_points'
            continue
        for check,instance in items:
            pts=xyz[(ids==instance)&(prov==1)];pts=pts[np.isfinite(pts).all(1)];stats=normal_statistics(pts);check['normal_statistics']=stats
            hf,vf=stats['horizontal_fraction'],stats['vertical_fraction']
            if stats['direct_point_count']<cfg['minimum_direct_points'] or stats['planar_sample_count']<cfg['minimum_planar_samples'] or hf is None or vf is None or hf<cfg['minimum_horizontal_fraction'] or vf>cfg['maximum_vertical_fraction']:
                check['reason']='unreliable_or_insufficient_horizontal_roof';continue
            z=pts[:,2];q90,q95,q99=map(float,np.quantile(z,[.90,.95,.99]));band=float(np.mean(np.abs(z-check['top_z_m'])<=cfg['top_band_half_width_m']))
            check.update(top_band_fraction=band,q90_m=q90,q95_m=q95,q99_m=q99,q95_minus_q90_m=q95-q90,q99_minus_q95_m=q99-q95)
            if band<cfg['minimum_top_band_fraction'] or q95-q90>cfg['maximum_q95_minus_q90_m'] or q99-q95>cfg['maximum_q99_minus_q95_m']:
                check['reason']='unstable_top_height_evidence';continue
            check.update(decision='retain_observed_boundary_roof',reason='direct_predicted_roof_support');retained.append(check['observation_id'])
    return sorted(retained),{'parameters':cfg,'uses_ground_truth':False,'feature_cache_read':False,'checks':checks,'retained_observation_ids':sorted(retained),'source_sha256':point_hashes,'runtime_data_reads':reads,'position_in_pipeline':'Before prune; retained candidates still undergo local/replace repair and final filtering/polish.'}
