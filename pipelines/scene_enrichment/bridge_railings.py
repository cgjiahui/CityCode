#!/usr/bin/env python3
"""Place native 4m guardrails on the outer two edges of predicted bridge spans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import CubicHermiteSpline

MESH='/Game/Road/Kit_Freeway_A/Mesh/SM_Railing_400_A_01.SM_Railing_400_A_01'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'ue_citycode'))
from prepare import rotation_matrix
from bridge_assets import span_spline, landing_crossfall


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rail_pose(start,end,outward_side,bounds,overlap=.05):
    """Native local +Y is the run; local negative X lies inside the road.

    Left edge runs backwards so both sides have a proper rotation with positive
    scales. Native Z=0 is the road-deck plane, not the mesh's hanging lower bbox.
    """
    start,end=np.asarray(start,float),np.asarray(end,float)
    forward=end-start;length=float(np.linalg.norm(forward));forward/=length
    heading=float(np.degrees(np.arctan2(forward[1],forward[0])))
    grade=float(np.degrees(np.arctan2(forward[2],np.linalg.norm(forward[:2]))))
    right=outward_side<0
    rotation=[0.,heading-90. if right else heading+90.,-grade if right else grade]
    matrix=rotation_matrix(rotation)
    direction=forward if right else -forward
    bounds=np.asarray(bounds,float)/100.
    scale=[1.,(length+overlap)/(bounds[1,1]-bounds[0,1]),1.]
    anchor=(start if right else end)-matrix[:,0]*bounds[1,0]-direction*(bounds[0,1]*scale[1]+overlap*.5)
    # Independent convention guard: the consumer's actual UE Rotator matrix is
    # checked against the desired native longitudinal and outward directions.
    if np.max(abs(matrix[:,1]-direction))>1e-8:raise ValueError('UE native railing axis differs from road chord')
    return anchor.tolist(),rotation,scale



def same_level_interior(points, fields, own_span):
    for identity,field in fields.items():
        if identity==own_span:continue
        a,b,width0,width1=field
        delta=b[:,:2]-a[:,:2]
        denominator=np.maximum(np.einsum('ij,ij->i',delta,delta),1e-12)
        inside=[]
        for point in points:
            t=np.einsum('ij,ij->i',point[:2]-a[:,:2],delta)/denominator
            valid=(t>=0)&(t<=1)
            clipped=np.clip(t,0.,1.)
            closest=a+clipped[:,None]*(b-a)
            width=width0+clipped*(width1-width0)
            distance=np.linalg.norm(closest[:,:2]-point[:2],axis=1)
            inside.append(bool(np.any(valid&(distance<width*.5-.3)&(abs(closest[:,2]-point[2])<=.75))))
        if inside[1] or sum(inside)>=2:return identity
    return None


def generate(proposal_path,catalog_path,spans_path,output):
    proposal=json.loads(proposal_path.read_text());spans=json.loads(spans_path.read_text())
    if sha(proposal_path)!=spans['source_proposal_sha256']:raise ValueError('Railings must use exactly the frozen swept-span prediction proposal')
    asset=next(r for r in json.loads(catalog_path.read_text())['assets'] if r['mesh_path']==MESH)
    native_length=(asset['bounds_cm'][1][1]-asset['bounds_cm'][0][1])/100.
    if not 3.5<native_length<4.5:raise ValueError('Native guardrail run length is not approximately 4m')
    available={r['span_id']for r in spans['templates']};span_records={r['span_id']:r for r in spans['templates']}
    rows=[];rejected=[];maximum_run_error=0.
    fields={};landing_offsets={}
    for span in proposal['continuous_segments']:
        if span['segment_id']not in available:continue
        modules=sorted([r for r in proposal['placements']if r['source_segment_id']==span['segment_id']],key=lambda r:r['placement_id'])
        p=span['profile'];s=np.asarray([r['s_m']for r in p]);xyz=np.asarray([r['position_m']for r in p])
        s,spline=span_spline(span)
        sample=np.linspace(0.,s[-1],int(np.ceil(s[-1]/2.))+1);points=spline(sample)
        points,_=landing_crossfall(span,sample,points)
        width=np.asarray([modules[0]['width_m']]+[min(a['width_m'],b['width_m'])for a,b in zip(modules[:-1],modules[1:])]+[modules[-1]['width_m']])
        widths=np.interp(sample,np.linspace(0.,s[-1],len(modules)+1),width)
        fields[span['segment_id']]=(points[:-1],points[1:],widths[:-1],widths[1:])
    for span in proposal['continuous_segments']:
        identity=span['segment_id']
        if identity not in available:continue
        modules=sorted([row for row in proposal['placements']if row['source_segment_id']==identity],key=lambda row:row['placement_id'])
        profile=span['profile'];s=np.asarray([p['s_m']for p in profile]);xyz=np.asarray([p['position_m']for p in profile])
        s,spline=span_spline(span)
        module_s=np.linspace(0.,s[-1],len(modules)+1)
        width=np.asarray([modules[0]['width_m']]+[min(a['width_m'],b['width_m'])for a,b in zip(modules[:-1],modules[1:])]+[modules[-1]['width_m']])
        count=max(1,int(np.ceil(s[-1]/native_length)));stations=np.linspace(0.,s[-1],count+1)
        center=spline(stations);tangent=spline(stations,1);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        side=np.column_stack((-tangent[:,1],tangent[:,0],np.zeros(len(tangent))));side/=np.linalg.norm(side,axis=1)[:,None]
        widths=np.interp(stations,module_s,width)
        for outward in (-1,1):
            lateral=.5*outward
            if span.get('ground_transition'):
                native_edges=span_records[identity].get('native_section_lateral_bounds_m')
                if native_edges is None:raise ValueError('Landing railings require actual swept section bounds')
                lateral=native_edges[0 if outward<0 else 1]/10.
            edge=center+side*widths[:,None]*lateral
            edge,offset=landing_crossfall(span,stations,edge)
            landing_offsets[identity]=max(landing_offsets.get(identity,0.),offset)
            for index,(a,b)in enumerate(zip(edge[:-1],edge[1:])):
                if min(widths[index:index+2])<6.:
                    rejected.append({'span_id':identity,'side':outward,'segment_index':index,'reason':'native_hanging_rail_profile_leaves_insufficient_road_width'});continue
                blocker=same_level_interior(np.asarray([a,(a+b)*.5,b]),fields,identity)
                if blocker:
                    rejected.append({'span_id':identity,'side':outward,'segment_index':index,'reason':'railing_would_block_same_level_neighbouring_deck','neighbour_span_id':blocker});continue
                anchor,rotation,scale=rail_pose(a,b,outward,asset['bounds_cm'])
                matrix=rotation_matrix(rotation);native=np.asarray(asset['bounds_cm'])/100.
                # Compare actual transformed longitudinal bbox with desired
                # endpoints, including the explicitly allowed 2.5cm overlap.
                midpoint_x=native[1,0]
                runpoints=[]
                for y in native[:,1]:
                    local=np.asarray([midpoint_x,y,0.])*scale
                    runpoints.append(np.asarray(anchor)+matrix@local)
                goal=np.asarray([a,b]) if outward<0 else np.asarray([b,a])
                error=np.max(np.linalg.norm(np.asarray(runpoints)-goal,axis=1))
                maximum_run_error=max(maximum_run_error,float(error))
                if error>.025001:raise ValueError('Guardrail native pivot/run mismatch')
                rows.append({'placement_id':f'{identity}_RAIL_{"R" if outward<0 else "L"}_{index+1:04d}',
                             'category':'bridge_railing','asset_id':MESH,'mesh_path':MESH,'position_m':anchor,
                             'rotation_deg':rotation,'scale':scale,'placement_source':'predicted_elevated_structure',
                             'source_span_id':identity,'road_edge_start_m':a.tolist(),'road_edge_end_m':b.tolist(),
                             'native_road_plane_z_cm':0.,'edge_side':'right'if outward<0 else'left',
                             'overlap_m':.05,'status':'ready'})
    result={'schema':'citycode.predicted_bridge_railings.v1','coordinate_system':{'world':'UE','units':'m','up':'Z'},
            'uses_original_bridge_actor_transforms':False,'uses_ground_truth_for_placement':False,
            'placements':rows,'placement_count':len(rows),'withheld_placements':rejected,'withheld_count':len(rejected),
            'source_proposal_sha256':sha(proposal_path),'source_span_manifest_sha256':sha(spans_path),
            'native_asset_catalog_sha256':sha(catalog_path),'code_sha256':sha(Path(__file__)),
            'native_asset':asset,'policy':{'only_outer_two_edges':True,'native_Y_run_m':native_length,
                                         'native_X_positive_outside':'Mesh local maxX near0 and body lies negativeX; align outermost face with deck edge',
                                         'native_Z_zero':'Road-deck plane; bbox bottom is hanging below the slab and is not used as a footing',
                                         'maximum_symmetric_chord_overlap_m':.05,'minimum_road_width_for_this_native_profile_m':6.,
                                         'junction_policy':'Suppress a candidate whose midpoint or two of three chord samples lie >0.3m inside another deck at matching height within0.75m; preserve rails across different bridge levels',
                                         'swept_geometry_frame':'Identical frozen centerline Hermite interpolation, conservative joint widths, and measured landing crossfall helper as native swept spans'},
            'qa':{'status':'PASS','actual_native_pivot_endpoint_maximum_error_with_intended_overlap_m':maximum_run_error,
                  'landing_crossfall_maximum_vertical_offset_m':max(landing_offsets.values(),default=0.),
                  'landing_crossfall_span_offsets_m':{k:v for k,v in landing_offsets.items()if v>0.},
                  'finite_positive_scale':all(np.all(np.isfinite(r['position_m']+r['rotation_deg']+r['scale']))and min(r['scale'])>0 for r in rows),
                  'ue_actor_creation_executed':False}}
    output.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proposal',type=Path,required=True);parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--span-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=generate(a.proposal,a.catalog,a.span_manifest,a.output)
    print(json.dumps({'placement_count':r['placement_count'],'withheld_count':r['withheld_count'],'qa':r['qa']}))
