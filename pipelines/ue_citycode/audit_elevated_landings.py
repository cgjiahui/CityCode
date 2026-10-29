"""Audit actual landing top triangles against the currently prepared terrain.

Checks the true native section edges plus three interior points. Vertical
triangle intersections avoid interpreting an underside/cap vertex as the deck
surface. Every ground layer remains in the report; no flat fallback is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_elevated_geometry import load_triangles, read
from interchange import digest_file, read_glb, accessor
from prepare_elevated_repair import write, height_hits
from bridge_assets import span_spline, landing_crossfall_samples


def top_triangle_hits(triangles, xy):
    hits = height_hits(triangles, xy)
    if not len(hits):
        raise ValueError("No actual deck triangle at physical section sample")
    return hits, float(hits.max())


def audit(source_manifest, surface_proposal, building_manifest, output):
    manifest, proposal, baseline = [Path(path).resolve() for path in (source_manifest, surface_proposal, building_manifest)]
    output = Path(output).resolve()
    if output in (manifest, proposal, baseline):
        raise ValueError("Audit output must not overwrite its inputs")
    before = {path: digest_file(path) for path in (manifest, proposal, baseline)}
    sources={r['span_id']:r for r in read(manifest)['templates']};rows=[];targets=[]
    for span in read(proposal)['continuous_segments']:
        if not span.get('ground_transition'):continue
        record=sources[span['segment_id']]
        tris,sha=load_triangles(manifest.parent/record['template_file'],record['position_m'],True)
        if sha!=record['template_sha256']:raise ValueError('Source changed')
        _,spline=span_spline(span);lateral,declared_ground,length,_=landing_crossfall_samples(span)
        end=spline(length);direction=spline(length,1);direction/=np.linalg.norm(direction)
        side=np.array([-direction[1],direction[0],0.]);side/=np.linalg.norm(side)
        vertices=tris.reshape(-1,3);terminal=vertices[np.abs((vertices-end)@direction)<3e-5]
        across=(terminal-end)@side;order=np.argsort(across);clusters=[]
        for index in order:
            if not clusters or across[index]-across[clusters[-1][-1]]>3e-5:clusters.append([index])
            else:clusters[-1].append(index)
        # At each real terminal lateral coordinate, top and underside have the
        # same XY. Select the actual top; retain the measured real end boundaries.
        top_indices=[max(group,key=lambda i:terminal[i,2])for group in clusters]
        x=across[top_indices]
        sample_x=np.linspace(x.min()+.00005,x.max()-.00005,5)
        # Vertical triangle intersections are essential: a cap/underside vertex
        # inside an untriangulated top face is not a sample of the upper surface.
        sample_xy=(end[None,:]+sample_x[:,None]*side-.0001*direction)[:,:2]
        reference=np.interp(sample_x,lateral,declared_ground)
        sample_rows=[]
        for xy,expected_ground in zip(sample_xy,reference):
            deck_hits,top=top_triangle_hits(tris,xy)
            sample={'xy_m':xy.tolist(),'actual_deck_top_z_m':float(top),'declared_ground_interpolated_z_m':float(expected_ground),
                'actual_top_vs_declared_ground_minus_025_m':float(top-(expected_ground-.025)),
                'actual_deck_hits_z_m':sorted(set(float(z)for z in deck_hits)),'actual_background_hits_z_m':[]}
            sample_rows.append(sample);targets.append(sample)
        rows.append({'span_id':span['segment_id'],'source_glb_sha256':sha,'physical_terminal_lateral_bounds_m':[float(x.min()),float(x.max())],
            'declared_contacts_outside_physical_section':int(((lateral<x.min()-3e-5)|(lateral>x.max()+3e-5)).sum()),
            'declared_contact_lateral_bounds_m':[float(lateral.min()),float(lateral.max())],
            'sampling':'actual physical left and right edges plus three evenly spaced interior locations; edge inset 0.05mm, longitudinal inset 0.1mm for serialized float32 boundary precision','samples':sample_rows})
    if not targets:
        result={'schema':'citycode.actual_release_landing_geometry_check.v3','ue_executed':False,
            'status':'not_applicable','landing_count':0,'sample_count':0,
            'reason':'Proposal declares no ground-transition spans','check_code_sha256':digest_file(__file__),
            'source_manifest_sha256':before[manifest],'surface_proposal_sha256':before[proposal],
            'building_manifest_sha256':before[baseline]}
        write(output,result)
        return result
    xy=np.array([s['xy_m']for s in targets]);part_proofs=[]
    for record in read(baseline)['meshes']:
        if record['kind']!='background':continue
        anchor=np.array(record['anchor_m'])
        for part in record['import_parts']:
            path=baseline.parent/part['file'];doc,binary,sha=read_glb(path)
            if sha!=part['sha256']:raise ValueError('Frozen background part changed')
            part_proofs.append({'file':part['file'],'sha256':sha})
            for mesh in doc['meshes']:
                for primitive in mesh['primitives']:
                    local,_=accessor(doc,binary,primitive['attributes']['POSITION']);world=local.astype(float)+anchor
                    low,high=world[:,:2].min(0),world[:,:2].max(0)
                    chosen=np.flatnonzero(np.all(xy>=low-1e-6,axis=1)&np.all(xy<=high+1e-6,axis=1))
                    if not len(chosen):continue
                    indices,_=accessor(doc,binary,primitive['indices']);triangles=world[indices.reshape(-1,3)]
                    lo,hi=triangles[:,:,:2].min(1),triangles[:,:,:2].max(1)
                    for i in chosen:
                        mask=np.all(xy[i]>=lo-1e-6,axis=1)&np.all(xy[i]<=hi+1e-6,axis=1)
                        hits=height_hits(triangles[mask],xy[i]);targets[i]['actual_background_hits_z_m'].extend(float(h)for h in hits)
            print('Checked actual background part '+str(len(part_proofs)),flush=True)
            del doc,binary
    maximum_interpolation=0.;maximum_ground=0.;maximum_highest_ground=0.;missing=0
    for sample in targets:
        hits=np.sort(sample['actual_background_hits_z_m']);hits=hits[np.r_[True,np.diff(hits)>.001]]if len(hits)else hits
        sample['actual_background_hits_z_m']=hits.tolist()
        if not len(hits):missing+=1;continue
        # Record every actual layer. The intended terrain layer is the one closest
        # to the producer's measured low-ground contact; no arbitrary flat fallback.
        ground=float(hits[np.argmin(abs(hits-sample['declared_ground_interpolated_z_m']))])
        sample['actual_background_ground_z_m']=ground
        sample['actual_top_minus_actual_ground_m']=sample['actual_deck_top_z_m']-ground
        sample['actual_ground_vs_declared_ground_m']=ground-sample['declared_ground_interpolated_z_m']
        sample['actual_top_minus_highest_background_hit_m']=sample['actual_deck_top_z_m']-float(hits.max())
        maximum_interpolation=max(maximum_interpolation,abs(sample['actual_top_vs_declared_ground_minus_025_m']))
        maximum_ground=max(maximum_ground,abs(sample['actual_top_minus_actual_ground_m']+.025))
        maximum_highest_ground=max(maximum_highest_ground,abs(sample['actual_top_minus_highest_background_hit_m']+.025))
    result={'schema':'citycode.actual_release_landing_geometry_check.v3','ue_executed':False,
        'status':'passed'if maximum_interpolation<3e-5 and max(maximum_ground,maximum_highest_ground)<.05 and not missing else'requires_review',
        'method':'Physical serialized source GLB terminal section edges plus three interior points; actual frozen background part triangle vertical intersections',
        'ground_layer_policy':'nearest actual triangle hit to declared measured low-ground contact; all hits retained; no fallback',
        'source_manifest_sha256':digest_file(manifest),'surface_proposal_sha256':digest_file(proposal),'building_manifest_sha256':digest_file(baseline),
        'check_code_sha256':digest_file(__file__),'background_parts':part_proofs,'landing_count':len(rows),'sample_count':len(targets),
        'missing_actual_ground_samples':missing,'maximum_actual_top_vs_declared_interpolation_error_m':maximum_interpolation,
        'maximum_actual_embed_error_relative_minus_025_m':maximum_ground,
        'maximum_highest_background_embed_error_relative_minus_025_m':maximum_highest_ground,
        'actual_ground_embed_tolerance_m':.05,'records':rows}
    if any(digest_file(path)!=sha for path,sha in before.items()):
        raise ValueError('Landing audit inputs changed during the check')
    write(output,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spans", "--source-span-manifest", dest="source_spans", type=Path, required=True)
    parser.add_argument("--proposal", "--surface-proposal", dest="proposal", type=Path, required=True)
    parser.add_argument("--buildings", "--building-manifest", dest="buildings", type=Path,
        default=Path(__file__).resolve().parents[2]/"build/ue_citycode/assembly_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_spans, args.proposal, args.buildings, args.output)
    print(json.dumps({key:value for key,value in result.items() if key not in ("records", "background_parts")}, indent=2))


if __name__ == "__main__":
    main()
