"""Prediction-only roof layers and observed boundary support."""
import argparse
from collections import defaultdict
import copy
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

from . import selection as base
from .height_refinement import install_guard, weighted_quantile
from .consensus import load_observations

OBS = base.SOURCE_RUN_ROOT / 'global_fused/building_observation_footprints.geojson'
INPUT = Path("UNCONFIGURED_INPUT")
TRIALS = {'C1_elevated_roof': {'support_buffer_m': 1.0, 'elevated': True, 'roof_min_patches': 1, 'roof_min_rise': 8, 'roof_min_area_fraction': 0.1}}


def point_support_polygon(points, bounds):
    x0, y0, x1, y1 = bounds
    points = points[(points[:, 0] >= x0-2) & (points[:, 0] <= x1+2) &
                    (points[:, 1] >= y0-2) & (points[:, 1] <= y1+2)]
    if len(points) < 200:
        return None
    resolution = .5
    origin = np.asarray([x0-3, y0-3])
    dims = np.ceil((np.asarray([x1+3, y1+3])-origin)/resolution).astype(int)+1
    cells = np.floor((points[:, :2]-origin)/resolution).astype(int)
    mask = np.zeros((dims[1], dims[0]), dtype=np.uint8)
    mask[cells[:, 1], cells[:, 0]] = 1
    # Preserve interior occlusions; this tests external support, not hole removal.
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, structure=np.ones((3,3)))).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3 and cv2.contourArea(contour) >= 4:
            xy = (contour[:, 0, :] + .5) * resolution + origin
            poly = Polygon(xy)
            if poly.area > 0:
                polygons.append(base.polygonal(poly))
    return unary_union(polygons).buffer(resolution*.71) if polygons else None


def prepare(cache):
    if cache.exists():
        raise FileExistsError(cache)
    reads = install_guard([OBS], expand_observation_points=True)
    fs = base.read_json(OBS)['features']; grouped = defaultdict(list)
    for f in fs:
        grouped[f['properties']['building_point_cloud_npz_path']].append(f)
    result = {}
    for path, features in sorted(grouped.items()):
        with np.load(base.PROJECT_ROOT / path) as a:
            xyz, ids, provenance = a['xyz_m'], a['instance_id'], a['provenance']
        for f in features:
            o = f['properties']; g = shape(f['geometry'])
            points = xyz[(ids == o['rgb_instance_id']) & (provenance == 1)]
            points = points[np.isfinite(points).all(axis=1)]
            direct = point_support_polygon(points, g.bounds)
            elevated = point_support_polygon(points[points[:, 2] >= o['base_z_m'] + 3], g.bounds)
            result[o['observation_id']] = {'point_count': len(points),
                'direct': mapping(direct) if direct is not None else None,
                'elevated': mapping(elevated) if elevated is not None else None}
        if len(result) % 200 < len(features):
            print('Support evidence',len(result),'/',len(fs),flush=True)
    base.write_json(cache, {'source_sha256': hashlib.sha256(OBS.read_bytes()).hexdigest(),
                           'observations': result, 'runtime_read_audit': sorted(reads)})


def boundary_refine(f, evidence, config):
    g = shape(f['geometry']); p = f['properties']
    ids = p.get('consensus_observation_ids', [p['canonical_observation_id']])
    field = 'elevated' if config.get('elevated') else 'direct'
    supports = [shape(evidence[i][field]) for i in ids if evidence[i][field] is not None]
    if not supports:
        return g, 'no_support_fallback'
    allowed = unary_union(supports).buffer(config['support_buffer_m'])
    result = g.intersection(allowed)
    if result.is_empty or result.area < .8*g.area:
        return g, 'insufficient_support_fallback'
    return base.polygonal(result).simplify(.15, preserve_topology=True), 'observed_boundary_clip'


def roof_evidence(features, candidates, config):
    gs = [shape(f['geometry']) for f in features]; tree = STRtree(gs)
    groups = [[] for g in gs]
    canonical_owner = {f['properties']['canonical_observation_id']: i for i,f in enumerate(features)}
    for c in candidates:
        o = c.observation_properties
        if o['stable_layer_count'] < 1 or o['boundary_status'] == 'boundary_clipped_exclude':
            continue
        if config.get('roof_require_core') and not o['reliable_core_contains_centroid']:
            continue
        options = []
        for i in tree.query(c.geometry,predicate='intersects'):
            i=int(i); g=gs[i]; p=features[i]['properties']
            if c.observation_id in canonical_owner and canonical_owner[c.observation_id] != i:
                continue
            a=g.intersection(c.geometry).area
            if (a/c.geometry.area >= config.get('roof_containment', .85)
                    and config.get('roof_min_area_fraction', .03) <= c.geometry.area/g.area <= .8
                    and g.covers(c.geometry.centroid) and o['top_z_m'] >= p['top_z_m']+config.get('roof_min_rise',4)):
                options.append((a/c.geometry.area, -g.area, i))
        if options:
            groups[max(options)[2]].append(c)
    return groups


def choose_roof_layers(f, children, min_patches):
    layers=[]; g=shape(f['geometry'])
    for anchor in sorted(children,key=lambda c:-c.observation_properties['top_z_m']):
        matches={}
        for c in children:
            dz=abs(c.observation_properties['top_z_m']-anchor.observation_properties['top_z_m'])
            a=c.geometry.intersection(anchor.geometry).area
            if dz<=6 and a/max(min(c.geometry.area,anchor.geometry.area),1e-9)>=.5:
                patch=c.observation_properties['patch_id']
                if patch not in matches or c.score>matches[patch].score:
                    matches[patch]=c
        if len(matches)<min_patches:
            continue
        members=list(matches.values()); z=weighted_quantile(
            [c.observation_properties['top_z_m'] for c in members], [max(c.score,.1) for c in members], .5)
        geom=unary_union([c.geometry for c in members]).intersection(g)
        if geom.area<20 or any(geom.intersection(shape(l['geometry'])).area>geom.area*.5 for l in layers):
            continue
        layers.append({'geometry':mapping(base.polygonal(geom)), 'top_z_m':z,
            'base_z_m':f['properties']['top_z_m'],
            'observation_ids':[c.observation_id for c in members], 'independent_patches':len(matches)})
    return layers


def split_entities(features, candidates, config):
    gs=[c.geometry for c in candidates];tree=STRtree(gs)
    owned={f['properties']['canonical_observation_id']:f['properties']['predicted_global_building_id'] for f in features}
    output=[];decisions=[]
    for f in features:
        g=shape(f['geometry']);gid=f['properties']['predicted_global_building_id'];pool=[]
        for j in tree.query(g,predicate='intersects'):
            c=candidates[int(j)];o=c.observation_properties
            if c.observation_id in owned and owned[c.observation_id]!=gid:continue
            if o['stable_layer_count']<1 or o['boundary_status']=='boundary_clipped_exclude':continue
            if not .1<=c.geometry.area/g.area<=.8:continue
            a=g.intersection(c.geometry).area
            if a/c.geometry.area>=.9 and g.covers(c.geometry.centroid):pool.append(c)
        hypotheses=[]
        for seed in sorted(pool,key=lambda c:(-c.score,c.observation_id)):
            members=[]
            for c in pool:
                a=seed.geometry.intersection(c.geometry).area
                if a/(seed.geometry.area+c.geometry.area-a)>=.5:members.append(c)
            patches={c.observation_properties['patch_id'] for c in members}
            if len(patches)>=config['split_min_patches']:hypotheses.append((seed,members,patches))
        chosen=[]
        for h in sorted(hypotheses,key=lambda h:(-len(h[2]),-h[0].score,-h[0].geometry.area)):
            if all(h[0].geometry.distance(old[0].geometry)>=config['split_gap_m'] for old in chosen):chosen.append(h)
        if len(chosen)<2:
            output.append(f);continue
        common=set.intersection(*(h[2] for h in chosen))
        covered=unary_union([h[0].geometry for h in chosen]).intersection(g).area/g.area
        if not common or covered<config['split_coverage']:
            output.append(f);continue
        children=[]
        for k,(c,members,patches) in enumerate(sorted(chosen,key=lambda h:(h[0].geometry.centroid.x,h[0].geometry.centroid.y)),1):
            geom=base.polygonal(c.geometry.intersection(g));o=c.observation_properties
            z=weighted_quantile([x.observation_properties['top_z_m'] for x in members],[max(x.score,.1) for x in members],.5)
            child=copy.deepcopy(f);p=child['properties'];child['geometry']=mapping(geom)
            p.update({'predicted_global_building_id':f'{gid}_S{k:02d}','parent_prediction_id':gid,
                'canonical_observation_id':c.observation_id,'canonical_patch_id':o['patch_id'],
                'canonical_instance_id':o['rgb_instance_id'],'canonical_rgb_path':o['rgb_path'],
                'canonical_source_instance_ids_path':o['source_instance_ids_path'],
                'canonical_bev_instance_ids_path':o['bev_instance_ids_path'],
                'canonical_building_point_cloud_npz_path':o['building_point_cloud_npz_path'],
                'base_z_m':o['base_z_m'],'top_z_m':z,'height_m':z-o['base_z_m'],
                'height_method':'split_child_observation_consensus','footprint_area_m2':geom.area,
                'centroid_x_m':geom.centroid.x,'centroid_y_m':geom.centroid.y,
                'consensus_observation_ids':[x.observation_id for x in members],
                'canonical_glb':'','canonical_glb_ue_z_up':''})
            children.append(child)
        output.extend(children);decisions.append({'parent_id':gid,'children':[f['properties']['predicted_global_building_id'] for f in children],
                                                'coverage':covered,'cooccurring_patches':sorted(common)})
    for i,f in enumerate(output,1):f['properties']['global_building_index']=i
    return output,decisions


def run(args):
    paths=[OBS,INPUT/'global_building_footprints.geojson',args.cache]
    reads=install_guard(paths)
    features=base.read_json(paths[1])['features']; cache=base.read_json(args.cache)
    assert cache['source_sha256']==hashlib.sha256(OBS.read_bytes()).hexdigest()
    evidence=cache['observations']; candidates=load_observations(base.SOURCE_RUN_ROOT,base.Parameters())
    for name in args.trials or TRIALS:
        config=TRIALS[name]; out=args.output_root/name
        if out.exists():
            raise FileExistsError(out)
        roof_groups=roof_evidence(features,candidates,config) if 'roof_min_patches' in config else [[] for _ in features]
        fs=copy.deepcopy(features); changed_geometry=changed_height=0; layer_count=0
        for f,original,children in zip(fs,features,roof_groups):
            p=f['properties'];g=shape(f['geometry'])
            if 'support_buffer_m' in config:
                new,method=boundary_refine(f,evidence,config)
                changed_geometry+=not new.equals(g);g=new;f['geometry']=mapping(g)
                p['boundary_refinement_method']=method
            if 'roof_min_patches' in config:
                layers=choose_roof_layers(f,children,config['roof_min_patches'])
                p['roof_layers']=layers;layer_count+=len(layers)
                if layers:
                    p['body_top_z_m']=p['top_z_m'];p['top_z_m']=max(p['top_z_m'],max(l['top_z_m'] for l in layers))
                    changed_height+=1;p['height_method']='body_plus_contained_predicted_roof_layers'
            p['height_m']=p['top_z_m']-p['base_z_m'];p['footprint_area_m2']=g.area
            p['centroid_x_m'],p['centroid_y_m']=g.centroid.x,g.centroid.y
            p['canonical_glb']=p['canonical_glb_ue_z_up']=''
        split_decisions=[]
        if 'split_min_patches' in config:fs,split_decisions=split_entities(fs,candidates,config)
        base.write_json(out/'global_building_footprints.geojson',{'type':'FeatureCollection','features':fs})
        base.write_jsonl(out/'global_buildings.jsonl',[f['properties'] for f in fs])
        base.write_json(out/'manifest.json',{'status':'complete','schema':'air2vectorcity.spatial_support_ablation.v4',
            'parameters':config,'prediction_contract':{'selection_uses_ground_truth':False,'entity_ids':'parent-child lineage' if split_decisions else 'unchanged from V3'},
            'changed_geometry':int(changed_geometry),'changed_height':changed_height,'roof_layer_count':layer_count,
            'split_decisions':split_decisions,'selected_entity_count':len(fs),
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'runtime_read_audit':sorted(reads)})
        print(name,changed_geometry,changed_height,layer_count,'split',len(split_decisions),'entities',len(fs),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--prepare',action='store_true');ap.add_argument('--output-root',type=Path)
    ap.add_argument('--trials',nargs='+',choices=list(TRIALS));args=ap.parse_args()
    if args.prepare: prepare(args.cache)
    elif args.output_root: run(args)
    else: ap.error('--output-root is required for trials')


if __name__=='__main__':
    main()
