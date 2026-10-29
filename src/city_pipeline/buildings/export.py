"""Export frozen prediction geometry as LoD1 and body-plus-roof-layer GLBs."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import trimesh
from shapely.geometry import Polygon, shape, mapping
from . import selection as base
from .height_refinement import install_guard


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trial-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True);args=ap.parse_args()
    source=args.trial_root;root=args.output_root
    if root.exists():raise FileExistsError(root)
    paths=[source/'manifest.json',source/'global_building_footprints.geojson']
    reads=install_guard(paths)
    prior=base.read_json(paths[0]);assert prior['status']=='complete'
    features=copy.deepcopy(base.read_json(paths[1])['features'])
    rows=[f['properties'] for f in features];records=[]
    for f in features:
        p=f['properties'];g=shape(f['geometry'])
        c=base.Candidate(p['global_building_index'],p,{},g,g,0,{})
        records.append({'id':p['predicted_global_building_id'],'index':p['global_building_index'],'candidate':c})
    coarse=base.export_glb(root/'lod1',records,rows)
    # Final artifact paths use the release manifest as their portable base.
    coarse.update({
        'full_city_coarse_glb':'lod1/full_city_coarse.glb',
        'full_city_coarse_glb_ue_z_up':'lod1/full_city_coarse_ue_z_up.glb',
        'canonical_building_glb_root':'lod1/canonical_building_glb',
        'canonical_building_glb_ue_z_up_root':'lod1/canonical_building_glb_ue_z_up',
    })
    world=trimesh.Scene();view=trimesh.Scene();piece_count=roof_count=0
    for f in features:
        p=f['properties'];gid=p['predicted_global_building_id'];g=shape(f['geometry'])
        p['coarse_glb']=f'lod1/canonical_building_glb/{gid}.glb'
        p['coarse_glb_ue_z_up']=f'lod1/canonical_building_glb_ue_z_up/{gid}.glb'
        body_top=p.get('body_top_z_m',p['top_z_m'])
        pieces=[('body',g,p['base_z_m'],body_top,base.deterministic_color(gid))]
        covered=Polygon();partition=[]
        for i,layer in enumerate(sorted(p.get('roof_layers',[]),key=lambda l:-l['top_z_m'])):
            region=shape(layer['geometry']).intersection(g).difference(covered)
            # Keep every positive-area roof piece; touching lines have no cap.
            if region.is_empty or region.area<=0.:continue
            region=base.polygonal(region);covered=covered.union(region)
            pieces.append((f'roof_{i:02d}',region,body_top,layer['top_z_m'],[241,157,69,255]))
            partition.append({**layer,'geometry':mapping(region),'base_z_m':body_top})
        p['roof_layers']=partition
        local_world=trimesh.Scene();local_view=trimesh.Scene();names=[]
        for role,geom,z0,z1,color in pieces:
            name=f'{gid}__{role}';names.append(name)
            mesh=base.extrude_geometry(geom,z0,z1,color,{'predicted_global_building_id':gid,'role':role})
            transformed=mesh.copy();transformed.apply_transform(base.UE_Z_UP_TO_GLTF_Y_UP)
            for scene in [world,local_world]:scene.add_geometry(mesh,node_name=name,geom_name=name)
            for scene in [view,local_view]:scene.add_geometry(transformed,node_name=name,geom_name=name)
            piece_count+=1;roof_count+=role!='body'
        wp=root/'canonical_building_glb_ue_z_up'/f'{gid}.glb';vp=root/'canonical_building_glb'/f'{gid}.glb'
        wp.parent.mkdir(parents=True,exist_ok=True);vp.parent.mkdir(parents=True,exist_ok=True)
        wp.write_bytes(local_world.export(file_type='glb'));vp.write_bytes(local_view.export(file_type='glb'))
        p['canonical_glb']=vp.relative_to(root).as_posix();p['canonical_glb_ue_z_up']=wp.relative_to(root).as_posix()
        p['structure_mesh_nodes']=names
    (root/'full_city_layered_ue_z_up.glb').write_bytes(world.export(file_type='glb'))
    (root/'full_city_layered.glb').write_bytes(view.export(file_type='glb'))
    base.write_json(root/'global_building_footprints.geojson',{'type':'FeatureCollection','features':features})
    base.write_jsonl(root/'global_buildings.jsonl',rows)
    base.write_json(root/'manifest.json',{'status':'complete','schema':'air2vectorcity.spatial_support_release.v4',
        'source_trial':os.path.relpath(source.resolve(),root.resolve()),
        'source_trial_path_resolution':'manifest_directory',
        'mesh_path_resolution':'manifest_directory',
        'runtime_read_audit_path_resolution':'manifest_directory',
        'source_sha256_path_resolution':'manifest_directory',
        'source_sha256':{os.path.relpath(p.resolve(),root.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'prediction_contract':prior['prediction_contract'],'parameters':prior['parameters'],
        'building_entity_count':len(features),'mesh_piece_count':piece_count,'roof_piece_count':roof_count,
        'layer_partition':'highest roof wins overlapping XY; body shared beneath upper pieces',
        'runtime_read_audit':sorted(os.path.relpath(p,root.resolve()) for p in reads),
        'outputs_path_resolution':'manifest_directory',
        'outputs':{'layered_glb':'full_city_layered.glb',
            'layered_ue_z_up_glb':'full_city_layered_ue_z_up.glb','coarse':coarse}})
    print('Exported',len(features),'buildings,',roof_count,'roof pieces')


if __name__=='__main__':main()
