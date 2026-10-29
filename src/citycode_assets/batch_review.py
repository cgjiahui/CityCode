"""Local source/image/model evidence. Reports and templates never approve assets."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import io
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .appearance import write_json, sha256_file
from .batch import read, proof, check_plan, verify


def _white(path, size):
    if not path.is_file():return Image.new('RGB',size,'#eeeeee')
    with Image.open(path) as opened:
        rgba=opened.convert('RGBA');image=Image.new('RGBA',rgba.size,'white');image.alpha_composite(rgba)
    image=image.convert('RGB');image.thumbnail(size)
    canvas=Image.new('RGB',size,'white');canvas.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2))
    return canvas


def review_camera(prepared, building):
    """Use the recorded camera or reconstruct the released proxy-render recipe."""
    provenance=prepared['provenance'];renderer=provenance['renderer'];derived=None
    if renderer.get('name')=='lod1_v4_sculpted_rough':
        from shapely.geometry import shape
        from .appearance import _shape_yaw
        yaw=float(provenance['shape_yaw_world_deg']);elevation=float(renderer['elevation_deg'])
        actual_yaw=_shape_yaw(shape(building['footprint_world']))
        difference=(yaw-actual_yaw+90)%180-90
        if (not np.isfinite([yaw,elevation,difference]).all() or abs(difference)>1e-4
                or abs(elevation-34)>1e-8):
            raise ValueError('Review camera metadata differs from the released structure/render recipe')
        az=math.radians(yaw+45);el=math.radians(elevation)
        depth=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
        right=np.cross([0,0,1],depth);right/=np.linalg.norm(right)
        derived=np.stack([right,np.cross(depth,right),depth],axis=1)
    recorded=renderer.get('camera_columns_world')
    if recorded is None:
        if derived is None:raise ValueError('Missing recorded review camera and supported structure-render recipe')
        return derived
    camera=np.asarray(recorded,dtype=float)
    if (camera.shape!=(3,3) or not np.isfinite(camera).all()
            or not np.allclose(camera.T@camera,np.eye(3),atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.det(camera),1,atol=1e-6)):
        raise ValueError('Review camera must contain finite orthonormal world axes')
    if derived is not None and not np.allclose(camera,derived,atol=1e-6,rtol=0):
        raise ValueError('Recorded review camera contradicts the structure-render recipe')
    return camera


def _report(root, kind, records, columns):
    folder=root/'reviews'/kind;folder.mkdir(parents=True,exist_ok=True)
    tiles=[]
    for record in records:
        aid=record['asset_id'];card=Image.new('RGB',(240*len(columns),276),'white');draw=ImageDraw.Draw(card)
        for i,(title,key) in enumerate(columns):
            card.paste(_white(root/record.get(key,'__missing__'),(232,232)),(i*240+4,36))
            draw.text((i*240+6,8),title,fill='black')
        card.save(folder/(aid+'.jpg'),quality=88)
        tiles.append('<figure><figcaption>'+html.escape(aid+' · '+record['status'])+'</figcaption><a href="'+
            html.escape(aid+'.jpg',quote=True)+'"><img loading="lazy" src="'+html.escape(aid+'.jpg',quote=True)+'"></a></figure>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>建筑输入与模型检查</title><style>body{font:15px/1.7 system-ui;max-width:1500px;margin:24px auto;padding:20px;color:#26313d;background:#f6f7f8}figure{background:white;padding:12px;margin:16px 0}img{max-width:100%}figcaption{font-weight:600}</style>
<h1>建筑输入与模型检查</h1><p>本页展示当前文件，生成预览不代表已通过审核。灰色格表示该步骤尚无结果；审核决定需单独保存。</p>'''+''.join(tiles)+'</html>'
    (folder/'index.html').write_text(document)
    write_json(folder/'index.json',{'schema':'citycode.batch_review_evidence.v1','kind':kind,'records':records,
        'inclusion_approved':False})
    # Keep a blank editable template separate from actual signed decisions.
    write_json(root/'reviews'/(kind+'.template.json'),{'schema':'citycode.batch_review.v1',
        'records':[{k:v for k,v in row.items() if k in {'asset_id','image_sha256','placement_identity'}}|
                   {'decision':None,'reason':''} for row in records]})
    return {'status':'evidence_ready_decisions_pending','records':len(records),
        'report':str(folder/'index.html'),'template':str(root/'reviews'/(kind+'.template.json'))}


def review_appearance(config):
    root=Path(config['output']);plan=check_plan(config);records=[]
    for row in plan['records']:
        folder=root/row['appearance_directory'];image=folder/'generated_main_oblique.png'
        record={'asset_id':row['asset_id'],'status':'image_ready' if image.is_file() else 'awaiting_image',
            'observed':os.path.relpath(folder/'inputs/source_primary.png',root),
            'structure':os.path.relpath(folder/'inputs/lod1_main_oblique.png',root),
            'image':os.path.relpath(image,root),'roof':os.path.relpath(folder/'inputs/roof_evidence.png',root)}
        if image.is_file():record['image_sha256']=sha256_file(image)
        records.append(record)
    return _report(root,'appearance',records,[('Observed','observed'),('Structure','structure'),('Completed image','image'),('Observed roof','roof')])


def load_render_object(path,matrix):
    from .lod import _read_glb, _accessor
    from .registration import read_glb_scene_geometry
    scene=read_glb_scene_geometry(path);data,doc,binary=_read_glb(path)
    uvs=[];materials=[];textures={};samplers={};factors={}
    for instance in scene['instances']:
        primitive=doc['meshes'][instance['mesh_index']]['primitives'][instance['primitive_index']]
        indices=(_accessor(data,doc,binary,primitive['indices'])[0].ravel() if 'indices' in primitive
                 else np.arange(_accessor(data,doc,binary,primitive['attributes']['POSITION'])[0].shape[0]))
        uvs.append(_accessor(data,doc,binary,primitive['attributes']['TEXCOORD_0'])[0][np.unique(indices)].copy())
        material=primitive['material'];materials.extend([material]*instance['triangle_count'])
        pbr=doc['materials'][material]['pbrMetallicRoughness'];info=pbr['baseColorTexture']
        if info.get('extensions') or info.get('texCoord',0)!=0:
            raise ValueError('Diagnostic renderer needs untransformed TEXCOORD_0')
        texture=doc['textures'][info['index']];image=doc['images'][texture.get('source',texture.get('extensions',{}).get('EXT_texture_webp',{}).get('source'))]
        if image.get('uri'):raise ValueError('Diagnostic renderer needs embedded images')
        view=doc['bufferViews'][image['bufferView']];start=binary+view.get('byteOffset',0)
        with Image.open(io.BytesIO(data[start:start+view['byteLength']])) as opened:
            textures[material]=np.asarray(opened.convert('RGB'))
        samplers[material]=doc.get('samplers',[])[texture['sampler']] if 'sampler' in texture else {}
        factors[material]=np.asarray(pbr.get('baseColorFactor',[1,1,1,1])[:3])
    uv=np.concatenate(uvs)
    if len(uv)!=len(scene['vertices']):raise ValueError('Actual mesh UV instances differ from geometry')
    matrix=np.asarray(matrix);world=scene['vertices']@matrix[:3,:3].T+matrix[:3,3]
    return {'world':world,'faces':np.ascontiguousarray(scene['faces'],np.int64),'uv':uv,
        'materials':np.asarray(materials),'textures':textures,'samplers':samplers,'factors':factors}


def _wrap(values, mode):
    if mode==10497:return np.mod(values,1)
    if mode==33648:
        values=np.mod(values,2);return np.where(values<=1,values,2-values)
    if mode!=33071:raise ValueError('Unknown glTF texture wrap mode')
    return np.clip(values,0,1)


def render(objects,camera,side=480,framing_objects=None):
    from .roof_reconciliation import _surface,pd,pi,pf
    fun=_surface();camera=np.asarray(camera)
    coords=[np.ascontiguousarray(o['world']@camera,np.float64) for o in objects]
    framing=[np.ascontiguousarray(o['world']@camera,np.float64) for o in framing_objects] if framing_objects is not None else coords
    lo=np.min([v[:,:2].min(0) for v in framing],axis=0);hi=np.max([v[:,:2].max(0) for v in framing],axis=0)
    cell=float(max(hi-lo)/(.90*side));origin=(lo+hi)/2-side*cell/2
    output=np.full((side,side,3),255,np.uint8);best=np.full((side,side),-np.inf,np.float32)
    for obj,vertices in zip(objects,coords):
        faces=obj['faces'];depth=np.empty((side,side),np.float32);ids=np.empty((side,side),np.int64)
        u=np.empty((side,side),np.float32);v=np.empty((side,side),np.float32)
        fun(vertices.ctypes.data_as(pd),faces.ctypes.data_as(pi),len(faces),*origin,cell,side,side,
            depth.ctypes.data_as(pf),ids.ctypes.data_as(pi),u.ctypes.data_as(pf),v.ctypes.data_as(pf))
        valid=(ids>=0)&(depth>best);fi=ids[valid];tri_uv=obj['uv'][faces[fi]]
        sample=u[valid,None]*tri_uv[:,0]+v[valid,None]*tri_uv[:,1]+(1-u[valid,None]-v[valid,None])*tri_uv[:,2]
        colors=np.empty((len(fi),3),np.float64)
        for material,image in obj['textures'].items():
            hit=obj['materials'][fi]==material;sampler=obj['samplers'][material];wrapped=sample[hit].copy()
            for axis,key in enumerate(['wrapS','wrapT']):wrapped[:,axis]=_wrap(wrapped[:,axis],sampler.get(key,10497))
            xy=np.minimum((wrapped*[image.shape[1],image.shape[0]]).astype(int),[image.shape[1]-1,image.shape[0]-1])
            colors[hit]=image[xy[:,1],xy[:,0]]*obj['factors'][material]
        triangle=obj['world'][faces];normal=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
        light=.4*camera[:,2]-.5*camera[:,0]+[0,0,1];light/=np.linalg.norm(light)
        shade=.78+.22*np.clip(normal@light,0,1)
        output[valid]=np.clip(colors*shade[fi,None],0,255);best[valid]=depth[valid]
    return Image.fromarray(output).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _render_one(task):
    config,row,candidates,neighbours=task;root=Path(config['output']);aid=row['asset_id']
    candidate=candidates[aid];folder=root/'reviews/models'/aid;folder.mkdir(parents=True,exist_ok=True)
    prepared=read(root/row['appearance_directory']/'appearance_input.json')
    building=read(verify(root,row['building'],Path(config['_project_root'])))
    camera=review_camera(prepared,building);az=math.atan2(camera[1,2],camera[0,2])+math.pi
    elevation=math.asin(camera[2,2]);depth=np.array([math.cos(elevation)*math.cos(az),math.cos(elevation)*math.sin(az),math.sin(elevation)])
    right=np.cross([0,0,1],depth);right/=np.linalg.norm(right);back=np.stack([right,np.cross(depth,right),depth],axis=1)
    record={'asset_id':aid,'status':read(root/'validation'/aid/'RESULT.json')['status']}
    from .scene_completion import placement_identity
    record['placement_identity']=placement_identity(candidate)
    for label,short in [('0.25M','high'),('0.1M','low'),('1M','master')]:
        if label not in candidate['lods']:continue
        path=verify(root,candidate['lods'][label]);obj=load_render_object(path,candidate['world_from_local_4x4'])
        for name,view in [('front',camera),('back',back),('roof',np.eye(3))]:
            destination=folder/(short+'_'+name+'.png');render([obj],view).save(destination)
            record[short+'_'+name]=os.path.relpath(destination,root)
    if neighbours:
        objects=[load_render_object(root/candidates[other]['lods']['0.1M']['path'],candidates[other]['world_from_local_4x4'])
            for other in [aid]+neighbours]
        for name,view in [('front',camera),('back',back)]:
            destination=folder/('scene_'+name+'.png');render(objects,view,side=640).save(destination)
            record['scene_'+name]=os.path.relpath(destination,root)
    record.update(image=os.path.relpath(root/row['appearance_directory']/'generated_main_oblique.png',root),
        observed=os.path.relpath(root/row['appearance_directory']/'inputs/source_primary.png',root))
    write_json(folder/'PROOF.json',{'placement_identity':record['placement_identity'],
        'neighbours':neighbours,'views':{k:proof(root/v,root) for k,v in record.items() if k.startswith(('high_','low_','master_','scene_'))},
        'lighting':'actual base color with diagnostic directional lighting; no PBR reflections',
        'scene_approval':False})
    return record


def review_models(config,*,workers=4):
    from shapely.geometry import shape
    from .batch_gpu import frozen_selection
    from .scene_completion import placement_identity,scene_identity
    root=Path(config['output']);selected=frozen_selection(config);candidates={};geometry={};missing=[]
    for row in selected:
        folder=root/'validation'/row['asset_id'];result_path=folder/'RESULT.json'
        if not result_path.exists() or not read(result_path).get('candidate_record'):
            missing.append(row['asset_id']);continue
        result=read(result_path);candidate=read(verify(root,result['candidate_record']));fit=read(folder/'fit.json')
        if candidate['world_from_local_4x4']!=fit['world_from_asset_4x4']:raise ValueError('Candidate placement differs from footprint evidence')
        candidates[row['asset_id']]=candidate;geometry[row['asset_id']]=shape(fit['actual_registered_footprint'])
    intersections=[];neighbours={aid:[] for aid in candidates}
    for aid,g in geometry.items():
        for other,t in geometry.items():
            if other<=aid or not g.intersects(t):continue
            area=g.intersection(t).area
            if area<=.1:continue
            intersections.append({'a':aid,'b':other,'intersection_m2':area,'fraction_a':area/g.area,
                'fraction_b':area/t.area,'inset1m_core_m2':g.buffer(-1).intersection(t.buffer(-1)).area})
            neighbours[aid].append(other);neighbours[other].append(aid)
    partials=read(root/'nonbuilding_sources.json')['source_records']
    context={'asset_instances':[{'id':aid,**row} for aid,row in candidates.items()],
        'surface_instances':[{'id':r['surface_instance_id'],'assembly_enabled':r['postprocess']['surface']['assembly_enabled'],**r}
            for r in partials if r['status']=='resolved_fragment']}
    folder=root/'reviews/models';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'scene_intersections.json',{'status':'measured_review_pending','scene_identity':scene_identity(context),
        'candidate_identities':{aid:placement_identity(r) for aid,r in candidates.items()},'missing_candidates':missing,
        'intersections':intersections,'inclusion_approved':False,
        'scope':'XY projections of actual registered high-LOD meshes; not a closed-solid collision simulation'})
    tasks=[(config,row,candidates,neighbours[row['asset_id']]) for row in selected if row['asset_id'] in candidates]
    results=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_render_one,t):t[1]['asset_id'] for t in tasks}
        for future in as_completed(futures):
            try:results.append(future.result())
            except Exception as exc:errors.append({'asset_id':futures[future],'error':str(exc)})
    columns=[('Observed','observed'),('Completed image','image'),('High front','high_front'),
         ('High back','high_back'),('Low front','low_front'),('Actual roof','high_roof')]
    if any('master_front' in record for record in results):
        columns += [('1M front','master_front'),('1M back','master_back'),('1M roof','master_roof')]
    report=_report(root,'models',sorted(results,key=lambda r:r['asset_id']),columns)
    return {**report,'missing':missing,'errors':errors,'scene_intersections':str(folder/'scene_intersections.json')}
