"""Source-evidence roof reconciliation of a fine neural GLB, on CPU.

Requires an offline C++11 compiler for the bundled upper-surface rasterizer.
Unknown roofs and all generated facade textures are retained. Explicit optional
geometry conformance is recorded separately from rigid registration. The new
candidate always needs actual dual-LOD and source semantic acceptance.
"""
from pathlib import Path
from functools import lru_cache
import copy,ctypes,io,json,subprocess,tempfile
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.spatial import cKDTree
import cv2
import shapely
from shapely.geometry import shape
from shapely import contains_xy,points,shortest_line,get_point,get_x,get_y,intersects_xy
from .lod import _read_glb,_accessor,inspect_glb
from .extrema_lod import _pack_glb
from .registration import read_glb_scene_geometry
from .appearance import sha256_file
pd=ctypes.POINTER(ctypes.c_double);pi=ctypes.POINTER(ctypes.c_int64);pf=ctypes.POINTER(ctypes.c_float)
@lru_cache(maxsize=1)
def _surface():
 temporary=tempfile.TemporaryDirectory(prefix='citycode-roof-raster-');so=Path(temporary.name)/'surface.so';cpp=Path(__file__).with_name('_roof_surface.cpp')
 subprocess.run(['g++','-std=c++11','-O2','-shared','-fPIC',str(cpp),'-o',str(so)],check=True,capture_output=True,timeout=60)
 lib=ctypes.CDLL(str(so));fun=lib.surface;fun.argtypes=[pd,pi,ctypes.c_int64,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,pf,pi,pf,pf];fun._owned_library=lib;fun._owned_directory=temporary
 return fun

def upper(world,faces,cell=.15,minimum_z=None):
 fun=_surface()
 lo=world[:,:2].min(0)-cell*2;hi=world[:,:2].max(0)+cell*2;w,h=np.ceil((hi-lo)/cell).astype(int);assert max(w,h)<4096
 v=np.ascontiguousarray(world,np.float64);f=np.ascontiguousarray(faces,np.int64);z=np.empty((h,w),np.float32);ids=np.empty((h,w),np.int64);u=np.empty((h,w),np.float32);q=np.empty((h,w),np.float32)
 fun(v.ctypes.data_as(pd),f.ctypes.data_as(pi),len(f),*lo,cell,w,h,z.ctypes.data_as(pf),ids.ctypes.data_as(pi),u.ctypes.data_as(pf),q.ctypes.data_as(pf))
 good=np.isfinite(z)
 if minimum_z is not None:good&=z>=minimum_z
 assert good.any();nearest=distance_transform_edt(~good,return_distances=False,return_indices=True);filled=z[tuple(nearest)]
 def query(xy):return map_coordinates(filled,[(xy[:,1]-lo[1])/cell-.5,(xy[:,0]-lo[0])/cell-.5],order=1,mode='nearest')
 return query
def conform_levels(world,faces,building):
 layers=building.get('roof_layers',[])
 if not layers:return world.copy(),{'applied':False,'reason':'No recorded source roof levels; neural image-inferred roof geometry retained.'}
 base=float(building['base_z_m']);top=float(building['top_z_m']);body=min(float(l['base_z_m']) for l in layers)
 # Facade ornaments can lie beyond the roof edge, where a vertical ray hits
 # only the ground floor. Such a sample is not a roof height: borrowing the
 # nearest supported upper roof prevents collapsing those facade columns.
 query=upper(world,faces,minimum_z=base+.3*(top-base));ceiling=query(world[:,:2]);target=np.full(len(world),body)
 for layer in layers:
  hit=contains_xy(shape(layer['geometry']),world[:,0],world[:,1]);target[hit]=np.maximum(target[hit],float(layer['top_z_m']))
 # Keep the base fixed and apply an explicit source-controlled column scale.
 # It changes geometry and is not described as a rigid registration repair.
 denominator=np.maximum(np.maximum(ceiling,world[:,2])-base,.01);ratio=(target-base)/denominator
 out=world.copy();out[:,2]=base+(world[:,2]-base)*ratio
 # The source declares the metric extrema; normalize only Z, recording it.
 before=[float(out[:,2].min()),float(out[:,2].max())];scale=(top-base)/(before[1]-before[0]);out[:,2]=base+(out[:,2]-before[0])*scale
 return out,{'applied':True,'method':'source_recorded_roof_levels_column_scale_base_anchored','source_layers':len(layers),'XY_changed':False,'triangles_added_or_deleted':0,'column_scale_quantiles':np.quantile(ratio,[0,.1,.5,.9,1]).tolist(),'final_global_Z_scale':scale,'max_vertex_Z_change_m':float(np.max(abs(out[:,2]-world[:,2]))),'pre_final_Z_range_m':before,'post_Z_range_m':[float(out[:,2].min()),float(out[:,2].max())],'inferred_facade_dimensions_changed':True,'GT_used':False}

def observed_gap_cap(world,faces,building,bev,rgba):
 """Cap deep/absent neural roof only where original source pixels support it.

 The source footprint, including its holes, limits the cap. Heights use recorded
 roof levels or nearby neural roofs; unknown heights are explicitly inferred.
 """
 h,w=rgba.shape[:2];v=world.copy();v[:,:2]=(np.linalg.inv(bev)@np.c_[world[:,:2],np.ones(len(world))].T).T[:,:2];v=np.ascontiguousarray(v);f=np.ascontiguousarray(faces,np.int64)
 z=np.empty((h,w),np.float32);ids=np.empty((h,w),np.int64);u=np.empty((h,w),np.float32);q=np.empty((h,w),np.float32);fun=_surface()
 fun(v.ctypes.data_as(pd),f.ctypes.data_as(pi),len(f),-.5,-.5,1.,w,h,z.ctypes.data_as(pf),ids.ctypes.data_as(pi),u.ctypes.data_as(pf),q.ctypes.data_as(pf))
 yy,xx=np.mgrid[:h,:w];xy=(bev@np.stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])).T[:,:2];geom=shape(building['footprint_world']);inside=contains_xy(geom,xy[:,0],xy[:,1]).reshape((h,w));observed=rgba[:,:,3]>=200;has_levels=bool(building.get('roof_layers'));supported=inside if has_levels else observed&inside;base=float(building['base_z_m']);height=float(building['height_m']);gap=supported&(z<base+.1*height)
 contours,hierarchy=cv2.findContours(gap.astype('u1'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);patches=[]
 if hierarchy is not None:
  for i,c in enumerate(contours):
   if hierarchy[0,i,3]>=0 or len(c)<3:continue
   def ring(c):return (bev@np.c_[c[:,0],np.ones(len(c))].T).T[:,:2]
   holes=[];child=int(hierarchy[0,i,2])
   while child>=0:
    if len(contours[child])>=3:holes.append(ring(contours[child]))
    child=int(hierarchy[0,child,0])
   g=shapely.make_valid(shapely.Polygon(ring(c),holes)).intersection(geom)
   if g.area>=.5:patches.append(g)
 if not patches:return np.empty((0,3)),np.empty((0,3),int),{'applied':False,'supported_deep_gap_pixels':int(gap.sum()),'added_faces':0}
 patch=shapely.union_all(patches);triangles=shapely.constrained_delaunay_triangles(patch);tri=[np.asarray(t.exterior.coords)[:3] for t in triangles.geoms if t.area>1e-8]
 if not tri:return np.empty((0,3)),np.empty((0,3),int),{'applied':False,'supported_deep_gap_pixels':int(gap.sum()),'added_faces':0}
 coords=np.asarray(tri).reshape(-1,2);valid=supported&np.isfinite(z)&(z>base+.3*height)
 if not valid.any():raise ValueError('Observed gap has no nearby neural roof height support')
 indices=np.flatnonzero(valid.ravel());tree=cKDTree(xy[indices]);near=tree.query(coords,k=min(8,len(indices)))[1];near=np.asarray(near).reshape((len(coords),-1));heights=np.median(z.ravel()[indices[near]],axis=1)
 layers=building.get('roof_layers',[])
 if layers:
  heights[:]=min(float(layer['base_z_m']) for layer in layers)
  for layer in layers:
   hit=intersects_xy(shape(layer['geometry']),coords[:,0],coords[:,1]);heights[hit]=np.maximum(heights[hit],float(layer['top_z_m']))
 out=np.c_[coords,heights];f=np.arange(len(out)).reshape(-1,3)
 return out,f,{'applied':True,'supported_deep_gap_pixels':int(gap.sum()),'added_faces':len(f),'cap_area_m2':float(patch.area),'source_footprint_holes_preserved':True,'minimum_component_area_m2':.5,'height_basis':'recorded_source_roof_levels' if layers else 'nearest_neural_roof_heights_inferred_not_measured','geometry_support':'current_source_footprint_and_recorded_roof_levels' if layers else 'current_source_footprint_and_observed_alpha','unknown_appearance_pixels_in_geometry_support':int((gap&~observed).sum()),'unknown_appearance_is_inferred_not_observed':True}

def reconcile_roof(source, destination, *, building, world_from_asset, observation, evidence_image,
                   constrain_recorded_levels=False, boundary_conformance=False,
                   repair_observed_gaps=False):
 """Write a new source-supported candidate; preserve the input fine mesh.

 Boundary conformance projects outside vertices to the current source boundary.
 Recorded-level conformance scales vertex Z by column; this can alter inferred
 facade dimensions. Both are explicit geometry operations, not bounded fitting.
 """
 src=Path(source).resolve();destination=Path(destination).resolve()
 if destination.exists() or destination==src:raise ValueError('A new distinct output path is required')
 destination.parent.mkdir(parents=True,exist_ok=True)
 M=np.asarray(world_from_asset,dtype=float)
 if M.shape!=(4,4) or not np.isfinite(M).all() or not np.allclose(M[3],[0,0,0,1]) or abs(np.linalg.det(M[:3,:3]))<1e-12:raise ValueError('Finite invertible affine placement required')
 if sha256_file(evidence_image)!=observation['files']['roof_evidence.png']:raise ValueError('Source image differs from recorded observation')
 bev=np.asarray(observation['bev_crop_pixel_center_to_world_xy_3x3']);rgba=np.asarray(Image.open(evidence_image).convert('RGBA'));h,w=rgba.shape[:2]
 texture=Image.fromarray(rgba).resize((2048,2048),Image.Resampling.LANCZOS);encoded=io.BytesIO();texture.save(encoded,format='PNG');png=encoded.getvalue()
 data,doc,binary=_read_glb(src);scene=read_glb_scene_geometry(src);assert len(scene['instances'])==len(doc['meshes'])==1 and len(doc['meshes'][0]['primitives'])==1
 inst=np.asarray(scene['instances'][0]['asset_from_node_4x4']);combined=M@inst;prim=doc['meshes'][0]['primitives'][0];position,acc=_accessor(data,doc,binary,prim['attributes']['POSITION']);normal,_=_accessor(data,doc,binary,prim['attributes']['NORMAL']);indices,_=_accessor(data,doc,binary,prim['indices']);faces=np.asarray(indices).reshape(-1,3).astype(np.int64);world=np.asarray(position)@combined[:3,:3].T+combined[:3,3];world,level=conform_levels(world,faces,building) if constrain_recorded_levels else (world.copy(), {"applied":False})
 boundary={'applied':False}
 if boundary_conformance:
  geom=shape(building['footprint_world']);outside=~intersects_xy(geom,world[:,0],world[:,1]);nearest=get_point(shortest_line(points(world[outside,:2]),geom),-1);xy=np.c_[get_x(nearest),get_y(nearest)];distance=np.linalg.norm(world[outside,:2]-xy,axis=1);world[outside,:2]=xy;boundary={'applied':True,'method':'outside_neural_vertices_project_to_nearest_current_source_boundary','vertices_changed':int(outside.sum()),'world_XY_displacement_quantiles_m':np.quantile(distance,[0,.1,.5,.9,1]).tolist(),'Z_unchanged':True,'unknown_source_regions_not_added':True}
 if level['applied'] or boundary['applied']:
  inv=np.linalg.inv(combined);position[:]=(world@inv[:3,:3].T+inv[:3,3]).astype(np.float32);acc['min']=position.min(0).tolist();acc['max']=position.max(0).tolist();tri=np.asarray(position,dtype=float)[faces];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);sums=np.zeros_like(position,dtype=np.float64)
  for c in range(3):np.add.at(sums,faces[:,c],fn)
  norm=np.linalg.norm(sums,axis=1);valid=norm>1e-12;normal[valid]=(sums[valid]/norm[valid,None]).astype(np.float32)
 tri=world[faces];centers=tri.mean(1);fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);cos=np.abs(fn[:,2])/np.maximum(np.linalg.norm(fn,axis=1),1e-12);query=upper(world,faces);visible=np.abs(centers[:,2]-query(centers[:,:2]))<.6
 pixels=(np.linalg.inv(bev)@np.c_[world[:,:2],np.ones(len(world))].T).T[:,:2];cp=(np.linalg.inv(bev)@np.c_[centers[:,:2],np.ones(len(centers))].T).T[:,:2]
 def supported(p):
  x,y=np.rint(p).astype(int).T;valid=(x>=0)&(x<w)&(y>=0)&(y<h);alpha=np.zeros(len(p),bool);alpha[valid]=rgba[y[valid],x[valid],3]>=200;return alpha
 vertex_ok=supported(pixels);roof=(cos>.55)&visible&supported(cp)&np.all(vertex_ok[faces],axis=1)
 payload=bytearray(data[binary:]);doc=copy.deepcopy(doc)
 def blob(value,target=None):
  payload.extend(b'\0'*(-len(payload)%4));view={'buffer':0,'byteOffset':len(payload),'byteLength':len(value)}
  if target:view['target']=target
  index=len(doc['bufferViews']);doc['bufferViews'].append(view);payload.extend(value);return index
 def accessor(array,kind,component,target):
  record={'bufferView':blob(array.tobytes(),target),'componentType':component,'count':len(array),'type':kind}
  if kind=='VEC3':record.update(min=array.min(axis=0).tolist(),max=array.max(axis=0).tolist())
  index=len(doc['accessors']);doc['accessors'].append(record);return index
 generated=copy.deepcopy(prim);observed=copy.deepcopy(prim);generated['indices']=accessor(faces[~roof].astype('<u4').reshape(-1),'SCALAR',5125,34963)
 observed['indices']=accessor(faces[roof].astype('<u4').reshape(-1),'SCALAR',5125,34963)
 # glTF images are sampled with the same V orientation as the source PNG.
 uv=((pixels+.5)/[w,h]).astype('<f4');observed['attributes']['TEXCOORD_0']=accessor(uv,'VEC2',5126,34962)
 ii=len(doc['images']);doc['images'].append({'bufferView':blob(png),'mimeType':'image/png'});si=len(doc.setdefault('samplers',[]));doc['samplers'].append({'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071});ti=len(doc['textures']);doc['textures'].append({'sampler':si,'source':ii});mi=len(doc['materials']);doc['materials'].append({'name':'Observed roof evidence; inferred roughness','pbrMetallicRoughness':{'baseColorTexture':{'index':ti},'metallicFactor':0.1,'roughnessFactor':0.8},'alphaMode':'OPAQUE','doubleSided':True});observed['material']=mi
 doc['meshes'][0]['primitives']=[p for p,count in [(generated,(~roof).sum()),(observed,roof.sum())] if count]
 gap={'applied':False,'added_faces':0}
 if repair_observed_gaps:
  cap,cap_faces,gap=observed_gap_cap(world,faces,building,bev,rgba)
  if len(cap_faces):
   inv=np.linalg.inv(combined);positions=(cap@inv[:3,:3].T+inv[:3,3]).astype('<f4');normal_local=combined[:3,:3].T@np.array([0.,0.,1.]);normal_local/=np.linalg.norm(normal_local);normals=np.tile(normal_local,(len(cap),1)).astype('<f4');px=(np.linalg.inv(bev)@np.c_[cap[:,:2],np.ones(len(cap))].T).T[:,:2];uv_cap=((px+.5)/[w,h]).astype('<f4')
   known=rgba[:,:,3]>=200
   if not known.any():raise ValueError('No observed roof RGB to infer a cap material')
   nearest=distance_transform_edt(~known,return_distances=False,return_indices=True);filled=rgba.copy();filled[~known]=rgba[nearest[0][~known],nearest[1][~known]];filled[:,:,3]=255;cap_image=io.BytesIO();Image.fromarray(filled).resize((2048,2048),Image.Resampling.LANCZOS).save(cap_image,format='PNG');cap_ii=len(doc['images']);doc['images'].append({'bufferView':blob(cap_image.getvalue()),'mimeType':'image/png'});cap_ti=len(doc['textures']);doc['textures'].append({'sampler':si,'source':cap_ii});cap_mi=len(doc['materials']);cap_material=copy.deepcopy(doc['materials'][mi]);cap_material['name']='Source-constrained roof cap; unknown RGB inferred from nearby observed roof';cap_material['pbrMetallicRoughness']['baseColorTexture']['index']=cap_ti;doc['materials'].append(cap_material)
   cap_primitive={'attributes':{'POSITION':accessor(positions,'VEC3',5126,34962),'NORMAL':accessor(normals,'VEC3',5126,34962),'TEXCOORD_0':accessor(uv_cap,'VEC2',5126,34962)},'indices':accessor(cap_faces.astype('<u4').reshape(-1),'SCALAR',5125,34963),'material':cap_mi,'mode':4};doc['meshes'][0]['primitives'].append(cap_primitive)
 doc['buffers']=[{'byteLength':len(payload)}];dst=destination;dst.write_bytes(_pack_glb(doc,bytes(payload)));info=inspect_glb(dst)
 return {'source':{'path':str(src),'sha256':sha256_file(src)},'result':{'path':str(destination),'sha256':info['sha256'],'faces':info['faces']},'source_observed_roof_faces':int(roof.sum()),'generated_material_faces':int((~roof).sum()),'roof_levels':level,'boundary_conformance':boundary,'observed_gap_repair':gap,'face_count_unchanged':len(faces)==info['faces'],'all_original_embedded_material_images_retained':True,'source_evidence_sha256':sha256_file(evidence_image),'source_code_sha256':sha256_file(__file__),'surface_code_sha256':sha256_file(Path(__file__).with_name('_roof_surface.cpp')),'UV_policy':'Generated facades retain their original UV and PBR; new visible roof primitive uses the recorded original source image transform.','source_alpha_policy':'centroid and all vertices alpha>=200; visible surface and abs(normal_z)>.55; unknown retains neural appearance.','candidate_requires_independent_QA':True,'ground_truth_used':False}

def main(argv=None):
 import argparse
 from .appearance import write_json
 parser=argparse.ArgumentParser(description=__doc__)
 for key in ['source','output','building','fit_record','observation','roof_image']:
  parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
 parser.add_argument('--constrain-recorded-levels',action='store_true')
 parser.add_argument('--boundary-conformance',action='store_true')
 parser.add_argument('--repair-observed-gaps',action='store_true')
 args=parser.parse_args(argv);proof=args.output.with_suffix('.glb.proof.json')
 if proof.exists():raise ValueError('A new proof output is required')
 building=json.loads(args.building.read_text());fit=json.loads(args.fit_record.read_text());observation=json.loads(args.observation.read_text())
 report=reconcile_roof(args.source,args.output,building=building,world_from_asset=fit['world_from_asset_4x4'],observation=observation,evidence_image=args.roof_image,constrain_recorded_levels=args.constrain_recorded_levels,boundary_conformance=args.boundary_conformance,repair_observed_gaps=args.repair_observed_gaps)
 report['inputs']={key:{'sha256':sha256_file(getattr(args,key)),'path':str(getattr(args,key))} for key in ['building','fit_record','observation','roof_image']}
 write_json(proof,report);print(json.dumps({'status':'candidate_requires_independent_QA','result':report['result'],'proof':str(proof)}));return 0

if __name__=='__main__':raise SystemExit(main())
