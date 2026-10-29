"""CPU-only coastal postprocess of saved Pixal3D. No API, inference or source mutation.

Published land.geojson supplies the outline for replay. For new scenes, supply
an outline in the same GeoJSON format. Input/output GLB axes are (UE X, UE Z-height, UE Y). The solid plinth
is closed; generated decorative surfaces remain independent surface meshes.
"""
import argparse, copy, importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import shape, Polygon
import trimesh

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'web_preview'))
from prepare_neural_background import extract_mesh, sha, write_glb
spec=importlib.util.spec_from_file_location('merger',ROOT/'pipelines/background_pixal3d/experiments/pixal3d_p384_v5/merge_fused_tiles_glb.py')
merger=importlib.util.module_from_spec(spec);spec.loader.exec_module(merger)


def polygon_parts(g):
 if g.geom_type=='Polygon':return [g]
 if hasattr(g,'geoms'):return [p for q in g.geoms for p in polygon_parts(q)]
 return []


def clip_mesh(mesh,land,floor=0.):
 """Exact XY triangle/polygon intersection; barycentric height and UV interpolation."""
 vertices=np.asarray(mesh.vertices); faces=np.asarray(mesh.faces); uv=np.asarray(mesh.visual.uv)
 keep=[];extra_vertices=[];extra_uv=[];extra_faces=[];crossings=0
 # Keep original indexed vertices for interior triangles; only split shoreline faces.
 for start in range(0,len(faces),100000):
  chunk=faces[start:start+100000]; triangles=vertices[chunk][:,:,[0,2]]
  polys=shapely.polygons(triangles)
  inside=shapely.covers(land,polys); outside=shapely.disjoint(land,polys)
  keep.append(chunk[inside]); crossings+=int((~inside & ~outside).sum())
  for face,tri in zip(chunk[~inside & ~outside],triangles[~inside & ~outside]):
   det=np.linalg.det(np.column_stack((tri[1]-tri[0],tri[2]-tri[0])))
   if abs(det)<1e-10:continue
   cut=Polygon(tri).intersection(land)
   for poly in polygon_parts(cut):
    if poly.area<1e-9:continue
    xy,idx=trimesh.creation.triangulate_polygon(poly,engine='earcut')
    weights12=np.linalg.solve(np.column_stack((tri[1]-tri[0],tri[2]-tri[0])),(xy-tri[0]).T).T
    weights=np.column_stack((1-weights12.sum(axis=1),weights12))
    v=weights@vertices[face];v[:,0]=xy[:,0];v[:,2]=xy[:,1]
    # Preserve source projected winding (including downward-facing triangles).
    ab=xy[idx[:,1]]-xy[idx[:,0]];ac=xy[idx[:,2]]-xy[idx[:,0]]
    cross=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    idx[cross*det<0]=idx[cross*det<0,::-1]
    extra_faces.append(idx+len(vertices)+sum(len(x) for x in extra_vertices));extra_vertices.append(v);extra_uv.append(weights@uv[face])
 if extra_vertices:
  vertices=np.concatenate([vertices,*extra_vertices]);uv=np.concatenate([uv,*extra_uv]);keep+=extra_faces
 f=np.concatenate(keep) if keep else np.empty((0,3),int)
 if not len(f):return None,{'source_faces':len(faces),'faces':0,'shoreline_crossings':crossings}
 used,inv=np.unique(f,return_inverse=True);v=vertices[used].copy();raised=int((v[:,1]<floor).sum());v[:,1]=np.maximum(v[:,1],floor)
 out=trimesh.Trimesh(vertices=v,faces=inv.reshape(-1,3),process=False)
 out.visual=trimesh.visual.TextureVisuals(uv=uv[used],material=mesh.visual.material)
 return out,{'source_faces':len(faces),'faces':len(out.faces),'shoreline_crossings':crossings,'vertices_raised_to_land_floor':raised}


def solid_and_water(land,water_z=-3.5,bottom=-8.,floor=0.):
 if not bottom<water_z<floor:raise ValueError('Expected bottom < water < land floor')
 body=trimesh.creation.extrude_polygon(land,height=floor-bottom,engine='earcut');body.vertices[:,2]+=bottom
 assert body.is_watertight and body.is_winding_consistent
 # Convert Z-up extrusion to historical glTF axes and correct reflection winding.
 body.vertices=body.vertices[:,[0,2,1]];body.faces=body.faces[:,::-1]
 body.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[87,83,73,255],roughnessFactor=1,doubleSided=True))
 water=trimesh.Trimesh(vertices=[[-1032,water_z,-832],[952.05,water_z,-832],[952.05,water_z,832.1],[-1032,water_z,832.1]],faces=[[0,2,1],[0,3,2]],process=False)
 water.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[47,67,78,255],roughnessFactor=.8,doubleSided=True))
 return body,water


def shoreline_skirt(mesh,land,floor):
 # Close the exposed cut between each generated coastline edge and the solid cap.
 edges=mesh.edges_sorted
 unique,counts=np.unique(edges,axis=0,return_counts=True);edges=unique[counts==1]
 if not len(edges):return None
 p=mesh.vertices[edges];xy=p[:,:,[0,2]]
 near=shapely.intersects_xy(land.boundary.buffer(.002),xy[:,:,0].reshape(-1),xy[:,:,1].reshape(-1)).reshape(-1,2).all(axis=1)
 p=p[near & (p[:,:,1].max(axis=1)>floor+.001)]
 if not len(p):return None
 v=np.concatenate([p,p.copy()],axis=1);v[:,2:,1]=floor
 idx=np.arange(len(v))[:,None]*4+np.array([[0,1,3,0,3,2]])
 wall=trimesh.Trimesh(vertices=v.reshape(-1,3),faces=idx.reshape(-1,3),process=False)
 wall.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[87,83,73,255],roughnessFactor=1,doubleSided=True))
 return wall


def save_textured_mesh(mesh, raw, output, name):
 # Preserve compressed source image bytes rather than expanding WebP to PNG.
 original, offset, _ = merger.parse_glb(raw)
 doc = {k:copy.deepcopy(original[k]) for k in ('asset','materials','images','textures')}
 doc.update(scene=0, scenes=[{'nodes':[0]}], nodes=[{'mesh':0,'name':name}],
            meshes=[{'name':name,'primitives':[{'attributes':{},'material':0}]}],
            bufferViews=[],accessors=[])
 binary=bytearray()
 def view(data):
  binary.extend(b'\0'*(-len(binary)%4));start=len(binary);binary.extend(data)
  doc['bufferViews'].append({'buffer':0,'byteOffset':start,'byteLength':len(data)})
  return len(doc['bufferViews'])-1
 with raw.open('rb') as f:
  for image in doc['images']:
   v=original['bufferViews'][image['bufferView']];f.seek(offset+v.get('byteOffset',0))
   image['bufferView']=view(f.read(v['byteLength']))
 def accessor(array,kind,component):
  a={'bufferView':view(array.tobytes()),'componentType':component,'type':kind,'count':len(array)}
  if kind=='VEC3':a.update(min=array.min(axis=0).tolist(),max=array.max(axis=0).tolist())
  doc['accessors'].append(a);return len(doc['accessors'])-1
 primitive=doc['meshes'][0]['primitives'][0]
 primitive['attributes']['POSITION']=accessor(np.asarray(mesh.vertices,dtype='<f4'),'VEC3',5126)
 uv=np.asarray(mesh.visual.uv,dtype='<f4').copy();uv[:,1]=1-uv[:,1]
 primitive['attributes']['TEXCOORD_0']=accessor(uv,'VEC2',5126)
 primitive['indices']=accessor(np.asarray(mesh.faces,dtype='<u4').reshape(-1),'SCALAR',5125)
 write_glb(output,doc,binary)


def run(source,land_path,output,water_z,bottom,floor):
 if output.exists():raise ValueError('Refusing to overwrite '+str(output))
 output.parent.mkdir(parents=True,exist_ok=True)
 source_hash=sha(source);land=shape(json.loads(land_path.read_text())['geometry'])
 if not land.is_valid:raise ValueError('Invalid land polygon')
 shapely.prepare(land)
 doc,offset,_=merger.parse_glb(source)
 if any(any(k in n for k in ('matrix','translation','rotation','scale')) for n in doc['nodes']):raise ValueError('Expected world-space source')
 records=[];parts=[];total=0
 with tempfile.TemporaryDirectory(prefix='.coast-',dir=output.parent) as td:
  work=Path(td)
  with source.open('rb') as stream:
   for i,m in enumerate(doc['meshes']):
    raw=work/'raw.glb';extract_mesh(stream,doc,offset,i,raw)
    scene=trimesh.load(raw,force='scene',process=False);mesh=next(iter(scene.geometry.values()))
    part_floor=floor-.03 if 'continuous_base' in m['name'] else floor
    clipped,record=clip_mesh(mesh,land,part_floor);record['name']=m['name'];records.append(record)
    if clipped is not None:
     part=work/f'part_{i:03d}.glb';save_textured_mesh(clipped,raw,part,m['name']);parts.append(part);total+=record['faces']
     skirt=shoreline_skirt(clipped,land,floor-.06)
     if skirt is not None:
      scene=trimesh.Scene();scene.add_geometry(skirt,node_name='shore_'+str(i));record['skirt_faces']=len(skirt.faces)
      part=work/f'skirt_{i:03d}.glb';scene.export(part);parts.append(part);total+=len(skirt.faces)
    if (i+1)%10==0:print(f'Coastal {i+1}/{len(doc["meshes"])}',flush=True)
  body,water=solid_and_water(land,water_z,bottom,floor-.06)
  scene=trimesh.Scene();scene.add_geometry(body,node_name='land_solid',geom_name='land_solid');scene.add_geometry(water,node_name='water_plane',geom_name='water_plane')
  part=work/'solid_water.glb';scene.export(part);parts.append(part);total+=len(body.faces)+len(water.faces)
  sources=[(p,*merger.parse_glb(p)) for p in parts];combined,bases=merger.merge_documents(sources,total)
  staging=output.with_suffix('.tmp');merger.write_glb(staging,combined,sources,bases);staging.replace(output)
 assert sha(source)==source_hash
 result={'schema':'citycode.coastal_postprocess.v1','source_sha256':source_hash,'land_sha256':sha(land_path),'output_sha256':sha(output),'faces':total,'bytes':output.stat().st_size,'water_z_m':water_z,'land_floor_z_m':floor,'bottom_z_m':bottom,'solid_thickness_m':floor-.06-bottom,'solid_cap_z_m':floor-.06,'underlay_floor_z_m':floor-.03,'land_above_water_min_m':floor-water_z,'body_watertight':bool(body.is_watertight),'body_winding_consistent':bool(body.is_winding_consistent),'water_planar':bool(np.all(water.vertices[:,1]==water_z)),'coordinates':'glTF (UE X, UE Z-height, UE Y)','processing':'Exact polygon triangle clipping with barycentric UV/height; land height floor; coast skirts; closed solid plinth; single water plane. Base-color textures retained, PBR normal/MR maps omitted.','lineage':'GT orthographic visible-height land classification; 1m derived extent, not native coastline','api_calls':0,'gpu_inference':False,'parts':records}
 output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='parts'}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--land',type=Path,default=ROOT/'data/smallcity_p384_background_coastal/land.geojson');p.add_argument('--output',type=Path,required=True);p.add_argument('--water-z',type=float,default=-3.5);p.add_argument('--bottom',type=float,default=-8.);p.add_argument('--land-floor',type=float,default=0.);a=p.parse_args()
 run(a.source,a.land,a.output,a.water_z,a.bottom,a.land_floor)
