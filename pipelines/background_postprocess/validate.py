"""Independently inspect saved geometry without loading full texture images."""
import argparse,json
from pathlib import Path
import numpy as np,shapely,trimesh
from shapely.geometry import shape
from coastal import merger,sha

def validate(model,land_path):
 doc,offset,_=merger.parse_glb(model);land=shape(json.loads(land_path.read_text())['geometry']).buffer(.002);shapely.prepare(land)
 def array(i):
  a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']]
  n={'SCALAR':1,'VEC2':2,'VEC3':3}[a['type']];dtype={5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']]
  assert 'byteStride' not in v
  return np.memmap(model,mode='r',dtype=dtype,offset=offset+v.get('byteOffset',0)+a.get('byteOffset',0),shape=(a['count'],n))
 outside=0;vertices=0;faces=0;body=False;water=False
 for m in doc['meshes']:
  for p in m['primitives']:
   v=array(p['attributes']['POSITION']);f=array(p['indices']).reshape(-1,3);faces+=len(f);vertices+=len(v)
   assert np.isfinite(v).all()
   if m['name']=='water_plane':water=bool(np.all(v[:,1]==-3.5));assert water;continue
   outside+=int((~shapely.intersects_xy(land,v[:,0],v[:,2])).sum())
   if m['name']=='land_solid':
    mesh=trimesh.Trimesh(vertices=v,faces=f,process=False);body=bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0);assert body
 assert outside==0 and body and water
 return {'status':'passed','sha256':sha(model),'faces':faces,'vertices':vertices,'land_vertices_outside_2mm_tolerance':outside,'solid_closed_outward_winding':body,'water_plane_z_m':-3.5,'api_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--land',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=validate(a.model,a.land);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
