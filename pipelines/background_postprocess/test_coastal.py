import unittest
import numpy as np
import shapely
from shapely.geometry import Polygon,box
import trimesh
from coastal import clip_mesh,solid_and_water,save_textured_mesh
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

class CoastalTests(unittest.TestCase):
 def test_crossing_interpolates_uv_and_height(self):
  mesh=trimesh.Trimesh(vertices=[[-1,2,0],[1,4,0],[0,6,2]],faces=[[0,1,2]],process=False)
  mesh.visual=trimesh.visual.TextureVisuals(uv=[[0,0],[1,0],[.5,1]])
  land=box(0,-1,3,3);out,_=clip_mesh(mesh,land)
  self.assertAlmostEqual(shapely.union_all(shapely.polygons(out.vertices[out.faces][:,:,[0,2]])).area,1.)
  self.assertTrue(np.all(out.vertices[:,0]>=0))
  for xyz,uv in zip(out.vertices,out.visual.uv):
   self.assertTrue(np.allclose(uv,[(xyz[0]+1)/2,xyz[2]/2]))
   self.assertAlmostEqual(xyz[1],xyz[0]+1.5*xyz[2]+3)
 def test_concavity_clips_even_when_vertices_inside(self):
  land=Polygon([(0,0),(3,0),(3,3),(2,3),(2,1),(1,1),(1,3),(0,3)])
  mesh=trimesh.Trimesh(vertices=[[.1,1,2.9],[2.9,1,2.9],[1.5,1,.1]],faces=[[0,1,2]],process=False)
  mesh.visual=trimesh.visual.TextureVisuals(uv=[[0,0],[1,0],[.5,1]])
  out,_=clip_mesh(mesh,land)
  self.assertLess(shapely.union_all(shapely.polygons(out.vertices[out.faces][:,:,[0,2]])).difference(land).area,1e-8)
 def test_writer_preserves_uv_orientation_and_texture(self):
  mesh=trimesh.Trimesh(vertices=[[0,1,0],[2,1,0],[0,1,2]],faces=[[0,1,2]],process=False)
  mesh.visual=trimesh.visual.TextureVisuals(uv=[[.1,.2],[.8,.2],[.1,.9]],material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new('RGB',(2,2),(12,34,56))))
  with TemporaryDirectory() as tmp:
   raw=Path(tmp)/'raw.glb';out=Path(tmp)/'out.glb';trimesh.Scene(mesh).export(raw)
   save_textured_mesh(mesh,raw,out,'roundtrip')
   loaded=next(iter(trimesh.load(out,force='scene',process=False).geometry.values()))
   self.assertTrue(np.allclose(loaded.visual.uv,mesh.visual.uv))
   self.assertTrue(np.allclose(loaded.vertices,mesh.vertices))
   self.assertEqual(loaded.visual.material.baseColorTexture.convert('RGB').getpixel((0,0)),(12,34,56))
 def test_closed_body_and_flat_water(self):
  body,water=solid_and_water(box(0,0,5,5))
  self.assertTrue(body.is_watertight);self.assertTrue(body.is_winding_consistent)
  self.assertTrue(np.all(water.vertices[:,1]==-3.5));self.assertEqual(body.bounds[:,1].tolist(),[-8,0])
  self.assertAlmostEqual(body.volume,200)
if __name__=='__main__':unittest.main()
