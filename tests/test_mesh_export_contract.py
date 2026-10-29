import unittest
import io
import numpy as np
import trimesh
from shapely.geometry import Polygon,MultiPolygon,box
from shapely.ops import unary_union
from city_pipeline.buildings.observations import extrude_geometry
class Contracts(unittest.TestCase):
 def check(self,g):
  original=g.wkb;m=extrude_geometry(g,-2.,10.,[50,60,70],{'id':'test'});tri=m.triangles;caps=[Polygon(t[:,:2]) for t in tri[np.all(np.isclose(tri[:,:,2],10),axis=1)]];cap=unary_union(caps)
  self.assertTrue(m.is_watertight);self.assertTrue(m.is_winding_consistent);self.assertAlmostEqual(g.symmetric_difference(cap).area,0.,places=8);self.assertAlmostEqual(m.volume,g.area*12,places=7);self.assertEqual(g.wkb,original);self.assertEqual(m.metadata['id'],'test');return m
 def test_courtyard(self):self.check(Polygon(box(0,0,10,10).exterior,[box(2,2,8,8).exterior]))
 def test_subfive_component(self):self.check(MultiPolygon([box(0,0,10,10),box(20,0,21,1)]))
 def test_entire_small_polygon(self):self.check(box(0,0,.1,.1))
 def test_thin_notch(self):self.check(Polygon([(0,0),(10,0),(10,10),(5.001,10),(5.001,1),(5.,1),(5.,10),(0,10)]))
 def test_collinear_vertices(self):self.check(Polygon([(0,0),(1,0),(2,0),(2,2),(0,2)]))
 def test_multiple_holes(self):self.check(Polygon(box(0,0,20,20).exterior,[box(1,1,2,2).exterior,box(5,5,10,10).exterior]))
 def test_clockwise(self):self.check(Polygon(list(box(0,0,10,10).exterior.coords)[::-1]))
 def test_world_coordinates(self):self.check(box(1500.1,-2340.1,1530.3,-2310.3))
 def test_glb_roundtrip(self):
  m=self.check(Polygon(box(0,0,10,10).exterior,[box(2,2,8,8).exterior]));loaded=trimesh.load(io.BytesIO(m.export(file_type='glb')),file_type='glb',force='mesh',process=False);self.assertTrue(loaded.is_watertight);self.assertAlmostEqual(loaded.volume,m.volume,places=6)
 def test_courtyard_point_contact(self):self.check(Polygon(box(0,0,10,10).exterior,[[(0,5),(2,4),(2,6),(0,5)]]))
 def test_invalid_not_silently_fixed(self):
  with self.assertRaises(ValueError):self.check(Polygon([(0,0),(1,1),(0,1),(1,0)]))
 def test_invalid_height(self):
  with self.assertRaises(ValueError):extrude_geometry(box(0,0,1,1),2,2,[0,0,0],{})
if __name__=='__main__':unittest.main()
