import unittest,sys
from pathlib import Path
import numpy as np
from shapely.geometry import box,Polygon,MultiPolygon
from scipy.ndimage import label
from citycode_assets.structure_conditioning import conditioned_structure_coordinates
from citycode_assets.placement import initial_generated_placement,NATIVE_TO_GLTF


def building(geom,height=8,roof_layers=None):
 M=np.eye(4);M[:3,3]=[geom.centroid.x,geom.centroid.y,2]
 return {'building_id':'synthetic','footprint_world':geom.__geo_interface__,'height_m':height,'base_z_m':2.,'top_z_m':2+height,'world_from_local_4x4':M.tolist(),'local_from_world_4x4':np.linalg.inv(M).tolist(),'roof_layers':roof_layers or []}
def original():
 return np.array([(x,y,z) for x in range(4,28) for y in range(4,28) for z in range(5,27) if x in [4,27] or y in [4,27] or z in [5,26]],np.int32)

def world_xy(coords,audit,b):
 M=np.asarray(initial_generated_placement(b,audit)['world_from_asset_4x4'])@NATIVE_TO_GLTF
 native=(coords+.5)/32-.5;return native@M[:3,:3].T+M[:3,3]

class SourceStructureConditioning(unittest.TestCase):
 def test_invalid_inputs_rejected(self):
  with self.assertRaises(ValueError):conditioned_structure_coordinates(np.empty((0,3),int),building(box(0,0,10,10)))
 def test_source_concavity_and_roof_height_span(self):
  b=building(Polygon([(0,0),(12,0),(12,4),(4,4),(4,12),(0,12)]));o,a=conditioned_structure_coordinates(original(),b)
  self.assertTrue(a['source_footprint_conditioned']);self.assertFalse(a['ground_truth_used']);self.assertEqual(o.shape[1],3);self.assertGreater(len(o),20);self.assertGreaterEqual(o.min(),0);self.assertLess(o.max(),32);self.assertEqual(o[:,2].min(),a['z_min']);self.assertEqual(o[:,2].max(),a['z_max'])
  p=world_xy(o,a,b);self.assertFalse(np.any((p[:,0]>6)&(p[:,1]>6)))
 def test_courtyard_not_filled_by_solid_image_structure(self):
  geom=Polygon([(0,0),(20,0),(20,20),(0,20)],holes=[[(6,6),(14,6),(14,14),(6,14)]]);b=building(geom);o,a=conditioned_structure_coordinates(original(),b);p=world_xy(o,a,b)
  self.assertFalse(np.any((p[:,0]>7)&(p[:,0]<13)&(p[:,1]>7)&(p[:,1]<13)))
 def test_disconnected_source_components_are_both_retained(self):
  b=building(MultiPolygon([box(-10,-3,-3,3),box(3,-3,10,3)]));o,a=conditioned_structure_coordinates(original(),b);mask=np.zeros((32,32),bool);mask[o[:,0],o[:,1]]=True
  self.assertEqual(label(mask)[1],2)
 def test_known_roof_layer_retained_in_world_frame(self):
  geom=box(0,0,20,20);b=building(geom,10,[{'geometry':box(0,0,10,20).__geo_interface__,'base_z_m':7.,'top_z_m':12.}]);o,a=conditioned_structure_coordinates(original(),b);p=world_xy(o,a,b)
  self.assertEqual(a['source_roof_layers'],1);left=p[(p[:,0]>2)&(p[:,0]<8),2].max();right=p[(p[:,0]>12)&(p[:,0]<18),2].max();self.assertGreater(left-right,3.)
 def test_world_translation_changes_placement_not_native_structure(self):
  b1=building(box(0,0,16,12));b2=building(box(100,-40,116,-28));x,a=conditioned_structure_coordinates(original(),b1);y,c=conditioned_structure_coordinates(original(),b2)
  self.assertEqual(len(x),len(y));np.testing.assert_allclose(world_xy(y,c,b2).mean(axis=0)-world_xy(x,a,b1).mean(axis=0),[100,-40,0],atol=.1)

if __name__=='__main__':unittest.main()
