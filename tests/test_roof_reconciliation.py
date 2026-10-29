"""Observed roof projection preserves fine facades and discloses geometry changes."""
from pathlib import Path
import copy,json,shutil,tempfile,unittest
import numpy as np
from PIL import Image
from shapely.geometry import box,mapping
from citycode_assets.roof_reconciliation import reconcile_roof,conform_levels,observed_gap_cap
from citycode_assets.lod import _read_glb,inspect_glb,sha256
from test_asset_delivery_contract import cube_glb

class RoofReconciliation(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
    def test_without_recorded_levels_geometry_is_unchanged(self):
        vertices=np.array([[0.,0,0],[0,1,4],[1,0,2]])
        result,record=conform_levels(vertices,[[0,1,2]],{'roof_layers':[]})
        np.testing.assert_array_equal(result,vertices);self.assertFalse(record['applied'])
    @unittest.skipUnless(shutil.which('g++'),'Offline CPU compiler required')
    def test_facade_outside_roof_edge_does_not_collapse_to_roof(self):
        vertices=np.array([[0.,0,0],[10,0,0],[10,10,0],[0,10,0],[0,0,10],[10,0,10],[10,10,10],[0,10,10],[-.5,0,0],[-.5,10,0],[-.25,5,2],[-.25,5,5],[-.25,5,8]])
        faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[8,0,3],[8,3,9]])
        building={'base_z_m':0.,'top_z_m':12.,'roof_layers':[{'geometry':mapping(box(0,0,5,10)),'base_z_m':8.,'top_z_m':12.}]}
        result,record=conform_levels(vertices,faces,building)
        np.testing.assert_array_equal(result[:,:2],vertices[:,:2])
        self.assertTrue(record['applied']);self.assertTrue(np.all(np.diff(result[-3:,2])>1.))
        self.assertLess(result[-3,2],4.);self.assertLess(result[-1,2],12.)
    @unittest.skipUnless(shutil.which('g++'),'Offline CPU compiler required')
    def test_roof_projection_preserves_source_bytes_faces_and_facade_material(self):
        source=self.root/'source.glb';cube_glb(source);before=source.read_bytes()
        image=self.root/'observed.png';Image.new('RGBA',(16,16),(30,60,220,255)).save(image)
        observation={'files':{'roof_evidence.png':sha256(image)},'bev_crop_pixel_center_to_world_xy_3x3':[[.1,0,-.2],[0,.1,-.2],[0,0,1]]}
        M=[[1,0,0,0],[0,0,-1,1],[0,1,0,0],[0,0,0,1]]
        output=self.root/'result.glb';report=reconcile_roof(source,output,building={'roof_layers':[]},world_from_asset=M,observation=observation,evidence_image=image)
        self.assertEqual(source.read_bytes(),before);self.assertEqual(inspect_glb(output)['faces'],12)
        self.assertGreater(report['source_observed_roof_faces'],0);self.assertGreater(report['generated_material_faces'],0)
        self.assertFalse(report['roof_levels']['applied']);self.assertFalse(report['ground_truth_used'])
        _,original,_=_read_glb(source);_,result,_=_read_glb(output)
        self.assertEqual(result['materials'][:len(original['materials'])],original['materials'])
        # Exercise the new cap's position accessor as well as the retained mesh.
        from unittest.mock import patch
        from citycode_assets.lod import _accessor
        cap=np.array([[.2,.2,1.],[.8,.2,1.],[.2,.8,1.]])
        with patch('citycode_assets.roof_reconciliation.observed_gap_cap',return_value=(cap,np.array([[0,1,2]]),{'applied':True,'added_faces':1})):
            reconcile_roof(source,self.root/'cap.glb',building={'roof_layers':[]},world_from_asset=M,observation=observation,evidence_image=image,repair_observed_gaps=True)
        payload,document,binary=_read_glb(self.root/'cap.glb')
        # The source fixture is deliberately minimal; this checks the added cap.
        primitive=document['meshes'][0]['primitives'][-1]
        values,accessor=_accessor(payload,document,binary,primitive['attributes']['POSITION'])
        np.testing.assert_array_equal(accessor['min'],values.min(axis=0))
        np.testing.assert_array_equal(accessor['max'],values.max(axis=0))
        with self.assertRaises(ValueError):reconcile_roof(source,output,building={},world_from_asset=M,observation=observation,evidence_image=image)
    def test_changed_observed_image_is_rejected_before_output(self):
        source=self.root/'source.glb';cube_glb(source);image=self.root/'observed.png';Image.new('RGBA',(16,16),'blue').save(image)
        with self.assertRaises(ValueError):reconcile_roof(source,self.root/'result.glb',building={},world_from_asset=np.eye(4),observation={'files':{'roof_evidence.png':'0'*64}},evidence_image=image)
        self.assertFalse((self.root/'result.glb').exists())

    @unittest.skipUnless(shutil.which('g++'),'Offline CPU compiler required')
    def test_supported_gap_cap_preserves_current_source_courtyard(self):
        from shapely.geometry import Polygon
        vertices=np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,.2,1],[.2,0,1]])
        # Real base/walls and a small surviving roof corner support the height.
        faces=np.array([[0,1,2],[0,2,3],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7],[4,9,8]])
        courtyard=box(.35,.35,.65,.65)
        footprint=Polygon(box(0,0,1,1).exterior.coords,[courtyard.exterior.coords])
        building={'base_z_m':0.,'height_m':1.,'footprint_world':mapping(footprint),'roof_layers':[]}
        bev=np.array([[.05,0,-.1],[0,.05,-.1],[0,0,1.]])
        rgba=np.full((25,25,4),255,np.uint8)
        cap,triangles,report=observed_gap_cap(vertices,faces,building,bev,rgba)
        self.assertTrue(report['applied']);self.assertGreater(len(triangles),0)
        import shapely
        actual=shapely.union_all(shapely.polygons(cap[triangles][:,:,:2]))
        self.assertLess(actual.intersection(courtyard).area,1e-10)
        self.assertLess(actual.difference(footprint).area,1e-10)

if __name__=='__main__':unittest.main()
