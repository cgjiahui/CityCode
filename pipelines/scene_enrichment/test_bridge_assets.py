"""Bridge template geometry/axis/UV contracts without Unreal or Epic fixtures."""
import tempfile
import json
import shutil
from pathlib import Path
import unittest
import numpy as np
from bridge_assets import (inverse_bilinear, regularize, write_glb, landing_crossfall,
                           landing_section_edges, section_polygon, polygon_triangles,
                           template_source_obj,sha)
from bridge_railings import rail_pose, same_level_interior
from prepare import rotation_matrix
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ue_citycode'))
from interchange import read_glb,accessor


class NativeBridgeTests(unittest.TestCase):
    def test_template_obj_relative_paths_survive_bundle_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'original';(bundle/'export').mkdir(parents=True);(bundle/'template').mkdir()
            obj=bundle/'export/native.obj';obj.write_text('v 0 1 2\n')
            manifest=bundle/'template/template_manifest.json'
            manifest.write_text(json.dumps({'source_obj_file':'../export/native.obj','source_obj_sha256':sha(obj)}))
            shutil.move(str(bundle),str(root/'moved'))
            resolved=template_source_obj(root/'moved/template/template_manifest.json')
            self.assertEqual(resolved.resolve(),root/'moved/export/native.obj')

    def test_template_obj_absolute_compatibility_override_and_sha_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);obj=root/'native.obj';obj.write_text('v 0 1 2\n')
            manifest=root/'template_manifest.json';manifest.write_text(json.dumps({'source_obj_file':str(obj),'source_obj_sha256':sha(obj)}))
            self.assertEqual(template_source_obj(manifest),obj)
            moved=root/'relocated.obj';obj.rename(moved)
            with self.assertRaises(FileNotFoundError):template_source_obj(manifest)
            self.assertEqual(template_source_obj(manifest,moved),moved)
            moved.write_text('v 0 1 3\n')
            with self.assertRaisesRegex(ValueError,'SHA256 differs'):template_source_obj(manifest,moved)

    @staticmethod
    def landing_span(reverse=False):
        direction=-1. if reverse else 1.
        # Deliberately shuffled contacts, including nonmonotonic crossfall.
        lateral=np.array([0.,5.,-2.5,-5.,2.5])
        ground=np.array([.63,.8,.51,.4,.55])
        return {'ground_transition':True,
                'profile':[{'s_m':0.,'position_m':[0.,0.,.375]},
                           {'s_m':40.,'position_m':[40.*direction,0.,.375]}],
                'landing_contact_xy_m':np.column_stack((np.full(5,40.*direction),lateral*direction)).tolist(),
                'landing_ground_z_m':ground.tolist()}

    def test_landing_top_and_bottom_follow_all_measured_contacts_in_both_directions(self):
        for reverse in (False,True):
            span=self.landing_span(reverse)
            xy=np.asarray(span['landing_contact_xy_m'])
            top=np.column_stack((xy,np.full(len(xy),.375)))
            bottom=top-[0.,0.,1.612]
            result,maximum=landing_crossfall(span,[40.],np.stack((top,bottom),axis=1)[None])
            np.testing.assert_allclose(result[0,:,0,2],np.asarray(span['landing_ground_z_m'])-.025,atol=1e-12)
            np.testing.assert_allclose(result[0,:,0]-result[0,:,1],[ [0.,0.,1.612] ]*5,atol=1e-12)
            self.assertAlmostEqual(maximum,.4)

    def test_landing_blend_is_local_and_does_not_extrapolate_or_change_ordinary_spans(self):
        span=self.landing_span()
        stations=np.array([0.,25.,32.5,40.])
        points=np.column_stack((stations,np.full(4,9.),np.full(4,.375)))
        warped,_=landing_crossfall(span,stations,points)
        np.testing.assert_allclose(warped[:,2]-.375,[0.,0.,.2,.4],atol=1e-12)
        ordinary=dict(span,ground_transition=False,inplace_landing_end='start')
        unchanged,maximum=landing_crossfall(ordinary,stations,points)
        np.testing.assert_array_equal(unchanged,points)
        self.assertEqual(maximum,0.)

    def test_crossfall_surface_cap_and_railing_use_identical_end_rings(self):
        span=self.landing_span()
        perimeter=np.array([[-5.,0.],[5.,0.],[5.,-1.612],[-5.,-1.612]])
        sections=[{'a':a.tolist(),'b':b.tolist(),'uv_a':[0.,0.],'uv_b':[1.,0.],
                   'uv_dx':[.2,0.],'material_slot':i} for i,(a,b)in enumerate(zip(perimeter,np.roll(perimeter,-1,axis=0)))]
        refined=landing_section_edges(sections,span,10.)
        polygon=section_polygon(refined)
        polygon_triangles(polygon)  # No open or untriangulable cap after refinement.
        top=polygon[abs(polygon[:,1])<1e-12]
        np.testing.assert_allclose(np.sort(top[:,0]),[-5.,-2.5,0.,2.5,5.])
        world=np.column_stack((np.full(len(polygon),40.),polygon[:,0],polygon[:,1]+.375))
        cap,_=landing_crossfall(span,[40.],world[None])
        for edge in refined:
            yz=np.asarray([edge['a'],edge['b']])
            ring=np.column_stack((np.full(2,40.),yz[:,0],yz[:,1]+.375))
            swept,_=landing_crossfall(span,[40.],ring[None])
            for point in swept[0]:
                self.assertLess(np.min(np.linalg.norm(cap[0]-point,axis=1)),1e-12)
        rail,_=landing_crossfall(span,[40.,40.],np.array([[40.,-5.,.375],[40.,5.,.375]]))
        for point in rail:
            self.assertLess(np.min(np.linalg.norm(cap[0]-point,axis=1)),1e-12)

    def test_inverse_quad_removes_both_centerline_shear_and_slanted_ends(self):
        quad=np.array([[-25,-4],[24,-7.5],[25,4],[-24.4,7.6]])
        st=np.array([[0,0],[1,0],[1,1],[0,1],[.3,.7],[.5,.5]])
        a,b,c=quad[1]-quad[0],quad[3]-quad[0],quad[2]-quad[1]-quad[3]+quad[0]
        points=quad[0]+st[:,0,None]*a+st[:,1,None]*b+(st[:,0]*st[:,1])[:,None]*c
        np.testing.assert_allclose(inverse_bilinear(points,quad),st,atol=1e-12)

    def test_standard_gltf_axis_and_winding_preserve_normals_and_uv(self):
        # Four independent outward triangles ensure each native material slot
        # survives importer material-slot matching, not only its name in JSON.
        base=np.array([[[-5,-5,0],[5,-5,0],[-5,5,0]]],float)
        triangles=np.repeat(base,4,axis=0)
        uv=np.repeat(np.array([[[.2,.3],[4.1,.3],[.2,7.2]]]),4,axis=0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.glb';write_glb(path,triangles,uv,np.arange(4))
            doc,binary,_=read_glb(path)
            self.assertEqual(len(doc['meshes'][0]['primitives']),4)
            for primitive in doc['meshes'][0]['primitives']:
                p,_=accessor(doc,binary,primitive['attributes']['POSITION']);n,_=accessor(doc,binary,primitive['attributes']['NORMAL']);t,_=accessor(doc,binary,primitive['attributes']['TEXCOORD_0'])
                # Simulate the actually measured native importer Y/Z swap.
                native=p[:,[0,2,1]];np.testing.assert_allclose(np.sort(native,axis=0),np.sort(triangles[0],axis=0))
                normal=np.cross(p[1]-p[0],p[2]-p[0]);normal/=np.linalg.norm(normal)
                np.testing.assert_allclose(n[0],normal)
                np.testing.assert_allclose(t,np.array([[.2,.7],[.2,-6.2],[4.1,.7]]),atol=1e-6)
                self.assertEqual(primitive['material'],doc['meshes'][0]['primitives'].index(primitive))


    def test_native_guardrails_both_sides_keep_positive_scale_and_zero_deck_plane(self):
        bounds=np.asarray([[-110.4,0.,-146.3],[4.4,399.86,96.3]])
        for heading in (0.,31.,90.,181.,-73.):
            for grade in (-.1,0.,.1):
                start=np.array([412.,-290.,11.])
                direction=np.array([np.cos(np.radians(heading)),np.sin(np.radians(heading)),grade]);direction/=np.linalg.norm(direction)
                end=start+direction*3.9
                for side in (-1,1):
                    anchor,rotation,scale=rail_pose(start,end,side,bounds)
                    matrix=rotation_matrix(rotation)
                    self.assertGreater(min(scale),0.)
                    self.assertAlmostEqual(np.linalg.det(matrix),1.,places=10)
                    local=np.array([bounds[1,0]/100.,0.,0.])*scale
                    native_start=np.asarray(anchor)+matrix@local
                    target=start if side<0 else end
                    self.assertAlmostEqual(float(np.linalg.norm(native_start-target)),.025,places=9)
                    # Z=0 is the deck plane. The native hanging slab should
                    # remain below it, not force the whole guardrail up 1.46m.
                    lower=np.asarray(anchor)+matrix@(np.array([bounds[1,0]/100.,0.,bounds[0,2]/100.])*scale)
                    self.assertLess(lower[2],native_start[2]-1.45)


    def test_guardrail_junction_filter_does_not_merge_different_height_decks(self):
        fields={'other':(np.array([[0.,0.,10.]]),np.array([[20.,0.,10.]]),np.array([10.]),np.array([10.]))}
        crossing=np.array([[5.,3.,10.],[7.,3.,10.],[9.,3.,10.]])
        self.assertEqual(same_level_interior(crossing,fields,'own'),'other')
        self.assertIsNone(same_level_interior(crossing+[0.,0.,5.],fields,'own'))
        self.assertIsNone(same_level_interior(crossing+[0.,5.,0.],fields,'own'))
        self.assertIsNone(same_level_interior(crossing,fields,'other'))


if __name__=='__main__':unittest.main()
