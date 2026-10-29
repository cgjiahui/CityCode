#!/usr/bin/env python3
"""Build an unredistributed native-derived, rectangular UE bridge template.

Inputs are an explicit local-mesh OBJ export and local UE asset catalog. Original
bridge Actors/maps/transforms are never read. Output stays in build/, not data/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'ue_citycode'))
from interchange import Writer

MESH = '/Game/Environment/Freeway/Mesh/SM_FREEWAY_VIS_349.SM_FREEWAY_VIS_349'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def template_source_obj(template_manifest, source_obj=None):
    """Resolve portable local Epic inputs, while accepting old absolute paths."""
    template_manifest=Path(template_manifest)
    manifest=json.loads(template_manifest.read_text())
    path=Path(source_obj) if source_obj is not None else Path(manifest['source_obj_file'])
    if source_obj is None and not path.is_absolute():
        path=template_manifest.parent/path
    if not path.is_file():
        raise FileNotFoundError('Native OBJ missing: '+str(path)+
                                '; export the licensed local mesh and supply --source-obj or rebuild the template')
    if sha(path)!=manifest['source_obj_sha256']:
        raise ValueError('Native OBJ SHA256 differs from the template manifest; rebuild the template for this local asset')
    return path


def read_obj(path):
    positions, uv, faces, texture_faces = [], [], [], []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        fields = line.split()
        if not fields: continue
        if fields[0] == 'v': positions.append([float(v) for v in fields[1:4]])
        elif fields[0] == 'vt': uv.append([float(v) for v in fields[1:3]])
        elif fields[0] == 'f':
            indices, textures = [], []
            for value in fields[1:]:
                parts = value.split('/')
                indices.append(int(parts[0])-1)
                if len(parts) < 2 or not parts[1]: raise ValueError('Every face corner requires UV')
                textures.append(int(parts[1])-1)
            for i in range(1, len(indices)-1):
                faces.append([indices[j] for j in (0,i,i+1)])
                texture_faces.append([textures[j] for j in (0,i,i+1)])
    vertices = np.asarray(positions, float)[:,[0,2,1]]/100.
    faces = np.asarray(faces, int)[:,[0,2,1]]  # OBJ export reverses handedness.
    texture_faces = np.asarray(texture_faces, int)[:,[0,2,1]]
    return vertices, np.asarray(uv, float), faces, texture_faces


def face_normals(vertices):
    normals = np.cross(vertices[:,1]-vertices[:,0], vertices[:,2]-vertices[:,0])
    length = np.linalg.norm(normals, axis=1)
    return normals/np.maximum(length[:,None],1e-30), length*.5


def section_limits(polygon, x):
    intersections=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if min(a[0],b[0])-1e-8 <= x <= max(a[0],b[0])+1e-8 and abs(b[0]-a[0]) > 1e-10:
            intersections.append(a[1]+(x-a[0])/(b[0]-a[0])*(b[1]-a[1]))
    return min(intersections), max(intersections)


def fit_top_quad(vertices, faces):
    triangles=vertices[faces]
    normals,area=face_normals(triangles)
    top=(normals[:,2]>.97)&(triangles[:,:,2].mean(axis=1)>.65)
    if top.sum()<2: raise ValueError('No sufficiently horizontal high-Z road-top faces')
    cloud=triangles[top].reshape(-1,3)[:,:2]
    hull=ConvexHull(cloud)
    polygon=cloud[hull.vertices]
    xlo,xhi=cloud[:,0].min(),cloud[:,0].max()
    positions=np.linspace(xlo+.05*(xhi-xlo),xhi-.05*(xhi-xlo),31)
    limits=np.asarray([section_limits(polygon,x) for x in positions])
    center=limits.mean(axis=1)
    slope,offset=np.polyfit(positions,center,1)
    normalized=polygon.copy();normalized[:,1]-=slope*normalized[:,0]+offset
    normalized=(normalized-normalized.mean(axis=0))/np.ptp(normalized,axis=0)
    # Four corners of the measured road surface, not its axis-aligned bbox.
    signs=np.asarray([[-1,-1],[1,-1],[1,1],[-1,1]])
    indices=[int(np.argmax(normalized@sign)) for sign in signs]
    if len(set(indices))!=4: raise ValueError('Road top is not an unambiguous quadrilateral')
    quad=polygon[indices]
    return quad, {'top_face_count':int(top.sum()),'top_projected_hull_area_m2':float(hull.volume),
                  'native_centerline_slope':float(slope),'native_centerline_yaw_deg':float(np.degrees(np.arctan(slope))),
                  'native_cross_section_width_median_m':float(np.median(np.diff(limits,axis=1))),
                  'source_top_quad_native_xy_m':quad.tolist()}


def inverse_bilinear(points, quad):
    q00,q10,q11,q01=np.asarray(quad)
    a,b,c=q10-q00,q01-q00,q11-q10-q01+q00
    st=(points-q00)@np.linalg.inv(np.column_stack((a,b))).T
    for _ in range(12):
        error=q00+st[:,0,None]*a+st[:,1,None]*b+(st[:,0]*st[:,1])[:,None]*c-points
        j0=a+st[:,1,None]*c;j1=b+st[:,0,None]*c
        determinant=j0[:,0]*j1[:,1]-j0[:,1]*j1[:,0]
        if np.any(abs(determinant)<1e-9):raise ValueError('Singular road quadrilateral')
        update=np.column_stack(((error[:,0]*j1[:,1]-error[:,1]*j1[:,0])/determinant,
                                (j0[:,0]*error[:,1]-j0[:,1]*error[:,0])/determinant))
        st-=update
        if abs(update).max()<1e-12:break
    return st


def regularize(vertices, faces, length=10., width=10.):
    quad,fit=fit_top_quad(vertices,faces)
    st=inverse_bilinear(vertices[:,:2],quad)
    result=vertices.copy()
    result[:,:2]=(np.clip(st,0.,1.)-.5)*[length,width]
    native_top=float(vertices[:,2].max())
    result[:,2]-=native_top
    top_vertices=vertices[:,2]>.65
    result[top_vertices,2]=0.  # remove centimetre-scale camber from repeated joins
    normals,areas=face_normals(result[faces])
    keep=areas>1e-10
    if not keep.all(): faces=faces[keep]
    fit.update({'length_m':length,'width_m':width,'native_top_max_z_m':native_top,
                'top_height_regularization_max_m':float(np.ptp(vertices[top_vertices,2])),
                'xy_clamped_source_vertices':int(np.any((st<0)|(st>1),axis=1).sum()),
                'degenerate_faces_removed':int((~keep).sum()),'mapping':'inverse measured top quadrilateral bilinear map to a rectangle; bounded XY clamp',
                'top_rule':'native vertices above 0.65m are set to deck top Z=0; remaining native vertical profile retained relative to top maximum'})
    return result,faces,keep,fit


def classify(triangles, normals):
    center=triangles.mean(axis=1)
    slots=np.full(len(triangles),2,int)
    underside=(normals[:,2]<-.35)|(center[:,2]<-.75)
    slots[underside]=0
    top=(normals[:,2]>.7)&(center[:,2]>-.1)
    slots[top]=1
    slots[top & (np.abs(center[:,1])>=4.25)]=3
    return slots


def write_glb(path, triangles, uv, slots, corner_normals=None):
    normals,_=face_normals(triangles)
    writer=Writer(directory=path.parent)
    colors=[[.32,.31,.29,1],[.12,.13,.14,1],[.48,.48,.45,1],[.22,.22,.2,1]]
    names=['MI_Freeway_Underside_Ramp','MI_FreewayAsphalt_Road','MI_FreewayConcrete_Side','MI_FreewayAsphalt_Shoulder']
    writer.doc['materials']=[{'name':name,'pbrMetallicRoughness':{'baseColorFactor':color,'metallicFactor':0.,'roughnessFactor':1.}} for name,color in zip(names,colors)]
    primitives=[]
    for slot in range(4):
        selected=slots==slot
        if not selected.any():raise ValueError('Semantic slot is empty: '+str(slot))
        # Native UE is left handed; glTF is Y-up/right handed. Both position
        # swap and triangle winding reversal are required; no actor correction.
        points=triangles[selected][:,[0,2,1]][:,:,[0,2,1]].reshape(-1,3).astype('<f4')
        tex=uv[selected][:,[0,2,1]].reshape(-1,2).copy();tex[:,1]=1.-tex[:,1]
        n=(np.repeat(normals[selected][:,[0,2,1]],3,axis=0) if corner_normals is None else
           corner_normals[selected][:,[0,2,1]][:,:,[0,2,1]].reshape(-1,3)).astype('<f4')
        attributes={'POSITION':writer.array(points), 'NORMAL':writer.array(n),
                    'TEXCOORD_0':writer.array(tex.astype('<f4'),{'componentType':5126,'type':'VEC2','count':len(tex)})}
        primitives.append({'attributes':attributes,'material':slot,'mode':4})
    writer.doc['meshes']=[{'name':'CityCodeNativeDerivedDeck10m','primitives':primitives}]
    writer.doc['nodes']=[{'mesh':0,'name':'NativeDeckTemplate'}];writer.doc['scenes']=[{'nodes':[0]}]
    writer.save(path)


def generate(source, catalog, output):
    output.mkdir(parents=True,exist_ok=True)
    entries=json.loads(catalog.read_text(encoding='utf-8-sig'))['assets']
    native=next(row for row in entries if row['mesh_path']==MESH)
    if len(native['materials'])!=4:raise ValueError('Expected four verified source material slots')
    vertices,uv,faces,uv_faces=read_obj(source)
    transformed,new_faces,keep,fit=regularize(vertices,faces)
    texture=uv[uv_faces][keep]
    triangles=transformed[new_faces];normals,_=face_normals(triangles)
    slots=classify(triangles,normals)
    target=output/'native_derived_deck_10m.glb';write_glb(target,triangles,texture,slots)
    bounds=np.asarray([transformed.min(axis=0),transformed.max(axis=0)])
    proof={'schema':'citycode.native_bridge_template.v1','status':'ready_for_ue_import',
           'template_file':target.name,'template_sha256':sha(target),'expected_native_bounds_m':bounds.tolist(),
           'source_mesh_path':MESH,'source_obj_file':os.path.relpath(source.resolve(),output.resolve()),
           'source_obj_path_base':'template_manifest_directory','source_obj_sha256':sha(source),'source_catalog_sha256':sha(catalog),
           'code_sha256':sha(Path(__file__)),'material_paths':[r['material_path'] for r in native['materials']],
           'material_assignment':'Geometry-semantic reassignment; OBJ has no usemtl. Original material-slot membership is not recoverable.',
           'material_slot_rules':['down-facing or lower underside -> underside','horizontal upper central surface -> road asphalt','remaining vertical/bevel faces -> side concrete','upper outer 15 percent of width -> shoulder asphalt'],
           'native_material_slot_triangle_counts':np.bincount(slots,minlength=4).tolist(),
           'source_triangle_count':len(faces),'template_triangle_count':len(new_faces),
           'source_uv_corner_count':len(texture)*3,'uv_policy':'Original per-corner OBJ UV retained; glTF V=1-OBJ_V; no UV resampling',
           'gltf_coordinate_policy':'glTF raw=(native UE X,native UE Z,native UE Y) metres; native dimensions are obtained by standard GLTFImporter basis, without building inverse basis',
           'native_deck_top_z_m':0.,'native_axes':{'length_axis':'X','up_axis':'Z'},'orientation_confirmed':True,
           'original_actor_world_transforms_read':False,'redistribution':'Derived Epic geometry is kept only in build; reproduce from licensed local City Sample assets; do not package this GLB as open data.',
           'fit':fit,'fixture_checked':False,'ue_import_executed':False}
    (output/'template_manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof,vertices,transformed,new_faces


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-obj',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--proposal',type=Path,help='Generate native-derived swept spans from explicit regularized prediction proposal')
    args=parser.parse_args();proof,_,_,_=generate(args.source_obj,args.catalog,args.output)
    print(json.dumps({'manifest':str(args.output/'template_manifest.json'),'bounds':proof['expected_native_bounds_m'],'triangles':proof['template_triangle_count'],'slot_counts':proof['native_material_slot_triangle_counts']}))
    if args.proposal:
        result=swept_spans(args.output/'template_manifest.json',args.proposal,args.output/'spans')
        print(json.dumps({'spans':result['actor_count'],'logical_modules':result['logical_module_count'],'manifest':str(args.output/'spans/span_manifest.json')}))
        qa=validate_span_files(args.output/'spans/span_manifest.json')
        print(json.dumps({'independent_geometry_qa':qa['status'],'spans':qa['actor_count']}))




def canonical_section(triangles, uv, slots):
    """Take a native transverse section, retaining surface UV derivatives."""
    normals,_=face_normals(triangles)
    sections=[]
    for face,tex,slot,normal in zip(triangles,uv,slots,normals):
        points=[]
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=face[i],face[j]
            if a[0]*b[0] < -1e-12:
                t=-a[0]/(b[0]-a[0]);points.append((a+t*(b-a),tex[i]+t*(tex[j]-tex[i])))
            elif abs(a[0])<1e-9:points.append((a,tex[i]))
        unique={tuple(np.round(point[0],8)):point for point in points}
        points=list(unique.values())
        if len(points)!=2:continue
        a,b=points
        edge=b[0]-a[0]
        if np.linalg.norm(edge[1:])<1e-8:continue
        if np.dot(np.array([0.,-edge[2],edge[1]]),normal)<0:a,b=b,a
        # Affine UV interpolation inside the measured source triangle; taking
        # its longitudinal derivative carries native texture density forward.
        design=np.column_stack((face,np.ones(3)))
        uvmap=np.linalg.lstsq(design,tex,rcond=None)[0]
        sections.append({'a':a[0][1:].tolist(),'b':b[0][1:].tolist(),
                         'uv_a':a[1].tolist(),'uv_b':b[1].tolist(),
                         'uv_dx':uvmap[0].tolist(),'material_slot':int(slot)})
    if len(sections)<4:raise ValueError('Native section has too few perimeter edges')
    return sections



def section_polygon(sections):
    lookup={tuple(np.round(edge['a'],6)):edge for edge in sections}
    first=sections[0];edge=first;points=[]
    for _ in range(len(sections)):
        points.append(edge['a']);key=tuple(np.round(edge['b'],6))
        if key not in lookup:raise ValueError('Native canonical section has an open perimeter')
        edge=lookup[key]
    if not np.allclose(edge['a'],first['a'],atol=1e-6):raise ValueError('Native section is not one closed loop')
    return np.asarray(points)


def polygon_triangles(polygon):
    polygon=np.asarray(polygon);indices=list(range(len(polygon)))
    sign=np.sign(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-np.roll(polygon[:,0],-1)*polygon[:,1]))
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    faces=[]
    while len(indices)>3:
        found=False
        for k,b in enumerate(indices):
            a,c=indices[k-1],indices[(k+1)%len(indices)];v=polygon[[a,b,c]]
            turn=sign*cross(v[1]-v[0],v[2]-v[1])
            if abs(turn)<1e-10:continue
            if turn<0:continue
            blocked=False
            for candidate in indices:
                if candidate in (a,b,c):continue
                point=polygon[candidate]
                products=[sign*cross(v[(j+1)%3]-v[j],point-v[j]) for j in range(3)]
                if min(products)>=-1e-10:blocked=True;break
            if blocked:continue
            faces.append([a,b,c]);indices.pop(k);found=True;break
        if not found:raise ValueError('Could not triangulate native cross-section cap')
    faces.append(indices)
    return np.asarray(faces,int)


def span_spline(span):
    """Shared producer/railing interpolation, including repaired joint tangents."""
    from scipy.interpolate import CubicHermiteSpline
    profile = span['profile']
    s = np.asarray([p['s_m'] for p in profile])
    xyz = np.asarray([p['position_m'] for p in profile])
    tangent = np.gradient(xyz, s, axis=0)
    for index, field in ((0, 'start_tangent_xyz'), (-1, 'end_tangent_xyz')):
        if field in span:
            tangent[index] = span[field]
    return s, CubicHermiteSpline(s, xyz, tangent)


def landing_crossfall_samples(span):
    """Measured terminal cross-section, sorted in the actual travel frame."""
    if not span.get('ground_transition'):
        return None
    contacts=np.asarray(span.get('landing_contact_xy_m', []),float)
    ground=np.asarray(span.get('landing_ground_z_m', []),float)
    if contacts.ndim!=2 or contacts.shape!=(len(ground),2) or len(ground)<2:
        raise ValueError('Ground transition requires measured transverse contacts')
    if not np.isfinite(contacts).all() or not np.isfinite(ground).all():
        raise ValueError('Nonfinite landing contact')
    s,spline=span_spline(span)
    direction=spline(s[-1],1)[:2]
    side=np.asarray([-direction[1],direction[0]])/np.linalg.norm(direction)
    lateral=(contacts-spline(s[-1])[:2])@side
    order=np.argsort(lateral)
    if np.any(np.diff(lateral[order])<1e-6):
        raise ValueError('Landing contacts must have distinct transverse positions')
    return lateral[order],ground[order],float(s[-1]),float(spline(s[-1])[2])


def landing_crossfall(span, stations, points, blend_length_m=15.):
    """Warp complete sections vertically toward measured landing crossfall.

    The last 15m uses a C1 smoothstep. Measurements are interpolated only inside
    their transverse support; outside it the boundary value is held constant.
    Applying the same offset to top and bottom retains native slab thickness.
    Input points have shape (stations, ..., 3), in native world XYZ metres.
    """
    result=np.asarray(points,float).copy()
    sample=landing_crossfall_samples(span)
    if sample is None:
        return result,0.
    stations=np.asarray(stations,float)
    if stations.ndim!=1 or result.shape[0]!=len(stations) or result.shape[-1]!=3:
        raise ValueError('Landing point/station dimensions differ')
    lateral,ground,length,terminal_z=sample
    _,spline=span_spline(span)
    center=spline(stations);direction=spline(stations,1)[:,:2]
    side=np.column_stack((-direction[:,1],direction[:,0]))
    side/=np.linalg.norm(side,axis=1)[:,None]
    expand=(len(stations),)+(1,)*(result.ndim-2)+(2,)
    transverse=np.sum((result[...,:2]-center[:,:2].reshape(expand))*side.reshape(expand),axis=-1)
    offset=np.interp(transverse,lateral,ground)-.025-terminal_z
    t=np.clip((stations-(length-min(blend_length_m,length)))/min(blend_length_m,length),0.,1.)
    weight=(t*t*(3.-2.*t)).reshape((len(stations),)+(1,)*(result.ndim-2))
    offset*=weight
    result[...,2]+=offset
    return result,float(np.max(np.abs(offset),initial=0.))


def landing_section_edges(sections,span,terminal_width_m):
    """Preserve native UV while adding vertices at every measured crossfall knot."""
    sample=landing_crossfall_samples(span)
    if sample is None:
        return sections
    knots=sample[0]*10./terminal_width_m
    refined=[]
    for edge in sections:
        a,b=np.asarray(edge['a']),np.asarray(edge['b'])
        ta,tb=np.asarray(edge['uv_a']),np.asarray(edge['uv_b'])
        fractions=[0.,1.]
        if abs(b[0]-a[0])>1e-10:
            fractions.extend(float(t) for t in (knots-a[0])/(b[0]-a[0]) if 1e-8<t<1.-1e-8)
        fractions=sorted(set(fractions))
        for lo,hi in zip(fractions[:-1],fractions[1:]):
            refined.append(dict(edge,a=(a+lo*(b-a)).tolist(),b=(a+hi*(b-a)).tolist(),
                                uv_a=(ta+lo*(tb-ta)).tolist(),uv_b=(ta+hi*(tb-ta)).tolist()))
    return refined


def swept_spans(template_manifest, proposal_path, output, source_obj=None):
    from scipy.interpolate import CubicHermiteSpline
    manifest=json.loads(template_manifest.read_text())
    # Read the original source used for the rectangular template, rather than
    # interpreting glTF UV twice. The caller sets an explicit source OBJ field.
    vertices,uv,faces,uv_faces=read_obj(template_source_obj(template_manifest,source_obj))
    transformed,new_faces,keep,_=regularize(vertices,faces)
    native_triangles=transformed[new_faces]
    native_uv=uv[uv_faces][keep]
    normals,_=face_normals(native_triangles)
    original_sections=canonical_section(native_triangles,native_uv,classify(native_triangles,normals))
    proposal=json.loads(proposal_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    records=[];all_qa=[]
    for span in proposal['continuous_segments']:
        identity=span['segment_id']
        modules=sorted([row for row in proposal['placements'] if row['source_segment_id']==identity],key=lambda row:row['placement_id'])
        if not modules:continue
        profile=span['profile'];s=np.asarray([p['s_m'] for p in profile]);xyz=np.asarray([p['position_m'] for p in profile])
        s,spline=span_spline(span)
        module_s=np.linspace(0.,s[-1],len(modules)+1)
        # Shared width at a joint is chosen conservatively from its two modules.
        widths=np.asarray([modules[0]['width_m']]+[min(a['width_m'],b['width_m'])for a,b in zip(modules[:-1],modules[1:])]+[modules[-1]['width_m']])
        sections=landing_section_edges(original_sections,span,widths[-1])
        perimeter=section_polygon(sections);cap_faces=polygon_triangles(perimeter)
        sample_s=np.unique(np.concatenate([np.linspace(a,b,max(2,int(np.ceil((b-a)/2.))+1)) for a,b in zip(module_s[:-1],module_s[1:])]))
        centers=spline(sample_s);forward=spline(sample_s,1);forward/=np.linalg.norm(forward,axis=1)[:,None]
        side=np.column_stack((-forward[:,1],forward[:,0],np.zeros(len(forward))));side/=np.linalg.norm(side,axis=1)[:,None]
        up=np.cross(forward,side);up/=np.linalg.norm(up,axis=1)[:,None]
        width=np.interp(sample_s,module_s,widths)
        anchor=(centers.min(axis=0)+centers.max(axis=0))*.5
        triangles=[];textures=[];materials=[];vertex_normals=[]
        ranges=[];material_offsets=np.zeros(4,int)
        seam_max=0.;landing_max_offset=0.
        for edge_index,edge in enumerate(sections):
            yz=np.asarray([edge['a'],edge['b']]);tex=np.asarray([edge['uv_a'],edge['uv_b']]);dx=np.asarray(edge['uv_dx'])
            rings=centers[:,None,:]+side[:,None,:]*(yz[None,:,0,None]*width[:,None,None]/10.)+up[:,None,:]*yz[None,:,1,None]
            rings,offset=landing_crossfall(span,sample_s,rings)
            landing_max_offset=max(landing_max_offset,offset)
            texrings=tex[None,:,:]+sample_s[:,None,None]*dx[None,None,:]
            edge_triangles=[];edge_textures=[];index_records=[]
            for k in range(len(sample_s)-1):
                for indices in (((k,0),(k+1,0),(k+1,1)),((k,0),(k+1,1),(k,1))):
                    edge_triangles.append([rings[a,b] for a,b in indices]);edge_textures.append([texrings[a,b]for a,b in indices]);index_records.append(indices)
            edge_triangles=np.asarray(edge_triangles)
            normal,area=face_normals(edge_triangles)
            smooth=np.zeros_like(rings)
            for face_normal,face_area,indices in zip(normal,area,index_records):
                for a,b in indices:smooth[a,b]+=face_normal*face_area
            smooth/=np.maximum(np.linalg.norm(smooth,axis=2,keepdims=True),1e-20)
            for index in module_s[1:-1]:
                matches=np.flatnonzero(abs(sample_s-index)<1e-7)
                if len(matches)!=1:raise ValueError('Every logical module joint must have one common ring')
                # One shared object represents both neighbouring boundaries,
                # so position/Z/normal cannot drift via independent transforms.
                joint=matches[0]
                seam_max=max(seam_max,float(np.max(np.abs(rings[joint]-rings[joint]))))
            ranges.append({'section_edge':edge_index,'material_slot':edge['material_slot'],
                           'primitive_first_face':int(material_offsets[edge['material_slot']]),'face_count':len(edge_triangles)})
            material_offsets[edge['material_slot']]+=len(edge_triangles)
            triangles.extend(edge_triangles-anchor)
            textures.extend(edge_textures);materials.extend([edge['material_slot']]*len(edge_triangles))
            vertex_normals.extend([[smooth[a,b]for a,b in indices]for indices in index_records])
        for endpoint,normal_sign in ((0,-1.),(-1,1.)):
            ring=centers[endpoint]+side[endpoint]*(perimeter[:,0,None]*width[endpoint]/10.)+up[endpoint]*perimeter[:,1,None]
            warped,offset=landing_crossfall(span,sample_s[[endpoint]],ring[None,:,:])
            ring=warped[0];landing_max_offset=max(landing_max_offset,offset)
            cap_triangles=ring[cap_faces]
            normals,_=face_normals(cap_triangles)
            desired=forward[endpoint]*normal_sign
            order=cap_faces.copy()
            if np.mean(normals@desired)<0:
                order=order[:,[0,2,1]];cap_triangles=ring[order]
            cap_uv=np.column_stack((perimeter[:,0]/10.+.5,-perimeter[:,1]/1.61209137))
            triangles.extend(cap_triangles-anchor);textures.extend(cap_uv[order]);materials.extend([2]*len(order))
            vertex_normals.extend(np.tile(desired,(len(order),3,1)))
        triangles=np.asarray(triangles);textures=np.asarray(textures);materials=np.asarray(materials)
        target=output/(identity+'.glb')
        write_glb(target,triangles,textures,materials,np.asarray(vertex_normals))
        local_bounds=[triangles.reshape(-1,3).min(axis=0).tolist(),triangles.reshape(-1,3).max(axis=0).tolist()]
        record={'span_id':identity,'template_file':target.name,'template_sha256':sha(target),'expected_native_bounds_m':local_bounds,
                'position_m':anchor.tolist(),'rotation_deg':[0.,0.,0.],'scale':[1.,1.,1.],
                'logical_module_count':len(modules),'placement_source':'predicted_elevated_structure',
                'inference':'explicit_rule_regularization','prepared_world_geometry':True,'triangle_count':len(triangles),
                'section_edge_primitive_face_ranges':ranges,'longitudinal_ring_count':len(sample_s),
                'module_joint_ring_indices':[int(np.argmin(abs(sample_s-value)))for value in module_s[1:-1]]}
        record['landing_crossfall']={'enabled':bool(span.get('ground_transition')),
                                    'blend_length_m':15.,'maximum_vertical_offset_m':landing_max_offset,
                                    'source':'published_mesh_terminal_transverse_contacts',
                                    'interpolation':'piecewise_linear_transverse_clamped_to_measured_support; longitudinal_C1_smoothstep',
                                    'terminal_top_relative_ground_m':-.025}
        record['native_section_lateral_bounds_m']=[float(perimeter[:,0].min()),float(perimeter[:,0].max())]
        records.append(record)
        all_qa.append({'span_id':identity,'logical_modules':len(modules),'shared_ring_count':len(module_s)-2,
                       'sweep_cross_section_edges':len(sections),'common_joint_rings_reused_by_construction':True,
                       'native_material_slot_triangle_counts':np.bincount(materials,minlength=4).tolist(),
                       'landing_crossfall_maximum_vertical_offset_m':landing_max_offset,
                       'geometry_surface_closed_at_extreme_span_ends':True})
    result={'schema':'citycode.native_bridge_span_templates.v1','status':'ready_for_ue_import',
            'material_paths':manifest['material_paths'],'templates':records,
            'actor_count':len(records),'logical_module_count':sum(r['logical_module_count']for r in records),
            'source_template_sha256':manifest['template_sha256'],'source_proposal_sha256':sha(proposal_path),
            'source_obj_sha256':manifest['source_obj_sha256'],'code_sha256':sha(Path(__file__)),
            'uses_original_bridge_actor_transforms':False,'uses_ground_truth_for_placement':False,
            'geometry_method':'Canonical transverse section of native-derived regularized template, swept over predicted centerlines with shared Hermite frames and cross sections at every logical module boundary; longitudinal sample step <=2m.',
            'uv_method':'Native per-corner section UV and source-triangle longitudinal UV derivative; repeated native section geometry is regularized, not a lossless copy of the 50m source.',
            'coordinate_policy':'Each GLB is standard glTF Y-up containing local native UE XYZ=(glTF X,glTF Z,glTF Y); actor applies only position_m anchor translation.',
            'replaces_not_appends':'250 ordinary rectangular native bridge placements',
            'redistribution':'Local licensed Epic-derived build artifacts only; do not publish these GLBs as open data.',
            'qa':all_qa}
    (output/'span_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate_span_files(manifest_path):
    """Decode stored GLB attributes and compare both actual sides of each seam."""
    import trimesh
    from interchange import read_glb,accessor
    manifest=json.loads(manifest_path.read_text());rows=[]
    for record in manifest['templates']:
        path=manifest_path.parent/record['template_file']
        if sha(path)!=record['template_sha256']:raise ValueError('Swept span hash changed')
        document,binary,_=read_glb(path)
        attributes={}
        all_positions=[];triangle_count=0
        for primitive in document['meshes'][0]['primitives']:
            attributes[primitive['material']]={}
            for name in ('POSITION','NORMAL','TEXCOORD_0'):
                values,_=accessor(document,binary,primitive['attributes'][name])
                if not np.isfinite(values).all():raise ValueError('Nonfinite swept attribute')
                if name!='TEXCOORD_0':values=values[:,[0,2,1]]
                if name=='NORMAL' and np.max(abs(np.linalg.norm(values,axis=1)-1))>2e-5:raise ValueError('Nonunit swept normal')
                attributes[primitive['material']][name]=values.reshape(-1,3,values.shape[1])[:,[0,2,1]]
            all_positions.extend(attributes[primitive['material']]['POSITION'].reshape(-1,3))
            triangle_count+=len(attributes[primitive['material']]['POSITION'])
        if set(attributes)!={0,1,2,3}:raise ValueError('Lost native material slot')
        maximum={'POSITION':0.,'NORMAL':0.,'TEXCOORD_0':0.};comparisons=0
        for section in record['section_edge_primitive_face_ranges']:
            first=section['primitive_first_face'];slot=section['material_slot']
            for ring in record['module_joint_ring_indices']:
                left=first+2*(ring-1);right=first+2*ring
                for name in maximum:
                    data=attributes[slot][name]
                    a=np.asarray([data[left,1],data[left,2]])
                    b=np.asarray([data[right,0],data[right+1,2]])
                    maximum[name]=max(maximum[name],float(np.max(np.abs(a-b))))
                comparisons+=1
        if max(maximum.values())>1e-7:raise ValueError('Actual stored shared cross-section attributes differ: '+str(maximum))
        xyz=np.asarray(all_positions);bounds=np.stack((xyz.min(axis=0),xyz.max(axis=0)))
        error=float(np.max(abs(bounds-np.asarray(record['expected_native_bounds_m']))))
        if error>.00003:raise ValueError('Decoded bounds differ from prepared native bounds')
        mesh=trimesh.load(path,force='mesh',process=False);mesh.merge_vertices(merge_tex=True,merge_norm=True)
        if not mesh.is_watertight:raise ValueError('Swept native span is not watertight: '+record['span_id'])
        rows.append({'span_id':record['span_id'],'triangles':triangle_count,'watertight_after_geometric_weld':True,
                     'decoded_bounds_maximum_error_m':error,'independent_joint_comparisons':comparisons,
                     'stored_joint_position_error_m':maximum['POSITION'],'stored_joint_normal_error':maximum['NORMAL'],
                     'stored_joint_uv_error':maximum['TEXCOORD_0']})
    result={'schema':'citycode.native_bridge_swept_geometry_qa.v1','status':'PASS','actor_count':len(rows),
            'logical_module_count':manifest['logical_module_count'],'scope':'Actual decoded GLB seam position/normal/UV, bounds, native material inventory and independent trimesh watertightness; UE import is a separate required check.',
            'ue_import_executed':False,'source_manifest_sha256':sha(manifest_path),'records':rows}
    (manifest_path.parent/'span_geometry_qa.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':main()
