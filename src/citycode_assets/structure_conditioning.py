"""Prediction-footprint conditioning of the32-cubed S1 support for refinement.

Image-only support remains evidence. Its roof profile is retained where no roof
layers were recovered. Current source geometry controls XY support and recorded
roof levels; neither reference buildings nor other assets enter this recipe.
"""
from __future__ import annotations
import copy
import numpy as np
from shapely import affinity
from shapely.geometry import shape,box
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion


def conditioned_structure_coordinates(original, building):
    from .generation import predicted_height_coordinates,stretch_height
    from .placement import initial_generated_placement,NATIVE_TO_GLTF
    footprint=shape(building['footprint_world'])
    corrected,audit=predicted_height_coordinates(original,footprint,float(building['height_m']))
    pose=initial_generated_placement(building,audit)
    native_from_world=np.linalg.inv(NATIVE_TO_GLTF)@np.linalg.inv(np.asarray(pose['world_from_asset_4x4']))
    a=native_from_world
    target=affinity.affine_transform(footprint,[a[0,0],a[0,1],a[1,0],a[1,1],a[0,3],a[1,3]])
    lo=np.asarray(target.bounds[:2]);hi=np.asarray(target.bounds[2:]);extent=float(max(hi-lo));factor=min(1.,.875/extent);offset=-factor*(lo+hi)/2
    # Keep the exact analytic placement recipe: this changes the target frame
    # of the existing control-to-model similarity, not a fitted output mesh.
    alignment=copy.deepcopy(audit['model_frame_alignment']);normal=np.asarray(alignment['target_normalization'],float)
    normal[:2]=factor*normal[:2]+offset;normal[2]*=factor;alignment['target_normalization']=normal.tolist();audit['model_frame_alignment']=alignment
    target=affinity.translate(affinity.scale(target,factor,factor,origin=(0,0)),*offset)
    xy_native=(corrected[:,:2]+.5)/32-.5;xy_grid=np.floor((factor*xy_native+offset+.5)*32).astype(np.int32)
    recentered=corrected.copy();recentered[:,:2]=xy_grid
    height=max(1,int(round(audit['requested_height_voxels']*factor)));recentered,stretch=stretch_height(recentered,height)
    audit.update({k:stretch[k] for k in ['requested_height_voxels','z_min','z_max']})
    zmin,zmax=audit['z_min'],audit['z_max'];mask=np.zeros((32,32),bool)
    for x in range(32):
        for y in range(32):
            cell=box(x/32-.5,y/32-.5,(x+1)/32-.5,(y+1)/32-.5)
            mask[x,y]=target.intersection(cell).area>.10/(32*32)
    if not mask.any():raise ValueError('Current footprint vanished on the32-cubed coarse support')
    coords=np.argwhere(mask);groups={}
    for x,y,z in recentered:
        if 0<=x<32 and 0<=y<32:groups[(int(x),int(y))]=max(groups.get((int(x),int(y)),zmin),int(z))
    if not groups:raise ValueError('No image-derived roof profile after recentering')
    keys=np.asarray(list(groups),int);tops=np.asarray([groups[tuple(p)] for p in keys],int);nearest=cKDTree(keys).query(coords)[1];roof=np.clip(tops[nearest],zmin,zmax)
    # Explicit recovered roof levels override the image profile only where
    # their source polygons declare them. Unknown roof geometry stays inferred.
    if building.get('roof_layers'):
        pose=initial_generated_placement(building,audit);native_from_world=np.linalg.inv(NATIVE_TO_GLTF)@np.linalg.inv(np.asarray(pose['world_from_asset_4x4']));a=native_from_world
        body_top=min(float(layer['base_z_m']) for layer in building['roof_layers']);native_z=float((a@np.array([0.,0.,body_top,1.]))[2]);roof[:]=int(np.clip(np.floor((native_z+.5)*32),zmin,zmax))
        from shapely import contains_xy
        centers=(coords+.5)/32-.5
        for layer in building['roof_layers']:
            p=affinity.affine_transform(shape(layer['geometry']),[a[0,0],a[0,1],a[1,0],a[1,1],a[0,3],a[1,3]])
            hit=contains_xy(p,centers[:,0],centers[:,1]);z=float((a@np.array([0.,0.,layer['top_z_m'],1.]))[2]);roof[hit]=np.maximum(roof[hit],int(np.clip(np.floor((z+.5)*32),zmin,zmax)))
    # Retain the global prediction height even when its sampled peak falls in
    # a subcell source feature; the nearest supported column carries that peak.
    if roof.max()<zmax:
        peak=keys[int(tops.argmax())];index=int(np.argmin(np.sum((coords-peak)**2,axis=1)));roof[index]=zmax
    edge=mask&~binary_erosion(mask,structure=np.ones((3,3)),border_value=0)
    result=set();roof_grid=np.full((32,32),-1,int)
    for (x,y),z in zip(coords,roof):
        x,y,z=int(x),int(y),int(z);roof_grid[x,y]=z
        result.add((x,y,zmin));result.add((x,y,z))
        if edge[x,y]:result.update((x,y,k) for k in range(zmin,z+1))
    for x,y,z in recentered:
        if 0<=x<32 and 0<=y<32 and mask[x,y] and zmin<=z<=roof_grid[x,y]:result.add((int(x),int(y),int(z)))
    out=np.asarray(sorted(result),np.int32)
    before_xy=set(map(tuple,recentered[:,:2].tolist()));after_xy=set(map(tuple,out[:,:2].tolist()))
    audit.update(before_tokens=int(len(original)),after_tokens=int(len(out)),xy_added=len(after_xy-before_xy),xy_removed=len(before_xy-after_xy),preexisting_z_gaps_closed=None,source_side_surfaces_completed=True,aligned_control_area=float(audit['aligned_control_area']*factor*factor))
    audit.update(mechanism='source_footprint_conditioned_S1_with_image_roof_and_recorded_source_roof_layers',footprint_mask_used=True,source_footprint_conditioned=True,image_only_original_tokens=int(len(original)),conditioned_tokens=int(len(out)),coarse_XY_grid_cells=int(mask.sum()),grid_size=32,source_roof_layers=len(building.get('roof_layers',[])),ground_truth_used=False,frame_recenter_scale=float(factor),frame_recenter_offset=offset.tolist(),roof_profile='recorded_source_levels' if building.get('roof_layers') else 'same_image_S1_inferred_roof',rasterization_policy='source_polygon_cell_intersection_area_over10percent; no convex hull or hole filling')
    return out,audit
