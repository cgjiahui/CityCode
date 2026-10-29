"""Export native observed evidence for each final predicted building identity."""
import argparse
from collections import OrderedDict
from pathlib import Path
import shutil

import cv2
import numpy as np
import trimesh
from PIL import Image
from shapely import affinity
from shapely.geometry import mapping, shape

from city_pipeline.runtime import read_json, write_json, sha256
from .observations import UE_Z_UP_TO_GLTF_Y_UP, extrude_geometry


def crop_bbox(mask, padding=0):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Observation mask has no target pixels')
    h, w = mask.shape
    return [max(0, int(xx.min())-padding), max(0, int(yy.min())-padding),
            min(w, int(xx.max())+1+padding), min(h, int(yy.max())+1+padding)]


def save_crop(rgb, mask, bbox, rgb_path, mask_path=None):
    x0, y0, x1, y1 = bbox
    crop = rgb[y0:y1, x0:x1]
    alpha = mask[y0:y1, x0:x1].astype(np.uint8)*255
    Image.fromarray(np.dstack([crop, alpha]), 'RGBA').save(rgb_path)
    if mask_path:
        Image.fromarray(alpha, 'L').save(mask_path)


def crop_camera(camera, bbox):
    x0, y0, x1, y1 = bbox
    k = np.asarray(camera['intrinsics']['K'], dtype=float).copy()
    k[0, 2] -= x0
    k[1, 2] -= y0
    return {'image_size_px': [x1-x0, y1-y0], 'crop_bbox_xyxy_px': bbox,
            'intrinsics': {'K': k.tolist()},
            'camera_to_world_4x4_m': camera['camera_to_world_4x4_m'],
            'world_to_camera_4x4_m': camera['world_to_camera_4x4_m'],
            'coordinate_conventions': camera.get('coordinate_conventions'),
            'source_view_id': camera['view_id']}


def export_evidence(source, release, output, data_root):
    source, release, output, data_root = map(lambda p: Path(p).resolve(),
                                              [source, release, output, data_root])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    resolve = lambda value: (data_root/value).resolve()
    observations = {f['properties']['observation_id']: f for f in
                    read_json(source/'global_fused/building_observation_footprints.geojson')['features']}
    e09_path = source/'e09_full_pipeline/manifest.json'
    e09 = {s['id']: s for s in read_json(e09_path)['samples']} if e09_path.is_file() else {}
    features = read_json(release/'global_building_footprints.geojson')['features']
    records, pending, sample_cache = [], [], OrderedDict()
    for f in features:
        props = f['properties']; gid = props['predicted_global_building_id']
        dest = output/'buildings'/gid
        (dest/'geometry').mkdir(parents=True)
        geom = shape(f['geometry']); cx, cy = geom.centroid.coords[0]
        base, height = float(props['base_z_m']), float(props['height_m'])
        local = affinity.translate(geom, xoff=-cx, yoff=-cy)
        for name, polygon in [('world', geom), ('local', local)]:
            write_json(dest/'geometry'/f'footprints_{name}.geojson', {
                'type': 'FeatureCollection', 'features': [{'type': 'Feature',
                'geometry': mapping(polygon), 'properties': {
                'building_id': gid, 'height_m': height, 'base_z_m': base if name=='world' else 0.0}}]})
        mesh = extrude_geometry(local, 0.0, height, [180, 190, 200, 255], {'building_id': gid})
        mesh.export(dest/'geometry/proxy_local_ue_z_up.glb')
        mesh.apply_transform(UE_Z_UP_TO_GLTF_Y_UP)
        mesh.export(dest/'geometry/proxy_local.glb')
        world_from_local = np.eye(4); world_from_local[:3, 3] = [cx, cy, base]
        canonical = props['canonical_observation_id']
        roles = {canonical: ['canonical']}
        for role, ids in (
            ('member', props.get('member_observation_ids', [])),
            ('consensus', props.get('consensus_observation_ids', [])),
            ('boundary_repair', props.get('v7_observation_ids', [])),
            ('roof_support', [oid for layer in props.get('roof_layers', [])
                              for oid in layer.get('observation_ids', [])]),
        ):
            for oid in ids:
                labels = roles.setdefault(oid, [])
                if role not in labels:
                    labels.append(role)
        missing = set(roles) - observations.keys()
        if missing:
            raise ValueError('Final building ' + gid + ' references missing observations: ' + ', '.join(sorted(missing)))
        exports = []
        for oid in roles:
            ob = observations[oid]['properties']; sid = ob['sample_id']
            if sid not in e09:
                pending.append({'building_id': gid, 'observation_id': oid,
                                'reason': 'E09 RGB/grid not included in supplied cache'})
                continue
            if sid not in sample_cache:
                s = e09[sid]; rgbp = resolve(ob['rgb_path']); camp = rgbp.parent/'camera.json'
                paths = {'source_rgb': rgbp, 'camera': camp,
                         'source_ids': resolve(ob['source_instance_ids_path']),
                         'bev_rgb': resolve(s['outputs']['bev_rgb']),
                         'bev_ids': resolve(ob['bev_instance_ids_path'])}
                if any(not p.is_file() for p in paths.values()):
                    pending.append({'building_id': gid, 'observation_id': oid,
                                    'reason': 'Supplied cache does not contain RGB/camera/instance evidence'})
                    continue
                camera = read_json(camp)
                if 'strict_valid_mask_path' in camera:
                    roi_reference = camera['strict_valid_mask_path']
                    if not isinstance(roi_reference, str) or not roi_reference:
                        raise ValueError('Camera strict-valid mask reference must be a nonempty path: ' + sid)
                    paths['strict_valid'] = (camp.parent / roi_reference).resolve()
                    if not paths['strict_valid'].is_file():
                        pending.append({'building_id': gid, 'observation_id': oid,
                                        'reason': 'Declared camera strict-valid mask is missing'})
                        continue
                    with Image.open(paths['strict_valid']) as roi:
                        if (list(roi.size) != camera['image_size_px']
                                or roi.mode not in ('1', 'L', 'I', 'I;16')):
                            raise ValueError('Camera strict-valid mask must be a native-size single-channel image: ' + sid)
                    # Keep the full sensor ROI beside the full sensor camera.
                    # Crop calibration is stored separately in observation.json.
                    camera['strict_valid_mask_path'] = 'strict_valid_mask.png'
                arrays = {k: np.asarray(Image.open(paths[k]).convert('RGB'))
                          for k in ['source_rgb', 'bev_rgb']}
                arrays.update({k: np.asarray(Image.open(paths[k])) for k in ['source_ids', 'bev_ids']})
                while len(sample_cache) >= 2:
                    sample_cache.popitem(last=False)
                sample_cache[sid] = (arrays, camera, s, paths)
            arrays, camera, s, paths = sample_cache[sid]
            sm = arrays['source_ids'] == int(ob['rgb_instance_id'])
            bm = arrays['bev_ids'] == int(ob['bev_instance_id'])
            if not sm.any() or not bm.any():
                pending.append({'building_id': gid, 'observation_id': oid,
                                'reason': 'No source or BEV pixels for this observation identity'})
                continue
            od = dest/'observations'/oid; od.mkdir(parents=True)
            sb, bb = crop_bbox(sm), crop_bbox(bm)
            cb = crop_bbox(sm, 32)
            save_crop(arrays['source_rgb'], sm, sb, od/'source_primary.png', od/'source_mask.png')
            save_crop(arrays['bev_rgb'], bm, bb, od/'roof_evidence.png', od/'bev_mask.png')
            x0, y0, x1, y1 = cb
            Image.fromarray(arrays['source_rgb'][y0:y1, x0:x1], 'RGB').save(od/'context.png')
            if 'strict_valid' in paths:
                shutil.copyfile(paths['strict_valid'], od/'strict_valid_mask.png')
            write_json(od/'camera.json', camera)
            crop_to_bev = np.array([[1, 0, bb[0]], [0, 1, bb[1]], [0, 0, 1]], float)
            crop_to_world = np.asarray(s['rasterization']['pixel_center_index_to_ue_world_xy_3x3']) @ crop_to_bev
            observation = {'schema': 'citycode.building_observed_evidence.v1',
                'building_id': gid, 'observation_id': oid, 'sample_id': sid,
                'patch_id': ob['patch_id'], 'view_id': ob['view_id'],
                'source_instance_id': ob['rgb_instance_id'], 'bev_instance_id': ob['bev_instance_id'],
                'is_canonical': oid==canonical, 'roles': roles[oid], 'source_bbox_xyxy_px': sb,
                'context_bbox_xyxy_px': cb, 'bev_bbox_xyxy_px': bb,
                'source_crop_camera': crop_camera(camera, sb),
                'context_crop_camera': crop_camera(camera, cb),
                'bev_crop_pixel_center_to_world_xy_3x3': crop_to_world.tolist(),
                'evidence_policy': 'Native observed masks retain the original observation identity. They may support cleaning, boundary repair or roof layers and need not equal the final footprint. Final geometry is stored separately; no GT identities or completed RGB are read.',
                'paths': {k: k+'.png' for k in ['source_primary','roof_evidence','context']},
                'source_sha256': {k: sha256(p) for k, p in paths.items()},
                'files': {p.name: sha256(p) for p in od.iterdir() if p.is_file()}}
            write_json(od/'observation.json', observation)
            exports.append({'observation_id': oid, 'is_canonical': oid==canonical,
                            'roles': roles[oid], 'path': str(od.relative_to(dest))})
        building = {'schema': 'citycode.building_generation_input.v1', 'building_id': gid,
            'canonical_observation_id': canonical, 'footprint_world': f['geometry'],
            'base_z_m': base, 'top_z_m': float(props['top_z_m']), 'height_m': height,
            'world_from_local_4x4': world_from_local.tolist(),
            'local_from_world_4x4': np.linalg.inv(world_from_local).tolist(),
            'coordinate_system': 'UE XYZ metres, +Z up; local origin at footprint centroid/base',
            'paths': {'footprints_world':'geometry/footprints_world.geojson',
                      'footprints_local':'geometry/footprints_local.geojson',
                      'proxy_local_ue_z_up':'geometry/proxy_local_ue_z_up.glb',
                      'proxy_local_gltf_y_up':'geometry/proxy_local.glb'},
            'observations': exports, 'roof_layers': props.get('roof_layers', []),
            'status': 'complete' if any(x['is_canonical'] for x in exports) else 'structure_only_evidence_pending'}
        write_json(dest/'building.json', building)
        records.append({'building_id': gid, 'path': str(dest.relative_to(output)),
                        'status': building['status'], 'observation_count': len(exports)})
    write_json(output/'manifest.json', {'schema': 'citycode.building_generation_inputs.v1',
        'status': 'complete' if not pending else 'structure_complete_with_pending_evidence',
        'building_count': len(records), 'records': records, 'pending': pending,
        'scope': 'Final predicted identity/structure, local LoD1 and native RGB evidence; no image/3D generation or GT evaluation.',
        'source_sha256': {'final_footprints': sha256(release/'global_building_footprints.geojson'),
                         'observations': sha256(source/'global_fused/building_observation_footprints.geojson')}})
    return output/'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['source', 'release', 'output', 'data-root']:
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    export_evidence(args.source, args.release, args.output, args.data_root)


if __name__ == '__main__':
    main()
