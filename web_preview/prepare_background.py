#!/usr/bin/env python3
"""Build independent, read-only browser assets from the frozen background replay."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import trimesh
from PIL import Image
from citycode_background.io import read, write, sha256, new_output
from citycode_background.raster import configure, pixels, release_pages

ROOT = Path(__file__).resolve().parents[1]


def prepare(bundle, reconstruction, output):
    bundle, reconstruction = Path(bundle).resolve(), Path(reconstruction).resolve()
    result = read(reconstruction/'RESULT.json')
    if result.get('status') != 'passed' or result.get('lineage') != 'historical_gt_assisted':
        raise ValueError('Expected a validated historical background replay')
    mesh = reconstruction/'mesh/background_ue_z_up.glb'
    if sha256(mesh) != result['mesh']['glb_sha256']:
        raise ValueError('Reconstructed background mesh checksum differs')
    out = new_output(output, bundle)
    configure(out/'scratch')
    Image.MAX_IMAGE_PIXELS = None
    grid = read(bundle/'global/pixel_to_world.json')
    rgb = pixels(bundle/'global/final_rgb.png')
    scale = 2048/max(rgb.shape[:2])
    resized = cv2.resize(rgb, (round(rgb.shape[1]*scale), round(rgb.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    release_pages(rgb); del rgb
    Image.fromarray(resized).save(out/'map.jpg', quality=93, subsampling=0)
    scene = trimesh.load(mesh, force='scene', process=False)
    for geometry in scene.geometry.values():
        material = geometry.visual.material
        texture = material.baseColorTexture.copy()
        texture.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        material.baseColorTexture = texture
    scene.export(out/'surfaces.glb')
    layers = {}
    for name in ['base', 'elevated']:
        transform, geometry_name = scene.graph[name]
        geometry = scene.geometry[geometry_name].copy(); geometry.apply_transform(transform)
        layers[name] = {'bounds': geometry.bounds.tolist(), 'faces': len(geometry.faces),
                        'vertices': len(geometry.vertices)}
    files = {role: {'path': name, 'sha256': sha256(out/name), 'bytes': (out/name).stat().st_size}
             for role, name in [('map', 'map.jpg'), ('surfaces', 'surfaces.glb')]}
    manifest = {'schema': 'citycode.background_web_preview.v1', 'lineage': 'historical_gt_assisted',
                'view_only': True, 'files': files, 'layers': layers,
                'bounds_xy': grid['raster_edge_bounds_xy_world_m'],
                'image_axis': grid['image_axis'], 'map_plane_z_m': -1.2,
                'map_plane_is_measured_surface': False,
                'coordinate_system': 'UE XYZ metres Z-up; already in world coordinates; no placement matrix',
                'source_rgb_sha256': sha256(bundle/'global/final_rgb.png'),
                'source_mesh_sha256': sha256(mesh), 'texture_size_px': [resized.shape[1], resized.shape[0]],
                'note': 'Historical GT-assisted background overlaid with current buildings; alignment not accepted. Elevated layer is a height surface, without bridge underside or piers.'}
    write(out/'manifest.json', manifest)
    shutil.rmtree(out/'scratch')
    print(json.dumps({'status': 'complete', 'output': str(out), 'layers': layers}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'data/smallcity_p384_background_history')
    p.add_argument('--reconstruction', type=Path, default=ROOT/'data/smallcity_p384_background_completed')
    p.add_argument('--output', type=Path, default=ROOT/'data/smallcity_p384_background_web_preview')
    args = p.parse_args()
    prepare(args.bundle, args.reconstruction, args.output)
