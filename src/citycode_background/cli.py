"""Frozen background replay. All commands are offline; no API client is installed here."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from .io import read, write, local, load_bundle, verify, sha256, new_output


def prepare_requests(root, cfg, output):
    # This exports a plan only. It never executes or authorizes billable requests.
    rows = []
    for version, ids in [('v9', [t['tile_id'] for t in cfg['tiles']]), ('v10', cfg['selected_ids'])]:
        prompt = local(root, f'prompts/{version}.txt').read_text()
        by_id = {t['tile_id']: t for t in cfg['tiles']}
        for tid in ids:
            t = by_id[tid]; source = local(root, t['files']['input_rgb'])
            dst = output/'inputs'/f'{tid}.png'; dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(source, dst)
            rows.append({'id': f'{version}/{tid}', 'input': f'inputs/{tid}.png', 'input_sha256': sha256(dst),
                         'prompt': prompt, 'size': [2048, 2048], 'quality': 'high', 'output_format': 'png',
                         'result': f'results/{version}/{tid}.png'})
    result = {'schema': 'citycode.background_requests.v1', 'status': 'plan_only_not_submitted',
              'historical_lineage': cfg['lineage'], 'model_identity': cfg['historical_rgb_model'],
              'network_calls': 0, 'new_generation_cost': 'depends on external provider; not estimated or incurred',
              'requests': rows}
    write(output/'requests.json', result)
    return {'status': result['status'], 'requests': len(rows), 'network_calls': 0}


def compare_structure(root, cfg, footprints, output):
    import cv2
    from .rgb import pixels, save
    grid = read(local(root, cfg['grid'])); width, height = grid['image_size_px']
    inverse = np.asarray(grid['ue_world_xy_to_pixel_center_index_3x3'])
    current = np.zeros((height, width), np.uint8)
    features = read(footprints)['features']
    # Fill one polygon at a time so a hole cannot erase another overlapping building.
    for feature in features:
        g = feature['geometry']
        if g['type'] not in ['Polygon', 'MultiPolygon']:
            raise ValueError('Expected building Polygon/MultiPolygon')
        polygons = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
        for polygon in polygons:
            rings = []
            for ring in polygon:
                a = np.asarray(ring)[:, :2]; h = np.column_stack([a, np.ones(len(a))]) @ inverse.T
                rings.append(np.round(h[:, :2]).astype(np.int32))
            points = np.vstack(rings); lo = np.maximum(points.min(0), 0); hi = np.minimum(points.max(0)+1, [width, height])
            x0, y0 = lo; x1, y1 = hi
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1-y0, x1-x0), np.uint8)
            cv2.fillPoly(mask, [rings[0]-lo], 1)
            for hole in rings[1:]:
                cv2.fillPoly(mask, [hole-lo], 0)
            current[y0:y1, x0:x1] |= mask
    old = pixels(local(root, 'global/building_union_mask.png'), 'L') > 0
    current = current > 0; intersection = int((old & current).sum()); union = int((old | current).sum())
    changed = old ^ current
    result = {'schema': 'citycode.background_structure_compatibility.v1', 'status': 'measured_not_automatically_accepted',
              'current_building_features': len(features), 'footprints_sha256': sha256(footprints),
              'rasterization': 'nearest pixel-centre polygon raster at 0.15m; old mask includes predicted/GT footprint union',
              'intersection_over_union': intersection/union if union else 1,
              'historical_only_pixels': int((old & ~current).sum()), 'current_only_pixels': int((current & ~old).sum()),
              'changed_pixels': int(changed.sum()), 'changed_area_m2': float(changed.sum()*grid['meters_per_pixel']**2),
              'map_frame_unchanged': True, 'new_api_calls': 0,
              'interpretation': 'Historical RGB/height reuse; changed footprints were NOT used to regenerate RGB or fill height holes. This comparison does not prove asset alignment.'}
    save(output/'footprint_difference.png', changed.astype(np.uint8)*255)
    write(output/'compatibility.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['verify', 'replay', 'mesh', 'prepare-requests', 'compare-structure']:
        p = commands.add_parser(name)
        p.add_argument('--bundle', type=Path, required=True)
        if name != 'verify':
            p.add_argument('--output', type=Path, required=True)
        if name in ['mesh', 'replay']:
            p.add_argument('--stride', type=int, default=16)
            p.add_argument('--texture-size', type=int, default=4096)
        if name == 'replay':
            p.add_argument('--rgb-only', action='store_true')
        if name == 'compare-structure':
            p.add_argument('--footprints', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'replay':
        import cv2
        if cv2.__version__ != "5.0.0":
            raise RuntimeError("Historical raster replay requires opencv-python-headless==5.0.0.93; install the background extra in a separate environment")
    result = verify(args.bundle)
    if args.command == 'verify':
        print(result)
        return
    root, cfg = load_bundle(args.bundle)
    out = new_output(args.output, root)
    write(out/'RUN_STATUS.json', {'status': 'running', 'command': args.command, 'api_calls': 0})
    try:
        if args.command == 'prepare-requests':
            result = prepare_requests(root, cfg, out)
        elif args.command == 'compare-structure':
            result = compare_structure(root, cfg, args.footprints, out)
        elif args.command == 'mesh':
            from .raster import configure
            configure(out/'raster_scratch')
            from .mesh import build
            result = build(root/'height/completed', root/'global/final_rgb.png', out/'mesh', args.stride, args.texture_size)
        else:
            from .raster import configure
            configure(out/'raster_scratch')
            from . import rgb
            result = {'schema': 'citycode.background_replay.v1', 'lineage': cfg['lineage'], 'input_verification': result,
                      'gpu_jobs': 0, 'api_calls': 0, 'neural_height_inference_performed': False,
                      'neural_background_generation_performed': False}
            steps = [('observation_fusion', rgb.fuse_observations), ('v9_stitch', rgb.stitch_v9),
                     ('water_restore', rgb.restore_water), ('v10_stitch', rgb.stitch_v10), ('patches', rgb.export_patches)]
            for name, function in steps:
                print(f'STAGE {name}', flush=True)
                result[name] = function(root, cfg, out)
                write(out/'RESULT.json', result)
            shutil.copy2(local(root, cfg['grid']), out/'pixel_to_world.json')
            if not args.rgb_only:
                from .mesh import build
                result['mesh'] = build(root/'height/completed', out/'background_rgb.png', out/'mesh', args.stride, args.texture_size)
            result['status'] = 'passed'
        if (out/'raster_scratch').exists():
            shutil.rmtree(out/'raster_scratch')
        write(out/'RESULT.json', result)
        write(out/'RUN_STATUS.json', {'status': 'complete', 'command': args.command, 'api_calls': 0, 'gpu_jobs': 0})
        print({'status': 'complete', 'output': str(out), 'api_calls': 0})
    except Exception as exc:
        write(out/'RUN_STATUS.json', {'status': 'failed', 'command': args.command, 'error': str(exc), 'api_calls': 0})
        raise
