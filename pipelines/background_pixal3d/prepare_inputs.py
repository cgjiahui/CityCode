"""Windowed version of the historical world-aligned Pixal3D input preparation."""
import gc
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from citycode_background.raster import configure, pixels, release_pages


def prepare_inputs(args, teacher_manifest):
    Image.MAX_IMAGE_PIXELS = None  # Full-city PNG was hash-checked by the entry point.
    output = args.inputs_output
    output.mkdir(parents=True, exist_ok=False)
    configure(output.parent / '_rgb_scratch')
    rgb = pixels(args.handoff / '01_global/rgb/background_rgb.png')
    arrays = {name: np.load(args.teacher_output / name, mmap_mode='r')
              for name in teacher_manifest['outputs'].values() if name.endswith('.npy')}
    transform = json.loads((args.teacher_output / 'pixel_to_world.json').read_text())
    inverse = np.asarray(transform['ue_world_xy_to_pixel_center_index_3x3'], np.float64)
    index = json.loads((args.dataset / 'patch_index.json').read_text())
    selected = set(args.patch_id or [])
    records = []
    for item in index['patches']:
        if selected and item['patch_id'] not in selected:
            continue
        x0, y0, x1, y1 = item['bounds_xy_world_m']
        axis = np.arange(2560, dtype=np.float32)
        wx, wy = x0 + (axis + .5) * .15, y0 + (axis + .5) * .15
        mx = np.ascontiguousarray(np.broadcast_to(inverse[0, 0] * wx + inverse[0, 2], (2560, 2560)), dtype=np.float32)
        my = np.ascontiguousarray(np.broadcast_to((inverse[1, 1] * wy + inverse[1, 2])[:, None], (2560, 2560)), dtype=np.float32)
        # Integer translation preserves the original OpenCV interpolation table indices.
        left = max(0, int(np.floor(mx.min())) - 8)
        top = max(0, int(np.floor(my.min())) - 8)
        right = min(rgb.shape[1], int(np.ceil(mx.max())) + 9)
        bottom = min(rgb.shape[0], int(np.ceil(my.max())) + 9)
        mx -= left
        my -= top
        roi = np.s_[top:bottom, left:right]
        window = {k: np.asarray(v[roi]) for k, v in arrays.items()}
        source = np.ascontiguousarray(rgb[roi])
        patch_rgb = cv2.remap(source, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
        def nearest(a):
            return cv2.remap(a, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        def weighted(field, valid):
            numerator = cv2.remap(np.where(valid, field, 0).astype(np.float32), mx, my,
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            weight = cv2.remap(valid.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            result = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=weight > 1e-6)
            result[weight < .01] = np.nan
            return result, weight
        base_valid = np.isfinite(window['base_surface_z_m.npy'])
        elevated_valid = np.isfinite(window['elevated_surface_z_m.npy'])
        elevated_nearest = nearest(elevated_valid.astype(np.uint8)) > 0
        fields = {}
        for layer, valid in [('base', base_valid), ('elevated', elevated_valid)]:
            height, weight = weighted(window[layer + '_surface_z_m.npy'], valid)
            confidence, _ = weighted(window[layer + '_confidence.npy'], valid)
            if layer == 'elevated':
                height[~elevated_nearest] = np.nan
                confidence[~elevated_nearest] = 0
            elif not (np.isfinite(height) & (weight > .01)).all():
                raise ValueError(f"Invalid base pixels in {item['patch_id']}")
            fields[layer + '_surface_z_m'] = height.astype(np.float32)
            fields[layer + '_confidence'] = np.nan_to_num(confidence, nan=0).astype(np.float32)
            for suffix in ('confidence_tier', 'provenance'):
                fields[layer + '_' + suffix] = nearest(window[layer + '_' + suffix + '.npy']).astype(np.uint8)
        fields['base_semantic_id'] = nearest(window['base_semantic_id.npy']).astype(np.uint8)
        patch = output / item['patch_id']
        for name in ('rgb', 'height', 'masks'):
            (patch / name).mkdir(parents=True)
        image_path = patch / 'rgb/background_rgb_world_aligned_rgba.png'
        rgba = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2RGBA)
        Image.fromarray(rgba).save(image_path)
        height_path = patch / 'height/background_25d_filled.npz'
        np.savez_compressed(height_path, **fields)
        Image.fromarray(elevated_nearest.astype(np.uint8) * 255).save(patch / 'masks/elevated_valid_mask.png')
        Image.fromarray(np.clip(fields['base_confidence'] * 255, 0, 255).astype(np.uint8)).save(patch / 'masks/base_confidence.png')
        record = {**item, 'data_bounds_xy_world_m': item['bounds_xy_world_m'],
                  'core_bounds_xy_world_m': [x0 + 32, y0 + 32, x1 - 32, y1 - 32],
                  'input': str(image_path.resolve()), 'height_bundle': str(height_path.resolve()),
                  'completed_world_height': str(height_path.resolve()),
                  'input_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                  'ortho_meters_per_pixel': .15, 'resolution_px': [2560, 2560],
                  'base_valid_fraction': 1., 'elevated_valid_fraction': float(elevated_nearest.mean())}
        (patch / 'input_metadata.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print(json.dumps({'stage': 'patch_inputs', 'patch_id': item['patch_id']}), flush=True)
        for a in arrays.values():
            release_pages(a)
        release_pages(rgb)
        del source, window, fields, patch_rgb, rgba
        gc.collect()
    manifest = {'schema': 'air2vectorcity.pixal3d_p384_handoff_v9_inputs.v1', 'status': 'complete',
                'patch_count': len(records), 'teacher': str(args.teacher_output.resolve()), 'patches': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    del rgb
    import shutil
    shutil.rmtree(output.parent / '_rgb_scratch')
