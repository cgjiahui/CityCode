"""Replay frozen BEV fusion, V9 feathering, shoreline restoration and V10.

The image API boundary is supplied PNG files. This module contains no API client.
Historical GT-derived raster support is deliberately retained and disclosed.
"""
import math
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .io import local, read
from .raster import pixels, save, empty, release_pages

Image.MAX_IMAGE_PIXELS = None
R = Image.Resampling
# Bound OpenCV worker buffers on large native-resolution rasters.
cv2.setNumThreads(1)


def compare_images(actual, expected):
    with Image.open(actual) as a, Image.open(expected) as b:
        if a.size != b.size or a.mode != b.mode:
            raise ValueError(f'Image grid/mode mismatch: {actual}')
        size, mode = a.size, a.mode
    a, b = pixels(actual, mode), pixels(expected, mode)
    different = maximum = 0
    for y in range(0, size[1], 128):
        delta = np.abs(a[y:y+128].astype(np.int16) - b[y:y+128].astype(np.int16))
        different += int(np.count_nonzero(delta)); maximum = max(maximum, int(delta.max()))
        release_pages(a); release_pages(b)
    if different:
        raise ValueError(f'Pixel replay mismatch: {actual}: {different} channels, max error {maximum}')
    return {'pixel_exact': True, 'size': list(size), 'mode': mode}


def fuse_observations(root, cfg, output):
    transform = read(local(root, cfg['grid']))
    width, height = transform['image_size_px']
    mpp = transform['meters_per_pixel']
    xmin, ymin, _, _ = transform['raster_edge_bounds_xy_world_m']
    inverse = np.asarray(transform['ue_world_xy_to_pixel_center_index_3x3'])
    rgb = empty('fusion_rgb.raw', (height, width, 3), np.uint8, 0)
    best = empty('fusion_best.raw', (height, width), np.float32, np.inf)
    for i, patch in enumerate(cfg['patches'], 1):
        files = patch['files']
        rgba = pixels(local(root, files['raw_bev_rgba']), 'RGBA')
        mask = pixels(local(root, files['building_union_mask']), 'L') > 0
        edited = rgba[..., :3].copy()
        edited[rgba[..., 3] == 0] = 255
        edited[mask] = 0
        if not np.array_equal(edited, pixels(local(root, files['input_rgb']))):
            raise ValueError(f'Patch edit input mismatch: {patch["patch_id"]}')
        t = read(local(root, files['pixel_to_world']))
        matrix = inverse @ np.asarray(t['pixel_center_index_to_ue_world_xy_3x3'])
        h, w = rgba.shape[:2]
        corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]) @ matrix.T
        corners = corners[:, :2] / corners[:, 2:]
        x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int)-3, 0)
        x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int)+4, [width, height])
        if x1 <= x0 or y1 <= y0:
            continue
        matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ matrix
        warped = cv2.warpPerspective(rgba[..., :3], matrix, (int(x1-x0), int(y1-y0)), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective((rgba[..., 3] > 0).astype(np.uint8), matrix,
                                   (int(x1-x0), int(y1-y0)), flags=cv2.INTER_NEAREST) > 0
        xs = xmin + (np.arange(x0, x1, dtype=np.float32)+.5)*mpp
        ys = ymin + (np.arange(y0, y1, dtype=np.float32)+.5)*mpp
        center = patch['center_world_m']
        score = (ys[:, None]-float(center[1]))**2 + (xs[None, :]-float(center[0]))**2
        target = best[y0:y1, x0:x1]
        take = valid & (score < target)
        rgb[y0:y1, x0:x1][take] = warped[take]
        target[take] = score[take]
        del rgba, edited, mask, warped, valid, score, take, target
        release_pages(rgb); release_pages(best)
        print(f'Fuse observations {i}/{len(cfg["patches"])}', flush=True)
    expected_missing = pixels(local(root, 'global/unobserved_mask.png'), 'L')
    building = pixels(local(root, 'global/building_union_mask.png'), 'L')
    for y in range(0, height, 128):
        missing = ~np.isfinite(best[y:y+128])
        if not np.array_equal(missing, expected_missing[y:y+128] > 0):
            raise ValueError('Global observation coverage differs from frozen input')
        rgb[y:y+128][missing] = 255
        rgb[y:y+128][building[y:y+128] > 0] = 0
        for array in [rgb, best, expected_missing, building]: release_pages(array)
    save(output / 'input_rgb.png', rgb)
    del rgb, best, missing, expected_missing, building
    return compare_images(output/'input_rgb.png', local(root, 'global/input_rgb.png'))


def feather(tile):
    x0, y0, x1, y1 = tile['pixel_bounds_xyxy']
    wx, wy = np.ones(x1-x0, np.float32), np.ones(y1-y0, np.float32)
    for side, v in [('left', wx), ('right', wx), ('top', wy), ('bottom', wy)]:
        n = int(tile['neighbor_overlaps_px'].get(side, 0))
        if n < 0 or n > len(v):
            raise ValueError('Invalid overlap')
        if n:
            if side in ['left', 'top']:
                v[:n] = np.linspace(0, 1, n, dtype=np.float32)
            else:
                v[-n:] = np.minimum(v[-n:], np.linspace(1, 0, n, dtype=np.float32))
    return wy[:, None] * wx[None, :]


def accumulate(root, cfg, output, corrected, name, base=None):
    width, height = read(local(root, cfg['grid']))['image_size_px']
    scratch = output / (name + '.scratch')
    scratch.mkdir()
    total = np.memmap(scratch/'sum.bin', mode='w+', dtype='float32', shape=(height, width, 3))
    weights = np.memmap(scratch/'weights.bin', mode='w+', dtype='float32', shape=(height, width))
    result = np.memmap(scratch/'rgb.bin', mode='w+', dtype='uint8', shape=(height, width, 3))
    try:
        for y in range(0, height, 128):
            total[y:y+128] = 0; weights[y:y+128] = 0
            release_pages(total); release_pages(weights)
        for tile, rgb in corrected:
            x0, y0, x1, y1 = map(int, tile['pixel_bounds_xyxy'])
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or rgb.shape != (y1-y0, x1-x0, 3):
                raise ValueError('Invalid tile image/bounds')
            w = feather(tile)
            total[y0:y1, x0:x1] += rgb.astype(np.float32) * w[..., None]
            weights[y0:y1, x0:x1] += w
            release_pages(total); release_pages(weights)
        for y in range(0, height, 256):
            sl = slice(y, min(y+256, height)); w = weights[sl]
            if base is None:
                if np.any(w <= 0):
                    raise ValueError('Uncovered mosaic pixels')
                mixed = total[sl] / w[..., None]
            else:
                strip = np.asarray(base[sl], dtype=np.float32)
                valid = w > 0
                generated = np.zeros_like(strip)
                generated[valid] = total[sl][valid] / w[valid, None]
                alpha = np.minimum(w, 1)
                mixed = strip * (1-alpha[..., None]) + generated * alpha[..., None]
            result[sl] = np.round(mixed).clip(0, 255).astype(np.uint8)
            for array in [total, weights, result, base]: release_pages(array)
        result.flush()
        save(output / name, result)
    finally:
        del total, weights, result
        shutil.rmtree(scratch)


def stitch_v9(root, cfg, output):
    # Check the actual API inputs against the frozen full-city input, not filenames.
    original = pixels(local(root, 'global/input_rgb.png'))
    for tile in cfg['tiles']:
        x0,y0,x1,y1 = tile['pixel_bounds_xyxy']
        crop = original[y0:y1,x0:x1]
        if not np.array_equal(crop, pixels(local(root, tile['files']['input_rgb']))):
            raise ValueError(f'Tile input mismatch: {tile["tile_id"]}')
        release_pages(original)
    del original, crop
    accumulate(root, cfg, output, ((t, pixels(local(root, t['v9_result']))) for t in cfg['tiles']), 'v9_rgb.png')
    return compare_images(output/'v9_rgb.png', local(root, 'global/v9_rgb.png'))


def restore_water(root, cfg, output):
    """Replay frozen water pixels, preserving newly recomputed V9 outside mask.

The standard memory-bounded profile treats the historical procedural water
treatment as cached input; it does not claim to rerun its random palette recipe.
    """
    source = pixels(output/'v9_rgb.png')
    cached = pixels(local(root, 'global/v9_water_restored.png'))
    mask = pixels(local(root, 'global/outside_water_mask.png'), 'L')
    result = empty('water_restored.raw', source.shape, np.uint8)
    for y in range(0, source.shape[0], 128):
        chosen = mask[y:y+128] > 0
        if not np.array_equal(source[y:y+128][~chosen], cached[y:y+128][~chosen]):
            raise ValueError('Frozen water treatment changed protected V9 pixels')
        result[y:y+128] = source[y:y+128]
        result[y:y+128][chosen] = cached[y:y+128][chosen]
        for array in [source, cached, mask, result]: release_pages(array)
    save(output/'v9_water_restored.png', result)
    report = compare_images(output/'v9_water_restored.png', local(root, 'global/v9_water_restored.png'))
    report['method'] = 'frozen water pixels; recomputed V9 preserved outside water mask'
    report['procedural_water_regenerated'] = False
    return report


def stitch_v10(root, cfg, output):
    by_id = {t['tile_id']: t for t in cfg['tiles']}
    size = tuple(read(local(root, cfg['grid']))['image_size_px'])
    base = pixels(output/'v9_water_restored.png')
    original = pixels(local(root, 'global/input_rgb.png'))
    outside = pixels(local(root, 'global/outside_water_mask.png'), 'L')
    with Image.open(local(root, 'historical_reference/ortho_rgba.png')) as g, \
         Image.open(local(root, 'historical_reference/water_rgb.png')) as water:
        gt = g.convert('RGBA')
        def tiles():
            for tid in cfg['selected_ids']:
                tile = by_id[tid]; bounds = tile['pixel_bounds_xyxy']; x0, y0, x1, y1 = bounds
                fallback = base[y0:y1,x0:x1]; orig = original[y0:y1,x0:x1]
                rgba = np.asarray(gt.crop((x0/size[0]*gt.width, y0/size[1]*gt.height,
                                          x1/size[0]*gt.width, y1/size[1]*gt.height)).resize((x1-x0, y1-y0), R.LANCZOS))
                alpha = rgba[..., 3].astype(np.float32)/255
                ref = np.round(rgba[..., :3]*alpha[..., None]+fallback*(1-alpha[..., None])).astype(np.uint8)
                water_proxy = (rgba[..., 3] > 0) & (rgba[..., :3].mean(2) < 90) & (rgba[..., 2].astype(np.int16)-rgba[..., 0].astype(np.int16) >= 5)
                building = pixels(local(root, tile['files']['building_mask']), 'L') > 0
                missing = pixels(local(root, tile['files']['unobserved_mask']), 'L') > 0
                protect = (water_proxy & ~building) | (outside[y0:y1,x0:x1] > 0)
                generated = pixels(local(root, tile['v10_result'])).copy()
                generated[protect & ~missing] = orig[protect & ~missing]
                generated[protect & missing] = ref[protect & missing]
                yield tile, generated
                for array in [base, original, outside]: release_pages(array)
            ops = [None, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.ROTATE_180]
            for i, tid in enumerate(cfg['water_tile_ids']):
                t = by_id[tid]; x0, y0, x1, y1 = t['pixel_bounds_xyxy']
                image = water.crop((0, 0, 1000, 1000)).resize((x1-x0, y1-y0), R.LANCZOS)
                if ops[i % 4] is not None:
                    image = image.transpose(ops[i % 4])
                yield t, np.asarray(image.convert('RGB'))
        accumulate(root, cfg, output, tiles(), 'background_rgb.png', base)
    del base, original, outside, gt
    return compare_images(output/'background_rgb.png', local(root, 'global/final_rgb.png'))


def export_patches(root, cfg, output):
    full = pixels(output/'background_rgb.png')
    inverse = np.asarray(read(local(root, cfg['grid']))['ue_world_xy_to_pixel_center_index_3x3'])
    records = []
    for i, patch in enumerate(cfg['patches'], 1):
        t = read(local(root, patch['files']['pixel_to_world']))
        w, h = t['output_size_px']; matrix = inverse @ np.asarray(t['pixel_center_index_to_ue_world_xy_3x3'])
        corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]) @ matrix.T
        x0, y0 = np.maximum(np.floor(corners[:, :2].min(0)).astype(int)-6, 0)
        x1, y1 = np.minimum(np.ceil(corners[:, :2].max(0)).astype(int)+7, [full.shape[1], full.shape[0]])
        matrix = matrix[:2].copy(); matrix[:, 2] -= [x0, y0]
        rgb = cv2.warpAffine(np.ascontiguousarray(full[y0:y1, x0:x1]), matrix, (w, h),
                             flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        folder = output/'patches'/patch['patch_id']; folder.mkdir(parents=True)
        save(folder/'rgb.png', rgb)
        shutil.copy2(local(root, patch['files']['pixel_to_world']), folder/'pixel_to_world.json')
        check = compare_images(folder/'rgb.png', local(root, patch['files']['repaired_rgb']))
        records.append({'patch_id': patch['patch_id'], **check})
        release_pages(full)
        print(f'Export repaired patches {i}/{len(cfg["patches"])}', flush=True)
    return records
