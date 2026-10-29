"""Prediction-only perspective evidence. Unknown or occluded is never background.

This module does not import evaluators or load reference geometry. Its functions
consume explicit camera arrays, cached predicted depth, V9.2 instance labels,
V7B background classes and the input strict-valid mask. No scene/object lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def project_points(xyz, camera):
    """Project world metres into image boundary coordinates (pixel i=[i,i+1))."""
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected N x 3 world points')
    transform = np.asarray(camera['world_to_camera_4x4_m'], dtype=float)
    intrinsic = np.asarray(camera['intrinsics']['K'], dtype=float)
    if transform.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError('Invalid camera matrix shape')
    if not np.isfinite(transform).all() or not np.isfinite(intrinsic).all():
        raise ValueError('Camera matrices must be finite')
    coordinates = np.c_[points, np.ones(len(points))] @ transform.T
    homogeneous = coordinates[:, :3] @ intrinsic.T
    uv = np.full((len(points), 2), np.nan)
    valid = np.isfinite(coordinates).all(axis=1) & (coordinates[:, 2] > 0)
    np.divide(homogeneous[:, :2], homogeneous[:, 2, None], out=uv,
              where=valid[:, None])
    return uv, coordinates[:, 2], valid


@dataclass
class ViewEvidence:
    camera: dict
    instance_ids: np.ndarray
    background_ids: np.ndarray
    depth_zc: np.ndarray
    strict_valid: np.ndarray
    background_core: np.ndarray

    @classmethod
    def from_arrays(cls, camera, instance_ids, background_ids, depth_zc,
                    strict_valid, background_margin_px=3, background_policy="v7b_agreement"):
        arrays = [np.asarray(x) for x in
                  (instance_ids, background_ids, depth_zc, strict_valid)]
        ids, bg, depth, valid = arrays
        if any(a.ndim != 2 or a.shape != ids.shape for a in arrays):
            raise ValueError('All evidence arrays must have the same native H x W shape')
        w, h = camera['image_size_px']
        if ids.shape != (h, w):
            raise ValueError('Evidence resolution differs from camera')
        if background_margin_px < 0 or int(background_margin_px) != background_margin_px:
            raise ValueError('Background erosion must be a nonnegative integer')
        # V7B 0 is unspecified black, not negative evidence. Only explicit
        # road/water/hardscape/elevated-road predictions outside all V9.2 roofs.
        if background_policy not in ('v7b_agreement', 'v92_explicit'):
            raise ValueError('Unknown semantic background policy')
        # V9.2 prompt explicitly defines black/road as nonbuilding. Its native
        # normalizer preserves all building-coloured pixels, so label zero in
        # the strict ROI is an explicit segmentation class, not missing XYZ.
        explicit = (ids == 0) & valid.astype(bool)
        if background_policy == 'v7b_agreement':
            explicit &= np.isin(bg, (1, 2, 3, 4))
        core = (binary_erosion(explicit, iterations=int(background_margin_px),
                               border_value=0)
                if background_margin_px else explicit)
        return cls(camera, ids, bg, depth, valid.astype(bool), core)

    @classmethod
    def load(cls, paths, background_margin_px=3, background_policy="v7b_agreement"):
        camera = json.loads(Path(paths['camera']).read_text())
        arrays = [np.asarray(Image.open(paths[name])) for name in
                  ('source_ids', 'background_ids', 'strict_valid')]
        depth = np.load(paths['predicted_depth'], mmap_mode='r', allow_pickle=False)
        return cls.from_arrays(camera, arrays[0], arrays[1], depth, arrays[2],
                               background_margin_px, background_policy)

    def classify(self, xyz, owner_instance_ids=(), depth_tolerance_m=3.0,
                 image_margin_px=2):
        if (not isinstance(depth_tolerance_m, (int, float))
                or not isinstance(image_margin_px, (int, float))
                or not np.isfinite(depth_tolerance_m) or not np.isfinite(image_margin_px)
                or depth_tolerance_m < 0 or image_margin_px < 0):
            raise ValueError('Tolerances must be finite and nonnegative')
        points = np.asarray(xyz, dtype=float)
        uv, zc, front = project_points(points, self.camera)
        h, w = self.instance_ids.shape
        valid = front & np.isfinite(uv).all(axis=1)
        valid &= ((uv[:, 0] >= image_margin_px) & (uv[:, 0] < w-image_margin_px)
                  & (uv[:, 1] >= image_margin_px) & (uv[:, 1] < h-image_margin_px))
        # Rendering outside the patch prism cannot establish negative evidence.
        bounds = self.camera.get('data_bounds_xy_world_m')
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            valid &= ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                      & (points[:, 1] >= y0) & (points[:, 1] <= y1))
        pixels = np.zeros((len(points), 2), dtype=np.int64)
        pixels[valid] = np.floor(uv[valid]).astype(np.int64)
        selected = np.flatnonzero(valid)
        observed_depth = np.full(len(points), np.nan)
        ids = np.zeros(len(points), dtype=self.instance_ids.dtype)
        explicit = np.zeros(len(points), dtype=bool)
        if len(selected):
            u, v = pixels[selected].T
            valid[selected] &= self.strict_valid[v, u]
            observed_depth[selected] = self.depth_zc[v, u]
            ids[selected] = self.instance_ids[v, u]
            explicit[selected] = self.background_core[v, u]
        valid &= np.isfinite(observed_depth) & (observed_depth > 0)
        occluded = valid & (zc > observed_depth + depth_tolerance_m)
        visible = valid & ~occluded
        same_instance = visible & np.isin(ids, list(owner_instance_ids)) & (ids > 0)
        depth_agrees = valid & (np.abs(zc-observed_depth) <= depth_tolerance_m)
        positive_surface = same_instance & depth_agrees
        explicit_background = visible & explicit
        other_instance = visible & (ids > 0) & ~np.isin(ids, list(owner_instance_ids))
        # Merely lacking a same-instance hit gives no negative evidence.
        unknown = ~(positive_surface | explicit_background | other_instance)
        return {'projected_uv': uv, 'sampled_instance_ids': ids,
                'depth_agrees': depth_agrees, 'visible': visible,
                'camera_depth_m': zc,
                'sampled_depth_m': observed_depth, 'valid_depth_and_roi': valid,
                'occluded': occluded, 'positive_surface': positive_surface,
                'same_instance_silhouette': same_instance,
                'explicit_background': explicit_background,
                'other_visible_instance': other_instance, 'unknown': unknown}
