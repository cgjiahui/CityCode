"""Metric vertical intersections of actual placed GLB triangles, on CPU."""
from pathlib import Path
import ctypes as C
import hashlib
import subprocess

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import make_valid


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Raster:
    def __init__(self, workdir):
        source = Path(__file__).with_name('_raster.cpp')
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        library = workdir / ('raster_' + sha256(source)[:16] + '.so')
        if not library.exists():
            subprocess.run(['c++', '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
        lib = self.lib = C.CDLL(str(library))
        lib.raster_create.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_int32), C.c_int64, C.c_double, C.c_double, C.c_double, C.c_int, C.c_int]
        lib.raster_create.restype = C.c_void_p
        for name in ['raster_hits', 'raster_lo', 'raster_hi']:
            getattr(lib, name).argtypes = [C.c_void_p]
            getattr(lib, name).restype = C.POINTER(C.c_float)
        lib.raster_offsets.argtypes = [C.c_void_p]
        lib.raster_offsets.restype = C.POINTER(C.c_int64)
        lib.raster_hit_count.argtypes = [C.c_void_p]
        lib.raster_hit_count.restype = C.c_int64
        lib.raster_destroy.argtypes = [C.c_void_p]

    def cast(self, vertices, faces, cell=.25):
        v = np.ascontiguousarray(vertices, dtype=np.float64)
        f = np.ascontiguousarray(faces, dtype=np.int32)
        if not np.isfinite(v).all() or not len(v) or not len(f):
            raise ValueError('Finite nonempty actual mesh is required')
        lo = np.floor(v[:, :2].min(0) / cell) * cell - cell
        dims = np.ceil((v[:, :2].max(0) - lo) / cell).astype(int) + 1
        w, h = map(int, dims)
        if min(w, h) <= 0 or max(w, h) > 8192:
            raise ValueError('Mesh raster exceeds the configured metric extent')
        handle = self.lib.raster_create(v.ctypes.data_as(C.POINTER(C.c_double)), f.ctypes.data_as(C.POINTER(C.c_int32)), len(f), *lo, float(cell), w, h)
        try:
            count = self.lib.raster_hit_count(handle)
            arrays = {key: np.ctypeslib.as_array(getattr(self.lib, name)(handle), shape=(n,)).copy()
                      for key, name, n in [('zmin', 'raster_lo', w*h), ('zmax', 'raster_hi', w*h), ('offsets', 'raster_offsets', w*h+1), ('hits', 'raster_hits', count)]}
            for key in ['zmin', 'zmax']:
                arrays[key] = arrays[key].reshape(h, w)
            arrays.update(origin=lo, cell=np.array(cell), shape=np.array([h, w], np.int32))
            return arrays
        finally:
            self.lib.raster_destroy(handle)


def footprint(mask, origin, cell):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] >= 0 or len(contour) < 3:
                continue
            holes = []
            child = int(hierarchy[0, i, 2])
            while child >= 0:
                if len(contours[child]) >= 3:
                    holes.append((contours[child][:, 0].astype(float) + .5) * cell + origin)
                child = int(hierarchy[0, child, 0])
            parts.append(make_valid(Polygon((contour[:, 0].astype(float) + .5) * cell + origin, holes)))
    return make_valid(unary_union(parts)) if parts else Polygon()
