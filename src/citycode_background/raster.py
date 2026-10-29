"""Memory-bounded PNG I/O backed by files in the explicit output directory.

Requires a C compiler and libpng development headers on Linux. The helper only
encodes/decodes user-selected local PNGs; it contains no subprocess/network API.
"""
import ctypes
import hashlib
import mmap
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

_CACHE = None
_LIB = None


def configure(path):
    global _CACHE, _LIB
    _CACHE = Path(path); _CACHE.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name('_png_stream.c')
    library = _CACHE/'png_stream.so'
    subprocess.run(['cc', '-O2', '-shared', '-fPIC', str(source), '-lpng', '-o', str(library)], check=True)
    _LIB = ctypes.CDLL(str(library))
    for name in ['bg_png_read', 'bg_png_write']:
        fn = getattr(_LIB, name)
        fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int


def release_pages(array):
    if isinstance(array, np.memmap):
        array.flush()
        if hasattr(array._mmap, 'madvise'):
            array._mmap.madvise(mmap.MADV_DONTNEED)


def empty(name, shape, dtype, fill=None):
    if _CACHE is None:
        return np.full(shape, fill, dtype=dtype) if fill is not None else np.empty(shape, dtype=dtype)
    p = _CACHE/name
    a = np.memmap(p, mode='w+', dtype=dtype, shape=shape)
    if fill is not None:
        for y in range(0, shape[0], 128):
            a[y:y+128] = fill
            release_pages(a)
    return a


def pixels(path, mode='RGB'):
    path = Path(path)
    if _CACHE is None:
        with Image.open(path) as im:
            return np.array(im.convert(mode))
    with Image.open(path) as im:
        width, height = im.size
        if im.format != 'PNG' or mode not in ['L', 'RGB', 'RGBA']:
            raise ValueError('Mapped raster reader expects PNG and L/RGB/RGBA')
    channels = {'L': 1, 'RGB': 3, 'RGBA': 4}[mode]
    key = hashlib.sha256(f'{path.resolve()}:{path.stat().st_mtime_ns}:{path.stat().st_size}:{mode}'.encode()).hexdigest()
    raw = _CACHE/(key+'.raw'); shape = (height,width) if channels == 1 else (height,width,channels)
    if not raw.exists():
        out = np.memmap(raw, mode='w+', dtype='uint8', shape=shape)
        status = _LIB.bg_png_read(str(path).encode(), out.ctypes.data, width, height, channels)
        if status:
            del out; raw.unlink(missing_ok=True)
            raise ValueError(f'PNG decode failed ({status}): {path}')
        release_pages(out); del out
    return np.memmap(raw, mode='r', dtype='uint8', shape=shape)


def save(path, array):
    a = np.asarray(array)
    if _LIB is None:
        Image.fromarray(a).save(path, compress_level=1)
        return
    if a.dtype != np.uint8 or not a.flags.c_contiguous or a.ndim not in [2,3]:
        raise ValueError('PNG encoder needs contiguous uint8 image')
    h,w = a.shape[:2]; channels = 1 if a.ndim == 2 else a.shape[2]
    if channels not in [1,3,4]:
        raise ValueError('Unsupported channel count')
    status = _LIB.bg_png_write(str(path).encode(), a.ctypes.data, w, h, channels)
    if status:
        raise ValueError(f'PNG encode failed ({status}): {path}')
    release_pages(array)
