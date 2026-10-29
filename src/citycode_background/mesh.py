"""CPU height surfaces, separate layers, no triangles across invalid pixels.

This is not neural/Pixal3D regeneration, a closed bridge volume, or a pier model.
"""
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from .io import read, write, sha256
from .raster import pixels, release_pages
import cv2


def surface(z, transform, stride=16, texture=None):
    if stride < 1 or z.ndim != 2 or min(z.shape) < 2:
        raise ValueError('Expected a 2D height grid and positive stride')
    matrix = np.asarray(transform['pixel_center_index_to_ue_world_xy_3x3'], dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError('Invalid world transform')
    if not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError('Only affine XY grids are supported')
    ys = np.unique(np.r_[np.arange(0, z.shape[0], stride), z.shape[0]-1])
    xs = np.unique(np.r_[np.arange(0, z.shape[1], stride), z.shape[1]-1])
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)]) @ matrix.T
    vertices = np.column_stack([xy[:, :2], z[np.ix_(ys, xs)].ravel()])
    index = np.arange(xx.size).reshape(xx.shape)
    valid_cells = np.zeros((len(ys)-1, len(xs)-1), dtype=bool)
    # Row-band sums avoid a full-resolution int64 integral image (>1 GiB).
    for i, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
        columns = (~np.isfinite(z[y0:y1+1])).sum(axis=0, dtype=np.int64)
        prefix = np.r_[0, columns.cumsum()]
        valid_cells[i] = prefix[xs[1:]+1] - prefix[xs[:-1]] == 0
    a = index[:-1, :-1][valid_cells]; b = index[:-1, 1:][valid_cells]
    c = index[1:, :-1][valid_cells]; d = index[1:, 1:][valid_cells]
    faces = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    if np.linalg.det(matrix[:2, :2]) < 0:
        faces = faces[:, ::-1]
    used = np.unique(faces); mapping = np.full(len(vertices), -1, dtype=np.int64); mapping[used] = np.arange(len(used))
    mesh = trimesh.Trimesh(vertices=vertices[used], faces=mapping[faces], process=False)
    if texture is not None and len(used):
        uv = np.column_stack([(xx.ravel()[used]+.5)/z.shape[1], 1-(yy.ravel()[used]+.5)/z.shape[0]])
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture)
    return mesh


def build(depth_root, rgb_path, output, stride=16, texture_size=4096):
    if stride < 1 or texture_size < 1:
        raise ValueError('Stride and texture size must be positive')
    depth_root, output = Path(depth_root), Path(output)
    transform = read(depth_root/'pixel_to_world.json')
    Image.MAX_IMAGE_PIXELS = None
    full = pixels(rgb_path)
    if list(full.shape[1::-1]) != transform['image_size_px']:
        raise ValueError('RGB and height grids differ')
    scale = min(1., texture_size / max(full.shape[:2]))
    size = (max(1, round(full.shape[1]*scale)), max(1, round(full.shape[0]*scale)))
    texture = Image.fromarray(cv2.resize(full, size, interpolation=cv2.INTER_AREA))
    release_pages(full); del full
    meshes = {}; counts = {}
    for layer in ['base', 'elevated']:
        path = depth_root/f'{layer}_surface_z_m.npy'; z = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(z.shape[::-1]) != transform['image_size_px']:
            raise ValueError('Height and transform grids differ')
        meshes[layer] = surface(z, transform, stride, texture)
        counts[layer] = {'vertices': len(meshes[layer].vertices), 'faces': len(meshes[layer].faces),
                         'valid_pixels': sum(int(np.isfinite(z[y:y+256]).sum()) for y in range(0, len(z), 256)),
                         'input_sha256': sha256(path)}
    if not any(m.faces.size for m in meshes.values()):
        raise ValueError('No valid surface cells')
    output.mkdir(parents=True, exist_ok=False)
    scene = trimesh.Scene()
    for name, mesh in meshes.items():
        if mesh.faces.size:
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    target = output/'background_ue_z_up.glb'; scene.export(target)
    restored = trimesh.load(target, force='scene', process=False)
    if sum(len(m.faces) for m in restored.geometry.values()) != sum(v['faces'] for v in counts.values()):
        raise ValueError('GLB roundtrip face count changed')
    if not all(np.isfinite(m.vertices).all() for m in restored.geometry.values()):
        raise ValueError('Nonfinite exported vertices')
    report = {'schema': 'citycode.background_surface.v1', 'status': 'passed', 'lineage': 'historical_gt_assisted',
              'method': 'CPU triangulation of frozen predicted heights with historical support masks',
              'coordinate_system': 'UE XYZ metres, Z up; no second placement transform',
              'layers': counts, 'stride_pixels': stride, 'nominal_spacing_m': stride*transform['meters_per_pixel'],
              'texture_size_px': list(texture.size), 'texture_source_sha256': sha256(rgb_path),
              'closed_volume': False, 'piers_reconstructed': False, 'layers_remain_separate': True,
              'all_pixels_in_each_coarse_face_cell_must_be_finite': True,
              'glb_sha256': sha256(target), 'gpu_jobs': 0, 'api_calls': 0}
    write(output/'manifest.json', report)
    return report
