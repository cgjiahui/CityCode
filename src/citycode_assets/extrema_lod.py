"""Finite CPU fallback for a generated low LOD that loses roof/base extrema.

An actual accepted high GLB supplies every position, normal, UV and image value.
The pinned MIT meshoptimizer backend returns indices only, with physical extrema
and their incident faces locked. The existing placement is never adjusted.

This entry supports the static single-primitive/single-instance GLBs produced by
the neural exporter. It preserves nonidentity glTF node transforms. Unsupported
deformations/instances are rejected, never flattened or silently duplicated.
Node.js with CPU WebAssembly is required; no network or GPU is used. A successful
candidate still needs the delivery pipeline's independent final QA and binding.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import shape

from .lod import _accessor, _read_glb, inspect_glb, sha256
from .registration import FitPolicy, overlap_metrics, raster_triangle_footprint, read_glb_scene_geometry

_HERE = Path(__file__).resolve().parent
_DEPENDENCY = _HERE / 'vendor/meshoptimizer_v025'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dependency_proof() -> dict[str, Any]:
    snapshot_path = _DEPENDENCY / 'SNAPSHOT.json'
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot.get('version') != '0.25.0' or snapshot.get('license') != 'MIT':
        raise ValueError('Unexpected CPU simplification dependency version/license')
    for record in snapshot['files']:
        path = _DEPENDENCY / record['path']
        if path.parent != _DEPENDENCY or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('CPU simplification dependency changed from packaged snapshot')
    return {'version': snapshot['version'], 'license': snapshot['license'], 'upstream': snapshot['upstream'],
            'snapshot_sha256': sha256(snapshot_path), 'files': snapshot['files']}


def _face_set(faces: np.ndarray) -> set[tuple[int, int, int]]:
    """Compare oriented triangles while allowing cyclic vertex order."""
    shifts = faces.argmin(axis=1)
    reordered = faces[np.arange(len(faces))[:, None], (np.arange(3)[None, :] + shifts[:, None]) % 3]
    return {tuple(row) for row in reordered.tolist()}


def extrema_lock_mask(positions, faces, world_from_vertices, tolerance_m):
    """Lock exact-position UV duplicates, two incident rings and each component.

    The world transform determines vertical extrema; no Y-up assumption is used.
    All supplied vertices must be used. The result records physical components so
    callers can reject a candidate that loses even a small disconnected surface.
    """
    positions = np.asarray(positions)
    faces = np.asarray(faces, dtype=np.int64)
    matrix = np.asarray(world_from_vertices, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('Finite XYZ positions required')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(positions):
        raise ValueError('Valid triangle indices required')
    if len(np.unique(faces)) != len(positions):
        raise ValueError('Lock-mask input must contain only used vertices')
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError('Finite invertible world placement required')
    if not np.allclose(matrix[3], [0, 0, 0, 1]) or not 0 < tolerance_m <= .20:
        raise ValueError('Affine placement and original vertical cap required')
    unique, inverse = np.unique(positions, axis=0, return_inverse=True)
    physical_faces = inverse[faces]
    edges = np.concatenate([physical_faces[:, [0, 1]], physical_faces[:, [1, 2]], physical_faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(len(unique), len(unique)))
    count, components = connected_components(graph, directed=False)
    heights = positions @ matrix[2, :3] + matrix[2, 3]
    patch = (heights <= heights.min() + tolerance_m) | (heights >= heights.max() - tolerance_m)
    for _ in range(2):
        physical = np.zeros(len(unique), dtype=bool);physical[inverse[patch]] = True
        patch = physical[inverse]
        patch[faces[np.any(patch[faces], axis=1)].ravel()] = True
    _, first_face = np.unique(components[physical_faces[:, 0]], return_index=True)
    patch[faces[first_face].ravel()] = True
    physical = np.zeros(len(unique), dtype=bool);physical[inverse[patch]] = True
    locks = physical[inverse].astype('u1')
    return locks, {'physical_component_count': int(count), 'physical_component_by_vertex': components[inverse],
                   'protected_faces': faces[np.all(locks[faces] != 0, axis=1)],
                   'world_extrema_m': [float(heights.min()), float(heights.max())]}


def _pack_glb(document, binary):
    encoded = json.dumps(document, separators=(',', ':'), ensure_ascii=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\0' * (-len(binary) % 4)
    body = struct.pack('<I4s', len(encoded), b'JSON') + encoded + struct.pack('<I4s', len(binary), b'BIN\0') + binary
    return struct.pack('<4sII', b'glTF', 2, len(body) + 12) + body


def _repack_original_attributes(data, document, binary_offset, arrays, source_primitive, indices):
    used, compact = np.unique(indices, return_inverse=True)
    output = copy.deepcopy(document)
    output['accessors'] = [];output['bufferViews'] = []
    payload = bytearray()
    def add_blob(blob, target=None):
        payload.extend(b'\0' * (-len(payload) % 4))
        view = {'buffer': 0, 'byteOffset': len(payload), 'byteLength': len(blob)}
        if target is not None: view['target'] = target
        index = len(output['bufferViews']);output['bufferViews'].append(view);payload.extend(blob)
        return index
    def add_accessor(array, original, target):
        record = copy.deepcopy(document['accessors'][original]);record.pop('byteOffset', None)
        record.update(bufferView=add_blob(array.tobytes(), target), count=len(array),
                      min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        index = len(output['accessors']);output['accessors'].append(record)
        return index
    primitive = output['meshes'][0]['primitives'][0]
    primitive['attributes'] = {name: add_accessor(array[used], source_primitive['attributes'][name], 34962) for name, array in arrays.items()}
    primitive['indices'] = add_accessor(compact.astype('<u4').reshape(-1, 1), source_primitive['indices'], 34963)
    # Source indices may be UNSIGNED_SHORT; the new buffer is always UINT32.
    output['accessors'][primitive['indices']]['componentType'] = 5125
    images = []
    for image, destination in zip(document.get('images', []), output.get('images', [])):
        if 'bufferView' not in image or 'uri' in image:
            raise ValueError('Embedded original image bytes are required')
        view = document['bufferViews'][image['bufferView']]
        offset = binary_offset + view.get('byteOffset', 0)
        blob = bytes(data[offset:offset + view['byteLength']])
        destination['bufferView'] = add_blob(blob)
        images.append({'sha256': _digest(blob), 'bytes': len(blob)})
    output['buffers'] = [{'byteLength': len(payload)}]
    return _pack_glb(output, bytes(payload)), used, images


def _validate_geometry(path, fit, target_faces, *, visual_completion=False):
    policy = FitPolicy();info = inspect_glb(path)
    scene = read_glb_scene_geometry(path)
    vertices, faces = scene['vertices'], scene['faces']
    matrix = np.asarray(fit['world_from_asset_4x4'], dtype=float)
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    target = shape(fit['target_geometry'])
    footprint, coarse = raster_triangle_footprint(vertices, faces, matrix)
    metrics = overlap_metrics(footprint, target)
    thresholds = {'iou': policy.minimum_footprint_iou, 'target_coverage': policy.minimum_target_coverage,
                  'asset_coverage': policy.minimum_asset_coverage}
    if visual_completion:
        thresholds = {'target_coverage': .20, 'asset_coverage': .20}
    fine = None
    if any(abs(metrics[name] - threshold) <= .02 for name, threshold in thresholds.items()):
        footprint, fine = raster_triangle_footprint(vertices, faces, matrix, cell_m=.05, max_dimension=4096)
        metrics = overlap_metrics(footprint, target)
    tolerance = min(.20, max(.05, .01 * (fit['predicted_top_z_m'] - fit['predicted_base_z_m'])))
    ground = float(world[:, 2].min() - fit['predicted_base_z_m'])
    top = float(world[:, 2].max() - fit['predicted_top_z_m'])
    errors = [name for name, threshold in thresholds.items() if metrics[name] < threshold]
    if abs(ground) > tolerance: errors.append('grounding')
    if abs(top) > tolerance: errors.append('top_height')
    if not 8 < info['faces'] <= 1.02 * target_faces: errors.append('face_budget')
    data, document, binary = _read_glb(path);textures = []
    for image in document.get('images', []):
        view = document['bufferViews'][image['bufferView']];offset = binary + view.get('byteOffset', 0)
        blob = bytes(data[offset:offset + view['byteLength']])
        with Image.open(io.BytesIO(blob)) as texture:
            texture.load()
            textures.append({'sha256': _digest(blob), 'bytes': len(blob), 'size': list(texture.size), 'format': texture.format})
            if texture.size != (2048, 2048): errors.append('texture_size')
    return {**info, 'errors': errors, 'overlap': metrics, 'ground_error_m': ground, 'top_error_m': top,
            'vertical_tolerance_m': tolerance, 'node_aware_bounds': [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            'projection': coarse, 'fine_projection': fine, 'textures': textures}


def export_extrema_preserving_lod(source_high: str | Path, destination_low: str | Path, *,
                                  fit_record: Mapping[str, Any], target_faces: int = 100000,
                                  node_executable: str = 'node', timeout_seconds: int = 120,
                                  strict_protected_faces: bool = False,
                                  target_relative_error: float = .01,
                                  validation_profile: str = 'source_reconstruction') -> dict[str, Any]:
    """Write a new candidate and truthful proof; never overwrite or re-place it.

    The default requires an accepted high-mesh fit using the original FitPolicy.
    Explicit visual_scene_completion permits only footprint disagreement with
    a partial source. All physical bounds, attributes and extrema remain gated;
    independent visual scene acceptance is still required afterwards.
    Independent final delivery QA is still required even when status is accepted.
    No existing sidecar, fit, approval manifest or binding is modified here.
    """
    started = time.monotonic();source_high = Path(source_high);destination_low = Path(destination_low)
    if destination_low.exists() or source_high.resolve() == destination_low.resolve():
        raise ValueError('A distinct, nonexistent candidate output is required')
    if isinstance(target_faces, bool) or not isinstance(target_faces, int) or not 8 < target_faces <= 100000:
        raise ValueError('An integer low-LOD target between9 and100000 is required')
    if not 1 <= timeout_seconds <= 600:
        raise ValueError('A finite CPU backend deadline of at most600 seconds is required')
    if not np.isfinite(target_relative_error) or not 0 < target_relative_error <= .02:
        raise ValueError('CPU simplification error budget must be positive and at most2 percent')
    fit = copy.deepcopy(dict(fit_record))
    source_sha = sha256(source_high)
    if validation_profile not in {'source_reconstruction', 'visual_scene_completion'}:
        raise ValueError('Unknown LOD validation profile')
    visual_completion = validation_profile == 'visual_scene_completion'
    reasons = fit.get('reasons', [])
    visual_fit = (visual_completion and fit.get('status') in {'accepted', 'review_required'}
                  and all(r in {'actual_mesh_iou_below_threshold', 'actual_mesh_target_coverage_below_threshold',
                                'actual_mesh_asset_coverage_below_threshold'} for r in reasons))
    if fit.get('source_sha256') != source_sha or not (visual_fit or fit.get('status') == 'accepted' and not reasons):
        raise ValueError('An accepted fit bound to the actual high bytes is required')
    if fit.get('policy') != asdict(FitPolicy()):
        raise ValueError('The unchanged original FitPolicy is required')
    height = float(fit['predicted_top_z_m'] - fit['predicted_base_z_m'])
    if not np.isfinite(height) or height <= 0: raise ValueError('Positive current predicted height is required')
    matrix = np.asarray(fit['world_from_asset_4x4'], dtype=float)
    dependency = _dependency_proof()
    node = shutil.which(node_executable)
    if node is None: raise RuntimeError('Node.js with CPU WebAssembly is required for the packaged offline backend')
    inspect_glb(source_high)
    data, document, binary = _read_glb(source_high)
    scene = read_glb_scene_geometry(source_high)
    if len(document['meshes']) != 1 or len(document['meshes'][0]['primitives']) != 1 or len(scene['instances']) != 1:
        raise ValueError('This CPU fallback requires one static primitive and one active scene instance')
    if document.get('skins') or document.get('animations') or set(document.get('extensionsRequired', [])) - {'EXT_texture_webp'}:
        raise ValueError('Skinned, animated or unsupported compressed geometry cannot use this fallback')
    primitive = document['meshes'][0]['primitives'][0]
    if set(primitive['attributes']) != {'POSITION', 'NORMAL', 'TEXCOORD_0'} or 'indices' not in primitive:
        raise ValueError('Indexed POSITION/NORMAL/TEXCOORD_0 source is required')
    original = {name: np.array(_accessor(data, document, binary, index)[0], copy=True) for name, index in primitive['attributes'].items()}
    if any(array.dtype != np.dtype('<f4') for array in original.values()):
        raise ValueError('Original FLOAT vertex attributes are required')
    source_indices = np.array(_accessor(data, document, binary, primitive['indices'])[0], dtype='<u4').ravel()
    source_used, remapped = np.unique(source_indices, return_inverse=True)
    arrays = {name: array[source_used] for name, array in original.items()}
    faces = remapped.astype('<u4').reshape(-1, 3)
    instance = np.asarray(scene['instances'][0]['asset_from_node_4x4'], dtype=float)
    locks, protection = extrema_lock_mask(arrays['POSITION'], faces, matrix @ instance, min(.20, max(.05, height * .01)))
    # The backend may discard a locked triangle with coincident seam vertices.
    # Reserve room for exact original faces and restore any that it discarded.
    protected_faces = protection['protected_faces']
    reserved = len(protected_faces) if strict_protected_faces and target_faces > len(protected_faces) + 8 else 0
    backend_target = target_faces - reserved
    destination_low.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='citycode-extrema-', dir=destination_low.parent) as directory:
        directory = Path(directory)
        arrays['POSITION'].astype('<f4').tofile(directory / 'positions.f32')
        np.concatenate([arrays['NORMAL'], arrays['TEXCOORD_0']], axis=1).astype('<f4').tofile(directory / 'attributes.f32')
        faces.astype('<u4').tofile(directory / 'indices.u32');locks.tofile(directory / 'locks.u8')
        (directory / 'config.json').write_text(json.dumps({'target_faces': backend_target,
            'target_relative_error':target_relative_error}))
        runner = _HERE / '_extrema_simplify.cjs'
        process = subprocess.run([node, str(runner), str(directory)], capture_output=True, text=True, timeout=timeout_seconds)
        if process.returncode:
            raise RuntimeError('Finite CPU simplification failed: ' + process.stderr[-1000:])
        backend = json.loads((directory / 'backend_result.json').read_text())
        result = np.fromfile(directory / 'simplified_indices.u32', dtype='<u4').reshape(-1, 3)
    if not len(result) or result.max() >= len(arrays['POSITION']): raise ValueError('Backend returned invalid indices')
    retained = _face_set(result)
    missing = np.asarray([face for face in protected_faces
                          if next(iter(_face_set(face[None, :]))) not in retained], dtype='<u4').reshape(-1, 3)
    if len(missing) and strict_protected_faces:
        result = np.concatenate([result, missing])
    blob, used, images = _repack_original_attributes(data, document, binary, arrays, primitive, result)
    # Exclusive create prevents a concurrent caller from replacing any existing asset.
    with destination_low.open('xb') as output: output.write(blob)
    low = _validate_geometry(destination_low, fit, target_faces, visual_completion=visual_completion)
    high = _validate_geometry(source_high, fit, 250000, visual_completion=visual_completion)
    errors = ['high:' + value for value in high['errors']] + ['low:' + value for value in low['errors']]
    protected_kept = _face_set(protection['protected_faces']) <= _face_set(result)
    component_labels = protection['physical_component_by_vertex']
    components_kept = len(np.unique(component_labels[result.ravel()])) == protection['physical_component_count']
    if not protected_kept: errors.append('protected_source_faces_lost')
    if not components_kept: errors.append('source_component_lost')
    bounds = np.asarray(high['node_aware_bounds']);low_bounds = np.asarray(low['node_aware_bounds'])
    drift = float(np.max(np.abs(low_bounds - bounds) / np.maximum(bounds[1] - bounds[0], 1e-6)))
    if drift > .025: errors.append('common_local_frame_drift')
    if sha256(source_high) != source_sha: raise ValueError('High source changed during export')
    actual_data, actual_document, actual_binary = _read_glb(destination_low)
    output_primitive = actual_document['meshes'][0]['primitives'][0]
    attributes_equal = all(np.array_equal(_accessor(actual_data, actual_document, actual_binary, output_primitive['attributes'][name])[0], array[used]) for name, array in arrays.items())
    if not attributes_equal: errors.append('source_attribute_values_changed')
    if [row['sha256'] for row in images] != [row['sha256'] for row in low['textures']]: errors.append('source_texture_bytes_changed')
    report = {'schema': 'citycode.extrema_preserving_lod.v1', 'status': 'accepted_candidate' if not errors else 'review_required',
        'validation_profile': validation_profile,
        'errors': errors, 'source_sha256': source_sha, 'source_bytes': source_high.stat().st_size,
        'raw_cache_sha256': fit.get('raw_cache_sha256'), 'low': low, 'high': high,
        'backend': backend, 'dependency': dependency, 'python_code_sha256': sha256(__file__), 'runner_code_sha256': sha256(runner),
        'world_from_asset_4x4': matrix.tolist(), 'existing_placement_unchanged': True,
        'same_source_normal_uv_material_images': attributes_equal and [row['sha256'] for row in images] == [row['sha256'] for row in low['textures']],
        'source_node_transforms_preserved': document['nodes'] == actual_document['nodes'],
        'protected_source_faces_retained': protected_kept, 'source_components_retained': components_kept,
        'physical_components': protection['physical_component_count'], 'locked_input_uv_vertices': int(locks.sum()),
        'protected_source_faces': len(protection['protected_faces']), 'lock_mask_sha256': _digest(locks.tobytes()),
        'protected_faces_restored_after_backend': len(missing) if strict_protected_faces else 0,
        'backend_target_faces_with_protected_reserve': backend_target,
        'strict_protected_faces': bool(strict_protected_faces),
        'selected_original_vertex_indices_sha256': _digest(source_used[used].astype('<u4').tobytes()),
        'common_local_frame_relative_drift': drift, 'canonical_sidecar_fit_or_binding_modified': False,
        'final_independent_delivery_qa_required': True, 'elapsed_seconds': time.monotonic() - started}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True);parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fit-record', type=Path, required=True);parser.add_argument('--target-faces', type=int, default=100000)
    parser.add_argument('--node', default='node')
    parser.add_argument('--validation-profile',choices=['source_reconstruction','visual_scene_completion'],default='source_reconstruction')
    args = parser.parse_args(argv)
    proof = args.output.with_suffix(args.output.suffix + '.proof.json')
    if proof.exists(): raise ValueError('A new proof path is required')
    report = export_extrema_preserving_lod(args.source, args.output, fit_record=json.loads(args.fit_record.read_text()),
                                          target_faces=args.target_faces, node_executable=args.node,
                                          validation_profile=args.validation_profile)
    with proof.open('x') as output: json.dump(report, output, indent=2, allow_nan=False);output.write('\n')
    print(json.dumps({'status': report['status'], 'errors': report['errors'], 'sha256': report['low']['sha256'],
                      'faces': report['low']['faces'], 'proof': str(proof)}))
    return 0 if report['status'] == 'accepted_candidate' else 2


if __name__ == '__main__': raise SystemExit(main())
