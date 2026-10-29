"""Copy explicitly selected prediction inputs into a self-contained cache."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import shutil

from city_pipeline.runtime import read_json, write_json, sha256

SCHEMA = 'city_pipeline.cached_input_bundle.v2'
E09_OUTPUTS = ('bev_rgb', 'surface_z', 'provenance', 'background_semantic_ids', 'final_ids')
CAMERA_KEYS = ('schema', 'view_id', 'image_size_px', 'intrinsics',
               'camera_to_world_4x4_m', 'world_to_camera_4x4_m',
               'coordinate_conventions', 'data_bounds_xy_world_m',
               'core_bounds_xy_world_m', 'ground_reference_z_m')
FORBIDDEN = ('global_vectors', 'gt_', 'evaluation_gt', 'depth_linear_m.exr',
             'building_instance_ids.png', '/semantic_ids.png', 'gt_prompt_mask')


def _inside(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('Bundle paths must be nonempty relative paths')
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Bundle path escapes its directory: ' + value)
    return path


def _prediction_source(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root) or any(t in str(path).lower() for t in FORBIDDEN):
        raise ValueError('Source is outside prediction inputs: ' + str(path))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bundle_cache(source_root, data_root, output, include_evidence=False, *,
                 include_visibility=False, dataset_root=None):
    """Freeze numerical inputs without relabelling calibrated depth as raw.

    Evidence includes native images and the five E09 outputs needed for image
    crops and numerical background surfaces. Reference annotations, previews,
    experiment caches and redundant point-cloud exports are not copied.
    """
    source_root, data_root, output = map(lambda p: Path(p).resolve(),
                                         (source_root, data_root, output))
    if output.exists():
        raise FileExistsError(output)
    observation_path = _prediction_source(
        data_root, source_root / 'global_fused/building_observation_footprints.geojson')
    doc = copy.deepcopy(read_json(observation_path))
    features = doc['features']
    by_sample = {}
    for feature in features:
        props = feature['properties']
        if any(k.lower().startswith(('gt_', 'ground_truth', 'oracle_')) for k in props):
            raise ValueError('Reference properties cannot enter a prediction cache')
        by_sample.setdefault(props['sample_id'], props.copy())

    # Preflight every copy before creating the output directory.
    plan, targets, optional = {}, {}, {}
    def add(value, destination, role):
        source = _prediction_source(data_root, value)
        destination = Path(destination).as_posix()
        if source in plan:
            return plan[source]['path']
        if destination in targets and targets[destination] != source:
            raise ValueError('Ambiguous bundle destination: ' + destination)
        targets[destination] = source
        plan[source] = {'path': destination, 'role': role}
        return destination

    roles = {'building_point_cloud_npz_path': 'points',
             'source_instance_ids_path': 'source_instance_ids',
             'bev_instance_ids_path': 'bev_instance_ids', 'rgb_path': 'rgb'}
    for feature in features:
        props = feature['properties']
        for key, value in list(props.items()):
            if not key.endswith('_path') or not isinstance(value, str):
                continue
            required = key == 'building_point_cloud_npz_path'
            if required or include_evidence or (include_visibility and key == 'source_instance_ids_path'):
                role = roles.get(key, key.removesuffix('_path'))
                destination = (Path('data/views') / props['sample_id'] / 'rgb.png'
                               if key == 'rgb_path' else Path('data') / role / Path(value).name)
                props[key] = add(value, destination, role)
            else:
                name = hashlib.sha256((key + ':' + props['sample_id']).encode()).hexdigest()[:16]
                rel = 'unbundled_evidence/' + name + '/' + Path(value).name
                optional[rel] = {'role': key, 'present': False}
                props[key] = rel

    cameras, visibility, e09 = {}, None, None
    if include_visibility:
        from city_pipeline.buildings.visibility_refinement import build_inventory
        visibility = build_inventory(observation_path, source_root, data_root)
    if include_evidence or include_visibility:
        for sid, props in sorted(by_sample.items()):
            view = ((Path(dataset_root).resolve() / 'patches' / props['patch_id'] /
                     'views' / props['view_id']) if dataset_root else
                    (data_root / props['rgb_path']).resolve().parent)
            camera_path = _prediction_source(data_root, view / 'camera.json')
            camera = read_json(camera_path)
            required = ('view_id', 'image_size_px', 'intrinsics',
                        'camera_to_world_4x4_m', 'world_to_camera_4x4_m', 'data_bounds_xy_world_m')
            if any(k not in camera for k in required):
                raise ValueError('Camera lacks projection fields: ' + sid)
            clean = {k: camera[k] for k in CAMERA_KEYS if k in camera}
            clean['strict_valid_mask_path'] = 'strict_valid_mask.png'
            cameras[sid] = clean
            valid = add(view / 'strict_valid_mask.png',
                        Path('data/views') / sid / 'strict_valid_mask.png', 'strict_valid')
            if visibility is not None:
                row = visibility[sid]
                row['camera'] = (Path('data/views') / sid / 'camera.json').as_posix()
                row['strict_valid'] = valid
                for key in ('source_ids', 'background_ids', 'predicted_depth'):
                    row[key] = add(row[key], Path('data') / key / Path(row[key]).name, key)

    source_metadata = {'observations_sha256': sha256(observation_path)}
    if include_evidence:
        e09_path = _prediction_source(data_root, source_root / 'e09_full_pipeline/manifest.json')
        source_metadata['e09_manifest_sha256'] = sha256(e09_path)
        original = read_json(e09_path)
        records = {row['id']: row for row in original['samples']}
        if not set(by_sample).issubset(records):
            raise ValueError('Missing E09 samples for prediction observations')
        samples = []
        for sid in sorted(by_sample):
            row = records[sid]
            outputs = {key: add(row['outputs'][key],
                              Path('data/e09') / key / Path(row['outputs'][key]).name,
                              'e09_' + key) for key in E09_OUTPUTS}
            samples.append({'id': sid, 'rasterization': row['rasterization'], 'outputs': outputs})
        e09 = {'schema': 'city_pipeline.cached_e09_surfaces.v1', 'status': 'complete',
               'sample_count': len(samples), 'samples': samples,
               'scope': 'Predicted BEV RGB, height surface, provenance and semantic/instance labels',
               'omitted_derivatives': ['previews', 'PLY', 'redundant point clouds', 'building-height arrays']}

    output.mkdir(parents=True)
    copied = []
    for source, record in sorted(plan.items(), key=lambda item: item[1]['path']):
        target = output / record['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha256(target)
        if digest != sha256(source):
            raise ValueError('Source changed while copying: ' + str(source))
        copied.append({**record, 'sha256': digest, 'bytes': target.stat().st_size})
    for sid, camera in cameras.items():
        write_json(output / 'data/views' / sid / 'camera.json', camera)
    obs_relative = 'source/global_fused/building_observation_footprints.geojson'
    write_json(output / obs_relative, doc)
    write_json(output / 'source/global_fused/manifest.json', {
        'schema': 'city_pipeline.cached_observations.v1', 'status': 'complete',
        'observation_count': len(features), 'sample_count': len(by_sample),
        'observations': obs_relative, 'uses_ground_truth': False})
    config = {'mode': 'cached', 'data_root': '.', 'source_run_root': 'source',
              'review_file': None}
    if e09 is not None:
        write_json(output / 'source/e09_full_pipeline/manifest.json', e09)
    if visibility is not None:
        write_json(output / 'visibility_inputs.json', {
            'schema': 'city_pipeline.visibility_inputs.v1', 'samples': visibility,
            'depth_stage': 'class_aware_calibrated_prediction', 'uses_ground_truth': False})
        config['visibility_inventory'] = 'visibility_inputs.json'
    write_json(output / 'buildings.json', config)
    copied_paths = {row['path'] for row in copied}
    for target in sorted(output.rglob('*')):
        if target.is_file() and target.relative_to(output).as_posix() not in copied_paths:
            copied.append({'path': target.relative_to(output).as_posix(), 'role': 'metadata',
                           'sha256': sha256(target), 'bytes': target.stat().st_size})
    write_json(output / 'INPUTS.json', {
        'schema': SCHEMA, 'status': 'complete', 'uses_ground_truth': False,
        'sample_count': len(by_sample), 'observation_count': len(features),
        'files': sorted(copied, key=lambda row: row['path']),
        'file_count': len(copied), 'total_bytes': sum(row['bytes'] for row in copied),
        'optional_unbundled_evidence': optional, 'source_content': source_metadata,
        'evidence_included': include_evidence, 'visibility_included': include_visibility,
        'scope': 'Prediction cache replay; no raw neural depth inference or reference annotations',
        'camera_metadata': 'Projection, coordinate and explicit ground-reference fields only',
        'all_paths_relative_to': 'bundle_directory'})
    validate_bundle(output, checksums=True)
    return output / 'buildings.json'


def validate_bundle(cache, checksums=True):
    """Check completeness and contained references before any prediction run."""
    root = Path(cache).resolve()
    manifest = read_json(root / 'INPUTS.json')
    if (manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete'
            or manifest.get('uses_ground_truth') is not False):
        raise ValueError('Incomplete or unsupported prediction cache')
    files = {}
    for row in manifest['files']:
        rel = row['path']; path = _inside(root, rel)
        if rel in files or any(t in rel.lower() for t in FORBIDDEN):
            raise ValueError('Duplicate or reference-data entry: ' + rel)
        if not path.is_file() or (root / rel).is_symlink():
            raise ValueError('Missing or linked bundle input: ' + rel)
        if path.stat().st_size != row['bytes'] or (checksums and sha256(path) != row['sha256']):
            raise ValueError('Changed bundle input: ' + rel)
        files[rel] = path
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'INPUTS.json'}:
        raise ValueError('Bundle contains unlisted or missing files')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Bundle must not contain symbolic links')
    if (manifest.get('file_count') != len(files) or
            manifest.get('total_bytes') != sum(p.stat().st_size for p in files.values())):
        raise ValueError('Bundle file totals do not match its inventory')
    def reference(value, optional=False):
        _inside(root, value)
        if value not in files and not (optional and value in manifest['optional_unbundled_evidence']):
            raise ValueError('Unlisted bundle reference: ' + value)
    config = read_json(root / 'buildings.json')
    permitted = {'mode', 'data_root', 'source_run_root', 'review_file',
                 'dataset_root', 'visibility_inventory'}
    if (set(config) - permitted or config.get('mode') != 'cached'
            or config.get('data_root') != '.' or config.get('source_run_root') != 'source'
            or config.get('review_file') is not None):
        raise ValueError('Cached config must contain only portable input paths and no review')
    if config.get('dataset_root') is not None:
        if not _inside(root, config['dataset_root']).is_dir():
            raise ValueError('Missing bundled dataset directory')
    observation_path = 'source/global_fused/building_observation_footprints.geojson'
    reference(observation_path)
    observations = read_json(root / observation_path)['features']
    samples = {f['properties']['sample_id'] for f in observations}
    for feature in observations:
        for key, value in feature['properties'].items():
            if key.lower().startswith(('gt_', 'ground_truth', 'oracle_')):
                raise ValueError('Reference properties in observations')
            if key.endswith('_path'):
                reference(value, optional=True)
    if config.get('visibility_inventory'):
        reference(config['visibility_inventory'])
        inventory = read_json(root / config['visibility_inventory'])
        if inventory.get('schema') != 'city_pipeline.visibility_inputs.v1' or set(inventory['samples']) != samples:
            raise ValueError('Visibility samples differ from observations')
        for row in inventory['samples'].values():
            if set(row) != {'camera', 'strict_valid', 'source_ids', 'background_ids', 'predicted_depth'}:
                raise ValueError('Unexpected visibility input roles')
            for value in row.values():
                reference(value)
            camera = read_json(files[row['camera']])
            if (set(camera) - set(CAMERA_KEYS) - {'strict_valid_mask_path'} or
                    camera.get('strict_valid_mask_path') != 'strict_valid_mask.png'):
                raise ValueError('Camera must contain projection metadata only')
            if files[row['strict_valid']] != files[row['camera']].parent / 'strict_valid_mask.png':
                raise ValueError('Camera ROI path differs from the visibility inventory')
    if manifest['evidence_included']:
        reference('source/e09_full_pipeline/manifest.json')
        e09 = read_json(root / 'source/e09_full_pipeline/manifest.json')
        if {row['id'] for row in e09['samples']} != samples:
            raise ValueError('E09 samples differ from observations')
        for row in e09['samples']:
            if set(row['outputs']) != set(E09_OUTPUTS):
                raise ValueError('Missing E09 numerical surface/evidence outputs')
            for value in row['outputs'].values():
                reference(value)
    if len(observations) != manifest['observation_count'] or len(samples) != manifest['sample_count']:
        raise ValueError('Incorrect observation inventory counts')
    return {'status': 'PASS', 'files': len(files), 'bytes': sum(p.stat().st_size for p in files.values()),
            'sample_count': len(samples), 'observation_count': len(observations),
            'checksums_verified': bool(checksums), 'self_contained': not bool(manifest['optional_unbundled_evidence'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--include-evidence', action='store_true')
    parser.add_argument('--include-visibility', action='store_true')
    args = parser.parse_args()
    print(bundle_cache(args.source_root, args.data_root, args.output, args.include_evidence,
                       include_visibility=args.include_visibility, dataset_root=args.dataset_root))


if __name__ == '__main__':
    main()
