"""One entry point for observation preparation and metric structure recovery."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from .dataset import discover_samples, validate_dataset

STRUCTURE_PROFILES = json.loads(Path(__file__).with_name('structure_profiles.json').read_text())
CLEAN_STRUCTURE_PROFILE = STRUCTURE_PROFILES['standard']['polish']


def structure_parameters(profile='standard', override=None):
    if profile not in STRUCTURE_PROFILES:
        raise ValueError('Unknown structure profile')
    result = copy.deepcopy(STRUCTURE_PROFILES[profile])
    if override is not None:
        supplied = json.loads(Path(override).read_text())
        if not isinstance(supplied, dict) or set(supplied) - {'profile', *result.keys()}:
            raise ValueError('Structure config accepts only global processing parameters')
        if supplied.get('profile', profile) != profile:
            raise ValueError('Only the standard release profile is supported')
        def merge(target, patch):
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = copy.deepcopy(value)
        merge(result, {k: v for k, v in supplied.items() if k != 'profile'})
    if any(not isinstance(value, dict) for value in result.values()):
        raise ValueError('Each standard processing section must be an object')
    from city_pipeline.buildings.parameters import normalize_postprocess, normalize_selection
    result['selection'] = normalize_selection(result['selection'])
    if result['visibility_refinement'] != {'method': 'local_surface_veto'}:
        raise ValueError('The release uses the fixed local_surface_veto refinement')
    if 'postprocess' in result:
        result['postprocess'] = normalize_postprocess(result['postprocess'])
    if result.get('polish'):
        from city_pipeline.buildings.polish import _config
        _config(result['polish'])
    if result.get('roof_retention'):
        from city_pipeline.buildings.roof_retention import _configuration
        _configuration(result['roof_retention'])
    if result.get('height_consensus'):
        from city_pipeline.buildings.height_consensus import normalize_config
        result['height_consensus'] = normalize_config(result['height_consensus'])
    return result


def _selection(parser):
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--patch', action='append', help='Patch ID; repeat to select a subset')
    parser.add_argument('--view', action='append', help='Defaults to views declared by the dataset')


def recovery_config(dataset, depth, masks, *, patches=None, views=None, priors=None,
                    structure_only=False, structure_config=None):
    dataset = Path(dataset).resolve()
    selected = discover_samples(dataset, patches, views)
    if len({sample.patch_id for sample in selected}) != len(selected):
        raise ValueError('The standard recipe requires one selected view per physical patch')
    depth, masks = Path(depth).resolve(), Path(masks).resolve()
    if not (depth / 'manifest.json').is_file():
        raise FileNotFoundError(depth / 'manifest.json')
    selected_ids = {sample.stem for sample in selected}
    depth_manifest = json.loads((depth / 'manifest.json').read_text())
    records = {}
    for item in depth_manifest.get('samples', []):
        sid = f"{item['patch_id']}__{item['view_id']}"
        if sid in records:
            raise ValueError(f'Duplicate depth observation: {sid}')
        records[sid] = item
    if selected_ids - records.keys():
        raise ValueError(f'Missing selected depth observations: {sorted(selected_ids - records.keys())}')
    for sid in selected_ids:
        item = records[sid]
        stage = item.get('depth_stage', depth_manifest.get('depth_stage'))
        calibration = str(item.get('scale_calibration', {}).get('method', '')).lower()
        if stage == 'already_calibrated_prediction' or 'already-calibrated' in calibration:
            raise ValueError('recover requires raw model depth; already calibrated cache is only valid for explicit compatibility checks')
        value = item.get('outputs', {}).get('depth_raw_zc_m')
        if not value:
            raise ValueError(f'Missing raw depth path: {sid}')
        if not (depth / value).is_file():
            raise FileNotFoundError(depth / value)
    for stage in ('structure_v9_2', 'structure_v7b'):
        folder = masks / stage
        if not (folder / 'manifest.json').is_file():
            raise FileNotFoundError(folder / 'manifest.json')
        mask_records = {}
        for item in json.loads((folder / 'manifest.json').read_text()).get('samples', []):
            if item['id'] in mask_records:
                raise ValueError(f"Duplicate mask observation: {stage}/{item['id']}")
            mask_records[item['id']] = item
        if selected_ids - mask_records.keys():
            raise ValueError(f'Missing selected masks: {stage}/{sorted(selected_ids - mask_records.keys())}')
        for sid in selected_ids:
            if not (folder / mask_records[sid]['result']).is_file():
                raise FileNotFoundError(folder / mask_records[sid]['result'])
    priors = Path(priors).resolve() if priors else dataset / 'priors.json'
    from city_pipeline.geometry.priors import read_plane_priors
    read_plane_priors(priors)
    return {
        'mode': 'raw', 'data_root': str(dataset), 'dataset_root': str(dataset),
        'depth_root': str(depth), 'mask_root': str(masks),
        'calibration_plane_policy': 'fixed-smallcity', 'priors_path': str(priors),
        'include_views': sorted({sample.view_id for sample in selected}),
        'sample_ids': [sample.stem for sample in selected], 'review_file': None,
        **structure_parameters(override=structure_config),
        'export_generation_evidence': not structure_only,
    }


def cached_recovery_config(cache, *, structure_only=False, structure_config=None):
    """Use the same release recipe with a checked, portable prediction bundle."""
    from city_pipeline.inputs import validate_bundle
    cache = Path(cache).resolve()
    validate_bundle(cache, checksums=True)
    paths = json.loads((cache / 'buildings.json').read_text())
    if paths.get('mode') != 'cached' or paths.get('review_file'):
        raise ValueError('A prediction-only cached input bundle is required')
    allowed = {'mode', 'data_root', 'source_run_root', 'review_file',
               'visibility_inventory', 'dataset_root'}
    if set(paths) - allowed:
        raise ValueError('Cached bundle must contain input paths, not a processing recipe')
    root = (cache / paths.get('data_root', '.')).resolve()
    if root != cache:
        raise ValueError('Cached data_root must be the bundle directory')
    cfg = {'mode': 'cached', 'data_root': str(root),
           'source_run_root': str((root / paths['source_run_root']).resolve()),
           'review_file': None, **structure_parameters(override=structure_config),
           'export_generation_evidence': not structure_only,
           'input_bundle_manifest': str(cache / 'INPUTS.json')}
    for key in ('visibility_inventory', 'dataset_root'):
        if paths.get(key):
            cfg[key] = str((root / paths[key]).resolve())
    return cfg


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check-inputs', help='Validate prepared observations')
    _selection(check)
    check.add_argument('--checksums', action='store_true')

    prepare = commands.add_parser('prepare-masks', help='Export both formal segmentation requests')
    _selection(prepare)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--working-size', type=int, default=2048)

    provider = commands.add_parser('run-masks', help='Execute explicitly configured Vertex requests')
    provider.add_argument('--requests', type=Path, required=True)
    provider.add_argument('--project')
    provider.add_argument('--location')
    provider.add_argument('--model')
    provider.add_argument('--max-requests', type=int)

    masks = commands.add_parser('import-masks', help='Import predicted PNG masks for geometry')
    _selection(masks)
    masks.add_argument('--results', type=Path, required=True)
    masks.add_argument('--output', type=Path, required=True)
    masks.add_argument('--working-size', type=int, default=2048)

    predict = commands.add_parser('predict-depth', help='Infer camera-forward depth with a full v2 checkpoint')
    _selection(predict)
    predict.add_argument('--output', type=Path, required=True)
    predict.add_argument('--checkpoint', type=Path, required=True)
    predict.add_argument('--model-source', type=Path)
    predict.add_argument('--device', default='cuda')
    predict.add_argument('--checkpoint-sha256')
    predict.add_argument('--inference-size', type=int, default=1024)

    depth = commands.add_parser('import-depth', help='Import predicted depth arrays and their provenance')
    _selection(depth)
    depth.add_argument('--input', type=Path, required=True)
    depth.add_argument('--output', type=Path, required=True)

    recover = commands.add_parser('recover', help='Recover footprints, heights, surfaces and generation evidence')
    _selection(recover)
    recover.add_argument('--depth', type=Path, required=True)
    recover.add_argument('--masks', type=Path, required=True)
    recover.add_argument('--priors', type=Path)
    recover.add_argument('--output', type=Path, required=True)
    recover.add_argument('--resume', action='store_true')
    recover.add_argument('--structure-config', type=Path,
                         help='Scene-wide numeric overrides for the standard recipe')
    recover.add_argument('--structure-only', action='store_true',
                         help='Defer native image evidence crops until building generation')

    cached = commands.add_parser('recover-cached', help='Run the same standard recipe from a portable prediction cache')
    cached.add_argument('--cache', type=Path, required=True)
    cached.add_argument('--output', type=Path, required=True)
    cached.add_argument('--resume', action='store_true')
    cached.add_argument('--structure-config', type=Path)
    cached.add_argument('--structure-only', action='store_true')

    verify = commands.add_parser('verify-structure', help='Check local mesh and footprint consistency')
    verify.add_argument('--run', type=Path, required=True)
    commands.add_parser('model-info', help='Print required depth model resources')
    return ap


def execute(args):
    if args.command == 'check-inputs':
        return validate_dataset(args.dataset, args.patch, args.view, args.checksums)
    if args.command in {'prepare-masks', 'import-masks', 'run-masks'}:
        from .segmentation import prepare_requests, import_results, run_vertex_requests
        if args.command == 'prepare-masks':
            return str(prepare_requests(args.dataset, args.output, patch_ids=args.patch,
                                       views=args.view, working_size=args.working_size))
        if args.command == 'import-masks':
            return str(import_results(args.dataset, args.results, args.output,
                                      patch_ids=args.patch, views=args.view, working_size=args.working_size))
        return run_vertex_requests(args.requests, project=args.project, location=args.location,
                                   model=args.model, max_requests=args.max_requests)
    if args.command in {'predict-depth', 'import-depth'}:
        from .depth import predict_dataset, import_predicted_depth
        if args.command == 'predict-depth':
            return str(predict_dataset(args.dataset, args.output, args.checkpoint,
                                      args.model_source, args.device, args.patch, args.view,
                                      args.inference_size, expected_sha256=args.checkpoint_sha256))
        return str(import_predicted_depth(args.dataset, args.input, args.output, args.patch, args.view))
    if args.command in {'recover', 'recover-cached'}:
        if args.command == 'recover':
            cfg = recovery_config(args.dataset, args.depth, args.masks, patches=args.patch,
                                  views=args.view, priors=args.priors,
                                  structure_only=args.structure_only, structure_config=args.structure_config)
        else:
            cfg = cached_recovery_config(args.cache, structure_only=args.structure_only,
                                         structure_config=args.structure_config)
        output = args.output.resolve()
        # Keep the config outside the fresh output: the runner creates that directory.
        config_path = output.parent / f'.{output.name}.config.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
        from city_pipeline.buildings.run import run
        release = run(config_path, output, args.resume)
        state = json.loads((output / 'RUN_STATUS.json').read_text())
        return {'release': str(release), 'building_generation_inputs':
                str(output / 'building_generation_inputs') if cfg['export_generation_evidence'] else None,
                'generation_evidence': state['generation_evidence'],
                'surface_outputs': str((Path(cfg['source_run_root']) if cfg['mode'] == 'cached'
                                        else output / 'upstream') / 'e09_full_pipeline')}
    if args.command == 'verify-structure':
        subprocess.run([sys.executable, '-m', 'city_pipeline.verify_buildings', '--run', str(args.run)], check=True)
        return json.loads((args.run / 'VALIDATION.json').read_text())
    if args.command == 'model-info':
        from .model_manifest import depth_model_requirements
        return depth_model_requirements()
    raise ValueError(f'Unknown command: {args.command}')


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        result = execute(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        ap.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
