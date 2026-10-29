"""Frozen building recipe using only modules shipped in this distribution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from city_pipeline.runtime import PACKAGE_ROOT, read_json, sha256, write_json

STAGES = [
    ('consensus', '01_consensus/manifest.json'),
    ('roof_statistics', 'roof_statistics.json'),
    ('height', '02_height/CAP1_P95/manifest.json'),
    ('support_statistics', 'observed_support.json'),
    ('spatial', '03_spatial/C1_elevated_roof/manifest.json'),
    ('spatial_export', '03_release/manifest.json'),
    ('boundary', '05_boundary/manifest.json'),
    ('prune', '06_prune/manifest.json'),
    ('local_repair', '07_repair/trials/local/manifest.json'),
    ('replace_repair', '07_repair/trials/replace/manifest.json'),
    ('filter', '08_filter/manifest.json'),
    ('final_export', '../release/manifest.json'),
]


def run(config, output, resume=False):
    cfg = read_json(config)
    if cfg.get('review_file') is not None:
        raise ValueError('External review configurations are not supported by the standard recipe')
    from .parameters import normalize_postprocess, normalize_selection
    postprocess = normalize_postprocess(cfg.get('postprocess'))
    selection = normalize_selection(cfg.get('selection'))
    visibility_config = cfg.get('visibility_refinement')
    if visibility_config is not None and visibility_config != {'method': 'local_surface_veto'}:
        raise ValueError('Unsupported visibility refinement recipe')
    root = (Path(config).resolve().parent / cfg.get('data_root', '.')).resolve()
    path = lambda key: (root / cfg[key]).resolve()
    mode = cfg.get('mode', 'cached')
    if mode not in {'raw', 'cached'}:
        raise ValueError('mode must be raw or cached')
    output = Path(output).resolve()
    source = path('source_run_root') if mode == 'cached' else output / 'upstream'
    work = output / 'work'
    env = os.environ.copy()
    env.update(CITY_PIPELINE_DATA_ROOT=str(root), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               MPLCONFIGDIR=str(output / '.matplotlib'))
    env['PYTHONPATH'] = os.pathsep.join([str(PACKAGE_ROOT.parent), env.get('PYTHONPATH', '')])
    # Guard is installed by each child before importing algorithm modules.
    if cfg.get('dataset_root'):
        env['CITY_PIPELINE_NO_GT_DATASET'] = str(path('dataset_root'))
    priors_path = None
    if mode == "raw" and cfg.get("calibration_plane_policy", "camera-ground") == "fixed-smallcity":
        priors_path = (root / cfg["priors_path"]).resolve() if cfg.get("priors_path") else path("dataset_root") / "priors.json"
        from city_pipeline.geometry.priors import read_plane_priors
        read_plane_priors(priors_path)
    commands = []
    polish_config = cfg.get('polish')
    polish_config_path = work / 'polish_config.json'
    height_config = cfg.get('height_consensus')
    height_config_path = work / 'height_consensus_config.json'
    inventory = path('visibility_inventory') if cfg.get('visibility_inventory') else None
    if mode == 'raw':
        cmd = [sys.executable, '-m', 'city_pipeline.geometry.orchestrator', '--profile', 'p384',
               '--phase', 'geometry', '--dataset-root', str(path('dataset_root')),
               '--depth-root', str(path('depth_root')), '--mask-root', str(path('mask_root')),
               '--run-root', str(source), '--source-evaluation', 'none', '--skip-release-verification',
               '--calibration-plane-policy', cfg.get('calibration_plane_policy', 'camera-ground')]
        if priors_path:
            cmd += ["--priors-path", str(priors_path)]
        if cfg.get("allow_cached_smoke_recalibration", False):
            cmd += ["--allow-cached-smoke-recalibration"]
        for view in cfg.get('include_views', ['tilt_light']):
            cmd += ['--include-view', view]
        for sid in cfg.get('sample_ids', []):
            cmd += ['--sample-id', sid]
        commands.append(('upstream', cmd, source / 'e10_height_guided/footprint_height_representation/manifest.json'))
        commands.append(('observations', [sys.executable, '-m', 'city_pipeline.buildings.observations',
            '--run-root', str(source), '--dataset-root', str(path('dataset_root')),
            '--output-root', str(source/'global_fused'), '--enable-near-cluster-consensus',
            '--enable-low-confidence-fragment-filter', '--fragment-minimum-area-m2', '20',
            '--fragment-maximum-elongation', '4.5', '--write-broad-fragment-quarantine',
            '--quarantine-maximum-area-m2', '75', '--quarantine-maximum-elongation', '4.5',
            '--quarantine-minimum-width-m', '5.5', '--quarantine-minimum-compactness', '.20'],
            source / 'global_fused/manifest.json'))
    for name, marker in STAGES:
        if name == 'final_export' and polish_config:
            # Raw outputs and collected inputs can have different caller-owned roots.
            point_data_root = Path(os.path.commonpath([str(root), str(source)]))
            commands.append(('polish', [sys.executable, '-m', 'city_pipeline.buildings.polish',
                '--input-root', str(work/'08_filter'),
                '--observations', str(source/'global_fused/building_observation_footprints.geojson'),
                '--data-root', str(root), '--point-boundary-root', str(point_data_root), '--output-root', str(work/'09_polish'),
                '--config', str(polish_config_path)], work/'09_polish/manifest.json'))
        if name == 'final_export' and height_config:
            final_input = work/'09_polish' if polish_config else work/'08_filter'
            commands.append(('height_consensus', [sys.executable, '-m', 'city_pipeline.buildings.height_consensus',
                '--input-root', str(final_input), '--observations', str(source/'global_fused/building_observation_footprints.geojson'),
                '--point-statistics', str(work/'roof_statistics.json'), '--output-root', str(work/'10_height_consensus'),
                '--config', str(height_config_path)], work/'10_height_consensus/manifest.json'))
        if name == 'final_export' and visibility_config:
            final_input = (work/'10_height_consensus' if height_config else
                           work/'09_polish' if polish_config else work/'08_filter')
            visibility_command = [sys.executable, '-m', 'city_pipeline.buildings.visibility_refinement',
                '--input-root', str(final_input),
                '--observations', str(source/'global_fused/building_observation_footprints.geojson'),
                '--source-run-root', str(source), '--data-root', str(root),
                '--output-root', str(work/'11_visibility')]
            if inventory:
                visibility_command += ['--inventory', str(inventory)]
            commands.append(('visibility_refinement', visibility_command, work/'11_visibility/manifest.json'))
        cmd = [sys.executable, '-m', 'city_pipeline.buildings.stage', '--stage', name,
               '--source', str(source), '--work', str(work)]
        if name in postprocess:
            cmd += ['--parameters', json.dumps(postprocess[name], sort_keys=True)]
        if name == 'consensus':
            cmd += ['--selection-parameters', json.dumps(selection, sort_keys=True)]
        if name == 'prune' and cfg.get('roof_retention'):
            cmd += ['--roof-retention', json.dumps(cfg['roof_retention'], sort_keys=True)]
        if name == 'final_export' and (visibility_config or polish_config or height_config):
            final_input = (work/'11_visibility' if visibility_config else
                           work/'10_height_consensus' if height_config else work/'09_polish')
            cmd += ['--final-input', str(final_input)]
        commands.append((name, cmd, (work / marker).resolve()))
    if cfg.get('export_generation_evidence', True):
        commands.append(('generation_evidence', [sys.executable, '-m', 'city_pipeline.buildings.evidence',
            '--source', str(source), '--release', str(output/'release'),
            '--output', str(output/'building_generation_inputs'), '--data-root', str(root)],
            output/'building_generation_inputs/manifest.json'))
    import hashlib
    signature = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for p in sorted(PACKAGE_ROOT.rglob('*.py')):
        signature.update(str(p.relative_to(PACKAGE_ROOT)).encode()); signature.update(p.read_bytes())
    public_package = PACKAGE_ROOT.parent/'citycode_perception'
    for p in sorted(public_package.rglob('*')):
        if p.is_file() and p.suffix in {'.py', '.json', '.txt'}:
            signature.update(str(p.relative_to(PACKAGE_ROOT.parent)).encode()); signature.update(p.read_bytes())
    # Include the actual cached point clouds, not just the top-level manifest.
    inputs = {}
    if priors_path:
        inputs[str(priors_path)] = sha256(priors_path)
    if mode == 'cached':
        obs = source / 'global_fused/building_observation_footprints.geojson'
        inputs[str(obs)] = sha256(obs)
        for f in read_json(obs)['features']:
            for key in ('building_point_cloud_npz_path',):
                value = f['properties'].get(key)
                if value:
                    p = str((root / value).resolve())
                    if p not in inputs: inputs[p] = sha256(p)
        if visibility_config:
            from .visibility_refinement import collect_input_paths
            for p in collect_input_paths(obs, source, root, inventory):
                if str(p) not in inputs:
                    inputs[str(p)] = sha256(p)
        if cfg.get('input_bundle_manifest'):
            bundle_manifest = path('input_bundle_manifest')
            inputs[str(bundle_manifest)] = sha256(bundle_manifest)
            for item in read_json(bundle_manifest)['files']:
                p = (bundle_manifest.parent/item['path']).resolve()
                if str(p) not in inputs:
                    inputs[str(p)] = sha256(p)
    else:
        for key in ('depth_root', 'mask_root'):
            for p in sorted(path(key).rglob('*')):
                if p.is_file(): inputs[str(p)] = sha256(p)
        dataset = path('dataset_root')
        # Hash only selected source-view RGB, but every relevant metadata file.
        for camera in sorted(dataset.glob('patches/*/views/*/camera.json')):
            sid=camera.parents[2].name+'__'+camera.parent.name
            if cfg.get('sample_ids') and sid not in cfg['sample_ids']:continue
            if camera.parent.name not in cfg.get('include_views',['tilt_light']):continue
            inputs[str(camera)]=sha256(camera)
            rgb=camera.parent/'rgb.png';inputs[str(rgb)]=sha256(rgb)
        for p in sorted(dataset.glob('patches/**/patch.json')): inputs[str(p)] = sha256(p)
        for p in sorted(dataset.glob('patches/**/strict_valid_mask.png')): inputs[str(p)] = sha256(p)
    signature.update(json.dumps(inputs, sort_keys=True).encode())
    fingerprint = signature.hexdigest()
    state_path = output/'RUN_STATUS.json'
    if output.exists():
        if not resume or not state_path.exists(): raise FileExistsError(output)
        state = read_json(state_path)
        if state['signature'] != fingerprint: raise ValueError('Code/config/input changed; use a fresh output.')
    else:
        state = {'signature': fingerprint, 'completed': {}, 'input_sha256': inputs, 'mode': mode,
                 'recipe': 'standard', 'selection': selection,
                 'visibility_refinement': visibility_config,
                 'neural_inference_executed_by_recover': False}
    (output/'logs').mkdir(parents=True, exist_ok=True); work.mkdir(exist_ok=True)
    if polish_config:
        write_json(polish_config_path, polish_config)
    if height_config:
        write_json(height_config_path, height_config)
    for name, command, marker in commands:
        if name in state['completed']:
            for filename, digest in state['completed'][name]['outputs'].items():
                if not Path(filename).is_file() or sha256(filename) != digest:
                    raise ValueError('Completed output changed: ' + filename)
            print('SKIP', name, flush=True); continue
        state.update(status='running', stage=name); write_json(state_path, state)
        print('RUN', name, flush=True)
        try:
            with (output/'logs'/f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not marker.is_file(): raise RuntimeError('Missing completion marker: '+str(marker))
            # Hash stage products for reliable resume; arrays/meshes included.
            if name == 'upstream':
                # E00 depths, E09 surfaces and E10 direct points are siblings of
                # the completion marker; every downstream dependency must be signed.
                paths = [p for p in source.rglob('*') if p.is_file()]
            else:
                paths = [marker] if marker.suffix == '.json' and marker.name != 'manifest.json' else [p for p in marker.parent.rglob('*') if p.is_file()]
            state['completed'][name] = {'outputs': {str(p): sha256(p) for p in paths}}
        except Exception:
            state['status'] = 'failed'; write_json(state_path, state); raise
    evidence_status = 'deferred_by_configuration'
    if cfg.get('export_generation_evidence', True):
        evidence_manifest = read_json(output/'building_generation_inputs/manifest.json')
        evidence_status = evidence_manifest['status']
        state['generation_evidence_pending_count'] = len(evidence_manifest.get('pending', []))
    state.update(status='complete', release=str(output/'release'), generation_evidence=evidence_status)
    write_json(state_path, state)
    return output/'release'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args(); run(args.config,args.output,args.resume)

if __name__ == '__main__': main()
