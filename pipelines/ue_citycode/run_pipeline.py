#!/usr/bin/env python3
"""Plan or execute the published CPU city-assembly workflow.

Run without --execute to inspect commands. Windows imports are a separate,
required stage between prepare and finalize; this entry never starts UE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENRICHMENT = Path('data/smallcity_p384_scene_enrichment')
DEFAULT_BINDING = Path('data/smallcity_p384_building_assets_completed/binding_manifest_physical.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def resolve(path, root=ROOT):
    path = Path(path)
    return (root/path).resolve() if not path.is_absolute() else path.resolve()


def name(path, root):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


@dataclass
class Step:
    label: str
    argv: list[str]
    outputs: list[Path] = field(default_factory=list)
    # Each declared QA result must affirm success, even when its process exits 0.
    qa: list[tuple[Path, str, tuple[str, ...]]] = field(default_factory=list)


def configuration(args, root=ROOT):
    root = Path(root).resolve()
    run = resolve(args.run_root, root)
    if not run.is_relative_to(root/'build') or run == root/'build':
        raise ValueError('--run-root must be a dedicated directory below this release build/')
    previous = read(run/'pipeline_prepare.json') if args.stage == 'finalize' and (run/'pipeline_prepare.json').is_file() else {}
    saved = previous.get('configuration', {})
    config = dict(stage=args.stage, root=root, run=run,
        lod=args.lod or saved.get('lod', '0.25M'),
        binding=resolve(args.binding or saved.get('binding', DEFAULT_BINDING), root),
        source_obj=resolve(args.source_obj or saved.get('source_obj', 'build/local_epic/SM_FREEWAY_VIS_349_987f07977f.obj'), root),
        workers=args.workers if args.workers is not None else saved.get('workers', 3),
        native_catalog=resolve(args.native_catalog, root) if args.native_catalog else run/'windows/native_bridge_catalog.json',
        native_proposal=resolve(args.native_proposal, root) if args.native_proposal else run/'windows/native_bridge_proposal.json')
    if not 1 <= config['workers'] <= 64:
        raise ValueError('--workers must be between 1 and 64')
    return config


def build_plan(config):
    root, run = config['root'], config['run']
    e = root/ENRICHMENT
    base, surfaces = run/'base', run/'surfaces'
    final = config['stage'] == 'finalize'
    out = run/'final/enrichment' if final else run/'enrichment'
    qa_out = run/'final' if final else out
    template = run/'native/template_manifest.json'
    steps = []

    def add(label, script, *argv, outputs=(), qa=()):
        steps.append(Step(label, [sys.executable, str(root/'pipelines'/script), *map(str, argv)], list(outputs), list(qa)))

    binding = ['--binding', config['binding']]
    if not final:
        common = ['--root', root, '--lod', config['lod'], *binding, '--output', base]
        add('bake_buildings_and_background', 'ue_citycode/prepare.py', *common, '--bake', '--workers', config['workers'], outputs=[base/'assembly_manifest.json'])
        add('filter_old_elevated_background', 'ue_citycode/filter_background.py',
            '--assembly', base/'assembly_manifest.json', '--source', base/'interchange/BACKGROUND_COASTAL.glb',
            '--output', base/'interchange/BACKGROUND_COASTAL_GROUND.glb',
            '--mask', e/'prediction_elevated_filter_mask.npz', outputs=[base/'interchange/BACKGROUND_COASTAL_GROUND.filter.json'])
        parts = base/'interchange/background_parts/manifest.json'
        add('split_background', 'ue_citycode/split_background.py', '--input', base/'interchange/BACKGROUND_COASTAL_GROUND.glb',
            '--output-directory', parts.parent, '--manifest', parts,
            '--filter-proof', base/'interchange/BACKGROUND_COASTAL_GROUND.filter.json', outputs=[parts])
        add('bind_background_parts', 'ue_citycode/prepare.py', *common, '--background-parts', parts, outputs=[base/'assembly_manifest.json'])
        add('repair_predicted_road_network', 'scene_enrichment/repair_elevated_network.py',
            '--input', e, '--ground', e/'props_ground', '--building-assembly', base/'assembly_manifest.json',
            '--maximum-approach-lateral-shift', '6', '--inplace-landings', '--withhold-isolated-fragments',
            '--design-completions', '--output', run/'network', outputs=[run/'network/network_connections.json'])
        add('prepare_local_native_template', 'scene_enrichment/bridge_assets.py', '--source-obj', config['source_obj'],
            '--catalog', e/'ue_source_asset_catalog.json', '--output', run/'native', outputs=[template])
        add('sweep_and_join_road_surface', 'scene_enrichment/stitch_bridge_network.py',
            '--proposal', run/'network/regularized_placement_proposal.json', '--connections', run/'network/network_connections.json',
            '--structure', e/'elevated_structure.json', '--template', template, '--output', surfaces,
            outputs=[surfaces/'joined/span_manifest.json'],
            qa=[(surfaces/'source_spans/span_geometry_qa.json', 'status', ('PASS',)),
                (surfaces/'joined/span_manifest.json', 'qa.boolean_status', ('PASS',))])
    enrichment_args = ['--network-proposal', surfaces/'surface_placement_proposal.json',
        '--centerlines', surfaces/'surface_centerlines.geojson', '--span-manifest', surfaces/'source_spans/span_manifest.json',
        '--joined-manifest', surfaces/'joined/span_manifest.json', '--template', template,
        '--catalog', e/'ue_asset_catalog.json', '--props-groups', e/'props_scene_review_candidate_groups.json',
        *binding, '--baseline-manifest', base/'assembly_manifest.json', '--output', out]
    if final:
        enrichment_args += ['--props-proposal', run/'enrichment/props_final_proposal.json',
            '--native-catalog', config['native_catalog'], '--native-proposal', config['native_proposal'],
            '--manifest-output', run/'final/assembly_candidate.json']
    add('finalize_verified_native_bindings' if final else 'place_rails_and_supports',
        'ue_citycode/prepare_elevated_repair.py', *enrichment_args,
        outputs=[out/'bridge_railings_proposal.json', out/'bridge_supports_proposal.json'])
    if final:
        scene = run/'final/citycode'
        add('compile_portable_citycode', 'ue_citycode/citycode.py', 'compile', '--root', root,
            '--config', root/'workflow/04_citycode_assembly/citycode.standard.json', '--output', scene,
            '--binding', config['binding'], '--assembly', run/'final/assembly_candidate.json',
            '--catalog', run/'final/assembly_candidate.catalog.json',
            '--elevated-centerlines', surfaces/'surface_centerlines.geojson',
            '--elevated-proposal', surfaces/'surface_placement_proposal.json',
            '--network', run/'network/network_connections.json',
            '--props-groups', run/'enrichment/props_final_groups.json', outputs=[scene/'manifest.json'])
        add('export_citycode_to_ue', 'ue_citycode/citycode.py', 'export-ue', '--root', root,
            '--scene', scene, '--output', run/'final/assembly_manifest.json',
            outputs=[run/'final/assembly_manifest.json', run/'final/assembly_manifest.catalog.json'])
        add('validate_complete_cpu_scene', 'ue_citycode/validate_cpu.py', '--root', root,
            '--manifest', run/'final/assembly_manifest.json', '--output', run/'final/cpu_validation.json',
            qa=[(run/'final/cpu_validation.json', 'status', ('passed',))])
    scene_args = ['--spans', surfaces/'joined/span_manifest.json', '--supports', out/'bridge_supports_proposal.json',
        '--railings', out/'bridge_railings_proposal.json', *binding]
    if not final:
        add('review_street_props', 'scene_enrichment/review_scene_props.py',
            '--proposal', e/'props_reviewed_proposal.json', '--groups', e/'props_reviewed_groups.json', *scene_args,
            '--review-lamps', '--review-blocked-cars', '--maximum-along-curb-m', '6', '--withhold-unresolved',
            '--output-prefix', run/'enrichment/props_final', outputs=[run/'enrichment/props_final_proposal.json', run/'enrichment/props_final_groups.json'])
    add('audit_street_props', 'scene_enrichment/props_scene_qa.py',
        '--proposal', run/'enrichment/props_final_proposal.json', '--groups', run/'enrichment/props_final_groups.json',
        *scene_args, '--output', qa_out/'props_final_qa.json',
        qa=[(qa_out/'props_final_qa.json', 'status', ('passed_with_explicit_contact_limits',))])
    add('audit_actual_building_intersections', 'ue_citycode/audit_elevated_geometry.py',
        '--source-spans', surfaces/'source_spans/span_manifest.json', '--joined', surfaces/'joined/span_manifest.json',
        '--buildings', base/'assembly_manifest.json', '--proposal', surfaces/'surface_placement_proposal.json',
        '--ground', e/'props_ground', '--output', qa_out/'building_qa.json',
        qa=[(qa_out/'building_qa.json', 'status', ('no_actual_surface_crossings_or_sampled_containment_found',))])
    add('audit_actual_landing_contacts', 'ue_citycode/audit_elevated_landings.py',
        '--source-spans', surfaces/'source_spans/span_manifest.json', '--proposal', surfaces/'surface_placement_proposal.json',
        '--buildings', base/'assembly_manifest.json', '--output', qa_out/'landing_qa.json',
        qa=[(qa_out/'landing_qa.json', 'status', ('passed',))])
    return steps


def required_inputs(config):
    root, e = config['root'], config['root']/ENRICHMENT
    paths = [config['binding'], root/'workflow/04_citycode_assembly/standard.json',
        root/'workflow/04_citycode_assembly/citycode.standard.json',
        root/'data/smallcity_p384_structure/release/global_building_footprints.geojson', e/'ue_asset_catalog.json',
        e/'props_scene_review_candidate_groups.json', e/'props_asset_catalog.jsonl', e/'props_ground/BACKGROUND_SAMPLING.json',
        root/'data/smallcity_p384_background_coastal/land.geojson', root/'data/smallcity_p384_background_history/height/completed',
        root/'data/smallcity_p384_gt_roads/source/road_sections.geojson', root/'data/smallcity_p384_gt_roads/source/lane_centerlines.geojson',
        root/'data/smallcity_p384_gt_roads/ground_only/road_graph.json']
    paths += [e/'props_ground'/f for f in ('continuous_top.npy', 'detail_near_ground_top.npy', 'detail_max_top.npy')]
    if config['stage'] == 'prepare':
        paths += [config['source_obj'], e/'ue_source_asset_catalog.json', e/'props_reviewed_proposal.json', e/'props_reviewed_groups.json',
            e/'prediction_elevated_filter_mask.npz', e/'elevated_structure.json', e/'regularized_placement_proposal.json',
            e/'prediction_evidence_grid.npz', root/'data/smallcity_p384_background_coastal/background.glb',
            root/'data/smallcity_p384_background_coastal/background.json']
    else:
        paths += [config['run']/'pipeline_prepare.json', config['native_catalog'], config['native_proposal']]
    return paths


def check_qa(step):
    for path, field_name, accepted in step.qa:
        value = read(path)
        for key in field_name.split('.'):
            value = value.get(key) if isinstance(value, dict) else None
        if value not in accepted:
            raise ValueError(f'{step.label}: {path.name} {field_name}={value!r}; required {accepted}. Agent review is required before continuing.')


def snapshot_paths(run):
    return [run/p for p in ('base/assembly_manifest.json', 'native/template_manifest.json',
        'surfaces/surface_placement_proposal.json', 'surfaces/surface_centerlines.geojson',
        'surfaces/source_spans/span_manifest.json', 'surfaces/joined/span_manifest.json',
        'enrichment/props_final_proposal.json', 'enrichment/props_final_groups.json',
        'enrichment/bridge_railings_proposal.json', 'enrichment/bridge_supports_proposal.json',
        'enrichment/props_final_qa.json', 'enrichment/building_qa.json', 'enrichment/landing_qa.json')]


def verify_native_import(config):
    catalog, proposal = read(config['native_catalog']), read(config['native_proposal'])
    if catalog.get('status') != 'native_bridge_template_imported' or any(catalog.get(k) is not True for k in ('ue_executed', 'bounds_verified', 'native_materials_verified')):
        raise ValueError('finalize requires real successful Windows native mesh import, bounds and material verification')
    expected = digest(config['run']/'surfaces/joined/span_manifest.json')
    assets = catalog.get('assets', [])
    if not assets or any(row.get('template_manifest_sha256') != expected for row in assets) or proposal.get('template_manifest_sha256') != expected:
        raise ValueError('Windows native import does not match this run joined-span manifest')


def preflight(config, steps):
    missing = [str(p) for p in required_inputs(config) if not p.exists()]
    missing += [s.argv[1] for s in steps if not Path(s.argv[1]).is_file()]
    if missing:
        raise FileNotFoundError('Required inputs missing:\n'+'\n'.join(missing))
    if config['stage'] == 'prepare':
        occupied = [config['run']/p for p in ('base', 'network', 'native', 'surfaces', 'enrichment', 'final', 'pipeline_prepare.json', 'pipeline_finalize.json') if (config['run']/p).exists()]
        if occupied:
            raise ValueError('Prepare refuses to overwrite an existing run; select a new --run-root: '+', '.join(map(str, occupied)))
    else:
        if (config['run']/'final').exists() or (config['run']/'pipeline_finalize.json').exists():
            raise ValueError('Finalize refuses to overwrite an existing final result; select a new run instead')
        previous = read(config['run']/'pipeline_prepare.json')
        if previous.get('status') != 'prepared_cpu_checks_passed':
            raise ValueError('This run has no successfully completed CPU prepare stage')
        saved = previous['configuration']
        if config['lod'] != saved['lod'] or config['binding'] != resolve(saved['binding'], config['root']):
            raise ValueError('Finalize LOD/binding must match the completed prepare stage')
        for filename, expected in previous['frozen_files'].items():
            if digest(resolve(filename, config['root'])) != expected:
                raise ValueError('Prepared input changed before finalize: '+filename)
        verify_native_import(config)
    missing_modules = [m for m in ('numpy', 'scipy', 'shapely', 'trimesh', 'manifold3d', 'PIL') if importlib.util.find_spec(m) is None]
    if missing_modules:
        raise ValueError('Install the CPU dependencies with pip install -e \'.[scene-assembly]\': '+', '.join(missing_modules))


def execute(config, steps, runner=None):
    """Run argv directly; never invoke a shell or interpret user paths as code."""
    preflight(config, steps)
    runner = runner or subprocess.run
    root, run = config['root'], config['run']
    run.mkdir(parents=True, exist_ok=True)
    if config['stage'] == 'finalize':
        (run/'final').mkdir()
    record_path = run/('pipeline_'+config['stage']+'.json')
    record = {'schema': 'citycode.cpu_pipeline_run.v1', 'stage': config['stage'], 'status': 'running', 'ue_executed': False,
        'pipeline_code_sha256': digest(__file__),
        'configuration': {k: name(config[k], root) if isinstance(config[k], Path) else config[k] for k in ('lod', 'binding', 'source_obj', 'workers')},
        'completed_steps': [], 'active_step': None}
    def save():
        record_path.write_text(json.dumps(record, indent=2)+'\n')
    # Claim this stage before any producer runs; concurrent invocations cannot
    # both overwrite the same outputs after passing their initial preflight.
    with record_path.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    try:
        for step in steps:
            record['active_step'] = step.label; save()
            print('Running '+step.label, flush=True)
            runner(step.argv, cwd=str(root), check=True)
            for path in step.outputs:
                if not path.is_file():
                    raise FileNotFoundError(step.label+' did not create '+str(path))
            check_qa(step)
            record['completed_steps'].append(step.label); save()
        if config['stage'] == 'prepare':
            frozen = snapshot_paths(run)+[config['binding'], root/ENRICHMENT/'ue_asset_catalog.json',
                root/ENRICHMENT/'props_scene_review_candidate_groups.json', root/ENRICHMENT/'props_asset_catalog.jsonl',
                root/ENRICHMENT/'props_ground/BACKGROUND_SAMPLING.json',
                root/'data/smallcity_p384_background_coastal/land.geojson',
                root/'data/smallcity_p384_gt_roads/source/road_sections.geojson',
                root/'data/smallcity_p384_gt_roads/source/lane_centerlines.geojson',
                root/'data/smallcity_p384_gt_roads/ground_only/road_graph.json']
            record['frozen_files'] = {name(path, root): digest(path) for path in frozen}
            record['status'] = 'prepared_cpu_checks_passed'
            record['next_step'] = 'Import surfaces/joined in the existing Windows UE project; copy real native_bridge_catalog.json and native_bridge_proposal.json into this run/windows, then run finalize.'
        else:
            record['status'] = 'finalized_cpu_checks_passed'
            record['final_manifest'] = name(run/'final/assembly_manifest.json', root)
            record['citycode_manifest'] = name(run/'final/citycode/manifest.json', root)
            record['next_step'] = 'Run the separate Windows assembly/import entry, independent saved-map readback, and visual review.'
        record['active_step'] = None; save()
    except Exception as error:
        record.update(status='failed', error=str(error)); save()
        raise
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'finalize'))
    p.add_argument('--run-root', type=Path, default=Path('build/ue_citycode_replay'))
    p.add_argument('--source-obj', type=Path, help='Licensed local City Sample mesh OBJ; prepare only')
    p.add_argument('--lod', choices=('0.1M', '0.25M', '1M'), help='Default 0.25M; finalize inherits the prepared run')
    p.add_argument('--binding', type=Path, help='Published physical binding; finalize inherits the prepared run')
    p.add_argument('--workers', type=int, help='CPU bake workers, default 3')
    p.add_argument('--native-catalog', type=Path, help='Actual successful Windows native import result for finalize')
    p.add_argument('--native-proposal', type=Path, help='Actual Windows native placement proposal for finalize')
    p.add_argument('--execute', action='store_true', help='Execute the plan; otherwise print only without creating files')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = configuration(args, ROOT)
        steps = build_plan(config)
        if not args.execute:
            print('PLAN ONLY: '+args.stage+'; no commands executed or files created. Paths resolve from '+str(ROOT))
            for step in steps:
                print('\n'+step.label+'\n  '+shlex.join(step.argv))
            missing = [str(p) for p in required_inputs(config) if not p.exists()]
            if missing:
                print('\nRequired before --execute:\n'+'\n'.join(missing))
            return 0
        result = execute(config, steps)
        print(json.dumps({k: result[k] for k in ('status', 'ue_executed', 'next_step')}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print('CPU pipeline stopped: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
