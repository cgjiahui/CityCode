"""Audit the shipped example, optionally replay its frozen final decisions.

This checks cached results and conditioning inputs. It does not call an image
provider or regenerate neural geometry, and never reports a first-attempt rate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(root, proof):
    relative = Path(proof['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.parent):
        raise ValueError('Evidence escaped the published data tree')
    if not path.is_file() or path.stat().st_size != proof['bytes'] or sha256(path) != proof['sha256']:
        raise ValueError('Frozen file missing or changed: ' + proof['path'])
    return path


def asset_route(root, row):
    provenance = row.get('provenance', {})
    kind = provenance.get('kind')
    if row['status'] == 'accepted_repaired':
        return 'cross_lod_repair'
    if kind in {'historical_asset_refit', 'historical_shared_asset_refit'}:
        return 'historical_reuse'
    if kind == 'historical_components_refit':
        return 'composite_reuse'
    if kind == 'source_conditioned_fine_generation_with_observed_roof_reconciliation':
        return 'source_conditioned_refinement'
    if kind == 'image_led_scene_completion':
        return 'image_led_regeneration'
    if kind == 'generated_full_2_8':
        if provenance.get('placement_initialization'):
            return 'source_role_orientation_recovery'
        sidecar = root / Path(row['lods']['0.1M']['path']).parent / 'lod_manifest.json'
        if json.loads(sidecar.read_text()).get('low_lod_derivation'):
            return 'extrema_preserving_lod'
        return 'ordinary_generation_after_review'
    raise ValueError('Unclassified active asset provenance: ' + str(kind))


def replay_orientation(root, frozen):
    """Rebase only file proofs, then execute the approved source-role initializer."""
    import numpy as np
    from citycode_assets.orientation import source_supported_yaw_initialization

    portable = json.loads(checked(root, frozen['orientation_approval']).read_text())
    original = json.loads(checked(root, portable['original_approval']).read_text())
    def without_proof_paths(value):
        if isinstance(value, dict):
            return {k: without_proof_paths(v) for k, v in value.items()
                    if not (k == 'path' and set(value) == {'path', 'sha256', 'bytes'})}
        return [without_proof_paths(v) for v in value] if isinstance(value, list) else value
    if without_proof_paths(original) != without_proof_paths(portable['cue']):
        raise ValueError('Portable approval changed the approved correspondence')
    def rebase(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256', 'bytes'}:
                return {**value, 'path': str(checked(root, value))}
            return {k: rebase(v) for k, v in value.items()}
        return [rebase(v) for v in value] if isinstance(value, list) else value
    cue = rebase(portable['cue'])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=root/'reproduction', suffix='.json',
                                         prefix='.orientation_replay_', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(cue, stream, ensure_ascii=False)
        result = source_supported_yaw_initialization(
            portable['original_prior_world_from_asset_4x4'],
            json.loads(Path(cue['current_building']['path']).read_text()), temporary,
            approved_cue_sha256=sha256(temporary), evidence_root=root.parent)
        difference = float(np.max(np.abs(np.asarray(result['source_supported_prior_world_from_asset_4x4']) -
            portable['expected_source_supported_prior_world_from_asset_4x4'])))
        if difference > 1e-8:
            raise ValueError('Replayed source-role prior differs from the frozen accepted prior')
        return {'building_id': cue['building_id'], 'status': 'passed',
                'maximum_matrix_difference': difference,
                'source_role_initializer_reexecuted': True,
                'final_bounded_fit_reexecuted': False, 'model_bytes_modified': False}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replay_cpu_repairs(root, binding, frozen):
    """Regenerate the two reviewed CPU mesh exceptions into temporary outputs."""
    from citycode_assets.extrema_lod import export_extrema_preserving_lod
    from citycode_assets.mesh_repair import repair_lods
    inputs = {r['building_id']: r for r in frozen['generation_inputs']}
    decisions = json.loads((root/'postprocess_decisions.json').read_text())
    decisions = {r['building_id']: r for r in decisions['records']}
    results = []
    with tempfile.TemporaryDirectory(prefix='citycode_cpu_replay_') as folder:
        for row in binding['buildings']:
            if not row.get('lods') or row['status'] == 'resolved_fragment':
                continue
            route = asset_route(root, row)
            sid = row['building_id']
            if route == 'extrema_preserving_lod':
                fit = json.loads(checked(root, inputs[sid]['fit_record']).read_text())
                output = Path(folder)/(sid+'.glb')
                result = export_extrema_preserving_lod(root/row['lods']['0.25M']['path'], output, fit_record=fit)
                if result['status'] != 'accepted_candidate' or sha256(output) != row['lods']['0.1M']['sha256']:
                    raise ValueError('Extrema LOD replay differs from delivered model')
                results.append({'building_id': sid, 'operation': route, 'status': 'passed',
                                'model_files_byte_identical': 1})
            elif route == 'cross_lod_repair':
                decision = decisions[sid]
                recipe = checked(root, decision['repair_recipe'])
                originals = row['postprocess']['repair']['original_lods']
                result = repair_lods(checked(root, originals['0.25M']), checked(root, originals['0.1M']),
                    row['world_from_local_4x4'], Path(folder)/sid, recipe,
                    expected_recipe_sha256=decision['repair_recipe']['sha256'])
                if any(result['outputs'][lod]['sha256'] != value['sha256'] for lod, value in row['lods'].items()):
                    raise ValueError('Cross-LOD repair replay differs from delivered model')
                results.append({'building_id': sid, 'operation': route, 'status': 'passed',
                                'model_files_byte_identical': 2})
    return results


def audit(root, *, verify_models=False, replay=False, orientation=False, cpu_repairs=False):
    root = Path(root).resolve()
    frozen = json.loads((root/'reproduction/manifest.json').read_text())
    manifest_path = checked(root, frozen['binding_manifest'])
    binding = json.loads(manifest_path.read_text())
    source = json.loads(checked(root, frozen['source_manifest']).read_text())
    by_source = {r['building_id']: r for r in binding['buildings']}
    source_ids = {r['building_id'] for r in source['records']}
    if source_ids != set(by_source) or source_ids != {r['building_id'] for r in frozen['source_buildings']}:
        raise ValueError('The upstream, frozen input and binding ID sets differ')
    for proof in frozen['source_buildings']:
        building = json.loads(checked(root, proof).read_text())
        if building['building_id'] != proof['building_id']:
            raise ValueError('Upstream building identity changed')
    for proof in frozen['packaged_files']:
        checked(root, proof)
    from citycode_assets.appearance import completion_contract
    generated = {r['building_id']: r for r in frozen['generation_inputs']}
    if len(generated) != len(frozen['generation_inputs']):
        raise ValueError('Duplicate frozen generation input')
    for sid, inputs in generated.items():
        image = checked(root, inputs['completed_image'])
        prepared = json.loads(checked(root, inputs['prepared_input']).read_text())
        receipt = json.loads(checked(root, inputs['completion_receipt']).read_text())
        building = root / by_source[sid]['structure_path']
        if (prepared['building_id'] != sid or receipt['status'] != 'complete'
                or prepared['provenance']['building_sha256'] != sha256(building)
                or receipt['request_identity_sha256'] != completion_contract(prepared)
                or receipt['output_sha256'] != sha256(image)):
            raise ValueError('Conditioning image is not bound to current inputs: ' + sid)
        if inputs.get('fit_record'):
            fit = json.loads(checked(root, inputs['fit_record']).read_text())
            if (inputs['fit_record']['sha256'] != by_source[sid]['provenance']['fit_record_sha256']
                    or fit['world_from_asset_4x4'] != by_source[sid]['world_from_local_4x4']):
                raise ValueError('Fit record differs from accepted placement: ' + sid)
    routes = Counter()
    assets = {}
    model_proofs = {}
    normal_repairs = set()
    expected_neural = set()
    for instance in binding['asset_instances']:
        row = by_source[instance['source_building_ids'][0]]
        route = asset_route(root, row)
        routes[route] += 1
        assets[instance['id']] = (instance, route)
        if row.get('provenance', {}).get('normal_repair_applied'):
            normal_repairs.add(instance['id'])
        if 'generation' in route or route in {'cross_lod_repair', 'extrema_preserving_lod',
                'source_role_orientation_recovery', 'source_conditioned_refinement'}:
            expected_neural.add(row['building_id'])
    if expected_neural != set(generated):
        raise ValueError('Frozen conditions do not cover every active neural asset')
    for row in binding['buildings']:
        for value in row.get('lods', {}).values():
            proof = {k: value[k] for k in ['path', 'sha256', 'bytes']}
            if proof['path'] in model_proofs and model_proofs[proof['path']] != proof:
                raise ValueError('Conflicting model identities')
            model_proofs[proof['path']] = proof
    if verify_models:
        for proof in model_proofs.values():
            checked(root, proof)
    inventory = []
    for sid, row in by_source.items():
        identity = row.get('asset_instance_id', '')
        instance, route = assets.get(identity, ({}, row['status']))
        inputs = generated.get(sid, {})
        inventory.append({'building_id': sid, 'status': row['status'], 'asset_instance_id': identity,
            'building_entity_id': instance.get('building_entity_id', ''), 'route': route,
            'conditioning_profile': inputs.get('structure_conditioning', ''),
            'appearance_directory': inputs.get('appearance_directory', ''),
            'normal_repair': identity in normal_repairs})
    report = {'schema': 'citycode.example_reproducibility_audit.v1', 'status': 'passed',
        'binding_sha256': sha256(manifest_path), 'source_instances': len(source_ids),
        'accepted_source_instances': sum(len(a['source_building_ids']) for a in binding['asset_instances']),
        'physical_building_assets': len(assets), 'logical_building_entities': len(binding['building_entities']),
        'source_status_counts': dict(Counter(r['status'] for r in by_source.values())),
        'physical_asset_routes': dict(routes), 'normal_repaired_physical_assets': len(normal_repairs),
        'source_files_verified': len(frozen['source_buildings']), 'generation_conditions_verified': len(generated),
        'conditioning_profiles': dict(Counter(r['structure_conditioning'] for r in generated.values())),
        'model_files_verified': len(model_proofs) if verify_models else 0,
        'all_input_only_generation_revalidated': False, 'first_attempt_success_count': None,
        'scope': 'Verified cached assets/conditions and frozen decisions, not neural regeneration from stage01 alone',
        'API_calls': 0, 'GPU_jobs': 0}
    if orientation:
        report['orientation_replay'] = replay_orientation(root, frozen)
    if cpu_repairs:
        report['cpu_mesh_repairs_replay'] = replay_cpu_repairs(root, binding, frozen)
    if replay:
        from citycode_assets.completion_replay import replay_scene_completion
        output = root / ('binding_manifest.audit-' + uuid.uuid4().hex + '.json')
        try:
            report['binding_replay'] = replay_scene_completion(root,
                root/'scene_completion/replay/decisions.json', output)
            if sha256(output) != sha256(manifest_path):
                raise ValueError('Final binding replay differs from the delivered binding')
            report['binding_replay']['starts_from_completed_physical_assets'] = 531
            report['binding_replay']['scope'] = 'Final 11 reviewed decisions over cached baseline; not full neural generation'
        finally:
            output.unlink(missing_ok=True)
    return report, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-models', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--replay-orientation', action='store_true')
    parser.add_argument('--replay-cpu-repairs', action='store_true')
    args = parser.parse_args()
    report, inventory = audit(args.assets_root, verify_models=args.verify_models,
                              replay=args.replay, orientation=args.replay_orientation, cpu_repairs=args.replay_cpu_repairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
