"""Replay frozen Agent-assisted asset decisions using published cached results.

This is a deterministic CPU assembly step. It neither re-runs the Agent nor
calls an image service, loads a generative model, or reads reference GT.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from .scene_completion import digest, validate_scene_completion_context

SCHEMA = 'citycode.reviewed_scene_completion.v1'
ACTIONS = {'retain_existing_model', 'regenerate_from_cached_image', 'regenerate_image_and_model'}
METADATA = {'updated_utc', 'scene_review', 'scene_completion', 'postprocessing', 'assembly_contract'}


def _file(root, record):
    relative = Path(record['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('Replay file must remain inside the published asset directory')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record.get('sha256'):
        raise ValueError('Replay file SHA256 changed: ' + record['path'])
    return data


def replay_scene_completion(assets_root, decisions_path, output_path):
    from shapely.geometry import shape
    from .cli import (ACCEPTED, PENDING, _policy, _validate_accepted,
                      _validate_asset_instances, _validate_building_entities, _validate_groups)
    from .postprocess_validation import validate_surface_instances

    root = Path(assets_root).resolve()
    output = Path(output_path).resolve()
    if output.parent != root or output.exists():
        raise ValueError('Replay output must be a new file in the asset directory')
    decisions = json.loads(Path(decisions_path).read_text())
    if decisions.get('schema') != SCHEMA:
        raise ValueError('Unsupported reviewed scene completion schema')
    original = json.loads(_file(root, decisions['input_manifest']))
    if original.get('schema') != 'citycode.building_asset_bindings.v1' or original.get('path_base') != 'manifest_directory':
        raise ValueError('Replay input must be a portable building binding manifest')
    result = copy.deepcopy(original)
    rows = {r['building_id']: r for r in result['buildings']}
    records = decisions['records']
    selected = {r['building_id']: r for r in records}
    if not records or len(selected) != len(records):
        raise ValueError('Reviewed records must be nonempty with unique source IDs')
    if set(decisions.get('output_metadata', {})) - METADATA:
        raise ValueError('Replay metadata cannot replace source or asset tables')
    provenance = decisions.get('review_provenance', {})
    if not provenance.get('decision_maker') or type(provenance.get('reference_data_used_for_review')) is not bool:
        raise ValueError('Review provenance must declare who decided and whether reference data was used')
    for sid, decision in selected.items():
        before = rows.get(sid)
        if before is None or before['status'] not in ACCEPTED | PENDING:
            raise ValueError('Reviewed source is absent or has an incompatible state')
        if digest(before) != decision.get('input_record_digest'):
            raise ValueError('Reviewed source record changed: ' + sid)
        if decision.get('action') not in ACTIONS or not decision.get('reason'):
            raise ValueError('Every reviewed source needs an explicit action and reason')
        after = copy.deepcopy(decision['building_record'])
        if after.get('building_id') != sid or after.get('status') not in ACCEPTED:
            raise ValueError('Completed building record must retain its source ID and accepted state')
        if after.get('structure_path') != before.get('structure_path'):
            raise ValueError('Replay cannot replace the original source structure')
        source = (root / before['structure_path']).resolve()
        if not source.is_relative_to(root.parent):
            raise ValueError('Original source must remain inside the published data tree')
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != decision.get('source_building_sha256'):
            raise ValueError('Original structure evidence changed: ' + sid)
        building = json.loads(payload)
        if (building['building_id'] != sid
                or not shape(building['footprint_world']).equals(shape(after['target_geometry']))
                or not shape(before['target_geometry']).equals(shape(after['target_geometry']))):
            raise ValueError('Replay must preserve the original predicted footprint')
        if not decision.get('evidence_files'):
            raise ValueError('Every reviewed source needs published visual evidence')
        for evidence in decision['evidence_files']:
            _file(root, evidence)
        if decision['action'] == 'retain_existing_model':
            if (after['lods'] != before.get('retained_candidate_lods', before.get('lods'))
                    or after['world_from_local_4x4'] != before['world_from_local_4x4']):
                raise ValueError('Retained model bytes or placement cannot change')
        else:
            _file(root, decision['completed_image'])
            generation = json.loads(_file(root, decision['generation_record']))
            if (generation.get('status') != 'complete' or generation.get('building_id') != sid
                    or generation['contract']['building_sha256'] != decision['source_building_sha256']
                    or generation['contract']['image_sha256'] != decision['completed_image']['sha256']
                    or after['provenance']['conditioning_image_sha256'] != decision['completed_image']['sha256']
                    or after['provenance']['raw_manifest_sha256'] != decision['generation_record']['sha256']):
                raise ValueError('Cached generated model is not bound to the reviewed source and image')
        _validate_accepted(root, after, _policy(original), geometry=False)
        rows[sid] = after
    # Surface changes affect default display only. Their original observations,
    # mesh, target geometry and placement remain in the published package.
    surfaces = {s['id']: s for s in result.get('surface_instances', [])}
    seen_surfaces = set()
    for decision in decisions.get('surface_decisions', []):
        identity = decision['id']
        if identity in seen_surfaces or identity not in surfaces or not decision.get('reason'):
            raise ValueError('Surface decisions require unique existing IDs and reasons')
        seen_surfaces.add(identity)
        surface = surfaces[identity]
        if digest(surface) != decision['input_record_digest'] or type(decision.get('assembly_enabled')) is not bool:
            raise ValueError('Surface decision does not match its frozen input')
        if len(surface['source_building_ids']) != 1:
            raise ValueError('Default surface decision needs one unambiguous source record')
        row = rows[surface['source_building_ids'][0]]
        surface['assembly_enabled'] = decision['assembly_enabled']
        row['postprocess']['surface']['assembly_enabled'] = decision['assembly_enabled']
        row['reason'] = decision['source_reason']
    result['buildings'] = [rows[r['building_id']] for r in result['buildings']]
    active = {r['id']: r for r in result['asset_instances']}
    for sid in sorted(selected):
        row = rows[sid]
        if sid in active and active[sid]['source_building_ids'] != [sid]:
            raise ValueError('A shared physical asset cannot be replaced one member at a time')
        active[sid] = {'id': sid, 'source_building_ids': [sid], 'status': row['status'],
                       'world_from_local_4x4': row['world_from_local_4x4'],
                       'lods': copy.deepcopy(row['lods']), 'building_entity_id': sid}
    result['asset_instances'] = list(active.values())
    entities = copy.deepcopy(result.get('building_entities', []))
    existing_parts = {aid for entity in entities for aid in entity['asset_instance_ids']}
    groups = decisions.get('building_entity_groups', [])
    grouped = [aid for group in groups for aid in group['asset_instance_ids']]
    if len(grouped) != len(set(grouped)) or set(grouped) - set(active):
        raise ValueError('Architectural groups need unique active parts')
    for entity in entities:
        members = set(entity['asset_instance_ids'])
        if members & set(grouped) and not members <= set(grouped):
            raise ValueError('An existing multipart entity cannot be split implicitly')
    entities = [e for e in entities if not set(e['asset_instance_ids']) & set(grouped)]
    for sid in sorted(set(selected) - set(grouped) - existing_parts):
        entities.append({'id': sid, 'asset_instance_ids': [sid], 'source_building_ids': [sid]})
    entities.extend(copy.deepcopy(groups))
    for entity in entities:
        for aid in entity['asset_instance_ids']:
            active[aid]['building_entity_id'] = entity['id']
    result['building_entities'] = entities
    result.update(copy.deepcopy(decisions.get('output_metadata', {})))
    result['counts'] = dict(Counter(r['status'] for r in result['buildings']))
    unfinished = any(r['status'] in PENDING for r in result['buildings'])
    result['status'] = 'in_progress' if unfinished else 'complete'
    result['fine_generation_complete'] = not unfinished
    _validate_groups(result, result['buildings'])
    count = _validate_asset_instances(result, result['buildings'])
    entities_count = _validate_building_entities(result)
    validate_surface_instances(result, result['buildings'])
    validate_scene_completion_context(result)
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != decisions.get('expected_output_manifest_sha256'):
        raise ValueError('Replayed manifest does not match the frozen reviewed result')
    with output.open('xb') as file:
        file.write(payload)
    return {'schema': 'citycode.scene_completion_replay.v1', 'status': 'passed',
            'output_sha256': actual, 'reviewed_instances': len(records),
            'physical_asset_instances': count, 'building_entities': entities_count,
            'review_reference_assisted': provenance['reference_data_used_for_review'],
            'reference_data_read_during_replay': False, 'API_calls': 0, 'GPU_jobs': 0,
            'source_or_model_bytes_modified': False}
