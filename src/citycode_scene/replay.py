"""Apply agent-reviewed placement decisions without changing any model bytes."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
from shapely import affinity
from shapely.geometry import shape, mapping

from .geometry import json_write, sha256


def delta_matrix(translation=(0., 0., 0.), horizontal_scale=1., pivot=(0., 0.)):
    delta = np.eye(4)
    delta[0, 0] = delta[1, 1] = float(horizontal_scale)
    delta[:2, 3] = np.asarray(pivot) * (1-horizontal_scale) + np.asarray(translation[:2])
    delta[2, 3] = translation[2]
    return delta


def validate_delta(delta, policy):
    d = np.asarray(delta, float)
    if d.shape != (4, 4) or not np.isfinite(d).all() or not np.allclose(d[3], [0, 0, 0, 1]):
        raise ValueError('Finite affine 4x4 delta required')
    if not np.allclose(d[:3, :3], np.diag([d[0, 0], d[0, 0], 1])):
        raise ValueError('Only uniform horizontal scaling and translation are allowed')
    if not policy['minimum_horizontal_scale'] <= d[0, 0] <= 1:
        raise ValueError('Horizontal scale exceeds the reviewed policy')
    if abs(d[2, 3]) > policy['maximum_ground_shift_m']:
        raise ValueError('Vertical adjustment exceeds the reviewed policy')
    return d


def apply_decisions(binding_path, decision_path, output_path):
    binding_path, decision_path, output_path = map(lambda p: Path(p).resolve(), (binding_path, decision_path, output_path))
    if output_path == binding_path:
        raise ValueError('Keep the generated binding immutable; write a separate physical binding')
    if output_path.parent != binding_path.parent:
        raise ValueError('Derived binding must keep the same relative asset path base')
    source = json.loads(binding_path.read_text())
    recipe = json.loads(decision_path.read_text())
    if recipe['status'] != 'agent_reviewed_frozen':
        raise ValueError('Draft proposals need agent review before application')
    if sha256(binding_path) != recipe['input_binding_sha256']:
        raise ValueError('Input binding changed; old placement decisions cannot be applied to new inputs')
    for scene_input in recipe.get('scene_inputs', []):
        if sha256(binding_path.parent/scene_input['path']) != scene_input['sha256']:
            raise ValueError(f"Scene evidence changed: {scene_input['role']}")
    policy = recipe['policy']
    out = deepcopy(source)
    instances = out['asset_instances'] + out.get('massing_instances', [])
    by = {r['id']: r for r in instances}
    decisions = recipe['decisions']
    if len({r['id'] for r in decisions}) != len(decisions):
        raise ValueError('Each physical placement may have only one frozen decision')
    deletions = [r for r in decisions if r['action'] == 'exclude']
    if len(deletions) > policy['maximum_exclusions']:
        raise ValueError('Exclusion budget exceeded; preserve the city coverage')
    deltas = {}
    for decision in decisions:
        aid = decision['id']
        if aid not in by:
            raise ValueError(f'Unknown physical placement: {aid}')
        if not decision.get('reason') or not decision.get('evidence'):
            raise ValueError('Every changed placement needs a reason and measured evidence')
        row = by[aid]
        current = np.asarray(row['world_from_local_4x4'], float)
        if not np.allclose(current, decision['before_world_from_local_4x4'], atol=1e-9, rtol=0):
            raise ValueError(f'Placement changed before replay: {aid}')
        if row['lods']['0.1M']['sha256'] != decision['model_01m_sha256']:
            raise ValueError(f'Model changed before replay: {aid}')
        if sha256(binding_path.parent/row['lods']['0.1M']['path']) != decision['model_01m_sha256']:
            raise ValueError(f'Actual model bytes changed before replay: {aid}')
        if decision['action'] == 'exclude':
            if not decision.get('bounded_search_exhausted') or not decision.get('severe_vehicle_or_building_conflict'):
                raise ValueError('Deletion requires severe measured conflict and failed bounded correction')
            row['assembly_enabled'] = False
        elif decision['action'] == 'adjust':
            delta = validate_delta(decision['world_delta_4x4'], policy)
            from citycode_assets.registration import read_glb_scene_geometry
            geometry = read_glb_scene_geometry(binding_path.parent/row['lods']['0.1M']['path'])
            world_vertices = geometry['vertices'] @ current[:3,:3].T + current[:3,3]
            pivot = (world_vertices[:,:2].min(0)+world_vertices[:,:2].max(0))/2
            if not np.allclose(pivot, decision['pivot_xy_m'], rtol=0, atol=1e-5):
                raise ValueError('Displacement reference must be the actual original mesh centre')
            displacement = delta[:2, :2] @ pivot + delta[:2, 3] - pivot
            if np.linalg.norm(displacement) > policy['maximum_horizontal_shift_m'] + 1e-8:
                raise ValueError('Horizontal displacement exceeds the reviewed policy')
            row['world_from_local_4x4'] = (delta @ current).tolist()
            if 'normal_transform_world_3x3' in row:
                row['normal_transform_world_3x3'] = np.linalg.inv(np.asarray(row['world_from_local_4x4'])[:3,:3]).T.tolist()
            deltas[aid] = delta
            if 'actual_footprint' in row:
                row['actual_footprint'] = mapping(affinity.affine_transform(shape(row['actual_footprint']),
                    [delta[0, 0], delta[0, 1], delta[1, 0], delta[1, 1], delta[0, 3], delta[1, 3]]))
            # Target observations remain evidence, not a geometry to move with a model.
        else:
            raise ValueError('Only explicit adjust or exclude decisions are replayed')
        row['physical_postprocess'] = {'action': decision['action'], 'reason': decision['reason'],
            'decision_id': aid, 'recipe_sha256': sha256(decision_path)}
    # Multi-part entities receive the same world correction, preserving relative joints.
    for entity in out.get('building_entities', []):
        group = [aid for aid in entity['asset_instance_ids'] if aid in by and by[aid].get('assembly_enabled', True)]
        if len(group) > 1 and any(aid in deltas for aid in group):
            values = [deltas.get(aid, np.eye(4)) for aid in group]
            if not all(np.allclose(values[0], d, atol=1e-8, rtol=0) for d in values[1:]):
                raise ValueError(f'Common world correction required for grouped parts: {group}')
    # Shared source IDs still refer to one physical placement.
    for group in out.get('asset_instance_groups', []):
        if group['id'] in by:
            group['world_from_local_4x4'] = by[group['id']]['world_from_local_4x4']
    out['physical_postprocess'] = {'schema': 'citycode.scene_physical_replay.v1',
        'source_binding': binding_path.name, 'source_binding_sha256': sha256(binding_path),
        'decision_path': str(decision_path.relative_to(binding_path.parent)) if decision_path.is_relative_to(binding_path.parent) else str(decision_path),
        'decision_sha256': sha256(decision_path), 'adjusted_placements': len(deltas), 'excluded_placements': len(deletions),
        'active_placements': sum(r.get('assembly_enabled', True) for r in instances),
        'geometry_bytes_modified': False, 'source_structure_records_preserved': len(out['buildings'])}
    json_write(output_path, out)
    return out['physical_postprocess']
