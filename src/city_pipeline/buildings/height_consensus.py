"""Prediction-only, bounded consensus of one height vote per input patch.

Depends only on NumPy, Shapely and the standard
library; it neither opens files nor accepts GT/evaluation/object-selector inputs.
World heights and all distance parameters are metres. Geometry and identity are
immutable; roof-layer entities and base priors remain unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
from shapely.geometry import shape
from shapely.strtree import STRtree

DEFAULT_CONFIG = {
    'profile': 'height-consensus-v1',
    'peer_iou_min': .5,
    'minimum_patch_count': 2,
    'huber_delta_m': 4.,
    'point_upper_adjustment_cap_m': 1.,
    'maximum_adjustment_m': 3.,
    'minimum_observation_quality': 40.,
    'minimum_direct_points': 200,
    'minimum_body_height_m': 2.,
    'change_tolerance_m': .01,
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalize_config(config=None):
    supplied = dict(config or {})
    unsupported = set(supplied) - set(DEFAULT_CONFIG)
    if unsupported:
        raise ValueError('Unsupported configuration keys: ' + ', '.join(sorted(unsupported)))
    cfg = {**DEFAULT_CONFIG, **supplied}
    for key in set(DEFAULT_CONFIG) - {'profile'}:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Finite numeric parameter required: ' + key)
    if not 0 < cfg['peer_iou_min'] <= 1 or cfg['huber_delta_m'] <= 0:
        raise ValueError('Invalid peer IoU or Huber radius')
    if not 0 <= cfg['minimum_observation_quality'] <= 100:
        raise ValueError('Quality threshold must be within [0,100]')
    for key in ['minimum_patch_count', 'minimum_direct_points']:
        if int(cfg[key]) != cfg[key] or cfg[key] < (2 if key == 'minimum_patch_count' else 1):
            raise ValueError('Invalid count parameter: ' + key)
        cfg[key] = int(cfg[key])
    for key in ['point_upper_adjustment_cap_m', 'maximum_adjustment_m',
                'minimum_body_height_m', 'change_tolerance_m']:
        if cfg[key] < 0:
            raise ValueError('Negative distance parameter: ' + key)
    return cfg


def huber_location(values, delta_m):
    """Equal-patch Huber M estimate, deterministic lower-median initialization."""
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('At least one finite height is required')
    # Match the existing weighted quantile convention for an even vote count.
    location = float(np.sort(values, kind='stable')[(len(values) - 1) // 2])
    for _ in range(30):
        residual = np.abs(values - location)
        weights = np.minimum(1., float(delta_m) / np.maximum(residual, 1e-8))
        updated = float(np.average(values, weights=weights))
        if abs(location - updated) < 1e-6:
            return updated
        location = updated
    return location


def _eligible(observation, cfg):
    if observation.get('boundary_status') == 'boundary_clipped_exclude':
        return False
    for key in ['top_z_m', 'preliminary_quality', 'source_edge_distance_px']:
        value = observation.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            return False
    return cfg['minimum_observation_quality'] <= observation['preliminary_quality'] <= 100


def _peer_groups(features, observation_features, cfg):
    geometries = [shape(feature['geometry']) for feature in features]
    if any(not geom.is_valid or geom.is_empty or geom.area <= 0 for geom in geometries):
        raise ValueError('Input footprint must be a nonempty valid polygon')
    groups = [[] for _ in features]
    observations = {}
    if not features:
        return groups
    tree = STRtree(geometries)
    for feature in observation_features:
        ob = feature['properties']
        oid = ob['observation_id']
        if oid in observations:
            raise ValueError('Duplicate observation identity')
        observations[oid] = ob
        if not _eligible(ob, cfg):
            continue
        geom = shape(feature['geometry'])
        if not geom.is_valid or geom.is_empty or geom.area <= 0:
            continue
        owners = []
        for index in tree.query(geom, predicate='intersects'):
            candidate = geometries[index]
            intersection = candidate.intersection(geom).area
            iou = intersection / max(candidate.area + geom.area - intersection, 1e-8)
            if iou >= cfg['peer_iou_min']:
                owners.append((iou, int(index)))
        if owners:
            iou, index = max(owners)
            groups[index].append((ob, iou))
    for feature, group in zip(features, groups):
        canonical_id = feature['properties']['canonical_observation_id']
        if canonical_id not in observations:
            raise ValueError('Canonical observation is missing from supplied prediction input')
        canonical = observations[canonical_id]
        if _eligible(canonical, cfg) and not any(ob['observation_id'] == canonical_id for ob, _ in group):
            # Existing canonical identity survives small boundary-repair IoU changes.
            group.append((canonical, .5))
    return groups


def _patch_votes(group, point_statistics, cfg):
    by_patch = {}
    for ob, iou in group:
        stats = point_statistics.get(ob['observation_id'])
        if not isinstance(stats, dict):
            continue
        q95, count = stats.get('q95'), stats.get('observed_count')
        if isinstance(q95, bool) or not isinstance(q95, (int, float)) or not np.isfinite(q95):
            continue
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not np.isfinite(count) or count < 0:
            continue
        stable_top = float(ob['top_z_m'])
        if count < cfg['minimum_direct_points']:
            q95 = stable_top
        top = stable_top + float(np.clip(q95 - stable_top, 0., cfg['point_upper_adjustment_cap_m']))
        quality = max(.1, float(ob['preliminary_quality']) / 100)
        core = 1. if ob['reliable_core_contains_centroid'] else .4
        edge = .6 if ob['source_edge_distance_px'] == 0 else 1.
        ranking_weight = iou ** 2 * quality * core * edge
        vote = {'observation_id': ob['observation_id'], 'patch_id': ob['patch_id'],
                'top_vote_m': top, 'stable_top_z_m': stable_top, 'point_q95_m': float(q95),
                'direct_point_count': int(count), 'preliminary_quality': float(ob['preliminary_quality']),
                'peer_iou': iou, 'ranking_weight': ranking_weight}
        old = by_patch.get(ob['patch_id'])
        if old is None or ranking_weight > old['ranking_weight']:
            by_patch[ob['patch_id']] = vote
    return sorted(by_patch.values(), key=lambda vote: vote['observation_id'])


def refine_height_consensus(prediction_features, observation_features, point_statistics, config=None):
    """Return (copied features, report) from already-predicted E10 observations.

    prediction_features: GeoJSON features with unique prediction ID, canonical
      observation ID, base_z_m/top_z_m/height_m and optional roof_layers.
    observation_features: E10 predicted footprint features; each observation has
      patch_id, observation_id, top_z_m, preliminary_quality, boundary_status,
      reliable_core_contains_centroid and source_edge_distance_px.
    point_statistics: mapping[observation_id] -> {q95, observed_count}; q95 is
      world Z of finite direct predicted E10 XYZ (provenance=1). No GT depth.
    config: global numeric parameters only; no per-object selection mechanism.

    At least two distinct patches contribute equally to the Huber estimate.
    Quality/core/edge influence which observation represents one patch, not the
    weight of that patch against other patches. Low-point statistics fall back to
    the stable predicted height. Heights change by at most maximum_adjustment_m
    relative to the original input top, while respecting the disclosed base.
    """
    cfg = normalize_config(config)
    result = copy.deepcopy(prediction_features)
    ids = [feature['properties']['predicted_global_building_id'] for feature in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate predicted building identity')
    groups = _peer_groups(result, observation_features, cfg)
    changes, protected, unsupported, infeasible_minimum = [], [], [], []
    for feature, group in zip(result, groups):
        props = feature['properties']
        gid = props['predicted_global_building_id']
        if props.get('roof_layers'):
            protected.append(gid)
            continue
        votes = _patch_votes(group, point_statistics, cfg)
        if len(votes) < cfg['minimum_patch_count']:
            unsupported.append(gid)
            continue
        base, current_top = float(props['base_z_m']), float(props['top_z_m'])
        if not np.isfinite([base, current_top]).all() or current_top < base:
            raise ValueError('Invalid predicted base/top height')
        geometry_hash, evidence_hash = _hash(feature['geometry']), _hash(votes)
        previous = props.get('height_consensus', {})
        same_input = (previous.get('configuration') == cfg and
                      previous.get('geometry_sha256') == geometry_hash and
                      previous.get('evidence_sha256') == evidence_hash and
                      abs(previous.get('result_top_z_m', float('inf')) - current_top) < 1e-8)
        original_top = float(previous['original_top_z_m']) if same_input else current_top
        estimate = huber_location([vote['top_vote_m'] for vote in votes], cfg['huber_delta_m'])
        adjustment = float(np.clip(estimate - original_top,
                                  -cfg['maximum_adjustment_m'], cfg['maximum_adjustment_m']))
        lower = max(base + cfg['minimum_body_height_m'], original_top - cfg['maximum_adjustment_m'])
        upper = original_top + cfg['maximum_adjustment_m']
        if lower > upper:
            # A minimum-body prior cannot override the explicit change bound.
            infeasible_minimum.append(gid)
            continue
        new_top = float(np.clip(original_top + adjustment, lower, upper))
        if abs(new_top - current_top) < cfg['change_tolerance_m']:
            continue
        props['top_z_m'] = new_top
        props['height_m'] = new_top - base
        if 'body_top_z_m' in props:
            props['body_top_z_m'] = new_top
        props['height_method'] = 'independent_patch_equal_weight_huber_consensus'
        props['height_consensus'] = {'configuration': cfg, 'original_top_z_m': original_top,
            'estimate_z_m': estimate, 'result_top_z_m': new_top,
            'geometry_sha256': geometry_hash, 'evidence_sha256': evidence_hash,
            'independent_patch_count': len(votes), 'votes': votes}
        for key in ['canonical_glb', 'canonical_glb_ue_z_up', 'coarse_glb', 'coarse_glb_ue_z_up']:
            props[key] = ''
        props['structure_mesh_nodes'] = []
        changes.append({'prediction_id': gid, 'original_top_z_m': original_top,
                        'new_top_z_m': new_top, 'patch_count': len(votes)})
    report = {'profile': cfg['profile'], 'parameters': cfg, 'entity_count': len(result),
        'changed_entity_count': len(changes), 'protected_roof_layer_entities': len(protected),
        'insufficient_patch_support_entities': len(unsupported),
        'infeasible_minimum_height_entities': len(infeasible_minimum),
        'geometry_and_ids_unchanged': True, 'base_priors_unchanged': True,
        'requires_mesh_regeneration': bool(changes), 'configured_individual_ids': False,
        'changes': changes}
    return result, report


def main(argv=None):
    """Isolated final-height stage; export regenerates meshes after this step."""
    import argparse
    from pathlib import Path
    from .height_refinement import install_guard
    from city_pipeline.runtime import read_json, write_json, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--point-statistics', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.input_root / 'global_building_footprints.geojson'
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [source, args.observations, args.point_statistics, args.config]
    from .polish import FORBIDDEN_PATH_PARTS
    if any(any(token in p.as_posix().lower() for token in FORBIDDEN_PATH_PARTS) for p in paths):
        raise PermissionError('Height consensus requires prediction inputs')
    reads = install_guard(paths + [output / 'global_building_footprints.geojson'])
    doc, observations, statistics, cfg = map(read_json, paths)
    if statistics.get('source_observations_sha256') != sha256(args.observations):
        raise ValueError('Point statistics belong to different predicted observations')
    result, report = refine_height_consensus(doc['features'], observations['features'], statistics['observations'], cfg)
    output.mkdir(parents=True)
    prediction = output / 'global_building_footprints.geojson'
    write_json(prediction, {**doc, 'features': result})
    write_json(output / 'manifest.json', {'status': 'complete', 'schema': 'citycode.height_consensus.v1',
              **report, 'source_sha256': {str(p): sha256(p) for p in paths},
              'prediction_sha256': sha256(prediction), 'runtime_read_audit': sorted(reads),
              'prediction_contract': 'Predicted geometry, observations and direct-XYZ summaries only. No GT or per-building configured decisions.'})
    print('Height consensus:', len(result), 'entities,', report['changed_entity_count'], 'updated', flush=True)
    return output


if __name__ == '__main__':
    main()
