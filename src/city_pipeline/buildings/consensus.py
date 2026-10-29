#!/usr/bin/env python3
"""Prediction-only alternative-observation selection. No evaluator is imported.

Reads the frozen E10 observation polygons and optionally their predicted XYZ.
All thresholds are scene-global; no reference geometry/count is an input.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from shapely.geometry import shape

from . import selection as base


def install_read_guard(source, cache, export_rasters=False):
    """Enforce the data whitelist at runtime, including reads by imported helpers."""
    observation_path = (source / 'global_fused/building_observation_footprints.geojson').resolve()
    allowed = {observation_path, cache.resolve(),
               (base.SOURCE_RUN_ROOT / 'global_fused/manifest.json').resolve()}
    reads = set()

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1]
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(c in mode for c in 'wax'):
            return
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix in {'.py', '.pyc', '.so'}:
            return
        if path.is_relative_to(base.PROJECT_ROOT.resolve()):
            if path not in allowed:
                raise PermissionError(f'Prediction data read outside whitelist: {path}')
            reads.add(str(path))

    sys.addaudithook(guard)
    document = base.read_json(observation_path)
    for feature in document['features']:
        o = feature['properties']
        for key in ['building_point_cloud_npz_path', 'source_instance_ids_path', 'bev_instance_ids_path']:
            allowed.add((base.PROJECT_ROOT / o[key]).resolve())
    # Empty-view identity exports need the immutable source observation mapping.
    if export_rasters:
        manifest = base.read_json(base.SOURCE_RUN_ROOT / 'global_fused/manifest.json')
        for sample in manifest['identity_chain']['sample_records']:
            allowed.add((base.PROJECT_ROOT / sample['mapping']).resolve())
    return reads, hashlib.sha256(observation_path.read_bytes()).hexdigest()


def load_observations(source, params):
    path = source / 'global_fused/building_observation_footprints.geojson'
    candidates = []
    for f in base.read_json(path)['features']:
        o = dict(f['properties'])
        original = base.polygonal(shape(f['geometry']))
        g = base.clean_geometry(original, params)
        prop = {'predicted_global_building_id': o['observation_id'],
                'canonical_observation_id': o['observation_id'],
                'canonical_score': o['canonical_score'], 'member_count': 1}
        metrics = base.shape_metrics(g)
        score = base.candidate_score(prop, o, metrics, params)
        candidates.append(base.Candidate(len(candidates), prop, o, original, g, score, metrics))
    return candidates


def point_evidence(candidates, cache):
    """Robust observed roof support, excluding filled/interpolated points."""
    if cache.exists():
        return base.read_json(cache)
    grouped = defaultdict(list)
    for c in candidates:
        grouped[c.observation_properties['building_point_cloud_npz_path']].append(c)
    result = {}
    for path, items in sorted(grouped.items()):
        with np.load(base.PROJECT_ROOT / path) as data:
            xyz, ids, provenance = data['xyz_m'], data['instance_id'], data['provenance']
        for c in items:
            o = c.observation_properties
            points = xyz[(ids == int(o['rgb_instance_id'])) & (provenance == 1)]
            points = points[np.isfinite(points).all(axis=1)]
            top = float(o['top_z_m'])
            z = points[:, 2]
            near = z[np.abs(z - top) <= 3.0]
            result[c.observation_id] = {
                'observed_point_count': len(z),
                'roof_support_fraction': float(len(near) / max(len(z), 1)),
                'roof_residual_m': float(np.median(np.abs(near - top))) if len(near) else 3.0,
                'observed_z_q90': float(np.quantile(z, .90)) if len(z) else top,
            }
        print(f'XYZ evidence {len(result)}/{len(candidates)}', flush=True)
    base.write_json(cache, result)
    return result


def solve(candidates, params, consensus_weight, area_weight, point_weight, evidence):
    edges, raw = base.relation_graph(candidates, params)
    # Cross-patch agreement counts independent patches, never duplicate observations
    # of one patch. Height agreement is soft: roofs can have several real levels.
    support = [dict() for _ in candidates]
    for i, j, inter, containment, iou in raw:
        a, b = candidates[i], candidates[j]
        if a.observation_properties['patch_id'] == b.observation_properties['patch_id']:
            continue
        if iou < .35:
            continue
        dz = abs(float(a.observation_properties['top_z_m']) - float(b.observation_properties['top_z_m']))
        agreement = iou * np.exp(-dz / 12.0)
        for u, v in [(i, j), (j, i)]:
            patch = candidates[v].observation_properties['patch_id']
            support[u][patch] = max(support[u].get(patch, 0), float(agreement))
    for i, c in enumerate(candidates):
        agreement = sum(sorted(support[i].values(), reverse=True)[:3])
        c.metrics['cross_patch_agreement'] = agreement
        c.metrics['independent_support_patches'] = len(support[i])
        c.score += consensus_weight * agreement
        # Sublinear area reward reduces incentive to replace a complete building
        # with tiny mutually compatible fragments, without preferring city blocks.
        c.score *= (c.geometry.area / 400.0) ** area_weight
        if evidence:
            e = evidence[c.observation_id]
            c.metrics.update(e)
            c.score += point_weight * e['roof_support_fraction'] * min(e['observed_point_count'] / 1000, 1)
    pool = []
    for i, c in enumerate(candidates):
        ok, reason = base.eligible(c, params)
        c.decision = reason
        if ok:
            pool.append(i)
    index = {v: k for k, v in enumerate(pool)}
    pairs = [(index[a], index[b]) for a, b in sorted(edges) if a in index and b in index]
    rr = np.repeat(np.arange(len(pairs)), 2)
    cc = np.asarray(pairs, dtype=int).ravel()
    matrix = coo_matrix((np.ones(len(cc)), (rr, cc)), shape=(len(pairs), len(pool))).tocsc()
    scores = np.asarray([candidates[i].score for i in pool])
    res = milp(-scores, integrality=np.ones(len(pool)), bounds=Bounds(0, 1),
               constraints=LinearConstraint(matrix, -np.inf, 1),
               options={'time_limit': 120, 'mip_rel_gap': 1e-5})
    if res.x is None:
        raise RuntimeError(f'No feasible selection: {res.message}')
    chosen = {pool[k] for k in np.flatnonzero(res.x > .5)}
    if any(a in chosen and b in chosen for a, b in edges):
        raise AssertionError('Selected conflict pair')
    for i in pool:
        candidates[i].decision = 'selected' if i in chosen else 'conflict_suppressed'
    return sorted(chosen), {'solver_status': int(res.status), 'solver_message': res.message,
                            'mip_gap': float(res.mip_gap), 'conflict_edges': len(pairs),
                            'eligible_count': len(pool)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-run-root', type=Path, default=base.SOURCE_RUN_ROOT)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--consensus-weight', type=float, default=1.0)
    ap.add_argument('--area-weight', type=float, default=.25)
    ap.add_argument('--point-weight', type=float, default=0)
    ap.add_argument('--point-cache', type=Path)
    ap.add_argument('--min-area', type=float, default=24)
    ap.add_argument('--conflict-containment', type=float, default=.5)
    ap.add_argument('--export-glb', action='store_true')
    ap.add_argument('--export-rasters', action='store_true')
    args = ap.parse_args()
    root = args.output_root.resolve()
    if (root / 'manifest.json').exists():
        raise FileExistsError(f'Use a fresh output directory: {root}')
    cache = args.point_cache or root / 'point_evidence.json'
    reads, source_hash = install_read_guard(args.source_run_root, cache, args.export_rasters)
    params = base.Parameters(min_area_m2=args.min_area, severe_containment=args.conflict_containment)
    candidates = load_observations(args.source_run_root, params)
    evidence = point_evidence(candidates, cache) if args.point_weight else {}
    selected, stats = solve(candidates, params, args.consensus_weight, args.area_weight, args.point_weight, evidence)
    records = base.build_records(selected, candidates)
    gr, obs, gf, of = base.prediction_rows(records)
    outputs = base.export_glb(root, records, gr) if args.export_glb else {}
    samples = base.identity_rasters(root, records) if args.export_rasters else []
    for row, feature in zip(gr, gf['features']):
        for key in ['canonical_glb', 'canonical_glb_ue_z_up']:
            feature['properties'][key] = row[key]
    base.write_json(root / 'global_building_footprints.geojson', gf)
    base.write_json(root / 'building_observation_footprints.geojson', of)
    base.write_jsonl(root / 'global_buildings.jsonl', gr)
    base.write_csv(root / 'global_buildings.csv', gr)
    base.write_jsonl(root / 'building_observations.jsonl', obs)
    base.write_csv(root / 'observation_to_global.csv', obs)
    base.write_json(root / 'global_building_index.json', {str(r['index']): r['id'] for r in records})
    base.write_jsonl(root / 'selection_decisions.jsonl', [
        {'observation_id': c.observation_id, 'decision': c.decision, 'score': c.score, **c.metrics}
        for c in candidates])
    manifest = {'status': 'complete', 'schema': 'citycode.observation_consensus.v2',
                'prediction_contract': {'selection_uses_ground_truth': False,
                    'geometry_source': 'cleaned frozen E10 observation',
                    'height_source': 'same frozen E10 observation',
                    'inputs': ['building_observation_footprints.geojson'] +
                        (['referenced predicted E10 XYZ, instance_id, provenance'] if evidence else []),
                    'upstream_scale_prior': 'inherited fixed-smallcity background plane calibration'},
                'parameters': {**asdict(params), 'consensus_weight': args.consensus_weight,
                    'area_weight': args.area_weight, 'point_weight': args.point_weight},
                'source_observations': len(candidates), 'selected_entity_count': len(records),
                'source_observations_sha256': source_hash,
                'runtime_read_audit': {'enabled': True, 'project_data_reads': sorted(reads)},
                'optimization': stats, 'outputs': outputs, 'identity_chain': {'sample_records': samples}}
    base.write_json(root / 'manifest.json', manifest)
    print({'output': str(root), 'selected_entity_count': len(records),
           'runtime_project_data_reads': len(reads), **stats}, flush=True)


if __name__ == '__main__':
    main()
