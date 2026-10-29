"""Fuse predicted observation geometry/heights after observation identity selection.

No reference data is read. Source observations are assigned to at most one selected
entity by reciprocal footprint agreement before contributing to its consensus.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree

from . import selection as base
from .consensus import load_observations

DEFAULT_INPUT = Path("UNCONFIGURED_DEFAULT_INPUT")
TRIALS = {'CAP1_P95': {'height_quantile': 0.5, 'geometry_vote': 0.5, 'point_quantile': 'q95', 'upper_adjustment_cap_m': 1.0}}


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind='stable')
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w), q * w.sum()), len(v) - 1)])


def peer_groups(features, candidates, min_iou):
    selected = [shape(f['geometry']) for f in features]
    tree = STRtree(selected)
    groups = [[] for _ in selected]
    lookup = {c.observation_id: c for c in candidates}
    for c in candidates:
        matches = []
        for i in tree.query(c.geometry, predicate='intersects'):
            g = selected[i]
            area = g.intersection(c.geometry).area
            iou = area / (g.area + c.geometry.area - area)
            if iou >= min_iou:
                matches.append((iou, int(i)))
        if not matches:
            continue
        iou, i = max(matches)
        o = c.observation_properties
        if o['boundary_status'] == 'boundary_clipped_exclude':
            continue
        # Geometric identity evidence, not roof-height agreement: using the latter
        # here would lock in the possibly wrong height being corrected.
        quality = max(.1, float(o['preliminary_quality']) / 100)
        core = 1.0 if o['reliable_core_contains_centroid'] else .4
        edge = .6 if o['source_edge_distance_px'] == 0 else 1.0
        weight = iou ** 2 * quality * core * edge
        groups[i].append((c, weight))
    final = []
    for f, group in zip(features, groups):
        canonical_id = f['properties']['canonical_observation_id']
        by_patch = {}
        for c, w in group:
            patch = c.observation_properties['patch_id']
            if patch not in by_patch or w > by_patch[patch][1]:
                by_patch[patch] = (c, w)
        canonical = lookup[canonical_id]
        patch = canonical.observation_properties['patch_id']
        if patch not in by_patch:
            by_patch[patch] = (canonical, .5)
        final.append(sorted(by_patch.values(), key=lambda item: item[0].observation_id))
    return final


def voted_geometry(original, peers, threshold, trim_only):
    if len(peers) < 2 or threshold <= 0:
        return original
    gs = [c.geometry for c, w in peers]
    weights = np.asarray([w for c, w in peers])
    cells = polygonize(unary_union([g.boundary for g in gs]))
    kept = []
    for cell in cells:
        point = cell.representative_point()
        support = sum(w for g, w in zip(gs, weights) if g.covers(point))
        if support >= threshold * weights.sum():
            kept.append(cell)
    if not kept:
        return original
    result = unary_union(kept).intersection(original if trim_only else original.buffer(4))
    if result.is_empty or not .65 <= result.area / original.area <= 1.35:
        return original
    return base.clean_geometry(base.polygonal(result), base.Parameters())


def refine(features, groups, config, point_statistics=None):
    out = []
    for f, peers in zip(features, groups):
        p = dict(f['properties'])
        original = shape(f['geometry'])
        g = voted_geometry(original, peers, config.get('geometry_vote', 0), config.get('trim_only', False))
        opening = config.get('opening_m', 0)
        if opening:
            smooth = g.buffer(-opening, join_style=2).buffer(opening, join_style=2).intersection(g)
            if not smooth.is_empty and smooth.area >= .8 * g.area:
                g = base.polygonal(smooth)
        erosion = config.get('erosion_m', 0)
        if erosion:
            inset = g.buffer(-erosion, join_style=2)
            if not inset.is_empty and inset.area >= .8 * g.area:
                g = base.polygonal(inset)
        heights = [float(c.observation_properties['top_z_m']) for c, w in peers]
        if config.get('point_quantile'):
            heights = [point_statistics[c.observation_id][config['point_quantile']] for c, w in peers]
        weights = [w for c, w in peers]
        z = weighted_quantile(heights, weights, config['height_quantile'])
        if config.get('upper_adjustment_cap_m'):
            stable_top = weighted_quantile([c.observation_properties['top_z_m'] for c, w in peers], weights, .5)
            z = stable_top + float(np.clip(z - stable_top, 0, config['upper_adjustment_cap_m']))
        # Keep geometry identity and base-height convention fixed for this ablation.
        p['top_z_m'] = max(z, p['base_z_m'] + 1)
        p['height_m'] = p['top_z_m'] - p['base_z_m']
        p['height_method'] = ('independent_patch_observed_xyz_' + config['point_quantile']
                              if config.get('point_quantile') else 'independent_patch_weighted_roof_quantile')
        p['footprint_area_m2'] = g.area
        p['centroid_x_m'], p['centroid_y_m'] = g.centroid.x, g.centroid.y
        p['consensus_observation_ids'] = [c.observation_id for c, w in peers]
        p['consensus_weights'] = weights
        p['consensus_top_z_m'] = heights
        p['height_evidence_range_m'] = max(heights) - min(heights)
        p['input_canonical_top_z_m'] = f['properties']['top_z_m']
        # Old mesh URIs describe the unrefined geometry, so never publish them.
        p['canonical_glb'] = p['canonical_glb_ue_z_up'] = ''
        out.append({'type': 'Feature', 'properties': p, 'geometry': mapping(g)})
    return out


def install_guard(paths, expand_observation_points=False):
    allowed = {p.resolve() for p in paths}
    reads = set()
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode, flags = args[1:3]
        if isinstance(mode, str) and any(c in mode for c in 'wax'):
            return
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix in {'.py', '.pyc', '.so'}:
            return
        if path.is_relative_to(base.PROJECT_ROOT.resolve()):
            if path not in allowed:
                raise PermissionError(f'Non-prediction input: {path}')
            reads.add(str(path))
    sys.addaudithook(guard)
    if expand_observation_points:
        for path in paths:
            if path.name == 'building_observation_footprints.geojson':
                for f in base.read_json(path)['features']:
                    allowed.add((base.PROJECT_ROOT / f['properties']['building_point_cloud_npz_path']).resolve())
    return reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--min-peer-iou', type=float, default=.5)
    ap.add_argument('--trial', choices=list(TRIALS), nargs='+')
    ap.add_argument('--point-statistics', type=Path)
    ap.add_argument('--export-glb', action='store_true')
    args = ap.parse_args()
    pred_path = args.input_root / 'global_building_footprints.geojson'
    obs_path = base.SOURCE_RUN_ROOT / 'global_fused/building_observation_footprints.geojson'
    reads = install_guard([pred_path, obs_path] + ([args.point_statistics] if args.point_statistics else []))
    features = base.read_json(pred_path)['features']
    candidates = load_observations(base.SOURCE_RUN_ROOT, base.Parameters())
    groups = peer_groups(features, candidates, args.min_peer_iou)
    trials = {name: TRIALS[name] for name in args.trial} if args.trial else {
        name: cfg for name, cfg in TRIALS.items() if 'point_quantile' not in cfg}
    point_statistics = None
    if args.point_statistics:
        stats = base.read_json(args.point_statistics)
        assert stats['source_observations_sha256'] == hashlib.sha256(obs_path.read_bytes()).hexdigest()
        point_statistics = stats['observations']
    if any(cfg.get('point_quantile') for cfg in trials.values()) and point_statistics is None:
        raise ValueError('Point-based trials require --point-statistics from roof_statistics')
    for name, config in trials.items():
        root = args.output_root / name
        if root.exists():
            raise FileExistsError(root)
        refined = refine(features, groups, config, point_statistics)
        rows = [f['properties'] for f in refined]
        outputs = {}
        if args.export_glb:
            records = []
            lookup = {c.observation_id: c for c in candidates}
            for f in refined:
                p = f['properties']; old = lookup[p['canonical_observation_id']]
                c = base.Candidate(old.source_index, p, old.observation_properties,
                                   old.geometry, shape(f['geometry']), old.score, old.metrics)
                records.append({'id': p['predicted_global_building_id'],
                                'index': p['global_building_index'], 'candidate': c})
            outputs = base.export_glb(root, records, rows)
        base.write_json(root / 'global_building_footprints.geojson', {'type': 'FeatureCollection', 'features': refined})
        base.write_jsonl(root / 'global_buildings.jsonl', rows)
        base.write_json(root / 'manifest.json', {
            'status': 'complete', 'schema': 'air2vectorcity.prediction_consensus_refinement.v3',
            'parameters': {**config, 'min_peer_iou': args.min_peer_iou},
            'selected_entity_count': len(refined),
            'prediction_contract': {'selection_uses_ground_truth': False,
                'geometry_source': 'weighted consensus of assigned predicted E10 observations',
                'height_source': ('weighted quantile of assigned observed predicted XYZ upper quantiles'
                    if config.get('point_quantile') else 'weighted quantile of assigned predicted E10 roof heights'),
                'entity_selection': 'frozen input identities; no deletions'},
            'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [pred_path, obs_path] + ([args.point_statistics] if args.point_statistics else [])},
            'runtime_read_audit': sorted(reads), 'outputs': outputs,
        })
        print(name, len(refined), flush=True)


if __name__ == '__main__':
    main()
