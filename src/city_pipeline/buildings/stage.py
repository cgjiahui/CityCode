"""Dispatch one standard reconstruction step in an isolated process."""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from . import selection as base
from .height_refinement import install_guard
from .parameters import normalize_selection, stage_parameters

STAGE_NAMES = (
    'consensus', 'roof_statistics', 'height', 'support_statistics', 'spatial',
    'spatial_export', 'boundary', 'prune', 'local_repair', 'replace_repair',
    'filter', 'final_export',
)
CLEANUP_INPUTS = {
    'boundary': '03_release',
    'prune': '05_boundary',
    'filter': '07_repair/trials/replace',
}
CLEANUP_OUTPUTS = {'boundary': '05_boundary', 'prune': '06_prune', 'filter': '08_filter'}


def invoke(module, arguments):
    """Adapt the existing module entry points without changing their algorithms."""
    previous = sys.argv
    try:
        sys.argv = [module.__name__, *map(str, arguments)]
        return module.main()
    finally:
        sys.argv = previous


def run_consensus_stage(args):
    from . import consensus
    selection = normalize_selection(json.loads(args.selection_parameters)
                                    if args.selection_parameters else None)
    invoke(consensus, [
        '--source-run-root', args.source, '--output-root', args.work / '01_consensus',
        '--consensus-weight', selection['consensus_weight'],
        '--area-weight', selection['area_weight'], '--min-area', selection['min_area_m2'],
        '--conflict-containment', selection['conflict_containment'],
    ])


def run_roof_statistics_stage(args):
    from . import roof_statistics
    invoke(roof_statistics, ['--output', args.work / 'roof_statistics.json'])


def run_height_stage(args):
    from . import height_refinement
    invoke(height_refinement, [
        '--input-root', args.work / '01_consensus', '--output-root', args.work / '02_height',
        '--trial', 'CAP1_P95', '--point-statistics', args.work / 'roof_statistics.json',
    ])


def run_spatial_stage(args):
    from . import spatial_support
    spatial_support.OBS = args.source / 'global_fused/building_observation_footprints.geojson'
    spatial_support.INPUT = args.work / '02_height/CAP1_P95'
    arguments = ['--cache', args.work / 'observed_support.json']
    if args.stage == 'support_statistics':
        arguments += ['--prepare']
    else:
        arguments += ['--output-root', args.work / '03_spatial', '--trials', 'C1_elevated_roof']
    invoke(spatial_support, arguments)


def run_export_stage(args):
    from . import export
    if args.stage == 'spatial_export':
        source = args.work / '03_spatial/C1_elevated_roof'
        output = args.work / '03_release'
    else:
        source = args.final_input or args.work / '08_filter'
        output = args.work.parent / 'release'
    # The intermediate export also normalizes overlapping roof layers. Keep
    # that behavior until roof normalization has its own validated operation.
    invoke(export, ['--trial-root', source, '--output-root', output])


def run_repair_stage(args, thresholds):
    from . import observation_repair
    observation_repair.ROOT = args.work / '07_repair'
    observation_repair.POOL = args.work / '05_boundary/global_building_footprints.geojson'
    observation_repair.OBS = args.source / 'global_fused/building_observation_footprints.geojson'
    name = 'local' if args.stage == 'local_repair' else 'replace'
    source = args.work / ('06_prune' if name == 'local' else '07_repair/trials/local')
    invoke(observation_repair, [
        '--input-root', source, '--extra-configs', json.dumps({name: thresholds}), '--configs', name,
    ])


def apply_boundary_inset(features, thresholds):
    """Apply the existing bounded inset and clip roof layers to the new body."""
    changes = []
    for feature in features:
        properties = feature['properties']
        geometry = shape(feature['geometry'])
        weak = (len(properties['consensus_observation_ids']) == 1
                or properties['canonical_boundary_status'] != 'interior_keep')
        if thresholds['scope'] == 'weak' and not weak:
            continue
        if thresholds['scope'] == 'single' and len(properties['consensus_observation_ids']) != 1:
            continue
        pieces = list(geometry.geoms) if geometry.geom_type == 'MultiPolygon' else [geometry]
        clean = []
        for polygon in pieces:
            inset = polygon.buffer(-thresholds['offset_m'], join_style=2)
            keep_original = (inset.is_empty or inset.geom_type != 'Polygon'
                             or inset.area < thresholds['minimum_component_area_retention'] * polygon.area)
            clean.append(polygon if keep_original else inset)
        refined = unary_union(clean)
        if geometry.symmetric_difference(refined).area < 1e-6:
            continue
        feature['geometry'] = mapping(refined)
        properties['footprint_area_m2'] = refined.area
        properties['centroid_x_m'] = refined.centroid.x
        properties['centroid_y_m'] = refined.centroid.y
        layers = []
        for layer in properties.get('roof_layers', []):
            clipped = shape(layer['geometry']).intersection(refined)
            if not clipped.is_empty and clipped.area >= .1:
                layers.append({**layer, 'geometry': mapping(clipped)})
        properties['roof_layers'] = layers
        if 'body_top_z_m' in properties:
            properties['top_z_m'] = max([properties['body_top_z_m']] + [x['top_z_m'] for x in layers])
            properties['height_m'] = properties['top_z_m'] - properties['base_z_m']
        changes.append({'id': properties['predicted_global_building_id'], 'action': 'boundary_inset'})
    return features, changes


def filter_features(features, observations, stage, thresholds, roof_retained=()):
    """Apply the original pruning or final size/height criteria in input order."""
    kept, changes = [], []
    for feature in features:
        properties = feature['properties']
        peers = [observations[key] for key in properties['consensus_observation_ids']]
        geometry = shape(feature['geometry'])
        if stage == 'prune':
            reject = (len(peers) == 1 and all(o['boundary_status'] != 'interior_keep' for o in peers)
                      and geometry.area <= thresholds['single_all_boundary_max_area_m2'])
        else:
            width = float(np.linalg.norm(
                np.diff(np.array(geometry.minimum_rotated_rectangle.exterior.coords), axis=0), axis=1).min())
            reject = (properties['height_m'] <= thresholds['height_le_m']
                      or (width < thresholds['width_lt_m'] and len(peers) == thresholds['support_equals']
                          and max(o['preliminary_quality'] for o in peers) < thresholds['quality_lt']))
        if stage == 'prune' and reject and properties['consensus_observation_ids'][0] in roof_retained:
            reject = False
            changes.append({'id': properties['predicted_global_building_id'],
                            'action': 'retain_with_predicted_roof_support'})
        if reject:
            changes.append({'id': properties['predicted_global_building_id'], 'action': 'delete'})
        else:
            kept.append(feature)
    return kept, changes


def prepare_roof_retention(args, features, observations, thresholds):
    if args.stage != 'prune' or not args.roof_retention:
        return set(), None
    from .roof_retention import observed_roof_retention
    config = json.loads(args.roof_retention)
    if config.get('prune_maximum_area_m2', 1200) != thresholds['single_all_boundary_max_area_m2']:
        raise ValueError('Roof retention must use the same prune area threshold')
    point_boundary = Path(os.path.commonpath([str(base.PROJECT_ROOT.resolve()), str(args.source.resolve())]))
    retained, report = observed_roof_retention(
        features, observations, base.PROJECT_ROOT, config, point_boundary_root=point_boundary)
    return set(retained), report


def write_cleanup_outputs(args, features, changes, thresholds, paths, audit, roof_report):
    output = args.work / CLEANUP_OUTPUTS[args.stage]
    output.mkdir(parents=True, exist_ok=False)
    payload = json.dumps({'type': 'FeatureCollection', 'features': features})
    (output / 'global_building_footprints.geojson').write_text(payload)
    (output / 'changes.json').write_text(json.dumps(changes, indent=2))
    manifest = {
        'status': 'complete', 'parameters': thresholds, 'building_count': len(features),
        'prediction_contract': 'Global prediction-side rules.',
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'prediction_sha256': hashlib.sha256(payload.encode()).hexdigest(),
        'runtime_read_audit': sorted(audit), 'roof_retention': roof_report,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(args.stage, len(features), len(changes), flush=True)


def run_cleanup_stage(args, thresholds):
    source = args.work / CLEANUP_INPUTS[args.stage] / 'global_building_footprints.geojson'
    observation_path = args.source / 'global_fused/building_observation_footprints.geojson'
    paths = [source, observation_path]
    audit = install_guard(paths, expand_observation_points=args.stage == 'prune' and bool(args.roof_retention))
    features = copy.deepcopy(json.loads(source.read_text())['features'])
    observations = {f['properties']['observation_id']: f['properties']
                    for f in json.loads(observation_path.read_text())['features']}
    retained, roof_report = prepare_roof_retention(args, features, observations, thresholds)
    if args.stage == 'boundary':
        features, changes = apply_boundary_inset(features, thresholds)
    else:
        features, changes = filter_features(features, observations, args.stage, thresholds, retained)
    write_cleanup_outputs(args, features, changes, thresholds, paths, audit, roof_report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGE_NAMES)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--final-input', type=Path)
    parser.add_argument('--parameters')
    parser.add_argument('--roof-retention')
    parser.add_argument('--selection-parameters')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base.SOURCE_RUN_ROOT = args.source
    thresholds = stage_parameters(args.stage, json.loads(args.parameters) if args.parameters else None)
    if args.stage == 'consensus':
        run_consensus_stage(args)
    elif args.stage == 'roof_statistics':
        run_roof_statistics_stage(args)
    elif args.stage == 'height':
        run_height_stage(args)
    elif args.stage in ('support_statistics', 'spatial'):
        run_spatial_stage(args)
    elif args.stage in ('spatial_export', 'final_export'):
        run_export_stage(args)
    elif args.stage in ('local_repair', 'replace_repair'):
        run_repair_stage(args, thresholds)
    else:
        run_cleanup_stage(args, thresholds)


if __name__ == '__main__':
    main()
