"""Conservative visible-background trimming with local predicted-surface protection.

All evidence is rebuilt from caller-owned cameras, masks, predicted depths and
direct instance points. This stage preserves building identities and heights;
it never imports evaluation code or reads a precomputed research decision mask.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from city_pipeline.runtime import read_json, sha256, write_json
from ._visibility.associations import associate, classify_hits
from ._visibility.boundary import gated_refine, parts, prepare_cells
from ._visibility.local_surface import LOCAL_CONFIG, build_queries, local_bands, reduce_view, restrict_masks
from ._visibility.projection import ViewEvidence
from ._visibility.weak_silhouette import extra_mask, refine


METHOD = 'local_surface_veto'
BOUNDARY_CONFIG = {
    'cell_size_m': 1., 'depth_tolerance_m': 3., 'background_margin_px': 3,
    'image_margin_px': 2, 'negative_fraction_per_cell': .6,
    'minimum_negative_patches': 2, 'height_uncertainty_m': 3.,
}
ASSOCIATION_CONFIG = {
    'quality_min': 50, 'iou_min': .65, 'containment_min': .85,
    'area_ratio_min': .5, 'area_ratio_max': 2., 'height_tolerance_m': 5.,
    'neighbor_overlap_area_m2': 20., 'neighbor_coverage_veto': .5,
}
WEAK_CONFIG = {'name': 'v92_associated_all5', 'policy': 'v92', 'samples': 5, 'release': 'associated'}
INPUT_KEYS = ('camera', 'strict_valid', 'source_ids', 'background_ids', 'predicted_depth')


def _prediction_path(value, root):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError('Prediction input paths must be nonempty')
    path = (Path(root) / value).resolve()
    if any(token in str(path).lower() for token in
           ('global_vectors', 'evaluation_gt', 'gt_', 'depth_linear_m.exr')):
        raise ValueError('Reference data is not a visibility input: ' + str(path))
    if not path.is_file():
        raise FileNotFoundError('Required visibility prediction input: ' + str(path))
    return path


def build_inventory(observations, source_run_root, data_root, inventory_path=None):
    """Resolve original prediction inputs; explicit inventory paths are local to it.

    Observation path fields remain relative to data_root in both raw and cached
    modes. Absolute caller paths are supported, including raw upstream outputs.
    The optional portable inventory contains a samples mapping with INPUT_KEYS.
    """
    data_root = Path(data_root).resolve()
    features = read_json(observations)['features']
    samples = {}
    if inventory_path is not None:
        inventory_path = Path(inventory_path).resolve()
        document = read_json(inventory_path)
        if document.get('uses_ground_truth', False) is not False:
            raise ValueError('Visibility inventory must contain prediction inputs only')
        if document.get('schema', 'city_pipeline.visibility_inputs.v1') != 'city_pipeline.visibility_inputs.v1':
            raise ValueError('Unsupported visibility inventory schema')
        for sample, row in document['samples'].items():
            samples[sample] = {key: str(_prediction_path(row[key], inventory_path.parent))
                               for key in INPUT_KEYS}
    else:
        manifest = Path(source_run_root).resolve() / 'e00_instance_aware_25d/manifest.json'
        outputs = {}
        for row in read_json(manifest)['samples']:
            if row['id'] in outputs:
                raise ValueError('Repeated upstream sample in visibility inventory')
            outputs[row['id']] = row['outputs']
        for feature in features:
            properties = feature['properties']
            sample = properties['sample_id']
            if sample in samples:
                continue
            if sample not in outputs:
                raise ValueError('Missing upstream visibility outputs for ' + sample)
            view = _prediction_path(properties['rgb_path'], data_root).parent
            rows = {'camera': view / 'camera.json', 'strict_valid': view / 'strict_valid_mask.png',
                    'source_ids': properties['source_instance_ids_path'],
                    'background_ids': outputs[sample]['v7b_class_ids'],
                    'predicted_depth': outputs[sample]['class_aware_depth_zc_m']}
            samples[sample] = {key: str(_prediction_path(value, data_root)) for key, value in rows.items()}
    expected = {feature['properties']['sample_id'] for feature in features}
    if set(samples) != expected:
        raise ValueError('Visibility inventory must exactly cover the observation samples')
    physical_patches = set()
    for sample in sorted(samples):
        patch, separator, view = sample.partition('__')
        if not separator or not patch or not view or patch in physical_patches:
            raise ValueError('The standard visibility recipe requires one view per physical patch')
        physical_patches.add(patch)
        camera = read_json(samples[sample]['camera'])
        bounds = np.asarray(camera.get('data_bounds_xy_world_m'), dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[:2] >= bounds[2:]):
            raise ValueError('Camera requires finite increasing world XY bounds')
    return dict(sorted(samples.items()))


def collect_input_paths(observations, source_run_root, data_root, inventory_path=None):
    """Return every actual source dependency for portable bundling/resume hashing."""
    observations = Path(observations).resolve()
    samples = build_inventory(observations, source_run_root, data_root, inventory_path)
    paths = {observations}
    paths.add(Path(inventory_path).resolve() if inventory_path is not None else
              Path(source_run_root).resolve() / 'e00_instance_aware_25d/manifest.json')
    paths.update(Path(value) for row in samples.values() for value in row.values())
    for feature in read_json(observations)['features']:
        properties = feature['properties']
        paths.add(_prediction_path(properties['building_point_cloud_npz_path'], data_root))
        if inventory_path is None:
            # RGB establishes the adjacent camera/ROI location in automatic mode.
            paths.add(_prediction_path(properties['rgb_path'], data_root))
    return sorted(paths)


def outer_proposal(features):
    result = deepcopy(features)
    for feature in result:
        if feature['properties'].get('roof_layers'):
            continue
        old = shape(feature['geometry'])
        shells = unary_union([Polygon(part.exterior) for part in parts(old)])
        removal = old.difference(shells.buffer(-2.))
        holes = unary_union([Polygon(ring) for part in parts(old) for ring in part.interiors])
        if not holes.is_empty:
            removal = removal.difference(holes.buffer(.75))
        new = old.difference(removal)
        if not new.is_empty and new.is_valid:
            feature['geometry'] = mapping(new)
    return result


def weak_scope(cells, owners, parent, links, samples):
    """Rebuild all top-height weak-silhouette tests from original image arrays."""
    count = len(cells)
    arrays = {key: np.zeros(count, dtype=bool) for key in
              ('all_foreground', 'source_surface', 'peer_surface', 'neighbor_or_unknown')}
    arrays['v92_other_negative5'] = np.zeros(count, dtype=np.uint32)
    if not count:
        return np.zeros(0, dtype=bool), arrays
    xyz = np.concatenate([cell['xyz'] for cell in cells])
    canonical_patches = np.asarray(links['canonical_patch_by_owner'])
    canonical_instances = np.asarray([f['properties']['canonical_instance_id'] for f in parent])
    config = BOUNDARY_CONFIG
    for sample, paths in sorted(samples.items()):
        patch = sample.partition('__')[0]
        x0, y0, x1, y1 = read_json(paths['camera'])['data_bounds_xy_world_m']
        relevant = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
                    & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)).reshape(-1, 5).any(axis=1)
        selected = np.flatnonzero(relevant)
        if not len(selected):
            continue
        indices = (selected[:, None] * 5 + np.arange(5)).ravel()
        expected = np.repeat(owners[selected], 5)
        same_source = canonical_patches[expected] == patch
        evidence = ViewEvidence.load(paths, config['background_margin_px'], 'v92_explicit')
        native = np.unique(evidence.instance_ids[evidence.instance_ids > 0])
        owner_map = {int(k): int(v) for k, v in links['assignments'].get(sample, {}).items()}
        negative = np.ones(len(indices), dtype=bool)
        flags = {key: np.zeros(len(indices), dtype=bool) for key in arrays if key != 'v92_other_negative5'}
        for offset in (-config['height_uncertainty_m'], 0., config['height_uncertainty_m']):
            query = xyz[indices].copy()
            query[:, 2] += offset
            states = evidence.classify(query, native, config['depth_tolerance_m'], config['image_margin_px'])
            negative &= states['explicit_background']
            hits = classify_hits(states, owner_map, expected, same_source, canonical_instances[expected])
            for key in flags:
                flags[key] |= hits[key]
        for key in flags:
            arrays[key][selected] |= flags[key].reshape(-1, 5).any(axis=1)
        arrays['v92_other_negative5'][selected] += (negative.reshape(-1, 5).all(axis=1)
                                                  & (canonical_patches[owners[selected]] != patch))
    return extra_mask(arrays, WEAK_CONFIG), arrays


def collect_bands(cells, owners, parent, observations, links, data_root):
    cell_xyz = np.asarray([cell['xyz'] for cell in cells], dtype=float)
    points = cell_xyz.reshape(-1, 3)
    point_owners = np.repeat(owners, 5)
    records = [[] for _ in points]
    by_id = {row['properties']['observation_id']: row for row in observations}
    if len(by_id) != len(observations):
        raise ValueError('Repeated observation identity')
    selected = set(map(int, owners))
    plan = defaultdict(list)
    canonical_counts = defaultdict(int)
    for row in links['diagnostics']:
        owner = row['owner_index']
        if owner not in selected:
            continue
        observation = by_id[row['observation_id']]
        properties = observation['properties']
        canonical = properties['observation_id'] == parent[owner]['properties']['canonical_observation_id']
        canonical_counts[owner] += canonical
        path = _prediction_path(properties['building_point_cloud_npz_path'], data_root)
        plan[path].append((owner, observation, canonical))
    if any(canonical_counts[owner] != 1 for owner in selected):
        raise ValueError('Every scoped owner needs its one original canonical observation')
    summaries = []
    for path, rows in sorted(plan.items()):
        with np.load(path, allow_pickle=False) as data:
            xyz, native_ids, provenance = data['xyz_m'], data['instance_id'], data['provenance']
        for owner, observation, canonical in sorted(rows, key=lambda r: (r[0], r[1]['properties']['observation_id'])):
            properties = observation['properties']
            selected_points = np.flatnonzero(point_owners == owner)
            bands, direct_count = local_bands(xyz, native_ids, provenance, int(properties['rgb_instance_id']),
                                               points[selected_points, :2], LOCAL_CONFIG)
            for index, band in zip(selected_points, bands):
                records[index].append({**band, 'observation_id': properties['observation_id'],
                    'patch_id': properties['patch_id'], 'canonical': canonical,
                    'same_source_patch': properties['patch_id'] == parent[owner]['properties']['canonical_patch_id']})
            summaries.append({'observation_id': properties['observation_id'], 'owner_index': int(owner),
                              'direct_point_count': direct_count, 'reliable_bands': sum(b['reliable'] for b in bands)})
    return cell_xyz, records, summaries


def query_local_surfaces(queries, owners, parent, links, samples):
    xyz, cell_indices = queries['xyz'], queries['cell_index']
    canonical_patches = np.asarray(links['canonical_patch_by_owner'])[owners]
    canonical_ids = np.asarray([f['properties']['canonical_instance_id'] for f in parent])[owners]
    results = []
    for sample, paths in sorted(samples.items()):
        patch = sample.partition('__')[0]
        x0, y0, x1, y1 = read_json(paths['camera'])['data_bounds_xy_world_m']
        within = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1))
        # Keep every query of a relevant cell; dropping out-of-ROI samples would
        # let a partially unobserved cell manufacture a complete background vote.
        selected = np.flatnonzero(np.isin(cell_indices, np.unique(cell_indices[within])))
        if not len(selected):
            continue
        ci = cell_indices[selected]
        view = ViewEvidence.load(paths, BOUNDARY_CONFIG['background_margin_px'], 'v92_explicit')
        native = np.unique(view.instance_ids[view.instance_ids > 0])
        states = view.classify(xyz[selected], native, BOUNDARY_CONFIG['depth_tolerance_m'],
                               BOUNDARY_CONFIG['image_margin_px'])
        owner_map = {int(k): int(v) for k, v in links['assignments'].get(sample, {}).items()}
        hits = classify_hits(states, owner_map, owners[ci], canonical_patches[ci] == patch, canonical_ids[ci])
        results.append({'patch_id': patch, **reduce_view(states, hits, ci, len(owners))})
    return results


def run(input_root, observations, source_run_root, data_root, output_root, inventory_path=None, method=METHOD):
    if method != METHOD:
        raise ValueError('Only the standard local-surface-veto recipe is supported')
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    source_manifest = input_root / 'manifest.json'
    if read_json(source_manifest).get('status') != 'complete':
        raise ValueError('Visibility refinement needs complete building input')
    footprint_path = input_root / 'global_building_footprints.geojson'
    inputs = [source_manifest, footprint_path,
              *collect_input_paths(observations, source_run_root, data_root, inventory_path)]
    hashes = {str(path): sha256(path) for path in sorted(set(inputs))}
    parent = read_json(footprint_path)['features']
    observation_features = read_json(observations)['features']
    samples = build_inventory(observations, source_run_root, data_root, inventory_path)
    proposal = outer_proposal(parent)
    print('Rebuilding conservative visibility evidence', len(samples), 'physical views', flush=True)
    baseline, baseline_report = gated_refine(parent, proposal, samples, BOUNDARY_CONFIG)
    cells = prepare_cells(parent, proposal, BOUNDARY_CONFIG['cell_size_m'])
    links = associate(parent, observation_features, ASSOCIATION_CONFIG)
    owner_lookup = {gid: i for i, gid in enumerate(links['owner_ids'])}
    owners = np.asarray([owner_lookup[cell['building_id']] for cell in cells], dtype=np.int64)
    accepted, top_arrays = weak_scope(cells, owners, parent, links, samples)
    scope = np.flatnonzero(accepted)
    local_report = {'scoped_cells': len(scope), 'query_count': 0, 'accepted_cells': 0}
    final, refinement_report = baseline, {'changes': [], 'added_removed_area_m2': 0.}
    if len(scope):
        scoped_cells, scoped_owners = [cells[int(i)] for i in scope], owners[scope]
        cell_xyz, bands, band_summary = collect_bands(scoped_cells, scoped_owners, parent,
                                                    observation_features, links, data_root)
        queries = build_queries(cell_xyz, bands, LOCAL_CONFIG)
        views = query_local_surfaces(queries, scoped_owners, parent, links, samples)
        masks = restrict_masks(np.ones(len(scope), dtype=bool), queries['canonical_all_five_reliable'],
            np.asarray(links['canonical_patch_by_owner'])[scoped_owners], views, LOCAL_CONFIG)
        final_mask = np.zeros(len(cells), dtype=bool)
        final_mask[scope] = masks[METHOD]
        parameters = {**WEAK_CONFIG, 'name': METHOD, 'local_parameters': LOCAL_CONFIG, 'local_query_rule': METHOD}
        final, refinement_report = refine(parent, baseline, cells, final_mask, parameters)
        local_report.update(query_count=len(queries['xyz']), accepted_cells=int(final_mask.sum()),
            canonical_all_five_reliable=int(queries['canonical_all_five_reliable'].sum()),
            source_surface_cells=int(masks['source_surface'].sum()), peer_surface_cells=int(masks['peer_surface'].sum()),
            neighbor_or_unknown_cells=int(masks['neighbor_or_unknown'].sum()), observation_bands=band_summary)
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError('Visibility source changed during inference: ' + path)
    write_json(output_root / 'global_building_footprints.geojson', {'type': 'FeatureCollection', 'features': final})
    write_json(output_root / 'changes.json', {'conservative_visibility': baseline_report,
        'local_surface': refinement_report, 'local_evidence': local_report, 'association_counts': links['counts']})
    write_json(output_root / 'manifest.json', {'status': 'complete', 'uses_ground_truth': False,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'method': METHOD,
        'building_entity_count': len(final), 'physical_view_count': len(samples),
        'prediction_contract': {'selection_uses_ground_truth': False, 'heights_unchanged': True,
                                'identities_unchanged': True, 'existing_holes_and_roofs_preserved': True},
        'source_sha256': hashes, 'parameters': {'visibility': BOUNDARY_CONFIG,
            'association': ASSOCIATION_CONFIG, 'local_surface': LOCAL_CONFIG}, 'source_unchanged': True,
        'evidence_recomputed_from_original_predictions': True})
    return output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--source-run-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path)
    args = parser.parse_args()
    run(args.input_root, args.observations, args.source_run_root, args.data_root,
        args.output_root, args.inventory)


if __name__ == '__main__':
    main()
