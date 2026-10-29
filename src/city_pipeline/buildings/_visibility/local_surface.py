"""Additional prediction-only local-surface protection for frozen weak cells.

No geometry edits, reference data, or building-specific exceptions live here.
The legacy top-height evidence is retained by intersecting with its acceptance
mask. Local points only supply extra query surfaces, never replacement heights.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


LOCAL_CONFIG = {
    'radius_m': 2., 'provenance': 1, 'minimum_points': 20,
    'maximum_q90_minus_q10_m': 6., 'query_margin_m': 3.,
    'points_per_cell': 5, 'minimum_other_background_patches': 2,
}


def local_bands(xyz, instance_ids, provenance, instance_id, query_xy,
                config=LOCAL_CONFIG):
    """Finite direct points from this exact image instance, within XY radius."""
    xyz = np.asarray(xyz)
    ids, provenance = np.asarray(instance_ids), np.asarray(provenance)
    query_xy = np.asarray(query_xy, dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or ids.shape != (len(xyz),)
            or provenance.shape != ids.shape or query_xy.ndim != 2
            or query_xy.shape[1] != 2 or not np.isfinite(query_xy).all()):
        raise ValueError('Malformed direct-point inputs')
    if instance_id <= 0:
        raise ValueError('A local band requires a foreground instance')
    direct = xyz[(ids == instance_id) & (provenance == config['provenance'])
                 & np.isfinite(xyz).all(axis=1)]
    tree = cKDTree(direct[:, :2]) if len(direct) else None
    neighbors = (tree.query_ball_point(query_xy, config['radius_m'])
                 if tree is not None else [[] for _ in query_xy])
    bands = []
    for indices in neighbors:
        values = direct[indices, 2]
        record = {'point_count': len(values), 'reliable': False}
        if len(values):
            q10, q50, q90 = np.quantile(values, [.1, .5, .9])
            record.update(q10_m=float(q10), q50_m=float(q50), q90_m=float(q90),
                          q90_minus_q10_m=float(q90-q10),
                          reliable=bool(len(values) >= config['minimum_points']
                                        and q90-q10 <= config['maximum_q90_minus_q10_m']))
        bands.append(record)
    return bands, len(direct)


def build_queries(cell_xyz, point_bands, config=LOCAL_CONFIG):
    """Require a reliable canonical band at all five XYs; query every reliable band.

    point_bands is ordered cell-major, then point-major. An associated peer is
    extra protection; it cannot rescue an unknown canonical sample. Multiple
    bands from a patch remain correlated and do not produce negative votes.
    """
    cell_xyz = np.asarray(cell_xyz, dtype=float)
    npoint = config['points_per_cell']
    if (cell_xyz.ndim != 3 or cell_xyz.shape[1:] != (npoint, 3)
            or not np.isfinite(cell_xyz).all()
            or len(point_bands) != len(cell_xyz)*npoint):
        raise ValueError('Expected aligned cells with five finite XYZ samples')
    point_ready = []
    for records in point_bands:
        oids = [r['observation_id'] for r in records]
        canonical = [r for r in records if r['canonical']]
        if len(set(oids)) != len(oids) or len(canonical) != 1:
            raise ValueError('Each point needs one unique canonical observation')
        point_ready.append(bool(canonical[0]['reliable']))
        for r in records:
            if r['reliable']:
                q = [r['q10_m'], r['q50_m'], r['q90_m']]
                if (not np.isfinite(q).all() or q != sorted(q)
                        or r['point_count'] < config['minimum_points']
                        or q[2]-q[0] > config['maximum_q90_minus_q10_m']):
                    raise ValueError('Invalid reliable local band')
    ready = np.asarray(point_ready, dtype=bool).reshape(-1, npoint).all(axis=1)
    xyz, cell_index, point_index, band_index, height_index = [], [], [], [], []
    for ci in np.flatnonzero(ready):
        for pi in range(npoint):
            for bi, band in enumerate(point_bands[ci*npoint+pi]):
                if not band['reliable']:
                    continue
                heights = (band['q10_m']-config['query_margin_m'],
                           band['q50_m'], band['q90_m']+config['query_margin_m'])
                for hi, z in enumerate(heights):
                    xyz.append([*cell_xyz[ci, pi, :2], z])
                    cell_index.append(ci); point_index.append(pi)
                    band_index.append(bi); height_index.append(hi)
    return {'xyz': np.asarray(xyz, dtype=float).reshape(-1, 3),
            'cell_index': np.asarray(cell_index, dtype=np.int64),
            'point_index': np.asarray(point_index, dtype=np.uint8),
            'band_index': np.asarray(band_index, dtype=np.int64),
            'height_index': np.asarray(height_index, dtype=np.uint8),
            'canonical_all_five_reliable': ready}


def reduce_view(states, hits, query_cell_index, cell_count):
    """Any local surface/unknown foreground vetoes; background requires every query.

    A query outside ROI, without depth, or occluded cannot supply a negative
    vote. A visible owned silhouette without matching depth remains weak, as
    in the unchanged legacy associated-silhouette rule.
    """
    indices = np.asarray(query_cell_index, dtype=int)
    if (indices.ndim != 1 or np.any(indices < 0)
            or np.any(indices >= cell_count)):
        raise ValueError('Invalid query-to-cell mapping')
    n = len(indices)
    required_states = ('explicit_background', 'valid_depth_and_roi', 'occluded')
    required_hits = ('source_surface', 'peer_surface', 'neighbor_or_unknown')
    if any(np.asarray(states[k]).shape != (n,) for k in required_states):
        raise ValueError('State/query mismatch')
    if any(np.asarray(hits[k]).shape != (n,) for k in required_hits):
        raise ValueError('Hit/query mismatch')
    query_count = np.bincount(indices, minlength=cell_count)
    result = {'query_count': query_count}
    for k in required_hits:
        value = np.zeros(cell_count, dtype=bool)
        np.logical_or.at(value, indices, np.asarray(hits[k], dtype=bool))
        result[k] = value
    valid_background = (np.asarray(states['explicit_background'], dtype=bool)
                        & np.asarray(states['valid_depth_and_roi'], dtype=bool)
                        & ~np.asarray(states['occluded'], dtype=bool))
    count = np.bincount(indices, weights=valid_background, minlength=cell_count)
    result['all_queries_explicit_background'] = (query_count > 0) & (count == query_count)
    result['invalid_or_occluded_query_count'] = np.bincount(
        indices, weights=(~np.asarray(states['valid_depth_and_roi'], dtype=bool)
                          | np.asarray(states['occluded'], dtype=bool)),
        minlength=cell_count).astype(np.int64)
    return result


def restrict_masks(legacy_accepted, canonical_ready, canonical_patches, view_results,
                   config=LOCAL_CONFIG):
    """Add only restrictions to the old mask, with one vote per other physical patch."""
    legacy = np.asarray(legacy_accepted, dtype=bool)
    ready = np.asarray(canonical_ready, dtype=bool)
    canonical_patches = np.asarray(canonical_patches)
    if legacy.ndim != 1 or ready.shape != legacy.shape or canonical_patches.shape != legacy.shape:
        raise ValueError('Cell evidence mismatch')
    source = np.zeros(len(legacy), dtype=bool)
    peer = source.copy(); neighbor = source.copy()
    negative = np.zeros(len(legacy), dtype=np.uint16)
    seen = set()
    for row in view_results:
        patch = row['patch_id']
        if patch in seen:
            raise ValueError('The frozen inventory must have one input view per physical patch')
        seen.add(patch)
        if any(np.asarray(row[k]).shape != legacy.shape for k in (
                'source_surface', 'peer_surface', 'neighbor_or_unknown',
                'all_queries_explicit_background')):
            raise ValueError('View/cell evidence mismatch')
        source |= row['source_surface']; peer |= row['peer_surface']
        neighbor |= row['neighbor_or_unknown']
        negative += (row['all_queries_explicit_background'] & (canonical_patches != patch))
    veto = source | peer | neighbor
    a = legacy & ready & ~veto
    b = a & (negative >= config['minimum_other_background_patches'])
    if np.any(a & ~legacy) or np.any(b & ~a):
        raise AssertionError('Local evidence broadened the legacy acceptance scope')
    return {'local_surface_veto': a, 'local_surface_background': b,
            'canonical_unknown': ~ready, 'source_surface': source,
            'peer_surface': peer, 'neighbor_or_unknown': neighbor,
            'other_explicit_background_patch_count': negative}
