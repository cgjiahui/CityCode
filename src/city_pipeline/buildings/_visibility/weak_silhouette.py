"""Test weak associated silhouettes without weakening real surfaces or neighbors."""
from __future__ import annotations
import copy
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
from .boundary import parts


def extra_mask(arrays, config):
    # source_only means the canonical source PATCH. Explicitly associated
    # instances within that same patch belong to the correlated source class;
    # unassociated/ambiguous instances remain protected neighbors/unknowns.
    if config['policy'] not in ('v7b', 'v92') or config['samples'] not in (3, 5):
        raise ValueError('Unrecognized explicit background rule')
    if config['release'] not in ('source_only', 'associated'):
        raise ValueError('Unrecognized weak silhouette rule')
    negative = arrays[config['policy']+'_other_negative'+str(config['samples'])] >= 2
    # "surface" is depth-consistent predicted evidence, never a GT surface.
    protected = arrays['neighbor_or_unknown'] | arrays['source_surface'] | arrays['peer_surface']
    if config['release'] == 'source_only': protected |= arrays['peer_silhouette']
    # Only study cells formerly vetoed by a building silhouette. Other new
    # background cells are not an unannounced change to the parent recipe.
    return arrays['all_foreground'] & negative & ~protected


def refine(parent, baseline, cells, accepted, config):
    parent_by_id = {f['properties']['predicted_global_building_id']: f for f in parent}
    base_by_id = {f['properties']['predicted_global_building_id']: f for f in baseline}
    if (len(parent_by_id) != len(parent) or len(base_by_id) != len(baseline)
            or set(parent_by_id) != set(base_by_id)):
        raise ValueError('Boundary experiment must retain all identities')
    if np.asarray(accepted).shape != (len(cells),): raise ValueError('Cell evidence mismatch')
    for gid, f in parent_by_id.items():
        p, b = f['properties'], base_by_id[gid]['properties']
        if any(p.get(k) != b.get(k) for k in ('canonical_observation_id', 'top_z_m', 'body_top_z_m',
                                            'base_z_m', 'height_m', 'roof_layers')):
            raise ValueError('Baseline changed identity/height/roof')
        if shape(base_by_id[gid]['geometry']).difference(shape(f['geometry'])).area > 1e-7:
            raise ValueError('Baseline must be shrink-only')
    removals = defaultdict(list)
    for cell, include in zip(cells, accepted):
        if include: removals[cell['building_id']].append(cell['proposed_removal'])
    result = copy.deepcopy(baseline); changes = []; rejected = []
    for f in result:
        p = f['properties']; gid = p['predicted_global_building_id']
        if gid not in removals: continue
        old = shape(parent_by_id[gid]['geometry']); base = shape(f['geometry'])
        original_parts = parts(old)
        removal = unary_union([old.difference(base), *removals[gid]])
        exterior = unary_union([Polygon(x.exterior) for x in original_parts]).boundary
        removal = unary_union([x for x in parts(removal) if x.distance(exterior) < 1e-7])
        if removal.is_empty: continue
        new = old.difference(removal)
        reason = None
        if not new.is_valid or new.is_empty: reason = 'invalid_or_empty'
        elif len(parts(new)) != len(original_parts): reason = 'component_count'
        elif old.area-new.area > .03*old.area: reason = 'total_area_limit'
        elif any(new.intersection(x).area < .95*x.area for x in original_parts): reason = 'component_retention'
        elif new.difference(base).area > 1e-7: reason = 'would_restore_baseline_trim'
        else:
            old_holes = unary_union([Polygon(r) for x in original_parts for r in x.interiors])
            new_holes = unary_union([Polygon(r) for x in parts(new) for r in x.interiors])
            if (sum(len(x.interiors) for x in parts(new)) != sum(len(x.interiors) for x in original_parts)
                    or old_holes.symmetric_difference(new_holes).area > 1e-7): reason = 'courtyard_change'
            elif any(shape(r['geometry']).difference(new).area > 1e-7 for r in p.get('roof_layers', [])):
                reason = 'roof_loss'
        if reason:
            rejected.append({'id': gid, 'reason': reason}); continue
        delta = base.area-new.area
        if delta < .05: continue
        f['geometry'] = mapping(new)
        p.update(footprint_area_m2=new.area, centroid_x_m=new.centroid.x, centroid_y_m=new.centroid.y)
        p['round4_weak_silhouette'] = {'method': 'explicit_background_over_weak_owned_silhouette',
                                      'parameters': config, 'removed_from_baseline_m2': delta}
        changes.append({'id': gid, 'removed_from_baseline_m2': delta})
    return result, {'accepted_weak_cells': int(np.asarray(accepted).sum()), 'changes': changes,
                    'rejected': rejected, 'added_removed_area_m2': sum(c['removed_from_baseline_m2'] for c in changes),
                    'GT_used': False, 'baseline_is_never_regrown': True}
