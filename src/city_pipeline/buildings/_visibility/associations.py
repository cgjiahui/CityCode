"""Conservative current-geometry associations; no historic global-ID joins."""
from collections import Counter

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


def associate(features, observations, config):
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in config.values()):
        raise ValueError('Association configuration must be finite numeric values')
    if (not 0 < config['iou_min'] <= 1 or not 0 < config['containment_min'] <= 1
            or not 0 < config['area_ratio_min'] <= config['area_ratio_max']
            or config['height_tolerance_m'] < 0 or config['neighbor_overlap_area_m2'] < 0
            or not 0 < config['neighbor_coverage_veto'] <= 1):
        raise ValueError('Invalid association threshold')
    geometries = [shape(f['geometry']) for f in features]
    if any(g.is_empty or not g.is_valid or not np.isfinite(g.bounds).all()
           or g.geom_type not in ('Polygon', 'MultiPolygon') for g in geometries):
        raise ValueError('Invalid current building geometry')
    surfaces = []
    for f, current in zip(features, geometries):
        p = f['properties']; roofs = [
            (shape(layer['geometry']).intersection(current), float(layer['top_z_m']))
            for layer in p.get('roof_layers', [])]
        body = current.difference(unary_union([r[0] for r in roofs])) if roofs else current
        layers = [(body, float(p.get('body_top_z_m', p['top_z_m']))), *roofs]
        if any(not np.isfinite(z) or not g.is_valid for g, z in layers):
            raise ValueError('Invalid current roof layer')
        surfaces.append(layers)
    tree = STRtree(geometries)
    by_id = {o['properties']['observation_id']: o for o in observations}
    if len(by_id) != len(observations):
        raise ValueError('Duplicate observation ID')
    canonical = {}
    canonical_patches = []
    for i, feature in enumerate(features):
        oid = feature['properties']['canonical_observation_id']
        if oid not in by_id or oid in canonical:
            raise ValueError('Missing or multiply owned canonical observation')
        current = feature['properties']; op = by_id[oid]['properties']
        for a, b in (('canonical_patch_id', 'patch_id'),
                     ('canonical_view_id', 'view_id'),
                     ('canonical_instance_id', 'rgb_instance_id')):
            if current[a] != op[b]: raise ValueError('Canonical image identity mismatch')
        canonical[oid] = i
        canonical_patches.append(by_id[oid]['properties']['patch_id'])
    assignments, diagnostics = {}, []
    image_keys = set()
    for observation in observations:
        p = observation['properties']; oid = p['observation_id']
        observation_geometry = shape(observation['geometry'])
        if (observation_geometry.is_empty or not observation_geometry.is_valid
                or observation_geometry.geom_type not in ('Polygon', 'MultiPolygon')
                or not np.isfinite(observation_geometry.bounds).all()):
            raise ValueError('Invalid observation geometry')
        if p['sample_id'] != p['patch_id']+'__'+p['view_id']:
            raise ValueError('Observation image does not match patch and view')
        if (not isinstance(p['rgb_instance_id'], int) or p['rgb_instance_id'] <= 0
                or not np.isfinite([p['top_z_m'], p['preliminary_quality']]).all()):
            raise ValueError('Invalid observation identity or quality')
        key = (p['sample_id'], int(p['rgb_instance_id']))
        if key in image_keys:
            raise ValueError('Duplicate source image instance')
        image_keys.add(key)
        if oid in canonical:
            owner, reason = canonical[oid], 'canonical_exclusive'
            evidence = []
        else:
            owner, reason, evidence = -1, 'unassociated', []
            eligible = (p['stable_layer_count'] >= 1
                        and p['preliminary_quality'] >= config['quality_min']
                        and p['boundary_status'] != 'boundary_clipped_exclude')
            if eligible:
                g = observation_geometry
                overlaps = []
                for index in tree.query(g, predicate='intersects'):
                    index = int(index); current = geometries[index]
                    compatible_layers = [(abs(float(p['top_z_m'])-z), g.intersection(surface).area/g.area)
                                         for surface, z in surfaces[index] if not surface.is_empty]
                    compatible_layers = [(dz, coverage) for dz, coverage in compatible_layers
                                         if dz <= config['height_tolerance_m']
                                         and coverage >= config['containment_min']]
                    dz = min((row[0] for row in compatible_layers), default=float('inf'))
                    inter = current.intersection(g).area
                    iou = inter/(g.area+current.area-inter)
                    containment = inter/g.area
                    overlaps.append((index, inter, inter/current.area))
                    ratio = g.area/current.area
                    if (iou >= config['iou_min'] and containment >= config['containment_min']
                            and config['area_ratio_min'] <= ratio <= config['area_ratio_max']
                            and dz <= config['height_tolerance_m']):
                        evidence.append({'owner_index': index, 'iou': iou,
                                         'containment': containment, 'height_delta_m': dz})
                evidence.sort(key=lambda x: (-x['iou'], x['owner_index']))
                if evidence:
                    if len(evidence) > 1:
                        reason = 'ambiguous_current_geometry'
                    elif any(j != evidence[0]['owner_index']
                             and area >= config['neighbor_overlap_area_m2']
                             and covered >= config['neighbor_coverage_veto']
                             for j, area, covered in overlaps):
                        reason = 'spans_another_selected_building'
                    else:
                        owner = evidence[0]['owner_index']; reason = 'current_geometry_and_height'
            else:
                reason = 'unreliable_observation'
        assignments.setdefault(key[0], {})[key[1]] = owner
        diagnostics.append({'observation_id': oid, 'sample_id': key[0],
                            'instance_id': key[1], 'owner_index': owner,
                            'reason': reason, 'geometry_candidates': evidence})
    return {'assignments': assignments, 'canonical_patch_by_owner': canonical_patches,
            'owner_ids': [f['properties']['predicted_global_building_id'] for f in features],
            'diagnostics': diagnostics,
            'counts': dict(Counter(d['reason'] for d in diagnostics))}


def classify_hits(states, owner_by_instance, expected_owner, same_source_patch,
                  canonical_instance_ids=None):
    """Foreground without a unique association remains protected, never background."""
    ids = np.asarray(states['sampled_instance_ids'])
    expected = np.asarray(expected_owner, dtype=int)
    same_source = np.asarray(same_source_patch, dtype=bool)
    if ids.shape != expected.shape or ids.shape != same_source.shape:
        raise ValueError('Hit classifications require aligned point arrays')
    lookup = np.full(max(int(ids.max(initial=0)), max(owner_by_instance, default=0))+1,
                     -1, dtype=int)
    for instance, owner in owner_by_instance.items():
        if instance <= 0:
            raise ValueError('Background cannot have an owner')
        lookup[instance] = owner
    foreground = states['visible'] & (ids > 0)
    same_building = foreground & (lookup[ids] == expected)
    source = same_building & same_source
    peer = same_building & ~same_source
    unknown = foreground & ~same_building
    selected_other = unknown & (lookup[ids] >= 0)
    canonical = np.zeros(len(ids), dtype=bool)
    if canonical_instance_ids is not None:
        canonical_ids = np.asarray(canonical_instance_ids, dtype=int)
        if canonical_ids.shape != ids.shape: raise ValueError('Canonical IDs do not align')
        canonical = source & (ids == canonical_ids)
    return {'source_silhouette': source, 'peer_silhouette': peer,
            'source_surface': source & states['depth_agrees'],
            'peer_surface': peer & states['depth_agrees'],
            'neighbor_or_unknown': unknown,
            'canonical_source': canonical,
            'same_patch_related': source & ~canonical,
            'other_selected_instance': selected_other,
            'ambiguous_or_unmapped': unknown & ~selected_other,
            'all_foreground': foreground}
