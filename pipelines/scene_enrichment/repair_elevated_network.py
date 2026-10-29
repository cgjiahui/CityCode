#!/usr/bin/env python3
"""Resolve observed road-layer continuity and assembly contact on a separate copy.

Consumes predicted profiles/layers and the released scene's mesh-height cache.
No road annotations, reference bridge placements or reference map are read.
The output is assembly design inference; it does not replace observed evidence.
"""
from __future__ import annotations

import argparse
import heapq
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial import cKDTree

from extract_elevated import Config, digest, placement_proposal, read_json, write_json


def stations(xyz):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(xyz)[:, :2], axis=0), axis=1))]


def constrained_height(target, s, fixed, maximum_grade=.1, maximum_change=None):
    """Least-change smooth heights with exact joints and hard grade bounds."""
    target, s = np.asarray(target, float), np.asarray(s, float)
    if len(target) < 2 or np.any(np.diff(s) <= 0):
        raise ValueError('Height fit needs distinct ordered stations')
    difference = np.diff(np.eye(len(s)), axis=0)
    bound = maximum_grade*np.diff(s)
    lower = np.full(len(s), -np.inf) if maximum_change is None else target-maximum_change
    upper = np.full(len(s), np.inf) if maximum_change is None else target+maximum_change
    for index, value in fixed.items():
        lower[index] = upper[index] = value
    second = np.diff(np.eye(len(s)), n=2, axis=0)
    hessian = np.eye(len(s))+2.*second.T@second
    def fun(z):
        residual = z-target
        return .5*float(residual@residual)+float(np.sum((second@z)**2))
    def jac(z):
        return hessian@z-target
    initial = np.clip(target, lower, upper)
    result = minimize(fun, initial, jac=jac, method='SLSQP',
                      bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(difference, -bound, bound),
                      options={'ftol': 1e-10, 'maxiter': 150})
    if not result.success or np.max(np.abs(np.diff(result.x)/np.diff(s))) > maximum_grade+1e-7:
        raise ValueError('No feasible height/grade fit: '+result.message)
    return result.x


def key(xy):
    return tuple(np.round(xy[:2], 5))


def endpoints(segments):
    groups = defaultdict(list)
    for segment in segments:
        for end, index in [('start', 0), ('end', -1)]:
            node = segment['profile'][index]
            groups[key(node['position_m'])].append((segment, end, node))
    return groups


def solve_joints(segments, maximum_grade=.1):
    """Only identical predicted topology nodes connect; XY crossings do not."""
    fixed = defaultdict(dict)
    rejected = []
    for xy, group in endpoints(segments).items():
        if len(group) < 2:
            continue
        values = [n['position_m'][2] for _, _, n in group]
        if np.ptp(values) > 2.:
            rejected.append({'xy_m': list(xy), 'reason': 'distinct_height_layers', 'heights_m': values})
            continue
        z = float(np.median(values))
        for segment, end, node in group:
            fixed[segment['segment_id']][0 if end == 'start' else len(segment['profile'])-1] = z
    # A failed member invalidates its joint, rather than leaving a falsely
    # labelled solved node. Compute all fits before mutating any profile.
    while True:
        fitted, failures = {}, []
        for segment in segments:
            identity = segment['segment_id']
            if not fixed[identity]:
                continue
            profile = segment['profile']
            try:
                fitted[identity] = constrained_height([p['position_m'][2] for p in profile],
                                                       [p['s_m'] for p in profile], fixed[identity],
                                                       maximum_grade, 2.)
            except ValueError:
                failures.append(identity)
        if not failures:
            break
        failed_xy = {key(s['profile'][i]['position_m']) for s in segments
                     if s['segment_id'] in failures for i in fixed[s['segment_id']]}
        for segment in segments:
            identity = segment['segment_id']
            for index in list(fixed[identity]):
                if key(segment['profile'][index]['position_m']) in failed_xy:
                    del fixed[identity][index]
        rejected.extend({'xy_m': list(xy), 'reason': 'joint_grade_fit_infeasible'} for xy in failed_xy)
    for segment in segments:
        identity = segment['segment_id']
        if identity not in fitted:
            continue
        for node, z in zip(segment['profile'], fitted[identity]):
            node.setdefault('pre_network_z_m', node['position_m'][2])
            node['position_m'][2] = float(z)
            node['network_height_adjustment_m'] = float(z-node['pre_network_z_m'])
    return rejected


class LayerEvidence:
    def __init__(self, path, cell_m):
        self.data = dict(np.load(path))
        self.origin = self.data['origin_xy_m']
        self.cell = cell_m
        self.width = self.data['height_m'].shape[1]
        unique, start, count = np.unique(self.data['layer_cell_index'], return_index=True, return_counts=True)
        self.lookup = {int(k): (int(a), int(n)) for k, a, n in zip(unique, start, count)}
        observed = np.isfinite(self.data['height_m'])
        iy, ix = np.nonzero(observed)
        self.observed_xy = np.column_stack([ix, iy])
        self.tree = cKDTree(self.origin+(self.observed_xy+.5)*cell_m)

    def candidates(self, xy, maximum_distance=1.6):
        distance, index = self.tree.query(xy)
        if distance > maximum_distance:
            return []
        ix, iy = self.observed_xy[index]
        a, n = self.lookup[int(iy*self.width+ix)]
        return [{'height_m': float(self.data['layer_height_m'][j]),
                 'views': int(self.data['layer_view_count'][j]),
                 'layer_index': int(j), 'distance_m': float(distance)} for j in range(a, a+n)]

    def selection(self, xy, expected, tolerance=3.):
        choices = [c for c in self.candidates(xy) if abs(c['height_m']-expected) <= tolerance]
        if not choices:
            return None
        return min(choices, key=lambda c: abs(c['height_m']-expected)+.5/max(1, c['views']))

    def supported_width(self, xy):
        if not hasattr(self, '_mask_radius'):
            self._mask_radius = ndimage.distance_transform_edt(self.data['retained_mask'])*self.cell
        coordinates = ((np.asarray(xy)-self.origin)/self.cell-.5).T[::-1]
        return 2*ndimage.map_coordinates(self._mask_radius, coordinates, order=1, mode='constant', cval=0.)


def build_connector(source, first, last, start_z, end_z, evidence, identity, maximum_grade=.1):
    source_nodes = source['profile'][first:last+1]
    xyz = np.asarray([n['position_m'] for n in source_nodes], float)
    s = stations(xyz)
    shortcut = False
    source_indices = np.arange(first, last+1)
    # A skeleton can walk around a tiny local hole inside a broad junction.
    # Only shortcut when the entire straight ribbon has measured class-4
    # width; this never connects disjoint masks or mere XY-close overpasses.
    chord = np.linalg.norm(xyz[-1, :2]-xyz[0, :2])
    if chord > 2 and s[-1] > 1.5*chord and hasattr(evidence, 'supported_width'):
        points = np.linspace(xyz[0], xyz[-1], max(3, int(np.ceil(chord/3.))+1))
        widths = evidence.supported_width(points[:, :2])
        if widths.min() >= 12.:
            source_indices = np.rint(np.linspace(first, last, len(points))).astype(int)
            source_nodes = [source['profile'][i] for i in source_indices]
            xyz, s, shortcut = points, stations(points), True
    if len(s) < 2 or s[-1] > 90. or abs(end_z-start_z) > maximum_grade*s[-1]+1e-6:
        raise ValueError('connector_length_or_grade')
    expected = np.interp(s, [0., s[-1]], [start_z, end_z])
    selected = [evidence.selection(p[:2], z) for p, z in zip(xyz, expected)]
    anchored = np.asarray([c is not None and c['views'] >= 2 for c in selected])
    if anchored.sum() < 2 or anchored.mean() < .45:
        raise ValueError('insufficient_continuous_layer_evidence')
    anchor_s = s[anchored]
    if len(anchor_s) > 1 and np.max(np.diff(anchor_s)) > 24.:
        raise ValueError('unobserved_layer_gap_exceeds_24m')
    observed = np.asarray([c['height_m'] if c is not None else z for c, z in zip(selected, expected)])
    z = constrained_height(observed, s, {0: start_z, len(s)-1: end_z}, maximum_grade)
    if max(abs(z[i]-c['height_m']) for i, c in enumerate(selected) if c and c['views'] >= 2) > 3.:
        raise ValueError('fit_exceeds_predicted_layer_tolerance')
    nodes = []
    for i, (source_node, value, layer) in enumerate(zip(source_nodes, z, selected)):
        node = deepcopy(source_node)
        node.update({'s_m': float(s[i]), 'source_profile_index': int(source_indices[i]),
                     'source_s_m': source_node['s_m'], 'width_m': min(32., source_node['width_m']),
                     'uncertainty_reasons': [], 'evidence_state': 'network_layer_continuity_inference',
                     'original_uncertainty_reasons': source_node['uncertainty_reasons'],
                     'selected_observed_layer': layer, 'height_anchor': bool(anchored[i]),
                     'height_source': 'predicted_layer_continuity_between_connected_spans'})
        node['position_m'][2] = float(value)
        if shortcut:
            node['position_m'][:2] = xyz[i, :2].tolist()
            node['width_m'] = float(min(32., widths[i]))
            node['predicted_mask_supported_shortcut'] = True
        nodes.append(node)
    return {'segment_id': identity, 'source_segment_id': source['segment_id'],
            'profile': nodes, 'length_m': float(s[-1]), 'anchor_count': int(anchored.sum()),
            'inference': 'network_layer_continuity_inference', 'connection_only': True,
            'predicted_mask_supported_shortcut': shortcut}


def alternate_path_length(segments, start, end):
    """Existing accepted graph path, without connecting projected crossings."""
    levels = {}
    for xy, group in endpoints(segments).items():
        groups = []
        for z in sorted(n['position_m'][2] for _, _, n in group):
            if not groups or z-groups[-1][0] > 2.:
                groups.append([z])
            else:
                groups[-1].append(z)
        levels[xy] = [float(np.mean(g)) for g in groups]
    def endpoint_key(point):
        xy = key(point)
        if xy not in levels:
            return (*xy, -1)
        layer = int(np.argmin(np.abs(np.asarray(levels[xy])-point[2])))
        return (*xy, layer) if abs(levels[xy][layer]-point[2]) <= 2. else (*xy, -1)
    adjacency = defaultdict(list)
    for segment in segments:
        a, b = [endpoint_key(segment['profile'][i]['position_m']) for i in (0, -1)]
        adjacency[a].append((b, segment['length_m']))
        adjacency[b].append((a, segment['length_m']))
    queue, distances = [(0., endpoint_key(start))], {endpoint_key(start): 0.}
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        if node == endpoint_key(end):
            return distance
        for neighbor, length in adjacency[node]:
            following = distance+length
            if following < distances.get(neighbor, float('inf')):
                distances[neighbor] = following
                heapq.heappush(queue, (following, neighbor))
    return float('inf')


def repair_gaps(structure, segments, evidence):
    """Restore only omitted intervals bounded by already accepted road nodes."""
    by_source = defaultdict(list)
    for segment in segments:
        by_source[segment['source_segment_id']].append(segment)
    accepted, rejected = [], []
    hubs = endpoints(segments)
    for source in structure['segments']:
        raw = source['profile']
        spans = by_source[source['segment_id']]
        covered = np.zeros(len(raw)-1, bool)
        anchors = {}
        for segment in spans:
            first, last = segment['profile'][0]['source_profile_index'], segment['profile'][-1]['source_profile_index']
            covered[first:last] = True
            anchors[first] = segment['profile'][0]['position_m'][2]
            anchors[last] = segment['profile'][-1]['position_m'][2]
        for i in [0, len(raw)-1]:
            group = hubs.get(key(raw[i]['position_m']), [])
            if group:
                heights = [n['position_m'][2] for _, _, n in group]
                if np.ptp(heights) <= 2.:
                    anchors[i] = float(np.median(heights))
        if len(anchors) < 2:
            continue
        indices = sorted(anchors)
        number = 0
        for first, last in zip(indices[:-1], indices[1:]):
            if np.any(covered[first:last]):
                continue
            number += 1
            identity = source['segment_id']+f'_CONNECTOR_{number:02d}'
            try:
                # Independent fits can leave a short gap marginally steeper
                # than the common grade. Resolve that small endpoint mismatch
                # across the adjacent spans; never flatten a different deck.
                za, zb = anchors[first], anchors[last]
                length = raw[last]['s_m']-raw[first]['s_m']
                excess = max(0., abs(zb-za)-.1*length)
                corrections = []
                if 0 < excess <= 1.:
                    shift = np.sign(zb-za)*(excess/2+1e-5)
                    za, zb = za+shift, zb-shift
                    for index, new_z in [(first, za), (last, zb)]:
                        for segment, end, _ in hubs[key(raw[index]['position_m'])]:
                            profile = segment['profile']
                            n = 0 if end == 'start' else len(profile)-1
                            fitted = constrained_height([p['position_m'][2] for p in profile],
                                                        [p['s_m'] for p in profile], {n: new_z}, .1, 1.)
                            corrections.append((segment, fitted))
                result = build_connector(source, first, last, za, zb, evidence, identity)
                if min(n['width_m'] for n in result['profile']) < 6.:
                    alternative = alternate_path_length(segments+accepted, result['profile'][0]['position_m'],
                                                        result['profile'][-1]['position_m'])
                    if alternative <= result['length_m']*3:
                        raise ValueError('narrow_redundant_branch_with_accepted_alternative_path')
                for segment, fitted in corrections:
                    for node, z in zip(segment['profile'], fitted):
                        node['position_m'][2] = float(z)
                accepted.append(result)
            except ValueError as error:
                rejected.append({'segment_id': identity, 'source_segment_id': source['segment_id'],
                                 'source_profile_range': [first, last], 'reason': str(error),
                                 'length_m': float(raw[last]['s_m']-raw[first]['s_m']),
                                 'endpoint_z_m': [anchors[first], anchors[last]]})
    return accepted, rejected


class MeshGround:
    """Direct published mesh raster sampling; no land/reference fallback."""
    def __init__(self, directory):
        self.directory = Path(directory)
        meta = read_json(self.directory/'BACKGROUND_SAMPLING.json')
        self.origin = np.asarray(meta['origin_xy'])
        self.cell = meta['cell_m']
        self.maps = {k: np.load(self.directory/(k+'.npy'), mmap_mode='r')
                     for k in ('continuous_top', 'detail_near_ground_top')}
        self.shape = self.maps['continuous_top'].shape

    def sample(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        ij = np.floor((xy-self.origin)/self.cell).astype(int)
        valid = np.all(ij >= 0, axis=1)&(ij[:, 0] < self.shape[1])&(ij[:, 1] < self.shape[0])
        z = np.full(len(xy), np.nan)
        if valid.any():
            ix, iy = ij[valid].T
            z[valid] = np.maximum(self.maps['continuous_top'][iy, ix], self.maps['detail_near_ground_top'][iy, ix])
        z[~np.isfinite(z)] = np.nan
        return z


class ActualBuildingFootprints:
    """Lazy projected *prepared mesh* footprint, avoiding rotated-bounds inflation.

    Roof/ledge projection is conservative with respect to height, but tests the
    vertices actually shipped to the scene rather than a source accessor box.
    """
    def __init__(self, manifest):
        from shapely.geometry import box
        self.path = Path(manifest)
        document = read_json(self.path)
        self.records = []
        for record in document['meshes']:
            if not record.get('id', '').startswith(('PGB_', 'MASSING_')):
                continue
            bounds = record.get('prepared', {}).get('world_bounds_m')
            if bounds is None:
                continue
            self.records.append((record, box(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])))
        self.footprints = {}
        self.intersections = {}
        self.geometry_inputs = {}

    def projected(self, record):
        import shapely
        from shapely.ops import unary_union
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'ue_citycode'))
        from interchange import read_glb, accessor
        identity = record['id']
        if identity in self.footprints:
            return self.footprints[identity]
        path = self.path.parent/record['import_file']
        document, binary, _ = read_glb(path)
        triangles = []
        for mesh in document['meshes']:
            for primitive in mesh['primitives']:
                positions, _ = accessor(document, binary, primitive['attributes']['POSITION'])
                if 'indices' in primitive:
                    indices, _ = accessor(document, binary, primitive['indices'])
                    positions = positions[indices.ravel()]
                # These prepared GLBs contain UE-local XYZ metres; import
                # basis compensation belongs to UE, so no axis swap here.
                triangle = positions.reshape(-1, 3, 3)+np.asarray(record['anchor_m'])
                xy = triangle[:, :, :2]
                a, b = xy[:, 1]-xy[:, 0], xy[:, 2]-xy[:, 0]
                area = abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/2
                triangles.append(xy[area > 1e-6])
        footprint = unary_union(shapely.polygons(np.concatenate(triangles))).buffer(0)
        self.footprints[identity] = footprint
        self.geometry_inputs[identity] = {'file': str(path), 'sha256': digest(path)}
        return footprint

    def intersects(self, corridor):
        found = []
        for record, bounds in self.records:
            if bounds.intersection(corridor).area <= 2.:
                continue
            area = self.projected(record).intersection(corridor).area
            if area > 2.:
                found.append({'id': record['id'], 'projected_intersection_m2': float(area)})
        self.last_intersections = found
        return bool(found)


def generate_approaches(structure, segments, ground, buildings=None, maximum_lateral_shift_m=0.):
    """Fit finite-length, scene-supported landings only at predicted leaf ends.

    A terminal lacking enough ground support or a collision-free corridor is
    retained as unresolved. Elevated junctions are never forced to ground.
    """
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    sources = {s['segment_id']: s for s in structure['segments']}
    leaves = {key(n['position_xy_m']) for n in structure['junctions_and_endpoints'] if n['kind'] == 'endpoint'}
    additions, decisions = [], []
    for xy, group in endpoints(segments).items():
        if len(group) != 1:
            continue
        segment, end, node = group[0]
        if segment.get('connection_only'):
            continue
        source = sources[segment['source_segment_id']]
        raw = source['profile']
        towards = 0 if end == 'start' else len(raw)-1
        if key(raw[towards]['position_m']) not in leaves:
            continue
        index = node['source_profile_index']
        # Do not cross a different accepted span further along this source.
        if any(s['source_segment_id'] == source['segment_id'] and s is not segment and
               ((end == 'start' and s['profile'][-1]['source_profile_index'] < index) or
                (end == 'end' and s['profile'][0]['source_profile_index'] > index)) for s in segments):
            continue
        path = list(range(index, towards-1, -1)) if end == 'start' else list(range(index, towards+1))
        xyz = np.asarray([raw[i]['position_m'] for i in path], float)
        xyz[0] = node['position_m']
        # The published geometric surface is used for assembly contact only.
        last = xyz[-1, :2]
        neighbor = np.asarray(raw[1 if towards == 0 else -2]['position_m'][:2])
        outward = last-neighbor
        outward /= np.linalg.norm(outward)
        initial_length = float(stations(xyz)[-1])
        width = float(min(node['width_m'], 14.))
        decision = {'segment_id': segment['segment_id'], 'end': end,
                    'source_terminal_xy_m': last.tolist(), 'status': 'withheld'}
        best = None
        shifts = [0.] if maximum_lateral_shift_m <= 0 else [0., -.5*maximum_lateral_shift_m,
                                                          .5*maximum_lateral_shift_m,
                                                          -maximum_lateral_shift_m, maximum_lateral_shift_m]
        # Prefer the unshifted predicted course even when a shifted course is
        # a few metres shorter. Lateral design inference is a last resort.
        for extension, lateral in [(e, d) for d in shifts for e in np.arange(0., 82., 2.)]:
            extra = np.arange(2., extension+.01, 2.)
            candidate = np.vstack([xyz, np.column_stack([last+extra[:, None]*outward,
                                                        np.full(len(extra), xyz[-1, 2])])]) if len(extra) else xyz.copy()
            s = stations(candidate)
            if len(s) < 3 or s[-1] < 12.:
                continue
            if lateral:
                tangent = np.gradient(candidate[:, :2], s, axis=0)
                normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                progress = np.minimum(1., s/max(20., .6*s[-1]))
                candidate[:, :2] += normals*(lateral*(3*progress**2-2*progress**3))[:, None]
                s = stations(candidate)
            center_z = ground.sample(candidate[:, :2])
            side = np.array([-outward[1], outward[0]])
            land_xy = np.array([candidate[-1, :2]+side*f*width for f in (-.5, -.25, 0., .25, .5)])
            land_z = ground.sample(land_xy)
            if not np.isfinite(land_z).all() or np.ptp(land_z) > .6:
                continue
            # Flat end sits just below all five actual surface samples. Deck
            # thickness extends downward into the ground, not into open air.
            landing = float(land_z.min()-.025)
            dz = candidate[0, 2]-landing
            if dz < -.2 or dz > .075*s[-1]:
                continue
            z = candidate[0, 2]+(landing-candidate[0, 2])*(3*(s/s[-1])**2-2*(s/s[-1])**3)
            try:
                # Last two station tops are buried to provide a horizontal
                # run-out instead of a vertical 1.6m native end cap.
                z = constrained_height(z, s, {0: candidate[0, 2], len(s)-1: landing,
                                               len(s)-2: landing}, .1)
            except ValueError:
                continue
            if np.isfinite(center_z).mean() < .95:
                continue
            widths = node['width_m']+(width-node['width_m'])*np.minimum(1., s/max(10., .5*s[-1]))
            strips = []
            for i in range(len(candidate)-1):
                delta = candidate[i+1, :2]-candidate[i, :2]
                normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
                a, b = candidate[i, :2], candidate[i+1, :2]
                strips.append(Polygon([a+normal*widths[i]/2, b+normal*widths[i+1]/2,
                                       b-normal*widths[i+1]/2, a-normal*widths[i]/2]))
            corridor = unary_union(strips)
            collision = (buildings.intersects(corridor) if hasattr(buildings, 'intersects') else
                         any(corridor.intersection(poly).area > 2. for _, poly in buildings or []))
            if collision:
                decision['reason'] = 'approach_would_intersect_existing_building'
                if hasattr(buildings, 'last_intersections'):
                    decision['blocking_buildings'] = buildings.last_intersections
                continue
            best = candidate, s, z, extension, lateral, land_xy, land_z
            break
        if best is None:
            decision.setdefault('reason', 'no_supported_grade_bounded_landing_within_80m_extension')
            decisions.append(decision)
            continue
        xyz, s, z, extension, lateral, land_xy, land_z = best
        decision.pop('reason', None)
        decision.pop('blocking_buildings', None)
        identity = segment['segment_id']+'_APPROACH_'+end.upper()
        profile = []
        for i, (point, value) in enumerate(zip(xyz, z)):
            item = deepcopy(node)
            item.update({'s_m': float(s[i]), 'position_m': [*point[:2].tolist(), float(value)],
                         'width_m': float(node['width_m']+(width-node['width_m'])*min(1., s[i]/max(10., .5*s[-1]))),
                         'source_profile_index': int(path[i]) if i < len(path) else int(towards),
                         'height_source': 'assembly_approach_to_published_mesh_ground',
                         'height_anchor': False, 'uncertainty_reasons': [],
                         'evidence_state': 'assembly_ground_transition_inference',
                         'ground_transition': True})
            profile.append(item)
        additions.append({'segment_id': identity, 'source_segment_id': source['segment_id'],
                          'length_m': float(s[-1]), 'profile': profile, 'anchor_count': 0,
                          'inference': 'assembly_ground_transition_inference', 'ground_transition': True,
                          'landing_contact_xy_m': land_xy.tolist(), 'landing_ground_z_m': land_z.tolist(),
                          'terminal_extension_m': float(extension), 'lateral_avoidance_m': float(lateral),
                          'native_deck_end_buried': True})
        decision.update({'status': 'approach_candidate', 'approach_segment_id': identity,
                         'length_m': float(s[-1]), 'source_tail_length_m': initial_length,
                         'terminal_extension_m': float(extension), 'landing_top_z_m': float(z[-1]),
                         'lateral_avoidance_m': float(lateral),
                         'ground_contact_range_m': [float(land_z.min()), float(land_z.max())]})
        decisions.append(decision)
    return additions, decisions


def inplace_landings(segments, decisions, ground, buildings):
    """Try contact inside an existing leaf corridor after external routes fail.

    Observed heights remain in the source package. This explicitly redesigns
    the assembly ramp and may lower it more than the ordinary join fit budget.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    by_id = {s['segment_id']: s for s in segments}
    for decision in decisions:
        if decision['status'] != 'withheld':
            continue
        span = by_id[decision['segment_id']]
        profile = span['profile']
        xyz = np.asarray([n['position_m'] for n in profile])
        s = np.asarray([n['s_m'] for n in profile])
        terminal = 0 if decision['end'] == 'start' else len(s)-1
        inside = 1 if terminal == 0 else len(s)-2
        far = len(s)-1 if terminal == 0 else 0
        direction = xyz[inside, :2]-xyz[terminal, :2]
        side = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
        sample_xy = xyz[terminal, :2]+np.linspace(-.5, .5, 5)[:, None]*side*profile[terminal]['width_m']
        land_z = ground.sample(sample_xy)
        if not np.isfinite(land_z).all() or np.ptp(land_z) > .6:
            decision['inplace_landing_rejected'] = 'terminal_ground_missing_or_excessive_crossfall'
            continue
        landing = float(land_z.min()-.025)
        minimum_grade = abs(xyz[far, 2]-landing)/s[-1]
        decision['inplace_minimum_required_grade'] = float(minimum_grade)
        if minimum_grade > .1:
            decision['inplace_landing_rejected'] = 'insufficient_existing_corridor_length_for_grade'
            continue
        target = xyz[:, 2].copy()
        target[terminal] = landing
        try:
            z = constrained_height(target, s, {terminal: landing, far: xyz[far, 2]}, .1)
        except ValueError:
            decision['inplace_landing_rejected'] = 'infeasible_height_profile'
            continue
        terrain = ground.sample(xyz[:, :2])
        if not np.isfinite(terrain).all() or np.any(z < terrain-.15):
            decision['inplace_landing_rejected'] = 'profile_would_pass_below_published_ground'
            continue
        strips = []
        for i in range(len(xyz)-1):
            a, b = xyz[i, :2], xyz[i+1, :2]
            delta = b-a
            normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
            wa, wb = profile[i]['width_m']/2, profile[i+1]['width_m']/2
            strips.append(Polygon([a+normal*wa, b+normal*wb, b-normal*wb, a-normal*wa]))
        corridor = unary_union(strips)
        collision = (buildings.intersects(corridor) if hasattr(buildings, 'intersects') else
                     any(corridor.intersection(poly).area > 2. for _, poly in buildings or []))
        if collision:
            decision['inplace_landing_rejected'] = 'existing_corridor_intersects_actual_building'
            continue
        for node, old, new in zip(profile, xyz[:, 2], z):
            node['pre_landing_z_m'] = float(old)
            node['position_m'][2] = float(new)
            node['height_source'] = 'assembly_inplace_terminal_to_published_ground'
        span.update({'inplace_landing_end': decision['end'], 'landing_contact_xy_m': sample_xy.tolist(),
                     'landing_ground_z_m': land_z.tolist(), 'native_deck_end_buried': True,
                     'assembly_height_reason': 'inplace_contact_without_extra_corridor_or_building_intrusion'})
        decision.pop('reason', None)
        decision.pop('blocking_buildings', None)
        decision.update({'status': 'inplace_landing_candidate', 'landing_top_z_m': landing,
                         'maximum_height_redesign_m': float(np.max(abs(z-xyz[:, 2]))),
                         'minimum_ground_clearance_m': float(np.min(z-terrain)),
                         'horizontal_runout': False,
                         'note': 'No vertical end step; corridor length limits approach grade easing.'})


def validate_candidate(structure, segments, evidence):
    """Independent numerical/lineage checks, not a final visual-acceptance claim."""
    original = {s['segment_id']: s for s in structure['segments']}
    largest_grade = 0.
    for segment in segments:
        profile = segment['profile']
        xyz = np.asarray([n['position_m'] for n in profile])
        s = np.asarray([n['s_m'] for n in profile])
        if not np.isfinite(xyz).all() or np.any(np.diff(s) <= 0):
            raise ValueError('Invalid network profile: '+segment['segment_id'])
        grade = float(np.max(np.abs(np.diff(xyz[:, 2])/np.diff(s))))
        if grade > .100001:
            raise ValueError('Network profile exceeds maximum grade')
        largest_grade = max(largest_grade, grade)
        if segment.get('ground_transition'):
            if xyz[-1, 2] != xyz[-2, 2] or np.any(xyz[-1, 2] > segment['landing_ground_z_m']):
                raise ValueError('Ground landing is not flat and buried')
            continue
        if segment.get('predicted_mask_supported_shortcut'):
            available = evidence.supported_width(xyz[:, :2])
            if np.any(available+1e-6 < np.array([n['width_m'] for n in profile])):
                raise ValueError('Shortcut leaves predicted semantic support')
        else:
            raw = original[segment['source_segment_id']]['profile']
            expected = np.asarray([raw[n['source_profile_index']]['position_m'][:2] for n in profile])
            if not np.allclose(xyz[:, :2], expected, atol=1e-7):
                raise ValueError('Network moved XY without supported shortcut declaration')
    joints = connection_contract(segments)
    return {'numerical_checks': 'passed', 'maximum_discrete_grade': largest_grade,
            'solved_shared_nodes': len(joints), 'checked_profiles': len(segments),
            'maximum_shared_node_height_error_m': max((j['height_spread_m'] for j in joints), default=0.),
            'scene_qa_complete': False,
            'remaining_checks': ['native_surface_union', 'actual_building_triangle_clearance',
                                 'final_ground_contact', 'roadside_props_and_vehicles', 'UE_visual_review']}


def filter_isolated_fragments(segments, grid, maximum_length_m=150.):
    """Withhold small interior airborne components, never delete observations."""
    by_id = {s['segment_id']: s for s in segments}
    adjacency = {identity: set() for identity in by_id}
    for group in endpoints(segments).values():
        for a, _, na in group:
            for b, _, nb in group:
                if abs(na['position_m'][2]-nb['position_m'][2]) < 1e-5:
                    adjacency[a['segment_id']].add(b['segment_id'])
    origin = np.asarray(grid['origin_xy_m'])
    maximum = origin+np.asarray(grid['shape_yx'][::-1])*grid['cell_size_m']
    remaining, withheld, components = set(by_id), set(), []
    while remaining:
        todo, component = [sorted(remaining)[0]], set()
        while todo:
            identity = todo.pop()
            if identity in component:
                continue
            component.add(identity)
            todo.extend(adjacency[identity]-component)
        remaining -= component
        length = sum(by_id[i]['length_m'] for i in component)
        grounded = [i for i in component if by_id[i].get('ground_transition') or by_id[i].get('inplace_landing_end')]
        points = np.asarray([n['position_m'][:2] for i in component for n in by_id[i]['profile']])
        boundary_distance = float(np.min(np.minimum(points-origin, maximum-points)))
        reject = length < maximum_length_m and not grounded and boundary_distance > 25.
        if reject:
            withheld |= component
        components.append({'segment_ids': sorted(component), 'length_m': float(length),
                           'grounded_segments': sorted(grounded), 'scene_edge_distance_m': boundary_distance,
                           'assembly_enabled': not reject,
                           'reason': 'short_isolated_airborne_interior_fragment' if reject else 'retained_network_component'})
    return [s for s in segments if s['segment_id'] not in withheld], components


def design_completions(structure, segments, evidence, buildings):
    """Design, never observed reconstruction, across high-layer evidence gaps.

    Only a long interior ungrounded corridor whose omitted source-chain tails
    terminate on accepted road nodes is eligible. XY stays on the prediction.
    """
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import unary_union
    _, components = filter_isolated_fragments(segments, structure['grid'])
    sources = {s['segment_id']: s for s in structure['segments']}
    by_id = {s['segment_id']: s for s in segments}
    hubs = endpoints(segments)
    additions, decisions = [], []
    for component in components:
        if component['grounded_segments'] or component['length_m'] < 150. or len(component['segment_ids']) != 1:
            continue
        existing = by_id[component['segment_ids'][0]]
        source = sources[existing['source_segment_id']]
        raw = source['profile']
        lo, hi = existing['profile'][0]['source_profile_index'], existing['profile'][-1]['source_profile_index']
        for first, last, side in [(0, lo, 'START'), (hi, len(raw)-1, 'END')]:
            if first == last:
                continue
            external_index = first if side == 'START' else last
            group = hubs.get(key(raw[external_index]['position_m']), [])
            if not group:
                continue
            external_heights = [n['position_m'][2] for _, _, n in group]
            if np.ptp(external_heights) > 2.:
                continue
            inside_z = existing['profile'][0 if side == 'START' else -1]['position_m'][2]
            outside_z = float(np.median(external_heights))
            za, zb = (outside_z, inside_z) if side == 'START' else (inside_z, outside_z)
            identity = source['segment_id']+'_DESIGN_COMPLETION_'+side
            decision = {'segment_id': identity, 'source_profile_range': [first, last],
                        'inference': 'assembly_design_completion', 'status': 'withheld'}
            xyz = np.array([n['position_m'] for n in raw[first:last+1]], float)
            s = stations(xyz)
            width = np.minimum(32., evidence.supported_width(xyz[:, :2]))
            if s[-1] > 90. or abs(zb-za) > .1*s[-1] or width.min() < 6.:
                decision['reason'] = 'design_length_grade_or_predicted_corridor_width'
                decisions.append(decision); continue
            # Densely check the proposed center path, not just sparse anchors.
            dense_s = np.linspace(0., s[-1], max(3, int(s[-1]*2)+1))
            dense_xy = np.column_stack([np.interp(dense_s, s, xyz[:, axis]) for axis in (0, 1)])
            available = evidence.supported_width(dense_xy)
            if available.min() < 6.:
                decision['reason'] = 'inter_sample_prediction_corridor_gap'
                decisions.append(decision); continue
            # Inscribed semantic width bounds the whole ribbon conservatively.
            width[:] = min(width.min(), available.min())
            z = constrained_height(np.interp(s, [0., s[-1]], [za, zb]), s, {0: za, len(s)-1: zb}, .1)
            strips = []
            for i in range(len(xyz)-1):
                a, b = xyz[i, :2], xyz[i+1, :2]
                delta = b-a; normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
                strips.append(Polygon([a+normal*width[i]/2, b+normal*width[i+1]/2,
                                       b-normal*width[i+1]/2, a-normal*width[i]/2]))
            corridor = unary_union(strips)
            if hasattr(buildings, 'intersects') and buildings.intersects(corridor):
                decision.update(reason='actual_building_projection_intersection', blocking_buildings=buildings.last_intersections)
                decisions.append(decision); continue
            incident = {existing['segment_id']} | {span['segment_id'] for span, _, _ in group}
            for xy_key in (key(xyz[0]), key(xyz[-1])):
                incident |= {span['segment_id'] for span, _, _ in hubs.get(xy_key, [])}
            checks, unsafe = [], False
            for other in segments:
                if other['segment_id'] in incident:
                    continue
                other_xyz = np.asarray([n['position_m'] for n in other['profile']])
                other_s = stations(other_xyz)
                line = LineString(other_xyz[:, :2])
                footprint = line.buffer(max(n['width_m'] for n in other['profile'])/2, cap_style='flat')
                overlap = corridor.intersection(footprint)
                if overlap.area <= .2:
                    continue
                points = [overlap.representative_point()]
                points.extend(Point(xy) for xy in dense_xy if overlap.buffer(.1).intersects(Point(xy)))
                own_line = LineString(xyz[:, :2])
                differences = []
                for point in points:
                    own_z = np.interp(own_line.project(point), s, z)
                    other_z = np.interp(line.project(point), other_s, other_xyz[:, 2])
                    differences.append(abs(float(own_z-other_z)))
                minimum = min(differences)
                checks.append({'other_segment_id': other['segment_id'], 'overlap_area_m2': float(overlap.area),
                               'minimum_deck_top_separation_m': minimum,
                               'approximate_underside_clearance_m': minimum-1.65})
                if minimum < 5.65:
                    unsafe = True
            decision['other_road_clearance_checks'] = checks
            if unsafe:
                decision['reason'] = 'unconnected_road_crossing_clearance_below_4m'
                decisions.append(decision); continue
            nodes = []
            for index, (point, value, w) in enumerate(zip(xyz, z, width)):
                node = deepcopy(raw[first+index])
                node.update({'position_m': [*point[:2].tolist(), float(value)], 's_m': float(s[index]),
                             'source_profile_index': first+index, 'source_s_m': node['s_m'],
                             'width_m': float(w), 'uncertainty_reasons': [],
                             'original_uncertainty_reasons': raw[first+index]['uncertainty_reasons'],
                             'height_anchor': False, 'evidence_state': 'assembly_design_completion',
                             'height_source': 'accepted_endpoint_grade_design_not_observed_height_recovery'})
                nodes.append(node)
            additions.append({'segment_id': identity, 'source_segment_id': source['segment_id'],
                              'profile': nodes, 'length_m': float(s[-1]), 'anchor_count': 0,
                              'inference': 'assembly_design_completion', 'connection_only': True,
                              'design_completion': True, 'observed_height_reconstruction': False})
            decision.update(status='assembly_design_candidate', length_m=float(s[-1]),
                            width_m=float(width.min()), maximum_grade=float(np.max(abs(np.diff(z)/np.diff(s)))),
                            predicted_semantic_centerline_coverage=1., endpoint_heights_m=[za, zb])
            decisions.append(decision)
    return additions, decisions


def terminal_report(structure, segments, ground):
    original = {s['segment_id']: s for s in structure['segments']}
    result = []
    groups = endpoints(segments)
    for xy, group in groups.items():
        if len(group) != 1:
            continue
        segment, end, node = group[0]
        src = original[segment['source_segment_id']]
        xyz = np.asarray(node['position_m'])
        ground_z = ground.sample([xyz[:2]])[0]
        index = node['source_profile_index']
        result.append({'segment_id': segment['segment_id'], 'end': end, 'position_m': xyz.tolist(),
                       'source_profile_index': index, 'source_profile_count': len(src['profile']),
                       'ground_z_m': float(ground_z) if np.isfinite(ground_z) else None,
                       'vertical_gap_m': float(xyz[2]-ground_z) if np.isfinite(ground_z) else None,
                       'original_graph_terminal': index in (0, len(src['profile'])-1),
                       'status': 'grounded_inplace' if segment.get('inplace_landing_end') == end else
                                 'grounded_approach' if segment.get('ground_transition') and end == 'end' else
                                 'needs_approach_or_explicit_terminal'})
    return result


def connection_contract(segments):
    records = []
    for xy, group in endpoints(segments).items():
        if len(group) < 2:
            continue
        zs = [node['position_m'][2] for _, _, node in group]
        if np.ptp(zs) > 1e-5:
            continue
        arms = []
        for segment, end, node in group:
            a = np.asarray(node['position_m'])
            b = np.asarray(segment['profile'][1 if end == 'start' else -2]['position_m'])
            tangent = b-a
            tangent /= np.linalg.norm(tangent)
            arms.append({'segment_id': segment['segment_id'], 'end': end,
                         'tangent_outward_xyz': tangent.tolist(), 'width_m': node['width_m']})
        records.append({'junction_id': f'NETWORK_JUNCTION_{len(records)+1:03d}',
                        'position_m': [*xy, float(np.mean(zs))], 'arms': arms,
                        'height_spread_m': float(np.ptp(zs))})
    return records


def run(input_directory, ground_directory, output, building_binding=None, building_assembly=None, maximum_lateral_shift_m=0., enable_inplace_landings=False, withhold_isolated_fragments=False, enable_design_completions=False):
    executing_digest = digest(Path(__file__))
    input_directory, ground_directory, output = map(Path, (input_directory, ground_directory, output))
    if output.resolve() == input_directory.resolve():
        raise ValueError('Repair must use a separate candidate directory')
    structure = read_json(input_directory/'elevated_structure.json')
    proposal = read_json(input_directory/'regularized_placement_proposal.json')
    if structure['uses_ground_truth'] or proposal['uses_ground_truth']:
        raise ValueError('Prediction-only source required')
    config = Config(**structure['config'])
    evidence = LayerEvidence(input_directory/'prediction_evidence_grid.npz', config.grid_m)
    segments = deepcopy(proposal['continuous_segments'])
    rejected_joints = solve_joints(segments)
    additions, rejected_gaps = repair_gaps(structure, segments, evidence)
    segments.extend(additions)
    rejected_joints.extend(solve_joints(segments))
    ground = MeshGround(ground_directory)
    buildings = None
    if building_assembly:
        buildings = ActualBuildingFootprints(building_assembly)
    elif building_binding:
        from props import load_buildings
        buildings = load_buildings(building_binding)
    approaches, approach_decisions = generate_approaches(structure, segments, ground, buildings, maximum_lateral_shift_m)
    segments.extend(approaches)
    if enable_inplace_landings:
        inplace_landings(segments, approach_decisions, ground, buildings)
    all_candidates = deepcopy(segments)
    components = []
    if withhold_isolated_fragments:
        segments, components = filter_isolated_fragments(segments, structure['grid'])
    design_segments, design_decisions = [], []
    if enable_design_completions:
        design_segments, design_decisions = design_completions(structure, segments, evidence, buildings)
        segments.extend(design_segments)
        all_candidates.extend(deepcopy(design_segments))
    validation = validate_candidate(structure, segments, evidence)
    terminals = terminal_report(structure, segments, ground)
    result = placement_proposal(segments, config)
    result.update({'schema': 'citycode.elevated_network_repair.v1',
                   'source': 'predicted_profile_layers_and_published_scene_contact',
                   'uses_ground_truth': False, 'continuous_segments': segments,
                   'continuous_segment_count': len(segments),
                   'continuous_length_m': sum(s['length_m'] for s in segments),
                   'inference': 'assembly_network_postprocessing',
                   'replaces_not_appends': str(input_directory/'regularized_placement_proposal.json')})
    for row in result['placements']:
        row['inference'] = 'assembly_network_postprocessing'
    output.mkdir(parents=True, exist_ok=True)
    if withhold_isolated_fragments:
        write_json(output/'all_candidates.json', {'schema': 'citycode.elevated_assembly_candidates.v1',
                                                  'continuous_segments': all_candidates, 'component_decisions': components,
                                                  'original_prediction_modified': False})
    write_json(output/'regularized_placement_proposal.json', result)
    lines = {'type': 'FeatureCollection', 'coordinate_system': proposal['coordinate_system'],
             'features': [{'type': 'Feature', 'id': s['segment_id'],
                           'properties': {k: v for k, v in s.items() if k != 'profile'},
                           'geometry': {'type': 'LineString', 'coordinates': [n['position_m'] for n in s['profile']]}}
                          for s in segments]}
    write_json(output/'regularized_centerlines.geojson', lines)
    report = {'schema': 'citycode.elevated_network_connections.v1', 'uses_ground_truth': False,
              'input_digests': {name: digest(input_directory/name) for name in
                               ('elevated_structure.json', 'regularized_placement_proposal.json', 'prediction_evidence_grid.npz')},
              'ground_metadata_digest': digest(ground_directory/'BACKGROUND_SAMPLING.json'),
              'building_binding_digest': digest(building_binding) if building_binding else None,
              'building_assembly_digest': digest(building_assembly) if building_assembly else None,
              'building_collision_geometry': 'prepared_scene_mesh_projected_triangles' if building_assembly else 'source_bounds_proxy',
              'sampled_building_geometry': buildings.geometry_inputs if building_assembly else None,
              'producer_digest': executing_digest,
              'original_segment_count': len(proposal['continuous_segments']),
              'connector_count': sum(s.get('connection_only', False) for s in segments), 'final_segment_count': len(segments),
              'retained_original_segment_count': sum(not s.get('connection_only') and not s.get('ground_transition') for s in segments),
              'candidate_segment_count': len(all_candidates), 'component_decisions': components,
              'design_completion_count': len(design_segments), 'design_completion_decisions': design_decisions,
              'approach_count': len(approaches), 'approach_decisions': approach_decisions,
              'inplace_landing_count': sum(s.get('inplace_landing_end') is not None for s in segments),
              'validation': validation,
              'maximum_grade': max(float(np.max(np.abs(np.diff([p['position_m'][2] for p in s['profile']]))/
                                                   np.diff([p['s_m'] for p in s['profile']]))) for s in segments),
              'junctions': connection_contract(segments), 'rejected_joints': rejected_joints,
              'rejected_connectors': rejected_gaps, 'open_terminals': terminals,
              'status': 'candidate_requires_mesh_junction_and_approach_qa'}
    write_json(output/'network_connections.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--ground', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--building-binding', type=Path)
    p.add_argument('--building-assembly', type=Path)
    p.add_argument('--maximum-approach-lateral-shift', type=float, default=0.)
    p.add_argument('--inplace-landings', action='store_true')
    p.add_argument('--withhold-isolated-fragments', action='store_true')
    p.add_argument('--design-completions', action='store_true')
    args = p.parse_args()
    report = run(args.input, args.ground, args.output, args.building_binding, args.building_assembly,
                 args.maximum_approach_lateral_shift, args.inplace_landings, args.withhold_isolated_fragments,
                 args.design_completions)
    print(json.dumps({k: v for k, v in report.items() if k not in ('junctions', 'input_digests', 'open_terminals', 'rejected_connectors', 'rejected_joints')}, indent=2))


if __name__ == '__main__':
    main()
