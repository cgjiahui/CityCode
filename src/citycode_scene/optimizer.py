"""Bounded geometric proposals. An agent reviews proposals before replay."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import math

import numpy as np
from shapely import affinity
from shapely.geometry import shape, mapping

from . import pairs
from .geometry import Raster, footprint
from .replay import delta_matrix


def contact_penalty(row, allowed_area=5.):
    if not row or row['same_building_family']:
        return 0.
    # Tiny facade/step contacts are explicitly tolerated, including tall seams.
    if row['classification'] in ('minor_contact', 'XY_only_no_common_solid', 'uncertain_open_surface'):
        return 0.
    return max(0., row['known_solid_overlap_area_m2']-allowed_area)


class PlacementSearch:
    def __init__(self, cache_dir, workdir, lane_geojson=None):
        self.cache_dir = Path(cache_dir)
        pairs.configure(cache_dir)
        self.original = {r['id']: deepcopy(r) for r in pairs.records}
        self.current = deepcopy(self.original)
        self.raster = Raster(Path(workdir)/'native')
        self.cell = pairs.CELL
        self.road_model = None
        self.pedestrian_model = None
        if lane_geojson:
            from .roads import RoadAudit
            self.road_model = RoadAudit(cache_dir, lane_geojson)
            self.pedestrian_model = RoadAudit(cache_dir, lane_geojson, kind='pedestrian', clearance_m=2.2)

    def road_measure(self, row):
        if self.road_model is None:
            return None
        return self.road_model.analyze(row, pairs.columns_for(row),
            [pairs.columns_for(c) for c in row.get('components', [])] or None)

    def walking_measure(self, row):
        if self.pedestrian_model is None:
            return None
        return self.pedestrian_model.analyze(row, pairs.columns_for(row),
            [pairs.columns_for(c) for c in row.get('components', [])] or None)

    @lru_cache(maxsize=32)
    def geometry(self, relative):
        with np.load(self.cache_dir/relative) as archive:
            return archive['vertices'], archive['faces']

    def transform(self, aid, delta):
        current = self.current[aid]
        original = self.original[aid]
        total = np.asarray(delta) @ np.asarray(current.get('_world_delta', np.eye(4)))
        out = {**current, '_world_delta': total.tolist()}
        if 'components' in current:
            out['components'] = [dict(c) for c in current['components']]
        for descriptor, original_component in [(out, original)] + list(zip(out.get('components', []), original.get('components', []))):
            v, f = self.geometry(original_component['geometry'])
            world = v @ total[:3, :3].T + total[:3, 3]
            columns = self.raster.cast(world, f, self.cell)
            descriptor['_columns'] = pairs.prepare_columns(columns)
            descriptor['world_bounds'] = [world.min(0).tolist(), world.max(0).tolist()]
            descriptor['footprint_world'] = mapping(footprint(np.isfinite(columns['zmin']), columns['origin'], self.cell))
            descriptor['sampled_projection_area_m2'] = float(np.isfinite(columns['zmin']).sum()*self.cell**2)
        out['world_from_local_4x4'] = (np.asarray(delta) @ np.asarray(current['world_from_local_4x4'])).tolist()
        return out

    def evaluate(self, changed):
        updated = {**self.current, **changed}
        rows = []
        visited = set()
        for aid, a in changed.items():
            poly = shape(a['footprint_world']).buffer(self.cell)
            for bid, b in updated.items():
                key = tuple(sorted([aid, bid]))
                if aid == bid or key in visited:
                    continue
                visited.add(key)
                if a['physical_family_id'] == b['physical_family_id']:
                    continue
                if min(a['world_bounds'][1][2], b['world_bounds'][1][2])-max(a['world_bounds'][0][2], b['world_bounds'][0][2]) <= .05:
                    continue
                if not poly.intersects(shape(b['footprint_world']).buffer(self.cell)):
                    continue
                row = pairs.measure(a, b)
                if row:
                    rows.append(row)
        return rows

    def apply(self, ids, delta):
        changed = {aid: self.transform(aid, delta) for aid in ids}
        self.current.update(changed)
        return self.evaluate(changed)

    def group(self, aid):
        # Formal multi-part buildings stay one rigid group; inferred wings are
        # adjusted separately in XY unless the reviewer explicitly binds them.
        entity = self.current[aid]['entity_id']
        return [bid for bid, r in self.current.items() if r['entity_id'] == entity]

    def propose_pair(self, aid, bid, *, max_shift=2., min_scale=.95, finalist_count=16):
        baseline_members = set(self.group(aid)) | set(self.group(bid))
        baseline = self.evaluate({member:self.current[member] for member in baseline_members})
        target_baseline = pairs.measure(self.current[aid], self.current[bid])
        base_pairs = {frozenset([p['asset_a'], p['asset_b']]): p for p in baseline}
        baseline_penalty = sum(contact_penalty(p) for p in baseline)
        road_before = {}
        walking_before = {}
        candidates = []
        for mobile, fixed in ((aid, bid), (bid, aid)):
            ids = self.group(mobile)
            if self.road_model:
                for member in ids:
                    road_before[member] = self.road_measure(self.current[member])
                    walking_before[member] = self.walking_measure(self.current[member])
            boxes = np.asarray([self.current[x]['world_bounds'] for x in ids])
            pivot = (boxes[:, 0, :2].min(0)+boxes[:, 1, :2].max(0))/2
            movable = {x: shape(self.current[x]['footprint_world']) for x in ids}
            neighbors = {x: shape(r['footprint_world']) for x, r in self.current.items() if x not in ids and any(shape(r['footprint_world']).distance(g) < max_shift+1 for g in movable.values())}
            target_poly = shape(self.current[fixed]['footprint_world'])
            c = np.mean(boxes[:, :, :2], axis=(0, 1))
            fbox = np.asarray(self.current[fixed]['world_bounds'])
            away = c-fbox[:, :2].mean(0)
            angles = list(np.arange(0, 2*math.pi, math.pi/4))
            if np.linalg.norm(away) > 1e-8:
                angles += [math.atan2(away[1], away[0])]
            shifts = [(0., 0.)] + [(float(length*math.cos(angle)), float(length*math.sin(angle)))
                       for length in [.25, .5, 1., 1.5, max_shift] for angle in angles]
            scales = sorted({s for s in [1., .995, .99, .98, .97, min_scale] if s >= min_scale}, reverse=True)
            before_xy = {x: sum(poly.intersection(g).area for poly in movable.values()) for x, g in neighbors.items()}
            for scale in scales:
                for tx, ty in shifts:
                    if (scale == 1 and tx == ty == 0) or math.hypot(tx, ty) > max_shift+1e-8:
                        continue
                    projected = {x: affinity.translate(affinity.scale(g, scale, scale, origin=tuple(pivot)), tx, ty) for x, g in movable.items()}
                    target_area = sum(g.intersection(target_poly).area for g in projected.values())
                    increase = sum(max(0., sum(g.intersection(poly).area for g in projected.values())-before_xy[x]-.5)
                                   for x, poly in neighbors.items() if x != fixed)
                    cost = math.hypot(tx, ty)+20*(1-scale)
                    proxy = target_area+5*increase+.15*cost
                    candidates.append((proxy, cost, ids, pivot, tx, ty, scale))
        # Retain candidates from both mobile sides, including inexpensive small
        # shrinks which an XY-only metric can rank below a larger displacement.
        finalists = sorted(candidates, key=lambda x: (x[0], x[1]))[:finalist_count]
        finalists += sorted(candidates, key=lambda x: (x[1], x[0]))[:finalist_count//2]
        results = []
        for _, cost, ids, pivot, tx, ty, scale in finalists:
            delta = delta_matrix((tx, ty, 0), scale, pivot)
            outside_budget = False
            for member in ids:
                total = delta @ np.asarray(self.current[member].get('_world_delta', np.eye(4)))
                original_center = np.asarray(self.original[member]['world_bounds'])[:, :2].mean(0)
                displacement = total[:2, :2] @ original_center + total[:2, 3] - original_center
                if np.linalg.norm(displacement) > max_shift+1e-8 or total[0, 0] < .90:
                    outside_budget = True
            if outside_budget:
                continue
            changed = {x: self.transform(x, delta) for x in ids}
            measured = self.evaluate(changed)
            target = next((r for r in measured if {r['asset_a'], r['asset_b']} == {aid, bid}), None)
            target_area = target['known_solid_overlap_area_m2'] if target else 0.
            new_clear = [r for r in measured if r['classification'] == 'clear_overlap' and
                         (frozenset([r['asset_a'], r['asset_b']]) not in base_pairs or
                          base_pairs[frozenset([r['asset_a'], r['asset_b']])]['classification'] != 'clear_overlap')]
            # A correction may not turn another previously tolerated pair into
            # a new >5m2 local penetration, even if it narrowly misses clear.
            new_excess = sum(max(0., contact_penalty(r)-contact_penalty(base_pairs.get(frozenset([r['asset_a'], r['asset_b']]))))
                             for r in measured if {r['asset_a'], r['asset_b']} != {aid, bid})
            penalty = sum(contact_penalty(r) for r in measured)
            road_after = [self.road_measure(r) for r in changed.values()] if self.road_model else []
            road_regression = any(
                (r['max_single_lane_obstructed_length_m'] >= 1 and r['max_single_lane_obstructed_length_m'] > road_before[r['id']]['max_single_lane_obstructed_length_m']+.25)
                or r['areas_m2']['occupied'] > road_before[r['id']]['areas_m2']['occupied']+1.
                for r in road_after)
            walking_after = [self.walking_measure(r) for r in changed.values()] if self.pedestrian_model else []
            walking_regression = any(
                r['max_single_lane_obstructed_length_m'] >= 3
                and r['max_single_lane_obstructed_length_m'] > walking_before[r['id']]['max_single_lane_obstructed_length_m']+2
                and r['areas_m2']['occupied'] > walking_before[r['id']]['areas_m2']['occupied']+1
                for r in walking_after)
            results.append({'ids': ids, 'pivot_xy_m': list(map(float, pivot)),
                'translation_xy_m': [tx, ty], 'scale_xy': scale, 'world_delta_4x4': delta.tolist(),
                'before_target': target_baseline, 'after_target': target,
                'target_area_after_m2': target_area, 'cost': cost,
                'new_clear_pairs': new_clear, 'other_contact_excess_increase_m2': new_excess,
                'penalty_before': baseline_penalty, 'penalty_after': penalty,
                'road_regression': road_regression, 'road_after': road_after,
                'walking_regression': walking_regression, 'walking_after': walking_after,
                'measured_pairs_after': measured,
                'verification': 'actual_transformed_meshes_recast_at_0.25m'})
        results.sort(key=lambda r: (r['road_regression'] or r['walking_regression'], bool(r['new_clear_pairs']), r['other_contact_excess_increase_m2'] > .5,
                                   r['target_area_after_m2'] > 5, r['penalty_after'], r['cost']))
        return results
