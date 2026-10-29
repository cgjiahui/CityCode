#!/usr/bin/env python3
"""Export current, height-aware contact evidence for the read-only city viewer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely import contains_xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from citycode_scene import pairs
from citycode_scene.roads import RoadAudit, road_height, union_occupancy

SCHEMA = 'citycode.collision_preview.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Audit geometry path must remain inside its cache')
    return path


def intersections(a, b):
    """Intersect occupied spans, preserving gaps and disjoint height layers."""
    out = []
    for lo, hi in a:
        for lo2, hi2 in b:
            low, high = max(lo, lo2), min(hi, hi2)
            if high > low:
                out.append((low, high))
    return pairs.merge(out)


def patch(x, y, lo, hi, cell):
    return {'center': [float(x), float(y), float((lo+hi)/2)],
            'size': [float(cell), float(cell), float(hi-lo)]}


def geometry_fields(patches, fallback_bounds, limit):
    """Focus on current contact heights; sampling never enlarges a cell."""
    if patches:
        centers = np.asarray([p['center'] for p in patches])
        sizes = np.asarray([p['size'] for p in patches])
        bounds = [(centers-sizes/2).min(0).tolist(), (centers+sizes/2).max(0).tolist()]
        # A representative actual interval is safer than an averaged height
        # which could fall into empty space between stacked contact layers.
        nearest = int(np.argmin(np.sum((centers-centers.mean(0))**2, axis=1)))
        focus = centers[nearest].tolist()
    else:
        bounds = fallback_bounds
        focus = np.asarray(fallback_bounds).mean(0).tolist()
    count = len(patches)
    if count > limit:
        selected = np.linspace(0, count-1, limit, dtype=int)
        patches = [patches[i] for i in selected]
    return {'focus_position': focus, 'bounds': bounds, 'patches': patches,
            'patches_truncated': count > limit, 'patch_count_before_sampling': count}


def asset_bounds(records):
    boxes = np.asarray([r['world_bounds'] for r in records])
    return [boxes[:, 0].min(0).tolist(), boxes[:, 1].max(0).tolist()]


def display_id(aid):
    if aid.startswith('PGB_SC_'):
        return str(int(aid.rsplit('_', 1)[-1]))
    if aid.startswith('MASSING_PGB_SC_'):
        numbers = aid.rsplit('_', 2)[-2:]
        return f'{int(numbers[0])}·{int(numbers[1])}'
    return aid.removeprefix('SHARED_').lstrip('0')


def building_patches(a, b):
    ca, cb = pairs.columns_for(a), pairs.columns_for(b)
    gx0, gy0 = max(ca['gx'], cb['gx']), max(ca['gy'], cb['gy'])
    gx1, gy1 = min(ca['gx']+ca['w'], cb['gx']+cb['w']), min(ca['gy']+ca['h'], cb['gy']+cb['h'])
    if gx1 <= gx0 or gy1 <= gy0:
        return []
    aa = (slice(gy0-ca['gy'], gy1-ca['gy']), slice(gx0-ca['gx'], gx1-ca['gx']))
    bb = (slice(gy0-cb['gy'], gy1-cb['gy']), slice(gx0-cb['gx'], gx1-cb['gx']))
    lo, hi, lo2, hi2 = ca['zmin'][aa], ca['zmax'][aa], cb['zmin'][bb], cb['zmax'][bb]
    possible = np.isfinite(lo) & np.isfinite(lo2) & (np.minimum(hi, hi2)-np.maximum(lo, lo2) > .05)
    out = []
    for y, x in zip(*np.where(possible)):
        gx, gy = int(gx0+x), int(gy0+y)
        az, _ = pairs.actual_spans(a, gx, gy)
        bz, _ = pairs.actual_spans(b, gx, gy)
        spans = intersections(az, bz)
        if sum(h-l for l, h in spans) <= .05:
            continue
        out.extend(patch((gx+.5)*pairs.CELL, (gy+.5)*pairs.CELL, l, h, pairs.CELL) for l, h in spans)
    return out


def _priority_pairs(assets, before_report):
    priority, before, fine = set(), {}, {}
    if before_report:
        path = Path(before_report)
        if path.is_dir():
            path = path/'buildings/PAIRS.json' if (path/'buildings/PAIRS.json').is_file() else path/'PAIRS.json'
        source = read(path)
        for row in source.get('pairs', []):
            if not row.get('same_building_family') and row.get('classification') == 'clear_overlap':
                key = tuple(sorted([row['asset_a'], row['asset_b']]))
                priority.add(key)
                before[key] = row['known_solid_overlap_area_m2']
    file = assets/'physical_postprocess/fine_pairs.json'
    review = assets/'physical_postprocess/review.json'
    if file.is_file() and review.is_file() and read(review).get('physical_binding_sha256') == sha(_binding_path(assets)):
        for row in read(file).get('checks', []):
            key = tuple(sorted(row['pair']))
            priority.add(key)
            before.setdefault(key, row.get('before_area_m2'))
            fine[key] = row.get('after_fine')
    return priority, before, fine


def _binding_path(root):
    physical = root/'binding_manifest_physical.json'
    return physical if physical.is_file() else root/'binding_manifest.json'


def _resolve_assets(root):
    root = Path(root).resolve()
    if (root/'binding_manifest.json').is_file() or (root/'binding_manifest_physical.json').is_file():
        return root
    matches = list(root.glob('data/*/binding_manifest_physical.json'))
    if len(matches) != 1:
        raise ValueError('Provide the asset release directory containing the binding manifest')
    return matches[0].parent


def _lane_path(assets, expected_sha, explicit):
    if explicit:
        path = Path(explicit).resolve()
        if sha(path) != expected_sha:
            raise ValueError('Lane input checksum differs from the current road audit')
        return path
    for path in assets.parent.rglob('lane_centerlines.geojson'):
        if sha(path) == expected_sha:
            return path.resolve()
    raise ValueError('Supply --lane-geojson matching the road audit checksum')


def vehicle_patches(meta, row, model):
    c = pairs.columns_for(meta)
    lane_by_id = {lane['lane_id']: lane for lane in model.lanes}
    components = [pairs.columns_for(x) for x in meta.get('components', [meta])]
    intervals = {}
    for lr in row['lanes']:
        if lr['occupied_area_m2'] <= 0:
            continue
        lane = lane_by_id[lr['lane_id']]
        poly = lane['corridor']
        origin, cell, h, w = c['origin'], float(c['cell']), c['h'], c['w']
        b = poly.bounds
        x0, x1 = max(0, math.floor((b[0]-origin[0])/cell)), min(w, math.ceil((b[2]-origin[0])/cell))
        y0, y1 = max(0, math.floor((b[1]-origin[1])/cell)), min(h, math.ceil((b[3]-origin[1])/cell))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        xy = origin+(np.column_stack([xx.ravel(), yy.ravel()])+.5)*cell
        keep = contains_xy(poly, xy[:, 0], xy[:, 1]) & np.isfinite(c['zmin'][yy.ravel(), xx.ravel()])
        xy = xy[keep]
        z, _ = road_height(xy, lane)
        occupied = union_occupancy(xy, z, components, model.clearance_m)['occupied']
        for point, road_z in zip(xy[occupied], z[occupied]):
            gx, gy = np.floor(point/cell).astype(int)
            spans, _ = pairs.actual_spans(meta, int(gx), int(gy))
            clipped = intersections(spans, [(road_z+.25, road_z+model.clearance_m)])
            # An open column may still contain an observed obstructing surface.
            if not clipped:
                for component in components:
                    ix, iy = np.floor((point-component['origin'])/cell).astype(int)
                    if not (0 <= ix < component['w'] and 0 <= iy < component['h']):
                        continue
                    pos = iy*component['w']+ix
                    zs = component['hits'][component['offsets'][pos]:component['offsets'][pos+1]]
                    clipped.extend((max(float(v)-.015, road_z+.25), min(float(v)+.015, road_z+model.clearance_m)) for v in zs if road_z+.25 <= v <= road_z+model.clearance_m)
            intervals.setdefault((int(gx), int(gy)), []).extend((l, h) for l, h in clipped if h > l)
    return [patch((gx+.5)*pairs.CELL, (gy+.5)*pairs.CELL, l, h, pairs.CELL)
            for (gx, gy), spans in sorted(intervals.items()) for l, h in pairs.merge(spans)]


def ground_patches(audit, row, meta):
    file = audit/'ground/samples'/f"{row['id']}.npz"
    if not file.is_file():
        raise ValueError('Ground contact samples are required for accurate preview locations')
    with np.load(file) as s:
        xy, bottom, scene, continuous = (s[k] for k in ['xy', 'bottom_z', 'scene_ground_z', 'continuous_z'])
        mask = np.isfinite(bottom) & np.isfinite(scene) & np.isfinite(continuous) & (scene-bottom > .5) & (scene-continuous > .1)
        return [patch(x, y, float(lo), float(hi), pairs.CELL)
                for (x, y), lo, hi in zip(xy[mask], bottom[mask], scene[mask])]


def prepare_collision_preview(release_root, audit_root, *, before_report=None, lane_geojson=None, output_path=None, max_patches=1000):
    assets, audit = _resolve_assets(release_root), Path(audit_root).resolve()
    if not 1 <= max_patches <= 1000:
        raise ValueError('max_patches must be between1 and1000')
    binding_file = _binding_path(assets)
    binding, binding_hash = read(binding_file), sha(binding_file)
    cache = audit/'cache'
    index, audit_summary = read(cache/'INDEX.json'), read(audit/'SUMMARY.json')
    ground, building_summary = read(audit/'ground/GROUND_CONTACTS.json'), read(audit/'buildings/SUMMARY.json')
    for source in [index, audit_summary, ground, building_summary]:
        if source.get('binding_sha256') != binding_hash:
            raise ValueError('Current binding checksum differs from the audit; recompute before exporting hotspots')
    enabled = {r['id']: r for key in ['asset_instances', 'massing_instances'] for r in binding.get(key, []) if r.get('assembly_enabled', True)}
    by = {r['id']: r for r in index['records']}
    if set(enabled) != set(by):
        raise ValueError('Audit placement IDs differ from the current enabled release placements')
    for aid, meta in by.items():
        if not np.allclose(meta['world_from_local_4x4'], enabled[aid]['world_from_local_4x4'], atol=1e-8, rtol=0):
            raise ValueError('Audit placement matrix differs from the current binding')
        if meta['source_model_sha256'] != enabled[aid]['lods']['0.1M']['sha256']:
            raise ValueError('Audit source model differs from the current binding')
        for descriptor in [meta]+meta.get('components', []):
            contained(cache, descriptor['columns'])
    pairs.configure(cache)
    priority, before, fine = _priority_pairs(assets, before_report)
    cases = []
    for row in read(audit/'buildings/PAIRS.json')['pairs']:
        if row.get('same_building_family') or row.get('known_solid_overlap_area_m2', 0) <= 0:
            continue
        ids = sorted([row['asset_a'], row['asset_b']])
        key = tuple(ids)
        records = [pairs.by[x] for x in ids]
        patches = building_patches(*records)
        if not patches:
            raise ValueError('Positive building audit contact has no current occupied mesh intervals')
        reviewed = key in priority
        boundary_review = row['classification'] in {'review_overlap', 'clear_overlap'}
        metrics = {'overlap_area_m2': row['known_solid_overlap_area_m2'], 'overlap_volume_m3': row['known_overlap_volume_m3'],
                   'max_vertical_overlap_m': row['max_common_vertical_occupied_length_m'], 'horizontal_depth_m': row['max_horizontal_interior_radius_m'],
                   'classification': row['classification'], 'sample_cell_m': pairs.CELL}
        if key in before:
            metrics['before_overlap_area_m2'] = before[key]
        if fine.get(key):
            metrics['fine_overlap_area_m2'] = fine[key]['known_solid_overlap_area_m2']
            metrics['fine_median_thickness_m'] = fine[key].get('common_vertical_thickness_median_m')
        cases.append({'id': 'building:'+'__'.join(ids), 'kind': 'building', 'asset_ids': ids,
                      'label': '建筑'+' / '.join(map(display_id, ids))+(' · 已复核保留' if reviewed else ' · 边界接触' if boundary_review else ' · 轻微接触'),
                      'severity': 'review' if reviewed or boundary_review else 'minor',
                      'review_status': 'reviewed_retained' if reviewed else 'boundary_review' if boundary_review else 'minor_contact',
                      'default_visible': reviewed, 'metrics': metrics, 'asset_bounds': asset_bounds(records),
                      'description': '当前模型之间的局部实体接触；已排除仅平面投影相交、共同高度为空的情况。',
                      'reason': ('保留少量接缝、檐口或边界接触；标记不是删除建筑的指令。' if reviewed or not boundary_review
                                 else '几何采样发现的边界接触，尚未逐处人工复核；是否调整需结合邻楼外观判断。'),
                      **geometry_fields(patches, asset_bounds(records), max_patches)})
    road_dir = audit/'roads/car_2m'
    road_summary = read(road_dir/'SUMMARY.json')
    if road_summary.get('source_cache_index_sha256') != sha(cache/'INDEX.json'):
        raise ValueError('Road audit was measured from another geometry cache')
    lanes = _lane_path(assets, road_summary['lane_geojson_sha256'], lane_geojson)
    model = RoadAudit(cache, lanes, clearance_m=float(road_summary['clearance_m']))
    vehicle_rows = [r for r in read(road_dir/'DETAILS.json')['instances'] if r['areas_m2']['occupied'] > 0]
    blocked = sum(r['max_single_lane_obstructed_length_m'] >= road_summary['thresholds']['one_lane_centerline_block_m'] for r in vehicle_rows)
    for row in vehicle_rows:
        meta = pairs.by[row['id']]
        if sha(cache/meta['columns']) != row['columns_sha256']:
            raise ValueError('Road occupancy columns changed after the audit')
        patches = vehicle_patches(meta, row, model)
        if not patches:
            raise ValueError('Positive road audit contact has no current vehicle-height mesh intervals')
        is_blocked = row['max_single_lane_obstructed_length_m'] >= road_summary['thresholds']['one_lane_centerline_block_m']
        cases.append({'id': 'vehicle:'+meta['id'], 'kind': 'vehicle', 'asset_ids': [meta['id']],
                      'label': '建筑'+display_id(meta['id'])+(' · 车道需复核' if is_blocked else ' · 车道边缘接触'),
                      'severity': 'review' if is_blocked else 'minor', 'review_status': 'minor_contact', 'default_visible': is_blocked,
                      'metrics': {'occupied_area_m2': row['areas_m2']['occupied'], 'max_intrusion_m': row['max_margin_intrusion_m'],
                                  'max_single_lane_obstructed_length_m': row['max_single_lane_obstructed_length_m'],
                                  'clearance_m': model.clearance_m, 'centerline_obstruction': is_blocked,
                                  'affected_lane_count': sum(x['occupied_area_m2'] > 0 for x in row['lanes'])},
                      'asset_bounds': meta['world_bounds'], 'description': '当前2米车体净空与模型的局部相交。',
                      'reason': '边缘接触与中心线阻塞分别统计；允许保留少量边缘接触。',
                      **geometry_fields(patches, meta['world_bounds'], max_patches)})
    for row in ground['records']:
        if not row['scene_category'].startswith('embedded_over_'):
            continue
        meta = pairs.by[row['id']]
        patches = ground_patches(audit, row, meta)
        if not patches:
            continue
        cases.append({'id': 'ground:'+meta['id'], 'kind': 'ground', 'asset_ids': [meta['id']],
                      'label': '建筑'+display_id(meta['id'])+' · 背景局部凸起', 'severity': 'review',
                      'review_status': 'background_review', 'default_visible': True,
                      'metrics': {'scene_gap_median_m': row['scene_ground_gap']['median_m'],
                                  'continuous_gap_median_m': row['continuous_ground_gap']['median_m'],
                                  'embedded_fraction': row['scene_ground_gap']['below_minus_0p50m_fraction'],
                                  'foot_area_m2': row['foot_sample_area_m2'], 'foot_samples': row['foot_samples']},
                      'asset_bounds': meta['world_bounds'], 'description': '附近背景细节高于建筑脚部；不代表建筑悬空。',
                      'reason': '保留建筑位置，避免为贴合背景局部凸起而抬高整栋建筑；后续可检查该处背景。',
                      **geometry_fields(patches, meta['world_bounds'], max_patches)})
    cases.sort(key=lambda x: (not x['default_visible'], x['kind'], x['id']))
    counts = {'building_contacts': sum(c['kind'] == 'building' for c in cases), 'vehicle_edge_contacts': sum(c['kind'] == 'vehicle' and not c['metrics']['centerline_obstruction'] for c in cases),
              'vehicle_centerline_obstructions': blocked, 'ground_background_reviews': sum(c['kind'] == 'ground' for c in cases),
              'default_visible_cases': sum(c['default_visible'] for c in cases)}
    payload = {'schema': SCHEMA, 'binding_sha256': binding_hash, 'coordinate_system': 'UE XYZ metres, Z up',
               'geometry_lod': '0.1M', 'counts': counts, 'default_case_ids': [c['id'] for c in cases if c['default_visible']],
               'cases': cases, 'source_audit_sha256': sha(audit/'SUMMARY.json'),
               'note': '仅显示当前摆放的实际接触；全城标记定位，选中后显示接触区域。已排除同组拼接与仅平面相交。'}
    output = Path(output_path) if output_path else assets/'physical_postprocess/collision_preview.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', '--assets-root', dest='release_root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--before-report', type=Path)
    parser.add_argument('--lane-geojson', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-patches', type=int, default=1000)
    args = parser.parse_args()
    value = prepare_collision_preview(args.release_root, args.audit_root, before_report=args.before_report,
                                      lane_geojson=args.lane_geojson, output_path=args.output, max_patches=args.max_patches)
    print(json.dumps({'binding_sha256': value['binding_sha256'], 'counts': value['counts'], 'case_count': len(value['cases'])}))


if __name__ == '__main__':
    main()
