#!/usr/bin/env python3
"""Read-only local preview of the released asset binding; no GPU or API calls."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
RECENT = [337, 364, 428, 429, 495, 496, 536, 553, 570, 571, 574]


def short_id(value):
    return str(int(value.rsplit('_', 1)[-1]))


def simplify_ring(ring, tolerance=0.65):
    """Display geometry only. Original structures and mesh files stay untouched."""
    points = [[round(p[0], 3), round(p[1], 3)] for p in ring]
    if len(points) <= 8:
        return points
    def rdp(points):
        if len(points) < 3:
            return points
        ax, ay = points[0]; bx, by = points[-1]
        length2 = (bx-ax)**2 + (by-ay)**2
        distance, pivot = 0, 0
        for i, (x, y) in enumerate(points[1:-1], 1):
            t = max(0, min(1, ((x-ax)*(bx-ax)+(y-ay)*(by-ay))/length2)) if length2 else 0
            d = (x-ax-t*(bx-ax))**2 + (y-ay-t*(by-ay))**2
            if d > distance:
                distance, pivot = d, i
        if distance > tolerance*tolerance:
            return rdp(points[:pivot+1])[:-1] + rdp(points[pivot:])
        return [points[0], points[-1]]
    # Split a closed ring to avoid the coincident-endpoint degeneracy.
    mid = len(points)//2
    result = rdp(points[:mid+1])[:-1]+rdp(points[mid:])
    return result if len(result) >= 4 else points


def display_polygons(geometry):
    if not geometry:
        return []
    coords = geometry['coordinates']
    return [[simplify_ring(ring) for ring in polygon]
            for polygon in ([coords] if geometry['type'] == 'Polygon' else coords)]


def add_background(scene, allowed, root):
    if root is None or not (root/'manifest.json').is_file():
        scene['background'] = None
        return
    root = root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != 'citycode.background_web_preview.v1':
        raise ValueError('Invalid background preview manifest')
    files = {}
    for role in ['map', 'surfaces'] + [role for role in ['neural','coastal','map_solid'] if role in manifest['files']]:
        row = manifest['files'][role]
        file = (root/row['path']).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Invalid background preview path')
        if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Background preview checksum failed')
        key = 'background/'+row['path']; allowed[key] = file
        files[role] = {**row, 'url': '/data/'+key}
    scene['background'] = {**manifest, 'files': files}
    scene['note'] = '当前建筑叠加历史GT辅助背景；可检查位置差异，尚未完成场景融合验收。'


def add_collision_audit(scene, allowed, root):
    """Embed only validated display evidence from the current physical binding."""
    scene['collision_audit'] = None
    file = (root/'physical_postprocess/collision_preview.json').resolve()
    if not file.is_file():
        return
    if not file.is_relative_to(root.resolve()):
        raise ValueError('Collision preview must remain inside the release')
    payload = json.loads(file.read_text())
    if payload.get('schema') != 'citycode.collision_preview.v1':
        raise ValueError('Invalid collision preview schema')
    if payload.get('binding_sha256') != scene['binding_sha256']:
        scene['collision_audit_status'] = 'binding_mismatch'
        return
    known = {a['id'] for a in scene.get('assets', []) if not a.get('surface') and a.get('enabled', True)}
    def vector(value, size=3):
        if not isinstance(value, list) or len(value) != size or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
            raise ValueError('Invalid collision preview coordinates')
        return value
    def bounds(value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError('Invalid collision preview bounds')
        lo, hi = vector(value[0]), vector(value[1])
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError('Invalid collision preview bounds')
        return [lo, hi]
    metric_keys = {
        'building': {'overlap_area_m2', 'overlap_volume_m3', 'max_vertical_overlap_m', 'horizontal_depth_m', 'classification', 'sample_cell_m', 'before_overlap_area_m2', 'fine_overlap_area_m2', 'fine_median_thickness_m'},
        'vehicle': {'occupied_area_m2', 'max_intrusion_m', 'max_single_lane_obstructed_length_m', 'clearance_m', 'centerline_obstruction', 'affected_lane_count'},
        'ground': {'scene_gap_median_m', 'continuous_gap_median_m', 'embedded_fraction', 'foot_area_m2', 'foot_samples'},
    }
    cases, seen = [], set()
    for row in payload.get('cases', []):
        kind, case_id = row.get('kind'), row.get('id')
        if kind not in metric_keys or not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('Invalid collision preview case')
        seen.add(case_id)
        if not isinstance(row.get('asset_ids'), list) or not row['asset_ids'] or any(not isinstance(aid, str) or aid not in known for aid in row['asset_ids']):
            raise ValueError('Collision preview refers to an unknown or disabled asset')
        if not isinstance(row.get('label'), str):
            raise ValueError('Invalid collision preview label')
        if row.get('severity') not in {'review', 'minor', 'resolved'} or row.get('review_status') not in {'reviewed_retained', 'boundary_review', 'minor_contact', 'background_review', 'resolved'}:
            raise ValueError('Invalid collision preview review status')
        patches = row.get('patches', [])
        if not isinstance(patches, list) or len(patches) > 1000:
            raise ValueError('Collision preview patch limit exceeded')
        cleaned = []
        for patch in patches:
            centre, size = vector(patch['center']), vector(patch['size'])
            if any(v <= 0 for v in size):
                raise ValueError('Invalid collision preview patch size')
            cleaned.append({'center': centre, 'size': size})
        metrics = {k: v for k, v in row.get('metrics', {}).items() if k in metric_keys[kind]}
        if any(not isinstance(v, (str, bool, int, float, type(None))) or isinstance(v, float) and not math.isfinite(v) for v in metrics.values()):
            raise ValueError('Invalid collision preview metrics')
        case = {k: row[k] for k in ['id', 'kind', 'asset_ids', 'label', 'severity', 'review_status']}
        case.update(default_visible=bool(row.get('default_visible')), metrics=metrics,
                    focus_position=vector(row['focus_position']), bounds=bounds(row['bounds']),
                    patches=cleaned, patches_truncated=bool(row.get('patches_truncated')))
        for key in ['description', 'reason']:
            if isinstance(row.get(key), str):
                case[key] = row[key]
        if 'asset_bounds' in row:
            case['asset_bounds'] = bounds(row['asset_bounds'])
        if isinstance(row.get('patch_count_before_sampling'), int):
            case['patch_count_before_sampling'] = row['patch_count_before_sampling']
        cases.append(case)
    counts = {
        'building_contacts': sum(c['kind'] == 'building' for c in cases),
        'vehicle_edge_contacts': sum(c['kind'] == 'vehicle' and not c['metrics'].get('centerline_obstruction') for c in cases),
        'vehicle_centerline_obstructions': sum(c['kind'] == 'vehicle' and bool(c['metrics'].get('centerline_obstruction')) for c in cases),
        'ground_background_reviews': sum(c['kind'] == 'ground' for c in cases),
        'default_visible_cases': sum(c['default_visible'] for c in cases),
    }
    # Source cache paths and arbitrary local files never enter /api/scene or
    # the allowed-file table. The overlay is wholly described by this whitelist.
    scene['collision_audit'] = {'schema': payload['schema'], 'binding_sha256': payload['binding_sha256'],
                                'coordinate_system': 'UE XYZ metres, Z up', 'geometry_lod': payload.get('geometry_lod', '0.1M'),
                                'counts': counts, 'default_case_ids': [c['id'] for c in cases if c['default_visible']], 'cases': cases,
                                'note': '当前模型的接触定位；轻微接触不等于需要删除建筑。'}


def build_scene(root, background_root=None):
    binding_path = root / 'binding_manifest_physical.json'
    if not binding_path.is_file():
        binding_path = root / 'binding_manifest.json'
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    physical = binding.get('physical_postprocess')
    if physical and physical['source_binding_sha256'] != hashlib.sha256((root/physical['source_binding']).read_bytes()).hexdigest():
        raise ValueError('Physical placement source changed; replay scene postprocessing before preview')
    buildings = {b['building_id']: b for b in binding['buildings']}
    allowed = {}
    members = {}
    for entity in binding.get('building_entities', []):
        for asset_id in entity['asset_instance_ids']:
            members[asset_id] = entity['asset_instance_ids']

    def file_url(relative):
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f'Invalid preview asset: {relative}')
        allowed[relative] = file
        return '/data/' + relative

    assets = []
    massing_ids={r['id'] for r in binding.get('massing_instances',[])}
    recipe_path = root/'massing_completion/recipe.json'
    decisions = {r['source_id']: r for r in json.loads(recipe_path.read_text()).get('source_decisions', [])} if recipe_path.is_file() else {}
    for item in binding.get('massing_instances',[]):
        related=[r['id'] for r in binding['massing_instances'] if r['source_building_ids']==item['source_building_ids']]
        members[item['id']]=related
    for item in binding['asset_instances'] + binding.get('massing_instances', []) + binding.get('surface_instances', []):
        surface = item['id'].startswith('SURFACE_')
        massing=item['id'] in massing_ids
        source_ids = item['source_building_ids']
        b = buildings.get(source_ids[0], {})
        polygons = []
        for source_id in source_ids:
            geometry = item.get('target_geometry') if massing else buildings.get(source_id, {}).get('target_geometry')
            if geometry:
                coords = geometry['coordinates']
                for poly in ([coords] if geometry['type'] == 'Polygon' else coords):
                    polygons.append([simplify_ring(ring) for ring in poly])
        matrix = item['world_from_local_4x4']
        bounds = item['lods']['0.1M']['bounds_local_accessor']
        corners = [[sum(matrix[j][k]*v[k] for k in range(3))+matrix[j][3] for j in range(3)]
                   for x in bounds for y in bounds for z in bounds
                   for v in [(x[0], y[1], z[2])]]
        world_bounds = [[min(p[j] for p in corners) for j in range(3)],
                        [max(p[j] for p in corners) for j in range(3)]]
        evidence = {}
        if massing:
            for role,name in [('front','parts_front.png'),('back','parts_back.png'),('context','after_front.png'),('source','observed.png')]:
                relative=f'massing_completion/review/{source_ids[0]}/{name}'
                if (root/relative).is_file():evidence[role]=file_url(relative)
        for role, relative in {
            'front': f'scene_completion/review/{item["id"]}/actual_model_front.png',
            'back': f'scene_completion/review/{item["id"]}/actual_model_back.png',
            'context': f'scene_completion/review/{item["id"]}/scene_front.jpg',
            'previous': f'fine_refinement/review/models/{item["id"]}/high_front.png',
            'appearance': f'scene_completion/appearance/{item["id"]}/generated_main_oblique.png',
        }.items():
            if (root / relative).is_file(): evidence[role] = file_url(relative)
        for role, filename in [('appearance', 'generated_main_oblique.png'), ('source', 'inputs/source_primary.png')]:
            relative = f'fine_refinement/inputs/{item["id"]}/appearance/{filename}'
            if role not in evidence and (root / relative).is_file(): evidence[role] = file_url(relative)
        if 'front' not in evidence:
            relative = f'fine_refinement/review/models/{item["id"]}/high_front.png'
            if (root / relative).is_file(): evidence['front'] = file_url(relative)
        assets.append({
            'id': item['id'], 'number': short_id(source_ids[0]), 'sources': source_ids,
            'status': item['status'], 'matrix': matrix, 'bounds': world_bounds,
            'center': [(world_bounds[0][j]+world_bounds[1][j])/2 for j in range(3)],
            'polygons': polygons, 'base': item['target_base_z_m'] if massing else b.get('base_z_m', world_bounds[0][2]),
            'top': item['target_top_z_m'] if massing else b.get('top_z_m', world_bounds[1][2]),
            'surface': surface, 'enabled': item.get('assembly_enabled', True),
            'massing':massing,'display_number':short_id(source_ids[0])+('·'+str(int(item['id'].rsplit('_',1)[-1])) if massing else ''),
            'donor_asset_id': item.get('donor_asset_id'),
            'parent_source_id': item.get('parent_building_id'),
            'layer': ('accepted_surface' if item.get('assembly_enabled', True) else 'observation_reference') if surface else 'building_model',
            'group': members.get(item['id'], [item['id']]),
            'recent': int(short_id(source_ids[0])) in RECENT and not surface,
            'review_candidate': not surface and (root/'fine_refinement/inputs'/item['id']).is_dir(),
            'reason': ('按观测的缺失裙楼、翼部或边带复用已有资产补全；未观测的外观与体量为近似推断。' if massing else b.get('reason', '')), 'evidence': evidence,
            'lods': {key: {'url': file_url(value['path']), 'bytes': value['bytes'], 'faces': value['faces']}
                     for key, value in item['lods'].items()},
        })
    massing_sources = []
    for sid in sorted({sid for part in binding.get('massing_instances', []) for sid in part['source_building_ids']}):
        parts = [part for part in assets if part['massing'] and sid in part['sources']]
        original = buildings.get(sid, {})
        decision = decisions.get(sid, {})
        geometry = decision.get('source_geometry') or original.get('postprocess', {}).get('source_target_geometry') or original.get('target_geometry')
        evidence = {}
        for role, name in [('before', 'before_front.png'), ('after', 'after_front.png'),
                           ('parts', 'parts_front.png'), ('source', 'observed.png')]:
            relative = f'massing_completion/review/{sid}/{name}'
            if (root/relative).is_file():
                evidence[role] = file_url(relative)
        massing_sources.append({'id': sid, 'number': short_id(sid), 'part_ids': [p['id'] for p in parts],
            'parent_source_id': decision.get('parent_building_id') or parts[0]['parent_source_id'],
            'polygons': display_polygons(geometry),
            'outline_z': max(p['top'] for p in parts)+.7,
            'description': decision.get('reason') or decision.get('prior_disposition_reason', '').split(' 原始表面')[0],
            'evidence': evidence})
    primary = [a for a in assets if not a['surface'] and a['enabled']]
    preview_root = (root/binding.get('web_preview_directory','../smallcity_p384_web_preview')).resolve()
    if not preview_root.is_relative_to(root.parent):raise ValueError('Preview directory must remain in the packaged data tree')
    preview_index = preview_root/'manifest.json'
    preview_summary = None
    if preview_index.is_file():
        preview = json.loads(preview_index.read_text())
        binding_hash = hashlib.sha256(binding_bytes).hexdigest()
        source_hashes = {binding_hash}
        if physical and not physical['geometry_bytes_modified']:
            source_hashes.add(physical['source_binding_sha256'])
        if preview.get('binding_sha256') in source_hashes and preview.get('count') == len(primary):
            by_id = {a['id']: a for a in primary}
            source_models = {item['id']: item['lods']['0.1M']['sha256'] for item in binding['asset_instances']+binding.get('massing_instances', [])}
            for record in preview['assets']:
                if record.get('source_sha256') != source_models.get(record['id']):
                    raise ValueError('Preview geometry does not match the current physical placement model')
                file = (preview_root/record['path']).resolve()
                if not file.is_relative_to(preview_root.resolve()) or not file.is_file():
                    raise ValueError('Invalid derived city preview path')
                if hashlib.sha256(file.read_bytes()).hexdigest() != record['preview_sha256']:
                    raise ValueError('Derived city preview checksum failed')
                key = 'preview/'+record['path'];allowed[key] = file
                by_id[record['id']]['lods']['preview'] = {'url':'/data/'+key,'bytes':record['bytes'],'faces':record['faces']}
            preview_summary = {k:preview[k] for k in ['count','bytes','faces','view_only']}
    scene = {
        'schema': 'citycode.web_scene_preview.v1',
        'binding_sha256': hashlib.sha256(binding_bytes).hexdigest(),
        'binding_name': binding_path.name,
        'physical_postprocess': physical,
        'coordinate_system': 'UE XYZ metres Z-up; glTF local transforms preserved',
        'counts': {'assets': len(primary), 'entities': len(binding['building_entities']),
                   'base_assets':len(binding['asset_instances']),'massing_parts':len(massing_ids),
                   'source_instances': len(binding['buildings']), 'surfaces': len(assets)-len(primary)},
        'recent': [f'PGB_SC_{n:06d}' for n in RECENT],
        'bounds': [[min(a['bounds'][0][j] for a in primary) for j in range(3)],
                   [max(a['bounds'][1][j] for a in primary) for j in range(3)]],
        'assets': assets,
        'massing_sources': massing_sources,
        'massing_pruning': {key: value for key, value in binding.get('massing_completion', {}).get('pruning', {}).items()
                            if key in {'previous_parts', 'retained_parts', 'removed_parts', 'removed_source_ids'}},
        'city_preview': preview_summary,
        'note': '建筑资产的空间预览；背景、道路和最终 CityCode 场景尚未装配。浅灰体块仅为结构概览。',
    }
    pruning=binding.get('massing_completion',{}).get('pruning',{})
    excluded=pruning.get('policy',{}).get('excluded_source_ids',[])
    scene['massing_pruning']['explicit_excluded_source_ids']=excluded
    scene['massing_pruning']['empty_after_part_review_source_ids']=[
        sid for sid in pruning.get('removed_source_ids',[]) if sid not in excluded]
    add_collision_audit(scene, allowed, root)
    add_background(scene, allowed, background_root)
    return scene, allowed


def make_handler(scene, allowed):
    scene_bytes = json.dumps(scene, ensure_ascii=False, separators=(',', ':')).encode()
    scene_gzip = gzip.compress(scene_bytes)
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self): self.serve(head=True)
        def do_GET(self): self.serve(head=False)
        def log_message(self, fmt, *args):
            if args and isinstance(args[0], str) and '/data/' in args[0]: return
            super().log_message(fmt, *args)
        def serve(self, head):
            path = unquote(urlsplit(self.path).path)
            if path == '/api/scene':
                compressed = 'gzip' in self.headers.get('Accept-Encoding', '')
                data = scene_gzip if compressed else scene_bytes
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Vary', 'Accept-Encoding')
                if compressed: self.send_header('Content-Encoding', 'gzip')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                if not head: self.wfile.write(data)
                return
            if path.startswith('/data/'):
                file = allowed.get(path[6:])
            else:
                dist = (HERE / 'dist').resolve()
                file = (dist / (path.lstrip('/') or 'index.html')).resolve()
                if not file.is_relative_to(dist): file = None
            if file is None or not file.is_file():
                self.send_error(404); return
            size = file.stat().st_size
            first, last, partial = 0, size-1, False
            if 'Range' in self.headers:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers['Range'])
                if not match or not any(match.groups()):
                    self.send_error(416); return
                start, end = match.groups()
                first, last = (int(start), int(end) if end else size-1) if start else (max(0, size-int(end)), size-1)
                last = min(last, size-1)
                if first > last or first >= size:
                    self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return
                partial = True
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', 'model/gltf-binary' if file.suffix == '.glb' else mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(last-first+1))
            self.send_header('Cache-Control', 'private, max-age=3600' if path.startswith('/data/') else 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if partial: self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
            self.end_headers()
            if head: return
            try:
                with file.open('rb') as stream:
                    stream.seek(first); remaining = last-first+1
                    while remaining:
                        data = stream.read(min(256*1024, remaining))
                        if not data: break
                        self.wfile.write(data); remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError): pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=HERE.parent/'data/smallcity_p384_building_assets_completed')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8794)
    parser.add_argument('--background-root', type=Path, default=HERE.parent/'data/smallcity_p384_background_web_preview')
    args = parser.parse_args()
    if not (HERE/'dist/index.html').is_file(): parser.error('Build the viewer first: npm ci && npm run build')
    scene, allowed = build_scene(args.assets_root.resolve(), args.background_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scene, allowed))
    print(f'CityCode preview: http://{args.host}:{args.port}/ | {scene["counts"]} | read-only', flush=True)
    # The handler keeps serialized JSON; release the construction-time scene tree.
    del scene
    import gc
    gc.collect()
    try:
        import ctypes
        trim = getattr(ctypes.CDLL(None), 'malloc_trim', None)
        if trim is not None: trim(0)
    except (OSError, AttributeError):
        pass
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
