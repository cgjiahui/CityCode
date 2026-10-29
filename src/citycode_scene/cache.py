"""Read a binding once; preserve GLB nodes, physical components and world poses."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import re

import numpy as np
from shapely.geometry import mapping

from citycode_assets.registration import read_glb_scene_geometry
from .geometry import Raster, footprint, json_write, sha256


def build_cache(binding_path, cache_dir, cell=.25, workers=4, reuse_cache_dir=None):
    binding_path, cache_dir = Path(binding_path).resolve(), Path(cache_dir).resolve()
    binding = json.loads(binding_path.read_text())
    binding_sha = sha256(binding_path)
    rows = [r for r in binding['asset_instances'] + binding.get('massing_instances', []) if r.get('assembly_enabled', True)]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Physical placements must have unique IDs')
    raster = Raster(cache_dir / 'native')
    old = {}
    reuse = Path(reuse_cache_dir).resolve() if reuse_cache_dir else None
    if reuse and (reuse/'INDEX.json').is_file():
        previous = json.loads((reuse/'INDEX.json').read_text())
        if previous['cell_m'] == cell:
            old = {r['id']: r for r in previous['records']}
    json_write(cache_dir/'INDEX.json', {'schema':'citycode.scene_physics_cache.v1',
        'binding_sha256':binding_sha, 'cell_m':cell, 'records':[], 'complete':False})

    def one(row):
        aid = row['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', aid):
            raise ValueError('Unsafe placement ID')
        folder = cache_dir/'instances'/aid
        folder.mkdir(parents=True, exist_ok=True)
        model = (binding_path.parent/row['lods']['0.1M']['path']).resolve()
        expected_sha = row['lods']['0.1M']['sha256']
        if sha256(model) != expected_sha:
            raise ValueError(f'Actual model identity changed: {aid}')
        matrix = np.asarray(row['world_from_local_4x4'], dtype=float)
        prior = old.get(aid)
        if prior and prior['source_model_sha256'] == expected_sha and np.array_equal(prior['world_from_local_4x4'], matrix):
            for file in (reuse/'instances'/aid).glob('*.npz'):
                destination = folder/file.name
                if destination.resolve() != file.resolve():
                    temporary = destination.with_suffix('.clone.tmp')
                    temporary.unlink(missing_ok=True)
                    try:
                        os.link(file, temporary)
                    except OSError:
                        import shutil
                        shutil.copy2(file, temporary)
                    temporary.replace(destination)
            meta = dict(prior)
            meta.update(status=row['status'], source_ids=row['source_building_ids'], entity_id=row.get('building_entity_id', aid),
                parent_source_id=row.get('parent_building_id'), source_model=str(model),
                target_base_z_m=row.get('target_base_z_m'), target_top_z_m=row.get('target_top_z_m'))
            json_write(folder/'metadata.json', meta)
            return meta
        scene = read_glb_scene_geometry(model)
        v = scene['vertices'] @ matrix[:3, :3].T + matrix[:3, 3]
        f = scene['faces'].astype(np.int32)
        np.savez(folder/'geometry.npz', vertices=v, faces=f)
        columns = raster.cast(v, f, cell)
        np.savez_compressed(folder/'columns.npz', **columns)
        mask = np.isfinite(columns['zmin'])
        counts = np.diff(columns['offsets'])
        active = int(np.count_nonzero(counts))
        instances = []
        fs = vs = 0
        groups = {}
        for instance in scene['instances']:
            instances.append({**instance, 'face_start': fs, 'vertex_start': vs})
            groups.setdefault(instance['node_index'], []).append((fs, fs+instance['triangle_count']))
            fs += instance['triangle_count']
            vs += instance['vertex_count']
        meta = {
            'id': aid, 'kind': 'massing' if aid.startswith('MASSING_') else 'base', 'status': row['status'],
            'source_ids': row['source_building_ids'], 'entity_id': row.get('building_entity_id', aid),
            'parent_source_id': row.get('parent_building_id'), 'world_bounds': [v.min(0).tolist(), v.max(0).tolist()],
            'source_model': str(model), 'source_model_sha256': expected_sha, 'lod': '0.1M',
            'triangles': len(f), 'vertices': len(v), 'world_from_local_4x4': matrix.tolist(),
            'geometry': str((folder/'geometry.npz').relative_to(cache_dir)),
            'columns': str((folder/'columns.npz').relative_to(cache_dir)), 'glb_instances': instances,
            'footprint_world': mapping(footprint(mask, columns['origin'], cell)),
            'sampled_projection_area_m2': active*cell**2,
            'column_raster': {'cell_m': cell, 'even_crossing_fraction': 1-int(np.count_nonzero(counts % 2))/max(1, active)},
            'target_base_z_m': row.get('target_base_z_m'), 'target_top_z_m': row.get('target_top_z_m'),
        }
        if len(groups) > 1:
            components = []
            for index, (node, ranges) in enumerate(groups.items()):
                selected = np.concatenate([f[start:end] for start, end in ranges])
                used = np.unique(selected)
                remap = np.full(len(v), -1, np.int32)
                remap[used] = np.arange(len(used))
                cv, cf = v[used], remap[selected]
                name = f'component_{index:02d}'
                np.savez(folder/(name+'_geometry.npz'), vertices=cv, faces=cf)
                cc = raster.cast(cv, cf, cell)
                np.savez_compressed(folder/(name+'_columns.npz'), **cc)
                components.append({'id': aid+f'__component{index:02d}', 'kind': 'composite_component',
                    'source_ids': meta['source_ids'], 'entity_id': meta['entity_id'],
                    'geometry': str((folder/(name+'_geometry.npz')).relative_to(cache_dir)),
                    'columns': str((folder/(name+'_columns.npz')).relative_to(cache_dir)),
                    'world_bounds': [cv.min(0).tolist(), cv.max(0).tolist()],
                    'footprint_world': mapping(footprint(np.isfinite(cc['zmin']), cc['origin'], cell)),
                    'sampled_projection_area_m2': float(np.isfinite(cc['zmin']).sum()*cell**2),
                    'glb_node_index': node, 'triangle_count': len(cf)})
            meta['components'] = components
        json_write(folder/'metadata.json', meta)
        return meta

    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(one, rows))
    if sha256(binding_path) != binding_sha:
        raise ValueError('Binding changed during actual-mesh cache construction')
    index = {'schema': 'citycode.scene_physics_cache.v1', 'binding': str(binding_path),
        'binding_sha256': binding_sha, 'coordinate_system': 'UE XYZ metres Z up; GLB nodes composed with world placement',
        'cell_m': cell, 'included_placements': len(rows),
        'excluded_observation_surfaces': len(binding.get('surface_instances', [])),
        'records': sorted(result, key=lambda r: r['id']), 'complete': True}
    json_write(cache_dir/'INDEX.json', index)
    return index
