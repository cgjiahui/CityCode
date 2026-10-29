"""Portable bundle contracts; no provider, research path or network fallback."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local(root, relative):
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError('Bundle paths must be relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Bundle path escapes root')
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen input: {relative}; no API fallback is available')
    return path


def load_bundle(root):
    root = Path(root).resolve()
    cfg = read(root / 'manifest.json')
    if cfg.get('schema') != 'citycode.background_history.v1' or cfg.get('lineage') != 'historical_gt_assisted':
        raise ValueError('Expected a disclosed historical background bundle')
    if cfg.get('network_policy') != 'offline_only_no_provider_client':
        raise ValueError('Unexpected execution policy')
    for key, ident in [('tiles', 'tile_id'), ('patches', 'patch_id')]:
        ids = [r[ident] for r in cfg[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate {key} IDs')
    ids = {t['tile_id'] for t in cfg['tiles']}
    selected, water = cfg['selected_ids'], cfg['water_tile_ids']
    if len(set(selected)) != len(selected) or len(set(water)) != len(water):
        raise ValueError('Duplicate selected tile IDs')
    if set(selected) & set(water) or not (set(selected) | set(water)) <= ids:
        raise ValueError('Invalid selected/water tile partition')
    for row in cfg['tiles'] + cfg['patches']:
        for value in row['files'].values():
            local(root, value)
        for key in ['v9_result', 'v10_result']:
            if row.get(key):
                local(root, row[key])
    return root, cfg


def verify(root):
    root, cfg = load_bundle(root)
    checksums = read(root / 'CHECKSUMS.json')
    seen = set()
    for row in checksums['files']:
        if row['path'] in seen:
            raise ValueError('Duplicate checksum entry')
        seen.add(row['path'])
        p = local(root, row['path'])
        if p.stat().st_size != row['bytes'] or sha256(p) != row['sha256']:
            raise ValueError(f'Frozen input checksum mismatch: {row["path"]}')
    required = {'manifest.json', cfg['grid'], 'height/completed/base_surface_z_m.npy',
                'height/completed/elevated_surface_z_m.npy'}
    for row in cfg['tiles'] + cfg['patches']:
        required.update(row['files'].values())
        required.update(row[k] for k in ['v9_result', 'v10_result'] if row.get(k))
    if not required <= seen:
        raise ValueError('Required input absent from checksum manifest')
    return {'status': 'passed', 'files_checked': len(seen), 'patches': len(cfg['patches']),
            'tiles': len(cfg['tiles']), 'network_calls': 0, 'lineage': cfg['lineage']}


def new_output(path, bundle):
    path, bundle = Path(path).resolve(), Path(bundle).resolve()
    if path == bundle or path.is_relative_to(bundle):
        raise ValueError('Output must be separate from the frozen bundle')
    path.mkdir(parents=True, exist_ok=False)
    return path
