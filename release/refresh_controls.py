#!/usr/bin/env python3
"""Review/refresh release control proofs; never accept changed data or weights.

Default is read-only. --write explicitly updates proofs for reviewed code,
configuration and model-resource metadata, while all binary/data proofs remain
strict. This is a maintainer operation, never a runtime fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'data/smallcity_p384_reused_generation_inputs/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def refresh(root, *, write=False):
    root = Path(root).resolve()
    path = root / MANIFEST
    doc = json.loads(path.read_text())
    changes = []
    for row in doc['required_files']:
        file = (path.parent / row['path']).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError('Missing/outside package: ' + row['path'])
        rel = file.relative_to(root)
        # Frozen inputs and model binaries are deliberately outside this allowlist.
        if rel.parts[0] not in {'workflow', 'src', 'pipelines', 'environment', 'models'}:
            continue
        if file.suffix not in {'.py', '.json', '.sh', '.ps1', '.txt', '.toml'}:
            continue
        actual = {'bytes': file.stat().st_size, 'sha256': digest(file)}
        if any(row[key] != value for key, value in actual.items()):
            changes.append({'path': rel.as_posix(),
                            'previous': {key: row[key] for key in actual}, 'current': actual})
            if write:
                row.update(actual)
    if write and changes:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    return {'status': 'updated' if write else 'review_only', 'changes': changes,
            'data_or_weight_proofs_modified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    print(json.dumps(refresh(args.root, write=args.write), ensure_ascii=False, indent=2))
