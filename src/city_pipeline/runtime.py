"""Data paths are caller-owned; code never searches the parent research checkout."""
from pathlib import Path
import hashlib
import json
import os

DATA_ROOT = Path(os.environ.get('CITY_PIPELINE_DATA_ROOT', '.')).resolve()
PACKAGE_ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(value, root=DATA_ROOT):
    return (Path(root) / value).resolve()
