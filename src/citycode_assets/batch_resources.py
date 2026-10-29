"""Portable identities for the exact code, checkpoints and offline GPU resources."""
from __future__ import annotations

from pathlib import Path

from .appearance import sha256_file


def resource_contract(config, *, full_hash=False, check_sizes=True):
    from .batch import digest, proof, read, verify
    project = Path(config['_project_root']).resolve()
    runtime_path = Path(config['runtime_config'])
    runtime = read(runtime_path)
    manifests = [project/runtime['model_manifest'],
                 project/runtime['base_resources_manifest'],
                 project/runtime['gpu_wheels_manifest']]
    if runtime.get('offline_wheels_manifest'):
        manifests.append(project/runtime['offline_wheels_manifest'])
    declarations = {}
    for manifest in manifests:
        value = read(manifest)
        items = value.get('files', [])
        for model in value.get('models', []):
            items += [model, model['architecture']]
        for item in items:
            record = {key:item[key] for key in ['path', 'bytes', 'sha256']}
            relative = Path(record['path'])
            path = (project/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(project):
                raise ValueError('Runtime resource must remain inside the public package')
            if relative.as_posix() in declarations and declarations[relative.as_posix()] != record:
                raise ValueError('Conflicting runtime resource identities')
            declarations[relative.as_posix()] = record
    records = [declarations[key] for key in sorted(declarations)]
    for item in records:
        if full_hash:
            verify(project, item)
        elif check_sizes:
            path = project/item['path']
            if not path.is_file() or path.stat().st_size != item['bytes']:
                raise ValueError('Runtime resource missing or size changed: '+item['path'])
    # Include the CPU repairs and export code as well as the inference adapter.
    # Documentation edits do not invalidate an already frozen computation.
    code = sorted(p for p in Path(__file__).parent.rglob('*') if p.is_file()
                  and p.suffix in {'.py', '.cpp', '.cjs', '.js', '.h'})
    identity = {'runtime_config':proof(runtime_path, project),
                'pipeline_config':proof(project/runtime['pipeline_config'], project),
                'resource_manifests':[proof(p, project) for p in manifests],
                'execution_code':[{**proof(p, Path(__file__).parent),
                    'path':'citycode_assets/'+p.relative_to(Path(__file__).parent).as_posix()} for p in code],
                'resources':records}
    return {**identity, 'identity_sha256':digest(identity)}


def verify_runtime_contract(config, frozen, *, full_hash=False, check_sizes=True):
    current = resource_contract(config, full_hash=full_hash, check_sizes=check_sizes)
    if current != frozen['runtime_resources']:
        raise ValueError('Frozen runtime, resource declarations or execution code changed')
    return current
