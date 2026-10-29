"""Initialize new-source generation and adopt externally completed images without an API call."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil

from PIL import Image

from .appearance import completion_contract, sha256_file, write_json


def initialize_generation(config_path, input_manifest, output, *, project_root,
                          completed_images=None, completed_images_manifest=None,
                          lod_targets=(1000000, 250000, 100000)):
    from .batch import read
    from .lod import generation_lod_targets

    project, config_path, inputs, output = map(lambda p: Path(p).resolve(),
                                             (project_root, config_path, input_manifest, output))
    targets = generation_lod_targets({'lod_targets': list(lod_targets)})
    if completed_images and completed_images_manifest:
        raise ValueError('Choose a completed-image directory or a manifest, not both')
    if not inputs.is_relative_to(project) or not output.is_relative_to(project):
        raise ValueError('Place new structure inputs and output under this release package')
    if output.parent != inputs.parents[2] or output == inputs.parents[1]:
        raise ValueError('Output must be a new sibling directory of the structure package')
    if output.exists() and any(output.iterdir()):
        raise ValueError('New generation needs an empty output directory; never overwrite the provided scene')
    upstream = read(inputs)
    sources = upstream['records']
    identifiers = [row['building_id'] for row in sources]
    if (not identifiers or len(identifiers) != upstream['building_count']
            or len(set(identifiers)) != len(identifiers)
            or any(not re.fullmatch(r'[A-Za-z0-9_-]+', sid) for sid in identifiers)):
        raise ValueError('Source manifest needs unique, safe building IDs and a consistent count')
    buildings = {}
    for row in sources:
        path = (inputs.parent / row['path'] / 'building.json').resolve()
        if not path.is_relative_to(inputs.parent) or read(path).get('building_id') != row['building_id']:
            raise ValueError('Source building path or identity differs from the manifest')
        if row.get('status') != 'complete' or read(path).get('status') != 'complete':
            raise ValueError('Finish upstream structure and observation export before generation: ' + row['building_id'])
        buildings[row['building_id']] = path

    policies = config_path.with_name(config_path.stem + '.sources.json')
    image_index = config_path.with_name(config_path.stem + '.images.json')
    new_files = [config_path, policies] + ([image_index] if completed_images else [])
    if any(path.exists() for path in new_files):
        raise ValueError('Initialization files already exist; resume the existing config or choose another name')
    imported = []
    if completed_images:
        directory = Path(completed_images).resolve()
        if not directory.is_dir():
            raise ValueError('Completed-image directory does not exist')
        for sid, source in buildings.items():
            image = directory / (sid + '.png')
            if image.is_file():
                imported.append({'building_id': sid, 'building_sha256': sha256_file(source),
                    'image': os.path.relpath(image, image_index.parent), 'sha256': sha256_file(image)})
        if not imported:
            raise ValueError('No <building_id>.png files match the current source manifest')
    elif completed_images_manifest:
        if not Path(completed_images_manifest).is_file():
            raise ValueError('Completed-image manifest does not exist')

    config = {'schema': 'citycode.unified_generation_config.v1',
        'project_root': os.path.relpath(project, config_path.parent),
        'execution_mode': 'generate', 'historical_models_allowed': False,
        'input_manifest': os.path.relpath(inputs, project),
        'dispositions': os.path.relpath(policies, project),
        'provider_config': 'workflow/02_building_assets/appearance.standard.json',
        'runtime_config': 'workflow/02_building_assets/runtime.standard.json',
        'output': os.path.relpath(output, project), 'lod_targets': list(targets),
        'provider_maximum_parallel': 4, 'automatic_provider_retries': 0}
    if completed_images or completed_images_manifest:
        config['completed_images_manifest'] = os.path.relpath(
            image_index if completed_images else Path(completed_images_manifest).resolve(), project)
    policy = {'schema': 'citycode.source_dispositions.v1', 'input_manifest_sha256': sha256_file(inputs),
        'reference_assisted_qualification': False, 'shared_physical_groups': [], 'building_entity_groups': [],
        'default_surface_source_ids': [],
        'sources': [{'building_id': sid, 'action': 'generate',
            'reason': 'Candidate from current source; completed image and actual model still require review.',
            'structure_conditioning': 'source_footprint', 'validation_profile': 'source_reconstruction',
            'postprocess': {}} for sid in identifiers]}
    if completed_images:
        write_json(image_index, {'schema': 'citycode.external_completed_images.v1', 'records': imported})
    write_json(policies, policy)
    write_json(config_path, config)
    return {'status': 'new_generation_initialized', 'config': str(config_path),
        'source_count': len(identifiers), 'external_images_indexed': len(imported),
        'lod_targets': list(targets), 'next_phase': 'prepare', 'API_calls': 0, 'GPU_jobs': 0,
        'automatic_scene_approval': False}


def import_completed_images(config, plan):
    """Bind explicit user-supplied images to source bytes, without inventing an API receipt."""
    from .batch import read, verify

    root = Path(config['output'])
    project = Path(config['_project_root'])
    index = Path(config['completed_images_manifest'])
    manifest = read(index)
    if manifest.get('schema') != 'citycode.external_completed_images.v1':
        raise ValueError('Unsupported completed-image manifest schema')
    records = manifest['records']
    by_id = {row['building_id']: row for row in records}
    selected = {row['building_id']: row for row in plan['records']}
    if len(by_id) != len(records) or set(by_id) - set(selected):
        raise ValueError('External images contain duplicate or unselected source IDs')
    results = []
    for sid, row in by_id.items():
        planned = selected[sid]
        members = planned.get('source_ids', [sid])
        member_binding = None
        building_path = verify(root, planned['building'], project)
        building = read(building_path)
        building_sha256 = planned['building']['sha256']
        if 'source_buildings' in row:
            declared = row['source_buildings']
            if ('building_sha256' in row or len(members) < 2 or len(set(members)) != len(members)
                    or not isinstance(declared, list)
                    or any(not isinstance(item, dict) or not item.get('building_id') or not item.get('sha256')
                           for item in declared)):
                raise ValueError('External source-member binding requires one explicit shared group: ' + sid)
            declared_by_id = {item['building_id']: item['sha256'] for item in declared}
            originals = planned.get('source_buildings', [])
            originals_by_id = {item['building_id']: item for item in originals}
            if (len(declared_by_id) != len(declared) or set(declared_by_id) != set(members)
                    or len(originals_by_id) != len(originals) or set(originals_by_id) != set(members)
                    or building.get('source_group', {}).get('members') != members):
                raise ValueError('External shared image members differ from the planned group: ' + sid)
            for member in members:
                original = originals_by_id[member]
                if declared_by_id[member] != original['sha256']:
                    raise ValueError('External shared image belongs to different source bytes: ' + member)
                source_path = verify(root, original, project)
                if read(source_path).get('building_id') != member:
                    raise ValueError('Shared source file identity differs from the plan: ' + member)
            member_binding = [{'building_id': member, 'sha256': declared_by_id[member]} for member in members]
        elif row.get('building_sha256') != building_sha256:
            raise ValueError('External image belongs to different building bytes: ' + sid)
        if building.get('building_id') != sid:
            raise ValueError('External image belongs to a different planned building: ' + sid)
        image = (index.parent / row['image']).resolve()
        if sha256_file(image) != row['sha256']:
            raise ValueError('External completed image changed: ' + sid)
        with Image.open(image) as opened:
            opened.load()
            if opened.format != 'PNG' or opened.size != (1024, 1024):
                raise ValueError('External completed image must be a 1024x1024 PNG: ' + sid)
        directory = root / planned['appearance_directory']
        prepared = read(directory / 'appearance_input.json')
        if (prepared['building_id'] != sid
                or prepared['provenance']['building_sha256'] != building_sha256):
            raise ValueError('External image does not match the prepared source: ' + sid)
        receipt = {'schema': 'citycode.appearance_completion.v1', 'status': 'complete',
            'building_id': sid, 'origin': 'external_completed_image',
            'request_identity_sha256': completion_contract(prepared),
            'request_identity_scope': 'local_input_binding_only_no_provider_request_claimed',
            'output': 'generated_main_oblique.png', 'output_sha256': row['sha256'],
            'external_manifest_sha256': sha256_file(index), 'source_building_sha256': building_sha256,
            'API_calls': 0, 'image_review_required': True}
        if member_binding is not None:
            receipt.update(source_binding_scope='shared_source_members', source_buildings=member_binding)
        target, saved = directory / receipt['output'], directory / 'completion.json'
        if saved.exists() or target.exists():
            if not (saved.is_file() and target.is_file() and read(saved) == receipt
                    and sha256_file(target) == row['sha256']):
                raise ValueError('Existing completed image differs; choose a new run: ' + sid)
        else:
            shutil.copyfile(image, target)
            write_json(saved, receipt)
        results.append({'asset_id': planned['asset_id'], 'image_sha256': row['sha256']})
    write_json(root / 'IMPORTED_IMAGES.json', {'status': 'complete', 'records': results,
        'manifest_sha256': sha256_file(index), 'API_calls': 0, 'automatic_approval': False})
    return results
