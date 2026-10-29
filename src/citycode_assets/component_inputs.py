"""Prepare optional component reruns from registered assets and saved RGB inputs.

The coarse geometry is explicitly asset-derived. It is not a new extraction,
an observed component segmentation, or a reference-building reconstruction.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
import trimesh

from .appearance import sha256_file
from .batch import proof, read, verify
from .registration import raster_triangle_footprint


def component_world_mesh(scene, container_id, world_from_asset):
    """Extract all actual geometry below one named container, preserving node poses."""
    if container_id not in scene.graph.nodes:
        raise ValueError('Component container is missing: ' + container_id)
    matrix = np.asarray(world_from_asset, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])):
        raise ValueError('Component extraction needs a finite affine world matrix')
    parents = scene.graph.transforms.parents
    vertices, faces, selected = [], [], []
    offset = 0
    for node in sorted(scene.graph.nodes_geometry):
        ancestor, visited = node, set()
        while ancestor != container_id:
            if ancestor in visited or ancestor not in parents:
                ancestor = None
                break
            visited.add(ancestor)
            ancestor = parents[ancestor]
        if ancestor is None:
            continue
        transform, geometry_id = scene.graph[node]
        mesh = scene.geometry[geometry_id]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('Component contains unsupported or empty geometry: ' + node)
        world = matrix @ np.asarray(transform, dtype=float)
        actual = np.asarray(mesh.vertices) @ world[:3, :3].T + world[:3, 3]
        vertices.append(actual)
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(actual)
        selected.append(node)
    if not vertices:
        raise ValueError('Component container has no mesh descendants: ' + container_id)
    return np.concatenate(vertices), np.concatenate(faces), selected


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def _write_verified(bundle, path, content):
    path = Path(path)
    if not path.resolve().is_relative_to(bundle):
        raise ValueError('Derived component output escaped the selected package')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing component input differs; choose a new output: ' + str(path))
    else:
        path.write_bytes(content)


def _read_project_file(project, value):
    path = Path(value).resolve()
    if not path.is_relative_to(project) or not path.is_file():
        raise ValueError('Component inputs must remain inside the selected project: ' + str(path))
    return path


def _polygonal(geometry):
    if geometry.geom_type in {'Polygon', 'MultiPolygon'}:
        return geometry
    return unary_union([part for part in getattr(geometry, 'geoms', [])
                        if part.geom_type in {'Polygon', 'MultiPolygon'} and part.area > 0])


def build_component_inputs(project, bundle, baseline_manifest, inventory, *, reference_records=None):
    """Build portable rerun inputs; never generate a model or approve a new mesh.

    ``reference_records`` may supply audited saved reference PNGs and camera
    metadata. Without it the current parent observation is used and explicitly
    labelled as parent context. All files are copied unchanged by checksum.
    """
    from city_pipeline.buildings.observations import extrude_geometry, UE_Z_UP_TO_GLTF_Y_UP

    project, bundle = Path(project).resolve(), Path(bundle).resolve()
    baseline_manifest = _read_project_file(project, baseline_manifest)
    if not bundle.is_relative_to(project/'data'):
        raise ValueError('Place the component input package inside the release data directory')
    inventory = read(inventory) if isinstance(inventory, (str, Path)) else inventory
    entries = [entry for entry in inventory['items'] if entry.get('route') == 'component']
    if not entries:
        raise ValueError('No component generation units were selected')
    keys = [(entry['asset_id'], entry['old_asset_id']) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('Component generation units must be unique')
    for pair in keys:
        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', item) for item in pair):
            raise ValueError('Unsafe component or asset identifier')
    references = {}
    for row in reference_records or []:
        key = (row['current_asset_id'], row['old_asset_id'])
        if key in references:
            raise ValueError('Duplicate saved component reference')
        references[key] = row
    if reference_records is not None and set(keys) - set(references):
        raise ValueError('Saved component references do not cover all generation units')
    baseline = read(baseline_manifest)
    assets = {row['id']: row for row in baseline['asset_instances']}
    sources = {row['building_id']: row for row in baseline['buildings']}
    dataset = bundle/'building_generation_inputs'
    records, images, groups = [], [], []
    for aid in sorted({entry['asset_id'] for entry in entries}):
        if aid not in assets or assets[aid]['status'] != 'accepted_composite':
            raise ValueError('Component task does not name a current composite asset: ' + aid)
        asset = assets[aid]
        selected = sorted([entry for entry in entries if entry['asset_id'] == aid], key=lambda row: row['old_asset_id'])
        source_ids = asset['source_building_ids']
        if any(entry['current_source_ids'] != source_ids for entry in selected):
            raise ValueError('Component source membership differs from the current asset')
        lod = verify(baseline_manifest.parent, asset['lods']['0.1M'], project)
        scene = trimesh.load_scene(lod, process=False)
        source_paths = [
            _read_project_file(project, baseline_manifest.parent/sources[sid]['structure_path']) for sid in source_ids]
        source_buildings = [read(path) for path in source_paths]
        if any(building['building_id'] != sid for building, sid in zip(source_buildings, source_ids)):
            raise ValueError('Current component source file identity differs')
        parent = source_buildings[0]
        support = unary_union([shape(building['footprint_world']) for building in source_buildings])
        if not support.is_valid or support.is_empty:
            raise ValueError('Current component source footprint must be valid')
        group = {'asset_id': aid, 'current_source_ids': source_ids,
                 'source_model': proof(lod, bundle),
                 'source_buildings': [{'building_id': sid, **proof(path, bundle)}
                                      for sid, path in zip(source_ids, source_paths)],
                 'source_world_from_local_4x4': asset['world_from_local_4x4'],
                 'task_ids': [], 'components': [],
                 'new_generated_models_require_review': True,
                 'previous_mesh_and_scene_approval_reusable': False}
        for index, entry in enumerate(selected):
            old_id = entry['old_asset_id']
            task_id = f'{aid}__C{index:02d}'
            completed = _read_project_file(project, bundle/'units'/old_id/'completed.png')
            if sha256_file(completed) != entry['image_sha256']:
                raise ValueError('Completed component image differs from the audited input: ' + old_id)
            with Image.open(completed) as image:
                if image.format != 'PNG' or image.size != (1024, 1024):
                    raise ValueError('Completed component image must be a 1024 PNG: ' + old_id)
            vertices, faces, nodes = component_world_mesh(scene, old_id, asset['world_from_local_4x4'])
            projected, projection = raster_triangle_footprint(vertices, faces, np.eye(4), cell_m=.10, max_dimension=4096)
            footprint = _polygonal(projected.intersection(support))
            base, top = float(vertices[:, 2].min()), float(vertices[:, 2].max())
            if (not footprint.is_valid or footprint.is_empty or footprint.area <= 0
                    or not np.isfinite([base, top]).all() or top <= base):
                raise ValueError('Component has no valid positive supported volume: ' + old_id)
            cx, cy = map(float, footprint.centroid.coords[0])
            local = affinity.translate(footprint, xoff=-cx, yoff=-cy)
            world_from_local = np.eye(4)
            world_from_local[:3, 3] = [cx, cy, base]
            folder = dataset/'buildings'/task_id
            for name, geometry in [('world', footprint), ('local', local)]:
                feature = {'type': 'FeatureCollection', 'features': [{'type': 'Feature',
                    'geometry': mapping(geometry), 'properties': {'building_id': task_id,
                    'height_m': top-base, 'base_z_m': base if name == 'world' else 0.}}]}
                _write_verified(bundle, folder/'geometry'/f'footprints_{name}.geojson', _json_bytes(feature))
            proxy = extrude_geometry(local, 0., top-base, [180, 190, 200, 255], {'building_id': task_id})
            for name, transform in [('proxy_local_ue_z_up.glb', np.eye(4)),
                                    ('proxy_local.glb', UE_Z_UP_TO_GLTF_Y_UP)]:
                mesh = proxy.copy()
                mesh.apply_transform(transform)
                proxy_scene = trimesh.Scene(base_frame='world')
                proxy_scene.add_geometry(mesh, node_name='proxy', geom_name='proxy')
                _write_verified(bundle, folder/'geometry'/name, proxy_scene.export(file_type='glb'))
            reference = references.get((aid, old_id))
            canonical = task_id+'__reference'
            obs_folder = folder/'observations'/canonical
            if reference is not None:
                observation = {'schema': 'citycode.building_observed_evidence.v1', 'building_id': task_id,
                    'observation_id': canonical, 'original_building_id': old_id,
                    'original_observation_id': reference.get('observation_id'),
                    'evidence_scope': 'saved_reference_input', 'is_canonical': True,
                    'source_camera_crop': copy.deepcopy(reference.get('source_camera_crop')),
                    'paths': {key: key+'.png' for key in ['source_primary', 'roof_evidence', 'context']}}
                reference_files = reference['reference_inputs']
            else:
                previous = next(row for row in parent['observations']
                                if row['observation_id'] == parent['canonical_observation_id'])
                parent_obs = _read_project_file(project, source_paths[0].parent/previous['path']/'observation.json')
                observation = read(parent_obs)
                original_paths = observation['paths']
                reference_files = {key+'.png': {'path': str(parent_obs.parent/original_paths[key])}
                                   for key in ['source_primary', 'roof_evidence', 'context']}
                observation.update(building_id=task_id, observation_id=canonical,
                    original_building_id=parent['building_id'], original_observation_id=parent['canonical_observation_id'],
                    evidence_scope='parent_source_context', is_canonical=True,
                    paths={key: key+'.png' for key in ['source_primary', 'roof_evidence', 'context']})
            observation['component_segmentation_observed'] = False
            observation['appearance_and_geometry_require_review'] = True
            reference_proofs = {}
            for key in ['source_primary', 'roof_evidence', 'context']:
                record = reference_files[key+'.png']
                packaged = bundle/'units'/old_id/'reference_inputs'/(key+'.png')
                source = _read_project_file(project, packaged if packaged.is_file() else record['path'])
                if record.get('sha256') and sha256_file(source) != record['sha256']:
                    raise ValueError('Saved component reference changed: ' + old_id)
                with Image.open(source) as image:
                    image.load()
                    if key != 'context' and image.mode != 'RGBA':
                        raise ValueError('Component appearance evidence must preserve source RGBA')
                target = obs_folder/(key+'.png')
                _write_verified(bundle, target, source.read_bytes())
                reference_proofs[key] = proof(target, bundle)
            _write_verified(bundle, obs_folder/'observation.json', _json_bytes(observation))
            prior = {'kind': 'asset_derived_component_prior', 'new_observation_extraction': False,
                'reference_geometry_used': False, 'source_model': proof(lod, bundle),
                'source_container_node': old_id, 'source_geometry_nodes': nodes,
                'source_world_from_local_4x4': asset['world_from_local_4x4'],
                'current_source_ids': source_ids, 'projection': projection,
                'projected_area_m2': float(projected.area), 'supported_area_m2': float(footprint.area),
                'world_z': [base, top], 'actual_component_triangles': len(faces),
                'old_roof_layers_reused': False, 'new_model_review_required': True}
            building = {'schema': 'citycode.building_generation_input.v1', 'status': 'complete',
                'building_id': task_id, 'canonical_observation_id': canonical,
                'footprint_world': mapping(footprint), 'base_z_m': base, 'top_z_m': top, 'height_m': top-base,
                'world_from_local_4x4': world_from_local.tolist(),
                'local_from_world_4x4': np.linalg.inv(world_from_local).tolist(),
                'coordinate_system': 'UE XYZ metres Z-up; local origin at supported footprint centroid and asset-derived base',
                'roof_layers': [], 'observations': [{'observation_id': canonical, 'is_canonical': True,
                    'roles': ['canonical', 'saved_reference_input' if reference else 'parent_source_context'],
                    'path': 'observations/'+canonical}],
                'paths': {'footprints_world': 'geometry/footprints_world.geojson',
                    'footprints_local': 'geometry/footprints_local.geojson',
                    'proxy_local_ue_z_up': 'geometry/proxy_local_ue_z_up.glb',
                    'proxy_local_gltf_y_up': 'geometry/proxy_local.glb'}, 'derived_prior': prior}
            building_path = folder/'building.json'
            _write_verified(bundle, building_path, _json_bytes(building))
            records.append({'building_id': task_id, 'path': 'buildings/'+task_id, 'status': 'complete'})
            images.append({'building_id': task_id, 'building_sha256': sha256_file(building_path),
                           'image': completed.relative_to(bundle).as_posix(), 'sha256': entry['image_sha256']})
            group['task_ids'].append(task_id)
            group['components'].append({'task_id': task_id, 'old_asset_id': old_id,
                'building': proof(building_path, bundle), 'completed_image': proof(completed, bundle),
                'reference_inputs': reference_proofs, 'derived_prior': prior})
        groups.append(group)
    manifest = {'schema': 'citycode.building_generation_inputs.v1', 'status': 'complete',
        'building_count': len(records), 'records': records, 'pending': [],
        'scope': 'Optional component reruns from asset-derived priors; these are not new source detections.',
        'source_binding_manifest': proof(baseline_manifest, bundle), 'reference_geometry_used': False}
    image_index = {'schema': 'citycode.external_completed_images.v1', 'records': images}
    group_index = {'schema': 'citycode.component_regeneration_groups.v1', 'groups': groups,
        'component_count': len(records), 'group_count': len(groups),
        'baseline_binding': proof(baseline_manifest, bundle),
        'assembly_policy': 'Use newly generated, reviewed component bytes and their actual fitted world poses; do not inherit the previous mesh or scene approval.',
        'automatic_assembly_approval': False}
    _write_verified(bundle, dataset/'manifest.json', _json_bytes(manifest))
    _write_verified(bundle, bundle/'components.images.json', _json_bytes(image_index))
    _write_verified(bundle, bundle/'component_groups.json', _json_bytes(group_index))
    return {'status': 'complete_component_rerun_inputs', 'component_count': len(records),
            'group_count': len(groups), 'manifest': str(dataset/'manifest.json'),
            'image_index': str(bundle/'components.images.json'), 'API_calls': 0, 'GPU_jobs': 0,
            'automatic_model_or_scene_approval': False}
