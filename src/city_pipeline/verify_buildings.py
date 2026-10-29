#!/usr/bin/env python3
"""Verify completed building geometry and exported meshes without using GT."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_paths(manifest, release):
    """Keep legacy absolute hashes; resolve declared relative hashes portably."""
    result = {}
    for name, digest in manifest['source_sha256'].items():
        path = Path(name)
        if not path.is_absolute():
            require(manifest.get('source_sha256_path_resolution') == 'manifest_directory',
                    'Relative source hash requires manifest_directory declaration')
            path = release / path
        result[path.resolve()] = digest
    return result


def check_source_hashes(paths):
    for path, digest in paths.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f'Export source SHA changed: {path}')


def world_mesh(scene, node):
    require(node in scene.graph.nodes_geometry, f'Missing mesh node: {node}')
    transform, geometry_name = scene.graph[node]
    require(np.isfinite(transform).all(), f'Nonfinite transform: {node}')
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    return mesh


def cap_audit(mesh, geometry, bottom, top):
    """Check both caps with explicit GLB float32 coordinate/perimeter bounds.

    In-memory extrusion checks are tighter. Components and holes with a core
    wider than coordinate quantization must retain coverage or void. Smaller
    details remain covered by the cap area bound and are reported explicitly.
    """
    require(np.isfinite([bottom, top]).all() and top > bottom,
            'Mesh piece must have finite positive height')
    require(geometry.geom_type in ('Polygon', 'MultiPolygon') and
            geometry.is_valid and not geometry.is_empty and geometry.area > 0,
            'Mesh reference must be a valid nonempty polygon')
    require(len(mesh.vertices) > 0 and np.isfinite(mesh.vertices).all(),
            'Mesh vertices must be finite and nonempty')
    xy_scale = max(1., float(np.abs(mesh.vertices[:, :2]).max()))
    xy_epsilon = max(1e-4, 2 * np.finfo(np.float32).eps * xy_scale)
    z_epsilon = max(1e-4, 2 * np.finfo(np.float32).eps * max(abs(bottom), abs(top), 1.))
    area_tolerance = max(.01, geometry.length * xy_epsilon * 2)
    triangles = np.asarray(mesh.triangles)
    components = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    holes = [Polygon(ring) for component in components for ring in component.interiors]
    # This erosion only defines observable detail at the exported precision;
    # the reference geometry used in both cap comparisons remains unchanged.
    component_cores = [p.buffer(-xy_epsilon) for p in components]
    hole_cores = [h.buffer(-xy_epsilon) for h in holes]
    protected_components = [p for p in component_cores if not p.is_empty]
    protected_holes = [h for h in hole_cores if not h.is_empty]
    caps = {}
    for label, elevation in [('bottom', bottom), ('top', top)]:
        selected = np.all(np.abs(triangles[:, :, 2] - elevation) <= z_epsilon, axis=1)
        other_elevation = top if label == 'bottom' else bottom
        selected &= (np.abs(triangles[:, :, 2] - elevation).mean(axis=1)
                     <= np.abs(triangles[:, :, 2] - other_elevation).mean(axis=1))
        # Exclude vertical walls even for pieces thinner than the Z tolerance.
        selected &= np.ptp(triangles[:, :, 2], axis=1) == 0
        polygons = [Polygon(t[:, :2]) for t in triangles[selected]]
        polygons = [polygon for polygon in polygons if polygon.is_valid and polygon.area > 0]
        cap = unary_union(polygons)
        delta = float(geometry.symmetric_difference(cap).area)
        triangle_error = abs(sum(p.area for p in polygons) - geometry.area)
        caps[label] = {
            'symmetric_difference_m2': delta,
            'triangle_area_error_m2': float(triangle_error),
            'missing_area_per_component_m2': [float(p.difference(cap).area) for p in components],
            'filled_area_per_hole_m2': [float(h.intersection(cap).area) for h in holes],
            'matches_geometry': bool(delta <= area_tolerance and triangle_error <= area_tolerance
                and all(p.intersection(cap).area > 0 for p in protected_components)
                and all(h.difference(cap).area > 0 for h in protected_holes)),
        }
    expected_volume = float(geometry.area * (top - bottom))
    volume_tolerance = area_tolerance * (top - bottom) + geometry.area * 2 * z_epsilon
    checks = {
        'watertight': bool(mesh.is_watertight),
        'consistent_winding': bool(mesh.is_winding_consistent),
        'positive_volume': bool(np.isfinite(mesh.volume) and mesh.volume > 0),
        'volume_matches_geometry': bool(abs(mesh.volume - expected_volume) <= volume_tolerance),
        'bottom_elevation_matches': bool(abs(float(mesh.bounds[0, 2]) - bottom) <= z_epsilon),
        'top_elevation_matches': bool(abs(float(mesh.bounds[1, 2]) - top) <= z_epsilon),
        'bottom_cap_matches_geometry': caps['bottom']['matches_geometry'],
        'top_cap_matches_geometry': caps['top']['matches_geometry'],
    }
    return {
        'pass': all(checks.values()), 'checks': checks, 'caps': caps,
        'cap_area_tolerance_m2': float(area_tolerance),
        'xy_float32_tolerance_m': float(xy_epsilon),
        'z_float32_tolerance_m': float(z_epsilon),
        'volume_m3': float(mesh.volume), 'expected_volume_m3': expected_volume,
        'volume_tolerance_m3': float(volume_tolerance),
        'reference_component_count': len(components), 'reference_hole_count': len(holes),
        'components_below_xy_resolution': len(components) - len(protected_components),
        'holes_below_xy_resolution': len(holes) - len(protected_holes),
    }


def verify_run(run, reference=None):
    run = Path(run)
    release = run / 'release'
    require(read_json(run / 'RUN_STATUS.json')['status'] == 'complete', 'Incomplete run')
    manifest = read_json(release / 'manifest.json')
    require(manifest['status'] == 'complete', 'Incomplete export')
    sources = source_paths(manifest, release)
    check_source_hashes(sources)
    features = read_json(release / 'global_building_footprints.geojson')['features']
    scene = trimesh.load(release / 'full_city_layered_ue_z_up.glb', force='scene', process=False)
    geometries, names, audits, ids = [], [], [], []
    roof_count = 0
    for feature in features:
        p = feature['properties']
        geometry = shape(feature['geometry'])
        gid = p['predicted_global_building_id']
        ids.append(gid)
        require(geometry.geom_type in ('Polygon', 'MultiPolygon') and geometry.is_valid and
                not geometry.is_empty and geometry.area > 0, f'Invalid footprint: {gid}')
        geometries.append(geometry)
        require(np.isfinite([p['height_m'], p['base_z_m'], p['top_z_m']]).all() and
                abs(p['height_m'] - (p['top_z_m'] - p['base_z_m'])) < 1e-6,
                f'Inconsistent height: {gid}')
        body_top = float(p.get('body_top_z_m', p['top_z_m']))
        listed = p['structure_mesh_nodes']
        roofs = p.get('roof_layers', [])
        require(len(listed) == 1 + len(roofs) and listed[0] == gid + '__body',
                f'Mesh-node/roof count mismatch: {gid}')
        pieces = [(listed[0], geometry, float(p['base_z_m']), body_top)]
        seen = []
        for layer, node in zip(roofs, listed[1:]):
            roof = shape(layer['geometry'])
            require(roof.difference(geometry).area < 1e-5 and
                    all(roof.intersection(other).area < 1e-5 for other in seen),
                    f'Roof outside footprint or overlapping another roof: {node}')
            seen.append(roof)
            pieces.append((node, roof, float(layer.get('base_z_m', body_top)), float(layer['top_z_m'])))
            roof_count += 1
        require(abs(max(top for _, _, _, top in pieces) - p['top_z_m']) < 1e-6,
                f'Top height differs from structure: {gid}')
        for node, footprint, bottom, top in pieces:
            names.append(node)
            audit = cap_audit(world_mesh(scene, node), footprint, bottom, top)
            require(audit['pass'], f'Mesh geometry mismatch: {node}: {audit["checks"]}')
            audits.append({'node': node, **audit})
        if manifest.get('mesh_path_resolution') == 'manifest_directory':
            for key in ('canonical_glb', 'canonical_glb_ue_z_up', 'coarse_glb', 'coarse_glb_ue_z_up'):
                path = Path(p[key])
                require(not path.is_absolute() and (release / path).is_file(),
                        f'Missing portable mesh artifact: {key}: {path}')
    require(len(ids) == len(set(ids)), 'Duplicate building identities')
    require(len(names) == len(set(names)) and set(names) == set(scene.graph.nodes_geometry),
            'Duplicate or unreferenced mesh nodes')
    require({scene.graph[node][1] for node in names} == set(scene.geometry), 'Unreferenced mesh geometry')
    tree = STRtree(geometries)
    severe = 0
    for i, geometry in enumerate(geometries):
        for j in tree.query(geometry, predicate='intersects'):
            if j > i:
                area = geometry.intersection(geometries[j]).area
                severe += int(area >= 20 and area / min(geometry.area, geometries[j].area) >= .5)
    require(severe == 0, 'Severe footprint overlap')
    result = {
        'status': 'PASS', 'buildings': len(features), 'roof_pieces': roof_count,
        'closed_meshes': len(names), 'severe_overlap_pairs': severe,
        'geometry_height_mesh_verified': True, 'mesh_cap_holes_components_verified': True,
        'mesh_comparison_precision': 'GLB float32; per-piece coordinate/perimeter error bounds',
        'mesh_piece_results': audits,
    }
    if reference:
        old_features = read_json(reference)['features']
        old = {f['properties']['predicted_global_building_id']: f for f in old_features}
        require(len(old) == len(old_features) and set(old) == set(ids), 'Reference identities differ')
        max_delta = 0.
        for feature in features:
            p = feature['properties']
            previous = old[p['predicted_global_building_id']]
            delta = shape(feature['geometry']).symmetric_difference(shape(previous['geometry'])).area
            max_delta = max(max_delta, delta)
            require(delta < 1e-5 and abs(p['top_z_m'] - previous['properties']['top_z_m']) < 1e-6,
                    'Reference footprint or top height differs')
            roofs, old_roofs = p.get('roof_layers', []), previous['properties'].get('roof_layers', [])
            require(len(roofs) == len(old_roofs), 'Reference roof count differs')
            for layer, old_layer in zip(roofs, old_roofs):
                require(shape(layer['geometry']).symmetric_difference(shape(old_layer['geometry'])).area < 1e-5
                        and abs(layer['top_z_m'] - old_layer['top_z_m']) < 1e-6, 'Reference roof differs')
        result['historical_v9_geometry_and_heights_match'] = True
        result['maximum_footprint_symmetric_difference_m2'] = max_delta
    check_source_hashes(sources)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    result = verify_run(args.run, args.reference)
    (args.run / 'VALIDATION.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print({key: value for key, value in result.items() if key != 'mesh_piece_results'})


if __name__ == '__main__':
    main()
