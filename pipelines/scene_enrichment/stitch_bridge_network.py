#!/usr/bin/env python3
"""Join native-derived deck solids while retaining native UV/material provenance.

Only accepted prediction-network connections define height-compatible joins.
Original observations remain untouched. Output is a licensed local build, not
redistributable Epic mesh data. Install the scene-assembly optional dependency.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from bridge_assets import sha, swept_spans, validate_span_files, span_spline, write_glb, face_normals
from extract_elevated import Config, placement_proposal
from repair_elevated_network import connection_contract
from interchange import read_glb, accessor


def common_surfaces(proposal, connections, config):
    result = deepcopy(proposal)
    spans = {s['segment_id']: s for s in result['continuous_segments']}
    # Flat junction islands cover crossing centres. The solid union removes
    # remaining overlap surfaces. Nearby islands merge
    # only along a real accepted span, never merely because XY projections cross.
    selected = []
    for joint in connections['junctions']:
        arms = joint['arms']
        vectors = [np.asarray(a['tangent_outward_xyz'][:2]) for a in arms]
        angle = 180.
        if len(vectors) == 2:
            angle = np.degrees(np.arccos(np.clip(vectors[0]@vectors[1]/np.linalg.norm(vectors[0])/np.linalg.norm(vectors[1]), -1, 1)))
        if len(arms) >= 3 or angle < 160.:
            selected.append({'joint': joint, 'radius': max(a['width_m'] for a in arms)*.35+1.})
    groups = [{i} for i in range(len(selected))]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                connect = False
                for a in groups[i]:
                    for b in groups[j]:
                        aa, bb = selected[a], selected[b]
                        ids_a = {r['segment_id'] for r in aa['joint']['arms']}
                        ids_b = {r['segment_id'] for r in bb['joint']['arms']}
                        delta = np.asarray(aa['joint']['position_m'])-bb['joint']['position_m']
                        if ids_a & ids_b and np.linalg.norm(delta[:2]) < aa['radius']+bb['radius'] and abs(delta[2]) < 2.:
                            connect = True
                if connect:
                    groups[i] |= groups[j]; groups.pop(j); changed = True; break
            if changed:
                break
    islands = []
    for group in groups:
        members = [selected[i] for i in group]
        islands.append({'geometry': unary_union([Point(r['joint']['position_m'][:2]).buffer(r['radius']) for r in members]),
                        'z': float(np.median([r['joint']['position_m'][2] for r in members])),
                        'ids': set().union(*[{a['segment_id'] for a in r['joint']['arms']} for r in members]),
                        'junctions': [r['joint']['junction_id'] for r in members]})
    # Also include spans entirely between island nodes. Do not change unrelated
    # overpassing spans merely because they lie in the same plan-view island.
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix, vstack, hstack, eye
    span_list = list(spans.values())
    offset = {}; xyzs = {}; n = 0
    for span in span_list:
        identity = span['segment_id']; offset[identity] = n
        xyzs[identity] = np.asarray([p['position_m'] for p in span['profile']])
        n += len(span['profile'])
    original = np.concatenate([xyzs[s['segment_id']][:, 2] for s in span_list])
    targets = np.r_[original, [i['z'] for i in islands]]
    total = len(targets); eq = []; rhs = []; inequalities = []; upper = []
    def equal(entries, value=0.):
        eq.append(entries); rhs.append(value)
    for joint in connections['junctions']:
        indices = [offset[a['segment_id']]+(0 if a['end']=='start' else len(spans[a['segment_id']]['profile'])-1) for a in joint['arms']]
        for index in indices[1:]: equal({indices[0]: 1., index: -1.})
    connected = {(a['segment_id'], a['end']) for j in connections['junctions'] for a in j['arms']}
    for span in span_list:
        identity = span['segment_id']; start = offset[identity]; xyz = xyzs[identity]
        s = np.asarray([p['s_m'] for p in span['profile']])
        for k, ds in enumerate(np.diff(s)):
            inequalities.extend([{start+k: -1., start+k+1: 1.}, {start+k: 1., start+k+1: -1.}])
            upper.extend([.100001*ds, .100001*ds])
        for end, k in (('start', 0), ('end', len(s)-1)):
            if (identity, end) not in connected: equal({start+k: 1.}, xyz[k, 2])
        if span.get('ground_transition'):
            # The last two ground-contact rings stay at their measured heights.
            equal({start+len(s)-2: 1.}, xyz[-2, 2])
        for i, island in enumerate(islands):
            if identity not in island['ids']: continue
            for k, point in enumerate(xyz):
                if island['geometry'].distance(Point(point[:2])) < 1e-8:
                    equal({start+k: 1., n+i: -1.})
    def sparse(rows, columns):
        matrix = lil_matrix((len(rows), columns))
        for i, row in enumerate(rows):
            for j, value in row.items(): matrix[i, j] = value
        return matrix.tocsr()
    grade = sparse(inequalities, total*2)
    absolute = vstack([hstack([eye(total), -eye(total)]), hstack([-eye(total), -eye(total)])])
    solution = linprog(np.r_[np.zeros(total), np.ones(n), np.full(len(islands), 3.)],
                      A_ub=vstack([grade, absolute]), b_ub=np.r_[upper, targets, -targets],
                      A_eq=sparse(eq, total*2), b_eq=rhs,
                      bounds=[(z-2., z+2.) for z in targets]+[(0., None)]*total, method='highs')
    if not solution.success:
        raise ValueError('Shared junction/grade constraints infeasible: '+solution.message)
    rows = []
    for span in span_list:
        identity = span['segment_id']; xyz = xyzs[identity]
        fitted = solution.x[offset[identity]:offset[identity]+len(xyz)]
        for node, z, old in zip(span['profile'], fitted, xyz[:, 2]):
            node['pre_surface_z_m'] = float(old); node['position_m'][2] = float(z)
        rows.append({'span_id': identity, 'maximum_height_change_m': float(np.max(abs(fitted-xyz[:, 2])))})
    for i, island in enumerate(islands): island['z'] = float(solution.x[n+i])
    # Reconcile non-island ends after the fits, and use one common tangent for
    # two-way joins so both independently swept endpoint rings are identical.
    new_joints = connection_contract(list(spans.values()))
    for joint in new_joints:
        arms = joint['arms']
        if len(arms) != 2:
            continue
        a, b = arms
        va, vb = np.asarray(a['tangent_outward_xyz']), np.asarray(b['tangent_outward_xyz'])
        common = va-vb; common /= np.linalg.norm(common[:2])
        for arm, direction in ((a, common), (b, -common)):
            sign = 1. if arm['end'] == 'start' else -1.
            spans[arm['segment_id']][arm['end']+'_tangent_xyz'] = (direction*sign).tolist()
    for span in spans.values():
        if span.get('ground_transition'):
            delta = np.asarray(span['profile'][-1]['position_m'])-span['profile'][-2]['position_m']
            delta[2] = 0.; delta /= np.linalg.norm(delta[:2])
            span['end_tangent_xyz'] = delta.tolist()
    modules = placement_proposal(list(spans.values()), config)
    result['placements'] = modules['placements']
    result['placement_count'] = len(result['placements'])
    result['surface_policy'] = 'Flat shared junction islands, bounded grade profiles, common two-arm endpoint frames, exact native solid union'
    grade_rows = []
    for span in spans.values():
        s, spline = span_spline(span)
        derivative = spline(np.linspace(0., s[-1], max(100, int(np.ceil(s[-1]/.1)))), 1)
        grade = np.abs(derivative[:, 2])/np.maximum(np.linalg.norm(derivative[:, :2], axis=1), 1e-12)
        grade_rows.append({'span_id': span['segment_id'], 'maximum_swept_grade': float(grade.max())})
    if max(r['maximum_swept_grade'] for r in grade_rows) > .13:
        raise ValueError('Actual swept grade exceeds the 13 percent assembly limit')
    return result, {'discrete_profile_grade_limit': .100001, 'actual_swept_grade_limit': .13, 'swept_grades': grade_rows,
                    'junction_islands': [{'junctions': i['junctions'], 'z_m': i['z'], 'area_m2': i['geometry'].area} for i in islands],
                    'changed_profiles': rows, 'junctions': new_joints}


def decode_native(path, anchor):
    doc, binary, _ = read_glb(path)
    triangles, textures, slots = [], [], []
    for primitive in doc['meshes'][0]['primitives']:
        p, _ = accessor(doc, binary, primitive['attributes']['POSITION'])
        uv, _ = accessor(doc, binary, primitive['attributes']['TEXCOORD_0'])
        if 'indices' in primitive:
            index, _ = accessor(doc, binary, primitive['indices'])
            p, uv = p[index.ravel()], uv[index.ravel()]
        p = p.reshape(-1, 3, 3)[:, [0, 2, 1]][:, :, [0, 2, 1]]+anchor
        uv = uv.reshape(-1, 3, 2)[:, [0, 2, 1]]; uv[:, :, 1] = 1.-uv[:, :, 1]
        triangles.extend(p); textures.extend(uv); slots.extend([primitive['material']]*len(p))
    return np.asarray(triangles), np.asarray(textures), np.asarray(slots)


def native_solid(triangles, uv, slots, original_ids):
    import manifold3d as m
    order = np.argsort(slots, kind='stable')
    triangles, uv, slots = triangles[order], uv[order], slots[order]
    properties = np.concatenate((triangles.reshape(-1, 3), uv.reshape(-1, 2)), axis=1)
    # Explicit position welding keeps per-corner UV seams separate. Float GLB
    # values are exact duplicates at their shared rings, not epsilon guesses.
    _, first, inverse = np.unique(properties[:, :3], axis=0, return_index=True, return_inverse=True)
    to = first[inverse]; fr = np.flatnonzero(to != np.arange(len(to)))
    used, count = np.unique(slots, return_counts=True)
    mesh = m.Mesh64(properties, np.arange(len(properties), dtype=np.uint64).reshape(-1, 3),
                    merge_from_vert=fr.astype(np.uint64), merge_to_vert=to[fr].astype(np.uint64),
                    run_index=np.r_[0, np.cumsum(count)*3].astype(np.uint64),
                    run_original_id=np.asarray([original_ids[int(i)] for i in used], np.uint32))
    solid = m.Manifold(mesh)
    if solid.status() != m.Error.NoError:
        raise ValueError('Input native deck is not a valid closed solid: '+str(solid.status()))
    return solid


def solid_arrays(result, ids):
    mesh = result.to_mesh64(); prop = np.asarray(mesh.vert_properties); faces = np.asarray(mesh.tri_verts)
    slots = np.full(len(faces), -1, int)
    for a, b, identity in zip(mesh.run_index[:-1], mesh.run_index[1:], mesh.run_original_id):
        slots[int(a)//3:int(b)//3] = ids.index(int(identity))
    if np.min(slots) < 0:
        raise ValueError('Boolean output lost source material provenance')
    return prop[faces, :3], prop[faces, 3:5], slots


def union_arrays(items):
    """Boolean union with exact source material runs and interpolated native UV."""
    import manifold3d as m
    start = m.Manifold.reserve_ids(4); ids = list(range(start, start+4))
    solids = [native_solid(p, uv, slot, ids) for p, uv, slot in items]
    result = m.Manifold.batch_boolean(solids, m.OpType.Add)
    if result.status() != m.Error.NoError or result.is_empty():
        raise ValueError('Deck solid union failed: '+str(result.status()))
    # Intersections of almost-coplanar imported surfaces create microscopic
    # slivers. Collapse them before float32 GLB export; movement is <0.5 mm.
    result = result.simplify(.0005)
    return (*solid_arrays(result, ids), float(result.volume()))


def stitch(span_manifest, output):
    manifest = json.loads(span_manifest.read_text()); output.mkdir(parents=True, exist_ok=True)
    records = manifest['templates']
    arrays = [decode_native(span_manifest.parent/r['template_file'], np.asarray(r['position_m'])) for r in records]
    triangles, uv, slots, volume = union_arrays(arrays)
    anchor = (triangles.reshape(-1, 3).min(0)+triangles.reshape(-1, 3).max(0))*.5
    target = output/'PRED_ELEVATED_CONNECTED_NETWORK.glb'
    normal, area = face_normals(triangles)
    tiny = np.linalg.norm(normal, axis=1) < .5
    if tiny.any():
        # A boolean kernel can retain zero-area topological seam triangles.
        # They rasterize no pixels, but glTF still requires finite unit normals.
        # Copy the nearest actual face normal; do not remove topology edges.
        from scipy.spatial import cKDTree
        good = np.flatnonzero(~tiny)
        nearest = cKDTree(triangles[good].mean(1)).query(triangles[tiny].mean(1))[1]
        normal[tiny] = normal[good[nearest]]
    write_glb(target, triangles-anchor, uv, slots, np.repeat(normal[:, None, :], 3, axis=1))
    # Validate the actual quantized file as a solid as well as its render
    # attributes. Subpixel coplanar boolean slivers may collapse in float32;
    # the solid kernel removes these when it verifies the serialized mesh.
    import manifold3d as m
    first = m.Manifold.reserve_ids(4); ids = list(range(first, first+4))
    decoded, decoded_uv, decoded_slots = decode_native(target, np.zeros(3))
    serialized_solid = native_solid(decoded, decoded_uv, decoded_slots, ids)
    decoded_area = np.linalg.norm(np.cross(decoded[:, 1]-decoded[:, 0], decoded[:, 2]-decoded[:, 0]), axis=1)*.5
    document, binary, _ = read_glb(target)
    for primitive in document['meshes'][0]['primitives']:
        normal, _ = accessor(document, binary, primitive['attributes']['NORMAL'])
        if not np.isfinite(normal).all() or np.max(abs(np.linalg.norm(normal, axis=1)-1.)) > 2e-5:
            raise ValueError('Serialized road has invalid rendering normals')
    record = {'span_id': 'PRED_ELEVATED_CONNECTED_NETWORK', 'template_file': target.name, 'template_sha256': sha(target),
              'expected_native_bounds_m': [(triangles-anchor).reshape(-1, 3).min(0).tolist(), (triangles-anchor).reshape(-1, 3).max(0).tolist()],
              'position_m': anchor.tolist(), 'rotation_deg': [0., 0., 0.], 'scale': [1., 1., 1.],
              'logical_module_count': manifest['logical_module_count'], 'placement_source': 'predicted_elevated_structure',
              'inference': 'prediction_network_join_and_native_solid_union', 'prepared_world_geometry': True,
              'triangle_count': len(triangles), 'source_span_ids': [r['span_id'] for r in records]}
    out = {**manifest, 'templates': [record], 'actor_count': 1, 'source_span_manifest_sha256': sha(span_manifest),
           'code_sha256': sha(Path(__file__)), 'geometry_method': 'Native-section swept spans; flat common junction surfaces; manifold solid union removes internal caps and overlapping surfaces, preserving per-corner native UV and material provenance',
           'qa': {'boolean_status': 'PASS', 'volume_m3': volume, 'source_spans': len(records),
                  'serialized_manifold_valid': True, 'serialized_collapsed_boolean_slivers': int(np.sum(decoded_area == 0)), 'sliver_simplification_tolerance_m': .0005,
                  'zero_area_topological_face_normals_from_nearest_valid_face': int(tiny.sum()),
                  'material_triangle_counts': np.bincount(slots, minlength=4).tolist(), 'ue_import_executed': False}}
    (output/'span_manifest.json').write_text(json.dumps(out, indent=2)+'\n')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('proposal', 'connections', 'structure', 'template', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-obj',type=Path,help='Relocated local native OBJ; SHA256 must match --template, otherwise rebuild that template')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    proposal, report = common_surfaces(json.loads(a.proposal.read_text()), json.loads(a.connections.read_text()),
                                       Config(**json.loads(a.structure.read_text())['config']))
    report.update(schema='citycode.elevated_shared_surfaces.v1', uses_ground_truth_for_placement=False,
                  source_digests={name: sha(getattr(a, name)) for name in ('proposal', 'connections', 'structure', 'template')},
                  code_sha256=sha(Path(__file__)), solid_kernel='manifold3d==3.5.4', ue_import_executed=False)
    if a.source_obj is not None:
        report['source_digests']['source_obj']=sha(a.source_obj)
    path = a.output/'surface_placement_proposal.json'; path.write_text(json.dumps(proposal, indent=2)+'\n')
    lines = {'type': 'FeatureCollection', 'coordinate_system': proposal['coordinate_system'],
             'features': [{'type': 'Feature', 'id': span['segment_id'],
                           'properties': {key: value for key, value in span.items() if key != 'profile'},
                           'geometry': {'type': 'LineString', 'coordinates': [node['position_m'] for node in span['profile']]}}
                          for span in proposal['continuous_segments']]}
    (a.output/'surface_centerlines.geojson').write_text(json.dumps(lines, indent=2)+'\n')
    (a.output/'junction_surfaces.json').write_text(json.dumps(report, indent=2)+'\n')
    swept_spans(a.template, path, a.output/'source_spans',source_obj=a.source_obj)
    validate_span_files(a.output/'source_spans/span_manifest.json')
    out = stitch(a.output/'source_spans/span_manifest.json', a.output/'joined')
    print(json.dumps(out['qa']))


if __name__ == '__main__':
    main()
