#!/usr/bin/env python3
"""Extrude the RGB map's published land extent for local viewing, on CPU.

Uses the same packaged land polygon as the coastal 3D background. The RGB
image remains unchanged: land is a flat textured cap with a closed base and
shore walls, and water is a separate textured plane outside that polygon.
Coordinates stay UE XYZ metres, Z up; the viewer must not swap Y and Z.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import shapely
from shapely.geometry import Polygon, box, shape
import trimesh

ROOT = Path(__file__).resolve().parents[1]
WALL_COLOR = [93, 88, 76, 255]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def polygons(geometry):
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [polygon for child in getattr(geometry, 'geoms', []) for polygon in polygons(child)]


def rgb_uv(vertices, bounds):
    """Trimesh UVs: export flips V, so actual glTF V follows image rows/+Y."""
    xmin, ymin, xmax, ymax = bounds
    vertices = np.asarray(vertices)
    return np.column_stack(((vertices[:, 0] - xmin) / (xmax - xmin),
                            1 - (vertices[:, 1] - ymin) / (ymax - ymin)))


def textured(mesh, image, bounds, name):
    material = trimesh.visual.material.PBRMaterial(
        name=name, baseColorTexture=image, baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0, roughnessFactor=1, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(uv=rgb_uv(mesh.vertices, bounds), material=material)
    return mesh


def planar_mesh(geometry, z):
    meshes = []
    for polygon in polygons(geometry):
        vertices, faces = trimesh.creation.triangulate_polygon(polygon, engine='earcut')
        mesh = trimesh.Trimesh(vertices=np.column_stack((vertices, np.full(len(vertices), z))),
                               faces=faces, process=False)
        downward = mesh.face_normals[:, 2] < 0
        mesh.faces[downward] = mesh.faces[downward, ::-1]
        meshes.append(mesh)
    if not meshes:
        raise ValueError('Expected positive-area water outside land')
    return trimesh.util.concatenate(meshes)


def footprint(mesh, upward_only=False):
    faces = mesh.faces[mesh.face_normals[:, 2] > .999] if upward_only else mesh.faces
    triangles = np.asarray(mesh.vertices)[faces, :2]
    return shapely.union_all(shapely.polygons(triangles))


def preserve_rgb_bytes(path, map_path):
    """Replace exporter JPEG recompression with the packaged image's exact bytes."""
    raw = Path(path).read_bytes()
    json_size = struct.unpack_from('<I', raw, 12)[0]
    doc = json.loads(raw[20:20 + json_size])
    binary_start = 20 + json_size + 8
    original_binary = memoryview(raw)[binary_start:]
    image_bytes = Path(map_path).read_bytes()
    with Image.open(map_path) as image:
        mime = Image.MIME[image.format]
    if mime not in {'image/png', 'image/jpeg'}:
        raise ValueError('RGB map must use a native glTF PNG or JPEG image')
    image_views = {image['bufferView'] for image in doc['images']}
    binary = bytearray()
    for index, view in enumerate(doc['bufferViews']):
        offset, size = view.get('byteOffset', 0), view['byteLength']
        contents = image_bytes if index in image_views else original_binary[offset:offset + size]
        binary.extend(b'\x00' * (-len(binary) % 4))
        view['byteOffset'] = len(binary)
        view['byteLength'] = len(contents)
        binary.extend(contents)
    for image in doc['images']:
        image['mimeType'] = mime
    doc['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary.extend(b'\x00' * (-len(binary) % 4))
    Path(path).write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(binary))
                           + struct.pack('<I4s', len(encoded), b'JSON') + encoded
                           + struct.pack('<I4s', len(binary), b'BIN\x00') + binary)
    for image in doc['images']:
        view = doc['bufferViews'][image['bufferView']]
        embedded = binary[view['byteOffset']:view['byteOffset'] + view['byteLength']]
        if hashlib.sha256(embedded).hexdigest() != sha(map_path):
            raise ValueError('Exported RGB image bytes differ from the source map')


def exported_mesh_arrays(path, node_name):
    """Read actual exported glTF accessors, including its texture V convention."""
    raw = Path(path).read_bytes()
    if raw[:4] != b'glTF':
        raise ValueError('Expected GLB')
    json_size, json_type = struct.unpack_from('<I4s', raw, 12)
    if json_type != b'JSON':
        raise ValueError('Expected GLB JSON chunk')
    doc = json.loads(raw[20:20 + json_size])
    binary_header = 20 + json_size
    binary_size, binary_type = struct.unpack_from('<I4s', raw, binary_header)
    if binary_type != b'BIN\x00':
        raise ValueError('Expected embedded GLB buffer')
    binary = memoryview(raw)[binary_header + 8:binary_header + 8 + binary_size]
    node = next(node for node in doc['nodes'] if node.get('name') == node_name)
    primitive = doc['meshes'][node['mesh']]['primitives'][0]

    def accessor(index):
        entry = doc['accessors'][index]
        view = doc['bufferViews'][entry['bufferView']]
        dtype = {5123: '<u2', 5125: '<u4', 5126: '<f4'}[entry['componentType']]
        width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[entry['type']]
        if view.get('byteStride'):
            raise ValueError('Unexpected strided generated accessor')
        offset = view.get('byteOffset', 0) + entry.get('byteOffset', 0)
        return np.frombuffer(binary, dtype=dtype, count=entry['count'] * width,
                             offset=offset).reshape(entry['count'], width)

    vertices = accessor(primitive['attributes']['POSITION'])
    uv = accessor(primitive['attributes']['TEXCOORD_0']) if 'TEXCOORD_0' in primitive['attributes'] else None
    faces = accessor(primitive['indices']).reshape(-1, 3)
    return vertices, faces, uv, doc


def prepare(preview_root, land_path, output, *, land_top=0., water_z=-3.5, bottom=-8.):
    preview_root, land_path, output = map(Path, (preview_root, land_path, output))
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Refusing to overwrite existing RGB solid output')
    if not bottom < water_z < land_top:
        raise ValueError('Expected bottom < water < land top')
    manifest = json.loads((preview_root / 'manifest.json').read_text())
    if manifest.get('schema') != 'citycode.background_web_preview.v1':
        raise ValueError('Expected a background preview manifest')
    map_path = (preview_root / manifest['files']['map']['path']).resolve()
    if not map_path.is_relative_to(preview_root.resolve()):
        raise ValueError('Map must be in the packaged preview directory')
    map_hash, land_hash = sha(map_path), sha(land_path)
    if map_hash != manifest['files']['map']['sha256']:
        raise ValueError('Packaged RGB map checksum differs')
    bounds = list(map(float, manifest['bounds_xy']))
    extent = box(*bounds)
    land = shape(json.loads(land_path.read_text())['geometry'])
    if not land.is_valid or land.is_empty or not polygons(land) or land.area <= 0:
        raise ValueError('Invalid packaged land polygon')
    if not extent.covers(land):
        raise ValueError('Land polygon extends beyond the RGB map bounds')
    water_polygon = extent.difference(land)
    bodies = []
    for polygon in polygons(land):
        body = trimesh.creation.extrude_polygon(polygon, height=land_top - bottom, engine='earcut')
        body.vertices[:, 2] += bottom
        bodies.append(body)
    solid = trimesh.util.concatenate(bodies)
    if not solid.is_watertight or not solid.is_winding_consistent or solid.volume <= 0:
        raise ValueError('RGB land body is not a consistently wound closed solid')
    cap = solid.submesh([np.flatnonzero(solid.face_normals[:, 2] > .999)], append=True)
    walls = solid.submesh([np.flatnonzero(solid.face_normals[:, 2] <= .999)], append=True)
    water = planar_mesh(water_polygon, water_z)
    image = Image.open(map_path)
    scene = trimesh.Scene()
    scene.graph.update(frame_to='land', frame_from=scene.graph.base_frame, matrix=np.eye(4))
    scene.add_geometry(textured(cap, image, bounds, 'RGB land'), node_name='land_top',
                       geom_name='land_top', parent_node_name='land')
    walls.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name='Land shore walls and bottom', baseColorFactor=WALL_COLOR,
        metallicFactor=0, roughnessFactor=1, doubleSided=True))
    scene.add_geometry(walls, node_name='land_walls_bottom', geom_name='land_walls_bottom', parent_node_name='land')
    scene.add_geometry(textured(water, image, bounds, 'RGB water'), node_name='water', geom_name='water')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.glb')
    try:
        scene.export(temporary)
        preserve_rgb_bytes(temporary, map_path)
        # Validate the exported float32 geometry, not just the source polygon.
        cap_vertices, cap_faces, cap_uv, doc = exported_mesh_arrays(temporary, 'land_top')
        water_vertices, water_faces, water_uv, _ = exported_mesh_arrays(temporary, 'water')
        wall_vertices, wall_faces, _, _ = exported_mesh_arrays(temporary, 'land_walls_bottom')
        exported_cap = trimesh.Trimesh(cap_vertices, cap_faces, process=False)
        exported_water = trimesh.Trimesh(water_vertices, water_faces, process=False)
        exported_walls = trimesh.Trimesh(wall_vertices, wall_faces, process=False)
        exported_body = trimesh.util.concatenate([exported_cap, exported_walls])
        exported_body.merge_vertices()
        if not exported_body.is_watertight or not exported_body.is_winding_consistent:
            raise ValueError('Exported RGB land body lost closure or winding')
        cap_footprint = footprint(exported_cap)
        water_footprint = footprint(exported_water)
        # Float32 GLB world positions introduce sub-millimetre boundary error.
        tolerance = max(1e-4, land.length * 1e-4)
        cap_error = cap_footprint.symmetric_difference(land).area
        water_error = water_footprint.symmetric_difference(water_polygon).area
        overlap_error = cap_footprint.intersection(water_footprint).area
        coverage_error = cap_footprint.union(water_footprint).symmetric_difference(extent).area
        if max(cap_error, water_error, overlap_error, coverage_error) > tolerance:
            raise ValueError('Exported land/water differs from the packaged boundary')
        if not np.allclose(cap_vertices[:, 2], land_top, atol=1e-6):
            raise ValueError('Land cap is not at the requested height')
        if not np.allclose(water_vertices[:, 2], water_z, atol=1e-6):
            raise ValueError('Water plane is not at the requested height')
        for vertices, uv in [(cap_vertices, cap_uv), (water_vertices, water_uv)]:
            expected_uv = np.column_stack(((vertices[:, 0] - bounds[0]) / (bounds[2] - bounds[0]),
                                            (vertices[:, 1] - bounds[1]) / (bounds[3] - bounds[1])))
            if not np.allclose(uv, expected_uv, atol=2e-7, rtol=0):
                raise ValueError('Exported glTF UVs do not follow image rows toward UE +Y')
        if any(any(key in node for key in ['matrix', 'translation', 'rotation', 'scale']) for node in doc['nodes']):
            raise ValueError('Unexpected transform in world-coordinate RGB map')
        if sha(map_path) != map_hash or sha(land_path) != land_hash:
            raise ValueError('Source map or land polygon changed during preparation')
        proof = {
            'schema': 'citycode.rgb_map_solid.v1', 'view_only': True,
            'coordinates': 'UE XYZ metres Z-up; no axis swap or placement matrix',
            'source_map_sha256': map_hash, 'land_sha256': land_hash,
            'land_polygon': 'same packaged land.geojson as the coastal 3D background',
            'bounds_xy': bounds, 'land_bounds': exported_body.bounds.tolist(),
            'water_bounds': exported_water.bounds.tolist(),
            'land_top_z_m': land_top, 'water_z_m': water_z, 'bottom_z_m': bottom,
            'land_thickness_m': land_top - bottom, 'land_above_water_m': land_top - water_z,
            'wall_color_rgba': WALL_COLOR, 'land_area_m2': land.area,
            'water_area_m2': water_polygon.area,
            'land_cap_faces': len(cap_faces), 'land_wall_bottom_faces': len(wall_faces),
            'water_faces': len(water_faces),
            'faces': len(cap_faces) + len(wall_faces) + len(water_faces),
            'body_watertight': bool(exported_body.is_watertight),
            'body_winding_consistent': bool(exported_body.is_winding_consistent),
            'water_planar': True, 'water_overlaps_land': False,
            'land_boundary_symmetric_difference_m2': cap_error,
            'water_boundary_symmetric_difference_m2': water_error,
            'land_water_overlap_m2': overlap_error,
            'combined_extent_symmetric_difference_m2': coverage_error,
            'float32_area_tolerance_m2': tolerance,
            'texture_orientation': 'actual glTF accessor: image top row at UE ymin uses V=0; rows increase toward +Y',
            'embedded_map_sha256': map_hash, 'rgb_texture_reencoded': False,
            'source_rgb_pixels_changed': False, 'api_calls': 0, 'gpu_inference': False,
            'measured_terrain': False,
            'processing': 'Flat RGB-textured land cap; closed neutral shore walls and bottom; RGB-textured water only outside the same published land polygon.',
            'bytes': temporary.stat().st_size, 'output_sha256': sha(temporary),
        }
        temporary.replace(output)
        metadata = output.with_suffix('.json')
        staging = metadata.with_suffix('.json.tmp')
        staging.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
        staging.replace(metadata)
    finally:
        temporary.unlink(missing_ok=True)
        image.close()
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-root', type=Path, default=ROOT / 'data/smallcity_p384_background_web_preview')
    parser.add_argument('--land', type=Path, default=ROOT / 'data/smallcity_p384_background_coastal/land.geojson')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--land-top', type=float, default=0.)
    parser.add_argument('--water-z', type=float, default=-3.5)
    parser.add_argument('--bottom', type=float, default=-8.)
    parser.add_argument('--register', action='store_true', help='Register the generated preview in its manifest after validation')
    args = parser.parse_args()
    output = args.output or args.preview_root / 'map_solid.glb'
    proof = prepare(args.preview_root, args.land, output, land_top=args.land_top, water_z=args.water_z, bottom=args.bottom)
    if args.register:
        relative = output.resolve().relative_to(args.preview_root.resolve()).as_posix()
        manifest_path = args.preview_root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['map_solid'] = {'path': relative, 'sha256': proof['output_sha256'], 'bytes': proof['bytes']}
        manifest['map_solid'] = proof
        staging = manifest_path.with_suffix('.json.tmp')
        staging.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        staging.replace(manifest_path)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
