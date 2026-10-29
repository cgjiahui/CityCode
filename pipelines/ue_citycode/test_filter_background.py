"""Small CPU fixtures test spatial and byte-preservation contracts."""
import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
from shapely.geometry import Polygon, box

from filter_background import (PredictionMask, clipped_triangle, filter_glb,
                               stable_serialized_pieces, weight_area)
from interchange import Writer, accessor, digest_file, read_glb


def fixture(path, triangles, image_payload=b"\x89PNG\r\n\x1a\noriginal-texture-bytes"):
    writer = Writer(directory=path.parent)
    writer.doc["images"] = [{"bufferView": writer.blob(image_payload), "mimeType": "image/png"}]
    for index, triangle in enumerate(triangles):
        vertices = np.asarray(triangle, "<f4")
        positions = writer.array(vertices)
        uv = writer.array(vertices[:, :2].copy(), {"componentType": 5126, "type": "VEC2", "count": 3})
        indices = writer.array(np.array([[0], [1], [2]], "<u4"),
                               {"componentType": 5125, "type": "SCALAR", "count": 3}, 34963)
        writer.doc["meshes"].append({"name": f"test_{index}", "primitives": [
            {"attributes": {"POSITION": positions, "TEXCOORD_0": uv}, "indices": indices}]})
        writer.doc["nodes"].append({"name": f"test_{index}", "mesh": index})
        writer.doc["scenes"][0]["nodes"].append(index)
    writer.save(path)


def output_triangles(path):
    doc, binary, _ = read_glb(path)
    result = {}
    for mesh in doc["meshes"]:
        arrays = []
        for p in mesh["primitives"]:
            v, _ = accessor(doc, binary, p["attributes"]["POSITION"])
            ix, _ = accessor(doc, binary, p["indices"])
            uv, _ = accessor(doc, binary, p["attributes"]["TEXCOORD_0"])
            arrays.append((v[ix.reshape(-1, 3)], uv[ix.reshape(-1, 3)]))
        result[mesh["name"]] = arrays
    binary.release()
    return doc, result


class TriangleClippingTests(unittest.TestCase):
    def test_large_triangle_crosses_mask_while_corners_and_centroid_miss(self):
        triangle = np.array([[-10, -10, 6], [10, -10, 6], [0, 10, 6]], float)
        region = box(0, 0, 1, 1)
        self.assertFalse(any(region.covers(__import__('shapely').geometry.Point(p)) for p in triangle[:, :2]))
        self.assertFalse(region.covers(__import__('shapely').geometry.Point(triangle[:, :2].mean(0))))
        pieces = clipped_triangle(triangle, region, 2)
        self.assertIsNotNone(pieces)
        total = sum(Polygon(weights @ triangle[:, :2]).area for weights in pieces)
        self.assertAlmostEqual(total, 199., places=7)
        for weights in pieces:
            self.assertLess(Polygon(weights @ triangle[:, :2]).intersection(region).area, 1e-8)

    def test_z_plane_clipping_preserves_low_portion_of_sloping_bridge_side(self):
        triangle = np.array([[0, 0, 0], [1, 0, 4], [0, 1, 4]], float)
        pieces = clipped_triangle(triangle, box(-1, -1, 2, 2), 2)
        self.assertAlmostEqual(sum(weight_area(p) for p in pieces), .125)
        for weights in pieces:
            self.assertLessEqual((weights @ triangle)[:, 2].max(), 2.+1e-10)

    def test_vertical_riser_clipping_preserves_ground_and_mask_exterior(self):
        triangle = np.array([[.5, -1, 0], [.5, 2, 0], [.5, .5, 6]], float)
        pieces = clipped_triangle(triangle, box(0, 0, 1, 1), 2)
        self.assertIsNotNone(pieces)
        self.assertGreater(sum(weight_area(p) for p in pieces), 0.)
        for weights in pieces:
            center = (weights @ triangle).mean(0)
            self.assertFalse(0 < center[1] < 1 and center[2] > 2.+1e-8)

    def test_nearly_vertical_float32_projection_does_not_amplify_xy_quantization(self):
        triangle = np.array([[500., 0., 0.], [501., .000003, 0.], [500.5, .000001, 8.]], float)
        region = box(500.25, -1, 500.75, 1)
        pieces = clipped_triangle(triangle, region, 2.)
        self.assertIsNotNone(pieces)
        for weights in pieces:
            # Repeat after actual GLB float32 serialization; clipping is stable.
            actual = (weights @ triangle).astype(np.float32).astype(float)
            again = clipped_triangle(actual, region, 2.)
            if again is not None:
                self.assertLess(.5-sum(weight_area(p) for p in again), 1e-5)

    def test_polygonized_mask_retains_holes_and_correct_metric_axes(self):
        cells = np.ones((5, 7), bool); cells[1:4, 2:5] = False
        mask = PredictionMask(cells, [100, -50], .75)
        self.assertAlmostEqual(mask.geometry.area, 26*.75**2)
        self.assertTrue(mask.geometry.covers(__import__('shapely').geometry.Point(100.1, -49.9)))
        self.assertFalse(mask.geometry.covers(__import__('shapely').geometry.Point(102, -48.5)))

    def test_serialized_fragment_guard_drops_only_tiny_new_artifacts(self):
        mask = PredictionMask([[True]], [500, 0], 1.)
        tiny = np.array([[500,0,4],[501,0,4],[500.5,.000003,4]],float)
        result, count, area = stable_serialized_pieces([np.eye(3)],tiny,mask,np.zeros(3),2.)
        self.assertEqual(result,[])
        self.assertEqual(count,1)
        self.assertLess(area,.00001)
        real_face = np.array([[500,0,4],[501,0,4],[500,1,4]],float)
        with self.assertRaises(ValueError):
            stable_serialized_pieces([np.eye(3)],real_face,mask,np.zeros(3),2.)


class StreamedDerivativeTests(unittest.TestCase):
    def test_same_xy_ground_kept_bridge_removed_outside_high_kept_empty_nodes_dropped(self):
        ground = [[.1, .1, 0], [.8, .1, 0], [.1, .8, 0]]
        high = [[x, y, 6] for x, y, _ in ground]
        outside = [[10, 10, 7], [11, 10, 7], [10, 11, 7]]
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/'source.glb', Path(folder)/'filtered.glb'
            fixture(source, [ground, high, outside]); before = digest_file(source)
            proof = filter_glb(source, destination, PredictionMask([[True]], [0, 0], 1))
            doc, meshes = output_triangles(destination)
            self.assertEqual(set(meshes), {'test_0', 'test_2'})
            self.assertEqual(proof['output_face_count'], 2)
            self.assertEqual(proof['empty_primitives_dropped'], 1)
            self.assertEqual(proof['empty_meshes_dropped'], 1)
            self.assertEqual(len(doc['nodes']), 2)
            self.assertTrue(proof['image_payloads_byte_identical'])
            self.assertEqual(digest_file(source), before)
            np.testing.assert_allclose(meshes['test_0'][0][0][0], ground)

    def test_no_axis_swap_and_world_anchor_applied_once(self):
        # Local Y=40 is far from the mask's Y=0; only adding the explicit -40
        # Y anchor reaches it. A Y/Z swap would fail to remove this surface.
        triangle = [[.1, 40.1, 3], [.8, 40.1, 3], [.1, 40.8, 3]]
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/'source.glb', Path(folder)/'filtered.glb'
            fixture(source, [triangle])
            proof = filter_glb(source, destination, PredictionMask([[True]], [0, 0], 1), anchor=[0, -40, 0])
            self.assertEqual(proof['output_face_count'], 0)
            self.assertEqual(proof['output_mesh_count'], 0)
            self.assertIsNone(proof['output_world_bounds_m'])

    def test_boundary_uv_interpolation_and_winding_preserve_original_triangle_plane(self):
        triangle = [[-10, -10, 6], [10, -10, 6], [0, 10, 6]]
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/'source.glb', Path(folder)/'filtered.glb'
            fixture(source, [triangle])
            proof = filter_glb(source, destination, PredictionMask([[True]], [0, 0], 1))
            _, meshes = output_triangles(destination)
            v, uv = meshes['test_0'][0]
            self.assertGreater(len(v), 1)
            np.testing.assert_allclose(uv, v[:, :, :2], atol=1e-6)
            normals = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
            self.assertTrue(np.all(normals[:, 2] > 0))
            self.assertTrue(proof['source_unchanged'])
            self.assertEqual(proof['primitives'][0]['mask_boundary_faces_clipped'], 1)

    def test_source_and_destination_cannot_be_same(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'source.glb'; fixture(p, [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
            with self.assertRaises(ValueError):
                filter_glb(p, p, PredictionMask([[True]], [0, 0], 1))


if __name__ == '__main__':
    unittest.main()
