import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
import trimesh

from city_pipeline.buildings import observations as implementation

HERE = Path(__file__).resolve().parent
FIXTURE = shape(json.loads((HERE / 'fixtures/mesh/earcut_touching_holes.geojson').read_text()))


class ConstrainedFallbackContracts(unittest.TestCase):
    def assert_mesh(self, geometry):
        original = geometry.wkb
        with contextlib.redirect_stdout(io.StringIO()):
            mesh = implementation.extrude_geometry(geometry, -2.5, 14.75, [10, 20, 30], {'test': True})
        triangles = mesh.triangles
        caps = triangles[np.all(triangles[:, :, 2] == 14.75, axis=1)]
        cap = unary_union([Polygon(triangle[:, :2]) for triangle in caps])
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(geometry.wkb, original)
        self.assertEqual(geometry.symmetric_difference(cap).area, 0.)
        self.assertAlmostEqual(mesh.volume, geometry.area * 17.25, delta=1e-6)
        self.assertEqual(mesh.bounds[0, 2], -2.5)
        self.assertEqual(mesh.bounds[1, 2], 14.75)
        return cap

    def test_real_earcut_failure_has_exact_cdt_cap_and_all_holes(self):
        self.assertTrue(FIXTURE.is_valid)
        self.assertEqual(len(FIXTURE.interiors), 23)
        v, f = trimesh.creation.triangulate_polygon(orient(FIXTURE, sign=1.), engine='earcut')
        wrong_cap = unary_union([Polygon(t) for t in v[f] if Polygon(t).area > 0])
        self.assertGreater(wrong_cap.symmetric_difference(FIXTURE).area, .06)
        cap = self.assert_mesh(FIXTURE)
        self.assertEqual(sum(Polygon(h).intersection(cap).area for h in FIXTURE.interiors), 0.)

    def test_successful_earcut_path_never_calls_fallback(self):
        with patch('shapely.constrained_delaunay_triangles', side_effect=AssertionError('must not run')):
            self.assert_mesh(Polygon(box(0, 0, 10, 10).exterior, [box(3, 3, 5, 5).exterior]))

    def test_reflected_real_fixture_preserves_geometry(self):
        self.assert_mesh(scale(FIXTURE, xfact=-1, yfact=1, origin=(0, 0)))

    def test_original_tiny_component_survives(self):
        self.assert_mesh(MultiPolygon([FIXTURE, box(4000., 4000., 4000.1, 4000.1)]))

    def test_corrupt_fallback_missing_one_triangle_is_rejected(self):
        polygon = orient(FIXTURE, sign=1.)
        triangles = shapely.constrained_delaunay_triangles(polygon)
        missing = GeometryCollection(list(triangles.geoms)[:-1])
        with patch('shapely.constrained_delaunay_triangles', return_value=missing):
            with self.assertRaisesRegex(ValueError, 'Constrained Delaunay changed polygon'):
                implementation.triangulate_exact(polygon)

    def test_fallback_cannot_move_vertices(self):
        polygon = orient(FIXTURE, sign=1.)
        triangles = list(shapely.constrained_delaunay_triangles(polygon).geoms)
        coordinates = list(triangles[0].exterior.coords)[:3]
        coordinates[0] = (coordinates[0][0] + 1e-9, coordinates[0][1])
        triangles[0] = Polygon(coordinates)
        with patch('shapely.constrained_delaunay_triangles', return_value=GeometryCollection(triangles)):
            with self.assertRaisesRegex(ValueError, 'new or moved vertex'):
                implementation.triangulate_exact(polygon)


if __name__ == '__main__':
    if not Path(implementation.__file__).resolve().is_relative_to(HERE.parent / 'src'):
        raise RuntimeError('Wrong source module: ' + implementation.__file__)
    unittest.main()
