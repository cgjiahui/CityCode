"""Native-texture solid joins: overlap removal, layer separation and roundtrip."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from stitch_bridge_network import union_arrays, decode_native
from bridge_assets import write_glb


def cube(center):
    mesh = trimesh.creation.box(extents=[1., 1., 1.])
    triangles = mesh.triangles+center
    uv = triangles[:, :, :2]*.37+np.array([.13, .29])
    slots = np.arange(len(triangles)) % 4
    return triangles, uv, slots


class DeckStitchTests(unittest.TestCase):
    def test_overlapping_solids_become_one_surface_preserving_native_uv(self):
        triangles, uv, slots, volume = union_arrays([cube([0., 0., 0.]), cube([.5, 0., 0.])])
        self.assertAlmostEqual(volume, 1.5, places=9)
        np.testing.assert_allclose(uv, triangles[:, :, :2]*.37+[.13, .29], atol=1e-10)
        self.assertEqual(set(slots), {0, 1, 2, 3})
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(triangles.size//3).reshape(-1, 3))
        self.assertTrue(mesh.is_watertight)
        self.assertEqual(len(mesh.split()), 1)

    def test_xy_crossing_does_not_fill_space_between_distinct_levels(self):
        triangles, _, _, volume = union_arrays([cube([0., 0., 0.]), cube([0., 0., 5.])])
        self.assertAlmostEqual(volume, 2., places=9)
        mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(triangles.size//3).reshape(-1, 3))
        self.assertEqual(len(mesh.split()), 2)
        self.assertTrue(mesh.is_watertight)

    def test_gltf_roundtrip_keeps_native_coordinates_uv_and_outward_winding(self):
        triangles, uv, slots, _ = union_arrays([cube([1., 2., 3.])])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'deck.glb'
            write_glb(path, triangles, uv, slots)
            p, t, s = decode_native(path, np.zeros(3))
        np.testing.assert_allclose(t, p[:, :, :2]*.37+[.13, .29], atol=2e-7)
        self.assertEqual(set(s), {0, 1, 2, 3})
        mesh = trimesh.Trimesh(vertices=p.reshape(-1, 3), faces=np.arange(p.size//3).reshape(-1, 3))
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.volume, 1., places=6)


if __name__ == '__main__':
    unittest.main()
