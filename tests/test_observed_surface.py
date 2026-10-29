"""Source-support tests for partial observed surface export."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import Polygon, box, mapping

from citycode_assets.observed_surface import (
    build_observed_surface, closed_mesh_budget, filter_components, grid_triangles,
)


def sample_grid(side=9):
    uv = np.asarray([(x, y) for y in range(side) for x in range(side)], dtype=np.int64)
    xyz = np.column_stack([uv * .15, np.full(len(uv), 12.0)])
    return xyz, uv


class ObservedSurfaceTests(unittest.TestCase):
    def test_missing_interior_sample_cannot_be_bridged_by_real_corners(self):
        xyz, uv = sample_grid(3)
        # All four coarse corners exist, but their interior is unobserved.
        keep = ~np.all(uv == [1, 1], axis=1)
        self.assertEqual(len(grid_triangles(xyz[keep], uv[keep], box(-1, -1, 2, 2), 2)), 0)

    def test_interior_height_jump_rejects_flat_corner_interpolation(self):
        xyz, uv = sample_grid(3)
        xyz[np.all(uv == [1, 1], axis=1), 2] += 8
        self.assertEqual(len(grid_triangles(xyz, uv, box(-1, -1, 2, 2), 2)), 0)

    def test_footprint_hole_is_preserved_even_between_grid_samples(self):
        xyz, uv = sample_grid(3)
        fp = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)],
                     holes=[[(.06, .06), (.10, .06), (.10, .10), (.06, .10)]])
        self.assertLess(len(grid_triangles(xyz, uv, fp, 2)), 2)

    def test_closed_budget_counts_boundary_walls(self):
        faces = np.asarray([[0, 1, 2], [0, 2, 3]])
        self.assertEqual(closed_mesh_budget(faces), {
            "top_triangle_count": 2, "boundary_edge_count": 4, "closed_triangle_count": 12})

    def test_tiny_disconnected_component_removed_without_changing_main(self):
        xyz, uv = sample_grid(9)
        main = grid_triangles(xyz, uv, box(-1, -1, 3, 3), 2)
        extra = np.asarray([[3, 0, 12], [3.15, 0, 12], [3, .15, 12]])
        all_xyz = np.concatenate([xyz, extra])
        faces = np.concatenate([main, [[len(xyz), len(xyz) + 1, len(xyz) + 2]]])
        kept, _, removed = filter_components(all_xyz, faces)
        np.testing.assert_array_equal(kept, main)
        self.assertEqual(sum(r["triangle_count"] for r in removed), 1)

    def test_export_replays_source_and_excludes_filled_or_subbase_samples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            building_dir = root / "building"
            observation_dir = building_dir / "observations" / "o1"
            point_dir = root / "cache" / "data" / "points"
            observation_dir.mkdir(parents=True)
            point_dir.mkdir(parents=True)
            xyz, uv = sample_grid(13)
            xyz = xyz.astype(np.float32)
            provenance = np.ones(len(xyz), dtype=np.uint8)
            provenance[np.all(uv == [1, 1], axis=1)] = 2
            xyz[np.all(uv == [3, 3], axis=1), 2] = -.5
            rgb = np.arange(len(xyz) * 3, dtype=np.uint16).reshape(-1, 3).astype(np.uint8)
            np.savez(point_dir / "sample__height_cleaned_building_points.npz", xyz_m=xyz, rgb=rgb,
                     instance_id=np.full(len(xyz), 7, dtype=np.uint16), provenance=provenance)
            observation = {"observation_id": "o1", "sample_id": "sample", "source_instance_id": 7,
                           "bev_instance_id": 7,
                           "bev_crop_pixel_center_to_world_xy_3x3": [[.15, 0, 0], [0, .15, 0], [0, 0, 1]]}
            (observation_dir / "observation.json").write_text(json.dumps(observation))
            building = {"building_id": "test", "canonical_observation_id": "o1", "base_z_m": .48,
                        "top_z_m": 12, "footprint_world": mapping(box(-.1, -.1, 2, 2)),
                        "observations": [{"observation_id": "o1", "path": "observations/o1"}], "roof_layers": []}
            bp = building_dir / "building.json"
            bp.write_text(json.dumps(building))
            result = build_observed_surface(bp, root / "cache", root / "out")
            self.assertEqual(result["status"], "supported_surface_candidate")
            self.assertLessEqual(result["closed_mesh_budget"]["closed_triangle_count"], 100000)
            self.assertEqual(result["policy"]["grid_stride"], 2)
            with np.load(root / "out" / "supported_surface.npz") as a:
                original_index = a["vertex_source_flat_index"]
                np.testing.assert_array_equal(a["vertices_world_xyz_m"], xyz[original_index])
                np.testing.assert_array_equal(a["vertex_rgb"], rgb[original_index])
                self.assertTrue(np.all(provenance[original_index] == 1))
                self.assertTrue(np.all(a["vertices_world_xyz_m"][:, 2] >= .6))
                self.assertEqual(len(np.unique(a["faces"])), len(a["vertices_world_xyz_m"]))
            serialized = json.dumps(result)
            self.assertNotIn(td, serialized)


if __name__ == "__main__":
    unittest.main()
