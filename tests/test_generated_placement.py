"""New buildings derive placement from current prediction and sampled pose."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, shape

from citycode_assets import generation
from citycode_assets.lod import _accessor, _read_glb
from citycode_assets.placement import initial_generated_placement, fit_generated_asset, load_placement_inputs
from citycode_assets.appearance import sha256_file
from test_asset_delivery_contract import cube_glb


def building(geometry, height=6, base=2):
    center = geometry.centroid
    world = np.eye(4); world[:3, 3] = [center.x, center.y, base]
    return {"building_id": "current_only", "footprint_world": geometry.__geo_interface__,
            "base_z_m": base, "top_z_m": base + height, "height_m": height,
            "world_from_local_4x4": world.tolist(), "local_from_world_4x4": np.linalg.inv(world).tolist()}


def simple_audit(record):
    geom = shape(record["footprint_world"])
    normalized, _, _ = generation.normalize_geometry(geom)
    diagonal = math.hypot(normalized.bounds[2] - normalized.bounds[0], normalized.bounds[3] - normalized.bounds[1])
    return {"height_m": record["height_m"], "footprint_area_m2": geom.area,
            "requested_height_voxels": 12, "z_min": 8, "z_max": 19,
            "model_frame_alignment": {"prediction_normalization": [0, 0, diagonal],
               "target_normalization": [.125, -.0625, .5], "rotation_deg": 90,
               "scale_after_area_normalization": 1.2, "translation": [.2, -.3]}}


def sampled_coords(geometry):
    normalized, _, _ = generation.normalize_geometry(geometry)
    native = affinity.translate(affinity.scale(affinity.rotate(normalized, 35, origin=(0, 0)), .7, .7, origin=(0, 0)), .05, -.02)
    xy = [(x, y) for x in range(32) for y in range(32)
          if native.contains(Point((x + .5) / 32 - .5, (y + .5) / 32 - .5))]
    return np.array([(x, y, z) for x, y in xy for z in range(8, 24)], dtype=np.int32)


class GeneratedPlacementContract(unittest.TestCase):
    def test_two_origins_axis_conversion_vertical_cells_and_matrix_closure(self):
        record = building(Polygon([(10, 20), (30, 20), (30, 30), (10, 30)]))
        pose = initial_generated_placement(record, simple_audit(record))
        world = np.array(pose["world_from_asset_4x4"])
        # The current footprint bbox centre maps to native (.225,-.2125),
        # hence glTF (.225, native_z, .2125). No old building is involved.
        lower = world @ [.225, -.25, .2125, 1]
        upper = world @ [.225, .125, .2125, 1]
        np.testing.assert_allclose(lower, [20, 25, 2, 1], atol=1e-8)
        np.testing.assert_allclose(upper, [20, 25, 8, 1], atol=1e-8)
        np.testing.assert_allclose(world @ pose["asset_from_world_4x4"], np.eye(4), atol=1e-8)
        np.testing.assert_allclose(np.array(pose["world_from_building_local_4x4"]) @ pose["building_local_from_asset_4x4"], world, atol=1e-8)
        np.testing.assert_allclose(pose["normal_asset_to_world_3x3"], np.linalg.inv(world[:3, :3]).T, atol=1e-8)
        self.assertGreater(np.linalg.det(world[:3, :3]), 0)
        self.assertFalse(pose["actual_mesh_acceptance_performed"])

    def _shape_roundtrip(self, geometry, force_topology_fallback=False):
        record = building(geometry, height=8)
        coords = sampled_coords(geometry)
        actual_fit = generation.fit
        counter = [0]
        def unstable(*args, **kwargs):
            counter[0] += 1
            if force_topology_fallback and counter[0] <= 2:
                raise RuntimeError("unstable courtyard support")
            return actual_fit(*args, **kwargs)
        with patch("citycode_assets.generation.fit", side_effect=unstable):
            _, audit = generation.predicted_height_coordinates(coords, geometry, record["height_m"])
        normalized, center, scale = generation.normalize_geometry(geometry)
        counter[0] = 0
        with patch("citycode_assets.generation.fit", side_effect=unstable):
            oracle = generation.align_control_to_baseline({"normalized": normalized,
                "ratio": record["height_m"] / math.sqrt(geometry.area)}, coords)
        pose = initial_generated_placement(record, audit)
        world = np.array(pose["world_from_asset_4x4"])
        # Native XY to world XY uses GLB (x,-native_y) horizontal coordinates.
        inverse = world[:2][:, [0, 2]] @ np.diag([1, -1])
        recovered = affinity.affine_transform(oracle["normalized_model_frame"],
            [inverse[0, 0], inverse[0, 1], inverse[1, 0], inverse[1, 1], world[0, 3], world[1, 3]])
        self.assertLess(recovered.symmetric_difference(geometry).area / geometry.area, 1e-5)
        self.assertEqual(len(recovered.interiors), len(geometry.interiors))
        if force_topology_fallback:
            self.assertTrue(pose["topology_fallback"]["used"])

    def test_ring_and_concave_contours_roundtrip_real_alignment(self):
        ring = Point(100, -200).buffer(10).difference(Point(100, -200).buffer(4))
        concave = Polygon([(100, -200), (120, -200), (120, -194), (108, -194), (108, -180), (100, -180)])
        for geometry in (ring, concave):
            with self.subTest(kind="ring" if geometry.interiors else "concave"):
                self._shape_roundtrip(geometry)

    def test_convex_topology_fallback_preserves_pose_and_original_hole(self):
        ring = Point(100, -200).buffer(10).difference(Point(102, -199).buffer(4))
        self._shape_roundtrip(ring, force_topology_fallback=True)

    def test_actual_mesh_fit_does_not_hide_wrong_generated_height(self):
        record = building(Polygon([(10, 20), (30, 20), (30, 30), (10, 30)]))
        audit = simple_audit(record)
        prior = np.array(initial_generated_placement(record, audit)["world_from_asset_4x4"])
        with tempfile.TemporaryDirectory() as temporary:
            for height, accepted in [(6, True), (12, False)]:
                path = Path(temporary) / f"generated_{height}.glb"
                cube_glb(path)
                # A textured actual GLB whose world box can be compared with
                # the prediction; the second mesh independently doubles Z.
                intended = np.array([[20, 0, 0, 10], [0, 0, -10, 30], [0, height, 0, 2], [0, 0, 0, 1]])
                mesh_transform = np.linalg.inv(prior) @ intended
                data, document, binary = _read_glb(path)
                vertices, _ = _accessor(data, document, binary, 0)
                normals, _ = _accessor(data, document, binary, 1)
                vertices[:] = vertices @ mesh_transform[:3, :3].T + mesh_transform[:3, 3]
                transformed = normals @ np.linalg.inv(mesh_transform[:3, :3])
                normals[:] = transformed / np.linalg.norm(transformed, axis=1)[:, None]
                path.write_bytes(data)
                fit = fit_generated_asset(path, record, audit, expected_source_sha256=sha256_file(path))
                self.assertEqual(fit["status"] == "accepted", accepted, fit["reasons"])
                if not accepted:
                    self.assertLess(fit["fit"]["relative_z_scale"], .8)

    def test_hash_bound_height_evidence_cannot_be_swapped(self):
        record = building(Polygon([(10, 20), (30, 20), (30, 30), (10, 30)]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "building.json"; output = root / "generated"
            source.write_text(json.dumps(record)); output.mkdir()
            (output / "height_correction.json").write_text(json.dumps(simple_audit(record)))
            np.savez(output / "stage1_image_only.npz", coords=np.array([[0, 0, 0]]))
            np.savez(output / "stage1_height_corrected.npz", coords=np.array([[0, 0, 8]]))
            (output / "raw_mesh.pt").write_bytes(b"fixture decoded cache")
            placement = {name: {"path": filename, "sha256": sha256_file(output / filename)} for name, filename in
                         [("height_correction", "height_correction.json"), ("image_only_structure", "stage1_image_only.npz"),
                          ("height_corrected_structure", "stage1_height_corrected.npz")]}
            (output / "raw_manifest.json").write_text(json.dumps({"status": "complete",
                "bytes": (output / "raw_mesh.pt").stat().st_size, "sha256": sha256_file(output / "raw_mesh.pt"), "contract": {
                "building_sha256": sha256_file(source)}, "placement_inputs": placement}))
            actual, audit = load_placement_inputs(source, output)
            self.assertEqual(actual["building_id"], record["building_id"])
            initial_generated_placement(actual, audit)
            (output / "height_correction.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                load_placement_inputs(source, output)


if __name__ == "__main__":
    unittest.main()
