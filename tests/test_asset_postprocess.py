"""Source replacements preserve identity, geometry, calibration and old assets."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from citycode_assets.lod import sha256
from citycode_assets.postprocess import file_proof, postprocess_assets, roof_uv, thin_surface, observed_roof_material, _json_write
from citycode_assets.registration import read_glb_scene_geometry
from test_asset_delivery_contract import cube_glb


class AssetPostprocessContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name) / "data"
        self.assets = self.data / "assets"; self.assets.mkdir(parents=True)
        self.structure = self.data / "structure"; self.folder = self.structure / "buildings/new"
        observation = self.folder / "observations/view"; observation.mkdir(parents=True)
        cube_glb(self.folder / "proxy.glb")
        image = Image.new("RGBA", (2, 2), (20, 40, 60, 255))
        image.save(observation / "roof_evidence.png")
        obs = {"building_id": "new", "observation_id": "view",
               "bev_crop_pixel_center_to_world_xy_3x3": [[.5, 0, .25], [0, -.5, -.25], [0, 0, 1]],
               "paths": {"roof_evidence": "roof_evidence.png"},
               "files": {"roof_evidence.png": sha256(observation / "roof_evidence.png")}}
        (observation / "observation.json").write_text(json.dumps(obs))
        translation = np.eye(4); translation[2, 3] = 2
        self.footprint = {"type": "Polygon", "coordinates": [[[0, -1], [1, -1], [1, 0], [0, 0], [0, -1]]]}
        building = {"building_id": "new", "canonical_observation_id": "view",
            "footprint_world": self.footprint, "base_z_m": 2, "top_z_m": 3,
            "world_from_local_4x4": translation.tolist(),
            "paths": {"proxy_local_gltf_y_up": "proxy.glb"},
            "observations": [{"observation_id": "view", "path": "observations/view"}]}
        (self.folder / "building.json").write_text(json.dumps(building))
        (self.structure / "manifest.json").write_text(json.dumps({"ids": ["old", "new"]}))
        self.old = {"building_id": "old", "status": "accepted_reuse", "asset_instance_id": "old",
                    "arbitrary_frozen_provenance": {"keep": [1, 2, 3]}}
        old_instance = {"id": "old", "source_building_ids": ["old"], "frozen": True}
        self.base = {"schema": "citycode.building_asset_bindings.v1", "path_base": "manifest_directory",
            "source_structure": {"instance_count": 2, "manifest_sha256": sha256(self.structure / "manifest.json"),
                                 "building_ids": ["old", "new"]},
            "buildings": [self.old, {"building_id": "new", "status": "pending_review",
                "structure_path": "../structure/buildings/new/building.json", "target_geometry": self.footprint}],
            "asset_instances": [old_instance], "asset_instance_groups": []}
        self.base_path = self.assets / "binding.json"; self.base_path.write_text(json.dumps(self.base))
        self.decision = {"building_id": "new", "action": "source_structure", "reason_code": "main_body_mismatch",
            "reason": "Use the verified source shape; facade detail remains unknown", "reviewer": "test",
            "evidence": [file_proof(self.folder / "building.json", self.assets),
                         file_proof(observation / "observation.json", self.assets)]}
        self.plan_path = self.assets / "plan.json"

    def run_plan(self, changes=None):
        plan = {"schema": "citycode.asset_postprocess_decisions.v1", "base_binding_sha256": sha256(self.base_path),
                "records": [self.decision]}
        if changes: changes(plan)
        self.plan_path.write_text(json.dumps(plan))
        output = self.assets / "processed.json"
        report = postprocess_assets(self.base_path, self.plan_path, self.structure, output)
        return report, json.loads(output.read_text())

    def test_source_rebuild_preserves_old_rows_and_exact_triangle_positions(self):
        _, result = self.run_plan()
        self.assertEqual(result["buildings"][0], self.old)
        self.assertEqual(result["asset_instances"][0], self.base["asset_instances"][0])
        self.assertEqual(result["source_structure"], self.base["source_structure"])
        new = result["buildings"][1]
        self.assertEqual(new["status"], "accepted_structure")
        self.assertFalse(new["postprocess"]["detailed_asset_accepted"])
        self.assertEqual(new["target_geometry"], self.footprint)
        for lod in new["lods"].values():
            scene = read_glb_scene_geometry(self.assets / lod["path"])
            world = np.asarray(new["world_from_local_4x4"])
            z = (scene["vertices"] @ world[:3, :3].T + world[:3, 3])[:, 2]
            self.assertAlmostEqual(z.min(), 2); self.assertAlmostEqual(z.max(), 3)
        self.assertEqual(new["lods"]["0.25M"]["sha256"], new["lods"]["0.1M"]["sha256"])

    def test_stale_input_and_replacement_of_accepted_assets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "other binding"):
            self.run_plan(lambda p: p.update(base_binding_sha256="0" * 64))
        self.decision["building_id"] = "old"
        with self.assertRaisesRegex(ValueError, "Accepted assets"):
            self.run_plan()

    def test_source_sha_drift_cannot_be_reapproved_by_the_exporter(self):
        evidence = self.decision["evidence"][0]
        evidence["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.run_plan()
        self.assertFalse((self.assets / "postprocessed_assets").exists())

    def test_pixel_center_uv_mapping_and_observed_surface_thickness(self):
        material = {"pixel_from_world": np.linalg.inv([[.5, 0, .25], [0, -.5, -.25], [0, 0, 1]]), "size": [2, 2]}
        uv = roof_uv(np.array([[.25, -.25, 3], [.75, -.75, 3]]), material)
        np.testing.assert_allclose(uv, [[.25, .25], [.75, .75]])
        vertices = np.array([[0., 0., 30.], [1., 0., 30.2], [0., 1., 30.1]])
        closed, faces = thin_surface(vertices, [[0, 1, 2]], .12)
        np.testing.assert_array_equal(closed[:3], vertices)
        np.testing.assert_allclose(closed[3:], vertices - [0, 0, .12])
        self.assertEqual(len(faces), 8)
        self.assertGreater(closed[:, 2].min(), 29)

    def test_unobserved_roof_border_stays_neutral_and_failed_json_is_retryable(self):
        material = observed_roof_material(self.folder / "building.json")
        import io
        with Image.open(io.BytesIO(material["png"])) as image:
            pixels = np.array(image)
        self.assertEqual(material["size"], [4, 4])
        np.testing.assert_array_equal(pixels[0], np.full((4, 3), 174))
        np.testing.assert_array_equal(pixels[1, 1], [20, 40, 60])
        np.testing.assert_allclose(roof_uv([[.25, -.25, 3]], material), [[.375, .375]])
        output = self.assets / "report.json"
        with self.assertRaises(ValueError):
            _json_write(output, {"bad": float("nan")})
        self.assertFalse(output.with_name("report.json.partial").exists())
        _json_write(output, {"status": "passed"})
        self.assertEqual(json.loads(output.read_text()), {"status": "passed"})


if __name__ == "__main__": unittest.main()
