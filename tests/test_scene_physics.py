"""Physical replay preserves assets and rechecks the scene evidence it used."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from citycode_assets.registration import read_glb_scene_geometry
from citycode_scene.cache import build_cache
from citycode_scene.geometry import sha256
from citycode_scene.replay import apply_decisions, delta_matrix
from citycode_scene.roads import column_occupancy, union_occupancy
from test_asset_delivery_contract import cube_glb


class ScenePhysicsContract(unittest.TestCase):
    def test_proposals_reject_scene_changes_since_real_audit(self):
        from citycode_scene.cli import audit, propose
        lanes = self.root/'lanes.geojson'
        lanes.write_text(json.dumps({'type':'FeatureCollection','features':[{
            'type':'Feature','properties':{'lane_id':'far_route','tags':['Vehicle'],'width_m':4},
            'geometry':{'type':'LineString','coordinates':[[100,100,0],[110,100,0]]}}]}))
        config={'building_binding':str(self.binding_path),'lane_xyz':str(lanes),'cpu_workers':1,
                'policy':{'maximum_horizontal_shift_m':2,'preferred_minimum_horizontal_scale':.95}}
        folder=self.root/'audit'
        audit(config,self.root,folder)
        self.assertEqual(propose(config,self.root,folder)['review_cases'],0)
        for path in [self.binding_path,lanes]:
            previous=path.read_text()
            path.write_text(previous+'\n')
            with self.assertRaisesRegex(ValueError,'changed after audit'):
                propose(config,self.root,folder)
            path.write_text(previous)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binding_path = self.root / "binding_manifest.json"
        self.recipe_path = self.root / "physical_decisions.json"
        self.output_path = self.root / "binding_physical.json"
        self.models = {}
        rows = []
        matrix = np.array([[0., -2., 0., 10.], [2., 0., 0., 20.],
                           [0., 0., 3., 1.], [0., 0., 0., 1.]])
        for aid in ("A", "B"):
            model = self.root / "assets" / aid / "model_01m.glb"
            if aid == "A":
                # Node transforms must be composed before measuring the mesh
                # centre used by the replay displacement guard.
                cube_glb(model, nodes=[{"translation": [.25, 0, 0], "children": [1]},
                                       {"mesh": 0, "translation": [.5, 0, 0]}])
            else:
                cube_glb(model, nodes=[{"mesh": 0}, {"mesh": 0, "translation": [.5, 0, 0]}], roots=[0, 1])
            world = matrix.copy()
            if aid == "B":
                world[:3, 3] += [3., 4., .5]
            rows.append({"id": aid, "source_building_ids": ["observed_"+aid],
                         "status": "accepted_generated", "building_entity_id": "whole_building",
                         "world_from_local_4x4": world.tolist(),
                         "normal_transform_world_3x3": np.linalg.inv(world[:3, :3]).T.tolist(),
                         "lods": {"0.1M": {"path": str(model.relative_to(self.root)), "sha256": sha256(model)}}})
            self.models[aid] = model
        self.binding = {"asset_instances": rows, "massing_instances": [],
                        "buildings": [{"building_id": "observed_A"}, {"building_id": "observed_B"}],
                        "building_entities": [{"id": "whole_building", "asset_instance_ids": ["A", "B"]}],
                        "asset_instance_groups": [{"id": "A", "building_ids": ["observed_A", "observed_alias"],
                                                   "world_from_local_4x4": copy.deepcopy(rows[0]["world_from_local_4x4"])}]}
        self.binding_path.write_text(json.dumps(self.binding))
        self.background = self.root / "background.glb"
        cube_glb(self.background)
        self.image = self.root / "condition.png"
        self.image.write_bytes(b"\x89PNG\r\n\x1a\ncondition-evidence")

    def recipe(self, delta=None):
        if delta is None:
            delta = delta_matrix((.25, -.25, .2))
        decisions = []
        for row in self.binding["asset_instances"]:
            current = np.array(row["world_from_local_4x4"])
            local = read_glb_scene_geometry(self.models[row["id"]])["vertices"]
            vertices = local @ current[:3, :3].T + current[:3, 3]
            centre = (vertices[:, :2].min(0)+vertices[:, :2].max(0))/2
            decisions.append({"id": row["id"], "action": "adjust", "world_delta_4x4": delta.tolist(),
                              "before_world_from_local_4x4": current.tolist(), "pivot_xy_m": centre.tolist(),
                              "model_01m_sha256": row["lods"]["0.1M"]["sha256"],
                              "reason": "Reviewed common correction preserves the building assembly.",
                              "evidence": ["actual_cube_geometry_and_scene_inputs"]})
        return {"schema": "citycode.scene_placement_decisions.v1", "status": "agent_reviewed_frozen",
                "input_binding_sha256": sha256(self.binding_path),
                "scene_inputs": [{"role": role, "path": str(path.relative_to(self.root)), "sha256": sha256(path)}
                                 for role, path in [("background", self.background), ("condition_image", self.image)]],
                "policy": {"minimum_horizontal_scale": .9, "maximum_ground_shift_m": 1.2,
                           "maximum_horizontal_shift_m": 2., "maximum_exclusions": 0},
                "decisions": decisions}

    def write_recipe(self, value):
        self.recipe_path.write_text(json.dumps(value))

    @staticmethod
    def archive(path):
        with np.load(path) as data:
            return {key: data[key].copy() for key in data.files}

    def test_world_correction_preserves_group_pose_normals_and_model_bytes(self):
        original_binding = self.binding_path.read_bytes()
        original_models = {aid: p.read_bytes() for aid, p in self.models.items()}
        delta = delta_matrix((.3, -.4, .2), .95, (12., 24.))
        recipe = self.recipe(delta)
        self.write_recipe(recipe)
        report = apply_decisions(self.binding_path, self.recipe_path, self.output_path)
        derived = json.loads(self.output_path.read_text())
        self.assertEqual(report["adjusted_placements"], 2)
        old = [np.array(r["world_from_local_4x4"]) for r in self.binding["asset_instances"]]
        new = [np.array(r["world_from_local_4x4"]) for r in derived["asset_instances"]]
        for before, after, row in zip(old, new, derived["asset_instances"]):
            np.testing.assert_allclose(after, delta @ before, atol=1e-12)
            np.testing.assert_allclose(row["normal_transform_world_3x3"], np.linalg.inv(after[:3, :3]).T, atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(new[0]) @ new[1], np.linalg.inv(old[0]) @ old[1], atol=1e-12)
        np.testing.assert_array_equal(derived["asset_instance_groups"][0]["world_from_local_4x4"], new[0])
        self.assertEqual(derived["buildings"], self.binding["buildings"])
        self.assertEqual(self.binding_path.read_bytes(), original_binding)
        self.assertEqual({aid: p.read_bytes() for aid, p in self.models.items()}, original_models)

        recipe["decisions"][1]["world_delta_4x4"] = delta_matrix((0., 0., .1)).tolist()
        self.write_recipe(recipe)
        with self.assertRaisesRegex(ValueError, "Common world correction"):
            apply_decisions(self.binding_path, self.recipe_path, self.root / "split_group.json")

    def test_changed_model_image_or_background_bytes_reject_old_recipe(self):
        self.write_recipe(self.recipe())
        for path, message in [(self.models["A"], "Actual model bytes changed"),
                              (self.background, "Scene evidence changed: background"),
                              (self.image, "Scene evidence changed: condition_image")]:
            with self.subTest(source=path.name):
                before = path.read_bytes()
                try:
                    path.write_bytes(before[:-1]+bytes([before[-1] ^ 1]))
                    with self.assertRaisesRegex(ValueError, message):
                        apply_decisions(self.binding_path, self.recipe_path, self.output_path)
                    self.assertFalse(self.output_path.exists())
                finally:
                    path.write_bytes(before)

    def test_old_recipe_cannot_apply_twice_to_derived_binding(self):
        self.write_recipe(self.recipe())
        apply_decisions(self.binding_path, self.recipe_path, self.output_path)
        once = self.output_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Input binding changed"):
            apply_decisions(self.output_path, self.recipe_path, self.root / "twice.json")
        self.assertEqual(self.output_path.read_bytes(), once)
        self.assertFalse((self.root / "twice.json").exists())

    def test_reusing_cache_replaces_existing_wrong_pose_and_component_arrays(self):
        original_dir, reused_dir = self.root / "cache_original", self.root / "cache_reused"
        original = build_cache(self.binding_path, original_dir, workers=1)
        changed = copy.deepcopy(self.binding)
        for row in changed["asset_instances"]:
            row["world_from_local_4x4"][0][3] += 7
        changed_path = self.root / "binding_shifted.json"
        changed_path.write_text(json.dumps(changed))
        build_cache(changed_path, reused_dir, workers=1)
        self.assertFalse(np.array_equal(self.archive(original_dir / original["records"][0]["geometry"])["vertices"],
                                       self.archive(reused_dir / original["records"][0]["geometry"])["vertices"]))
        restored = build_cache(self.binding_path, reused_dir, workers=1, reuse_cache_dir=original_dir)
        self.assertTrue(restored["complete"])
        self.assertEqual(restored["binding_sha256"], original["binding_sha256"])
        for left, right in zip(original["records"], restored["records"]):
            self.assertEqual(right["world_bounds"], left["world_bounds"])
            self.assertEqual(len(right.get("components", [])), len(left.get("components", [])))
            for before_path in (original_dir / "instances" / left["id"]).glob("*.npz"):
                after_path = reused_dir / "instances" / right["id"] / before_path.name
                before, after = self.archive(before_path), self.archive(after_path)
                self.assertEqual(before.keys(), after.keys())
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key], err_msg=f"{right['id']}/{before_path.name}:{key}")

    def test_road_deck_below_surface_is_clear_but_headroom_body_blocks(self):
        columns = {"offsets": np.array([0, 2, 4, 6]), "hits": np.array([-1., 0., .5, 1.5, 6., 7.])}
        value = column_occupancy(np.array([0, 1, 2]), np.zeros(3), columns, 4.5)
        np.testing.assert_array_equal(value["deck"], [True, False, False])
        np.testing.assert_array_equal(value["occupied"], [False, True, False])
        np.testing.assert_array_equal(value["solid"], [False, True, False])
        np.testing.assert_allclose(value["vertical_overlap"], [0., 1., 0.])

    def test_overlapping_components_union_preserves_occupied_middle(self):
        model = self.root / "composite.glb"
        cube_glb(model, nodes=[{"mesh": 0}, {"mesh": 0, "translation": [0, 0, .5]}], roots=[0, 1])
        matrix = np.diag([1., 1., 2., 1.]); matrix[2, 3] = 1.
        binding = {"asset_instances": [{"id": "COMPOSITE", "status": "accepted_composite",
                    "source_building_ids": ["observed"], "world_from_local_4x4": matrix.tolist(),
                    "lods": {"0.1M": {"path": model.name, "sha256": sha256(model)}}}]}
        source = self.root / "composite_binding.json"
        source.write_text(json.dumps(binding))
        cache = self.root / "composite_cache"
        record = build_cache(source, cache, workers=1)["records"][0]
        self.assertEqual(len(record["components"]), 2)
        columns = self.archive(cache / record["columns"])
        components = [self.archive(cache / r["columns"]) for r in record["components"]]
        xy, road_z = np.array([[.375, .375]]), np.array([2.25])
        # Physical cubes occupy [1,3] and [2,4]. A short navigation interval
        # through their shared middle is occupied even though aggregate parity
        # would incorrectly pair [1,2] and [3,4], leaving a fictitious gap.
        ij = np.floor((xy-columns["origin"])/columns["cell"]).astype(int)
        ids = ij[:, 1]*columns["zmin"].shape[1]+ij[:, 0]
        wrong_aggregate = column_occupancy(ids, road_z, columns, .5)
        correct_union = union_occupancy(xy, road_z, components, .5)
        self.assertFalse(bool(wrong_aggregate["occupied"][0]))
        self.assertTrue(bool(correct_union["occupied"][0]))
        self.assertAlmostEqual(float(correct_union["vertical_overlap"][0]), .25)


if __name__ == "__main__":
    unittest.main()
