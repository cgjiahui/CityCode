"""Isolated CPU tests: approved source cues and real scene-aware GLB fitting."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from shapely.geometry import box

from citycode_assets import orientation as candidate


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PUBLIC_REGISTRATION = PACKAGE / "src/citycode_assets/registration.py"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rotation(angle):
    radians = np.deg2rad(angle)
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    return matrix


class SourceOrientation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.asset = self.root / "synthetic_actual.glb"
        self.asset.write_bytes(b"synthetic actual model evidence; geometry tests replace this with a real GLB")
        self.source = self.root / "synthetic_current_prediction_observation.json"
        write(self.source, {"basis": "synthetic current observation for CPU regression; no production building"})
        self.role_evidence = self.root / "synthetic_actual_role_selection.json"
        write(self.role_evidence, {"selection": "two distinct labeled synthetic actual mesh regions"})
        footprint = box(94, -21.5, 106, -18.5)
        world = np.eye(4)
        world[:2, 3] = footprint.centroid.coords[0]
        self.building = {"building_id": "synthetic_source_roles", "footprint_world": footprint.__geo_interface__,
            "base_z_m": 2., "top_z_m": 8., "height_m": 6.,
            "world_from_local_4x4": world.tolist(), "local_from_world_4x4": np.linalg.inv(world).tolist()}
        self.building_path = self.root / "current_building.json"
        write(self.building_path, self.building)
        self.building = json.loads(self.building_path.read_text())
        self.prior = np.array([[1., 0, 0, 100], [0, 0, -1., -20], [0, 1., 0, 2.], [0, 0, 0, 1.]])
        self.model_points = [[-4., 6., 0.], [4., 6., 0.]]
        self.cue_path = self.root / "reviewed_orientation_trial.json"

    def document(self, angle=90., model_points=None):
        model_points = self.model_points if model_points is None else model_points
        q = rotation(angle)
        anchors = []
        for i, model in enumerate(model_points):
            world = (self.prior @ np.r_[model, 1.])[:2]
            source = q[:2, :2] @ world + [50., -35.]
            anchors.append({"role_id": f"distinct_region_{i}", "status": "approved_correspondence",
                "source_evidence_kind": "qualified_predicted_region", "model_evidence_kind": "actual_mesh_region",
                "source_anchor_world_xy_m": source.tolist(), "model_anchor_asset_gltf_xyz": model,
                "source_evidence": [candidate.proof(self.source)], "model_evidence": [candidate.proof(self.role_evidence)]})
        return {"schema": "citycode.source_role_orientation_approval.v1", "status": "approved_for_initialization_trial",
            "building_id": self.building["building_id"], "source_basis": "current_prediction_observation",
            "selection_method": "directed_source_role_correspondence", "reviewer": "isolated_cpu_fixture",
            "ground_truth_or_oracle_used": False, "target_iou_used_to_choose_orientation": False,
            "coordinate_systems": {"source": "UE world XY metres Z-up", "model": "default-scene glTF XYZ Y-up including node transforms"},
            "current_building": candidate.proof(self.building_path), "actual_model": candidate.proof(self.asset),
            "original_prior_sha256": candidate.matrix_identity(self.prior), "anchors": anchors}

    def initialize(self, document=None, approved_sha=None):
        write(self.cue_path, self.document() if document is None else document)
        return candidate.source_supported_yaw_initialization(self.prior, self.building, self.cue_path,
            approved_cue_sha256=approved_sha or candidate.sha256_file(self.cue_path), evidence_root=self.root)

    def test_directed_two_roles_resolve_large_and_180_degree_rotations(self):
        for angle in [90., -129.69054631013614, 180., -179.9]:
            with self.subTest(angle=angle):
                result = self.initialize(self.document(angle))
                self.assertAlmostEqual(abs(result["initial_yaw_correction_degrees"]), abs(angle), places=10)
                self.assertFalse(result["original_prior_yaw_bound_satisfied_claimed"])

    def test_rigid_change_preserves_scale_vertical_handedness_normals_and_closure(self):
        self.prior[:3, :3] = self.prior[:3, :3] @ np.diag([-2., 4., 3.])
        result = self.initialize(self.document(77.42300114720177))
        corrected = np.asarray(result["source_supported_prior_world_from_asset_4x4"])
        delta = np.asarray(result["rigid_world_correction_4x4"])
        np.testing.assert_allclose(delta @ self.prior, corrected, atol=1e-12)
        np.testing.assert_array_equal(self.prior[2], corrected[2])
        np.testing.assert_allclose(np.linalg.svd(self.prior[:3, :3])[1], np.linalg.svd(corrected[:3, :3])[1], atol=1e-12)
        self.assertEqual(np.sign(np.linalg.det(self.prior[:3, :3])), np.sign(np.linalg.det(corrected[:3, :3])))
        center = np.r_[self.building["world_from_local_4x4"][0][3], self.building["world_from_local_4x4"][1][3], 14., 1.]
        np.testing.assert_allclose(delta @ center, center, atol=1e-12)
        np.testing.assert_allclose(np.asarray(result["asset_from_world_4x4"]) @ corrected, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(np.asarray(result["world_from_building_local_4x4"]) @ np.asarray(result["building_local_from_asset_4x4"]), corrected, atol=1e-12)
        np.testing.assert_allclose(result["normal_asset_to_world_3x3"], np.linalg.inv(corrected[:3, :3]).T, atol=1e-12)
        self.assertLess(result["local_world_closure_error"], 1e-10)

    def test_three_directed_noncollinear_anchors_use_rotation_without_scaling(self):
        result = self.initialize(self.document(-43., [[-4., 6., 0.], [4., 6., 0.], [1., 4., -2.]]))
        self.assertAlmostEqual(result["initial_yaw_correction_degrees"], -43., places=10)
        np.testing.assert_allclose(result["anchor_centered_residual_m"], 0., atol=1e-12)

    def test_coarse_or_unapproved_cues_cannot_enter_trial(self):
        for change in [{"status": "coarse_role_cues_for_independent_initializer_review"}, {"reviewer": ""},
                       {"selection_method": "footprint_iou_maximum"}, {"source_basis": "GT"},
                       {"ground_truth_or_oracle_used": True}, {"target_iou_used_to_choose_orientation": True}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                document = self.document(); document.update(change); self.initialize(document)
        with self.assertRaisesRegex(ValueError, "approved cue SHA"):
            self.initialize(approved_sha="0" * 64)

    def test_each_anchor_needs_approved_actual_source_and_model_evidence(self):
        for key, value in [("status", "pending"), ("source_evidence", []), ("model_evidence", []),
                           ("source_evidence_kind", "oracle_landmark"), ("model_evidence_kind", "imagined_roof")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                doc = self.document(); doc["anchors"][0][key] = value; self.initialize(doc)

    def test_degenerate_or_duplicate_directed_roles_are_rejected(self):
        for kind in ["one", "source_duplicate", "model_duplicate", "role_duplicate"]:
            doc = self.document()
            if kind == "one": doc["anchors"] = doc["anchors"][:1]
            elif kind == "source_duplicate": doc["anchors"][1]["source_anchor_world_xy_m"] = doc["anchors"][0]["source_anchor_world_xy_m"]
            elif kind == "model_duplicate": doc["anchors"][1]["model_anchor_asset_gltf_xyz"] = doc["anchors"][0]["model_anchor_asset_gltf_xyz"]
            else: doc["anchors"][1]["role_id"] = doc["anchors"][0]["role_id"]
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.initialize(doc)

    def test_reflected_role_mapping_cannot_be_repaired_with_a_rotation(self):
        doc = self.document(0., [[-4., 6., 0.], [4., 6., 0.], [1., 4., -2.]])
        for row in doc["anchors"]: row["source_anchor_world_xy_m"][0] *= -1
        with self.assertRaisesRegex(ValueError, "Reflected"):
            self.initialize(doc)

    def test_nonfinite_singular_tilted_prior_and_nonfinite_anchor_fail(self):
        for bad in [np.full((4, 4), np.nan), np.zeros((4, 4)), np.eye(4)]:
            self.setUp_for_finite_prior()
            with self.subTest(prior=bad.tolist()), self.assertRaises(ValueError):
                doc = self.document(); self.prior = bad; self.initialize(doc)
        self.setUp_for_finite_prior()
        self.prior[0, 1] = .2
        with self.assertRaisesRegex(ValueError, "without tilt"):
            self.initialize(self.document())
        self.setUp_for_finite_prior()
        doc = self.document(); doc["anchors"][0]["source_anchor_world_xy_m"][0] = float("nan")
        # JSON NaN itself is rejected at serialization; load a deliberately bad
        # external document to exercise the actual numeric input guard.
        self.cue_path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            candidate.source_supported_yaw_initialization(self.prior, self.building, self.cue_path,
                approved_cue_sha256=candidate.sha256_file(self.cue_path), evidence_root=self.root)

    def setUp_for_finite_prior(self):
        self.prior = np.array([[1., 0, 0, 100], [0, 0, -1., -20], [0, 1., 0, 2.], [0, 0, 0, 1.]])

    def test_stale_evidence_changed_structure_or_another_prior_is_rejected(self):
        doc = self.document(); self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Evidence bytes changed"): self.initialize(doc)
        doc = self.document(); doc["original_prior_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "another raw initial prior"): self.initialize(doc)
        doc = self.document(); self.building["height_m"] += 1.
        with self.assertRaisesRegex(ValueError, "different current structure"): self.initialize(doc)

    def real_glb(self):
        # Reuse the package's established fixture writer (POSITION/NORMAL/UV,
        # material/texture and transformed active node), not any real building.
        import sys
        sys.path.insert(0, str(PACKAGE / "tests"))
        sys.path.insert(0, str(PACKAGE / "src"))
        from test_asset_delivery_contract import cube_glb
        cube_glb(self.asset, nodes=[{"mesh": 0, "scale": [12., 6., 3.], "translation": [-6., 0., -1.5]}])

    def bounded_trial(self, registration_path=PUBLIC_REGISTRATION):
        self.real_glb()
        self.prior = rotation(90.) @ self.prior
        # Original yaw rotation is around world origin; reset translation to
        # its actual input prior. The initializer rotates around target center.
        self.prior[:2, 3] = [100., -20.]
        doc = self.document(-90.)
        for row in doc["anchors"]:
            x, _, z = row["model_anchor_asset_gltf_xyz"]
            row["source_anchor_world_xy_m"] = [100. + x, -20. - z]
        write(self.cue_path, doc)
        before = self.asset.read_bytes()
        original = candidate.load_registration(registration_path, candidate.sha256_file(registration_path))
        old = original.fit_predicted_asset(self.asset, self.building, self.prior,
            expected_source_sha256=candidate.sha256_file(self.asset), policy=original.FitPolicy())
        trial = candidate.fit_with_source_orientation(self.asset, self.building, self.prior, self.cue_path,
            approved_cue_sha256=candidate.sha256_file(self.cue_path), evidence_root=self.root,
            registration_path=registration_path, expected_registration_sha256=candidate.sha256_file(registration_path))
        self.assertEqual(self.asset.read_bytes(), before)
        return old, trial

    def test_real_scene_glb_new_initialization_passes_original_bounded_fit(self):
        for registration_path in [PUBLIC_REGISTRATION]:
            self.setUp_for_finite_prior()
            with self.subTest(registration=registration_path):
                old, trial = self.bounded_trial(registration_path)
                self.assertEqual(old["status"], "review_required")
                self.assertEqual(trial["numeric_status"], "accepted", trial["bounded_fit"]["reasons"])
                fitted = trial["bounded_fit"]
                self.assertEqual(fitted["policy"], candidate.ORIGINAL_POLICY)
                self.assertLessEqual(abs(fitted["fit"]["yaw_change_degrees"]), 10. + 1e-8)
                self.assertAlmostEqual(trial["initialization"]["initial_yaw_correction_degrees"], -90., places=10)
                self.assertFalse(trial["original_raw_prior_yaw_bound_satisfied_claimed"])
                self.assertFalse(trial["release_approved"])

    def test_actual_height_failure_is_preserved_not_promoted(self):
        self.building.update(top_z_m=14., height_m=12.)
        write(self.building_path, self.building)
        _, trial = self.bounded_trial()
        self.assertEqual(trial["numeric_status"], "review_required")
        self.assertIn("height_scale_change_exceeds_bound", trial["bounded_fit"]["reasons"])
        self.assertFalse(trial["release_approved"])

    def test_impossible_xy_coverage_is_preserved_not_promoted(self):
        self.building["footprint_world"] = box(40., -35., 160., -5.).__geo_interface__
        write(self.building_path, self.building)
        self.building = json.loads(self.building_path.read_text())
        _, trial = self.bounded_trial()
        self.assertEqual(trial["numeric_status"], "review_required")
        self.assertIn("actual_mesh_target_coverage_below_threshold", trial["bounded_fit"]["reasons"])
        self.assertEqual(trial["bounded_fit"]["policy"], candidate.ORIGINAL_POLICY)

    def test_registration_source_sha_and_original_policy_are_required(self):
        with self.assertRaisesRegex(ValueError, "registration source SHA"):
            candidate.load_registration(PUBLIC_REGISTRATION, "0" * 64)
        bad = self.root / "relaxed_policy.py"
        original_text = PUBLIC_REGISTRATION.read_text()
        bad.write_text(original_text.replace("maximum_yaw_change_degrees: float = 10.0", "maximum_yaw_change_degrees: float = 180.0"))
        with self.assertRaisesRegex(ValueError, "must not be relaxed"):
            candidate.load_registration(bad, candidate.sha256_file(bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
