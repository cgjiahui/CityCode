"""Delivery metadata keeps evidence and resumable identities without local paths."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from citycode_assets.metadata import sanitize_delivery_metadata, sanitize_completed_sidecars


class AssetMetadataContract(unittest.TestCase):
    def result(self):
        return {"building_id": "current_A", "old_asset_id": "historic_B", "status": "completed",
                "source_sha256": "a" * 64, "raw_cache_sha256": "b" * 64, "code_sha256": "c" * 64,
                "started_utc": "2026-10-15T01:00:00Z", "completed_utc": "2026-10-15T01:01:00Z",
                "elapsed_seconds": 60, "binding_requires_final_fit_acceptance": True,
                "lods": {label: {"path": f"assets/current_A/{name}", "sha256": h * 64,
                                  "faces": faces, "bytes": faces,
                                  "encoded_normals_and_uv": True}
                         for label, name, h, faces in [("0.25M", "model_025m.glb", "d", 250000),
                                                     ("0.1M", "model_01m.glb", "e", 100000)]},
                "source": {"building_id": "current_A", "old_asset_id": "historic_B",
                           "source_glb": "/mnt/private/model.glb", "source_sha256": "a" * 64,
                           "original_source_glb": "C:\\private\\old.glb", "original_source_sha256": "f" * 64,
                           "raw_cache_path": "/mnt/private/raw_mesh.pt", "raw_cache_sha256": "b" * 64,
                           "fit_record": "relative/private/fit.json", "fit_record_sha256": "1" * 64,
                           "normal_repair_applied": True, "status": "accepted", "reasons": [],
                           "fit": {"actual_mesh_overlap": {"iou": .91, "target_coverage": .96},
                                   "debug_locator": "Temporary /tmp/private/cache.pt was used",
                                   "/mnt/private/key.json": "not for delivery"},
                           "policy": {"minimum_footprint_iou": .70},
                           "world_from_asset_4x4": [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]]}}

    def test_no_private_paths_and_complete_cache_contract_is_preserved(self):
        original = self.result(); before = copy.deepcopy(original)
        cleaned = sanitize_delivery_metadata(original)
        self.assertEqual(original, before)
        text = json.dumps(cleaned)
        for fragment in ("/mnt/private", "C:\\\\private", "relative/private", "/tmp/private"):
            self.assertNotIn(fragment, text)
        for key in ("source_sha256", "raw_cache_sha256", "code_sha256", "lods", "status", "elapsed_seconds"):
            self.assertEqual(cleaned[key], original[key])
        self.assertEqual(cleaned["source"]["original_source_sha256"], "f" * 64)
        self.assertTrue(cleaned["source"]["normal_repair_applied"])
        self.assertEqual(cleaned["registration"]["fit"]["actual_mesh_overlap"]["iou"], .91)
        self.assertEqual(cleaned["registration"]["world_from_asset_4x4"], original["source"]["world_from_asset_4x4"])
        self.assertTrue(cleaned["binding_requires_final_fit_acceptance"])
        self.assertEqual(sanitize_delivery_metadata(cleaned), cleaned)

    def test_absolute_owned_lod_path_is_rejected_without_silent_loss(self):
        original = self.result(); original["lods"]["0.1M"]["path"] = "/mnt/private/model_01m.glb"
        with self.assertRaisesRegex(ValueError, "owned relative"):
            sanitize_delivery_metadata(original)

    def test_batch_changes_only_completed_public_sidecars_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            completed = root / "assets/current_A/lod_manifest.json"
            running = root / "assets/current_B/lod_manifest.json"
            research = root / "private/records/current_A.json"
            for path in (completed, running, research):
                path.parent.mkdir(parents=True, exist_ok=True)
            result = self.result(); completed.write_text(json.dumps(result))
            pending = copy.deepcopy(result); pending["status"] = "running"
            running.write_text(json.dumps(pending)); research.write_text(json.dumps(result))
            model = completed.parent / "model_025m.glb"; model.write_bytes(b"unchanged mesh")
            before = {path: path.read_bytes() for path in (running, research, model)}
            report = sanitize_completed_sidecars(root)
            self.assertEqual(report["changed"], 1)
            self.assertEqual(report["skipped_incomplete_or_changed"], 1)
            for path, payload in before.items():
                self.assertEqual(path.read_bytes(), payload)
            report = sanitize_completed_sidecars(root)
            self.assertEqual(report["changed"], 0)
            self.assertEqual(report["unchanged"], 1)


if __name__ == "__main__":
    unittest.main()
