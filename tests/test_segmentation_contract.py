"""Offline contracts: native masks, geometry reads and immutable provenance."""

import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from citycode_perception.segmentation import (
    STAGES, import_results, prepare_requests, run_vertex_requests,
)


class SegmentationContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.patch_id = "SC_P320_R+000_C+003"
        self.stem = self.patch_id + "__tilt_light"
        self.view = self.dataset / "patches" / self.patch_id / "views" / "tilt_light"
        self.view.mkdir(parents=True)
        Image.new("RGB", (64, 64), (85, 120, 140)).save(self.view / "rgb.png")
        Image.new("L", (64, 64), 255).save(self.view / "strict_valid_mask.png")
        (self.view / "camera.json").write_text(json.dumps({"image_size_px": [64, 64]}))
        (self.view.parents[1] / "patch.json").write_text("{}")
        (self.dataset / "dataset_manifest.json").write_text(json.dumps({"selected_views": ["tilt_light"]}))
        (self.dataset / "patch_index.json").write_text(json.dumps({"patches": [{"patch_id": self.patch_id}]}))

    def external_masks(self, *, size=(64, 64)):
        root = self.root / "external"
        for stage in STAGES:
            directory = root / stage / "results"
            directory.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", size, (0, 0, 0))
            color = (190, 60, 210) if stage == "structure_v9_2" else (0, 0, 255)
            for x in range(size[0] // 2, size[0]):
                for y in range(size[1]):
                    image.putpixel((x, y), color)
            image.save(directory / f"{self.stem}.png")
        return root

    def test_export_and_native_import_are_geometry_readable_without_network(self):
        output = self.root / "masks"
        with patch("citycode_perception.segmentation.urlopen", side_effect=AssertionError("network forbidden")):
            requests_path = prepare_requests(self.dataset, output, working_size=64)
            root_manifest = json.loads(requests_path.read_text())
            self.assertEqual(root_manifest["request_count"], 2)
            masks = import_results(self.dataset, self.external_masks(), output, working_size=64)
        for stage in STAGES:
            directory = masks / stage
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            # This is the exact id/result lookup consumed by the geometry reader.
            paths = {item["id"]: directory / item["result"] for item in manifest["samples"]}
            with Image.open(paths[self.stem]) as mask:
                self.assertEqual(mask.size, (1024, 1024))
                self.assertEqual(mask.mode, "RGB")
                self.assertEqual(mask.getpixel((511, 1)), (0, 0, 0))
                self.assertNotEqual(mask.getpixel((512, 1)), (0, 0, 0))
                self.assertEqual(len(mask.getcolors()), 2)  # no interpolated category colours
            imported = manifest["samples"][0]["import"]
            self.assertEqual(imported["image"]["resampling"], "NEAREST")
            self.assertTrue((directory / imported["raw"]).is_file())
        # Re-import of a normalized result reuses its original raw PNG.
        self.assertEqual(import_results(self.dataset, masks, masks, working_size=64), masks)

    def test_malformed_second_stage_never_produces_complete_output(self):
        external = self.external_masks()
        Image.new("RGB", (64, 32), (0, 0, 255)).save(external / "structure_v7b" / "results" / f"{self.stem}.png")
        target = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "dimensions"):
            import_results(self.dataset, external, target, working_size=64)
        self.assertFalse(target.exists())

    def test_prompt_or_source_change_cannot_reuse_existing_masks(self):
        target = self.root / "prepared"
        prepare_requests(self.dataset, target, working_size=64)
        Image.new("RGB", (64, 64), (12, 13, 14)).save(self.view / "rgb.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_requests(self.dataset, target, working_size=64)

    def test_external_manifest_source_hash_is_checked(self):
        external = self.external_masks()
        folder = external / "structure_v9_2"
        (folder / "manifest.json").write_text(json.dumps({"samples": [{"id": self.stem,
            "source_sha256": "0" * 64, "result": f"results/{self.stem}.png"}]}))
        with self.assertRaisesRegex(ValueError, "RGB mismatch"):
            import_results(self.dataset, external, self.root / "rejected", working_size=64)

    def test_optional_vertex_adapter_payload_is_offline_tested(self):
        output = self.root / "vertex"
        prepare_requests(self.dataset, output, working_size=64)
        image = BytesIO()
        Image.new("RGB", (1024, 1024), (0, 0, 255)).save(image, format="PNG")
        response = {"candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": "image/png", "data": base64.b64encode(image.getvalue()).decode()
        }}]}}], "responseId": "offline-fixture"}
        captured = []

        def fake_urlopen(request, *, timeout):
            captured.append(request)
            return BytesIO(json.dumps(response).encode())

        with patch.dict("os.environ", {"GOOGLE_OAUTH_ACCESS_TOKEN": "offline-fixture-token"}), \
                patch("citycode_perception.segmentation.urlopen", side_effect=fake_urlopen):
            summary = run_vertex_requests(output, project="test-project", location="global",
                                          model="test-image-model", max_requests=1)
        self.assertEqual(summary["completed_requests"], 1)
        payload = json.loads(captured[0].data)
        self.assertEqual(payload["generationConfig"]["imageConfig"]["imageSize"], "1K")
        self.assertTrue(captured[0].full_url.startswith("https://aiplatform.googleapis.com/"))
        receipt = (output / "structure_v9_2" / "api_metadata" / f"{self.stem}.json").read_text()
        self.assertNotIn("offline-fixture-token", receipt)


if __name__ == "__main__":
    unittest.main()
