"""Native conditioning must preserve source identity and actual provider bytes."""
import base64
import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import numpy as np
from PIL import Image

from citycode_assets import appearance, provider
from citycode_assets.cli import main


class AppearanceProvider(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="appearance_provider_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "appearance"
        (self.directory / "inputs").mkdir(parents=True)
        refs = []
        for index, name in enumerate(provider.INPUT_NAMES):
            path = self.directory / "inputs" / name
            Image.new("RGBA" if index < 3 else "RGB", (16 + index, 20 + index), "white").save(path, format="PNG")
            with Image.open(path) as image:
                refs.append({"path": "inputs/" + name, "sha256": appearance.sha256_file(path),
                             "mode": image.mode, "size": list(image.size)})
        prompt = "Reconstruct this current building; predicted height 12 metres."
        (self.directory / "prompt.txt").write_text(prompt + "\n")
        self.record = {"schema": "citycode.appearance_input.v1", "status": "prepared", "building_id": "current_A",
                       "input_identity_sha256": "a" * 64, "inputs": refs, "prompt": "prompt.txt",
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "model": provider.MODEL,
                       "quality": provider.QUALITY, "size": "1024x1024", "transport": provider.transport_for()}
        appearance.write_json(self.directory / "appearance_input.json", self.record)
        image = io.BytesIO()
        Image.new("RGB", (1024, 1024), "white").save(image, format="PNG")
        self.output = image.getvalue()
        self.response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{
            "inlineData": {"mimeType": "image/png", "data": base64.b64encode(self.output).decode()}}]}}],
                         "usageMetadata": {"totalTokenCount": 4000}}

    def complete(self):
        return appearance.complete_building(self.directory)

    def test_exact_four_reference_native_body_and_legacy_identity(self):
        wire, metadata = provider.build_native_request([self.directory / r["path"] for r in self.record["inputs"]],
                                                        (self.directory / "prompt.txt").read_text(), root=self.directory)
        body = json.loads(wire)
        parts = body["contents"][0]["parts"]
        self.assertEqual(body["contents"][0]["role"], "user")
        for row, part in zip(self.record["inputs"], parts[:4]):
            self.assertEqual(base64.b64decode(part["inlineData"]["data"]), (self.directory / row["path"]).read_bytes())
        self.assertEqual(parts[4]["text"], (self.directory / "prompt.txt").read_text().strip())
        self.assertEqual(body["generationConfig"], provider.GENERATION_CONFIG)
        self.assertEqual(metadata["actual_http_request_sha256"], hashlib.sha256(wire).hexdigest())
        self.assertNotIn('"data":', json.dumps(metadata))
        legacy = {k: v for k, v in self.record.items() if k != "transport"}
        legacy.update(model="gpt-image-2", quality="medium")
        value = {k: legacy[k] for k in ("model", "quality", "size", "prompt_sha256")}
        value["input_sha256"] = [r["sha256"] for r in legacy["inputs"]]
        self.assertEqual(appearance.completion_contract(legacy), hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        changed = copy.deepcopy(self.record)
        changed["transport"]["provider_host"] = "another.example"
        self.assertNotEqual(appearance.completion_contract(changed), appearance.completion_contract(self.record))

    def test_explicit_https_configuration_and_no_redirects(self):
        config = provider.provider_config()
        for url in ("http://example.test/v1", "https://user:secret@example.test/v1", "https://example.test/v1?key=secret"):
            config["base_url"] = url
            with self.assertRaises(ValueError):
                provider.provider_config(config)
        config = provider.provider_config()
        config["api_key"] = "must-not-be-stored"
        with self.assertRaises(ValueError):
            provider.provider_config(config)
        config = provider.provider_config()
        config["model"] = "gemini-3.1-flash-image-preview"
        self.assertEqual(provider.transport_for(config)["endpoint_path"], "/v1beta/models/gemini-3.1-flash-image-preview:generateContent")
        self.assertEqual(provider.DEFAULT_CONFIG["model"], "gemini-3-pro-image-preview")
        config["model"] = "../another/route"
        with self.assertRaises(ValueError):
            provider.provider_config(config)
        with self.assertRaisesRegex(ValueError, "redirects"):
            provider._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://another.example")

    def test_complete_binds_all_receipts_and_cache_checks_actual_inputs(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(provider, "_send_native", return_value=(self.response, 200)) as send:
            receipt = self.complete()
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual((self.directory / "generated_main_oblique.png").read_bytes(), self.output)
            for name, field in (("request.json", "request_sha256"), ("provider_request.json", "provider_request_sha256"),
                                ("provider_response.json", "provider_response_sha256")):
                self.assertEqual(receipt[field], appearance.sha256_file(self.directory / name))
            self.assertEqual(self.complete(), receipt)
            self.assertEqual(send.call_count, 1)
            path = self.directory / self.record["inputs"][1]["path"]
            Image.new("RGBA", (17, 21), "black").save(path, format="PNG")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.complete()
            self.assertEqual(send.call_count, 1)
        self.assertNotIn("fixture-secret", "".join(p.read_text() for p in self.directory.glob("*.json")))

    def test_http_transport_uses_only_configured_host_native_header_and_finite_timeout(self):
        config = provider.provider_config()
        config.update(base_url="https://configured.example/v1", timeout_seconds=123)
        test = self
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(test.response).encode()
        class Opener:
            def open(self, request, timeout):
                test.assertEqual(request.full_url, "https://configured.example/v1beta/models/gemini-3-pro-image-preview:generateContent")
                test.assertEqual(request.get_header("X-goog-api-key"), "fixture-secret")
                test.assertEqual(request.get_header("Content-type"), "application/json")
                test.assertEqual(request.data, b"exact-wire-fixture")
                test.assertEqual(timeout, 123)
                return Response()
        with patch.object(provider.urllib.request, "build_opener", return_value=Opener()) as build:
            body, status = provider._send_native(b"exact-wire-fixture", config, "fixture-secret")
        self.assertEqual(status, 200)
        self.assertEqual(body, self.response)
        self.assertIsInstance(build.call_args.args[0], provider._NoRedirect)

    def test_cached_prompt_and_same_size_receipt_corruption_are_rejected(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), patch.object(provider, "_send_native", return_value=(self.response, 200)) as send:
            self.complete()
            original = (self.directory / "prompt.txt").read_bytes()
            (self.directory / "prompt.txt").write_text("different building height")
            with self.assertRaisesRegex(ValueError, "Prompt changed"):
                self.complete()
            (self.directory / "prompt.txt").write_bytes(original)
            path = self.directory / "provider_response.json"
            data = bytearray(path.read_bytes()); data[-2] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                self.complete()
            self.assertEqual(send.call_count, 1)

    def test_legacy_copy_helper_cannot_create_incomplete_native_cache(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), patch.object(provider, "_send_native", return_value=(self.response, 200)):
            receipt = self.complete()
        target = self.root / "other_building"
        shutil.copytree(self.directory / "inputs", target / "inputs")
        shutil.copyfile(self.directory / "prompt.txt", target / "prompt.txt")
        record = copy.deepcopy(self.record)
        record.update(building_id="current_B", input_identity_sha256="b" * 64)
        appearance.write_json(target / "appearance_input.json", record)
        self.assertFalse(appearance.reuse_completed_image(target, self.directory))
        self.assertFalse((target / "completion.json").exists())
        self.assertFalse((target / "generated_main_oblique.png").exists())
        self.assertEqual(json.loads((self.directory / "completion.json").read_text()), receipt)

    def test_wrong_size_multiple_images_and_thought_only_are_rejected(self):
        small = io.BytesIO(); Image.new("RGB", (512, 512)).save(small, format="PNG")
        bad = copy.deepcopy(self.response)
        bad["candidates"][0]["content"]["parts"][0]["inlineData"]["data"] = base64.b64encode(small.getvalue()).decode()
        with self.assertRaisesRegex(ValueError, "already be an actual"):
            provider.decode_native_response(bad)
        bad = copy.deepcopy(self.response)
        bad["candidates"][0]["content"]["parts"] *= 2
        with self.assertRaisesRegex(ValueError, "exactly one"):
            provider.decode_native_response(bad)
        bad["candidates"][0]["content"]["parts"] = [bad["candidates"][0]["content"]["parts"][0]]
        bad["candidates"][0]["content"]["parts"][0]["thought"] = True
        with self.assertRaisesRegex(ValueError, "exactly one"):
            provider.decode_native_response(bad)

    def test_jpeg_keeps_original_payload_and_exact_decoded_rgb_pixels(self):
        yy, xx = np.indices((1024, 1024))
        pixels = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
        original = io.BytesIO()
        Image.fromarray(pixels).save(original, format="JPEG", quality=91)
        payload = original.getvalue()
        response = copy.deepcopy(self.response)
        response["candidates"][0]["content"]["parts"][0]["inlineData"] = {
            "mimeType": "image/jpeg", "data": base64.b64encode(payload).decode()}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(provider, "_send_native", return_value=(response, 200)) as send:
            receipt = self.complete()
            proof = receipt["payload_provenance"]
            original_path = self.directory / proof["original_payload"]["path"]
            self.assertEqual(original_path.read_bytes(), payload)
            self.assertEqual(proof["original_payload"]["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(proof["conversion"]["operation"], "jpeg_decode_to_png")
            self.assertFalse(proof["conversion"]["resized"])
            self.assertFalse(proof["conversion"]["edited"])
            self.assertFalse(proof["conversion"]["orientation_transformed"])
            with Image.open(io.BytesIO(payload)) as source, Image.open(self.directory / receipt["output"]) as target:
                self.assertEqual(target.format, "PNG")
                self.assertEqual(target.mode, "RGB")
                self.assertEqual(target.size, (1024, 1024))
                self.assertEqual(source.convert("RGB").tobytes(), target.tobytes())
                self.assertEqual(proof["conversion"]["source_pixel_sha256"], hashlib.sha256(target.tobytes()).hexdigest())
            self.assertEqual(proof["conversion"]["source_pixel_sha256"], proof["conversion"]["canonical_pixel_sha256"])
            saved = json.loads((self.directory / "provider_response.json").read_text())
            self.assertEqual(saved["payload_provenance"], proof)
            self.assertEqual(saved["candidates"][0]["parts"][0]["image_payload_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(self.complete(), receipt)
            self.assertEqual(send.call_count, 1)
            original_path.write_bytes(payload[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "payload|proof"):
                self.complete()
            self.assertEqual(send.call_count, 1)

    def test_png_preserves_original_bytes_and_alpha_without_reencoding(self):
        stream = io.BytesIO()
        Image.new("RGBA", (1024, 1024), (20, 70, 90, 31)).save(stream, format="PNG")
        payload = stream.getvalue()
        response = copy.deepcopy(self.response)
        response["candidates"][0]["content"]["parts"][0]["inlineData"]["data"] = base64.b64encode(payload).decode()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(provider, "_send_native", return_value=(response, 200)):
            receipt = self.complete()
        self.assertEqual((self.directory / receipt["output"]).read_bytes(), payload)
        proof = receipt["payload_provenance"]
        self.assertEqual((self.directory / proof["original_payload"]["path"]).read_bytes(), payload)
        self.assertEqual(proof["conversion"]["operation"], "none_original_png")
        self.assertEqual(proof["conversion"]["pixel_mode"], "RGBA")
        self.assertEqual(proof["original_payload"]["sha256"], proof["canonical_png"]["sha256"])

    def test_jpeg_cache_uses_historical_proof_and_pixels_across_png_encoders(self):
        stream = io.BytesIO()
        Image.new("RGB", (1024, 1024), (6, 12, 18)).save(stream, format="JPEG")
        response = copy.deepcopy(self.response)
        response["candidates"][0]["content"]["parts"][0]["inlineData"] = {
            "mimeType": "image/jpeg", "data": base64.b64encode(stream.getvalue()).decode()}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(provider, "_send_native", return_value=(response, 200)) as send:
            receipt = self.complete()
            output = self.directory / receipt["output"]
            original_encoded = output.read_bytes()
            with Image.open(output) as opened:
                pixels = opened.copy()
            compressed_differently = io.BytesIO()
            pixels.save(compressed_differently, format="PNG", compress_level=0)
            self.assertNotEqual(original_encoded, compressed_differently.getvalue())
            historical_version = "fixture-historical-pillow"

            def save_historical_cache(data):
                output.write_bytes(data)
                proof = receipt["payload_provenance"]
                proof["canonical_png"].update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                proof["conversion"]["pillow_version"] = historical_version
                receipt["output_sha256"] = hashlib.sha256(data).hexdigest()
                saved = json.loads((self.directory / "provider_response.json").read_text())
                saved.update(image_sha256=receipt["output_sha256"], image_bytes=len(data), payload_provenance=proof)
                appearance.write_json(self.directory / "provider_response.json", saved)
                receipt["provider_response_sha256"] = appearance.sha256_file(self.directory / "provider_response.json")
                appearance.write_json(self.directory / "completion.json", receipt)

            save_historical_cache(compressed_differently.getvalue())
            with patch.object(provider, "PILLOW_VERSION", "fixture-newer-pillow"), \
                    patch.object(provider, "_normalize_payload", side_effect=AssertionError("Cache must not re-encode")):
                self.assertEqual(self.complete(), receipt)
            self.assertEqual(receipt["payload_provenance"]["conversion"]["pillow_version"], historical_version)
            self.assertEqual(send.call_count, 1)
            value = pixels.getpixel((10, 10))
            pixels.putpixel((10, 10), (value[0] ^ 1, value[1], value[2]))
            changed = io.BytesIO()
            pixels.save(changed, format="PNG")
            # Refresh file hashes so the negative case specifically exercises
            # decoded-pixel verification rather than an earlier file-SHA check.
            save_historical_cache(changed.getvalue())
            with self.assertRaisesRegex(ValueError, "pixel"):
                self.complete()
            self.assertEqual(send.call_count, 1)

    def test_rejected_images_preserve_actual_payloads_and_http200_diagnostics(self):
        stream = io.BytesIO()
        Image.new("RGB", (1024, 512), "orange").save(stream, format="JPEG")
        cases = []
        small = copy.deepcopy(self.response)
        small["candidates"][0]["content"]["parts"][0]["inlineData"] = {
            "mimeType": "image/jpeg", "data": base64.b64encode(stream.getvalue()).decode()}
        cases.append((small, "1024"))
        multiple = copy.deepcopy(self.response)
        multiple["candidates"][0]["content"]["parts"] *= 2
        cases.append((multiple, "exactly one"))
        mismatched = copy.deepcopy(self.response)
        mismatched["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"] = "image/jpeg"
        cases.append((mismatched, "MIME"))
        for response, message in cases:
            with self.subTest(message=message), patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), \
                    patch.object(provider, "_send_native", return_value=(response, 200)) as send:
                with self.assertRaises(RuntimeError):
                    self.complete()
                self.assertEqual(send.call_count, 1)
                saved = json.loads((self.directory / "provider_response.json").read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["status_code"], 200)
                self.assertIn(message, saved["validation_error"])
                self.assertFalse((self.directory / "generated_main_oblique.png").exists())
                for part in saved["candidates"][0]["parts"]:
                    retained = part["original_payload"]
                    path = self.directory / retained["path"]
                    self.assertEqual(appearance.sha256_file(path), retained["sha256"])
                    self.assertEqual(path.stat().st_size, retained["bytes"])
                self.assertNotIn('"data":', json.dumps(saved))

    def test_only_one_non_thought_payload_is_selected(self):
        response = copy.deepcopy(self.response)
        thought = copy.deepcopy(response["candidates"][0]["content"]["parts"][0])
        thought["thought"] = True
        response["candidates"][0]["content"]["parts"].insert(0, thought)
        data, metadata = provider.decode_native_response(response)
        self.assertEqual(data, self.output)
        self.assertEqual(metadata["non_thought_image_count"], 1)
        self.assertEqual(len(metadata["candidates"][0]["parts"]), 2)

    def test_explicit_cli_and_single_http_failure_keep_diagnostics_without_secrets(self):
        error = urllib.error.HTTPError("https://example.test", 503, "unavailable", {}, io.BytesIO(json.dumps({
            "error": {"type": "new_api_error", "code": "system_memory_overloaded",
                      "message": "system memory overloaded fixture-secret " + "A" * 150}}).encode()))
        output = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-secret"}), \
                patch.object(provider, "_send_native", side_effect=error) as send, contextlib.redirect_stdout(output):
            status = main(["complete-appearance", "--appearance", str(self.directory)])
        self.assertEqual(status, 1)
        self.assertEqual(send.call_count, 1)
        response = json.loads((self.directory / "provider_response.json").read_text())
        self.assertEqual(response["status_code"], 503)
        self.assertEqual(response["provider_error"]["code"], "system_memory_overloaded")
        self.assertFalse((self.directory / "generated_main_oblique.png").exists())
        public_text = output.getvalue() + "".join(p.read_text() for p in self.directory.glob("*.json"))
        self.assertNotIn("fixture-secret", public_text)
        self.assertNotIn("A" * 100, public_text)

    def test_prepare_uses_package_native_provider_without_network(self):
        source = self.root / "structure/current_A"
        observation = source / "observations/observed_A"
        observation.mkdir(parents=True)
        for name in ("source_primary", "roof_evidence", "context"):
            Image.new("RGB" if name == "context" else "RGBA", (32, 24), "white").save(observation / (name + ".png"))
        appearance.write_json(observation / "observation.json", {"building_id": "current_A", "observation_id": "observed_A",
                              "paths": {n: n + ".png" for n in ("source_primary", "roof_evidence", "context")}})
        (source / "proxy.glb").write_bytes(b"renderer input fixture")
        appearance.write_json(source / "building.json", {"schema": "citycode.building_generation_input.v1", "status": "complete",
            "building_id": "current_A", "height_m": 12.0, "canonical_observation_id": "observed_A",
            "footprint_world": {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 6], [0, 6], [0, 0]]]},
            "observations": [{"observation_id": "observed_A", "path": "observations/observed_A"}],
            "paths": {"proxy_local_ue_z_up": "proxy.glb"}, "world_from_local_4x4": np.eye(4).tolist()})
        def render(mesh, destination, **kwargs):
            Image.new("RGBA", (1024, 1024), "white").save(destination)
        with patch.object(appearance, "render_spatial_proxy", side_effect=render) as renderer, \
                patch.object(provider, "_send_native", side_effect=AssertionError("Preparation cannot send images")):
            record = appearance.prepare_building(source / "building.json", self.root / "prepared")
            self.assertEqual(record["model"], provider.MODEL)
            self.assertEqual(record["quality"], "native_1k")
            self.assertEqual(record["transport"], provider.transport_for())
            self.assertEqual(appearance.prepare_building(source / "building.json", self.root / "prepared"), record)
            self.assertEqual(renderer.call_count, 1)


if __name__ == "__main__":
    unittest.main()
