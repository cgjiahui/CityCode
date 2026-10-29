"""CPU regression tests for actual streams, stale recipes and bounded replay."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from citycode_assets import mesh_repair as repair
from citycode_assets.registration import read_glb_scene_geometry


def fixture(path, *, color=(40, 80, 120), missing=False, node_transform=False):
    vertices = np.array([[x, y, 0] for y in range(3) for x in range(3)], dtype="<f4")
    faces = []
    for y in range(2):
        for x in range(2):
            a = 3 * y + x
            faces.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]])
    faces = np.asarray(faces, dtype="<u4")
    if missing:
        faces = faces[:-1]
    normals = np.tile(np.array([0, 0, 1], dtype="<f4"), (len(vertices), 1))
    uv = (vertices[:, :2] * .005).astype("<f4")
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    payloads = [vertices.tobytes(), normals.tobytes(), uv.tobytes(), faces.tobytes(), buffer.getvalue()]
    binary, views = bytearray(), []
    for payload in payloads:
        binary.extend(b"\0" * (-len(binary) % 4))
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary.extend(payload)
    nodes = [{"mesh": 0}]
    if node_transform:
        nodes[0]["translation"] = [1, 0, 0]
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": nodes, "meshes": [{"primitives": [{"attributes": {
               "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
           "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
           "accessors": [{"bufferView": i, "componentType": kind, "count": count, "type": name}
                         for i, kind, count, name in [(0, 5126, 9, "VEC3"), (1, 5126, 9, "VEC3"),
                                                     (2, 5126, 9, "VEC2"), (3, 5125, faces.size, "SCALAR")]],
           "images": [{"bufferView": 4, "mimeType": "image/png"}],
           "textures": [{"source": 0, "sampler": 0}], "samplers": [{"wrapS": 33071, "wrapT": 33071}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                           "metallicFactor": 0., "roughnessFactor": .8}}]}
    repair._write(path, doc, binary)


def read_output(path):
    # The public input contract accepts one primitive; repaired outputs contain two.
    data = path.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + size])
    start = 28 + size
    return doc, data[start:]


class MeshRepair(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.high, self.low = self.root / "high.glb", self.root / "low.glb"
        fixture(self.high)
        fixture(self.low, color=(150, 90, 30), missing=True)
        self.matrix = np.eye(4)
        self.recipe = {"schema": repair.RECIPE_SCHEMA, "status": "reviewed_actual_triangle_selection",
                       "reviewer": "synthetic_cpu_regression", "building_id": "synthetic",
                       "inputs": {label: {k: repair._proof(p)[k] for k in ("sha256", "bytes")}
                                  for label, p in [("0.25M", self.high), ("0.1M", self.low)]},
                       "world_from_asset_4x4": self.matrix.tolist(), "low_endpoint_passes": [],
                       "donor_triangle_ids": {"0.25M": [0], "0.1M": [7]},
                       "face_budgets": {"0.25M": 250000, "0.1M": 100000},
                       "residual_openings": "Synthetic fixture; no watertightness assertion"}

    def replay(self, recipe=None, *, expected=None):
        plan = copy.deepcopy(self.recipe if recipe is None else recipe)
        sha = hashlib.sha256(repair._canonical(plan)).hexdigest() if expected is None else expected
        return repair.repair_lods(self.high, self.low, self.matrix, self.root / "result", plan,
                                  expected_recipe_sha256=sha)

    def test_actual_donor_triangle_uv_normal_image_and_sampler_survive_relocation(self):
        before = {p: p.read_bytes() for p in (self.high, self.low)}
        report = self.replay()
        for label, source, name in [("0.25M", self.high, "model_025m.glb"),
                                    ("0.1M", self.low, "model_01m.glb")]:
            out = self.root / "result" / name
            od, ob = repair._read(source)
            doc, binary = read_output(out)
            self.assertEqual(binary[:len(ob)], ob)
            self.assertEqual(doc["materials"][0], od["materials"][0])
            self.assertEqual(doc["textures"][1]["sampler"], 1)
            for index in range(3):
                np.testing.assert_array_equal(repair._accessor(doc, binary, index), repair._accessor(od, ob, index))
            scene = read_glb_scene_geometry(out)
            self.assertEqual(len(scene["faces"]), report["outputs"][label]["faces"])
        # The last high triangle fills the missing low triangle with its own PBR.
        low_doc, low_bin = read_output(self.root / "result/model_01m.glb")
        high_doc, high_bin = repair._read(self.high)
        patch = low_doc["meshes"][0]["primitives"][1]
        actual_pos = repair._accessor(low_doc, low_bin, patch["attributes"]["POSITION"])
        _, donor_faces = repair._arrays(high_doc, high_bin)
        expected = repair._accessor(high_doc, high_bin, 0)[np.unique(donor_faces[7])]
        np.testing.assert_array_equal(actual_pos, expected)
        for p, contents in before.items():
            self.assertEqual(p.read_bytes(), contents)
        self.assertFalse(report["release_approved"])
        self.assertFalse(report["watertightness_guaranteed"])
        self.assertFalse(report["pose_changed"])
        moved = self.root / "moved"
        (self.root / "result").rename(moved)
        self.assertEqual(len(read_glb_scene_geometry(moved / "model_01m.glb")["faces"]), 8)

    def test_stale_input_and_recipe_sha_are_rejected_before_output(self):
        with self.assertRaisesRegex(ValueError, "recipe SHA"):
            self.replay(expected="0" * 64)
        with self.low.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.replay()
        self.assertFalse((self.root / "result").exists())

    def test_recipe_path_requires_exact_file_bytes(self):
        p = self.root / "reviewed.json"
        p.write_text(json.dumps(self.recipe, indent=2) + "\n")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        result = repair.repair_lods(self.high, self.low, self.matrix, self.root / "result", p,
                                   expected_recipe_sha256=sha)
        self.assertEqual(result["recipe_sha256"], sha)
        self.assertEqual((self.root / "result/repair_recipe.json").read_bytes(), p.read_bytes())

    def test_report_input_paths_are_relative_and_relocate_with_the_input_package(self):
        report = self.replay()
        self.assertEqual(report["path_base"], "report_directory")
        for label, original in [("0.25M", self.high), ("0.1M", self.low)]:
            item = report["inputs"][label]
            self.assertFalse(Path(item["path"]).is_absolute())
            self.assertEqual((self.root / "result" / item["path"]).resolve(), original.resolve())
            self.assertEqual(item["sha256"], hashlib.sha256(original.read_bytes()).hexdigest())
            self.assertEqual(item["bytes"], original.stat().st_size)
        relocated = self.root / "relocated_package"
        shutil.copytree(self.root, relocated, ignore=shutil.ignore_patterns("relocated_package"))
        moved_report = json.loads((relocated / "result/repair_report.json").read_text())
        for item in moved_report["inputs"].values():
            resolved = relocated / "result" / item["path"]
            self.assertEqual(hashlib.sha256(resolved.read_bytes()).hexdigest(), item["sha256"])
        for item in moved_report["outputs"].values():
            resolved = relocated / "result" / item["path"]
            self.assertEqual(hashlib.sha256(resolved.read_bytes()).hexdigest(), item["sha256"])

    def test_unreviewed_invalid_and_duplicate_triangle_selections_fail(self):
        for changes in [("status", "pending"), ("reviewer", "")]:
            plan = copy.deepcopy(self.recipe)
            plan[changes[0]] = changes[1]
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.replay(plan)
        for ids in [[-1], [8], [1, 1], [1.2], [True], []]:
            plan = copy.deepcopy(self.recipe)
            plan["donor_triangle_ids"]["0.1M"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.replay(plan)
        self.assertFalse((self.root / "result").exists())

    def test_pose_and_transform_assumptions_cannot_silently_change(self):
        for value in [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3)]:
            with self.subTest(matrix=value.tolist()), self.assertRaises(ValueError):
                repair.repair_lods(self.high, self.low, value, self.root / "result", self.recipe,
                                  expected_recipe_sha256=hashlib.sha256(repair._canonical(self.recipe)).hexdigest())
        plan = copy.deepcopy(self.recipe)
        plan["world_from_asset_4x4"][0][3] = .1
        with self.assertRaisesRegex(ValueError, "original pose"):
            self.replay(plan)
        fixture(self.high, node_transform=True)
        self.recipe["inputs"]["0.25M"] = {k: repair._proof(self.high)[k] for k in ("sha256", "bytes")}
        with self.assertRaisesRegex(ValueError, "identity"):
            self.replay()

    def test_face_budget_cannot_be_raised_and_incomplete_result_is_not_installed(self):
        for budget in [100001, 7]:
            plan = copy.deepcopy(self.recipe)
            plan["face_budgets"]["0.1M"] = budget
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "face budget"):
                self.replay(plan)
        self.assertFalse((self.root / "result").exists())

    def test_replay_does_not_overwrite_existing_result(self):
        self.replay()
        with self.assertRaises(FileExistsError):
            self.replay()

    def test_endpoint_replay_preserves_streams_and_checks_chart_edges(self):
        doc, binary = repair._read(self.high)
        # Collapse a coplanar grid edge; surviving UVs exactly follow the plane.
        passes = [{"original_triangles": 8, "result_triangles": 6, "endpoint_collapses": [
            {"original_vertex_remap": {"4": 1}, "removed_triangles": 2}]}]
        changed, data, summary = repair._replay_low(doc, binary, passes, self.matrix)
        self.assertEqual(summary[0]["removed_triangles"], 2)
        self.assertEqual(bytes(data[:len(binary)]), binary)
        _, faces = repair._arrays(changed, data)
        self.assertEqual(len(faces), 6)
        self.assertFalse(np.any(faces == 4))
        for index in range(3):
            np.testing.assert_array_equal(repair._accessor(doc, binary, index),
                                          repair._accessor(changed, data, index))
        bad = copy.deepcopy(passes)
        bad[0]["endpoint_collapses"][0]["original_vertex_remap"] = {"0": 8}
        with self.assertRaisesRegex(ValueError, "edge neighbour"):
            repair._replay_low(doc, binary, bad, self.matrix)
        bad = copy.deepcopy(passes)
        bad[0]["endpoint_collapses"][0]["removed_triangles"] = 1
        with self.assertRaisesRegex(ValueError, "removed triangle count"):
            repair._replay_low(doc, binary, bad, self.matrix)

    def test_endpoint_replay_rejects_normal_seams_and_non_affine_UV(self):
        doc, binary = repair._read(self.high)
        passes = [{"original_triangles": 8, "result_triangles": 6, "endpoint_collapses": [
            {"original_vertex_remap": {"4": 1}, "removed_triangles": 2}]}]
        for index, replacement, expected in [(1, [.5, 0, .866], "normal/UV chart"),
                                              (2, [.02, .02], "UV-affine")]:
            data = bytearray(binary)
            spec = doc["accessors"][index]
            view = doc["bufferViews"][spec["bufferView"]]
            values = repair._accessor(doc, binary, index)
            values[4] = replacement
            start = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
            data[start:start + values.nbytes] = values.tobytes()
            with self.subTest(stream=index), self.assertRaisesRegex(ValueError, expected):
                repair._replay_low(doc, data, passes, self.matrix)


if __name__ == "__main__":
    unittest.main()
