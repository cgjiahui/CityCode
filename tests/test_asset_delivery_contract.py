"""Portable assets must prove both geometry variants and every source identity."""
import base64
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import numpy as np

from citycode_assets.cli import main, validate_assets


def cube_glb(path, *, offset_x=0.0, nodes=None, roots=None, triangle_repetitions=1):
    vertices = np.array([[x + offset_x, y, z] for x, y, z in
                         [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]], dtype="<f4")
    normals = vertices - np.array([.5 + offset_x, .5, .5], dtype="<f4")
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    uv = np.zeros((8, 2), dtype="<f4")
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                      [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype="<u2")
    faces = np.tile(faces, (triangle_repetitions, 1))
    image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg==")
    binary, views = bytearray(), []
    for payload in (vertices.tobytes(), normals.astype("<f4").tobytes(), uv.tobytes(), faces.tobytes(), image):
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary.extend(payload); binary.extend(b"\x00" * (-len(binary) % 4))
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0] if roots is None else roots}],
                "nodes": [{"mesh": 0}] if nodes is None else nodes,
                "meshes": [{"primitives": [{"attributes": {
                    "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
                "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
                "accessors": [{"bufferView": i, "componentType": component, "count": count, "type": kind}
                              for i, component, count, kind in [(0, 5126, 8, "VEC3"), (1, 5126, 8, "VEC3"),
                                                                (2, 5126, 8, "VEC2"), (3, 5123, faces.size, "SCALAR")]],
                "images": [{"bufferView": 4, "mimeType": "image/png"}], "textures": [{"source": 0}],
                "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]}
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    body = struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body)


class AssetDeliveryContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "package"
        self.root.mkdir()
        matrix = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        self.record = {"building_id": "current_A", "status": "accepted_reuse",
                       "local_coordinate_system": "glTF Y-up", "world_from_local_4x4": matrix,
                       "building_local_from_asset_4x4": copy.deepcopy(matrix),
                       "world_from_building_local_4x4": np.eye(4).tolist(),
                       "normal_transform_world_3x3": np.array(matrix)[:3, :3].tolist(),
                       "target_geometry": {"type": "Polygon", "coordinates": [[[0, -1], [1, -1], [1, 0], [0, 0], [0, -1]]]},
                       "base_z_m": 0, "top_z_m": 1, "lods": {}}
        for label, name in [("0.25M", "model_025m.glb"), ("0.1M", "model_01m.glb")]:
            path = self.root / "assets/current_A" / name
            cube_glb(path)
            self.record["lods"][label] = {"path": path.relative_to(self.root).as_posix(),
                                         "bytes": path.stat().st_size, "faces": 12,
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.manifest = {"schema": "citycode.building_asset_bindings.v1", "path_base": "manifest_directory",
                         "source_structure": {"instance_count": 1, "building_ids": ["current_A"],
                                              "manifest_sha256": "a" * 64,
                                              "evidence_package": "../independent_structure"},
                         "buildings": [self.record]}
        self.path = self.root / "bindingmanifest.json"

    def validate(self, *, complete=True, geometry=False):
        self.path.write_text(json.dumps(self.manifest))
        return validate_assets(self.path, require_complete=complete, geometry=geometry)

    def test_both_lods_and_matrices_survive_relocation(self):
        report = self.validate(geometry=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertTrue(report["complete"])
        moved = self.root.parent / "moved"
        shutil.move(self.root, moved)
        report = validate_assets(moved / self.path.name, require_complete=True, geometry=True)
        self.assertEqual(report["status"], "passed", report["errors"])
        self.assertEqual(set(report["records"][0]["lods"]), {"0.25M", "0.1M"})

    def test_smaller_lod_is_checked_independently(self):
        row = self.record["lods"]["0.1M"]
        path = self.root / row["path"]
        cube_glb(path, offset_x=.4)
        row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report = self.validate(geometry=True)
        self.assertEqual(report["status"], "failed")
        self.assertIn("0.1M mesh footprint fails", report["errors"][0]["message"])

    def test_composition_and_normal_matrix_cannot_be_fabricated(self):
        self.record["building_local_from_asset_4x4"][0][3] = 2
        report = self.validate()
        self.assertIn("do not compose", report["errors"][0]["message"])
        self.record["building_local_from_asset_4x4"][0][3] = 0
        self.record["normal_transform_world_3x3"] = np.eye(3).tolist()
        self.assertIn("inverse transpose", self.validate()["errors"][0]["message"])

    def test_missing_or_pending_source_ids_prevent_completion(self):
        self.manifest["source_structure"].update(instance_count=2, building_ids=["current_A", "current_B"])
        report = self.validate(complete=False)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["complete"])
        self.assertEqual(report["missing_ids"], ["current_B"])
        self.assertEqual(self.validate()["status"], "failed")
        self.manifest["buildings"].append({"building_id": "current_B", "status": "pending_generation"})
        self.assertEqual(self.validate()["status"], "failed")

    def test_exclusion_keeps_identity_and_needs_actual_evidence(self):
        self.manifest["buildings"] = [{"building_id": "current_A", "status": "excluded_incomplete",
                                      "reason": "Only an isolated facade strip is observed",
                                      "evidence": [{"observation_id": "observed_A", "source_sha256": "b" * 64}]}]
        report = self.validate()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["complete"])
        del self.manifest["buildings"][0]["evidence"]
        self.assertEqual(self.validate()["status"], "failed")

    def test_paths_and_same_size_corruption_are_rejected(self):
        self.record["lods"]["0.1M"]["path"] = "../outside.glb"
        self.assertIn("inside", self.validate()["errors"][0]["message"])
        self.record["lods"]["0.1M"]["path"] = "assets/current_A/model_01m.glb"
        path = self.root / self.record["lods"]["0.1M"]["path"]
        payload = bytearray(path.read_bytes()); payload[-1] ^= 1; path.write_bytes(payload)
        self.assertIn("SHA256 mismatch", self.validate()["errors"][0]["message"])

    def test_composite_parts_cannot_replace_whole_building_lods(self):
        self.record["status"] = "accepted_composite"
        self.record["parts"] = [copy.deepcopy(self.record["lods"])]
        del self.record["lods"]
        self.assertIn("whole-building", self.validate()["errors"][0]["message"])

    def test_shared_physical_asset_requires_complete_explicit_group(self):
        other = copy.deepcopy(self.record)
        other["building_id"] = "current_B"
        self.manifest["buildings"].append(other)
        self.manifest["source_structure"].update(instance_count=2, building_ids=["current_A", "current_B"])
        with self.assertRaisesRegex(ValueError, "Duplicate physical"):
            self.validate()
        left = {"type": "Polygon", "coordinates": [[[0, -1], [.5, -1], [.5, 0], [0, 0], [0, -1]]]}
        right = {"type": "Polygon", "coordinates": [[[.5, -1], [1, -1], [1, 0], [.5, 0], [.5, -1]]]}
        for row, source in [(self.record, left), (other, right)]:
            row.update(asset_instance_group_id="physical_A", source_target_geometry=source)
        self.manifest["asset_instance_groups"] = [{"id": "physical_A", "building_ids": ["current_A", "current_B"],
            "lods": copy.deepcopy(self.record["lods"]),
            "world_from_local_4x4": copy.deepcopy(self.record["world_from_local_4x4"])}]
        report = self.validate(geometry=True)
        self.assertTrue(report["complete"])
        self.assertEqual(report["asset_instance_groups"], 1)
        self.manifest["asset_instance_groups"][0]["building_ids"] = ["current_A"]
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.validate()

    def test_cli_rejects_bad_schema_with_a_small_structured_report(self):
        self.manifest["schema"] = "unknown"
        self.path.write_text(json.dumps(self.manifest))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["validate-assets", "--manifest", str(self.path)])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_count"], 1)
        self.assertNotIn("Traceback", output.getvalue())

    def test_strict_cli_summary_keeps_full_pending_ids_only_in_report(self):
        ids = [f"pending_{i:04d}" for i in range(400)]
        self.manifest["source_structure"].update(instance_count=len(ids), building_ids=ids)
        self.manifest["buildings"] = [{"building_id": sid, "status": "pending_generation"} for sid in ids]
        self.path.write_text(json.dumps(self.manifest))
        destination = self.root / "validation.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["validate-assets", "--manifest", str(self.path), "--require-complete", "--output", str(destination)])
        summary = json.loads(output.getvalue())
        detailed = json.loads(destination.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(summary["pending_instances"], 400)
        self.assertEqual(summary["errors"][0]["pending_instances"], 400)
        self.assertNotIn("pending_ids", summary["errors"][0])
        self.assertNotIn(ids[0], output.getvalue())
        self.assertLess(len(output.getvalue()), 2000)
        self.assertEqual(detailed["pending_ids"], ids)
        self.assertEqual(detailed["errors"][0]["pending_ids"], ids)

    def test_grounding_tolerance_caps_at_twenty_centimetres(self):
        world = np.array(self.record["world_from_local_4x4"], dtype=float)
        world[2, 1] = 200
        world[2, 3] = .21
        self.record.update(world_from_local_4x4=world.tolist(),
                           building_local_from_asset_4x4=world.tolist(),
                           normal_transform_world_3x3=np.linalg.inv(world[:3, :3]).T.tolist(),
                           top_z_m=200)
        report = self.validate()
        self.assertEqual(report["status"], "failed")
        self.assertIn("predicted base/top", report["errors"][0]["message"])

    def test_optional_assembly_table_matches_only_accepted_membership(self):
        self.record["asset_instance_id"] = "physical_A"
        self.manifest["asset_instances"] = [{"id": "physical_A", "source_building_ids": ["current_A"],
            "status": "accepted_reuse", "world_from_local_4x4": copy.deepcopy(self.record["world_from_local_4x4"]),
            "lods": copy.deepcopy(self.record["lods"])}]
        report = self.validate()
        self.assertTrue(report["complete"])
        self.assertEqual(report["physical_asset_instances"], 1)
        self.manifest["asset_instances"][0]["world_from_local_4x4"][0][3] = .001
        with self.assertRaisesRegex(ValueError, "placement differ"):
            self.validate()
        self.manifest["asset_instances"][0]["world_from_local_4x4"][0][3] = 0
        self.record["status"] = "pending_lod_export"
        with self.assertRaisesRegex(ValueError, "without accepted"):
            self.validate(complete=False)

    def test_shared_group_cannot_split_into_two_assembly_instances(self):
        other = copy.deepcopy(self.record)
        other["building_id"] = "current_B"
        self.manifest["buildings"].append(other)
        self.manifest["source_structure"].update(instance_count=2, building_ids=["current_A", "current_B"])
        left = {"type": "Polygon", "coordinates": [[[0, -1], [.5, -1], [.5, 0], [0, 0], [0, -1]]]}
        right = {"type": "Polygon", "coordinates": [[[.5, -1], [1, -1], [1, 0], [.5, 0], [.5, -1]]]}
        for row, source, identity in [(self.record, left, "physical_A"), (other, right, "physical_B")]:
            row.update(asset_instance_group_id="shared_A_B", source_target_geometry=source,
                       asset_instance_id=identity)
        self.manifest["asset_instance_groups"] = [{"id": "shared_A_B", "building_ids": ["current_A", "current_B"],
            "lods": copy.deepcopy(self.record["lods"]),
            "world_from_local_4x4": copy.deepcopy(self.record["world_from_local_4x4"])}]
        self.manifest["asset_instances"] = [{"id": row["asset_instance_id"], "source_building_ids": [row["building_id"]],
            "status": row["status"], "lods": copy.deepcopy(row["lods"]),
            "world_from_local_4x4": copy.deepcopy(row["world_from_local_4x4"])} for row in [self.record, other]]
        with self.assertRaisesRegex(ValueError, "group ID as the physical"):
            self.validate()
        for row in [self.record, other]:
            row["asset_instance_id"] = "shared_A_B"
        instance = self.manifest["asset_instances"][0]
        instance.update(id="shared_A_B", source_building_ids=["current_A", "current_B"])
        self.manifest["asset_instances"] = [instance]
        report = self.validate(geometry=True)
        self.assertTrue(report["complete"])
        self.assertEqual(report["physical_asset_instances"], 1)

    def test_copied_model_at_same_placement_still_requires_group(self):
        other = copy.deepcopy(self.record)
        other["building_id"] = "current_B"
        for label, row in other["lods"].items():
            source = self.root / row["path"]
            destination = self.root / "assets/current_B" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            row["path"] = destination.relative_to(self.root).as_posix()
        self.manifest["buildings"].append(other)
        self.manifest["source_structure"].update(instance_count=2, building_ids=["current_A", "current_B"])
        with self.assertRaisesRegex(ValueError, "Duplicate physical"):
            self.validate()
        for key in ["world_from_local_4x4", "building_local_from_asset_4x4"]:
            other[key][0][3] = 2
        other["target_geometry"]["coordinates"] = [[[2, -1], [3, -1], [3, 0], [2, 0], [2, -1]]]
        report = self.validate(geometry=True)
        self.assertTrue(report["complete"])

    def test_face_budget_counts_every_actual_scene_instance(self):
        row = self.record["lods"]["0.1M"]
        path = self.root / row["path"]
        # One encoded mesh is within budget; two scene nodes draw 102,024 faces.
        cube_glb(path, nodes=[{"mesh": 0}, {"mesh": 0}], roots=[0, 1], triangle_repetitions=4251)
        row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), faces=51012)
        report = self.validate()
        self.assertEqual(report["status"], "failed")
        self.assertIn("Actual scene triangle count", report["errors"][0]["message"])
        row["faces"] = 102024
        self.assertEqual(self.validate()["status"], "failed")
        # Correctly declared instancing within the budget is accepted.
        cube_glb(path, nodes=[{"mesh": 0}, {"mesh": 0}], roots=[0, 1])
        row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), faces=24)
        report = self.validate(geometry=True)
        self.assertTrue(report["complete"])
        self.assertEqual(report["records"][0]["lods"]["0.1M"]["faces"], 24)
        self.assertEqual(report["records"][0]["lods"]["0.1M"]["encoded_mesh_faces"], 12)

    def test_nested_node_transforms_are_kept_for_both_lods(self):
        nodes = [{"translation": [2, 3, 4], "rotation": [0, 0, 2 ** -.5, 2 ** -.5], "children": [1]},
                 {"mesh": 0, "scale": [2, 3, 4]}]
        node_transform = np.array([[0, -3, 0, 2], [2, 0, 0, 3], [0, 0, 4, 4], [0, 0, 0, 1]])
        world = np.array(self.record["world_from_local_4x4"]) @ np.linalg.inv(node_transform)
        self.record.update(world_from_local_4x4=world.tolist(), building_local_from_asset_4x4=world.tolist(),
                           normal_transform_world_3x3=np.linalg.inv(world[:3, :3]).T.tolist())
        for row in self.record["lods"].values():
            path = self.root / row["path"]
            cube_glb(path, nodes=nodes)
            row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report = self.validate(geometry=True)
        self.assertTrue(report["complete"], report["errors"])
        # Only the lower-detail file loses its native placement.
        bad_nodes = copy.deepcopy(nodes)
        bad_nodes[1]["translation"] = [.8, 0, 0]
        row = self.record["lods"]["0.1M"]
        path = self.root / row["path"]
        cube_glb(path, nodes=bad_nodes)
        row.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report = self.validate(geometry=True)
        self.assertEqual(report["status"], "failed")
        self.assertIn("Actual 0.1M mesh footprint fails", report["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
