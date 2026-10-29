"""Postprocessing cannot relabel wrong geometry or manufacture whole buildings."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from citycode_assets.postprocess_validation import (
    validate_evidence_only_record, validate_postprocess_record, validate_surface_instances,
)


def mesh_glb(path, vertices, faces, building_id=None):
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<u4")
    positions = vertices.tobytes()
    indices = faces.tobytes()
    binary = positions + indices
    document = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
                        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": len(vertices)},
                      {"bufferView": 1, "componentType": 5125, "type": "SCALAR", "count": faces.size}],
    }
    if building_id is not None:
        document["meshes"][0]["extras"] = {"predicted_global_building_id": building_id}
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    chunks = struct.pack("<I4s", len(encoded), b"JSON") + encoded
    chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)


class PostprocessValidation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name) / "data"
        self.root = self.data / "assets"
        self.root.mkdir(parents=True)
        self.source = self.data / "structure" / "building_generation_inputs" / "buildings" / "A"
        self.source.mkdir(parents=True)
        self.vertices = np.array([[x, y, z] for x, y, z in [
            [-1, 0, -1.5], [1, 0, -1.5], [1, 0, 1.5], [-1, 0, 1.5],
            [-1, 5, -1.5], [1, 5, -1.5], [1, 5, 1.5], [-1, 5, 1.5]]], dtype=float)
        self.faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                               [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                               [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
        self.proxy = self.source / "proxy.glb"
        mesh_glb(self.proxy, self.vertices, self.faces)
        self.geometry = {"type": "Polygon", "coordinates": [
            [[9, 18.5], [11, 18.5], [11, 21.5], [9, 21.5], [9, 18.5]]]}
        self.translation = np.eye(4)
        self.translation[:3, 3] = [10, 20, 7]
        self.world = np.array([[1, 0, 0, 10], [0, 0, -1, 20],
                               [0, 1, 0, 7], [0, 0, 0, 1.]])
        self.building = {"building_id": "A", "footprint_world": self.geometry,
                         "base_z_m": 7, "top_z_m": 12,
                         "world_from_local_4x4": self.translation.tolist(),
                         "paths": {"proxy_local_gltf_y_up": "proxy.glb"}}
        self.building_path = self.source / "building.json"
        self.building_path.write_text(json.dumps(self.building))
        self.evidence = self.source / "observation.json"
        self.evidence.write_text('{"observed":true}')
        self.lods = {}
        for label, name in [("0.25M", "high.glb"), ("0.1M", "low.glb")]:
            path = self.root / name
            mesh_glb(path, self.vertices, self.faces)
            self.lods[label] = self.proof(path)
        self.record = {"building_id": "A", "status": "accepted_structure",
                       "target_geometry": copy.deepcopy(self.geometry),
                       "world_from_local_4x4": self.world.tolist(),
                       "base_z_m": 7, "top_z_m": 12, "lods": self.lods,
                       "postprocess": {
                           "schema": "citycode.asset_postprocess_record.v1",
                           "method": "source_structure", "representation": "structure",
                           "detailed_asset_accepted": False,
                           "source_target_geometry": copy.deepcopy(self.geometry),
                           "source_base_z_m": 7, "source_top_z_m": 12,
                           "source": {"building": self.proof(self.building_path),
                                      "proxy": self.proof(self.proxy),
                                      "evidence": [self.proof(self.evidence)]}}}

    def proof(self, path):
        import os
        return {"path": Path(os.path.relpath(path, self.root)).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}

    def test_same_source_geometry_survives_reordered_vertices_and_faces(self):
        # Legitimate normal/UV/material export may reorder the source topology.
        permutation = np.array([7, 2, 6, 0, 5, 3, 1, 4])
        inverse = np.argsort(permutation)
        path = self.root / "low.glb"
        mesh_glb(path, self.vertices[permutation], inverse[self.faces[::-1, ::-1]])
        self.record["lods"]["0.1M"] = self.proof(path)
        report = validate_postprocess_record(self.root, self.record, geometry=True)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["actual_source_geometry_checked"])
        self.assertFalse(report["detailed_semantic_acceptance_inferred"])
        json.dumps(report, allow_nan=False)

    def test_direct_UE_translation_does_not_place_Y_up_asset(self):
        self.record["world_from_local_4x4"] = self.translation.tolist()
        with self.assertRaisesRegex(ValueError, "Y-up frame"):
            validate_postprocess_record(self.root, self.record)

    def test_changed_actual_mesh_cannot_keep_structural_label(self):
        vertices = self.vertices.copy()
        vertices[6, 0] += .2
        path = self.root / "high.glb"
        mesh_glb(path, vertices, self.faces)
        self.record["lods"]["0.25M"] = self.proof(path)
        with self.assertRaisesRegex(ValueError, "triangle positions"):
            validate_postprocess_record(self.root, self.record, geometry=True)

    def layered_structure(self):
        # A broad lower body and a smaller raised roof preserve two heights.
        lower = self.vertices.copy()
        lower[lower[:, 1] == 5, 1] = 3
        upper = self.vertices.copy()
        upper[:, [0, 2]] *= .5
        upper[upper[:, 1] == 0, 1] = 3
        local = np.concatenate([lower, upper])
        faces = np.concatenate([self.faces, self.faces + len(lower)])
        # The canonical release is already in WORLD glTF coordinates.
        world_gltf = local + [10, 7, -20]
        source = self.data / "structure" / "release" / "canonical_building_glb" / "A.glb"
        mesh_glb(source, world_gltf, faces, building_id="A")
        self.record["postprocess"]["source"].update(
            layered_mesh=self.proof(source), geometry_source_frame="world_gltf_y_up")
        for label, name in [("0.25M", "high.glb"), ("0.1M", "low.glb")]:
            path = self.root / name
            mesh_glb(path, local, faces)
            self.record["lods"][label] = self.proof(path)
        return local, faces, source

    def test_layered_world_source_preserves_lower_body_and_raised_roof(self):
        self.layered_structure()
        result = validate_postprocess_record(self.root, self.record, geometry=True)
        self.assertEqual(result["source_geometry_frame"], "world_gltf_y_up")
        self.assertEqual(result["lods"]["0.1M"]["triangles"], 24)
        self.assertLess(result["lods"]["0.1M"]["position_tolerance_m"], .0001)
        json.dumps(result, allow_nan=False)

    def test_layered_source_rejects_discarding_the_raised_roof(self):
        local, _, _ = self.layered_structure()
        path = self.root / "low.glb"
        mesh_glb(path, local[:8], self.faces)
        self.record["lods"]["0.1M"] = self.proof(path)
        with self.assertRaisesRegex(ValueError, "triangle positions"):
            validate_postprocess_record(self.root, self.record, geometry=True)

    def test_layered_source_rejects_applying_world_translation_twice(self):
        local, faces, _ = self.layered_structure()
        path = self.root / "high.glb"
        mesh_glb(path, local + [10, 7, -20], faces)
        self.record["lods"]["0.25M"] = self.proof(path)
        with self.assertRaisesRegex(ValueError, "triangle positions"):
            validate_postprocess_record(self.root, self.record, geometry=True)

    def test_layered_file_identity_cannot_be_relabelled_by_its_filename(self):
        local, faces, source = self.layered_structure()
        mesh_glb(source, local + [10, 7, -20], faces, building_id="different_building")
        self.record["postprocess"]["source"]["layered_mesh"] = self.proof(source)
        with self.assertRaisesRegex(ValueError, "building identity"):
            validate_postprocess_record(self.root, self.record, geometry=True)

    def test_stale_source_and_symlink_escape_are_rejected(self):
        self.evidence.write_text('{"observed":false}')
        with self.assertRaisesRegex(ValueError, "SHA256 or byte"):
            validate_postprocess_record(self.root, self.record)
        outside = self.data.parent / "private.json"
        outside.write_text("{}")
        link = self.source / "escaped.json"
        link.symlink_to(outside)
        proof = self.proof(link)
        self.record["postprocess"]["source"]["evidence"] = [proof]
        with self.assertRaisesRegex(ValueError, "escapes"):
            validate_postprocess_record(self.root, self.record)

    def test_wrong_representation_and_detailed_claim_are_rejected(self):
        self.record["postprocess"]["representation"] = "repaired_detailed"
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_postprocess_record(self.root, self.record)
        self.record["postprocess"]["representation"] = "structure"
        self.record["postprocess"]["detailed_asset_accepted"] = True
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            validate_postprocess_record(self.root, self.record)

    def fragment_row(self):
        row = copy.deepcopy(self.record)
        row.update(status="resolved_fragment", surface_instance_id="roof_A")
        row["postprocess"].update(method="observed_fragment_surface", representation="observed_surface",
                                  complete_building=False,
                                  surface={"parent_building_id": None})
        return row

    def supported_fragment(self, grid_stride=1):
        from shapely import polygons, union_all
        from shapely.geometry import mapping

        row = self.fragment_row()
        oid, sample = "observation_A", "patch_A"
        od = self.source / "observations" / oid
        od.mkdir(parents=True)
        self.building.update(canonical_observation_id=oid,
                             observations=[{"observation_id": oid, "path": "observations/" + oid}])
        self.building_path.write_text(json.dumps(self.building))
        row["postprocess"]["source"]["building"] = self.proof(self.building_path)
        grid = np.array([[.15, 0, 9.2], [0, .15, 19], [0, 0, 1.]])
        grid_side = 3 * grid_stride + 1
        all_uv = np.array([[x, y] for y in range(grid_side) for x in range(grid_side)], dtype=np.int64)
        # A real source surface deliberately differs from the metadata roof top.
        all_xyz = np.column_stack([all_uv[:, 0] * .15 + 9.2,
                                   all_uv[:, 1] * .15 + 19,
                                   10 + all_uv[:, 0] * .02]).astype(np.float32)
        all_rgb = np.tile(np.array([180, 150, 100], dtype=np.uint8), (len(all_xyz), 1))
        flat = np.flatnonzero(np.all(all_uv % grid_stride == 0, axis=1))
        xyz, rgb, uv = all_xyz[flat], all_rgb[flat], all_uv[flat]
        faces = []
        for y in range(3):
            for x in range(3):
                n = 4 * y + x
                faces.extend([[n, n + 1, n + 5], [n, n + 5, n + 4]])
        faces = np.asarray(faces, dtype=np.int64)
        point_path = self.source / (sample + "__height_cleaned_building_points.npz")
        np.savez_compressed(point_path, xyz_m=all_xyz, rgb=all_rgb,
                            instance_id=np.full(len(all_xyz), 4), provenance=np.ones(len(all_xyz), dtype=np.uint8))
        observation = {"building_id": "A", "observation_id": oid, "sample_id": sample,
                       "source_instance_id": 4, "bev_instance_id": 4,
                       "bev_crop_pixel_center_to_world_xy_3x3": grid.tolist()}
        op = od / "observation.json"
        op.write_text(json.dumps(observation))
        support = self.root / "support.npz"
        extra = {} if grid_stride == 1 else {"triangle_component_index": np.zeros(len(faces), dtype=np.int64)}
        np.savez_compressed(support, vertices_world_xyz_m=xyz.astype(np.float64), faces=faces,
                            vertex_rgb=rgb, vertex_source_flat_index=flat,
                            vertex_original_grid_uv=uv, vertex_source_instance_id=np.full(len(xyz), 4),
                            vertex_source_provenance=np.ones(len(xyz), dtype=np.uint8),
                            vertex_observation_index=np.zeros(len(xyz), dtype=np.int16),
                            triangle_observation_index=np.zeros(len(faces), dtype=np.int16),
                            triangle_roof_layer_index=np.full(len(faces), -1, dtype=np.int16), **extra)
        footprint = union_all(polygons(xyz[faces, :2]))
        coverage = footprint.area / 6
        metadata = {"schema": "citycode.observed_fragment_surface.v1" if grid_stride == 1 else "citycode.observed_fragment_surface.v2",
                    "policy": {"grid_stride": grid_stride}, "building_id": "A",
                    "status": "supported_surface_candidate", "independent_building": False,
                    "source_building": self.proof(self.building_path), "surface": self.proof(support),
                    "vertex_count": len(xyz), "triangle_count": len(faces),
                    "source_footprint_planar_coverage": coverage,
                    "source_observations": [{"observation_id": oid, "sample_id": sample,
                         "source_instance_id": 4, "bev_instance_id": 4,
                         "observation": self.proof(op), "source_points": self.proof(point_path),
                         "grid_pixel_center_to_world_xy_3x3": grid.tolist(), "roof_layer_index": -1}]}
        mp = self.root / "support.json"
        mp.write_text(json.dumps(metadata))
        row["postprocess"]["surface"].update(
            metadata=self.proof(mp), support=self.proof(support),
            thickness_m=.12, thickness_is_visualization_only=True,
            visible_source_coverage=coverage)
        row.update(target_geometry=mapping(footprint), base_z_m=float(xyz[:, 2].min()) - .12,
                   top_z_m=float(xyz[:, 2].max()))
        bottom = xyz.astype(float) - [0, 0, .12]
        world_vertices = np.concatenate([xyz, bottom])
        all_faces = [*faces.tolist(), *(faces[:, ::-1] + len(xyz)).tolist()]
        edges = {}
        for face in faces:
            for a, b in zip(face, np.roll(face, -1)):
                key = tuple(sorted((int(a), int(b))))
                edges[key] = edges.get(key, 0) + 1
        for (a, b), count in edges.items():
            if count == 1:
                all_faces.extend([[a, b, b + len(xyz)], [a, b + len(xyz), a + len(xyz)]])
        local = (world_vertices - self.world[:3, 3]) @ np.linalg.inv(self.world[:3, :3]).T
        for label, name in [("0.25M", "surface_high.glb"), ("0.1M", "surface_low.glb")]:
            path = self.root / name
            mesh_glb(path, local, all_faces)
            row["lods"][label] = self.proof(path)
        return row, world_vertices, np.asarray(all_faces), point_path

    def test_fragment_replays_actual_source_points_instead_of_global_roof(self):
        row, _, _, _ = self.supported_fragment()
        report = validate_postprocess_record(self.root, row, geometry=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["support"]["source_vertices_replayed"], 16)
        self.assertLess(row["top_z_m"], self.building["top_z_m"])
        self.assertFalse(report["complete_building"])
        json.dumps(report, allow_nan=False)

    def test_fragment_cannot_be_extruded_down_to_ground(self):
        row, world_vertices, faces, _ = self.supported_fragment()
        world_vertices[16:, 2] = 7
        local = (world_vertices - self.world[:3, 3]) @ np.linalg.inv(self.world[:3, :3]).T
        path = self.root / "surface_low.glb"
        mesh_glb(path, local, faces)
        row["lods"]["0.1M"] = self.proof(path)
        with self.assertRaisesRegex(ValueError, "triangle positions|thin source roof"):
            validate_postprocess_record(self.root, row, geometry=True)

    def test_fragment_rejects_unqualified_source_samples_even_with_updated_hash(self):
        row, _, _, point_path = self.supported_fragment()
        with np.load(point_path, allow_pickle=False) as loaded:
            points = {key: loaded[key] for key in loaded.files}
        points["provenance"][0] = 2
        np.savez_compressed(point_path, **points)
        metadata_path = self.root / row["postprocess"]["surface"]["metadata"]["path"]
        metadata = json.loads(metadata_path.read_text())
        metadata["source_observations"][0]["source_points"] = self.proof(point_path)
        metadata_path.write_text(json.dumps(metadata))
        row["postprocess"]["surface"]["metadata"] = self.proof(metadata_path)
        with self.assertRaisesRegex(ValueError, "qualified source sample"):
            validate_postprocess_record(self.root, row, geometry=True)

    def test_coarse_fragment_cannot_bridge_an_unknown_interior_grid_sample(self):
        row, _, _, point_path = self.supported_fragment(grid_stride=2)
        self.assertEqual(validate_postprocess_record(self.root, row, geometry=True)["status"], "passed")
        with np.load(point_path, allow_pickle=False) as loaded:
            points = {key: loaded[key] for key in loaded.files}
        # The four coarse corners remain observed. Its middle pixel does not.
        points["provenance"][8] = 2
        np.savez_compressed(point_path, **points)
        metadata_path = self.root / row["postprocess"]["surface"]["metadata"]["path"]
        metadata = json.loads(metadata_path.read_text())
        metadata["source_observations"][0]["source_points"] = self.proof(point_path)
        metadata_path.write_text(json.dumps(metadata))
        row["postprocess"]["surface"]["metadata"] = self.proof(metadata_path)
        with self.assertRaisesRegex(ValueError, "unsupported original grid neighborhood"):
            validate_postprocess_record(self.root, row, geometry=True)

    def test_repair_report_is_bound_to_both_actual_replacements(self):
        row = copy.deepcopy(self.record)
        row["status"] = "accepted_repaired"
        row["postprocess"].update(method="local_mesh_repair", representation="repaired_detailed")
        path = self.root / "repair_report.json"
        report = {"status": "passed", "building_id": "A",
                  "output_lod_hashes": {key: value["sha256"] for key, value in self.lods.items()}}
        path.write_text(json.dumps(report))
        row["postprocess"]["repair"] = {"original_lods": copy.deepcopy(self.lods),
                                         "report": self.proof(path)}
        self.assertEqual(validate_postprocess_record(self.root, row)["status"], "passed")
        report["output_lod_hashes"]["0.1M"] = "0" * 64
        path.write_text(json.dumps(report))
        row["postprocess"]["repair"]["report"] = self.proof(path)
        with self.assertRaisesRegex(ValueError, "actual replacement LODs"):
            validate_postprocess_record(self.root, row)

    def test_surface_table_cannot_count_parent_as_fragment_source(self):
        row = self.fragment_row()
        parent = {"building_id": "B", "status": "accepted_reuse"}
        row["postprocess"]["surface"]["parent_building_id"] = "B"
        surface = {"id": "roof_A", "source_building_ids": ["A"], "status": "resolved_fragment",
                   "world_from_local_4x4": row["world_from_local_4x4"], "lods": row["lods"],
                   "parent_building_id": "B"}
        manifest = {"surface_instances": [surface]}
        self.assertEqual(validate_surface_instances(manifest, [row, parent]), 1)
        surface["source_building_ids"].append("B")
        with self.assertRaisesRegex(ValueError, "own fragment"):
            validate_surface_instances(manifest, [row, parent])

    def test_duplicate_or_missing_surface_source_is_rejected(self):
        row = self.fragment_row()
        surface = {"id": "roof_A", "source_building_ids": ["A"], "status": "resolved_fragment",
                   "world_from_local_4x4": row["world_from_local_4x4"], "lods": row["lods"],
                   "parent_building_id": None}
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_surface_instances({"surface_instances": [surface, copy.deepcopy(surface)]}, [row])
        with self.assertRaisesRegex(ValueError, "every resolved fragment"):
            validate_surface_instances({"surface_instances": []}, [row])

    def test_disabled_reference_surface_must_be_explicit_in_both_records(self):
        row = self.fragment_row()
        row["postprocess"]["surface"]["assembly_enabled"] = False
        surface = {"id": "roof_A", "source_building_ids": ["A"], "status": "resolved_fragment",
                   "world_from_local_4x4": row["world_from_local_4x4"], "lods": row["lods"],
                   "parent_building_id": None}
        with self.assertRaisesRegex(ValueError, "assembly_enabled differs"):
            validate_surface_instances({"surface_instances": [surface]}, [row])
        surface["assembly_enabled"] = False
        self.assertEqual(validate_surface_instances({"surface_instances": [surface]}, [row]), 1)
        surface["assembly_enabled"] = 0
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_surface_instances({"surface_instances": [surface]}, [row])

    def test_evidence_only_retention_proves_sources_without_approving_a_model(self):
        row = copy.deepcopy(self.record)
        row.update(status="excluded_incomplete")
        row.pop("lods")
        row["postprocess"].update(method="retain_fragment_evidence", representation="evidence_only")
        report = validate_evidence_only_record(self.root, row)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["renderable_asset_approved"])
        row["lods"] = self.lods
        with self.assertRaisesRegex(ValueError, "cannot carry accepted model"):
            validate_evidence_only_record(self.root, row)


if __name__ == "__main__":
    unittest.main()
