"""Prediction-only structure cleanup contracts with synthetic point evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape

from city_pipeline.buildings.polish import apply_polish, normal_statistics, _apply_geometry


def feature(gid, geometry, support=1, height=50, **extra):
    p = {"predicted_global_building_id": gid, "consensus_observation_ids": [gid + str(i) for i in range(support)],
         "canonical_observation_id": gid + "0", "canonical_instance_id": 1,
         "footprint_area_m2": geometry.area, "centroid_x_m": geometry.centroid.x,
         "centroid_y_m": geometry.centroid.y, "base_z_m": 0, "body_top_z_m": height,
         "top_z_m": height, "height_m": height}
    p.update(extra)
    return {"type": "Feature", "properties": p, "geometry": mapping(geometry)}


class PolishContractTests(unittest.TestCase):
    def scene(self, folder, *, vertical=True, point_count=900, neighbor_quality=90, strong_support=2):
        thin = Polygon([(0, 0), (10, 0), (10, .5), (.5, .5), (.5, 10), (0, 10)])
        candidates = [feature("arbitrary-facade", thin, canonical_building_point_cloud_npz_path="cloud.npz"),
                      feature("arbitrary-neighbor", box(10.5, 0, 40.5, 20), support=strong_support)]
        x, y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 40, 30))
        if vertical:
            points = np.column_stack([np.zeros(x.size), x.ravel(), y.ravel()])
        else:
            points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 50.)])
        points = points[:point_count]
        np.savez(Path(folder) / "cloud.npz", xyz_m=points.astype(np.float32),
                 instance_id=np.ones(len(points), np.uint16), provenance=np.ones(len(points), np.uint8))
        observations = {oid: {"preliminary_quality": neighbor_quality} for oid in candidates[1]["properties"]["consensus_observation_ids"]}
        return candidates, observations

    def test_raw_normal_evidence_separates_facade_and_roof(self):
        for vertical, remaining in [(True, 1), (False, 2)]:
            with self.subTest(vertical=vertical), tempfile.TemporaryDirectory() as folder:
                features, observations = self.scene(folder, vertical=vertical)
                before = copy.deepcopy(features)
                result, report = apply_polish(features, observations, folder, {"remove_tiny_facades": True})
                self.assertEqual(len(result), remaining)
                self.assertEqual(features, before)
                self.assertFalse(report["uses_ground_truth"])
                self.assertEqual(len(report["source_sha256"]), 1)
                self.assertTrue(report["fragment_checks"][0]["normal_statistics"]["planar_sample_count"] >= 200)

    def test_neighbor_strength_and_multiple_observations_protect_entities(self):
        for quality, support, candidate_support in [(84, 2, 1), (90, 1, 1), (90, 2, 2)]:
            with self.subTest(quality=quality, support=support, candidate_support=candidate_support), tempfile.TemporaryDirectory() as folder:
                fs, obs = self.scene(folder, neighbor_quality=quality, strong_support=support)
                fs[0]["properties"]["consensus_observation_ids"] = ["candidate" + str(i) for i in range(candidate_support)]
                result, report = apply_polish(fs, obs, folder, {"remove_tiny_facades": True})
                self.assertEqual(len(result), 2)
                self.assertEqual(report["runtime_data_reads"], [])

    def test_insufficient_missing_and_filled_point_evidence_retains_prediction(self):
        for mode in ["few", "missing", "filled", "degenerate"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                fs, obs = self.scene(folder, point_count=100 if mode == "few" else 900)
                p = Path(folder) / "cloud.npz"
                if mode == "missing":
                    p.unlink()
                elif mode in {"filled", "degenerate"}:
                    with np.load(p) as data:
                        xyz, ids = data["xyz_m"], data["instance_id"]
                    if mode == "degenerate":
                        xyz[:] = 0
                    np.savez(p, xyz_m=xyz, instance_id=ids,
                             provenance=np.full(len(ids), 4 if mode == "filled" else 1, np.uint8))
                result, report = apply_polish(fs, obs, folder, {"remove_tiny_facades": True})
                self.assertEqual(len(result), 2)
                self.assertEqual(report["removed_by_global_rule"], [])

    def test_no_ids_in_config_and_no_reference_point_cloud_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            fs, obs = self.scene(folder)
            with self.assertRaises(ValueError):
                apply_polish(fs, obs, folder, {"removed_ids": ["arbitrary-facade"]})
            fs[0]["properties"]["canonical_building_point_cloud_npz_path"] = "evaluation_gt_only/cloud.npz"
            with self.assertRaises(PermissionError):
                apply_polish(fs, obs, folder, {"remove_tiny_facades": True})
            fs[0]["properties"]["canonical_building_point_cloud_npz_path"] = "../cloud.npz"
            with self.assertRaises(PermissionError):
                apply_polish(fs, obs, folder, {"remove_tiny_facades": True})

    def test_component_cleanup_clips_roofs_and_preserves_heights(self):
        body, fragment = box(0, 0, 20, 20), box(30, 0, 31, 1)
        fs = [feature("roofed", MultiPolygon([body, fragment]),
                      top_z_m=60, height_m=60, roof_layers=[
                          {"geometry": mapping(MultiPolygon([box(1, 1, 3, 3), fragment])), "top_z_m": 60, "base_z_m": 50}])]
        result, report = apply_polish(fs, {}, ".", {"operations": [{"kind": "components", "minimum_area": 3, "maximum_removed_ratio": .01}]})
        self.assertTrue(shape(result[0]["geometry"]).equals(body))
        p = result[0]["properties"]
        self.assertEqual((p["body_top_z_m"], p["top_z_m"], p["height_m"]), (50, 60, 60))
        self.assertTrue(shape(p["roof_layers"][0]["geometry"]).equals(box(1, 1, 3, 3)))
        self.assertEqual(len(report["operations"][0]["changes"]), 1)
        large_fragment = MultiPolygon([body, box(30, 0, 32.5, 2)])
        retained, _ = apply_polish([feature("large", large_fragment)], {}, ".", {"operations": [{"kind": "components", "minimum_area": 6}]})
        self.assertTrue(shape(retained[0]["geometry"]).equals(large_fragment))

    def test_severe_overlap_reverts_geometry_change(self):
        # A thin bay removes under15% area. Closing it would swallow a predicted
        # neighboring building, so both candidate geometries must be reverted.
        g = Polygon([(0, 0), (40, 0), (40, 40), (22, 40), (22, 0.1), (20, 0.1), (20, 40), (0, 40)])
        peer = box(20.1, 5, 21.9, 35)
        result, report = _apply_geometry([feature("body", g), feature("peer", peer)], {"kind": "closing", "radius": 1.5})
        self.assertEqual(report["overlap_reverted_indices"], [0, 1])
        self.assertEqual(report["changes"], [])
        self.assertTrue(shape(result[0]["geometry"]).equals(g))

    def test_empty_config_is_noop_and_normals_are_deterministic(self):
        fs = [feature("random-id", box(0, 0, 10, 10))]
        result, report = apply_polish(fs, {}, ".", {})
        self.assertEqual(result, fs)
        self.assertEqual(report["operations"], [])
        self.assertEqual(report["runtime_data_reads"], [])
        points = np.random.default_rng(51).normal(size=(1000, 3)); points[:, 2] = 30
        self.assertEqual(normal_statistics(points), normal_statistics(points))

    def test_isolated_cli_publishes_complete_manifest_without_meshes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "input"; source.mkdir()
            (source / "global_building_footprints.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feature("plain", box(0, 0, 10, 10))]}))
            (root / "observations.json").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
            (root / "config.json").write_text("{}")
            output = root / "output"
            command = [sys.executable, "-m", "city_pipeline.buildings.polish", "--input-root", str(source),
                       "--observations", str(root / "observations.json"), "--data-root", str(root),
                       "--output-root", str(output), "--config", str(root / "config.json")]
            process = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["building_count"], 1)
            self.assertEqual(len(manifest["source_sha256"]), 3)
            self.assertEqual(list(output.glob("*.glb")), [])
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
