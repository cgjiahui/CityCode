"""Background boundary tests: integrity, uncovered pixels, holes and layer export."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import trimesh
from citycode_background.io import local, write, sha256, verify, new_output
from citycode_background.mesh import surface, build
from citycode_background.rgb import accumulate, compare_images


class BackgroundContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.grid = {'image_size_px': [5, 5], 'meters_per_pixel': .15,
                     'pixel_center_index_to_ue_world_xy_3x3': [[.15, 0, 10], [0, .15, 20], [0, 0, 1]]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapped_png_matches_pillow_in_all_supported_modes(self):
        from citycode_background import raster
        rng = np.random.default_rng(7)
        try:
            raster.configure(self.root/'png_cache')
            for mode, channels in [('L', 1), ('RGB', 3), ('RGBA', 4)]:
                shape = (257, 193) if channels == 1 else (257, 193, channels)
                data = rng.integers(0, 256, shape, dtype=np.uint8)
                src = self.root/f'{mode}.png'; dst = self.root/f'{mode}_out.png'
                Image.fromarray(data).save(src)
                mapped = raster.pixels(src, mode)
                self.assertIsInstance(mapped, np.memmap)
                np.testing.assert_array_equal(mapped, data)
                raster.save(dst, mapped)
                with Image.open(dst) as im: np.testing.assert_array_equal(np.asarray(im), data)
            # Alpha must be dropped, not composited on black, for source RGB.
            rgba = raster.pixels(self.root/'RGBA.png', 'RGB')
            with Image.open(self.root/'RGBA.png') as im:
                np.testing.assert_array_equal(rgba, np.asarray(im.convert('RGB')))
        finally:
            raster._CACHE = None
            raster._LIB = None

    def test_paths_and_frozen_output(self):
        (self.root/'ok').write_text('ok')
        self.assertEqual(local(self.root, 'ok'), self.root/'ok')
        for name in ['../outside', str(self.root/'ok')]:
            with self.assertRaises(ValueError): local(self.root, name)
        with self.assertRaises(FileNotFoundError): local(self.root, 'missing.png')
        with self.assertRaises(ValueError): new_output(self.root/'result', self.root)

    def test_v10_water_protection_preserves_read_only_generation_cache(self):
        from citycode_background import raster
        from citycode_background.rgb import stitch_v10
        write(self.root/'grid.json', self.grid)
        for folder in ['global', 'historical_reference', 'output']:
            (self.root/folder).mkdir()
        for name, mode, color in [
            ('global/input_rgb.png', 'RGB', (10, 20, 30)),
            ('global/final_rgb.png', 'RGB', (10, 20, 30)),
            ('global/outside_water_mask.png', 'L', 255),
            ('output/v9_water_restored.png', 'RGB', (0, 0, 0)),
            ('generated.png', 'RGB', (80, 90, 100)),
            ('building.png', 'L', 0), ('missing.png', 'L', 0),
            ('historical_reference/ortho_rgba.png', 'RGBA', (100, 100, 100, 255)),
            ('historical_reference/water_rgb.png', 'RGB', (0, 0, 0)),
        ]:
            Image.new(mode, (5, 5), color).save(self.root/name)
        tile = {'tile_id': 'one', 'pixel_bounds_xyxy': [0, 0, 5, 5], 'neighbor_overlaps_px': {},
                'v10_result': 'generated.png', 'files': {'building_mask': 'building.png', 'unobserved_mask': 'missing.png'}}
        cfg = {'grid': 'grid.json', 'tiles': [tile], 'selected_ids': ['one'], 'water_tile_ids': []}
        before = sha256(self.root/'generated.png')
        try:
            raster.configure(self.root/'png_cache')
            result = stitch_v10(self.root, cfg, self.root/'output')
            self.assertTrue(result['pixel_exact'])
            np.testing.assert_array_equal(raster.pixels(self.root/'generated.png')[0, 0], [80, 90, 100])
            self.assertEqual(sha256(self.root/'generated.png'), before)
        finally:
            raster._CACHE = None
            raster._LIB = None

    def test_coarse_mesh_does_not_cross_interior_hole(self):
        z = np.ones((5, 5), np.float32)
        complete = surface(z, self.grid, stride=2)
        z[1, 1] = np.nan  # All coarse corner samples remain finite.
        hole = surface(z, self.grid, stride=2)
        self.assertEqual(len(complete.faces), 8)
        self.assertEqual(len(hole.faces), 6)
        self.assertTrue(np.isfinite(hole.vertices).all())

    def test_negative_transform_preserves_upward_winding(self):
        self.grid['pixel_center_index_to_ue_world_xy_3x3'][0][0] = -.15
        mesh = surface(np.ones((5, 5)), self.grid, stride=2)
        self.assertTrue((mesh.face_normals[:, 2] > .99).all())

    def test_mesh_keeps_two_heights_at_same_xy(self):
        depth = self.root/'depth'; depth.mkdir()
        write(depth/'pixel_to_world.json', self.grid)
        np.save(depth/'base_surface_z_m.npy', np.ones((5, 5), np.float32))
        np.save(depth/'elevated_surface_z_m.npy', np.full((5, 5), 8., np.float32))
        Image.new('RGB', (5, 5), 'red').save(self.root/'rgb.png')
        with patch('socket.create_connection', side_effect=AssertionError('Network forbidden')):
            report = build(depth, self.root/'rgb.png', self.root/'mesh', stride=2, texture_size=5)
        scene = trimesh.load(self.root/'mesh/background_ue_z_up.glb', force='scene', process=False)
        self.assertEqual(set(scene.geometry), {'base', 'elevated'})
        self.assertAlmostEqual(scene.geometry['elevated'].vertices[:, 2].mean(), 8.)
        self.assertEqual(report['api_calls'], 0)

    def test_missing_mosaic_coverage_fails_and_removes_scratch(self):
        write(self.root/'grid.json', self.grid)
        tile = {'pixel_bounds_xyxy': [0, 0, 2, 2], 'neighbor_overlaps_px': {}}
        with self.assertRaisesRegex(ValueError, 'Uncovered'):
            accumulate(self.root, {'grid': 'grid.json'}, self.root,
                       [(tile, np.zeros((2, 2, 3), np.uint8))], 'out.png')
        self.assertFalse((self.root/'out.png.scratch').exists())

    def test_pixel_difference_not_hidden_by_matching_dimensions(self):
        Image.new('RGB', (5, 5), 'red').save(self.root/'a.png')
        Image.new('RGB', (5, 5), 'blue').save(self.root/'b.png')
        with self.assertRaisesRegex(ValueError, 'Pixel replay mismatch'):
            compare_images(self.root/'a.png', self.root/'b.png')

    def test_checksum_tamper_rejected_without_generation_fallback(self):
        cfg = {'schema': 'citycode.background_history.v1', 'lineage': 'historical_gt_assisted',
               'network_policy': 'offline_only_no_provider_client', 'grid': 'grid.json',
               'tiles': [], 'patches': [], 'selected_ids': [], 'water_tile_ids': []}
        write(self.root/'manifest.json', cfg); write(self.root/'grid.json', self.grid)
        for layer in ['base', 'elevated']:
            p = self.root/f'height/completed/{layer}_surface_z_m.npy';p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.ones((5, 5)))
        files = [{'path': p.relative_to(self.root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                 for p in self.root.rglob('*') if p.is_file()]
        write(self.root/'CHECKSUMS.json', {'files': files})
        with patch('socket.create_connection', side_effect=AssertionError('Network forbidden')):
            self.assertEqual(verify(self.root)['status'], 'passed')
            (self.root/'grid.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'): verify(self.root)


if __name__ == '__main__':
    unittest.main()
