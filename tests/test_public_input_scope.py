"""Public sample discovery must work with image/camera inputs alone."""
import ast
from pathlib import Path
import tempfile
import unittest

from city_pipeline.geometry.orchestrator import discover_samples


class PublicInputScope(unittest.TestCase):
    def test_observation_discovery_has_no_reference_depth_prerequisite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            view = root/'patches/PATCH/views/tilt_light'
            view.mkdir(parents=True)
            (view/'camera.json').write_text('{}')
            (view/'rgb.png').write_bytes(b'input-image-placeholder')
            rows = discover_samples(root, [], ['tilt_light'])
            self.assertEqual(rows[0]['id'], 'PATCH__tilt_light')
            self.assertEqual(set(rows[0]['paths']), {'rgb', 'camera'})

    def test_public_geometry_modules_no_longer_expose_legacy_evaluation_mains(self):
        root = Path(__file__).resolve().parents[1]
        for name in ['camera_geometry', 'roof_masks']:
            tree = ast.parse((root/f'src/city_pipeline/geometry/{name}.py').read_text())
            self.assertNotIn('main', {node.name for node in tree.body if isinstance(node, ast.FunctionDef)})
        self.assertFalse((root/'pipelines/background_postprocess/derive_land.py').exists())


if __name__ == '__main__':
    unittest.main()
