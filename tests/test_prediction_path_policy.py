"""Reference-annotation directories cannot supply recovery evidence."""
from pathlib import Path
import tempfile
import unittest

from city_pipeline.inputs import _prediction_source
from city_pipeline.buildings.polish import _prediction_path as point_path
from city_pipeline.buildings.visibility_refinement import _prediction_path as image_path


class PredictionPathPolicy(unittest.TestCase):
    def test_reserved_annotation_prefix_is_blocked_in_all_evidence_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('gt_annotations', 'evaluation_gt', 'global_vectors', 'predicted_points'):
                relative = Path(name) / 'cloud.npz'
                target = root / relative
                target.parent.mkdir()
                target.write_bytes(b'input provenance test')
                readers = (lambda: _prediction_source(root, relative),
                           lambda: point_path(relative, root),
                           lambda: image_path(relative, root))
                for reader in readers:
                    with self.subTest(name=name, reader=reader):
                        if name == 'predicted_points':
                            self.assertEqual(reader(), target)
                        else:
                            with self.assertRaises((ValueError, PermissionError)):
                                reader()


if __name__ == '__main__':
    unittest.main()
