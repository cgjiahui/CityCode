"""The release has one prediction-only recipe and bounded global selection knobs."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from city_pipeline.buildings.parameters import normalize_selection
from citycode_perception.cli import STRUCTURE_PROFILES, parser, recovery_config, structure_parameters


class StandardRecipe(unittest.TestCase):
    def test_multiple_views_of_one_patch_rejected_before_expensive_recovery(self):
        samples = [SimpleNamespace(patch_id='P', view_id=view) for view in ['light', 'front']]
        with patch('citycode_perception.cli.discover_samples', return_value=samples):
            with self.assertRaisesRegex(ValueError, 'one selected view per physical patch'):
                recovery_config('data', 'absent-depth', 'absent-masks')

    def test_only_one_release_recipe_and_both_input_entries_share_it(self):
        self.assertEqual(list(STRUCTURE_PROFILES), ['standard'])
        cfg = structure_parameters()
        self.assertEqual(cfg['selection']['area_weight'], .5)
        self.assertEqual(cfg['visibility_refinement'], {'method': 'local_surface_veto'})
        self.assertEqual(cfg['postprocess']['replace_repair']['max_shift'], 3.)
        for command, inputs in [('recover', ['--dataset', 'data', '--depth', 'depth', '--masks', 'masks']),
                                ('recover-cached', ['--cache', 'cache'])]:
            args = parser().parse_args([command, *inputs, '--output', 'result'])
            self.assertFalse(hasattr(args, 'polish'))
            self.assertFalse(args.structure_only)

    def test_selection_rejects_reference_paths_ids_and_invalid_numbers(self):
        for invalid in [{'building_ids': ['x']}, {'gt_file': 'reference.json'},
                        {'area_weight': True}, {'area_weight': float('nan')},
                        {'consensus_weight': float('inf')}, {'area_weight': 1.1},
                        {'min_area_m2': 0}, {'conflict_containment': 0}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_selection(invalid)

    def test_global_override_preserves_other_release_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps({'postprocess': {'boundary': {'offset_m': 1.0}}}))
            cfg = structure_parameters(override=path)
            self.assertEqual(cfg['postprocess']['boundary']['offset_m'], 1.)
            self.assertEqual(cfg['postprocess']['replace_repair']['max_shift'], 3.)
            self.assertEqual(cfg['selection']['area_weight'], .5)
            for invalid in [{'visibility_refinement': None},
                            {'visibility_refinement': {'method': 'weak_silhouette'}},
                            {'selection': {'selected_ids': ['x']}},
                            {'profile': 'clean-v2'}]:
                path.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    structure_parameters(override=path)


if __name__ == '__main__':
    unittest.main()
