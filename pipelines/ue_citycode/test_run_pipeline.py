"""Orchestration contracts only: no city geometry, UE, network, or GPU work."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def config(self, root, stage='prepare', *args):
        return pipeline.configuration(pipeline.parser().parse_args([stage, *args]), root)

    def test_prepare_plan_is_ordered_and_passes_explicit_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.config(root, 'prepare', '--workers', '4', '--lod', '1M', '--binding', 'data/my city/binding.json')
            steps = pipeline.build_plan(config)
            labels = [s.label for s in steps]
            self.assertLess(labels.index('filter_old_elevated_background'), labels.index('repair_predicted_road_network'))
            self.assertLess(labels.index('sweep_and_join_road_surface'), labels.index('place_rails_and_supports'))
            self.assertLess(labels.index('review_street_props'), labels.index('audit_street_props'))
            for step in steps:
                if step.label in ('bake_buildings_and_background', 'bind_background_parts', 'place_rails_and_supports', 'review_street_props', 'audit_street_props'):
                    self.assertEqual(step.argv[step.argv.index('--binding')+1], str(config['binding']))
            self.assertIn('1M', steps[0].argv)
            self.assertFalse(config['run'].exists())

    def test_default_main_does_not_run_or_write(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, 'ROOT', Path(folder)), patch.object(pipeline, 'execute') as execute:
            # configuration receives its stable module root in production;
            # no directory is created even with missing external OBJ inputs.
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(pipeline.main(['prepare']), 0)
            self.assertIn('PLAN ONLY', output.getvalue())
            execute.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_refuses_build_root_and_paths_outside_build(self):
        with tempfile.TemporaryDirectory() as folder:
            for path in ('build', 'data/scene', '../outside'):
                with self.assertRaises(ValueError):
                    self.config(Path(folder), 'prepare', '--run-root', path)
            for workers in ('-1', '0', '65'):
                with self.assertRaises(ValueError):
                    self.config(Path(folder), 'prepare', '--workers', workers)

    def test_finalize_inherits_prepare_configuration_and_uses_separate_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root/'build/ue_citycode_replay'; run.mkdir(parents=True)
            (run/'pipeline_prepare.json').write_text(json.dumps({'configuration': {'lod': '1M', 'binding': 'data/custom.json', 'workers': 7}}))
            config = self.config(root, 'finalize')
            steps = pipeline.build_plan(config)
            self.assertEqual(config['lod'], '1M'); self.assertEqual(config['workers'], 7)
            self.assertEqual(config['binding'], root/'data/custom.json')
            self.assertIn(str(run/'final/enrichment'), steps[0].argv)
            self.assertNotIn('review_street_props', [s.label for s in steps])
            self.assertIn('validate_complete_cpu_scene', [s.label for s in steps])
            labels = [s.label for s in steps]
            self.assertLess(labels.index('compile_portable_citycode'), labels.index('export_citycode_to_ue'))
            self.assertLess(labels.index('export_citycode_to_ue'), labels.index('validate_complete_cpu_scene'))
            self.assertIn(str(run/'final/assembly_candidate.json'), steps[0].argv)
            compiler = steps[labels.index('compile_portable_citycode')]
            self.assertIn(str(run/'network/network_connections.json'), compiler.argv)
            self.assertIn(str(run/'enrichment/props_final_groups.json'), compiler.argv)
            self.assertTrue(all(s.qa[0][0].is_relative_to(run/'final') for s in steps if s.qa))

    def test_qa_rejects_review_or_unknown_status_even_after_successful_process(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'qa.json'
            step = pipeline.Step('audit', [], qa=[(path, 'status', ('passed',))])
            for status in ('requires_review', 'requires_geometry_review', 'failed', None):
                path.write_text(json.dumps({'status': status}))
                with self.assertRaises(ValueError): pipeline.check_qa(step)
            path.write_text(json.dumps({'status': 'passed'})); pipeline.check_qa(step)

    def test_successful_process_with_review_result_stops_before_next_step(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(Path(folder)); qa = config['run']/'qa.json'
            calls = []
            def runner(argv, **kwargs):
                calls.append(argv)
                qa.write_text(json.dumps({'status': 'requires_review'}))
            steps = [pipeline.Step('audit', ['python', 'audit.py'], qa=[(qa, 'status', ('passed',))]), pipeline.Step('should_not_run', ['python', 'next.py'])]
            with patch.object(pipeline, 'preflight'), self.assertRaises(ValueError):
                pipeline.execute(config, steps, runner)
            self.assertEqual(len(calls), 1)
            self.assertEqual(pipeline.read(config['run']/'pipeline_prepare.json')['status'], 'failed')

    def test_subprocess_failure_is_recorded_and_not_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(Path(folder))
            def runner(argv, **kwargs): raise subprocess.CalledProcessError(4, argv)
            with patch.object(pipeline, 'preflight'), self.assertRaises(subprocess.CalledProcessError):
                pipeline.execute(config, [pipeline.Step('failing', ['python', 'x.py'])], runner)
            state = pipeline.read(config['run']/'pipeline_prepare.json')
            self.assertEqual(state['status'], 'failed'); self.assertEqual(state['active_step'], 'failing')

    def test_existing_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(Path(folder)); (config['run']/'base').mkdir(parents=True)
            marker = config['run']/'base/accepted'; marker.write_text('keep')
            with patch.object(pipeline, 'required_inputs', return_value=[]), self.assertRaisesRegex(ValueError, 'overwrite'):
                pipeline.preflight(config, [])
            self.assertEqual(marker.read_text(), 'keep')

    def test_native_import_must_be_real_and_match_this_run(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.config(Path(folder), 'finalize'); run = config['run']
            (run/'surfaces/joined').mkdir(parents=True); (run/'windows').mkdir()
            manifest = run/'surfaces/joined/span_manifest.json'; manifest.write_text('{}')
            expected = pipeline.digest(manifest)
            catalog = dict(status='native_bridge_template_imported', ue_executed=True, bounds_verified=True,
                native_materials_verified=True, assets=[{'template_manifest_sha256': expected}])
            config['native_catalog'].write_text(json.dumps(catalog))
            config['native_proposal'].write_text(json.dumps({'template_manifest_sha256': expected}))
            pipeline.verify_native_import(config)
            catalog['ue_executed'] = False; config['native_catalog'].write_text(json.dumps(catalog))
            with self.assertRaises(ValueError): pipeline.verify_native_import(config)
            catalog['ue_executed'] = True; config['native_catalog'].write_text(json.dumps(catalog))
            manifest.write_text('{"changed":true}')
            with self.assertRaises(ValueError): pipeline.verify_native_import(config)

    def test_finalize_refuses_a_modified_prepared_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); config=self.config(root, 'finalize'); run=config['run'];run.mkdir(parents=True)
            frozen=run/'frozen.json';frozen.write_text('{"ok":true}')
            state={'status':'prepared_cpu_checks_passed','configuration':{'lod':'0.25M','binding':str(pipeline.DEFAULT_BINDING)},
                   'frozen_files':{str(frozen.relative_to(root)):pipeline.digest(frozen)}}
            (run/'pipeline_prepare.json').write_text(json.dumps(state));frozen.write_text('{"changed":true}')
            with patch.object(pipeline, 'required_inputs', return_value=[]), self.assertRaisesRegex(ValueError,'Prepared input changed'):
                pipeline.preflight(config, [])


if __name__ == '__main__':
    unittest.main()
