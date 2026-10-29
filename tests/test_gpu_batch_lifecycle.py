"""The default task may finish after any wall time; explicit limits still stop it."""
import json
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from citycode_assets import batch_gpu, batch_resources


class GPUTimePolicy(unittest.TestCase):
    def execute(self, deadline):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'FROZEN.json').write_text('{}')
            (root/'assets/one').mkdir(parents=True)
            (root/'assets/one/export_manifest.json').write_text('{}')
            config = {'output': str(root), '_config_path': str(root/'config.json')}
            children = []

            def launch(*args, **kwargs):
                class Child:
                    pid = 987654321
                    polls = 0

                    def poll(self):
                        self.polls += 1
                        return None if self.polls < 4 else 0

                    def wait(self, timeout=None):
                        return 0
                child = Child(); children.append(child)
                return child

            ticks = iter([0]+[50000+i for i in range(100)])
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
            with patch.dict('sys.modules', {'torch': torch}), \
                 patch.object(batch_gpu, 'frozen_selection', return_value=[{'asset_id': 'one'}]), \
                 patch.object(batch_resources, 'verify_runtime_contract'), \
                 patch.object(batch_gpu.subprocess, 'Popen', side_effect=launch), \
                 patch.object(batch_gpu.time, 'monotonic', side_effect=lambda: next(ticks)), \
                 patch.object(batch_gpu.time, 'sleep'), \
                 patch.object(batch_gpu.os, 'killpg') as kill:
                result = batch_gpu.run_gpu_batch(config, gpu_count=1, deadline_seconds=deadline)
                return result, json.loads((root/'execution/PROGRESS.json').read_text()), kill.call_count, len(children)

    def test_default_allows_completion_after_more_than_twelve_hours(self):
        result, progress, kills, processes = self.execute(None)
        self.assertFalse(progress['stopped'])
        self.assertEqual([p['phase'] for p in progress['phases']], ['infer', 'export'])
        self.assertEqual(kills, 0)
        self.assertEqual(processes, 2)
        self.assertTrue(result['allocation_should_terminate_now'])

    def test_explicit_requested_deadline_stops_before_export(self):
        _, progress, kills, processes = self.execute(60)
        self.assertTrue(progress['stopped'])
        self.assertEqual(processes, 1)
        self.assertGreater(kills, 0)

    def test_shell_entry_only_passes_requested_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);python=root/'python'
            python.write_text('#!/bin/bash\nif [[ "$1" == "-c" ]]; then echo "$CITYCODE_TEST_OUT"; else printf "%s\\n" "$@"; fi\n')
            python.chmod(0o755)
            script=Path(__file__).resolve().parents[1]/'workflow/02_building_assets/run_gpu.sh'
            env=dict(os.environ,CITYCODE_PYTHON=str(python),CITYCODE_TEST_OUT=str(root/'output'))
            default=subprocess.run(['bash',str(script),'config.json','2'],env=env,capture_output=True,text=True,check=True)
            explicit=subprocess.run(['bash',str(script),'config.json','2','600'],env=env,capture_output=True,text=True,check=True)
            self.assertNotIn('--deadline-seconds',default.stdout)
            self.assertIn('--deadline-seconds\n600',explicit.stdout)


if __name__ == '__main__':
    unittest.main()
