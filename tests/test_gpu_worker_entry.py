"""Launch the actual nested export module without GPU work or model weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from citycode_assets import batch
from citycode_assets.appearance import sha256_file, write_json
import test_unified_batch as fixtures
from test_asset_delivery_contract import cube_glb


class GPUWorkerModuleEntry(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UnifiedBuildingBatch()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, _ = self.fixture.completed_review()
        project = self.fixture.project
        (project/'trellis/trellis2').mkdir(parents=True)
        runtime = batch.read(self.fixture.config['runtime_config'])
        runtime.update(trellis_root='trellis', model_package_root='.')
        write_json(self.fixture.config['runtime_config'], runtime)
        config = {key: value for key, value in self.fixture.config.items() if not key.startswith('_')}
        config['project_root'] = '.'
        self.config_path = project/'configuration.json'
        write_json(self.config_path, config)
        batch.freeze(self.fixture.config)
        raw = self.root/'generation/A_B/raw_mesh.pt'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'CPU fixture raw identity; no torch.load expected for verified existing export')
        write_json(raw.with_name('raw_manifest.json'), {'status': 'complete', 'sha256': sha256_file(raw)})
        lods = {}
        for label, filename in [('0.25M', 'model_025m.glb'), ('0.1M', 'model_01m.glb')]:
            path = self.root/'assets/A_B'/filename
            cube_glb(path)
            lods[label] = batch.proof(path, self.root)
        self.lods = lods
        write_json(self.root/'assets/A_B/export_manifest.json', {
            'status': 'completed', 'raw_cache_sha256': sha256_file(raw), 'lods': lods})
        (self.root/'execution/logs').mkdir(parents=True)

    def launch(self):
        # The parent itself runs with __name__ == '__main__'. Its export_rank
        # must launch the real package module again rather than -m __main__.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                   PYTHONPATH=str(Path(batch.__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH', ''))
        return subprocess.run([sys.executable, '-u', '-m', 'citycode_assets.batch_gpu',
            '--config', str(self.config_path), '--mode', 'export', '--rank', '0', '--gpu-count', '1'],
            cwd=self.fixture.project, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60)

    def test_nested_module_entry_reuses_verified_export_without_cuda(self):
        before = {row['path']: sha256_file(self.root/row['path']) for row in self.lods.values()}
        result = self.launch()
        log = (self.root/'execution/logs/export_A_B.log').read_text()
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+log)
        self.assertNotIn('__main__.__spec__', log)
        records = batch.read(self.root/'execution/export_rank0.json')['records']
        self.assertEqual(records, [{'asset_id': 'A_B', 'exit_code': 0, 'status': 'complete'}])
        self.assertEqual(before, {row['path']: sha256_file(self.root/row['path']) for row in self.lods.values()})

    def test_real_nested_entry_still_rejects_changed_raw_bytes(self):
        raw = self.root/'generation/A_B/raw_mesh.pt'
        raw.write_bytes(b'tampered')
        result = self.launch()
        log = (self.root/'execution/logs/export_A_B.log').read_text()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Raw generation bytes changed', log)
        self.assertNotIn('__main__.__spec__', log)
        self.assertEqual(batch.read(self.root/'execution/export_rank0.json')['records'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
