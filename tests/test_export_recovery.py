"""CPU low-export recovery requires exact current identities and an explicit review."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'workflow/02_building_assets/recover_exports.py'
spec=importlib.util.spec_from_file_location('release_export_recovery',path)
recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)


class RecoveryBoundary(unittest.TestCase):
    def test_missing_master_or_present_low_is_not_silently_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'assets/one';folder.mkdir(parents=True)
            raw=root/'generation/one';raw.mkdir(parents=True)
            (raw/'raw_manifest.json').write_text(json.dumps({'status':'complete','sha256':'raw',
                'contract':{'building_sha256':'source'}}))
            config={'output':str(root),'lod_targets':[1000000,250000,100000]}
            row={'asset_id':'one','building':{'sha256':'source'}}
            (folder/'model_025m.glb').write_bytes(b'current high')
            with self.assertRaisesRegex(ValueError,'exactly one missing'):
                recovery.recovery_inputs(config,row)
            (folder/'model_1m.glb').write_bytes(b'current master')
            with patch.object(recovery,'inspect_glb'):
                identity,_=recovery.recovery_inputs(config,row)
            self.assertEqual(set(identity['input_lods']),{'1M','0.25M'})
            (folder/'model_01m.glb').write_bytes(b'existing low')
            with self.assertRaisesRegex(ValueError,'exactly one missing'):
                recovery.recovery_inputs(config,row)

    def test_review_is_bound_to_current_high_raw_and_source(self):
        expected={'asset_id':'one','source_building_sha256':'source','raw_cache_sha256':'raw',
            'raw_manifest_sha256':'metadata','input_lods':{'0.25M':{'sha256':'actual high'}}}
        approved={**expected,'decision':'recover_0.1M_from_0.25M','reason':'Current high mesh visually checked'}
        recovery.verify_review(expected,approved)
        recovery.verify_review(expected,{**approved,'target_relative_error':.02})
        for key in ('source_building_sha256','raw_cache_sha256','raw_manifest_sha256','input_lods'):
            stale=copy.deepcopy(approved);stale[key]='different'
            with self.assertRaisesRegex(ValueError,'current source'):
                recovery.verify_review(expected,stale)
        with self.assertRaises(ValueError):recovery.verify_review(expected,{**approved,'decision':None})
        with self.assertRaisesRegex(ValueError,'1 or2 percent'):
            recovery.verify_review(expected,{**approved,'target_relative_error':.025})

    def test_simplification_error_budget_cannot_be_unbounded(self):
        from citycode_assets.extrema_lod import export_extrema_preserving_lod
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for error in (0,-1,.021,float('nan'),float('inf')):
                with self.assertRaisesRegex(ValueError,'at most2 percent'):
                    export_extrema_preserving_lod(root/'high.glb',root/'low.glb',fit_record={},target_relative_error=error)


if __name__=='__main__':unittest.main()
