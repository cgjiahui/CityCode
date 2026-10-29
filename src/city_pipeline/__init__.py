"""Portable structure reconstruction; CityCode conversion is in citycode_core."""
import os
import sys
from pathlib import Path

# Defense against accidental research-checkout imports after extraction.
class _RejectResearchImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'bev_recovery' or fullname.startswith('bev_recovery.'):
            raise ImportError('Research checkout imports are forbidden: ' + fullname)
        return None

sys.meta_path.insert(0, _RejectResearchImports())

if os.environ.get('CITY_PIPELINE_NO_GT_DATASET'):
    _dataset = Path(os.environ['CITY_PIPELINE_NO_GT_DATASET']).resolve()
    def _guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        mode, flags = args[1:3]
        if isinstance(mode, str) and any(x in mode for x in 'wax'): return
        if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR): return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(_dataset) and p.name not in {'rgb.png','camera.json','patch.json','strict_valid_mask.png','priors.json','patch_index.json','dataset_manifest.json','checksums.sha256'}:
            raise PermissionError('Inference annotation read forbidden: '+str(p))
    sys.addaudithook(_guard)
