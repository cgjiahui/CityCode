#!/usr/bin/env python3
"""Portable RGB + two-layer teacher -> Pixal3D -> continuous city background.

prepare is CPU-only. plan prints commands. generate explicitly runs GPU work.
No stage submits image API requests or downloads models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
V9 = HERE / 'experiments/pixal3d_p384_handoff_v9'
V5 = HERE / 'experiments/pixal3d_p384_v5'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def safe_child(root, name):
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside input directory: {name}')
    return result


def prepare(args):
    package = args.inputs_package.resolve()
    bundle = args.bundle.resolve()
    out = args.output.resolve()
    if out.exists() or any(out.is_relative_to(p) for p in (package, bundle)):
        raise ValueError('Use a new output outside the packaged input directories')
    manifest = read(package / 'manifest.json')
    rgb = safe_child(bundle, manifest['rgb_bundle_relative'])
    checked(rgb, manifest['rgb_sha256'])
    checked(package / 'teacher_additions.zip', manifest['teacher_archive_sha256'])
    index = read(package / 'patch_index.json')
    ids = {p['patch_id'] for p in index['patches']}
    if set(args.patch_id or []) - ids:
        raise ValueError('Unknown patch ID')
    out.mkdir(parents=True)
    teacher = out / 'teacher'
    teacher.mkdir()
    # Duplicate arrays remain read-only references to the original bundle.
    with zipfile.ZipFile(package / 'teacher_additions.zip') as archive:
        for name, info in manifest['teacher_files'].items():
            target = safe_child(teacher, name)
            if 'bundle_relative' in info:
                source = safe_child(bundle, info['bundle_relative'])
                checked(source, info['sha256'])
                target.symlink_to(source)
            else:
                with archive.open(info['archive_member']) as src, target.open('xb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
                checked(target, info['sha256'])
    for name in ('manifest.json', 'pixel_to_world.json'):
        (teacher / name).write_bytes((package / ('teacher_' + name)).read_bytes())
    layout = out / '_input_layout'
    rgb_link = layout / 'handoff/01_global/rgb/background_rgb.png'
    rgb_link.parent.mkdir(parents=True)
    rgb_link.symlink_to(rgb)
    dataset = layout / 'dataset'
    dataset.mkdir()
    (dataset / 'patch_index.json').write_text(json.dumps(index))
    spec = importlib.util.spec_from_file_location('portable_prepare', HERE / 'prepare_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare_inputs(SimpleNamespace(
        inputs_output=out / 'inputs', teacher_output=teacher,
        dataset=dataset, handoff=layout / 'handoff', patch_id=args.patch_id,
    ), read(teacher / 'manifest.json'))
    prepared = read(out / 'inputs/manifest.json')
    result = {'status': 'prepared', 'patch_count': len(prepared['patches']),
              'lineage': 'historical_gt_assisted', 'api_calls': 0,
              'gpu_inference': False, 'teacher_source': str(package)}
    (out / 'PREPARATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def model_paths(args):
    root = args.models_root.resolve()
    return {'model': root / 'Pixal3D', 'dino': root / 'dinov3-vitl16-pretrain-lvd1689m',
            'moge': root / 'moge-2-vitl', 'birefnet': root / 'BiRefNet',
            'naf': RELEASE / 'third_party/NAF'}


def preflight(args):
    paths = model_paths(args)
    required = [paths['model'] / 'pipeline.json', paths['model'] / 'ckpts',
                paths['dino'] / 'config.json', paths['moge'] / 'model.pt',
                paths['birefnet'] / 'config.json', paths['naf'] / 'hubconf.py',
                args.torch_cache / 'hub/checkpoints/naf_release.pth']
    config = paths['model'] / 'pipeline.json'
    if config.is_file():
        for stem in read(config)['args']['models'].values():
            required.extend([safe_child(paths['model'], stem + suffix)
                             for suffix in ('.json', '.safetensors')])
    for key in ('dino', 'birefnet'):
        if not list(paths[key].glob('*.safetensors')) and not list(paths[key].glob('*.bin')):
            required.append(paths[key] / 'model.safetensors')
    missing = [str(p) for p in required if not p.exists()]
    modules = ['torch', 'torchvision', 'cv2', 'numpy', 'scipy', 'trimesh',
               'transformers', 'nvdiffrast', 'cumesh', 'o_voxel', 'flex_gemm', 'moge',
               'natten', 'safetensors', 'diffusers', 'timm', 'einops', 'utils3d']
    sys.path.insert(0, str(RELEASE / 'third_party/MoGe'))
    backend = os.environ.get('ATTN_BACKEND', 'flash_attn_3')
    if backend in ('flash_attn', 'flash_attn_3'):
        modules.append('flash_attn_interface' if backend == 'flash_attn_3' else 'flash_attn')
    absent = [m for m in modules if importlib.util.find_spec(m) is None]
    return {'status': 'prerequisites_present' if not missing and not absent else 'missing_prerequisites',
            'missing_paths': missing, 'missing_modules': absent,
            'gpu_execution_validated': False,
            'note': 'Presence check only; CUDA extensions, GPU architecture and complete local model weights must also be compatible.'}


def commands(args):
    root = args.work.resolve()
    inputs, teacher, runs = root / 'inputs', root / 'teacher', root / 'runs'
    fusion, final, validation = root / 'fusion', root / 'continuous', root / 'validation'
    py = sys.executable
    def cmd(script, *parts):
        return [py, str(script), *map(str, parts)]
    generate = cmd(V9 / 'run_patch.py', '--inputs', inputs, '--teacher', teacher,
                   '--output', runs, '--mode', 'structure_guided', '--resolution', 1536,
                   '--hr-surface-band-voxels', 1, '--height-projection-residual', 0.05,
                   '--texture-size', 4096, '--decimation-target', 1000000)
    for key, value in model_paths(args).items():
        generate += ['--' + key, str(value)]
    return [
        generate,
        cmd(V9 / 'canonicalize_all.py', '--runs', runs, '--inputs', inputs, '--teacher', teacher, '--residual', 0.01),
        cmd(V9 / 'fuse_global_tiles.py', '--runs', runs, '--inputs', inputs, '--output', fusion),
        cmd(V9 / 'validate_global_scene.py', '--runs', runs, '--inputs', inputs, '--teacher', teacher, '--output', validation),
        cmd(V9 / 'add_continuous_base_fill.py', '--source-fusion', fusion, '--inputs', inputs, '--output', final, '--grid-m', 1, '--full-underlay', '--underlay-offset-m', 0.03),
        cmd(V5 / 'merge_fused_tiles_glb.py', '--fusion', final),
        cmd(V9 / 'build_rasterized_top_preview.py', '--fusion', final, '--output', final / 'full_scene_top_surface_2m.glb', '--bin-m', 2),
        cmd(V5 / 'convert_fused_tiles_meshlab.py', '--fusion', final),
        cmd(V5 / 'convert_glb_textures.py', final / 'full_scene_single_owner.glb', final / 'full_scene_single_owner_meshlab.glb', '--jpeg-quality', 95),
        cmd(V9 / 'accept_continuous_v2.py', '--fusion', final, '--reference-validation', validation / 'validation.json', '--output', root / 'acceptance'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare', help='CPU-only: rebuild generation inputs from frozen RGB and teacher')
    prep.add_argument('--bundle', type=Path, default=RELEASE / 'data/smallcity_p384_background_history')
    prep.add_argument('--inputs-package', type=Path, default=RELEASE / 'data/smallcity_p384_background_pixal3d_inputs')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--patch-id', action='append')
    for name in ('check', 'plan', 'generate'):
        sub = subs.add_parser(name)
        sub.add_argument('--models-root', type=Path, default=RELEASE / 'models/background_pixal3d')
        sub.add_argument('--torch-cache', type=Path, default=RELEASE / 'models/background_pixal3d/torch_cache')
        if name != 'check':
            sub.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'check':
        result = preflight(args)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'prerequisites_present' else 2
    elif args.command == 'plan':
        print(json.dumps({'commands': commands(args), 'requires_gpu': True, 'api_calls': 0}, indent=2))
    else:
        report = preflight(args)
        if report['status'] != 'prerequisites_present':
            print(json.dumps(report, indent=2))
            return 2
        prepared = read(args.work / 'PREPARATION.json')
        if prepared['patch_count'] != 79:
            raise ValueError('The full-scene acceptance contract requires all 79 prepared patches')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', WANDB_MODE='offline',
                   TORCH_HOME=str(args.torch_cache.resolve()),
                   FLEX_GEMM_AUTOTUNE_CACHE_PATH=str(args.work.resolve() / 'autotune_cache.json'))
        env['PYTHONPATH'] = os.pathsep.join([str(HERE / 'offline_guard'), str(RELEASE / 'third_party/MoGe'), env.get('PYTHONPATH', '')])
        for command in commands(args):
            subprocess.run(command, env=env, cwd=HERE, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
