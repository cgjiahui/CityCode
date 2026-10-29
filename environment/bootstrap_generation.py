#!/usr/bin/env python3
"""Install the pinned local wheelhouse into a new isolated Python3.10 venv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,10) or platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('This frozen wheelhouse requires Python3.10 on Linux x86_64')
    project=Path(__file__).resolve().parents[1]
    manifest_path=project/'environment/OFFLINE_WHEELS.json'
    manifest=json.loads(manifest_path.read_text())
    for row in manifest['files']:
        path=(project/row['path']).resolve()
        if (Path(row['path']).is_absolute() or not path.is_relative_to(project/'environment/wheels')
                or not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']):
            raise ValueError('Offline dependency missing or changed: '+row['path'])
    if args.check_only:
        print(json.dumps({'status':'offline_wheels_verified','files':len(manifest['files'])}));return 0
    if args.venv is None:parser.error('--venv is required for installation')
    destination=args.venv.resolve();marker=destination/'CITYCODE_ENVIRONMENT.json'
    identity={'offline_wheels_sha256':sha(manifest_path),
              'requirements_sha256':sha(project/'environment/generation.requirements.txt')}
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get('identity')!=identity:
            raise ValueError('Choose a new empty directory or resume the same CityCode environment')
    destination.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps({'status':'installing','identity':identity},indent=2)+'\n')
    venv.EnvBuilder(with_pip=True,clear=False,symlinks=True).create(destination)
    python=destination/'bin/python'
    env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_NO_CACHE_DIR='1',
             PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None)
    subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(project/'environment/wheels'),
        '-r',str(project/'environment/generation.requirements.txt')],env=env,check=True)
    subprocess.run([str(python),'-m','pip','install','--no-index','--no-build-isolation','--no-deps',
        '-e',str(project)],env=env,check=True)
    subprocess.run([str(python),'-m','pip','check'],env=env,check=True)
    subprocess.run([str(python),'-c','import torch, numpy, shapely, cv2, ninja; '
        'assert torch.__version__ == "2.6.0+cu124"; assert shapely.__version__ == "2.1.2"; '
        'assert ninja.BIN_DIR; print("Core imports and Ninja executable verified; GPU load still pending")'],
        env=env,check=True)
    subprocess.run([str(python),str(project/'environment/check_generation_imports.py')],
        env=env,check=True)
    marker.write_text(json.dumps({'status':'installed_CPU_verified_GPU_pending','identity':identity,
        'python':sys.version.split()[0],'isolated_system_site_packages':False,
        'GPU_load_verified':False},indent=2)+'\n')
    print(json.dumps({'status':'installed_CPU_verified_GPU_pending','venv':str(destination)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
