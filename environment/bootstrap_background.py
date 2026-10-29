"""Install the frozen background runtime from local wheels without networking."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv
import re


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv', type=Path, default=root/'environment/.venv-background')
    args = p.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Bundled Linux wheels target Python 3.10. Use a Python 3.10 interpreter, or install the background extra normally in your own environment.')
    target = args.venv.resolve()
    if not (target/'bin/python').exists():
        venv.EnvBuilder(with_pip=True).create(target)
    version = re.search(r'(?m)^version = "([^"]+)"', (root/'pyproject.toml').read_text()).group(1)
    wheel = root/f'dist/citycode_perception-{version}-py3-none-any.whl'
    if not wheel.exists():
        raise FileNotFoundError('Build the project wheel first; no download fallback is used')
    subprocess.run([str(target/'bin/python'), '-m', 'pip', 'install', '--no-index',
                    '--constraint', str(root/'environment/background.requirements.txt'),
                    '--find-links', str(root/'environment/wheels'),
                    '--find-links', str(root/'environment/background_wheels'),
                    str(wheel)+'[background]'], check=True)
    subprocess.run([str(target/'bin/python'), '-m', 'citycode_background', '--help'], check=True)
    print(f'Background runtime ready: {target}/bin/city-background')


if __name__ == '__main__':
    main()
