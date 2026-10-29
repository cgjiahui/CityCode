#!/usr/bin/env python3
"""Run the CPU scene audit and replay from an unpacked release checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from citycode_scene.cli import main

if __name__ == '__main__':
    main()
