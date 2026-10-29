#!/usr/bin/env python3
"""Source-checkout entry for the portable CityCode compiler and UE adapter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from citycode_core.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
