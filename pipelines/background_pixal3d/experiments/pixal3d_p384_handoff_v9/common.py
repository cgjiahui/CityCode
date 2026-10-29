from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
OUTPUTS = EXPERIMENT / "outputs"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def smoothstep01(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def halo_weight(height: int, width: int, meters_per_pixel: float, halo_m: float) -> np.ndarray:
    rows = np.minimum(np.arange(height) + 0.5, height - np.arange(height) - 0.5)
    columns = np.minimum(np.arange(width) + 0.5, width - np.arange(width) - 0.5)
    distance_m = np.minimum(rows[:, None], columns[None, :]) * meters_per_pixel
    return smoothstep01(distance_m / halo_m).astype(np.float32)




