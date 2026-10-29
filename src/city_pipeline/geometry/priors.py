"""Explicit numerical plane priors, kept separate from measured elevations."""
from pathlib import Path
import math

from city_pipeline.runtime import read_json


def read_plane_priors(path):
    path = Path(path).resolve()
    doc = read_json(path)
    if doc.get('unit') not in {'metre', 'meter', 'm'}:
        raise ValueError('Plane priors must declare metre units')
    planes = doc.get('predicted_background_semantic_id_to_prior_z_m')
    if not isinstance(planes, dict) or set(planes) != {'1', '2', '3'}:
        raise ValueError('Plane priors must specify predicted road/water/hardscape classes 1/2/3')
    result = {int(k): float(v) for k, v in planes.items()}
    if any(isinstance(v, bool) for v in planes.values()) or not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Plane priors must be finite scalar elevations')
    return result
