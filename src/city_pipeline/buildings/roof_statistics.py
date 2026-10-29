"""Summarize observed predicted roof XYZ without any GT input."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import selection as base
from .consensus import install_read_guard


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reads, source_hash = install_read_guard(base.SOURCE_RUN_ROOT, args.output)
    fs = base.read_json(base.SOURCE_RUN_ROOT / 'global_fused/building_observation_footprints.geojson')['features']
    grouped = defaultdict(list)
    for f in fs:
        o = f['properties']; grouped[o['building_point_cloud_npz_path']].append(o)
    result = {}
    for path, observations in sorted(grouped.items()):
        with np.load(base.PROJECT_ROOT / path) as data:
            xyz, ids, provenance = data['xyz_m'], data['instance_id'], data['provenance']
        for o in observations:
            z = xyz[(ids == o['rgb_instance_id']) & (provenance == 1), 2]
            z = z[np.isfinite(z)]
            qs = np.quantile(z, [.90, .95, .98, .99, .995]) if len(z) >= 200 else [o['top_z_m']] * 5
            result[o['observation_id']] = {'observed_count': len(z),
                **{k: float(v) for k, v in zip(['q90','q95','q98','q99','q995'],qs)}}
    base.write_json(args.output, {'source_observations_sha256': source_hash,
        'method': 'quantiles of provenance=1 predicted E10 world Z; at least 200 points',
        'observations': result, 'runtime_read_audit': sorted(reads)})
    print('Predicted XYZ statistics:', len(result), 'observations from', len(grouped), 'patches')


if __name__ == '__main__':
    main()
