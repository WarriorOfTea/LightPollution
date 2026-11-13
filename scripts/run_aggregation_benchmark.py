"""Measure existing source aggregation error with fixed power and atmosphere."""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.data import load_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, Sources, integrate_direction
from lightpollution.sources import regional_sources

ROOT = Path(__file__).resolve().parents[1]


def response(observer, sources, az, alt, settings):
    # Fixed chunking limits memory for the finer source partition. It changes
    # only floating-point summation grouping, not geometry or power.
    result = 0.
    for start in range(0, len(sources.power), 512):
        chunk = Sources(sources.positions_m[start:start+512], sources.power[start:start+512])
        result += integrate_direction(observer, chunk, az, alt, Atmosphere(), settings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113/aggregation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observer = Observer(40.451111, -3.726111, 650.)
    protocol = {'id': 'LP-source-aggregation-v1',
        'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'observer': asdict(observer), 'atmosphere': asdict(Atmosphere()), 'periods': ['2017-03', '2017-11'],
        'blocks_pixels': [12, 6, 3], 'directions_az_alt_deg': [[150, 10], [0, 90]],
        'chunk_sources': 512, 'settings': asdict(SolverSettings(192, 24)),
        'fixed_C_power_mask_and_all_other_inputs': True, 'DSM_and_finite_local_cells': False,
        'reference': 'block=3 plus numerical refinement; not an exact spatial solution or a larger region.',
        'no_residual_based_case_selection': True, 'physical_order': 1,
        'code_sha256': {p: file_hash(ROOT/p) for p in ['scripts/run_aggregation_benchmark.py',
            'src/lightpollution/sources.py', 'src/lightpollution/physics.py']}}
    write_json(args.output/'protocol.json', protocol)
    records = []
    for period in protocol['periods']:
        cache = ROOT/f'data/seasonal-ground-20261110/madrid-matched-2017/{period}/source-grid.npz'
        arrays, bounds, meta = load_region(cache)
        for az, alt in protocol['directions_az_alt_deg']:
            variants = []
            for block in protocol['blocks_pixels']:
                sources, provenance = regional_sources(arrays, bounds, observer, block=block,
                    normalisation_radiance_area=meta['reference_radiance_area_sum'])
                t = perf_counter()
                value = response(observer, sources, az, alt, SolverSettings(192, 24))
                seconds = perf_counter()-t
                refined = response(observer, sources, az, alt, SolverSettings(384, 48)) if block == 3 else None
                variants.append({'block': block, 'effective_sources': len(sources.power), 'total_power': float(sources.power.sum()),
                    'radiance': value, 'runtime_seconds': seconds, 'refined_radiance': refined,
                    'source_selection': provenance})
            reference = variants[-1]['radiance']
            for v in variants:
                v['relative_to_block_3_minus_one'] = v['radiance']/reference-1
            record = {'period': period, 'azimuth_deg': az, 'altitude_deg': alt, 'cache_sha256': file_hash(cache),
                'variants': variants, 'block3_ray_path_refinement_relative': abs(reference/variants[-1]['refined_radiance']-1),
                'power_preserved_relative_span': float(np.ptp([v['total_power'] for v in variants])/variants[-1]['total_power'])}
            records.append(record)
            write_json(args.output/f'{period}-alt-{alt}.json', record)
    summary = {'geometries': len(records), 'comparisons': len(records)*2,
        'max_abs_block_12_vs_3_relative': max(abs(r['variants'][0]['relative_to_block_3_minus_one']) for r in records),
        'max_abs_block_6_vs_3_relative': max(abs(r['variants'][1]['relative_to_block_3_minus_one']) for r in records),
        'max_ray_path_refinement_relative': max(r['block3_ray_path_refinement_relative'] for r in records),
        'max_power_relative_span': max(r['power_preserved_relative_span'] for r in records),
        'does_not_test_missing_nearby_sources_or_outside_ROI': True}
    write_json(args.output/'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
