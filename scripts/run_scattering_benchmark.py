"""Freeze and run controlled 1/2/3-order comparisons, never a field accuracy score."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.models.scattering_orders import LambertianCap, OrderSettings, integrate_orders
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer, SolverSettings, integrate_direction

ROOT = Path(__file__).resolve().parents[1]


def contrast(result, numerator, denominator):
    """Paired ratio of cumulative means; use their full covariance."""
    means = np.asarray(result['exact_order_radiance'])
    covariance = np.asarray(result['mean_covariance'])
    a = (np.arange(len(means)) < numerator).astype(float)
    b = (np.arange(len(means)) < denominator).astype(float)
    x, y = a @ means, b @ means
    if y <= 0:
        return {'relative_increment': None, 'standard_error': None}
    gradient = a/y - b*x/y**2
    se = float(np.sqrt(max(0, gradient @ covariance @ gradient)))
    return {'relative_increment': float(x/y-1), 'standard_error': se,
        'approximate_95_interval': [float(x/y-1-1.96*se), float(x/y-1+1.96*se)]}


def scene():
    return Observer(0., 0., 20.), LambertianCap(-np.rad2deg(60000/EARTH_RADIUS_M), 0., 40000., 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113/scattering')
    parser.add_argument('--samples', type=int, default=2097152)
    parser.add_argument('--worker-order', type=int)
    parser.add_argument('--seed', type=int, default=3011)
    args = parser.parse_args()
    observer, cap = scene()
    if args.worker_order:
        value = integrate_orders(observer, cap, 180., 45., Atmosphere(),
            OrderSettings(samples=args.samples, max_order=args.worker_order, seed=args.seed))
        # Linux ru_maxrss can preserve a pre-exec high-water value from a
        # large parent. /proc/self/status refers to THIS executable's lifetime.
        status = Path('/proc/self/status')
        value['rusage_maxrss_kib_including_possible_pre_exec_history'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        value['peak_rss_kib'] = (int(next(line.split()[1] for line in status.read_text().splitlines()
            if line.startswith('VmHWM:'))) if status.exists() else None)
        value['peak_rss_method'] = '/proc/self/status VmHWM; whole worker including Python/imports'
        print(json.dumps(value))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'id': 'LP-clean-orders-v1', 'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'purpose': 'Paired physical sensitivity; not agreement with measurements.',
        'observer': asdict(observer), 'cap': asdict(cap), 'azimuth_deg': 180.,
        'samples_per_case': args.samples, 'seed': args.seed, 'batch_size': 8192,
        'aod_values': [.02, .10, .30], 'altitudes_deg': [10., 45., 90.],
        'base_atmosphere': asdict(Atmosphere()), 'changed_parameters': ['aerosol_optical_depth', 'altitude_deg', 'max_order'],
        'fixed_parameters': ['source power and footprint', 'wavelength 550 nm', 'g', 'SSA', 'all scale heights', 'observer'],
        'absent_processes': ['cloud', 'Moon', 'Sun', 'airglow', 'snow', 'ground reflection', 'DSM', 'lamp spectra', 'camera', 'eye'],
        'reference_checks': ['Independent production first order with area/ray/path refinement.',
            'Independent seed for AOD=.10, altitude=45.', 'Orders 4/5 at AOD=.10/.30, altitude=45.'],
        'acceptance_before_results': {'first_order_difference_max_standard_errors': 4.,
            'deterministic_refinement_relative_max': .005, 'normal_interval_min_luminous_paths': 100},
        'timing': {'fresh_processes': True, 'samples': 524288, 'repeats_per_order': 2,
            'includes_imports': False, 'memory_includes_imports': True,
            'memory_method': '/proc/self/status VmHWM, excluding pre-exec parent high-water',
            'same_sample_count_not_same_error': True},
        'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count()},
        'source_sha256': {p: file_hash(ROOT/p) for p in [
            'src/lightpollution/physics.py', 'src/lightpollution/models/scattering_orders.py',
            'scripts/run_scattering_benchmark.py']}}
    write_json(args.output/'protocol.json', protocol)
    rows = []
    for tau in protocol['aod_values']:
        atm = replace(Atmosphere(), aerosol_optical_depth=tau)
        for alt in protocol['altitudes_deg']:
            label = f'aod-{tau:.2f}-alt-{int(alt)}'
            result = integrate_orders(observer, cap, 180., alt, atm,
                OrderSettings(samples=args.samples, seed=args.seed))
            references = []
            for nr, nphi, segments, path in [(12, 32, 256, 24), (24, 64, 512, 48), (32, 96, 1024, 64)]:
                value = integrate_direction(observer, cap.quadrature_sources(nr, nphi), 180., alt,
                    atm, SolverSettings(segments, path))
                references.append({'surface_radial_order': nr, 'surface_azimuth_count': nphi,
                    'ray_segments': segments, 'path_quadrature': path, 'radiance': value})
            ref = references[-1]['radiance']
            se = result['exact_order_standard_error'][0]
            result.update(case_id=label, deterministic_first_order=references,
                first_order_z=(result['exact_order_radiance'][0]-ref)/se if se else None,
                reference_refinement_relative=abs(references[-2]['radiance']/ref-1),
                cumulative_two_over_one=contrast(result, 2, 1), cumulative_three_over_two=contrast(result, 3, 2))
            write_json(args.output/f'{label}.json', result)
            rows.append(result)
            print(json.dumps({'case_id': label, 'hits': result['luminous_paths_by_order'],
                'first_order_z': result['first_order_z'], '2/1-1': result['cumulative_two_over_one'],
                '3/2-1': result['cumulative_three_over_two'], 'seconds': result['runtime_seconds']}), flush=True)
    extras = []
    for tau in [.1, .3]:
        value = integrate_orders(observer, cap, 180., 45., replace(Atmosphere(), aerosol_optical_depth=tau),
            OrderSettings(samples=args.samples, max_order=5, seed=args.seed))
        value['cumulative_five_over_three'] = contrast(value, 5, 3)
        value['case_id'] = f'tail-aod-{tau:.2f}-alt-45'
        extras.append(value)
        write_json(args.output/f"{value['case_id']}.json", value)
        print(json.dumps({'tail': tau, '5/3-1': value['cumulative_five_over_three'], 'hits': value['luminous_paths_by_order']}), flush=True)
    repeat = integrate_orders(observer, cap, 180., 45., Atmosphere(),
        OrderSettings(samples=args.samples, seed=args.seed+1))
    write_json(args.output/'independent-seed.json', repeat)
    timing = []
    for seed in [701, 702]:
        for order in [1, 2, 3]:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-order', str(order),
                '--samples', '524288', '--seed', str(seed)], capture_output=True, text=True, check=True)
            value = json.loads(process.stdout)
            timing.append(value)
            print(json.dumps({'timing_order': order, 'seed': seed, 'seconds': value['runtime_seconds'], 'rss_kib': value['peak_rss_kib']}), flush=True)
    write_json(args.output/'timing.json', {'runs': timing, 'hardware': protocol['hardware']})
    baseline = next(r for r in rows if r['case_id'] == 'aod-0.10-alt-45')
    differences = np.asarray(repeat['exact_order_radiance'])-baseline['exact_order_radiance']
    combined_se = np.sqrt(np.asarray(repeat['exact_order_standard_error'])**2+np.asarray(baseline['exact_order_standard_error'])**2)
    summary = {'protocol_sha256': file_hash(args.output/'protocol.json'), 'cases': len(rows),
        'total_paths_including_tail_repeat_and_timing': (len(rows)+len(extras)+1)*args.samples+6*524288,
        'all_first_orders_within_four_standard_errors': all(abs(r['first_order_z']) <= 4 for r in rows),
        'max_first_order_abs_z': max(abs(r['first_order_z']) for r in rows),
        'max_deterministic_relative_refinement': max(r['reference_refinement_relative'] for r in rows),
        'min_luminous_paths_per_exact_order_main_cases': min(min(r['luminous_paths_by_order']) for r in rows),
        'independent_seed_exact_order_z': (differences/combined_se).tolist(),
        'second_order_relative_increment_range': [min(r['cumulative_two_over_one']['relative_increment'] for r in rows),
                                                  max(r['cumulative_two_over_one']['relative_increment'] for r in rows)],
        'third_order_relative_increment_range': [min(r['cumulative_three_over_two']['relative_increment'] for r in rows),
                                                 max(r['cumulative_three_over_two']['relative_increment'] for r in rows)],
        'field_validation': False, 'production_model_replaced': False}
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
