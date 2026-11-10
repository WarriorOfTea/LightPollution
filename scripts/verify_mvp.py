"""Reproduce MVP cases and declared-ray convergence on pinned local NASA inputs."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import resource
import time

import numpy as np

from lightpollution.adapters.storage import write_json
from lightpollution.application.planner import Planner, compare_results, export_result
from lightpollution.domain.scene import DomainError, SceneSpec
from lightpollution.interfaces.commands import offline_example
from lightpollution.physics import SolverSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('data/mvp-verification-20261111'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/mvp-release-20261111'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    planner = Planner(args.workspace)
    requests = {'offline-demo': offline_example()}
    peter = deepcopy(offline_example())
    peter.update(region='petersburg-2025', period='2025')
    requests['petersburg-annual'] = peter
    for month in ['2017-03', '2017-11']:
        value = deepcopy(offline_example())
        value.update(region='madrid-2017', period=month, time='2017-11-19T22:30:00Z',
                     observer={'latitude_deg': 40.451111, 'longitude_deg': -3.726111, 'height_m': 20})
        requests['madrid-' + month] = value
    results, timings = {}, {}
    for name, request in requests.items():
        start = time.perf_counter()
        first = planner.run(SceneSpec.from_mapping(request))
        first_seconds = time.perf_counter() - start
        second = planner.run(SceneSpec.from_mapping(request))
        assert second.value['cache']['scene_hit']
        assert first.value['sky'] == second.value['sky']
        export_result(first, args.output / name)
        timings[name] = {'first_seconds': first_seconds,
                         'repeat_seconds': second.value['runtime']['elapsed_seconds'],
                         'cache': first.value['cache'], 'repeat_scene_hit': True}
        results[name] = first.value
        print(name, timings[name], flush=True)
    seasonal = compare_results(results['madrid-2017-03'], results['madrid-2017-11'])
    write_json(args.output / 'seasonal-comparison.json', seasonal)
    changed = deepcopy(peter)
    changed['atmosphere'] = {'aerosol_optical_depth': .2}
    other = planner.run(SceneSpec.from_mapping(changed))
    assert other.value['calibration']['source_coefficient'] == results['petersburg-annual']['calibration']['source_coefficient']
    export_result(other, args.output / 'petersburg-aod-02')
    write_json(args.output / 'atmosphere-comparison.json', compare_results(results['petersburg-annual'], other.value))
    for label, request in [('twilight', {**peter, 'time': '2026-06-21T12:00:00Z'}),
                           ('empty-monthly-QA', {**peter, 'region': 'petersburg-monthly-2025', 'period': '2025-07'})]:
        try:
            planner.run(SceneSpec.from_mapping(request))
        except DomainError as exc:
            write_json(args.output / (label + '.json'), exc.to_dict())
        else:
            raise AssertionError('Unsupported case was accepted: ' + label)
    spec = SceneSpec.from_mapping(peter)
    group = planner.sources.prepare('petersburg-2025', '2025', spec.observer)
    convergence = []
    # This threshold is fixed before the comparison; it is not a physical error bound.
    tolerance = .01
    for az, alt in [(0, 10), (180, 10), (180, 45), (0, 90)]:
        direction = (az, alt) if alt <= 10 else None
        sources, _ = group.select('2025', 12, direction)
        refined, _ = group.select('2025', 6, direction)
        baseline = planner.solver.direction(spec.observer, sources, az, alt, spec.atmosphere, SolverSettings(192, 24))
        quadrature = planner.solver.direction(spec.observer, sources, az, alt, spec.atmosphere, SolverSettings(384, 48))
        aggregation = planner.solver.direction(spec.observer, refined, az, alt, spec.atmosphere, SolverSettings(192, 24))
        record = {'azimuth_deg': az, 'altitude_deg': alt, 'baseline': baseline,
                  'refined_quadrature': quadrature, 'refined_aggregation': aggregation,
                  'quadrature_relative_difference': abs(quadrature / baseline - 1),
                  'aggregation_relative_difference': abs(aggregation / baseline - 1)}
        convergence.append(record)
        print('convergence', record, flush=True)
    passed = all(max(r['quadrature_relative_difference'], r['aggregation_relative_difference']) <= tolerance for r in convergence)
    write_json(args.output / 'numerical-validation.json', {'declared_relative_tolerance': tolerance,
        'selected_directions': len(convergence), 'passed': passed, 'records': convergence,
        'scope': 'Only these four rays for this source region; not a full-sky or physical accuracy bound.'})
    write_json(args.output / 'benchmark.json', {'cases': timings,
        'peak_process_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'platform_note': 'Linux ru_maxrss in KiB; process-wide peak across the complete script.',
        'new_NASA_downloads': 0, 'source_files': 'Pinned pre-existing local HDF5, verified against package manifests.'})
    if not passed:
        raise AssertionError('Declared numerical tolerance was not met; inspect numerical-validation.json')


if __name__ == '__main__':
    main()
