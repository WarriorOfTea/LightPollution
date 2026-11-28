"""Recover scattering orders and a flux-conserving angular response on the transects."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from run_halo_ground_validation import (
    BERLIN,
    ENHANCED,
    HELD,
    INPUTS,
    REFERENCE,
    LowAngleBoundary,
    collect_tasks,
    period_boundary,
    replay,
    setup_boundary,
)


class RedistributionBoundary(LowAngleBoundary):
    labels = (*LowAngleBoundary.labels, 'city_middle_angle_correction')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upper_bounds = np.r_[self.upper_bounds, self.upper_bounds[HELD] / 2]

    def sample(self, points, outgoing):
        values, support = super().sample(points, outgoing)
        mu = np.einsum('ij,ij->i', points, outgoing) / np.linalg.norm(points, axis=1)
        slope = values[:, ENHANCED] - values[:, HELD]
        return np.column_stack((values, slope * 4 * np.clip(mu, 0, 0.5))), support


def project(row, replayed, archived):
    means, covariance, orders = replayed
    weights = np.array(row['spectral_weights'])
    terms = weights @ means
    predicted = terms[0] + terms[ENHANCED] + row['natural']
    np.testing.assert_allclose(
        predicted, archived['prediction']['off_h2']['linear'], rtol=3e-11
    )
    contrast = np.zeros(len(terms))
    contrast[[HELD, ENHANCED]] = [-1, 1]
    cov = np.tensordot(weights**2, covariance, axes=1)
    return {
        **{key: row[key] for key in ('key', 'night', 'site', 'band')},
        'fitted_scene': archived['fitted_scene'],
        'single_and_multiple': np.tensordot(weights, orders, axes=1).tolist(),
        'tail_standard_error': float(np.sqrt(max(0, contrast @ cov @ contrast))),
    }


def observation_key(row):
    return tuple(row[k] for k in ('key', 'night', 'site', 'band'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('maps', 'boundaries', 'previous_ground', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    geometry = read_json(args.boundaries)
    previous = read_json(args.previous_ground)
    if file_hash(args.boundaries) != previous['provenance']['boundaries.json']:
        raise ValueError(
            'Expected the same municipality boundaries as in the baseline.'
        )
    archive = {
        observation_key(row): row for row in previous['rows'] if row['fitted_scene']
    }
    berlin, reference, inputs = (
        read_json(path) for path in (BERLIN, REFERENCE, INPUTS)
    )
    groups, tasks = collect_tasks(berlin, reference, inputs)
    count, paths, rows = 0, 0, []
    for period, readings in groups.items():
        selected = [row for row in readings if observation_key(row) in archive]
        if not selected:
            continue
        city = selected[0]['city']
        boundary = period_boundary(
            period,
            city,
            args.maps,
            geometry,
            berlin,
            reference,
            inputs,
            RedistributionBoundary,
        )
        keys = list(dict.fromkeys(row['key'] for row in selected))
        with ProcessPoolExecutor(
            args.workers, initializer=setup_boundary, initargs=(boundary,)
        ) as pool:
            futures = {
                k: pool.submit(replay, tasks[k], include_orders=True) for k in keys
            }
            for row in selected:
                key = row['key']
                rows.append(
                    project(row, futures[key].result(), archive[observation_key(row)])
                )
        for key in keys:
            task = tasks[key]
            count += len(task[7])
            paths += task[6] * len(task[7])
        print(period, len(selected), flush=True)
    if len(rows) != len(archive) or {observation_key(row) for row in rows} != set(
        archive
    ):
        raise ValueError('Missing ground replay rows.')
    write_json(
        args.output,
        {
            'rows': rows,
            'spectral_integrations': count,
            'paths': paths,
            'previous_ground_sha256': file_hash(args.previous_ground),
            'terms': list(RedistributionBoundary.labels),
        },
    )


if __name__ == '__main__':
    main()
