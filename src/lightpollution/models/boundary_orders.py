"""Paired all-order research transport for bounded spatial/angular boundaries.

Uses the same free-path/phase sampler and RNG streams as all_orders.py.
Every boundary variant is evaluated on exactly the same complete histories.
"""

from dataclasses import asdict
from time import perf_counter
from typing import Protocol

import numpy as np

from ..physics import EARTH_RADIUS_M, Atmosphere, Observer
from .all_orders import AllOrderSettings
from .scattering_orders import _collision, sample_phase


class BoundedBoundary(Protocol):
    labels: tuple[str, ...]
    upper_bounds: np.ndarray

    def sample(self, points, outgoing) -> tuple[np.ndarray, np.ndarray]: ...


def paired_ratio(result, numerator, denominator):
    labels = result['labels']
    i, j = labels.index(numerator), labels.index(denominator)
    means, cov = np.asarray(result['all_order_radiance']), np.asarray(result['mean_covariance'])
    if means[j] <= 0:
        return {'ratio': None, 'normal_95_interval': None}
    r = means[i]/means[j]
    grad = np.zeros(len(means))
    grad[i] += 1/means[j]
    grad[j] -= means[i]/means[j]**2
    se = float(np.sqrt(max(0., grad @ cov @ grad)))
    eligible = min(result['positive_paths'][i], result['positive_paths'][j]) >= 100 and min(result['effective_score_samples'][i], result['effective_score_samples'][j]) >= 100
    return {'ratio': float(r), 'relative_change': float(r-1), 'ratio_standard_error': se,
            'normal_95_interval': [float(r-1.96*se), float(r+1.96*se)] if eligible else None,
            'delta_mag_artificial_only': float(-2.5*np.log10(r)) if r > 0 else None,
            'normal_intervals_eligible': bool(eligible),
            'scope': 'Paired marginal MC uncertainty only; no atmospheric/source uncertainty or field accuracy.'}


def integrate_boundary_variants(observer: Observer, boundary: BoundedBoundary,
                               azimuth_deg: float, altitude_deg: float,
                               atmosphere: Atmosphere = Atmosphere(),
                               settings: AllOrderSettings = AllOrderSettings()):
    if observer.height_m >= atmosphere.top_m:
        raise ValueError('Observer must be inside the atmosphere.')
    direction = observer.direction(azimuth_deg, altitude_deg)
    bounds = np.asarray(boundary.upper_bounds, dtype=float)
    v, k = len(boundary.labels), settings.recorded_orders
    if not v or len(set(boundary.labels)) != v or bounds.shape != (v,) or np.any(~np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError('Unique boundary labels and finite nonnegative bounds are required.')
    started = perf_counter()
    sums, seconds = np.zeros(v), np.zeros((v, v))
    order_sums = np.zeros((k+1, v))
    order_seconds = np.zeros((v, k+1, k+1))
    hits, positive = np.zeros((k+1, v), dtype=np.int64), np.zeros(v, dtype=np.int64)
    angular_edges = np.array([0., 20., 40., 60., 75., 85., 90.])
    angular_sums = np.zeros((len(angular_edges)-1, v))
    outcomes = dict(space=0, ground=0, absorbed=0)
    support_counts = np.zeros(3, dtype=np.int64)
    max_order, vertices = 0, 0
    for batch, offset in enumerate(range(0, settings.samples, settings.batch_size)):
        size = min(settings.batch_size, settings.samples-offset)
        points = np.broadcast_to(observer.position, (size, 3)).copy()
        directions = np.broadcast_to(direction, (size, 3)).copy()
        weights, original = np.ones(size), np.arange(size)
        scores = np.zeros((size, k+1, v))
        order = 0
        while len(points):
            rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 0]))
            end, collision, ground = _collision(points, directions, atmosphere, rng, settings.max_null_events)
            reached = ~collision & ground
            outcomes['ground'] += int(reached.sum())
            outcomes['space'] += int((~collision & ~ground).sum())
            if np.any(reached):
                p, outgoing = end[reached], -directions[reached]
                values, code = boundary.sample(p, outgoing)
                if values.shape != (len(p), v) or np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > bounds[None, :]*(1+1e-12)):
                    raise ValueError('Boundary score is invalid or exceeds its declared bound.')
                code = np.asarray(code)
                if code.shape != (len(p),) or not np.all(np.isin(code, [0, 1, 2])):
                    raise ValueError('Invalid boundary support codes.')
                support_counts += np.bincount(code.astype(int), minlength=3)
                if order:
                    values = values*weights[reached, None]
                    category = min(order-1, k)
                    scores[original[reached], category, :] = values
                    hits[category] += np.count_nonzero(values, axis=0)
                    mu = np.clip(np.einsum('ij,ij->i', outgoing, p/np.linalg.norm(p, axis=1)[:, None]), 0, 1)
                    bins = np.clip(np.searchsorted(angular_edges, np.rad2deg(np.arccos(mu)), side='right')-1, 0, len(angular_edges)-2)
                    np.add.at(angular_sums, bins, values)
            points, directions = end[collision], directions[collision]
            original, weights = original[collision], weights[collision]
            if not len(points):
                break
            br, ba, extinction = atmosphere.coefficients(np.maximum(0, np.linalg.norm(points, axis=1)-EARTH_RADIUS_M))
            scattering = br+ba
            survival = scattering/extinction
            if settings.absorption == 'implicit':
                alive = scattering > 0
                weights *= survival
            else:
                absorption_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 2]))
                alive = absorption_rng.random(len(points)) < survival
            outcomes['absorbed'] += int((~alive).sum())
            points, directions, original, weights = (value[alive] for value in (points, directions, original, weights))
            if not len(points):
                break
            if order >= settings.safety_scattering_events:
                raise RuntimeError('Safety event budget reached; all-order result is NOT returned or truncated.')
            phase_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 1]))
            directions = sample_phase(directions, br[alive]/scattering[alive], atmosphere.aerosol_g, phase_rng)
            order += 1
            vertices += len(points)
            max_order = max(max_order, order)
        total = scores.sum(axis=1)
        sums += total.sum(axis=0)
        seconds += total.T @ total
        order_sums += scores.sum(axis=0)
        order_seconds += np.einsum('nkv,nlv->vkl', scores, scores)
        positive += np.count_nonzero(total, axis=0)
    n = settings.samples
    means = sums/n
    covariance = (seconds-np.outer(sums, sums)/n)/(n*(n-1))
    order_cov = (order_seconds-np.einsum('kv,lv->vkl', order_sums, order_sums)/n)/(n*(n-1))
    se = np.sqrt(np.maximum(0, np.diag(covariance)))
    effective = np.divide(sums*sums, np.diag(seconds), out=np.zeros(v), where=np.diag(seconds) > 0)
    hoeffding = bounds*np.sqrt(np.log(40)/(2*n))
    return {'solver': 'research-spherical-boundary-all-orders-mc-1',
        'settings': asdict(settings), 'observer': asdict(observer), 'atmosphere': asdict(atmosphere),
        'azimuth_deg': azimuth_deg, 'altitude_deg': altitude_deg, 'labels': list(boundary.labels),
        'all_order_radiance': means.tolist(), 'all_order_standard_error': se.tolist(),
        'mean_covariance': covariance.tolist(), 'positive_paths': positive.tolist(),
        'effective_score_samples': effective.tolist(),
        'normal_95_intervals': [[float(x-1.96*s), float(x+1.96*s)] if hit >= 100 and ess >= 100 else None for x, s, hit, ess in zip(means, se, positive, effective)],
        'hoeffding_marginal_95_intervals': np.stack((np.maximum(0, means-hoeffding), np.minimum(bounds, means+hoeffding)), axis=1).tolist(),
        'bounded_scores': bounds.tolist(),
        'category_labels': [f'exact_{j}' for j in range(1, k+1)]+[f'above_{k}'],
        'category_mean_radiance': (order_sums/n).tolist(),
        'category_mean_covariance_per_variant': order_cov.tolist(), 'positive_paths_by_category': hits.tolist(),
        'emission_zenith_bin_edges_deg': angular_edges.tolist(), 'emission_bin_radiance': (angular_sums/n).tolist(),
        'path_outcomes': outcomes, 'ground_support_counts': dict(zip(('outside_region', 'missing_common_support', 'known'), map(int, support_counts))),
        'max_scattering_order_seen': max_order, 'mean_scattering_vertices_per_path': vertices/n,
        'finite_scattering_order_truncation': False, 'safety_limit_hit': False,
        'runtime_seconds': perf_counter()-started, 'production_solver_changed': False,
        'limitations': ['Conditional on the common observed support; missing/outside light is not inferred.',
            'Scalar monochromatic proxy, absorbing spherical ground; no DSM, ground re-reflection, clouds, Moon or natural sky.',
            'Angular continuation and fixed spectrum are scenarios. MC intervals do not measure physical accuracy.',
            'Bounds and Hoeffding intervals are marginal, not simultaneous over variants/directions.',
            'Effective score samples is a weight-concentration diagnostic, not a proof of normal coverage.']}
