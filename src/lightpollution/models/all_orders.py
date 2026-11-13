"""Bounded backward transport estimator without a scattering-order cutoff.

The same spherical atmosphere, ground cap, free paths and reciprocal phases
as scattering_orders.py. All paths are followed to a boundary (or physical
absorption in analog mode). A safety budget causes FAILURE, never a published
truncated answer. Recording only five explicit orders does not stop transport.
"""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from ..physics import EARTH_RADIUS_M, Atmosphere, Observer
from .scattering_orders import LambertianCap, _collision, sample_phase


@dataclass(frozen=True)
class AllOrderSettings:
    samples: int = 131072
    seed: int = 3011
    batch_size: int = 8192
    recorded_orders: int = 5
    absorption: str = 'implicit'
    safety_scattering_events: int = 100000
    max_null_events: int = 100000

    def __post_init__(self):
        for name in ('samples', 'seed', 'batch_size', 'recorded_orders',
                     'safety_scattering_events', 'max_null_events'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be an integer in its permitted range.')
        if self.samples < 2 or self.recorded_orders > 64:
            raise ValueError('Need >=2 paths and at most64 explicitly recorded orders.')
        if self.absorption not in ('implicit', 'analog'):
            raise ValueError('Absorption must be implicit or analog.')


def relative_tail(result, up_to_order):
    """Paired all/partial increment and missing fraction; no independent-error sum."""
    means = np.asarray(result['category_mean_radiance'])
    covariance = np.asarray(result['mean_covariance'])
    k = result['settings']['recorded_orders']
    if not 1 <= up_to_order <= k:
        raise ValueError('The partial sum must be among the explicitly recorded orders.')
    a = np.ones(k+1)
    b = (np.arange(k+1) < up_to_order).astype(float)
    total, partial = float(a @ means), float(b @ means)
    if total <= 0 or partial <= 0:
        return {'up_to_order': up_to_order, 'increment_relative_to_partial': None,
            'missing_fraction_of_full': None, 'normal_intervals_eligible': False}
    grad_ratio = a/partial - b*total/partial**2
    grad_missing = -b/total + a*partial/total**2
    se_ratio = float(np.sqrt(max(0, grad_ratio @ covariance @ grad_ratio)))
    se_missing = float(np.sqrt(max(0, grad_missing @ covariance @ grad_missing)))
    hits = result['luminous_paths_by_category']
    eligible = sum(hits[:up_to_order]) >= 100 and sum(hits[up_to_order:]) >= 100
    ratio, missing = total/partial-1, 1-partial/total
    return {'up_to_order': up_to_order, 'increment_relative_to_partial': ratio,
        'increment_standard_error': se_ratio,
        'missing_fraction_of_full': missing, 'missing_fraction_standard_error': se_missing,
        'normal_intervals_eligible': bool(eligible),
        'increment_normal_95_interval': [ratio-1.96*se_ratio, ratio+1.96*se_ratio] if eligible else None,
        'missing_fraction_normal_95_interval': [missing-1.96*se_missing, missing+1.96*se_missing] if eligible else None,
        'tail_luminous_paths': int(sum(hits[up_to_order:])),
        'interval_scope': 'Marginal numerical sampling error; neither atmospheric uncertainty nor field accuracy.'}


def integrate_all_orders(observer: Observer, cap: LambertianCap, azimuth_deg: float,
                         altitude_deg: float, atmosphere: Atmosphere = Atmosphere(),
                         settings: AllOrderSettings = AllOrderSettings()) -> dict:
    """Estimate the infinite Neumann sum by finite random-length path histories.

    Implicit absorption retains the OLD random prefixes exactly. Analog mode
    randomly absorbs photons and leaves surviving weights=1, providing an
    independent absorption check. Both estimators have scores in [0,Lcap].
    No small-weight deletion, Russian roulette or finite-depth bias is used.
    """
    if observer.height_m >= atmosphere.top_m:
        raise ValueError('Observer must be inside the atmosphere.')
    direction = observer.direction(azimuth_deg, altitude_deg)
    if not np.all(np.isfinite(direction)):
        raise ValueError('Direction must be finite.')
    started = perf_counter()
    k = settings.recorded_orders
    sums, seconds = np.zeros(k+1), np.zeros((k+1, k+1))
    hits = np.zeros(k+1, dtype=np.int64)
    histogram = {}
    outcomes = dict(space=0, dark_ground=0, luminous_ground=0, absorbed=0)
    max_order_seen = 0
    scattering_vertices = 0
    for batch, offset in enumerate(range(0, settings.samples, settings.batch_size)):
        size = min(settings.batch_size, settings.samples-offset)
        points = np.broadcast_to(observer.position, (size, 3)).copy()
        directions = np.broadcast_to(direction, (size, 3)).copy()
        weights = np.ones(size)
        original = np.arange(size)
        scores = np.zeros((size, k+1))
        order = 0
        while len(points):
            rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 0]))
            end, collision, ground = _collision(points, directions, atmosphere, rng, settings.max_null_events)
            reached = ~collision & ground
            luminous = reached & cap.contains(end)
            if order and np.any(luminous):
                values = weights[luminous]*cap.radiance
                category = min(order-1, k)
                scores[original[luminous], category] = values
                hits[category] += int(np.sum(luminous))
                entry = histogram.setdefault(order, {'paths': 0, 'sum': 0., 'sum_squares': 0.})
                entry['paths'] += len(values)
                entry['sum'] += float(values.sum())
                entry['sum_squares'] += float(values @ values)
            outcomes['luminous_ground'] += int(np.sum(luminous))
            outcomes['dark_ground'] += int(np.sum(reached & ~luminous))
            outcomes['space'] += int(np.sum(~collision & ~ground))
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
            outcomes['absorbed'] += int(np.sum(~alive))
            points, directions, original, weights = (value[alive] for value in (points, directions, original, weights))
            if not len(points):
                break
            if order >= settings.safety_scattering_events:
                raise RuntimeError('Safety event budget reached; all-order result is NOT returned or truncated.')
            phase_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 1]))
            directions = sample_phase(directions, br[alive]/scattering[alive], atmosphere.aerosol_g, phase_rng)
            order += 1
            scattering_vertices += len(points)
            max_order_seen = max(max_order_seen, order)
        sums += scores.sum(axis=0)
        seconds += scores.T @ scores
    n = settings.samples
    means = sums/n
    covariance = (seconds - np.outer(sums, sums)/n)/(n*(n-1))
    cumulative_matrix = np.tril(np.ones((k+1, k+1)))
    cumulative = cumulative_matrix @ means
    cumulative_covariance = cumulative_matrix @ covariance @ cumulative_matrix.T
    standard_errors = np.sqrt(np.maximum(0, np.diag(cumulative_covariance)))
    normal_total = [float(cumulative[-1]-1.96*standard_errors[-1]),
                    float(cumulative[-1]+1.96*standard_errors[-1])] if sum(hits) >= 100 else None
    # A deliberately conservative bound remains valid even for zero observed
    # hits. It must not be presented as a useful relative-error guarantee.
    hoeffding = cap.radiance*np.sqrt(np.log(40)/(2*n))
    orders = {str(order): {'luminous_paths': entry['paths'], 'mean_radiance': entry['sum']/n,
        'standard_error': float(np.sqrt(max(0, (entry['sum_squares']-entry['sum']**2/n)/(n*(n-1))))) }
        for order, entry in sorted(histogram.items())}
    return {'solver': 'research-spherical-cap-all-orders-mc-1', 'settings': asdict(settings),
        'observer': asdict(observer), 'cap': asdict(cap), 'atmosphere': asdict(atmosphere),
        'azimuth_deg': azimuth_deg, 'altitude_deg': altitude_deg,
        'category_labels': [f'exact_{j}' for j in range(1, k+1)]+[f'above_{k}'],
        'category_mean_radiance': means.tolist(), 'mean_covariance': covariance.tolist(),
        'luminous_paths_by_category': hits.tolist(),
        'cumulative_labels': [f'up_to_{j}' for j in range(1, k+1)]+['all_orders'],
        'cumulative_radiance': cumulative.tolist(), 'cumulative_standard_error': standard_errors.tolist(),
        'all_order_radiance': float(cumulative[-1]), 'all_order_standard_error': float(standard_errors[-1]),
        'all_order_normal_95_interval': normal_total,
        'all_order_hoeffding_marginal_95_interval': [max(0., float(cumulative[-1]-hoeffding)),
            min(cap.radiance, float(cumulative[-1]+hoeffding))],
        'luminous_order_histogram': orders, 'path_outcomes': outcomes,
        'max_scattering_order_seen': max_order_seen,
        'max_luminous_order_seen': max(histogram, default=0),
        'mean_scattering_vertices_per_path': scattering_vertices/n,
        'runtime_seconds': perf_counter()-started,
        'finite_scattering_order_truncation': False, 'safety_limit_hit': False,
        'bounded_score': [0., cap.radiance], 'production_solver_changed': False,
        'limitations': ['Finite Lambertian cap, scalar monochromatic atmospheric model, absorbing ground and dark outer boundary.',
            'All orders refers to the specified RTE, not exact atmospheric microphysics or every physical process.',
            'Finite samples still give statistical error; observed maximum order is not a physical cutoff.',
            'Implicit absorbed outcome counts only zero-scattering terminations; other absorption is in weights.',
            'Normal intervals are approximate; rare-event zero is not evidence of a zero mean.']}
