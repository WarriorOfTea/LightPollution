"""Research-only, scalar spherical transport resolved by scattering order.

Backward Monte Carlo with extinction delta tracking, implicit absorption and
reciprocal Rayleigh/HG direction sampling. The boundary is a FINITE Lambertian
cap: scores are bounded by its radiance, unlike point-light next-event estimates
whose variance can diverge near a source. No ground reflection, DSM, Moon or
spectral integration is silently added to the production single-order solver.
"""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from ..physics import EARTH_RADIUS_M, Atmosphere, Observer, Sources, unit_position


@dataclass(frozen=True)
class LambertianCap:
    latitude_deg: float
    longitude_deg: float
    radius_m: float
    radiance: float = 1.0

    def __post_init__(self):
        values = list(asdict(self).values())
        if (not np.all(np.isfinite(values)) or not -90 <= self.latitude_deg <= 90
                or not -180 <= self.longitude_deg <= 180
                or not 0 < self.radius_m <= np.pi * EARTH_RADIUS_M
                or self.radiance < 0):
            raise ValueError('A cap requires finite coordinates, positive arc radius and nonnegative radiance.')

    @property
    def centre(self):
        return unit_position(self.latitude_deg, self.longitude_deg)

    def contains(self, points):
        return np.asarray(points) @ self.centre / EARTH_RADIUS_M >= np.cos(self.radius_m / EARTH_RADIUS_M)

    def quadrature_sources(self, radial_order=24, azimuth_count=96):
        """Surface quadrature for the EXISTING independent first-order solver.

        P = pi * L * dA makes its normalized Lambertian point emission equal
        to the radiance of the surface element, including its projected area.
        """
        if radial_order < 2 or azimuth_count < 4:
            raise ValueError('Surface quadrature needs >=2 radial nodes and >=4 azimuths.')
        x, weights = np.polynomial.legendre.leggauss(radial_order)
        # Accurate cap width for small angles; avoid subtractive loss in 1-cos.
        width = 2 * np.sin(self.radius_m / (2 * EARTH_RADIUS_M))**2
        mu = 1 - width * (x + 1) / 2
        phi = 2 * np.pi * (np.arange(azimuth_count) + .5) / azimuth_count
        centre = self.centre
        axis = np.eye(3)[np.argmin(abs(centre))]
        e1 = np.cross(centre, axis)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(centre, e1)
        ring = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
        points = mu[:, None, None] * centre + np.sqrt(np.maximum(0, 1-mu**2))[:, None, None] * ring
        area = EARTH_RADIUS_M**2 * weights * width / 2 * (2*np.pi/azimuth_count)
        return Sources(EARTH_RADIUS_M * points.reshape(-1, 3),
            np.repeat(np.pi * self.radiance * area, azimuth_count))


@dataclass(frozen=True)
class OrderSettings:
    samples: int = 131072
    max_order: int = 3
    seed: int = 3011
    batch_size: int = 8192
    max_null_events: int = 100000

    def __post_init__(self):
        for name in ('samples', 'max_order', 'seed', 'batch_size', 'max_null_events'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'{name} must be an integer in its permitted range.')
        if self.samples < 2 or self.max_order > 32:
            raise ValueError('At least two paths and at most 32 scattering orders are supported.')


def boundary_distance(points, directions, top_m):
    """Distance to first opaque ground / vacuum boundary; vectors must be unit.

    Stable near-ground root avoids cancellation when altitude is very small.
    Return ground=True only when Earth is reached before atmospheric exit.
    """
    radius2 = np.einsum('ij,ij->i', points, points)
    b = np.einsum('ij,ij->i', points, directions)
    c = np.maximum(0., radius2 - EARTH_RADIUS_M**2)
    disc = b*b-c
    ground = (b < 0) & (disc >= 0)
    root = np.sqrt(np.maximum(disc, 0))
    ground_distance = np.full(len(points), np.inf)
    ground_distance[ground] = c[ground] / (-b[ground]+root[ground])
    top = -b+np.sqrt(np.maximum(0, b*b+(EARTH_RADIUS_M+top_m)**2-radius2))
    return np.minimum(ground_distance, top), ground & (ground_distance <= top)


def _collision(points, directions, atmosphere, rng, max_null_events):
    """Exact free-path distribution for the prescribed exponential medium."""
    distance, ground = boundary_distance(points, directions, atmosphere.top_m)
    majorant = atmosphere.rayleigh_sea_level + atmosphere.aerosol_extinction_sea_level
    travelled = np.zeros(len(points))
    collision = np.zeros(len(points), dtype=bool)
    pending = np.arange(len(points))
    if majorant == 0:
        return points + distance[:, None]*directions, collision, ground
    for _ in range(max_null_events):
        if not len(pending):
            break
        travelled[pending] += rng.exponential(1/majorant, len(pending))
        pending = pending[travelled[pending] < distance[pending]]
        if not len(pending):
            break
        candidate = points[pending] + travelled[pending, None]*directions[pending]
        height = np.maximum(0, np.linalg.norm(candidate, axis=1)-EARTH_RADIUS_M)
        extinction = atmosphere.coefficients(height)[2]
        if np.any(extinction > majorant*(1+1e-12)):
            raise RuntimeError('The extinction majorant was violated.')
        accepted = rng.random(len(pending)) < extinction/majorant
        collision[pending[accepted]] = True
        pending = pending[~accepted]
    else:
        raise RuntimeError('Null-collision limit reached; no silently truncated estimate is returned.')
    length = np.where(collision, travelled, distance)
    return points + length[:, None]*directions, collision, ground


def sample_phase(directions, rayleigh_fraction, g, rng):
    """Sample a normalized reciprocal phase; positive cosine is forward."""
    count = len(directions)
    molecular = rng.random(count) < np.asarray(rayleigh_fraction)
    cosine = np.empty(count)
    pending = np.flatnonzero(molecular)
    while len(pending):
        c = rng.uniform(-1, 1, len(pending))
        accepted = rng.random(len(pending)) < (1+c*c)/2
        cosine[pending[accepted]] = c[accepted]
        pending = pending[~accepted]
    u = rng.random(np.sum(~molecular))
    cosine[~molecular] = (2*u-1 if abs(g) < 1e-8 else
        (1+g*g-((1-g*g)/(1-g+2*g*u))**2)/(2*g))
    cosine = np.clip(cosine, -1, 1)
    phi = 2*np.pi*rng.random(count)
    axis = np.eye(3)[np.argmin(abs(directions), axis=1)]
    e1 = np.cross(directions, axis)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(directions, e1)
    perpendicular = np.cos(phi)[:, None]*e1 + np.sin(phi)[:, None]*e2
    return cosine[:, None]*directions + np.sqrt(np.maximum(0, 1-cosine*cosine))[:, None]*perpendicular


def integrate_orders(observer: Observer, cap: LambertianCap, azimuth_deg: float,
                     altitude_deg: float, atmosphere: Atmosphere = Atmosphere(),
                     settings: OrderSettings = OrderSettings()) -> dict:
    """Estimate exactly 1..K scatterings; cumulative sums include <=K.

    A path scores ONLY upon reaching the luminous ground before the next real
    extinction event. Albedo weights account for absorption. Null events do not
    count as scattering. Ground is absorbing after emission; outer space dark.
    Independent streams per batch/order preserve paired prefixes for K=1,2,3.
    """
    if observer.height_m >= atmosphere.top_m:
        raise ValueError('Observer must be inside the atmosphere.')
    direction = observer.direction(azimuth_deg, altitude_deg)
    if not np.all(np.isfinite(direction)):
        raise ValueError('Direction must be finite.')
    start_time = perf_counter()
    k = settings.max_order
    sums, seconds = np.zeros(k), np.zeros((k, k))
    hits = np.zeros(k, dtype=np.int64)
    outcomes = dict(space=0, dark_ground=0, luminous_ground=0, order_cutoff=0, absorbed=0)
    for batch, offset in enumerate(range(0, settings.samples, settings.batch_size)):
        size = min(settings.batch_size, settings.samples-offset)
        points = np.broadcast_to(observer.position, (size, 3)).copy()
        directions = np.broadcast_to(direction, (size, 3)).copy()
        weights = np.ones(size)
        original = np.arange(size)
        scores = np.zeros((size, k))
        for order in range(k+1):
            if not len(points):
                break
            rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 0]))
            end, collision, ground = _collision(points, directions, atmosphere, rng, settings.max_null_events)
            reached = ~collision & ground
            luminous = reached & cap.contains(end)
            if order:
                scores[original[luminous], order-1] = weights[luminous]*cap.radiance
                hits[order-1] += np.sum(luminous)
            outcomes['luminous_ground'] += int(np.sum(luminous))
            outcomes['dark_ground'] += int(np.sum(reached & ~luminous))
            outcomes['space'] += int(np.sum(~collision & ~ground))
            if order == k:
                outcomes['order_cutoff'] += int(np.sum(collision))
                break
            points, directions = end[collision], directions[collision]
            original, weights = original[collision], weights[collision]
            if not len(points):
                break
            br, ba, extinction = atmosphere.coefficients(np.maximum(0, np.linalg.norm(points, axis=1)-EARTH_RADIUS_M))
            scattering = br+ba
            positive = scattering > 0
            outcomes['absorbed'] += int(np.sum(~positive))
            weights *= scattering/extinction
            points, directions, original, weights = (v[positive] for v in (points, directions, original, weights))
            if not len(points):
                break
            phase_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, batch, order, 1]))
            directions = sample_phase(directions, br[positive]/scattering[positive], atmosphere.aerosol_g, phase_rng)
        sums += scores.sum(axis=0)
        seconds += scores.T @ scores
    n = settings.samples
    means = sums/n
    covariance = (seconds - np.outer(sums, sums)/n)/(n*(n-1))
    cumulative_matrix = np.tril(np.ones((k, k)))
    cumulative = cumulative_matrix @ means
    cumulative_covariance = cumulative_matrix @ covariance @ cumulative_matrix.T
    return {'solver': 'research-spherical-cap-backward-mc-1', 'settings': asdict(settings),
        'observer': asdict(observer), 'cap': asdict(cap), 'atmosphere': asdict(atmosphere),
        'azimuth_deg': azimuth_deg, 'altitude_deg': altitude_deg,
        'exact_order_radiance': means.tolist(),
        'exact_order_standard_error': np.sqrt(np.maximum(0, np.diag(covariance))).tolist(),
        'mean_covariance': covariance.tolist(), 'cumulative_radiance': cumulative.tolist(),
        'cumulative_standard_error': np.sqrt(np.maximum(0, np.diag(cumulative_covariance))).tolist(),
        'luminous_paths_by_order': hits.tolist(), 'path_outcomes': outcomes,
        'runtime_seconds': perf_counter()-start_time,
        'units': 'same radiance units as cap, monochromatic at atmosphere.reference_wavelength_nm',
        'production_solver_changed': False,
        'limitations': ['Finite uniform Lambertian cap; no NASA reconstruction or point-source lookup.',
            'Absorbing spherical ground, no DSM, reflection, Moon, natural light or polarization.',
            'Monte Carlo standard errors are not errors against real observations.',
            'Normal confidence intervals require sufficient luminous hits; rare-event zeros are not certified zeros.']}
