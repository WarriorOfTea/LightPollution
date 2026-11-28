"""Research TOA transfer of a grey DNB proxy, with separately scored scattering orders."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..physics import EARTH_RADIUS_M, Atmosphere
from .all_orders import AllOrderSettings
from .scattering_orders import _collision, sample_phase


LOW_ANGLE_COSINE = float(np.sin(np.deg2rad(10.0)))


def _low_angle_integral(mu):
    scale = LOW_ANGLE_COSINE
    return scale**2 * (1 - (1 + mu / scale) * np.exp(-mu / scale))


def emission_radiance(mu):
    """Lambert and low-angle radiance per unit upward flux density, in sr^-1."""
    mu = np.clip(np.asarray(mu), 0, 1)
    normalizer = 2 * np.pi * _low_angle_integral(1.0)
    return np.stack(
        (
            np.full_like(mu, 1 / np.pi, dtype=float),
            np.exp(-mu / LOW_ANGLE_COSINE) / normalizer,
        ),
        axis=-1,
    )


def flux_below(elevation_deg):
    """Fractions of the two templates' upward power emitted below the given elevation."""
    if not 0 <= elevation_deg <= 90:
        raise ValueError('Elevation must lie in [0, 90].')
    mu = np.sin(np.deg2rad(elevation_deg))
    return np.array([mu**2, _low_angle_integral(mu) / _low_angle_integral(1.0)])


@dataclass(frozen=True)
class HaloKernel:
    direct: np.ndarray
    single: np.ndarray
    multiple: np.ndarray
    hits: tuple[int, int]
    outside_window: int


def build_halo_kernel(
    zenith_deg,
    azimuth_deg,
    cell_east_m,
    cell_north_m,
    radius_m,
    atmosphere: Atmosphere,
    settings: AllOrderSettings,
    *,
    angular_radiance=emission_radiance,
) -> HaloKernel:
    """Backtrace an observed pixel. Approximates translation of the kernel on a local grid.

    Array rows increase southward and columns eastward. A coefficient weights
    unit upward flux density of a cell, rather than a point-source power.
    """
    if (
        not np.all(
            np.isfinite([zenith_deg, azimuth_deg, cell_east_m, cell_north_m, radius_m])
        )
        or not 0 <= zenith_deg < 75
        or min(cell_east_m, cell_north_m, radius_m) <= 0
        or settings.absorption != 'implicit'
    ):
        raise ValueError(
            'Need finite local grid, angle below 75 degrees and implicit absorption.'
        )
    east_half = int(np.ceil(radius_m / cell_east_m))
    north_half = int(np.ceil(radius_m / cell_north_m))
    shape = (2 * north_half + 1, 2 * east_half + 1)
    ground = np.array([EARTH_RADIUS_M, 0.0, 0.0])
    zenith, azimuth = np.deg2rad([zenith_deg, azimuth_deg])
    mu = np.cos(zenith)
    outward = np.array(
        [mu, np.sin(zenith) * np.sin(azimuth), np.sin(zenith) * np.cos(azimuth)]
    )
    top_r = EARTH_RADIUS_M + atmosphere.top_m
    projection = EARTH_RADIUS_M * mu
    distance = np.sqrt(projection**2 + top_r**2 - EARTH_RADIUS_M**2) - projection
    # Start just inside the atmospheric boundary. The path beyond it is vacuum.
    entry = ground + (distance - 1e-3) * outward
    direct = angular_radiance(mu) * np.exp(
        -atmosphere.optical_depth(ground, entry, order=48)
    )
    templates = len(direct)
    scores = np.zeros((2, templates, np.prod(shape)))
    hit_counts = [0, 0]
    outside = 0
    for batch, offset in enumerate(range(0, settings.samples, settings.batch_size)):
        count = min(settings.batch_size, settings.samples - offset)
        points = np.broadcast_to(entry, (count, 3))
        directions = np.broadcast_to(-outward, (count, 3))
        weights = np.ones(count)
        order = 0
        while len(points):
            rng = np.random.default_rng(
                np.random.SeedSequence([settings.seed, batch, order, 0])
            )
            end, collision, touches_ground = _collision(
                points, directions, atmosphere, rng, settings.max_null_events
            )
            reached = ~collision & touches_ground
            if order and np.any(reached):
                points_on_ground = end[reached]
                horizontal = np.linalg.norm(points_on_ground[:, 1:], axis=-1)
                arc = EARTH_RADIUS_M * np.arctan2(horizontal, points_on_ground[:, 0])
                scale = np.divide(
                    arc, horizontal, out=np.ones_like(arc), where=horizontal > 0
                )
                # Reverse source displacement because this is a convolution kernel.
                x = east_half - np.rint(
                    points_on_ground[:, 1] * scale / cell_east_m
                ).astype(int)
                y = north_half + np.rint(
                    points_on_ground[:, 2] * scale / cell_north_m
                ).astype(int)
                inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
                outgoing_mu = (
                    np.einsum('ij,ij->i', -directions[reached], points_on_ground)
                    / EARTH_RADIUS_M
                )
                order_bin = int(order > 1)
                hit_counts[order_bin] += int(reached.sum())
                outside += int((~inside).sum())
                index = y[inside] * shape[1] + x[inside]
                value = (
                    angular_radiance(outgoing_mu[inside])
                    * weights[reached][inside, None]
                )
                for template in range(templates):
                    scores[order_bin, template] += np.bincount(
                        index, value[:, template], minlength=np.prod(shape)
                    )
            points, directions, weights = (
                end[collision],
                directions[collision],
                weights[collision],
            )
            if not len(points):
                break
            rayleigh, aerosol, extinction = atmosphere.coefficients_at(points)
            scattering = rayleigh + aerosol
            alive = scattering > 0
            if not np.any(alive):
                break
            if order >= settings.safety_scattering_events:
                raise RuntimeError(
                    'Scattering safety limit reached before complete histories.'
                )
            weights = weights[alive] * scattering[alive] / extinction[alive]
            points = points[alive]
            phase_rng = np.random.default_rng(
                np.random.SeedSequence([settings.seed, batch, order, 1])
            )
            directions = sample_phase(
                directions[alive],
                rayleigh[alive] / scattering[alive],
                atmosphere.aerosol_g,
                phase_rng,
            )
            order += 1
    single, multiple = scores.reshape((2, templates, *shape)) / settings.samples
    return HaloKernel(direct, single, multiple, tuple(hit_counts), outside)


def spread_flux(image, kernel):
    """Linear convolution, centred and zero-padded beyond the supplied source map."""
    source = np.asarray(image, dtype=float)
    filt = np.asarray(kernel, dtype=float)
    if source.ndim != 2 or filt.ndim != 2 or any(k % 2 != 1 for k in filt.shape):
        raise ValueError('Need a two-dimensional image and an odd centred kernel.')
    shape = tuple(n + k - 1 for n, k in zip(source.shape, filt.shape, strict=True))
    axes = (0, 1)
    result = np.fft.irfft2(
        np.fft.rfft2(source, s=shape, axes=axes)
        * np.fft.rfft2(filt, s=shape, axes=axes),
        s=shape,
        axes=axes,
    )
    y, x = (k // 2 for k in filt.shape)
    return result[y : y + source.shape[0], x : x + source.shape[1]]


def fit_nonnegative(components, observed, weights):
    """Small, exact active-set enumeration for the four linear proxy components."""
    x, y, w = (np.asarray(a, dtype=float) for a in (components, observed, weights))
    if (
        x.ndim != 2
        or not 1 <= x.shape[1] <= 4
        or y.shape != (len(x),)
        or w.shape != y.shape
        or len(y) < x.shape[1]
        or not all(np.all(np.isfinite(v)) for v in (x, y, w))
        or np.any(w <= 0)
    ):
        raise ValueError(
            'Expected finite data with one to four components and positive weights.'
        )
    a, b = x * w[:, None], y * w
    scales = np.linalg.norm(a, axis=0)
    usable = np.flatnonzero(scales > 0)
    scales[scales == 0] = 1.0
    a /= scales
    best = np.zeros(x.shape[1])
    loss = float(b @ b)
    for n in range(1, len(usable) + 1):
        for cols in combinations(usable, n):
            design = a[:, cols]
            c, *_ = np.linalg.lstsq(design, b, rcond=None)
            if np.any(c < 0):
                continue
            residual = design @ c - b
            value = float(residual @ residual)
            if value < loss:
                loss = value
                best.fill(0.0)
                best[list(cols)] = c / scales[list(cols)]
    return best, loss
