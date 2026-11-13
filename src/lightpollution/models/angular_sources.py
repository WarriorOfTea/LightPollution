"""Research boundary radiances for comparing Black Marble viewing categories.

The maps are DNB proxies used in ONE monochromatic atmosphere, not measured
550 nm spectral radiances. Unknown cells are excluded from every variant on
the same support. No per-variant flux normalisation or spectral inversion.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..data import Bounds, PIXELS_PER_DEGREE


def horizontal_radiance_factor(mu):
    """pi*f_H(mu)/mu for the existing k=4 intensity law; mu must be positive."""
    mu = np.asarray(mu, dtype=float)
    if np.any(~np.isfinite(mu)) or np.any((mu <= 0) | (mu > 1)):
        raise ValueError('Radiance conversion needs 0 < mu <= 1.')
    return 2.5 * (1-mu)**4 / mu


@lru_cache(maxsize=16)
def bin_horizontal_factor(lo_deg, hi_deg, measure='solid_angle'):
    """Illustrative bin average; daily viewing-angle weights are NOT available."""
    if not 0 <= lo_deg < hi_deg < 90:
        raise ValueError('The viewing bin must lie in [0,90).')
    x, w = np.polynomial.legendre.leggauss(96)
    if measure == 'solid_angle':
        lo, hi = np.cos(np.deg2rad([hi_deg, lo_deg]))
        return float(w @ horizontal_radiance_factor(lo+(x+1)*(hi-lo)/2)/2)
    if measure == 'zenith_angle':
        theta = np.deg2rad(lo_deg+(x+1)*(hi_deg-lo_deg)/2)
        return float(w @ horizontal_radiance_factor(np.cos(theta))/2)
    raise ValueError('Unknown angular averaging measure.')


def eta_from_ratio(ratio, measure='solid_angle'):
    """Unconstrained diagnostic fit. Nonphysical eta is retained as incompatibility."""
    near = bin_horizontal_factor(0, 20, measure)
    off = bin_horizontal_factor(40, 60, measure)
    ratio = np.asarray(ratio, dtype=float)
    return (ratio-1)/((off-1)-ratio*(near-1))


def empirical_radiance(near, off, mu, horizon_multiplier=1.):
    """Constant in the measured bins, linear in mu across the missing ranges.

0--20deg is near; 20--40deg interpolates; 40--60deg is off. At 90deg
the value is horizon_multiplier*off. This is an explicit family of scenarios,
not a recovered angular distribution. Bin plateaus avoid inventing bin centres.
"""
    if not np.isfinite(horizon_multiplier) or horizon_multiplier < 0:
        raise ValueError('The horizon multiplier must be finite and nonnegative.')
    mu = np.clip(np.asarray(mu), 0, 1)
    c20, c40 = np.cos(np.deg2rad([20, 40]))
    t = np.clip((c20-mu)/(c20-c40), 0, 1)
    value = (1-t)*near+t*off
    horizon = off * (horizon_multiplier+(1-horizon_multiplier)*mu/.5)
    return np.where(mu < .5, horizon, value)


@dataclass
class AngularGridBoundary:
    bounds: Bounds
    near: np.ndarray
    off: np.ndarray
    all_angle: np.ndarray
    support: np.ndarray
    regional_eta: float

    labels = ('near_lambertian', 'off_lambertian', 'all_lambertian',
              'angular_horizon_zero', 'angular_horizon_hold', 'angular_horizon_double',
              'mixed_eta_0.1_cap85', 'mixed_eta_regional_cap85', 'mixed_eta_regional_cap88')

    def __post_init__(self):
        self.support = np.asarray(self.support, dtype=bool).copy()
        if self.support.shape != self.bounds.shape:
            raise ValueError('Support shape disagrees with bounds.')
        if not 0 <= self.regional_eta < 1 or not np.isfinite(self.regional_eta):
            raise ValueError('The regional diagnostic mixture requires 0 <= eta < 1.')
        for name in ('near', 'off', 'all_angle'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != self.bounds.shape or np.any(~np.isfinite(value[self.support])) or np.any(value[self.support] < 0):
                raise ValueError('Boundary maps must be finite and nonnegative on the common support.')
            setattr(self, name, np.where(self.support, value, 0.))
        n, o, a = [float(v.max(initial=0)) for v in (self.near, self.off, self.all_angle)]
        gn = bin_horizontal_factor(0, 20)
        mixed = []
        for eta, angle in ((.1, 85), (self.regional_eta, 85), (self.regional_eta, 88)):
            factor = (1-eta+eta*horizontal_radiance_factor(np.cos(np.deg2rad(angle))))/(1-eta+eta*gn)
            mixed.append(n*float(factor))
        self.upper_bounds = np.array([n, o, a, max(n, o), max(n, o), max(n, 2*o), *mixed])

    def sample(self, points, outgoing):
        """Return NxV radiances and support codes: 0 outside, 1 missing, 2 known."""
        points = np.asarray(points)
        lat = np.rad2deg(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
        lon = np.rad2deg(np.arctan2(points[:, 1], points[:, 0]))
        rows = np.floor((self.bounds.north-lat)*PIXELS_PER_DEGREE).astype(int)
        cols = np.floor((lon-self.bounds.west)*PIXELS_PER_DEGREE).astype(int)
        inside = (rows >= 0) & (rows < self.bounds.shape[0]) & (cols >= 0) & (cols < self.bounds.shape[1])
        ri, ci = np.clip(rows, 0, self.bounds.shape[0]-1), np.clip(cols, 0, self.bounds.shape[1]-1)
        supported = inside & self.support[ri, ci]
        code = np.where(inside, np.where(supported, 2, 1), 0)
        n, o, a = [value[ri, ci] for value in (self.near, self.off, self.all_angle)]
        mu = np.clip(np.einsum('ij,ij->i', outgoing, points/np.linalg.norm(points, axis=1)[:, None]), 0, 1)
        values = [n, o, a, *(empirical_radiance(n, o, mu, tail) for tail in (0., 1., 2.))]
        gn = bin_horizontal_factor(0, 20)
        for eta, angle in ((.1, 85), (self.regional_eta, 85), (self.regional_eta, 88)):
            # A declared cap on RADIANCE, not silently the unbounded intensity law.
            h = horizontal_radiance_factor(np.maximum(mu, np.cos(np.deg2rad(angle))))
            values.append(n*(1-eta+eta*h)/(1-eta+eta*gn))
        return np.where(supported[:, None], np.stack(values, axis=1), 0.), code
