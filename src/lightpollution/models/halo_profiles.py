"""Energy-neutral angular perturbations and fits to differences of halo radiance."""

import numpy as np


KAPPA_BOUNDS = (-2.0, 5.0 + 2.0 * np.sqrt(6.0))


def redistribution_radiance(mu):
    """Unit OffNadir perturbation of H2, where mu is sin(elevation)."""
    mu = np.clip(np.asarray(mu, dtype=float), 0, 1)
    return np.maximum(0.0, 1 - 2 * mu) * (1 - 4 * mu)


def redistribution_flux_below(elevation_deg):
    """Integral of mu times the perturbation, without the factor 2*pi*OffNadir."""
    if not 0 <= elevation_deg <= 90:
        raise ValueError('Elevation must lie in [0, 90].')
    mu = np.minimum(0.5, np.sin(np.deg2rad(elevation_deg)))
    return 0.5 * (mu * (1 - 2 * mu)) ** 2


def profile_normal_terms(contrast, covariance, residual, response):
    """Normal equation for a scalar shape using possibly redundant differences."""
    contrast = np.asarray(contrast, dtype=float)
    if not len(contrast):
        return None
    variance = contrast @ covariance @ contrast.T
    values, vectors = np.linalg.eigh(variance)
    active = values > max(0.0, values[-1]) * 1e-10
    if not np.any(active):
        return None
    whitened = (vectors[:, active].T @ contrast) / np.sqrt(values[active, None])
    y, x = whitened @ residual, whitened @ response
    denominator = float(x @ x)
    if denominator <= 0:
        return None
    return {
        'numerator': float(x @ y),
        'denominator': denominator,
        'rank': int(active.sum()),
    }


def pool_profile_fits(normals):
    """Combine fits with equal weight per pass after accounting for profile rank."""
    if not normals:
        return None
    numerator = sum(n['numerator'] / n['rank'] for n in normals)
    denominator = sum(n['denominator'] / n['rank'] for n in normals)
    optimum = numerator / denominator
    return {
        'unclipped_kappa': float(optimum),
        'kappa': float(np.clip(optimum, *KAPPA_BOUNDS)),
        'at_bound': bool(not KAPPA_BOUNDS[0] < optimum < KAPPA_BOUNDS[1]),
        'weighted_curvature': float(denominator),
    }
