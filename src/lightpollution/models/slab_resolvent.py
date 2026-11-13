"""Isotropic slab integral equation solved as a finite linear system.

An independent reference for spherical Monte Carlo, not a replacement for
the horizontally varying spherical city problem. Spatial/angular quadrature
error remains; no finite scattering-order truncation enters the linear solve.
"""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class SlabSettings:
    layers: int = 512
    angular_order: int = 128
    optical_depth: float = 1.
    single_scattering_albedo: float = 1.
    view_cosine: float = 1.
    recorded_orders: int = 5
    series_absolute_tolerance: float = 1e-10
    series_safety_iterations: int = 100000

    def __post_init__(self):
        for name in ('layers', 'angular_order', 'recorded_orders', 'series_safety_iterations'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Slab discretization and iteration settings must be positive integers.')
        if self.layers < 4 or self.angular_order < 4:
            raise ValueError('Use >=4 spatial and angular nodes.')
        values = [self.optical_depth, self.single_scattering_albedo, self.view_cosine, self.series_absolute_tolerance]
        if (not np.all(np.isfinite(values)) or self.optical_depth <= 0
                or not 0 <= self.single_scattering_albedo <= 1 or not 0 < self.view_cosine <= 1
                or self.series_absolute_tolerance <= 0):
            raise ValueError('Invalid finite slab optical parameters or tolerance.')


def slab_system(settings: SlabSettings):
    """K,b,e: source S=b+KS and emerging radiance L=e dot S.

    Bottom boundary radiance=1 in the upper hemisphere. Both boundaries have
    zero incoming SCATTERED light. Integrate each cell's constant source
    analytically along characteristics; quadrature remains only in angle.
    """
    n = settings.layers
    nodes, weights = np.polynomial.legendre.leggauss(settings.angular_order)
    mu, weights = (nodes+1)/2, weights/2
    edges = np.linspace(0, settings.optical_depth, n+1)
    centres = (edges[1:]+edges[:-1])/2
    direct_mean = .5*(np.exp(-centres[:, None]/mu) @ weights)
    half = settings.optical_depth/(2*n)
    distance = abs(centres[:, None]-centres[None, :])
    kernel = np.zeros((n, n))
    for m, w in zip(mu, weights, strict=True):
        cell = np.exp(-np.maximum(0, distance-half)/m)*(-np.expm1(-2*half/m))
        np.fill_diagonal(cell, 2*(-np.expm1(-half/m)))
        kernel += .5*w*cell
    kernel *= settings.single_scattering_albedo
    forcing = settings.single_scattering_albedo*direct_mean
    escape = np.exp(-edges[:-1]/settings.view_cosine)*(-np.expm1(-np.diff(edges)/settings.view_cosine))
    return kernel, forcing, escape


def solve_slab(settings: SlabSettings = SlabSettings()):
    started = perf_counter()
    kernel, forcing, escape = slab_system(settings)
    assembled = perf_counter()
    q = float(np.max(np.sum(abs(kernel), axis=1)))
    if not q < 1:
        raise RuntimeError('The selected infinity-norm contraction certificate is unavailable; do not invent a tail bound.')
    system = np.eye(settings.layers)-kernel
    source = np.linalg.solve(system, forcing)
    solved = perf_counter()
    residual = float(np.max(abs(system @ source-forcing)))
    radiance = float(escape @ source)
    terms = []
    term = forcing.copy()
    for _ in range(settings.recorded_orders):
        terms.append(float(escape @ term))
        term = kernel @ term
    # Neumann iteration with an actual norm bound on the omitted tail,
    # independent of the value returned by the linear system solve.
    field, term = np.zeros_like(forcing), forcing.copy()
    tail_bound = float('inf')
    for iteration in range(1, settings.series_safety_iterations+1):
        field += term
        term = kernel @ term
        tail_bound = float(np.sum(escape)*np.max(abs(term))/(1-q))
        if tail_bound <= settings.series_absolute_tolerance:
            break
    else:
        raise RuntimeError('Series verification did not achieve its certified tail tolerance.')
    iterative_radiance = float(escape @ field)
    return {'solver': 'isotropic-slab-discrete-resolvent-1', 'settings': asdict(settings),
        'all_order_radiance': radiance, 'exact_order_radiance': terms,
        'cumulative_radiance': np.cumsum(terms).tolist(),
        'contraction_q_inf': q, 'absolute_algebraic_residual_inf': residual,
        'radiance_algebraic_error_bound': float(np.sum(escape)*residual/(1-q)),
        'verification_series_radiance': iterative_radiance,
        'verification_series_iterations': iteration, 'verification_series_tail_bound': tail_bound,
        'linear_vs_series_difference': abs(radiance-iterative_radiance),
        'assembly_seconds': assembled-started, 'linear_solve_seconds': solved-assembled,
        'total_runtime_seconds': perf_counter()-started,
        'finite_scattering_order_truncation': False, 'explicit_inverse_formed': False,
        'limitations': ['Isotropic, horizontally uniform plane-parallel slab, unit ground radiance.',
            'Linear-system and certified series errors do not bound spatial/angular quadrature error.',
            'This is an independent numerical reference, not a spherical city or field prediction.']}
