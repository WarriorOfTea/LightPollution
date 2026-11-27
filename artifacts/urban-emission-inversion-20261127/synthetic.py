"""Synthetic identifiability check for urban radiance inversions, with optical depths held known."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls

from lightpollution.adapters.storage import write_json

WAVELENGTHS_NM = np.array([450.0, 590.0, 720.0])
SPECTRA = np.array([[0.32, 0.51, 0.17], [0.02, 0.78, 0.20]])
BANDS = {'DNB': [0.0, 0.90, 0.70], 'blue': [1.0, 0.0, 0.0], 'amber': [0.0, 1.0, 0.0]}
TRUE_POWER = np.array([0.30, 0.20, 0.08, 0.42])
TRUE_HEIGHT_KM = 1.40
TAU_550 = 0.12
HEIGHTS_KM = np.linspace(0.55, 3.05, 101)
OFFSETS_KM = np.array([8.0, 12.0, 18.0, 25.0, 34.0, 46.0, 60.0, 78.0])
LOW_ANGLE_SCALE = np.sin(np.deg2rad(6.0))
FRAC_LED = np.array([1.0, 1.0, 0.0, 0.0])
VIEWS = [
    ('DNB', 0),
    ('DNB', 48),
    ('blue', 0),
    ('amber', 0),
    ('DNB', -32),
    ('blue', -32),
]
SEED = 1729


def angular_integral(mu):
    u = mu / LOW_ANGLE_SCALE
    return LOW_ANGLE_SCALE**2 * (1 - (1 + u) * np.exp(-u))


def angular_intensity(mu):
    low = mu * np.exp(-mu / LOW_ANGLE_SCALE) / (2 * np.pi * angular_integral(1.0))
    return np.stack((low, mu / np.pi), axis=-1)


def fraction_below_10():
    mu = np.sin(np.deg2rad(10.0))
    return np.tile([angular_integral(mu) / angular_integral(1.0), mu**2], 2)


@lru_cache(None)
def quadrature(order):
    return np.polynomial.legendre.leggauss(order)


def scattered_spectrum(
    offset_km, angle_deg, height_km, order=144, elevated_fraction=0.0
):
    """Slant viewing ray through an exponential, two-component clear atmosphere."""
    nodes, weights = quadrature(order)
    altitude = 50 * (nodes + 1)
    angle = np.deg2rad(angle_deg)
    x = np.asarray(offset_km)[:, None] + altitude * np.tan(angle)
    distance = np.hypot(x, altitude)
    mu = altitude / distance
    cos_scatter = (x * np.sin(angle) + altitude * np.cos(angle)) / distance
    phase_m = 3 / (16 * np.pi) * (1 + cos_scatter**2)
    g_aerosol = 0.73
    phase_a = (1 - g_aerosol**2) / (
        4 * np.pi * (1 + g_aerosol**2 - 2 * g_aerosol * cos_scatter) ** 1.5
    )
    layer_heights = np.array([height_km, 5.0, 8.0])
    bottom = np.exp(-altitude[:, None] / layer_heights)
    tau_a = TAU_550 * (550 / WAVELENGTHS_NM) ** 1.2
    tau_m = 0.095 * (550 / WAVELENGTHS_NM) ** 4
    tau_layers = np.vstack(
        ((1 - elevated_fraction) * tau_a, elevated_fraction * tau_a, tau_m)
    )
    beta = bottom[:, :, None] * tau_layers / layer_heights[None, :, None]
    toward_scatter = (1 - bottom) @ tau_layers
    toward_space = bottom @ tau_layers
    transmission = np.exp(
        -toward_scatter[None, :, :] / mu[:, :, None]
        - toward_space[None, :, :] / np.cos(angle)
    )
    aerosol_beta = beta[:, :2].sum(axis=1)
    scatter = (
        0.93 * phase_a[:, :, None] * aerosol_beta + phase_m[:, :, None] * beta[:, 2]
    )
    common = (
        transmission
        * scatter
        * (50 * weights / np.cos(angle))[None, :, None]
        / distance[:, :, None] ** 2
    )
    return np.einsum('rzl,rza->rla', common, angular_intensity(mu))


def basis(spec, band):
    response = np.asarray(BANDS[band])
    return np.einsum('rla,sl,l->rsa', spec, SPECTRA, response).reshape(len(spec), 4)


def offsets_for_view(angle):
    return OFFSETS_KM if angle == 0 else np.r_[-OFFSETS_KM[::-1], OFFSETS_KM]


def observation_matrices(height, order=144, elevated_fraction=0.0):
    return [
        basis(
            scattered_spectrum(
                offsets_for_view(theta), theta, height, order, elevated_fraction
            ),
            band,
        )
        for band, theta in VIEWS
    ]


def ratio_generators(fraction, values):
    """Rays spanning the cone q >= 0, (values - fraction) @ q = 0."""
    shifted = values - fraction
    eye = np.eye(len(values))
    generators = [eye[i] for i in np.flatnonzero(np.isclose(shifted, 0, atol=1e-11))]
    for i in np.flatnonzero(shifted > 1e-11):
        for j in np.flatnonzero(shifted < -1e-11):
            ray = -shifted[j] * eye[i] + shifted[i] * eye[j]
            generators.append(ray / ray.sum())
    return np.column_stack(generators)


def profile_ratio(values, design, y, height_cost):
    grid = np.linspace(values.min(), values.max(), 101)
    costs = []
    for target in grid:
        generators = ratio_generators(target, values)
        candidates = [
            nnls(x @ generators, y)[1] ** 2 + cost
            for x, cost in zip(design, height_cost, strict=True)
        ]
        costs.append(float(min(candidates)))
    return grid, np.array(costs)


def measured_data(matrices):
    rng = np.random.default_rng(SEED)
    observations, lower = [], []
    for x in matrices:
        signal = x @ TRUE_POWER
        floor = 0.012 * signal.max()
        background_sd = 0.018 * signal.max()
        sd = np.hypot(0.065 * signal, floor)
        covariance = np.diag(sd**2) + background_sd**2
        root = cholesky(covariance, lower=True)
        # Readings are expressed after independent estimation of the background level.
        y = signal + root @ rng.standard_normal(len(signal))
        observations.append(y)
        lower.append(root)
    return observations, lower


def fit_design(
    groups, observations, lower, grid_design, truth_design, height_prior, ratios
):
    y = np.concatenate(
        [solve_triangular(lower[i], observations[i], lower=True) for i in groups]
    )
    design = [
        np.vstack([solve_triangular(lower[i], matrices[i], lower=True) for i in groups])
        for matrices in grid_design
    ]
    height_cost = (
        ((HEIGHTS_KM - TRUE_HEIGHT_KM) / height_prior) ** 2
        if height_prior
        else np.zeros(len(HEIGHTS_KM))
    )
    fits = [nnls(x, y) for x in design]
    costs = np.array([r[1] ** 2 for r in fits]) + height_cost
    best = int(np.argmin(costs))
    power = fits[best][0]
    fit_band_errors = {}
    for i in (4, 5):
        ref = truth_design[i] @ TRUE_POWER
        predicted = grid_design[best][i] @ power
        fit_band_errors[VIEWS[i][0]] = float(
            np.linalg.norm(predicted - ref) / np.linalg.norm(ref)
        )
    modes = {}
    for name, value in ratios.items():
        grid, loss = profile_ratio(value, design, y, height_cost)
        selected = grid[loss <= costs[best] + 3.84]
        modes[name] = {
            'estimate': float(value @ power / power.sum()),
            'min': float(selected[0]),
            'max': float(selected[-1]),
            'grid': grid.tolist(),
            'delta_chi_sq': (loss - costs[best]).tolist(),
        }
    acceptable_height = HEIGHTS_KM[costs <= costs[best] + 3.84]
    svals = np.linalg.svd(design[best], compute_uv=False)
    return {
        'group_indices': list(groups),
        'n': len(y),
        'H_km': float(HEIGHTS_KM[best]),
        'H_interval_km': [float(acceptable_height[0]), float(acceptable_height[-1])],
        'height_anchor_sigma_km': height_prior,
        'chi_sq': float(costs[best]),
        'power': power.tolist(),
        'modes': modes,
        'heldout_relative_l2': fit_band_errors,
        'singular_values_at_fixed_height': svals.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    power_nodes, power_weights = quadrature(144)
    normalized = np.pi * power_weights @ angular_intensity((1 + power_nodes) / 2)
    np.testing.assert_allclose(normalized, [1.0, 1.0], atol=1e-12)
    truth = observation_matrices(TRUE_HEIGHT_KM)
    refined = observation_matrices(TRUE_HEIGHT_KM, order=288)
    convergence = max(
        float(np.max(np.abs(a - b) / np.maximum(b, 1e-30)))
        for a, b in zip(truth, refined, strict=True)
    )
    assert convergence < 0.004
    observed, lower = measured_data(truth)
    grid = [observation_matrices(h) for h in HEIGHTS_KM]
    ratios = {'fraction_below_10deg': fraction_below_10(), 'fraction_led': FRAC_LED}
    designs = [
        ('nadir_dnb', (0,), None),
        ('second_dnb_view', (0, 1), None),
        ('add_blue_amber', (0, 1, 2, 3), None),
        ('independent_aerosol_height', (0, 1, 2, 3), 0.20),
    ]
    packed = {
        name: fit_design(groups, observed, lower, grid, truth, prior, ratios)
        for name, groups, prior in designs
    }
    # Deliberately generate a scene outside the fitted family, with the same total AOD.
    elevated_fraction = 0.28
    missed_layer = observation_matrices(
        TRUE_HEIGHT_KM, elevated_fraction=elevated_fraction
    )
    target, target_noise = measured_data(missed_layer)
    mismatch = fit_design(
        (0, 1, 2, 3), target, target_noise, grid, missed_layer, None, ratios
    )
    result = {
        'protocol': 'Synthetic point source, optical depths fixed, exponential molecular/aerosol extinction, first scattering only.',
        'seed': SEED,
        'max_integration_refinement_relative': convergence,
        'wavelengths_nm': WAVELENGTHS_NM.tolist(),
        'hypothetical_band_responses': BANDS,
        'example_lamp_energy_in_three_bins': SPECTRA.tolist(),
        'offsets_km': OFFSETS_KM.tolist(),
        'band_and_signed_zenith_degrees': VIEWS,
        'tau_a_550': TAU_550,
        'power_truth': TRUE_POWER.tolist(),
        'aerosol_height_truth_km': TRUE_HEIGHT_KM,
        'moment_truth': {k: float(v @ TRUE_POWER) for k, v in ratios.items()},
        'angular_template_power_fractions_below_10deg': fraction_below_10()[
            :2
        ].tolist(),
        'height_grid_km': HEIGHTS_KM.tolist(),
        'observations': [
            {'y': y.tolist(), 'noise_cholesky': root.tolist()}
            for y, root in zip(observed, lower, strict=True)
        ],
        'fits': packed,
        'out_of_family': {
            'truth_aod_fraction_at_5_km_scale': elevated_fraction,
            'truth_aerosol_component_scales_km': [TRUE_HEIGHT_KM, 5.0],
            'one_layer_fit': mismatch,
        },
    }
    write_json(args.output, result)
    print(
        'omitted_upper_aerosol',
        mismatch['H_km'],
        mismatch['chi_sq'],
        mismatch['heldout_relative_l2'],
        {k: v['estimate'] for k, v in mismatch['modes'].items()},
        flush=True,
    )
    for k, v in packed.items():
        print(
            k,
            v['n'],
            v['H_km'],
            v['H_interval_km'],
            v['heldout_relative_l2'],
            {n: [x['estimate'], x['min'], x['max']] for n, x in v['modes'].items()},
            flush=True,
        )


if __name__ == '__main__':
    main()
