"""Isolate scattering, angular power, scene geometry and nuisance terms in the A1 fit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import PIXELS_PER_DEGREE
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.satellite_halo import (
    build_halo_kernel,
    fit_nonnegative,
    spread_flux,
)
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M
from run_halo_city_fits import LOCATIONS, responses, statistics
from run_viirs_halo_recovery import LABELS, block_measurements, preflight


def angular_basis(mu):
    middle = 4 * np.clip(mu, 0, 0.5) * np.maximum(0, 1 - 2 * mu)
    return np.concatenate((responses(mu), np.asarray(middle)[..., None]), axis=-1)


def gaussian(dx, dy, fwhm_km):
    sigma = fwhm_km * 1000 / np.sqrt(8 * np.log(2))
    nx, ny = np.ceil(4 * sigma / np.array([dx, dy])).astype(int)
    u, v = np.arange(-nx, nx + 1) * dx, np.arange(-ny, ny + 1) * dy
    values = np.exp(-0.5 * (u[None, :] ** 2 + v[:, None] ** 2) / sigma**2)
    return values / values.sum()


def spatial_terms(data, kernel, weak_context, include_middle):
    n, o = data['paired_city_near'], data['paired_city_off']
    neighbor = data['neighbor_flux']
    missing_off = np.maximum(0, data['city_flux'] - np.pi * n)
    fields = (
        ((missing_off, 0), (n, 2), (o, 3)),
        ((o, 4),),
        ((neighbor, 0),),
        ((weak_context, 0),),
    )
    if include_middle:
        fields += (((o, 5),),)
    expand = lambda image, coeff, filters: (
        image * coeff if filters is None else np.maximum(0, spread_flux(image, filters))
    )
    return [
        np.stack(
            [
                sum(
                    expand(
                        image,
                        kernel.direct[index],
                        None if filters is None else filters[index],
                    )
                    for image, index in group
                )
                for group in fields
            ],
            axis=-1,
        )
        for filters in (None, kernel.single, kernel.multiple)
    ]


def measurement_vectors(data, terms):
    shape = data['raw'].shape
    features = (
        *(terms[:, :, i] for i in range(terms.shape[-1])),
        np.broadcast_to(data['north'], shape),
        np.broadcast_to(data['east'], shape),
    )
    y, x, ii, jj, labels = block_measurements(data, features)
    validation = ((ii // 4) + 2 * (jj // 4)) % 3 == 0
    for code in (0, 2, 3):
        label = labels == code
        if (
            min(
                np.count_nonzero(label & validation),
                np.count_nonzero(label & ~validation),
            )
            < 6
        ):
            validation[label] = False
    weights = np.ones(len(y))
    for code in range(len(LABELS)):
        selected = labels == code
        calibration = y[selected & ~validation]
        weights[selected] = 1 / (
            max(0.10, float(np.median(calibration))) * np.sqrt(len(calibration))
        )
    return y, x[:, :-2], x[:, -2:], labels, validation, weights


def regression(design, target, weights, gradient, maximum=None):
    """Nonnegative components and a free signed background gradient, if requested."""
    a = design * weights[:, None]
    b = target * weights
    if gradient.shape[1]:
        g = gradient * weights[:, None]
        joint = np.column_stack((b, a))
        joint -= g @ np.linalg.lstsq(g, joint, rcond=None)[0]
        b, a = joint[:, 0], joint[:, 1:]
    c, _ = fit_nonnegative(a, b, np.ones(len(b)))
    if maximum is not None and c[0] > maximum:
        endpoint, _ = fit_nonnegative(a[:, 1:], b - maximum * a[:, 0], np.ones(len(b)))
        c = np.r_[maximum, endpoint]
    slope = np.linalg.lstsq(
        gradient * weights[:, None], (target - design @ c) * weights, rcond=None
    )[0]
    return c, slope


def contrasts(positions, labels, use, azimuth):
    """Near-equal radial and cross-track bins on opposite sides of the city."""
    north, east = positions.T
    phi = np.deg2rad(azimuth)
    along = north * np.cos(phi) + east * np.sin(phi)
    across = east * np.cos(phi) - north * np.sin(phi)
    r = np.hypot(north, east)
    direction = abs(along) >= abs(across)
    groups = np.column_stack((np.floor((r - 18) / 5), np.floor(across / 8))).astype(int)
    keep = use & (labels == 1) & direction
    pairs = []
    for group in np.unique(groups[keep], axis=0):
        subset = keep & np.all(groups == group, axis=1)
        front = np.flatnonzero(subset & (along > 0))
        back = np.flatnonzero(subset & (along < 0))
        if min(front.size, back.size) >= 2:
            pairs.append((front, back))
    return pairs


def difference(values, pairs):
    return np.array([values[a].mean(axis=0) - values[b].mean(axis=0) for a, b in pairs])


def fit_vector(
    vectors, azimuth, method='original', fixed_kappa=None, controls_only=False
):
    y, x, pos, classes, test, weight = vectors
    base, tail, neighbor, weak = x[:, :4].T
    conserving = method in ('conserve', 'contrast_conserve')
    if conserving:
        tail = tail - x[:, 4]
    gradient = pos / 65 if method in ('plane', 'weak_and_plane') else pos[:, :0]
    nuisance = np.column_stack((neighbor, np.ones(len(y))))
    if method in ('weak', 'weak_and_plane'):
        nuisance = np.column_stack((nuisance, weak))
    training_contrasts = contrasts(pos, classes, ~test, azimuth)
    testing_contrasts = contrasts(pos, classes, test, azimuth)
    fit = ~test
    if controls_only or method.startswith('contrast'):
        fit = fit & (classes != 1)
    if method.startswith('contrast'):
        c, slope = regression(
            nuisance[fit], (y - base)[fit], weight[fit], gradient[fit]
        )
        predicted = base + nuisance @ c
        d = difference(tail, training_contrasts)
        if not len(d) or not np.any(d):
            return None
        value = float(d @ difference(y - predicted, training_contrasts) / (d @ d))
        kappa = np.clip(value, -1, 8 if conserving else np.inf)
        model = predicted + kappa * tail
    elif fixed_kappa is not None:
        kappa = fixed_kappa
        offset = base + kappa * tail
        c, slope = regression(
            nuisance[fit], (y - offset)[fit], weight[fit], gradient[fit]
        )
        model = offset + nuisance @ c + gradient @ slope
    else:
        design = np.column_stack((tail, nuisance))
        c, slope = regression(
            design[fit],
            (y - base + tail)[fit],
            weight[fit],
            gradient[fit],
            maximum=9 if conserving else None,
        )
        kappa, c = c[0] - 1, c[1:]
        model = base + kappa * tail + nuisance @ c + gradient @ slope
    d_observed = difference(y, testing_contrasts)
    d_model = difference(model, testing_contrasts)
    field = (classes == 1) & test
    return {
        'kappa': float(kappa),
        'context_and_background': c.tolist(),
        'background_gradient_per_65km': slope.tolist(),
        'test': statistics(y, model, classes, test),
        'halo_test_observed_mean': float(y[field].mean()),
        'contrast': {
            'train_pairs': len(training_contrasts),
            'test_pairs': len(testing_contrasts),
            'observed': d_observed.tolist(),
            'predicted': d_model.tolist(),
        },
    }


def view_profile(vectors):
    """Describe how much of the tail signature remains after the fitted nuisance terms."""
    _, x, pos, labels, test, weight = vectors
    selected = (labels == 1) & ~test
    y = x[selected, 1] * weight[selected]
    nuisance = np.column_stack(
        (x[selected, 2:4], np.ones(int(selected.sum())), pos[selected] / 65)
    )
    a = nuisance * weight[selected, None]
    left = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return float(np.linalg.norm(left) / np.linalg.norm(y))


def flux_integrals(data, kappas):
    cell_area = np.cos(np.deg2rad(data['bounds'].latitude))[:, None]
    near = []
    for key in ('city_flux', 'paired_city_near', 'paired_city_off'):
        near.append(float(np.sum(cell_area * data[key])))
    total, n, o = near
    mu_nodes, weights = np.polynomial.legendre.leggauss(24)

    def integrate(high, kappa, conserving=False):
        edges = sorted(
            set(
                [0.0, high]
                + [
                    v
                    for v in (0.5, np.cos(np.deg2rad(40)), np.cos(np.deg2rad(20)))
                    if v < high
                ]
            )
        )
        power = 0.0
        for lo, hi in zip(edges, edges[1:]):
            mu = lo + (mu_nodes + 1) * (hi - lo) / 2
            response = angular_basis(mu)
            value = (
                (total - np.pi * n) * response[:, 0]
                + n * response[:, 2]
                + o
                * (
                    response[:, 3]
                    + kappa * (response[:, 4] - conserving * response[:, 5])
                )
            )
            power += np.pi * (hi - lo) * (weights @ (value * mu))
        return power

    out = {
        'paired_off_over_all_near': o / (total / np.pi),
        'paired_near_fraction': n / (total / np.pi),
    }
    power_h2 = integrate(1, 1)
    for name, kappa in kappas.items():
        conserve = name in ('conserve', 'contrast_conserve')
        uniform = integrate(1, kappa, conserve)
        if conserve:
            np.testing.assert_allclose(uniform, integrate(1, 0), rtol=1e-12)
        out[name] = {
            'power_over_h2': uniform / power_h2,
            'below_10_deg_fraction': integrate(np.sin(np.deg2rad(10)), kappa, conserve)
            / uniform,
            'below_25_deg_fraction': integrate(np.sin(np.deg2rad(25)), kappa, conserve)
            / uniform,
        }
    return out


def evaluate(scene, data, samples):
    air = Atmosphere(**scene['atmosphere'])
    z, az = (scene[k] for k in ('zenith_deg', 'azimuth_deg'))
    dy = EARTH_RADIUS_M * np.pi / (180 * PIXELS_PER_DEGREE)
    dx = dy * np.cos(np.deg2rad(data['lat']))
    weak = np.where(
        (~data['city_mask'])
        & (data['near_radiance'] >= 0.3)
        & (data['near_radiance'] < 2),
        np.pi * data['near_radiance'],
        0,
    )
    settings = AllOrderSettings(
        samples=samples, batch_size=32768, seed=scene['seed'], recorded_orders=1
    )

    def simulate(atmosphere, *, seed=None, baseline=False, wavelength=None, zenith=z):
        at = atmosphere
        if wavelength is not None:
            # The archived grey calculation has no independent measured spectral mixture.
            at = replace(
                at,
                reference_wavelength_nm=wavelength,
                aerosol_optical_depth=at.aerosol_optical_depth
                * (650 / wavelength) ** 1.1,
            )
        kernel = build_halo_kernel(
            zenith,
            az,
            dx,
            dy,
            150_000,
            at,
            settings if seed is None else replace(settings, seed=seed),
            angular_radiance=angular_basis if baseline else responses,
        )
        return spatial_terms(
            data, kernel, weak if baseline else np.zeros_like(weak), baseline
        )

    parts = simulate(air, baseline=True)
    all_terms = sum(parts)
    vectors = measurement_vectors(data, all_terms)
    fits = {
        name: fit_vector(vectors, az, name)
        for name in (
            'original',
            'conserve',
            'contrast',
            'contrast_conserve',
            'weak',
            'plane',
            'weak_and_plane',
        )
    }
    fits.update(
        {m: fit_vector(vectors, az, fixed_kappa=k) for m, k in (('h1', 0), ('h2', 1))}
    )
    fits['single'] = fit_vector(measurement_vectors(data, parts[0] + parts[1]), az)
    for name, alt in (
        (
            'height_low',
            replace(air, aerosol_scale_height_m=air.aerosol_scale_height_m * 0.65),
        ),
        (
            'height_high',
            replace(air, aerosol_scale_height_m=air.aerosol_scale_height_m * 1.8),
        ),
        (
            'aod_low',
            replace(air, aerosol_optical_depth=air.aerosol_optical_depth * 0.6),
        ),
        (
            'aod_high',
            replace(air, aerosol_optical_depth=air.aerosol_optical_depth * 1.5),
        ),
        ('g_062', replace(air, aerosol_g=0.62)),
    ):
        fits[name] = fit_vector(measurement_vectors(data, sum(simulate(alt))), az)
    for wave in (550, 790):
        fits[f'wave_{wave}'] = fit_vector(
            measurement_vectors(data, sum(simulate(air, wavelength=wave))), az
        )
    if scene['city'] == 'Tucson':
        # Barentine et al. (2020), the phase-function proxy used on the ground.
        g = float(np.interp(650, [400, 680], [0.68, 0.64]))
        fits['tucson_ground_g'] = fit_vector(
            measurement_vectors(data, sum(simulate(replace(air, aerosol_g=g)))), az
        )
    z_lo, z_hi = scene['sensor_zenith_05_95_deg']
    zenith_weight = np.clip((data['sensor_zenith'] - z_lo) / (z_hi - z_lo), 0, 1)
    zenith_weight = np.where(np.isfinite(zenith_weight), zenith_weight, 0.5)[..., None]
    varying = (
        sum(simulate(air, zenith=z_lo)) * (1 - zenith_weight)
        + sum(simulate(air, zenith=z_hi)) * zenith_weight
    )
    fits['pixel_zenith'] = fit_vector(measurement_vectors(data, varying), az)
    if scene['night'] in ('2017-03-29', '2013-12-29', '2019-04-03'):
        fits['second_seed'] = fit_vector(
            measurement_vectors(data, sum(simulate(air, seed=scene['seed'] + 13))), az
        )

    short, wing = gaussian(dx, dy, 1.1), gaussian(dx, dy, 10.0)
    blurred = np.stack(
        [
            0.98 * spread_flux(term, short) + 0.02 * spread_flux(term, wing)
            for term in np.moveaxis(all_terms, -1, 0)
        ],
        axis=-1,
    )
    fits['blur_stress'] = fit_vector(measurement_vectors(data, blurred), az)
    y, _, pos, labels, test, _ = vectors
    kappa = fits['original']['kappa']
    if samples == scene['samples']:
        np.testing.assert_allclose(
            kappa, scene['scenarios']['a1_anchored']['kappa'], rtol=1e-10, atol=1e-10
        )
    neighbor = fits['original']['context_and_background'][0]
    order_means = []
    for part in parts:
        _, design, _, _, _, _ = measurement_vectors(data, part)
        halo = (labels == 1) & test
        order_means.append(
            {
                'tail_response_mean': float(design[halo, 1].mean()),
                'city_mean': float((design[halo, 0] + kappa * design[halo, 1]).mean()),
                'neighbor_mean': float(neighbor * design[halo, 2].mean()),
            }
        )
    power = flux_integrals(
        data,
        {
            k: fits[k]['kappa']
            for k in ('original', 'conserve', 'h1', 'h2', 'contrast_conserve')
            if fits[k] is not None
        },
    )
    result = {
        'fits': fits,
        'angular_power': power,
        'order_means': order_means,
        'tail_norm_after_halo_nuisance_fraction': view_profile(vectors),
        'test_halo_quantiles': np.quantile(
            y[(labels == 1) & test], [0.1, 0.5, 0.9]
        ).tolist(),
        'zenith_deg': z,
        'azimuth_deg': az,
        'atmosphere': scene['atmosphere'],
    }
    return result, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('daily', 'maps', 'boundaries', 'previous_satellite', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--samples', type=int, default=1_048_576)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    blob = read_json(args.previous_satellite)
    geometry = read_json(args.boundaries)
    if file_hash(args.boundaries) != blob['boundaries_sha256']:
        raise ValueError('Expected the same city boundaries as in the baseline.')
    results, blocks = {}, {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {}
        for key, scene in blob['scenes'].items():
            if scene['status'] != 'evaluated':
                continue
            city, day = key.split('/')
            info, data = preflight(
                args.daily,
                city,
                date.fromisoformat(day),
                geometry,
                args.maps,
                location=LOCATIONS[city],
                minimum_class_cells=1,
                compare_a2=False,
            )
            if (
                info['source_template_files'] != scene['source_template_files']
                or info['files'] != scene['files']
            ):
                raise ValueError(f'Input provenance differs for {key}')
            jobs[key] = pool.submit(evaluate, scene, data, args.samples)
        for key, job in jobs.items():
            results[key], blocks[key] = job.result()
            fits = results[key]['fits']
            print(
                key,
                {
                    k: round(fits[k]['kappa'], 3)
                    for k in ('original', 'conserve', 'weak')
                },
                flush=True,
            )
    for key, vector in blocks.items():
        candidates = [k for k in results if k != key and k[:-3] == key[:-3]]
        if not candidates:
            continue
        az = results[key]['azimuth_deg']
        prediction = results[key]['fits']
        for name in ('original', 'conserve', 'contrast_conserve'):
            donors = [k for k in candidates if results[k]['fits'][name] is not None]
            if donors:
                kappa = float(
                    np.median([results[k]['fits'][name]['kappa'] for k in donors])
                )
                method = 'original' if name == 'original' else 'conserve'
                predictive = fit_vector(
                    vector, az, method, fixed_kappa=kappa, controls_only=True
                )
                prediction['leave_pass_out_' + name] = predictive | {'donors': donors}
        prediction['leave_pass_out_h2'] = fit_vector(
            vector, az, fixed_kappa=1, controls_only=True
        )
    write_json(
        args.output,
        {
            'scenes': results,
            'samples_per_kernel': args.samples,
            'kernel_count': len(results) * 10
            + sum(
                ('second_seed' in s['fits']) + ('tucson_ground_g' in s['fits'])
                for s in results.values()
            ),
            'prior_satellite_sha256': file_hash(args.previous_satellite),
            'angstrom_exponent_for_wavelength_stress': 1.1,
        },
    )


if __name__ == '__main__':
    main()
