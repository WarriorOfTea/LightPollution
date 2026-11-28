"""Fit angular city parameters on VIIRS A1 overpasses for the ground transects."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds, PIXELS_PER_DEGREE
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.angular_sources import empirical_radiance
from lightpollution.models.satellite_halo import (
    build_halo_kernel,
    emission_radiance,
    fit_nonnegative,
    flux_below,
    spread_flux,
)
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M
from run_viirs_halo_recovery import (
    CITY,
    DATA,
    LABELS,
    ROOT,
    block_measurements,
    preflight,
)


LOCATIONS = {name: settings[:4] for name, settings in CITY.items()}
LOCATIONS['Berlin'] = (52.517, 13.388, Bounds(12.25, 51.9, 14.55, 53.15), 45)
LOCATIONS['Balaguer'] = (41.7911, 0.806, Bounds(-0.2, 41.15, 1.8, 42.45), 250)
DAYS = {
    'Berlin': ('2017-03-28', '2017-03-29'),
    'Madrid': ('2012-02-19', '2012-02-20', *CITY['Madrid'][-1]),
    'Tucson': tuple(sorted((*CITY['Tucson'][-1], '2019-04-01'))),
    'Balaguer': tuple(f'2016-05-{d:02}' for d in range(2, 13)),
}
HALO = LABELS[1]


def responses(mu):
    """Unit-flux Lambert/exponential, unit Near/Off, and the positive Off tail."""
    return np.concatenate(
        (
            emission_radiance(mu),
            np.stack(
                (
                    empirical_radiance(1.0, 0.0, mu),
                    empirical_radiance(0.0, 1.0, mu),
                    np.maximum(0.0, 1 - 2 * mu),
                ),
                axis=-1,
            ),
        ),
        axis=-1,
    )


def atmosphere_for_scene(city, day, source_reference, archives, override=None):
    berlin, observation, tucson = archives
    if override is not None:
        case = override
    elif city == 'Berlin':
        case = {
            'parameters': {'aerosol_optical_depth': berlin['aod_550']['aeronet']},
            'angstrom': berlin['angstrom_exponent'],
        }
    elif city == 'Tucson':
        case = tucson[day + '/daylight-aeronet']
    else:
        name = (
            'balaguer'
            if city == 'Balaguer'
            else {'2013-12-04': 'madrid-dec3', '2013-12-29': 'madrid-dec29'}.get(
                day, 'madrid-feb'
            )
        )
        case = observation[name]
    atmosphere = Atmosphere(
        **case['parameters'], rayleigh_multiplier=np.exp(-source_reference / 8000)
    )
    return replace(
        atmosphere,
        reference_wavelength_nm=650.0,
        aerosol_optical_depth=atmosphere.aerosol_optical_depth
        * (550 / 650) ** case['angstrom'],
    )


def statistics(y, prediction, classes, mask):
    scores = {}
    for code, label in enumerate(LABELS):
        selected = mask & (classes == code)
        residual = (prediction - y)[selected]
        if residual.size:
            scores[label] = {
                'n': int(residual.size),
                'mae': float(abs(residual).mean()),
                'bias': float(residual.mean()),
            }
    return scores


def evaluate(info, data, atmosphere, samples, seed):
    dy = EARTH_RADIUS_M * np.pi / (180 * PIXELS_PER_DEGREE)
    dx = dy * np.cos(np.deg2rad(data['lat']))
    kernel = build_halo_kernel(
        info['zenith_deg'],
        info['azimuth_deg'],
        dx,
        dy,
        150_000,
        atmosphere,
        AllOrderSettings(
            samples=samples, batch_size=32768, seed=seed, recorded_orders=1
        ),
        angular_radiance=responses,
    )
    spread = kernel.single + kernel.multiple

    def response(image, component):
        return image * kernel.direct[component] + np.maximum(
            0.0, spread_flux(image, spread[component])
        )

    n, o = data['paired_city_near'], data['paired_city_off']
    lambert = response(data['city_flux'], 0)
    low = response(data['city_flux'], 1)
    other = response(data['neighbor_flux'], 0)
    held = (
        response(np.maximum(0.0, data['city_flux'] - np.pi * n), 0)
        + response(n, 2)
        + response(o, 3)
    )
    tail = response(o, 4)
    y, x, ii, jj, classes = block_measurements(
        data, (lambert, low, other, np.ones_like(held), held, tail)
    )
    test = ((ii // 4) + 2 * (jj // 4)) % 3 == 0
    halo = classes == 1
    if min(np.count_nonzero(halo & test), np.count_nonzero(halo & ~test)) < 6:
        return info | {'status': 'insufficient_spatial_halo_blocks'}
    for code in (0, 2, 3):
        label = classes == code
        if min(np.count_nonzero(label & test), np.count_nonzero(label & ~test)) < 6:
            test[label] = False
        if not np.count_nonzero(label & ~test):
            return info | {'status': 'missing_training_' + LABELS[code]}
    train = ~test
    weight = np.ones(len(y))
    for code in range(len(LABELS)):
        selection = classes == code
        sample = y[selection & train]
        weight[selection] = 1 / (
            max(0.10, float(np.median(sample))) * np.sqrt(len(sample))
        )

    def fit(design, offset):
        coef, _ = fit_nonnegative(design[train], (y - offset)[train], weight[train])
        signal = offset + design @ coef
        return {
            'coefficients': coef.tolist(),
            'test': statistics(y, signal, classes, test),
            'train': statistics(y, signal, classes, train),
        }

    variants = {
        'a1_lambert': fit(x[:, [0, 2, 3]], np.zeros_like(y)),
        'a1_free': fit(x[:, :4], np.zeros_like(y)),
        'h1': fit(x[:, [2, 3]], x[:, 4]),
        'h2': fit(x[:, [2, 3]], x[:, 4] + x[:, 5]),
    }
    if np.linalg.norm(x[train, 5] * weight[train]) > 0:
        anchored = fit(x[:, [5, 2, 3]], x[:, 4] - x[:, 5])
        anchored['kappa'] = anchored['coefficients'][0] - 1
        variants['a1_anchored'] = anchored
    free = np.asarray(variants['a1_free']['coefficients'][:2])
    low_fraction = float(free @ flux_below(10) / free.sum()) if free.sum() > 0 else None
    return info | {
        'status': 'evaluated',
        'samples': samples,
        'seed': seed,
        'atmosphere': asdict(atmosphere),
        'free_power_fraction_below_10': low_fraction,
        'scenarios': variants,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('daily', 'maps', 'boundaries', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--samples', type=int, default=1_048_576)
    parser.add_argument('--seed', type=int, default=18115)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--city', choices=DAYS, action='append')
    parser.add_argument('--atmosphere-overrides', type=Path)
    args = parser.parse_args()
    geometry = read_json(args.boundaries)
    overrides = (
        read_json(args.atmosphere_overrides) if args.atmosphere_overrides else {}
    )
    archives = (
        read_json(ROOT / 'artifacts/berlin-propagation-20261126/results.json'),
        read_json(DATA / 'observations.json')['atmospheres'],
        read_json(DATA / 'weather-audit.json')['aod_recalculations']['atmospheres'],
    )
    scenes = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {}
        for city in args.city or DAYS:
            days = DAYS[city]
            location = LOCATIONS[city]
            for day in days:
                utc_day = date.fromisoformat(day)
                key = city + '/' + day
                missing = [
                    tile
                    for tile in location[2].tiles()
                    if not any(
                        args.daily.glob(f'VNP46A1.A{utc_day:%Y%j}.{tile}.002.*.h5')
                    )
                ]
                if missing:
                    scenes[key] = {
                        'city': city,
                        'night': day,
                        'status': 'missing_daily_a1',
                        'missing_tiles': missing,
                    }
                    continue
                info, data = preflight(
                    args.daily,
                    city,
                    utc_day,
                    geometry,
                    args.maps,
                    location=location,
                    minimum_class_cells=1,
                    compare_a2=False,
                )
                if data is None:
                    scenes[key] = info | {'status': 'excluded_by_clear_dark_support'}
                    print(key, scenes[key]['status'], flush=True)
                    continue
                override = overrides.get(key)
                if override is not None:
                    info['atmospheric_input'] = override
                atmosphere = atmosphere_for_scene(
                    city, day, location[-1], archives, override
                )
                jobs[key] = pool.submit(
                    evaluate, info, data, atmosphere, args.samples, args.seed
                )
        for key, job in jobs.items():
            scenes[key] = job.result()
            print(key, scenes[key]['status'], flush=True)
    write_json(
        args.output, {'boundaries_sha256': file_hash(args.boundaries), 'scenes': scenes}
    )


if __name__ == '__main__':
    main()
