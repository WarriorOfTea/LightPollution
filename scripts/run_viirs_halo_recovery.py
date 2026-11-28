"""Fit a two-template DNB source on screened TOA VIIRS scenes, with spatial holdouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import read_json, write_json, file_hash
from lightpollution.adapters.viirs_daily import clear_dark_a1, read_daily_region
from lightpollution.data import Bounds, PIXELS_PER_DEGREE, good_mask, read_region
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.satellite_halo import (
    build_halo_kernel,
    fit_nonnegative,
    flux_below,
    spread_flux,
)
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, cell_areas
from run_city_source_transport import grid_city_indices


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/multicity-propagation-20261126'
CITY = {
    'Tucson': (
        32.2226,
        -110.9747,
        Bounds(-111.75, 31.6, -110.15, 32.85),
        700,
        ('2019-03-29', '2019-03-31', '2019-04-03', '2019-04-07', '2019-04-08'),
    ),
    'Madrid': (
        40.4169,
        -3.7035,
        Bounds(-4.5, 39.8, -2.9, 41.05),
        500,
        ('2013-12-04', '2013-12-29'),
    ),
}
LABELS = (
    'city_core',
    'halo_18_to_43_km',
    'dark_control_46_to_65_km',
    'bright_neighbor_sources',
)
OPTIONS = {
    'direct': (0, 2, 3),
    'all_lambert': (0, 2, 3),
    'single_mixture': (0, 1, 2, 3),
    'all_mixture': (0, 1, 2, 3),
}


def source_template(directory, bounds, night, geometry, city):
    month = night.strftime('%Y-%m')
    fields, tiles = read_region(
        directory / f'{city.lower()}-{month}', bounds, night.year, night.month
    )
    near = 'NearNadir_Composite_Snow_Free'
    valid = good_mask(
        fields[near], fields[near + '_Quality'], fields[near + '_Num']
    ) & np.isin(fields['Land_Water_Mask'], [0, 1, 5])
    off = 'OffNadir_Composite_Snow_Free'
    paired = valid & good_mask(
        fields[off], fields[off + '_Quality'], fields[off + '_Num']
    )
    radiance = np.where(valid, fields[near], 0.0)
    city_mask = grid_city_indices(bounds, {city: geometry[city]}, (city,)) == 1
    return (
        radiance,
        valid,
        city_mask,
        paired,
        np.where(paired, fields[off], 0.0),
        {v['filename']: v['sha256'] for v in tiles.values()},
    )


def region_masks(radiance, mapped, city_mask, bounds, lat, lon):
    metres_per_degree = EARTH_RADIUS_M * np.pi / 180
    north = (bounds.latitude[:, None] - lat) * metres_per_degree / 1000
    east = (
        (bounds.longitude[None, :] - lon)
        * metres_per_degree
        * np.cos(np.deg2rad(lat))
        / 1000
    )
    r = np.hypot(north, east)
    # Use the monthly field to screen strong direct lights, including small towns.
    luminous = (radiance >= 2.0).astype(float)
    close_to_lights = spread_flux(luminous, np.ones((11, 11))) > 0.5
    weak = mapped & ~city_mask & ~close_to_lights & (radiance < 0.85)
    core = city_mask & (r <= 18) & (radiance >= 2.0)
    halo = weak & (18 < r) & (r <= 43)
    background = weak & (46 <= r) & (r <= 65)
    neighbors = ~city_mask & (r <= 60) & (radiance >= 2.0)
    return (core, halo, background, neighbors), north, east, weak


def preflight(
    directory,
    city,
    night,
    geometry,
    maps,
    *,
    location=None,
    minimum_class_cells=160,
    compare_a2=True,
    a1_product='VNP46A1',
):
    lat, lon, bounds, ref = CITY[city][:4] if location is None else location
    a1, manifest = read_daily_region(directory, night, bounds, a1_product)
    corrected = np.full(bounds.shape, np.nan)
    if compare_a2:
        a2, more = read_daily_region(directory, night, bounds, a1_product[:-1] + '2')
        manifest.extend(more)
        good = (a2['Mandatory_Quality_Flag'] == 0) & (a2['Snow_Flag'] == 0)
        corrected = np.where(
            good & (a2['DNB_BRDF-Corrected_NTL'] >= 0),
            a2['DNB_BRDF-Corrected_NTL'],
            np.nan,
        )
    radiance, mapped, city_mask, paired, off, map_hash = source_template(
        maps, bounds, night, geometry, city
    )
    masks, north, east, weak = region_masks(
        radiance, mapped, city_mask, bounds, lat, lon
    )
    passes = a1['UTC_Time'][masks[0] & np.isfinite(a1['UTC_Time'])]
    if not len(passes):
        raise ValueError(f'No overpass crosses the urban support on {night} {city}.')
    hour = float(np.median(passes))
    pass_mask = np.abs(a1['UTC_Time'] - hour) <= 0.05
    ok = clear_dark_a1(a1) & pass_mask
    geometry_mask = pass_mask & masks[0]
    zenith = float(np.nanmedian(a1['Sensor_Zenith'][geometry_mask]))
    radians = np.deg2rad(a1['Sensor_Azimuth'][geometry_mask])
    azimuth = float(
        np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    )
    raw = a1['DNB_At_Sensor_Radiance']
    groups = {}
    for name, region in zip(LABELS, masks, strict=True):
        valid = ok & region
        both = valid & np.isfinite(corrected)
        groups[name] = {
            'candidate_cells': int((region & pass_mask).sum()),
            'a1_cells': int(valid.sum()),
            'paired_a1_a2_cells': int(both.sum()),
            'a1_median': float(np.median(raw[both])) if both.any() else None,
            'a2_median': float(np.median(corrected[both])) if both.any() else None,
            'a1_minus_a2_median': float(np.median((raw - corrected)[both]))
            if both.any()
            else None,
        }
    instant = datetime.combine(
        night, datetime.min.time(), tzinfo=timezone.utc
    ) + timedelta(hours=hour)
    info = {
        'city': city,
        'night': night.isoformat(),
        'bounds': asdict(bounds),
        'source_altitude_m': ref,
        'utc': instant.isoformat(),
        'zenith_deg': zenith,
        'azimuth_deg': azimuth,
        'lunar_altitude_core_deg': float(
            90 - np.nanmedian(a1['Lunar_Zenith'][geometry_mask])
        ),
        'solar_altitude_core_deg': float(
            90 - np.nanmedian(a1['Solar_Zenith'][geometry_mask])
        ),
        'source_template_files': map_hash,
        'files': manifest,
        'support': groups,
        'sensor_zenith_05_95_deg': np.percentile(
            a1['Sensor_Zenith'][ok], [5, 95]
        ).tolist()
        if ok.any()
        else None,
    }
    if min(v['a1_cells'] for v in groups.values()) < minimum_class_cells:
        return info, None
    local = {
        'bounds': bounds,
        'lat': lat,
        'raw': raw,
        'corrected': corrected,
        'valid': ok,
        'masks': masks,
        'north': north,
        'east': east,
        'near_radiance': radiance,
        'city_mask': city_mask,
        'weak_surface': weak,
        'sensor_zenith': a1['Sensor_Zenith'],
        'city_flux': np.where(city_mask, np.pi * radiance, 0.0),
        'paired_city_near': np.where(city_mask & paired, radiance, 0.0),
        'paired_city_off': np.where(city_mask & paired, off, 0.0),
        'neighbor_flux': np.where(
            ~city_mask & (radiance >= 2.0), np.pi * radiance, 0.0
        ),
    }
    return info, local


def block_measurements(data, features):
    """Areas are grouped in 4×4 native cells. Keep blocks with ≥8 eligible cells."""
    raw, valid = data['raw'], data['valid']
    n, m = (s // 4 for s in raw.shape)

    def blocks(a):
        return a[: 4 * n, : 4 * m].reshape(n, 4, m, 4)

    result = []
    for idx, region in enumerate(data['masks']):
        support = region & valid
        count = blocks(support).sum(axis=(1, 3))
        ii, jj = np.where(count >= 8)
        divisor = count[ii, jj]
        avg = lambda a: (
            blocks(np.where(support, a, 0.0)).sum(axis=(1, 3))[ii, jj] / divisor
        )
        result.append(
            (
                avg(raw),
                np.column_stack([avg(a) for a in features]),
                ii,
                jj,
                np.full(len(ii), idx),
            )
        )
    return [
        np.concatenate([m[i] for m in result], axis=0) for i in range(len(result[0]))
    ]


def scores(observed, predicted, use, classes):
    result = {}
    for i, name in enumerate(LABELS):
        mask = use & (classes == i)
        residual = (predicted - observed)[mask]
        result[name] = {
            'n': int(mask.sum()),
            'mean_observed': float(observed[mask].mean()),
            'mean_predicted': float(predicted[mask].mean()),
            'bias': float(residual.mean()),
            'mae': float(np.mean(np.abs(residual))),
        }
    return result


def atmosphere_for(info, atmosphere_audit, baseline, aod_factor):
    if info['city'] == 'Tucson':
        case = atmosphere_audit['aod_recalculations']['atmospheres'][
            info['night'] + '/daylight-aeronet'
        ]
    else:
        case = baseline['atmospheres'][
            'madrid-dec3' if info['night'] == '2013-12-04' else 'madrid-dec29'
        ]
    a = Atmosphere(
        **case['parameters'],
        rayleigh_multiplier=np.exp(-info['source_altitude_m'] / 8000),
    )
    return replace(
        a,
        reference_wavelength_nm=650.0,
        aerosol_optical_depth=aod_factor
        * a.aerosol_optical_depth
        * (550 / 650) ** case['angstrom'],
    )


def project_kernel(data, kernel, method):
    city, neighbor = data['city_flux'], data['neighbor_flux']
    lam = city * kernel.direct[0]
    low = city * kernel.direct[1]
    outside = neighbor * kernel.direct[0]
    if method != 'direct':
        spread = (
            kernel.single
            if method == 'single_mixture'
            else kernel.single + kernel.multiple
        )
        lam += np.maximum(0.0, spread_flux(city, spread[0]))
        low += np.maximum(0.0, spread_flux(city, spread[1]))
        outside += np.maximum(0.0, spread_flux(neighbor, spread[0]))
    return lam, low, outside, np.ones_like(city)


def evaluate(info, data, atmosphere, samples, seed, out):
    # The kernel is spherical at the reference location and translated on the local grid.
    dy = EARTH_RADIUS_M * np.pi / (180 * PIXELS_PER_DEGREE)
    dx = dy * np.cos(np.deg2rad(data['lat']))
    settings = AllOrderSettings(
        samples=samples, batch_size=32768, seed=seed, recorded_orders=1
    )
    kernel = build_halo_kernel(
        info['zenith_deg'], info['azimuth_deg'], dx, dy, 150_000, atmosphere, settings
    )
    scenarios = {}
    flux_w = float(
        ((data['city_flux'] * cell_areas(data['bounds'].latitude)[:, None]).sum())
        * 1e-5
    )
    for name, active in OPTIONS.items():
        features = project_kernel(data, kernel, name)
        y, x, ii, jj, c = block_measurements(data, features)
        # Hold out whole ∼7 km blocks, with the same split for all models.
        test = ((ii // 4) + 2 * (jj // 4)) % 3 == 0
        if any(
            min(((c == i) & test).sum(), ((c == i) & ~test).sum()) < 6
            for i in range(len(LABELS))
        ):
            raise ValueError(
                f'Insufficient blocked validation coverage for {info["city"]} {info["night"]}.'
            )
        weight = np.ones(len(y))
        for k in range(len(LABELS)):
            sample = y[(c == k) & ~test]
            weight[c == k] = 1 / (
                max(0.10, float(np.median(sample))) * np.sqrt(len(sample))
            )
        small, loss = fit_nonnegative(x[~test][:, active], y[~test], weight[~test])
        coef = np.zeros(4)
        coef[list(active)] = small
        f = np.asarray(coef[:2])
        area = f.sum()
        weighted = x[~test][:, active] * weight[~test, None]
        norms = np.linalg.norm(weighted, axis=0)
        nonzero = norms > 0
        condition = float(np.linalg.cond(weighted[:, nonzero] / norms[nonzero]))
        scenarios[name] = {
            'coefficients': coef.tolist(),
            'below_10deg_fraction': float(f @ flux_below(10) / area) if area else None,
            'equivalent_upward_DNB_W': float(area * flux_w),
            'weighted_training_loss': loss,
            'condition': condition if np.isfinite(condition) else None,
            'test': scores(y, x @ coef, test, c),
            'train': scores(y, x @ coef, ~test, c),
        }
        if name == 'all_mixture':
            pred = sum(v * d for v, d in zip(coef, features, strict=True))
            precisely_once = sum(
                coef[i] * np.maximum(0.0, spread_flux(flux, kernel.single[shape]))
                for i, flux, shape in (
                    (0, data['city_flux'], 0),
                    (1, data['city_flux'], 1),
                    (2, data['neighbor_flux'], 0),
                )
            )
            more_orders = sum(
                coef[i] * np.maximum(0.0, spread_flux(flux, kernel.multiple[shape]))
                for i, flux, shape in (
                    (0, data['city_flux'], 0),
                    (1, data['city_flux'], 1),
                    (2, data['neighbor_flux'], 0),
                )
            )
            halo = data['valid'] & data['masks'][1]
            denom = precisely_once[halo] + more_orders[halo]
            info['halo_scatter_medians'] = {
                'single': float(np.median(precisely_once[halo])),
                'multiple': float(np.median(more_orders[halo])),
                'multiple_share_of_scatter_mean_signal': float(
                    more_orders[halo].sum() / denom.sum()
                )
                if denom.sum()
                else None,
            }
            np.savez_compressed(
                out / f'{info["city"].lower()}-{info["night"]}.npz',
                observed=np.where(data['valid'], data['raw'], np.nan),
                corrected=np.where(data['valid'], data['corrected'], np.nan),
                predicted=pred,
                scattered=precisely_once + more_orders,
                multiple=more_orders,
                city_flux=data['city_flux'],
                latitude=data['bounds'].latitude,
                longitude=data['bounds'].longitude,
                x=x,
                y=y,
                test=test,
                classes=c,
                north=data['north'],
                east=data['east'],
            )
    info.update(
        atmosphere=asdict(atmosphere),
        samples=samples,
        seed=seed,
        source_1x_equivalent_DNB_W=flux_w,
        kernel_ground_hits=list(kernel.hits),
        kernel_outside_window=kernel.outside_window,
        scenarios=scenarios,
    )
    print(
        info['city'],
        info['night'],
        'fit',
        scenarios['all_mixture']['coefficients'],
        flush=True,
    )
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('daily', 'monthly', 'boundaries', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--samples', type=int, default=1_048_576)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=18115)
    parser.add_argument('--aod-factor', type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(args.aod_factor) or args.aod_factor <= 0:
        parser.error('--aod-factor must be positive and finite')
    boundary = read_json(args.boundaries)
    atm = read_json(DATA / 'weather-audit.json')
    baseline = read_json(DATA / 'results.json')
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {}
        for city, (*_, dates) in CITY.items():
            for d in dates:
                info, data = preflight(
                    args.daily, city, date.fromisoformat(d), boundary, args.monthly
                )
                key = f'{city}/{d}'
                if data is None:
                    results[key] = info | {'status': 'excluded_by_clear_dark_support'}
                    print(key, 'excluded', flush=True)
                    continue
                atmosphere = atmosphere_for(info, atm, baseline, args.aod_factor)
                jobs[key] = pool.submit(
                    evaluate,
                    info,
                    data,
                    atmosphere,
                    args.samples,
                    args.seed,
                    args.output,
                )
        for key, job in jobs.items():
            results[key] = job.result() | {'status': 'evaluated'}

    # Two low-power nights from the Tucson experiment, with opposite viewing directions.
    transfer = {}
    a, b = '2019-03-31', '2019-04-03'
    for train, target in ((a, b), (b, a)):
        if any(
            results['Tucson/' + d]['status'] != 'evaluated' for d in (train, target)
        ):
            continue
        with np.load(args.output / f'tucson-{target}.npz') as ar:
            x, y, test, c = (ar[n] for n in ('x', 'y', 'test', 'classes'))
            coefficients = np.array(
                results['Tucson/' + train]['scenarios']['all_mixture']['coefficients']
            )
            artificial = x[:, :3] @ coefficients[:3]
            background = max(0.0, float(np.median((y - artificial)[~test & (c == 2)])))
            pred = artificial + background
            transfer[f'{train} → {target}'] = {
                'background_from_target_dark_blocks': background,
                'test': scores(y, pred, test, c),
            }
    write_json(
        args.output / 'results.json',
        {
            'scenes': results,
            'held_out_dates': transfer,
            'city_boundaries_sha256': file_hash(args.boundaries),
            'aod_factor': args.aod_factor,
            'lambert_and_low_angle_fraction_below_10': flux_below(10).tolist(),
            'validation_policy': 'Spatially blocked; date transfer freezes all artificial components and updates only dark-block background.',
            'version': 'Research prototype grey atmosphere, no directional fit from A2.',
        },
    )


if __name__ == '__main__':
    main()
