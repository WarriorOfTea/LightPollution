"""Replay four ground transects with city light estimated only from satellite halos."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.data import Bounds
from lightpollution.models.all_orders import AllOrderSettings
from lightpollution.models.boundary_orders import integrate_boundary_variants
from lightpollution.models.satellite_halo import LOW_ANGLE_COSINE, emission_radiance
from lightpollution.models.spectra import wavelength_atmosphere
from lightpollution.physics import Atmosphere, Observer
from run_berlin_propagation import BerlinBoundary, FORMS, map_layers, pair_class
from run_city_source_transport import grid_city_indices
from run_viirs_halo_recovery import DATA, ROOT


BERLIN = ROOT / 'artifacts/berlin-propagation-20261126/results.json'
REFERENCE = DATA / 'results.json'
INPUTS = DATA / 'observations.json'
NEW = ('a1_lambert', 'a1_free', 'a1_shape_only', 'a1_anchored')
POLICY = {
    'Madrid/2012-02-19': '2012-02-20',
    'Madrid/2013-12-03': '2013-12-04',
    'Madrid/2013-12-28': '2013-12-29',
    'Balaguer/2016-05-03': '2016-05-04',
}
LAM, HELD, ENHANCED, LOW = 1, 3, 4, 8


class LowAngleBoundary(BerlinBoundary):
    labels = (*BerlinBoundary.labels, 'city_low_angle_exponential')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        low_limit = np.pi * emission_radiance(0.0)[1] * self.upper_bounds[LAM]
        self.upper_bounds = np.r_[self.upper_bounds, low_limit]

    def sample(self, points, outgoing):
        original, support = super().sample(points, outgoing)
        mu = np.einsum('ij,ij->i', points, outgoing) / np.linalg.norm(points, axis=1)
        low = np.pi * emission_radiance(mu)[:, 1] * original[:, LAM]
        return np.column_stack((original, low)), support


def select_scene(city, target, scenes):
    """Use the ground night or the closest qualified date in the same month."""
    target_day = date.fromisoformat(target)
    max_separation = 8 if city == 'Balaguer' else 2
    candidates = []
    for key, value in scenes.items():
        if (
            value['city'] != city
            or value['status'] != 'evaluated'
            or 'a1_anchored' not in value['scenarios']
        ):
            continue
        day = date.fromisoformat(value['night'])
        if (day.year, day.month) != (target_day.year, target_day.month):
            continue
        separation = abs((day - target_day).days)
        if separation <= max_separation:
            candidates.append((separation, value['night'], key))
    if not candidates:
        return None, None
    separation, _, key = min(candidates)
    return key, separation


def city_weights(scene):
    weights = {k: np.r_[1.0, v, np.zeros(4)] for k, v in FORMS.items()}
    if scene is None:
        return weights
    variants = scene['scenarios']
    free = np.asarray(variants['a1_free']['coefficients'][:2])
    low, high = np.cos(np.deg2rad(20.0)), 1.0
    # Match the radiance average in the observed NearNadir solid-angle band.
    at_zero = emission_radiance(0.0)[1]
    near_exponential = (
        np.pi
        * at_zero
        * LOW_ANGLE_COSINE
        * (np.exp(-low / LOW_ANGLE_COSINE) - np.exp(-high / LOW_ANGLE_COSINE))
        / (high - low)
    )
    near_scale = float(free @ np.array([1.0, near_exponential]))
    kappa = variants['a1_anchored']['kappa']
    for name in NEW:
        weights[name] = np.zeros(9)
        weights[name][0] = 1.0
    weights['a1_lambert'][LAM] = variants['a1_lambert']['coefficients'][0]
    weights['a1_free'][[LAM, LOW]] = free
    if near_scale > 0:
        weights['a1_shape_only'][[LAM, LOW]] = free / near_scale
    else:
        del weights['a1_shape_only']
    weights['a1_anchored'][[HELD, ENHANCED]] = [1 - kappa, kappa]
    return weights


# The workers inherit a fixed city source map for one period.
_BOUNDARY = None


def setup_boundary(value):
    global _BOUNDARY
    _BOUNDARY = value


def replay(job, *, include_orders=False):
    lat, lon, height, aperture, reference, case, samples, bands, old = job
    observer = Observer(lat, lon, height)
    base = Atmosphere(
        **case['parameters'], rayleigh_multiplier=float(np.exp(-reference / 8000))
    )
    _BOUNDARY.at(observer)
    new, covariances, by_order = [], [], []
    for band, previous in zip(bands, old, strict=True):
        wave = band['wavelength_nm']
        model = wavelength_atmosphere(base, wave, case['angstrom'])
        if 'phase_nodes' in case:
            g = float(np.interp(wave, *case['phase_nodes']))
            model = replace(model, aerosol_g=g)
        computed = integrate_boundary_variants(
            observer,
            _BOUNDARY,
            0.0,
            90.0,
            model,
            AllOrderSettings(samples=samples, seed=previous['seed'], recorded_orders=1),
            aperture_radius_deg=aperture,
        )
        means = np.array(computed['all_order_radiance'])
        np.testing.assert_allclose(
            means[:LOW], previous['all_order_radiance'], rtol=3e-11, atol=3e-12
        )
        new.append(means)
        covariances.append(computed['mean_covariance'])
        if include_orders:
            by_order.append(computed['category_mean_radiance'])
    if include_orders:
        return np.array(new), np.array(covariances), np.array(by_order)
    return np.array(new), np.array(covariances)


def collect_tasks(berlin, reference, inputs):
    groups = defaultdict(list)
    tasks = {}
    for row in berlin['rows']:
        key = f'berlin-2017-03/{row["stop"]}'
        old = row['atmospheres']['aeronet']['spectral_radiance_runs']
        bands = berlin['spectral_bands']
        case = {
            'parameters': {'aerosol_optical_depth': berlin['aod_550']['aeronet']},
            'angstrom': berlin['angstrom_exponent'],
        }
        tasks[key] = (
            row['latitude'],
            row['longitude'],
            row['observer_height_m'],
            berlin['aperture_radius_deg'],
            0.0,
            case,
            berlin['samples_per_run'],
            bands,
            old,
        )
        groups['berlin-2017-03'].append(
            {
                'key': key,
                'city': 'Berlin',
                'night': '2017-03-29',
                'site': row['site'],
                'distance_km': row['distance_km'],
                'band': 'photopic',
                'utc_interpretations': [row['utc']],
                'observed': row['luminance_mcd_m2'],
                'natural': berlin['mcd_m2_natural'],
                'zero': 1.0,
                'spectral_weights': [
                    b['mcd_m2_per_dnb_unit']['equal_dnb'] for b in bands
                ],
                'baseline': row['prediction']['aeronet']['equal_dnb'],
            }
        )
    for row in reference['rows']:
        key = row['run']
        period, atm, lat, lon, elevation, aperture = reference['runs'][key]['geometry']
        base_height = inputs['cities'][row['city']]['source_reference_m']
        bands = reference['spectral_bands']
        tasks[key] = (
            lat,
            lon,
            elevation - base_height,
            aperture,
            base_height,
            inputs['atmospheres'][atm],
            reference['samples_per_band'],
            bands,
            reference['runs'][key]['bands'],
        )
        band = row['band']
        calibration = reference['calibration'][band]
        groups[period].append(
            {
                'key': key,
                'city': row['city'],
                'night': row['night'],
                'site': row['site'],
                'distance_km': row['distance_km'],
                'band': band,
                'utc_interpretations': row.get('utc_interpretations'),
                'observed': row['observed_W_m2_sr'],
                'natural': 0.17 * calibration['natural_per_mcd_m2_W_m2_sr'],
                'zero': calibration['vega_zero_W_m2_sr'],
                'spectral_weights': [
                    b['W_m2_sr_per_dnb'][band]['equal_dnb'] for b in bands
                ],
                'baseline': row['prediction'],
            }
        )
    return groups, tasks


def project(row, means, covariance, scenes):
    city, night, band = (row[k] for k in ('city', 'night', 'band'))
    target = POLICY.get(city + '/' + night, night)
    scene_id, date_distance = select_scene(city, target, scenes)
    scene = scenes[scene_id] if scene_id else None
    weights = np.array(row['spectral_weights'])
    terms = weights @ means
    band_covariance = np.tensordot(weights**2, covariance, axes=1)
    observed = row['observed']
    predictions = {}
    for model, components in city_weights(scene).items():
        artificial = float(components @ terms)
        total = artificial + row['natural']
        if model in row['baseline']:
            k = 'total_mcd_m2' if city == 'Berlin' else 'total_W_m2_sr'
            np.testing.assert_allclose(total, row['baseline'][model][k], rtol=3e-11)
        absolute = (
            total
            if band == 'photopic' and city == 'Berlin'
            else (
                total * 683000
                if band == 'photopic'
                else -2.5 * np.log10(total / row['zero'])
            )
        )
        nominal_error = 100 * (total / observed - 1)
        se = (
            100
            * float(np.sqrt(max(0.0, components @ band_covariance @ components)))
            / observed
        )
        predictions[model] = {
            'display_value': float(absolute),
            'relative_pct': nominal_error,
            'linear': total,
            'monte_carlo_se_pct': se,
        }
    offsets = None
    if scene_id and row['utc_interpretations']:
        a = datetime.fromisoformat(scene['utc'])
        offsets = [
            (a - datetime.fromisoformat(s)).total_seconds() / 3600
            for s in row['utc_interpretations']
        ]
    result = {
        k: row[k]
        for k in (
            'key',
            'city',
            'night',
            'site',
            'band',
            'distance_km',
            'observed',
            'natural',
        )
    }
    result['city_dominant'] = all(
        row['baseline'][m]['city_fraction'] > 0.5
        for m in ('lambert', 'off_h1', 'off_h2')
    )
    result['observed_display'] = (
        observed
        if city == 'Berlin'
        else (
            observed * 683000
            if band == 'photopic'
            else float(-2.5 * np.log10(observed / row['zero']))
        )
    )
    return result | {
        'fitted_scene': scene_id,
        'calendar_offset_days': date_distance,
        'utc_offset_hours': offsets,
        'test_scene_requested': target,
        'prediction': predictions,
    }


def metrics(rows, modes):
    result = {}
    for model in modes:
        available = [r for r in rows if model in r['prediction']]
        if not available:
            continue
        errors = np.array([r['prediction'][model]['relative_pct'] for r in available])
        result[model] = {
            'n': len(available),
            'mae_pct': float(abs(errors).mean()),
            'bias_pct': float(errors.mean()),
        }
    return result


def compare_pairs(rows):
    values = {mode: {'n': 0, 'correct': 0} for mode in (*FORMS, *NEW)}
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if (a['city'], a['night'], a['band']) != (
                b['city'],
                b['night'],
                b['band'],
            ) or a['site'] == b['site']:
                continue
            sign = pair_class(a['observed'], b['observed'], 0.10)
            if not sign:
                continue
            for mode in values.keys() & a['prediction'].keys() & b['prediction'].keys():
                x, y = a['prediction'][mode]['linear'], b['prediction'][mode]['linear']
                values[mode]['n'] += 1
                values[mode]['correct'] += int(pair_class(x, y, 0.10) == sign)
    return values


def period_boundary(
    period, city, maps, geometry, berlin, reference, inputs, boundary_type
):
    year, month = map(int, period.split('-')[-2:])
    bounds = (
        Bounds(**berlin['bounds'])
        if city == 'Berlin'
        else Bounds(*inputs['cities'][city]['bounds'])
    )
    raster, sha1 = map_layers(maps, period, month, bounds=bounds, year=year)
    annual, sha2 = map_layers(
        maps, f'{city.lower()}-{year}', None, bounds=bounds, year=year
    )
    recorded = (
        berlin['tiles_sha256']
        if city == 'Berlin'
        else reference['map_diagnostics'][period]['tile_sha256']
    )
    if (sha1 | sha2) != recorded:
        raise ValueError(f'Map hash mismatch for {period}.')
    city_mask = grid_city_indices(bounds, {city: geometry[city]}, (city,)) == 1
    return boundary_type(raster, annual, city_mask, bounds=bounds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('maps', 'boundaries', 'satellite', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometry = read_json(args.boundaries)
    satellite = read_json(args.satellite)
    if file_hash(args.boundaries) != satellite['boundaries_sha256']:
        raise ValueError('Satellite and ground city boundaries must match.')
    berlin, reference, inputs = (
        read_json(path) for path in (BERLIN, REFERENCE, INPUTS)
    )
    groups, tasks = collect_tasks(berlin, reference, inputs)
    rows = []
    for period, readings in groups.items():
        city = readings[0]['city']
        boundary = period_boundary(
            period,
            city,
            args.maps,
            geometry,
            berlin,
            reference,
            inputs,
            LowAngleBoundary,
        )
        keys = list(dict.fromkeys(row['key'] for row in readings))
        computed = {}
        with ProcessPoolExecutor(
            args.workers, initializer=setup_boundary, initargs=(boundary,)
        ) as p:
            futures = {p.submit(replay, tasks[k]): k for k in keys}
            for completed in as_completed(futures):
                computed[futures[completed]] = completed.result()
        del boundary
        rows.extend(
            project(row, *computed[row['key']], satellite['scenes']) for row in readings
        )
        print(
            period,
            len(readings),
            'measurements replayed through',
            len(keys),
            'fields of view',
            flush=True,
        )
    groups = {}
    for city in ('Berlin', 'Madrid', 'Tucson', 'Balaguer', 'All'):
        group = [r for r in rows if city == 'All' or city == r['city']]
        both = [r for r in group if all(k in r['prediction'] for k in NEW)]
        common = {
            'all': both,
            'same_night': [r for r in both if r['calendar_offset_days'] == 0],
            'transferred': [r for r in both if r['calendar_offset_days'] != 0],
            'city_dominant': [r for r in both if r['city_dominant']],
        }
        groups[city] = {
            'ground_n': len(group),
            'paired_n': len(both),
            'baseline_all': metrics(group, FORMS),
            'paired': {k: metrics(v, (*FORMS, *NEW)) for k, v in common.items()},
            'brightness_order': compare_pairs(both),
        }
    paths = (BERLIN, REFERENCE, INPUTS, args.boundaries, args.satellite)
    write_json(
        args.output / 'results.json',
        {
            'observations': len(rows),
            'ground_paths': sum(t[6] * len(t[7]) for t in tasks.values()),
            'ground_spectral_integrations': sum(len(t[7]) for t in tasks.values()),
            'provenance': {p.name: file_hash(p) for p in paths},
            'rows': rows,
            'metrics': groups,
        },
    )


if __name__ == '__main__':
    main()
