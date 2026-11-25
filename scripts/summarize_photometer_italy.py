"""Score the Italian all-order runs against independent held-out SQM windows."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from run_photometer_italy import BACKGROUND_MAG

VARIANTS = ('base', 'spatial', 'regional')
WEATHER_LIMIT = {'cloud_cover': 15, 'precipitation': 0, 'relative_humidity_2m': 95,
                 'era5_land_snow_depth': .01}


def weather_summary(points, site):
    maxima, missing = {}, {}
    for field in WEATHER_LIMIT:
        locations = (points[site],) if field == 'relative_humidity_2m' else points.values()
        values = np.array([point[field] for point in locations], dtype=float)
        if values.shape != (len(locations), 2):
            raise ValueError(f'Expected two endpoint values for {field}.')
        valid = np.isfinite(values) & (values >= 0)
        missing[field] = int(values.size - valid.sum())
        maxima[field] = float(values[valid].max()) if valid.any() else None
    clear = (not any(missing.values())
             and maxima['cloud_cover'] <= WEATHER_LIMIT['cloud_cover']
             and maxima['precipitation'] == WEATHER_LIMIT['precipitation']
             and maxima['relative_humidity_2m'] <= WEATHER_LIMIT['relative_humidity_2m']
             and maxima['era5_land_snow_depth'] < WEATHER_LIMIT['era5_land_snow_depth'])
    return {'passes': clear, 'maxima': maxima, 'missing': missing,
            'era5_cells': len({tuple(p['returned_latlon']) for p in points.values()}),
            'era5_land_cells': len({tuple(p['snow_latlon']) for p in points.values()})}


def signed_errors(rows, training, run_values, mask, natural):
    scales = []
    for index in training:
        anchor = rows[index]
        artificial = anchor['observed_relative'] - natural
        calibration_radiance = run_values[anchor['runs']['base']][..., mask]
        if np.any(calibration_radiance <= 0) or artificial <= 0:
            raise ValueError('The common calibration must have a positive artificial signal.')
        scales.append(artificial / calibration_radiance)
    scale = np.median(scales, axis=0)
    residuals = []
    for row in rows:
        observed = row['observed_relative']
        radiance = np.stack([run_values[row['runs'][v]][..., mask] for v in VARIANTS], axis=-1)
        predicted = natural + scale[..., None] * radiance
        residuals.append(100 * (predicted / observed - 1))
    return np.stack(residuals), scale


def aggregate(actual, draws, indices):
    if not indices:
        return None
    estimates = actual[indices]
    score_draws = np.abs(draws[indices]).mean(axis=0)
    mape = np.abs(estimates).mean(axis=0)
    intervals = np.quantile(score_draws, [.025, .975], axis=0).T
    contrast_draws = score_draws[:, 1:] - score_draws[:, :1]
    spatial_minus_regional = score_draws[:, 1] - score_draws[:, 2]
    contrasts = {'spatial_minus_base': (mape[1] - mape[0], contrast_draws[:, 0]),
                 'regional_minus_base': (mape[2] - mape[0], contrast_draws[:, 1]),
                 'spatial_minus_regional': (mape[1] - mape[2], spatial_minus_regional)}
    return {'n': len(indices),
            'mape_pct': dict(zip(VARIANTS, mape.tolist(), strict=True)),
            'signed_mean_pct': dict(zip(VARIANTS, estimates.mean(axis=0).tolist(), strict=True)),
            'median_abs_pct': dict(zip(VARIANTS, np.median(abs(estimates), axis=0).tolist(), strict=True)),
            'mc_95_pct': dict(zip(VARIANTS, intervals.tolist(), strict=True)),
            'mape_differences_pp': {name: {'value': float(value),
                                           'mc_95': np.quantile(sample, [.025, .975]).tolist()}
                                    for name, (value, sample) in contrasts.items()}}


def site_summary(doc, weather, distance_reference=None):
    runs = doc['runs']
    seeds = [r['settings']['seed'] for r in runs.values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Distinct transport runs must have independent seeds.')
    if any(run['finite_scattering_order_truncation'] for run in runs.values()):
        raise ValueError('Every result must include all scattering orders.')
    rows = []
    for window in doc['rows']:
        quality = weather_summary(weather[window['date']], doc['site'])
        base = runs[window['runs']['base']]
        unmasked, beyond_3km, beyond_10km, _ = base['all_order_radiance']
        near_3 = 100 * (1 - beyond_3km / unmasked)
        near_3_10 = 100 * (1 - beyond_10km / beyond_3km)
        near_10 = 100 * (1 - beyond_10km / unmasked)
        rows.append({**window, 'weather': quality, 'artificial_from_lt3km_pct': near_3,
                     'artificial_from_3_to_10km_pct_of_outside_3km': near_3_10,
                     'artificial_from_lt10km_pct': near_10,
                     'passes_distance_check': near_3 <= 20 and near_10 < 50})
    clear = [i for i, row in enumerate(rows) if row['weather']['passes']]
    if len(clear) < 2:
        raise ValueError('Expected a calibration and at least one weather-screened night.')
    calibration, *controls = clear
    natural = rows[calibration]['natural_relative']
    if not np.allclose([row['natural_relative'] for row in rows], natural, rtol=1e-12, atol=0):
        raise ValueError('The common zenith background must be fixed.')
    reference_month = rows[calibration]['date'][:7]

    rng = np.random.default_rng(10495)
    means = {name: np.asarray(run['all_order_radiance']) for name, run in runs.items()}
    draws = {name: rng.multivariate_normal(means[name], run['mean_covariance'], size=16_384)
             for name, run in runs.items()}
    summaries, results = {}, {}
    for label, mask, level in (('monthly', 1, natural), ('fixed_map', 3, natural),
                               ('exclude_10km', 2, natural), ('all_distances', 0, natural),
                               ('brighter_background', 1, 10**(-.4*(21.3 - 22))),
                               ('darker_background', 1, 10**(-.4*(21.9 - 22)))):
        errors, scale = signed_errors(rows, [calibration], means, mask, level)
        errors_mc, _ = signed_errors(rows, [calibration], draws, mask, level)
        summaries[label] = aggregate(errors, errors_mc, controls)
        summaries[label]['common_scale'] = float(scale)
        results[label] = (errors, errors_mc)

    errors, errors_mc = results['monthly']
    for row, signed, sampled in zip(rows, errors, errors_mc, strict=True):
        row['signed_error_pct'] = dict(zip(VARIANTS, signed.tolist(), strict=True))
        row['mc_se_pp'] = dict(zip(VARIANTS, sampled.std(axis=0, ddof=1).tolist(), strict=True))
        row['natural_fraction_pct'] = 100 * natural / row['observed_relative']
        row['beyond_first_order_pct'] = {}
        for variant, name in row['runs'].items():
            case = runs[name]
            radiant = case['all_order_radiance'][1]
            row['beyond_first_order_pct'][variant] = 100 * (
                1 - case['category_mean_radiance'][0][1] / radiant)
    by_month = defaultdict(list)
    for i in controls:
        by_month[rows[i]['date'][:7]].append(i)
    folds = defaultdict(list)
    for index in clear:
        folds[rows[index]['date'][:7]].append(index)
    leave_month_out = None
    if len(folds) > 1:
        cv_errors, cv_draws = errors.copy(), errors_mc.copy()
        for held_out in folds.values():
            training = [i for i in clear if i not in held_out]
            predictions, _ = signed_errors(rows, training, means, 1, natural)
            sampled, _ = signed_errors(rows, training, draws, 1, natural)
            cv_errors[held_out], cv_draws[held_out] = predictions[held_out], sampled[held_out]
        leave_month_out = aggregate(cv_errors, cv_draws, clear)
    if distance_reference is None:
        distance_reference = {row['date'] for row in rows if row['passes_distance_check']}
    test_distance = [i for i in controls if rows[i]['date'] in distance_reference]
    test_all = [i for i in range(len(rows)) if i != calibration]
    across_months = [i for key, indices in by_month.items() if key != reference_month for i in indices]
    return {'site': doc['site'], 'calibration_date': rows[calibration]['date'],
            'calibration_passes_distance_check': rows[calibration]['passes_distance_check'],
            'n_windows': len(rows), 'n_readings': sum(r['readings'] for r in rows),
            'clear_windows': len(clear), 'clear_readings': sum(rows[i]['readings'] for i in clear),
            'n_transport_runs': len(runs),
            'trajectories': sum(r['settings']['samples'] for r in runs.values()),
            'max_scattering_order_seen': max(r['max_scattering_order_seen'] for r in runs.values()),
            'minimum_effective_samples_main_mask': min(r['effective_score_samples'][1]
                                                       for r in runs.values()),
            'background_mag': BACKGROUND_MAG, 'reference_map_month': doc['reference_map_month'],
            'summaries': summaries,
            'monthly_groups': {key: aggregate(errors, errors_mc, indices)
                               for key, indices in sorted(by_month.items())},
            'held_out_months': aggregate(errors, errors_mc, across_months),
            'leave_one_month_out_common_median_fit': leave_month_out,
            'clear_and_distance': aggregate(errors, errors_mc, test_distance)
            if rows[calibration]['date'] in distance_reference else None,
            'all_except_calibration': aggregate(errors, errors_mc, test_all),
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    weather_input = directory / 'era5-input.json'
    weather = read_json(weather_input)
    if weather['endpoint_suffixes_utc'] != ['T22:00', 'T23:00']:
        raise ValueError('Unexpected weather times.')
    summaries = {}
    for site in ('Galzignano', 'Arcugnano'):
        source = directory / (site.lower() + '.json')
        summaries[site] = {**site_summary(read_json(source), weather['sites'][site]),
                           'transport_sha256': file_hash(source)}
    annual_source = directory / 'galzignano-annual.json'
    annual = None
    if annual_source.is_file():
        annual_doc = read_json(annual_source)
        if annual_doc['site'] != 'Galzignano' or 'annual_map_year' not in annual_doc:
            raise ValueError('Expected the annual-map control on Galzignano.')
        matched_dates = {row['date'] for row in summaries['Galzignano']['rows']
                         if row['passes_distance_check']}
        annual = {**site_summary(annual_doc, weather['sites']['Galzignano'], matched_dates),
                  'map_year': annual_doc['annual_map_year'],
                  'transport_sha256': file_hash(annual_source)}
    output = {'model_order': VARIANTS,
              'error_convention': '100 * (predicted total / measured total - 1)',
              'mc_interval_scope': 'Numerical draws only, sharing the calibration and boundary-mask covariance.',
              'weather_limits': WEATHER_LIMIT, 'weather_input_sha256': file_hash(weather_input),
              'local_distance_check': {'max_artificial_from_lt3km_pct': 20,
                                       'require_artificial_from_lt10km_pct_strictly_below': 50},
              'sites': summaries, 'annual_source_control': annual}
    write_json(directory / 'comparison.json', output)
    fieldnames = ('source_map', 'site', 'date_utc', 'sqm_readings', 'use', 'weather_pass', 'distance_pass',
                  'max_cloud_pct', 'aod_regional', 'aod_at_station', 'natural_fraction_pct',
                  *[f'{v}_error_pct' for v in VARIANTS], *[f'{v}_mc_se_pp' for v in VARIANTS])
    with (directory / 'nightly-errors.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        series = [('monthly', summary) for summary in summaries.values()]
        if annual is not None:
            series.append((f'annual_{annual["map_year"]}', annual))
        for source_map, summary in series:
            for row in summary['rows']:
                clear = row['weather']['passes']
                use = 'calibration' if row['date'] == summary['calibration_date'] else (
                    'control' if clear else 'weather_rejected')
                writer.writerow({'source_map': source_map, 'site': summary['site'],
                                 'date_utc': row['date'], 'sqm_readings': row['readings'],
                                 'use': use, 'weather_pass': clear,
                                 'distance_pass': row['passes_distance_check'],
                                 'max_cloud_pct': row['weather']['maxima']['cloud_cover'],
                                 'aod_regional': row['aod_regional'], 'aod_at_station': row['aod_at_station'],
                                 'natural_fraction_pct': row['natural_fraction_pct'],
                                 **{f'{v}_error_pct': row['signed_error_pct'][v] for v in VARIANTS},
                                 **{f'{v}_mc_se_pp': row['mc_se_pp'][v] for v in VARIANTS}})
    for site, summary in summaries.items():
        print(site, summary['summaries']['monthly']['mape_pct'], flush=True)


if __name__ == '__main__':
    main()
