"""Fit city spectrum and angular parameters on two nights and score held-out nights."""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import file_hash, read_json, write_json, write_npz
from lightpollution.models.spectra import ARCSEC2_PER_SR, band_integral
from run_city_source_transport import ART, CITY_CODES


PHOTOMETRY = ART / 'photometer-sqm-residuals-20261126'
COHORTS = {'Verona': ('Campofontana', 'Cattignano', 'Marana', 'Montebaldo'), 'Vicenza': ('Arcugnano',),
           'Padova': ('Galzignano',), 'Pordenone': ('Piadera',), 'Treviso': ('Arson',), 'Madrid': ('AstMon_UCM',)}
SPECTRA = ('HP1', 'LED-B2', 'LED-B3')
PARAMETERS = ('spectral_only', 'angular_only', 'joint', 'joint_LED_B3', 'joint_horizon_4')
REFERENCE = 'lambert_reference'
FIXED_ANGLE = 'satellite_angles'
AREA_METHOD = 'GISCO City 2024, polygon membership of native Black Marble pixel centers'


def bands():
    library = load_spectral_library()
    report = read_json(PHOTOMETRY / 'results.json')
    sqm = report['band_projections']['mean']
    dnb_wave, dnb_curve = np.loadtxt(PHOTOMETRY / 'viirs-dnb-rsr.dat', unpack=True)
    wl = np.unique(np.r_[library.wavelengths, dnb_wave])
    wl = wl[(wl >= 380) & (wl <= 780)]
    dnb = np.interp(wl, dnb_wave, dnb_curve, left=0, right=0)
    v = np.interp(wl, library.wavelengths, library.responses['johnson_v'])
    spectra_v = {}
    for name in SPECTRA:
        template = np.interp(wl, library.wavelengths, library.templates[name])
        spectra_v[name] = float(band_integral(wl, template, v) / band_integral(wl, template, dnb))
    # Use nW/cm²/sr for the prediction and zero point, consistently with the source basis.
    zero_v = library.zero_v * ARCSEC2_PER_SR * 1e5
    return {'SQM': {'lamp': sqm['artificial_SQM_over_DNB'], 'zero': sqm['vega_zero_W_m2_sr']*1e5,
                    'natural_at_v22': zero_v*10**(-.4*22)*sqm['natural_SQM_over_V']},
            'V': {'lamp': spectra_v, 'zero': zero_v, 'natural_at_v22': zero_v*10**(-.4*22)}}


def reference_factor(band):
    return (band['lamp']['HP1'] + band['lamp']['LED-B2']) / 2


def reference_atmosphere(row):
    return 'power_regional' if row['band'] == 'V' else 'regional'


def load_transport(paths, replacement=None):
    observations, map_diagnostics, counts = {}, {}, {}
    arrays, original = {}, {}
    for path in paths:
        series = path.parent.name
        if series in counts:
            raise ValueError(f'Series name is duplicated: {series}.')
        runs = []
        for period, batch in read_json(path).items():
            names = batch['columns']
            map_diagnostics[f'{series}/{period}'] = batch['map']
            for key, row in batch['records'].items():
                if path == replacement:
                    prior = observations[key]
                    if any(row[name] != prior[name] for name in ('night', 'utc', 'site', 'band', 'magnitude')):
                        raise ValueError(f'Original and fallback photometry differ for {key}.')
                    original[key] = prior
                elif key in observations:
                    raise ValueError(f'Duplicate observation: {key}.')
                row['series'] = series
                row['city_map_support'] = batch['map']
                row['basis'] = {}
                for mode, values in row.pop('models').items():
                    runs.append((key, mode, values))
                    mean = dict(zip(names, values['all_order_radiance'], strict=True))
                    contributions = sum(mean.get(f'{city}_lambert', 0) for city in CITY_CODES)
                    if not 0 <= contributions <= mean['total_lambert'] + 1e-13:
                        raise ValueError(f'City components exceed total source signal in {key}/{mode}.')
                    row['basis'][mode] = mean
                observations[key] = row
        counts[series] = {'windows': len({key for key, _, _ in runs}),
                          'unique_runs': len({v['seed'] for _, _, v in runs}),
                          'trajectories': sum(v['samples'] for v in {v['seed']: v for _, _, v in runs}.values())}
        arrays.update({series + '_' + name: np.asarray(values) for name, values in {
            'keys': [key for key, _, _ in runs], 'models': [m for _, m, _ in runs], 'columns': names,
            'means': [v['all_order_radiance'] for _, _, v in runs],
            'covariance': [v['mean_covariance'] for _, _, v in runs],
            'effective_samples': [v['effective_score_samples'] for _, _, v in runs],
            'seeds': [v['seed'] for _, _, v in runs], 'samples': [v['samples'] for _, _, v in runs],
        }.items()})
    return observations, original, map_diagnostics, counts, arrays


def components(row, city, mode):
    mean = row['basis'][mode]
    return np.array([mean.get(f'{city}_{name}', 0.) for name in ('lambert', 'off_hold', 'horizon_slope')])


def city_map_sufficient(row, city, fraction):
    stats = row['city_map_support'][city]
    return (stats['near_supported'] >= fraction*stats['polygon_pixels'] and
            (stats['common_fraction_of_near_dnb'] or 0.) >= fraction)


def training_pair(cohort, observations, city, reserved):
    keys = [k for k in cohort if observations[k]['night'] not in reserved and
            city_map_sufficient(observations[k], city, .5) and
            components(observations[k], city, reference_atmosphere(observations[k]))[0] > 0]
    # Two Madrid dates were already the development dates of the earlier pilot.
    if city == 'Madrid':
        return tuple(k for k in ('Madrid-2017-03-19', 'Madrid-2017-11-19') if k in keys)
    if len({observations[k]['night'] for k in keys}) < 3:
        return ()
    design = {}
    for key in keys:
        row = observations[key]
        mode = reference_atmosphere(row)
        design[key] = components(row, city, mode)[1:] / row['basis'][mode]['total_lambert']

    def rank(pair):
        first, second = map(observations.get, pair)
        a, b = map(design.get, pair)
        return (first['site'] != second['site'], first['map_period'] != second['map_period'],
                abs(float(a[0]*b[1] - a[1]*b[0])), -(first['cloud_pct'] + second['cloud_pct']))

    return max((pair for pair in combinations(keys, 2)
                if observations[pair[0]]['night'] != observations[pair[1]]['night']), key=rank)


def observed_and_natural(row, band):
    return band['zero'] * 10**(-.4*row['magnitude']), row['natural_v_relative']*band['natural_at_v22']


def best_fit(train, city, band, spectrum, angular, led='LED-B2', horizon_max=2.):
    hp, led_coefficient = band['lamp']['HP1'], band['lamp'][led]
    lamp_fraction = np.linspace(0, 1, 201) if spectrum else np.array([.5])
    factors = hp + lamp_fraction*(led_coefficient - hp)
    data, rest = [], []
    for row in train:
        mode = reference_atmosphere(row)
        obs, natural = observed_and_natural(row, band)
        lamp, anchor, slope = components(row, city, mode)
        total = row['basis'][mode]['total_lambert']
        data.append(np.array([lamp, anchor, slope])/obs)
        rest.append((natural + (total - lamp)*reference_factor(band))/obs - 1)
    lambert, hold, low_angle = np.asarray(data).T
    error = np.asarray(rest)[None, :] + factors[:, None]*(hold if angular else lambert)
    if angular:
        slopes = factors[:, None] * low_angle
        norm = (slopes**2).sum(axis=1)
        shift = np.clip(np.divide(-(error*slopes).sum(axis=1), norm,
                                  out=np.zeros_like(norm), where=norm > 0), -1., horizon_max - 1.)
        error += slopes * shift[:, None]
        horizon = 1 + shift
    else:
        horizon = np.ones_like(lamp_fraction)
    rmse = np.sqrt((error**2).mean(axis=1))*100
    best = int(np.argmin(rmse))
    close = lamp_fraction[rmse <= rmse[best] + 1]
    return {'led_template': led, 'led_fraction_dnb': float(lamp_fraction[best]), 'horizon_over_off': float(horizon[best]),
            'angular': angular, 'training_rmse_pct': float(rmse[best]), 'train_profile_span_pp': float(rmse.max()-rmse.min()),
            'led_fraction_within_1pp_train_rmse': [float(close.min()), float(close.max())]}


def prediction(row, mode, band, city_params):
    total = row['basis'][mode]['total_lambert']
    observed, natural = observed_and_natural(row, band)
    signal = natural + total * reference_factor(band)
    for city, params in city_params.items():
        lambert, hold, slope = components(row, city, mode)
        strength = hold + (params['horizon_over_off'] - 1)*slope if params['angular'] else lambert
        p = params['led_fraction_dnb']
        fraction = p*band['lamp'][params['led_template']] + (1-p)*band['lamp']['HP1']
        signal += fraction*strength - lambert*reference_factor(band)
    ratio = signal / observed
    return {'pct': float((ratio-1)*100), 'mag': float(-2.5*np.log10(ratio)),
            'predicted_mag': float(-2.5*np.log10(signal/band['zero']))}


def metrics(records, keys):
    by_atmosphere = defaultdict(list)
    for key in keys:
        for mode, result in records[key].items():
            by_atmosphere[mode].append(result)
    summary = {}
    for mode, rows in by_atmosphere.items():
        summary[mode] = {}
        for variant in rows[0]:
            pct = np.array([r[variant]['pct'] for r in rows])
            mag = np.array([r[variant]['mag'] for r in rows])
            summary[mode][variant] = {'n': len(rows), 'bias_pct': float(pct.mean()), 'mae_pct': float(np.abs(pct).mean()),
                                      'bias_mag': float(mag.mean()), 'mae_mag': float(np.abs(mag).mean()),
                                      'rmse_mag': float(np.sqrt((mag**2).mean()))}
    return summary


def range_summary(values):
    return np.quantile(values, [0, .5, 1]).tolist() if values else None


def city_best_possible(row, mode, city, band, horizon_max):
    observed, natural = observed_and_natural(row, band)
    city_lambert, hold, slope = components(row, city, mode)
    outside = row['basis'][mode]['total_lambert'] - city_lambert
    fixed = natural + outside * reference_factor(band)
    gains = [band['lamp'][name] for name in SPECTRA]
    lo = fixed + min(gains) * max(0., min(city_lambert, hold - slope))
    hi = fixed + max(gains) * max(city_lambert, hold + (horizon_max - 1)*slope)
    return 100*abs(np.clip(observed, lo, hi)/observed - 1)


def analyze(paths, replacement=None):
    inputs = paths + [replacement] if replacement else paths
    observations, original, satellite, counts, arrays = load_transport(inputs, replacement)
    calibrated, table, windows = {}, {}, {}
    projection = bands()
    reserved = set()
    protected = set()
    cohorts = {city: sorted(k for k, row in observations.items() if row['site'].removeprefix('SQM_IT_') in sites)
               for city, sites in COHORTS.items()}
    city_anchors = {}
    for city in sorted(cohorts, key=lambda name: len(cohorts[name])):
        keys = cohorts[city]
        anchors = training_pair(keys, observations, city, reserved | protected)
        city_anchors[city] = anchors
        reserved.update(observations[k]['night'] for k in anchors)
        spare = {observations[k]['night'] for k in keys} - reserved
        if len(anchors) == 2 and len(spare) == 1:
            protected.update(spare)

    for city, cohort in cohorts.items():
        if not cohort:
            continue
        band = projection[observations[cohort[0]]['band']]
        anchors = city_anchors[city]
        holdout = [k for k in cohort if observations[k]['night'] not in reserved]
        anchor_rows = [observations[k] for k in anchors]
        parameters = {FIXED_ANGLE: {'led_template': 'LED-B2', 'led_fraction_dnb': .5,
                                   'horizon_over_off': 1., 'angular': True}}
        if len(anchors) == 2 and holdout:
            parameters.update(dict(zip(PARAMETERS, (
                best_fit(anchor_rows, city, band, True, False), best_fit(anchor_rows, city, band, False, True),
                best_fit(anchor_rows, city, band, True, True), best_fit(anchor_rows, city, band, True, True, led='LED-B3'),
                best_fit(anchor_rows, city, band, True, True, horizon_max=4.)), strict=True)))
            calibrated[city] = parameters['joint']
        predicted = {}
        for key in cohort:
            row = observations[key]
            predicted[key] = {}
            for mode in row['basis']:
                predicted[key][mode] = {REFERENCE: prediction(row, mode, band, {})} | {
                    name: prediction(row, mode, band, {city: value}) for name, value in parameters.items()}
            terms = row['basis'][reference_atmosphere(row)]
            total = terms['total_lambert']
            windows[key] = {'city': city, 'site': row['site'], 'night': row['night'], 'utc': row['utc'],
                            'series': row['series'], 'map_period': row['map_period'], 'band': row['band'],
                            'original_map_period': original[key]['map_period'] if key in original else None,
                            'n_readings': row['readings'], 'observed_mag': row['magnitude'], 'aod': row['aod'],
                            'split': 'anchor' if key in anchors else 'holdout' if key in holdout else 'same_anchor_night',
                            'city_fraction_pct': 100*terms.get(f'{city}_lambert', 0)/total,
                            'inside_10km_pct': 100*terms['inside_10km']/total,
                            'predictions': predicted[key]}
        anchor_maps = {r['map_period'] for r in anchor_rows}
        site_set = {r['site'] for r in anchor_rows}
        strata = {'holdout': holdout,
                   'unseen_map': [k for k in holdout if observations[k]['map_period'] not in anchor_maps] if anchors else [],
                   'unseen_station': [k for k in holdout if observations[k]['site'] not in site_set] if anchors else [],
                   'inside_10km_lt50pct': [k for k in holdout if windows[k]['inside_10km_pct'] < 50],
                   'well_mapped_city': [k for k in holdout if city_map_sufficient(observations[k], city, .8)],
                   'same_three_veneto_atmospheres': [k for k in holdout if 'spatial_vertical' in observations[k]['basis']],
                   'original_148': [k for k in holdout if observations[k]['tier'] in ('ideal','nearly_ideal')]}
        strata.update({f'station/{site}': [k for k in holdout if observations[k]['site'].endswith(site)] for site in COHORTS[city]})
        attainable = {}
        for horizon in (2, 4):
            errors = np.asarray([city_best_possible(observations[k], reference_atmosphere(observations[k]),
                                                    city, band, horizon) for k in holdout])
            if len(errors):
                attainable[str(horizon)] = {'mae_pct_floor': float(errors.mean()),
                                            'measurements_inside_model_range': int(np.count_nonzero(errors < 1e-10)),
                                            'windows': len(errors)}
        table[city] = {'windows': len(cohort), 'nights': len({observations[k]['night'] for k in cohort}),
                       'sites': sorted({observations[k]['site'] for k in cohort}),
                       'source_share_pct_min_median_max': range_summary([windows[k]['city_fraction_pct'] for k in cohort]),
                       'inside_10km_pct_min_median_max': range_summary([windows[k]['inside_10km_pct'] for k in cohort]),
                       'anchor_keys': anchors, 'anchor_maps': sorted(anchor_maps), 'parameters': parameters,
                       'optimistic_city_only_non_forecast': attainable,
                       'training_metrics': metrics(predicted, anchors),
                       'held_out': {name: {'nights': len({observations[k]['night'] for k in keys}),
                                           'metrics': metrics(predicted, keys)} for name, keys in strata.items()}}
        map_control = {}
        for key in holdout:
            if key not in original:
                continue
            row, prior = observations[key], original[key]
            map_control[key] = {mode: {'original_monthly': prediction(prior, mode, band, {}),
                                      'annual': prediction(row, mode, band, {})} for mode in row['basis']}
        table[city]['annual_fallback_no_fit_control'] = metrics(map_control, map_control.keys())

    valid = [k for k, row in windows.items() if row['split'] == 'holdout' and row['band'] == 'SQM']
    combined = {k: {mode: {REFERENCE: prediction(observations[k], mode, projection['SQM'], {}),
                            'city_calibrations': prediction(observations[k], mode, projection['SQM'],
                                {city: parameters for city, parameters in calibrated.items() if city != 'Madrid'})}
                     for mode in observations[k]['basis']} for k in valid}
    subtotals = {city: metrics(combined, [k for k in valid if windows[k]['city'] == city]) for city in cohorts if city != 'Madrid'}
    ratios = {city: {} for city in CITY_CODES}
    for city in CITY_CODES:
        for site in sorted({r['site'] for r in observations.values()}):
            share = [float(100*components(r, city, reference_atmosphere(r))[0]/r['basis'][reference_atmosphere(r)]['total_lambert'])
                     for r in observations.values() if r['site'] == site]
            if max(share) > 0:
                ratios[city][site] = range_summary(share)
    summary = {'bands': projection, 'scoring': 'Equal window weights, relative flux least squares on two nights per city',
               'cities': table, 'reserved_nights': sorted(reserved), 'transport': counts, 'source_map_diagnostics': satellite,
               'city_share_by_site_pct_min_median_max': ratios, 'all_cities_together_on_holdout': subtotals,
               'geo': AREA_METHOD, 'inputs': {p.parent.name: file_hash(p) for p in inputs}}
    return summary, windows, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('transport', nargs='+', type=Path)
    parser.add_argument('--annual-fallback', type=Path)
    args = parser.parse_args()
    summary, rows, basis = analyze(args.transport, args.annual_fallback)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'results.json', summary)
    write_json(args.output / 'windows.json', rows)
    write_npz(args.output / 'transport-basis.npz', **basis)


if __name__ == '__main__':
    main()
