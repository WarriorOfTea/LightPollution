"""Compare the inversion diagnostics with the unchanged ground observations."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnose_halo_ground import observation_key
from lightpollution.adapters.storage import file_hash, read_json, write_json
from run_halo_city_fits import HALO
from run_halo_ground_validation import BERLIN, ENHANCED, HELD, REFERENCE
from run_viirs_halo_recovery import DATA


CITIES = ('Berlin', 'Madrid', 'Tucson')
CONSERVING = {
    'conserve',
    'contrast_conserve',
    'leave_pass_out_conserve',
    'leave_pass_out_contrast_conserve',
}


def old_band_kernels():
    berlin, reference = read_json(BERLIN), read_json(REFERENCE)
    kernels = {}
    for row in berlin['rows']:
        key = f'berlin-2017-03/{row["stop"]}'
        bands = berlin['spectral_bands']
        runs = row['atmospheres']['aeronet']['spectral_radiance_runs']
        kernels[key, 'photopic'] = ([b['mcd_m2_per_dnb_unit'] for b in bands], runs)
    for row in reference['rows']:
        k, band = row['run'], row['band']
        kernels[k, band] = (
            [b['W_m2_sr_per_dnb'][band] for b in reference['spectral_bands']],
            reference['runs'][k]['bands'],
        )
    return kernels


def utc_qualities():
    reference, berlin = read_json(REFERENCE), read_json(BERLIN)
    input_rows = [
        r | {'key': r['run']} for r in reference['rows'] if 'astro_checks' in r
    ]
    result = {observation_key(row): row['astro_checks'] for row in input_rows}
    for row in berlin['rows']:
        result[
            (f'berlin-2017-03/{row["stop"]}', '2017-03-29', row['site'], 'photopic')
        ] = [
            {
                'sun_alt_deg': row['sun_altitude_deg'],
                'moon_alt_deg': row['moon_altitude_deg'],
            }
        ]
    return result


def ground_row(row, orders, satellite, kernel, eph):
    scene = satellite[row['fitted_scene']]
    predictions = row['prediction']
    observed = row['observed']
    values = np.array(orders['single_and_multiple'])
    held = values[:, HELD]
    tails = values[:, ENHANCED] - held
    middle = values[:, -1]
    baseline = predictions['off_h1']['linear']
    kappa = scene['fits']['original']['kappa']
    addition = 100 * (kappa - 1) * tails / observed
    np.testing.assert_allclose(
        baseline + kappa * tails.sum(), predictions['a1_anchored']['linear'], rtol=3e-11
    )
    simulated = {}
    for mode, result in scene['fits'].items():
        if result is not None:
            response = tails - middle if mode in CONSERVING else tails
            simulated[mode] = baseline + result['kappa'] * response.sum()
    simulated['zero_natural'] = simulated['original'] - row['natural']
    simulated['h2_zero_natural'] = simulated['h2'] - row['natural']
    single_ground_base = row['natural'] + values[0, 0] + held[0]
    simulated['single_ground'] = single_ground_base + kappa * tails[0]
    simulated['single_both'] = (
        single_ground_base + scene['fits']['single']['kappa'] * tails[0]
    )
    bands, runs = kernel
    spectral = np.array([b['all_order_radiance'] for b in runs])
    tail_spectral = spectral[:, ENHANCED] - spectral[:, HELD]
    for spectrum in ('HP1', 'LED-B2'):
        sky = np.array([b[spectrum] for b in bands]) @ tail_spectral
        simulated['original_tail_' + spectrum] = baseline + kappa * sky
    simulated['a1_free'] = predictions['a1_free']['linear']
    artificial = values[:, [0, HELD]].sum(axis=1) + kappa * tails
    total_tail_se = 100 * abs(kappa - 1) * orders['tail_standard_error'] / observed
    return {
        **{
            k: row[k]
            for k in (
                'city',
                'night',
                'site',
                'band',
                'distance_km',
                'fitted_scene',
                'utc_offset_hours',
                'calendar_offset_days',
                'city_dominant',
                'key',
            )
        },
        'naive_natural_pct_of_observed': 100 * row['natural'] / observed,
        'tail_extra_points_by_order': addition.tolist(),
        'tail_extra_points_mcse': total_tail_se,
        'original_artificial_pct_by_order': (100 * artificial / observed).tolist(),
        'relative_pct': {
            name: 100 * (value / observed - 1) for name, value in simulated.items()
        },
        'max_ground_sun_deg': max(e['sun_alt_deg'] for e in eph),
        'max_ground_moon_deg': max(e['moon_alt_deg'] for e in eph),
    }


def scores(rows):
    by_method = defaultdict(list)
    for row in rows:
        for method, value in row['relative_pct'].items():
            by_method[method].append(value)
    return {
        method: {
            'n': len(e),
            'mae_pct': float(np.mean(np.abs(e))),
            'bias_pct': float(np.mean(e)),
            'positive': int(np.count_nonzero(np.array(e) > 0)),
        }
        for method, e in by_method.items()
    }


def ground_summary(rows):
    terms = np.array([r['tail_extra_points_by_order'] for r in rows])
    s = terms.sum(axis=0)
    return {
        'n': len(rows),
        'metrics': scores(rows),
        'mean_tail_extra_pct_by_order': terms.mean(axis=0).tolist(),
        'mean_natural_pct_of_observed': float(
            np.mean([r['naive_natural_pct_of_observed'] for r in rows])
        ),
        'multiple_share_of_added_skyglow': float(s[1] / s.sum()),
        'median_mc_error_on_delta_points': float(
            np.median([r['tail_extra_points_mcse'] for r in rows])
        ),
        'max_mc_error_on_delta_points': max(r['tail_extra_points_mcse'] for r in rows),
        'tail_larger_than_5mcse_count': sum(
            sum(r['tail_extra_points_by_order']) > 5 * r['tail_extra_points_mcse']
            for r in rows
        ),
        'max_sun_alt_deg': max(r['max_ground_sun_deg'] for r in rows),
        'max_moon_alt_deg': max(r['max_ground_moon_deg'] for r in rows),
    }


def satellite_summary(scenes):
    modes = sorted({m for r in scenes.values() for m, v in r['fits'].items() if v})
    data = {}
    for m in modes:
        fit = [r['fits'][m] for r in scenes.values() if r['fits'].get(m)]
        ordinary = np.array([f['test'][HALO]['mae'] for f in fit])
        signed = np.array([f['test'][HALO]['bias'] for f in fit])
        both = [f for f in fit if f['contrast']['test_pairs']]
        contrast_error = [
            np.mean(
                np.abs(
                    np.subtract(f['contrast']['predicted'], f['contrast']['observed'])
                )
            )
            for f in both
        ]
        data[m] = {
            'passes': len(fit),
            'test_blocks': sum(f['test'][HALO]['n'] for f in fit),
            'halo_mae': float(ordinary.mean()),
            'halo_bias': float(signed.mean()),
            'contrast_passes': len(both),
            'contrast_pairs': sum(f['contrast']['test_pairs'] for f in both),
            'contrast_mae': float(np.mean(contrast_error)) if both else None,
            'min_max_kappa': [
                min(f['kappa'] for f in fit),
                max(f['kappa'] for f in fit),
            ],
        }
    return data


def matched_summaries(rows, scenes):
    controls = {
        'contrast_conserve': 'h2',
        'leave_pass_out_original': 'leave_pass_out_h2',
        'leave_pass_out_conserve': 'leave_pass_out_h2',
        'leave_pass_out_contrast_conserve': 'leave_pass_out_h2',
        'tucson_ground_g': 'original',
        'second_seed': 'original',
    }
    comparisons = {}
    for method, baseline in controls.items():
        selected = {key: v for key, v in scenes.items() if v['fits'].get(method)}
        metrics = satellite_summary(selected)
        common = [v for v in rows if method in v['relative_pct']]
        ground = scores(common)
        comparisons[method] = {
            'control': baseline,
            'satellite': {k: metrics[k] for k in (baseline, method)},
            'ground': {k: ground[k] for k in (baseline, method)},
        }
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous_ground', 'ground_replay', 'satellite', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    archived = [r for r in read_json(args.previous_ground)['rows'] if r['fitted_scene']]
    ground = read_json(args.ground_replay)
    if ground['previous_ground_sha256'] != file_hash(args.previous_ground):
        raise ValueError('Inconsistent ground archive and replay.')
    replay = {observation_key(v): v for v in ground['rows']}
    satellite = read_json(args.satellite)
    scene_data = satellite['scenes']
    kernels = old_band_kernels()
    quality = utc_qualities()
    rows = [
        ground_row(
            r,
            replay[observation_key(r)],
            scene_data,
            kernels[r['key'], r['band']],
            quality[observation_key(r)],
        )
        for r in archived
    ]
    samples = {
        'All': rows,
        'same_night': [r for r in rows if r['calendar_offset_days'] == 0],
    }
    for city in CITIES:
        subset = [r for r in rows if r['city'] == city]
        samples[city] = subset
        for label, lo, hi in (('near', 0, 15), ('mid', 15, 35), ('far', 35, 1000)):
            segment = [r for r in subset if lo <= r['distance_km'] < hi]
            if segment:
                samples[f'{city}/{label}'] = segment
        dominant = [r for r in subset if r['city_dominant']]
        if dominant:
            samples[city + '/city_dominant'] = dominant
    samples['Tucson/less_cloud'] = [
        r for r in samples['Tucson'] if r['night'] not in ('2019-04-01', '2019-04-07')
    ]
    for r in rows:
        for label in (f'{r["city"]}/{r["night"]}', f'{r["city"]}/band/{r["band"]}'):
            samples.setdefault(label, []).append(r)
    weather = read_json(DATA / 'weather-audit.json')['weather_by_night']
    write_json(
        args.output,
        {
            'ground': {name: ground_summary(group) for name, group in samples.items()},
            'satellite': satellite_summary(scene_data),
            'matched': matched_summaries(rows, scene_data),
            'rows': rows,
            'ground_integrations': ground['spectral_integrations'],
            'ground_paths': ground['paths'],
            'satellite_kernels': satellite['kernel_count'],
            'satellite_paths': satellite['samples_per_kernel']
            * satellite['kernel_count'],
            'weather_by_night': weather,
            'sources_sha256': {
                p.name: file_hash(p)
                for p in (args.previous_ground, args.ground_replay, args.satellite)
            },
        },
    )


if __name__ == '__main__':
    main()
