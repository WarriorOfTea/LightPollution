"""Compare Venetian SQM observations and all-order predictions using the stored scale."""

import argparse
import csv
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from summarize_photometer_vertical_italy import score

VARIANTS = ('base', 'regional', 'vertical', 'regional_vertical')


def compare_site(site, selection, archive, profiles, rng):
    anchor_date = selection['calibration_date']
    scale = selection['summaries']['monthly']['common_scale']
    natural = next(r['natural_relative'] for r in selection['rows'] if r['date'] == anchor_date)
    archive_rows = {r['date']: r for r in archive['rows']}
    anchor = archive_rows[anchor_date]
    anchor_base = anchor['runs']['base']
    base_anchor_radiance, _ = score(archive['runs'][anchor_base], 'outside_3km')
    if not np.isclose(scale * base_anchor_radiance, anchor['observed_relative'] - natural):
        raise ValueError('The stored calibration must agree with its archived inputs.')
    old_draws = {}
    for name, result in archive['runs'].items():
        mean, se = score(result, 'outside_3km')
        old_draws[name] = rng.normal(mean, se, 16_384)
    # Vary only the Monte Carlo estimate of the one already fixed reference.
    sampled_scale = scale * base_anchor_radiance / old_draws[anchor_base]
    rows, stochastic_errors = [], []
    for observation in selection['rows']:
        day = observation['date']
        keys = archive_rows[day]['runs']
        profile = profiles[(site, day)]
        measured = observation['observed_relative']
        if observation['natural_relative'] != natural:
            raise ValueError('The common background in the archived comparison must be fixed.')
        raw = profile['artificial_radiance']
        model = {'base': raw['base'], 'regional': raw['regional_exponential'],
                 'vertical': raw['spatial_layered'], 'regional_vertical': raw['regional_layered']}
        sampled = [old_draws[keys['base']], old_draws[keys['regional']]]
        sampled += [rng.normal(model[key]['mean'], model[key]['mc_se'], len(sampled_scale))
                    for key in VARIANTS[2:]]
        errors_draw = 100 * ((natural + sampled_scale * np.array(sampled)) / measured - 1)
        predicted = {key: natural + scale * model[key]['mean'] for key in VARIANTS}
        errors = {key: 100 * (predicted[key] / measured - 1) for key in VARIANTS}
        for key in ('base', 'regional'):
            if not np.isclose(errors[key], observation['signed_error_pct'][key], atol=1e-9, rtol=0):
                raise ValueError(f'Unexpected difference from the saved {key} run.')
        rows.append({'site': site, 'date_utc': day, 'readings': observation['readings'],
                     'anchor': day == anchor_date, 'weather_pass': observation['weather']['passes'],
                     'same_prior_six': profile['same_prior_six'],
                     'measured_100_eq_natural': 100 * measured / natural,
                     'model_100_eq_natural': {key: 100 * val / natural for key, val in predicted.items()},
                     'error_pct': errors,
                     'mc_se_pp': dict(zip(VARIANTS, errors_draw.std(axis=1, ddof=1).tolist(), strict=True)),
                     'raw_artificial': {key: model[key]['mean'] for key in VARIANTS}})
        stochastic_errors.append(errors_draw)
    return rows, stochastic_errors


def stats(rows, draws, indices):
    if not indices:
        raise ValueError('Empty comparison group.')
    errors = np.array([[rows[i]['error_pct'][m] for m in VARIANTS] for i in indices])
    mape = np.abs(errors).mean(axis=0)
    sampled_mape = np.stack([abs(draws[i]) for i in indices]).mean(axis=0)
    change = sampled_mape[2] - sampled_mape[1]
    return {'n': len(indices), 'readings': sum(rows[i]['readings'] for i in indices),
            'mape_pct': dict(zip(VARIANTS, mape.tolist(), strict=True)),
            'bias_pct': dict(zip(VARIANTS, errors.mean(axis=0).tolist(), strict=True)),
            'vertical_minus_regional_mape_pp': float(mape[2] - mape[1]),
            'vertical_minus_regional_mape_mc_95_pp': np.quantile(change, [.025, .975]).tolist()}


def temporal_control(rows):
    lookup = {r['date_utc']: r for r in rows if r['site'] == 'Galzignano'}
    first, last = (lookup[d] for d in ('2023-02-12', '2023-02-15'))
    return {'dates': [first['date_utc'], last['date_utc']],
            'measured_total_change_pct': 100 * (last['measured_100_eq_natural'] / first['measured_100_eq_natural'] - 1),
            'measured_minus_nominal_background_change_pct':
                100 * ((last['measured_100_eq_natural'] - 100) / (first['measured_100_eq_natural'] - 100) - 1),
            'raw_artificial_change_pct': {m: 100 * (last['raw_artificial'][m] / first['raw_artificial'][m] - 1)
                                          for m in VARIANTS}}


def write_csv(path, rows):
    fields = ['site', 'date_utc', 'readings', 'anchor', 'weather_pass', 'same_prior_six',
              'measured_100_eq_natural', *[name + suffix for name in VARIANTS
              for suffix in ('_100_eq_natural', '_error_pct', '_mc_se_pp')]]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields[:7]} | {
                name + suffix: row[key][name] for name in VARIANTS
                for suffix, key in (('_100_eq_natural', 'model_100_eq_natural'),
                                    ('_error_pct', 'error_pct'), ('_mc_se_pp', 'mc_se_pp'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('vertical', type=Path)
    args = parser.parse_args()
    previous = args.previous / 'comparison.json'
    vertical = args.vertical / 'comparison.json'
    prior, new = read_json(previous), read_json(vertical)
    if new['selection_sha256'] != file_hash(previous):
        raise ValueError('The experiments used different SQM windows.')
    profiles = {(row['site'], row['date']): row for row in new['windows']}
    rng = np.random.default_rng(10116)
    rows, draws = [], []
    for site, payload in prior['sites'].items():
        archive = read_json(args.previous / (site.lower() + '.json'))
        site_rows, uncertainty = compare_site(site, payload, archive, profiles, rng)
        rows.extend(site_rows)
        draws.extend(uncertainty)
    if set(profiles) != {(row['site'], row['date_utc']) for row in rows}:
        raise ValueError('Incomplete match between optical transport and SQM.')
    groups = {'the_same_six': [i for i, r in enumerate(rows) if r['same_prior_six']]}
    for site in prior['sites']:
        groups[site] = [i for i, r in enumerate(rows) if r['site'] == site and r['weather_pass'] and not r['anchor']]
    result = {'inputs': {'prior_sha256': file_hash(previous), 'vertical_sha256': file_hash(vertical)},
              'scale': '100 = original nominal natural brightness; calibrated source scale reused from prior.',
              'model_order': VARIANTS,
              'groups': {name: stats(rows, draws, indices) for name, indices in groups.items()},
              'february_change_without_source_scale': temporal_control(rows), 'rows': rows}
    write_json(args.vertical / 'measured-comparison.json', result)
    write_csv(args.vertical / 'measured-comparison.csv', rows)
    for group, details in result['groups'].items():
        print(group, details['mape_pct'], details['vertical_minus_regional_mape_mc_95_pp'])


if __name__ == '__main__':
    main()
