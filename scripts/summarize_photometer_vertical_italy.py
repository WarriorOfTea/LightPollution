"""Calculate uncalibrated radiance differences and numerical MC errors."""

import argparse
import csv
from pathlib import Path

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json

CONTRASTS = {
    'regional_vs_base': ('regional_layered', 'base'),
    'spatial_vs_base': ('spatial_layered', 'base'),
    'spatial_vs_regional': ('spatial_layered', 'regional_layered'),
    'vertical_regional_vs_exp': ('regional_layered', 'regional_exponential'),
    'vertical_spatial_vs_exp': ('spatial_layered', 'spatial_exponential'),
}
RADIUS_LABELS = ('outside_0km', 'outside_3km', 'outside_10km')


def score(run, label):
    index = run['labels'].index(label)
    return (run['all_order_radiance'][index], run['all_order_standard_error'][index])


def contrast(new, reference):
    value, sigma = new
    base, base_sigma = reference
    if min(value, base) <= 0:
        raise ValueError('A pair requires positive radiance in both runs.')
    ratio = value / base
    # Independent transport seeds for each model. The result excludes field uncertainty.
    standard_error = ratio * np.hypot(sigma / value, base_sigma / base) * 100
    return {'pct': 100 * (ratio - 1), 'mc_se_pp': float(standard_error)}


def window_rows(previous, transport):
    comparison = read_json(previous / 'comparison.json')
    cases = {(row['site'], row['date']): row for row in transport['rows']}
    output = []
    for site, payload in comparison['sites'].items():
        path = previous / (site.lower() + '.json')
        if file_hash(path) != transport['inputs'][site]:
            raise ValueError(f'Archived exponential results changed at {site}.')
        old = read_json(path)
        rows = {r['date']: r for r in old['rows']}
        for observation in payload['rows']:
            date = observation['date']
            row = cases.pop((site, date))
            earlier = rows[date]['runs']
            variants = {**row['variants'], 'base': old['runs'][earlier['base']],
                        'regional_exponential': old['runs'][earlier['regional']],
                        'spatial_exponential': old['runs'][earlier['spatial']]}
            settings = [r['settings']['seed'] for r in variants.values()]
            if len(settings) != len(set(settings)) or any(
                    r['finite_scattering_order_truncation'] for r in variants.values()):
                raise ValueError('Expected independent seeds and untruncated transport.')
            changes = {}
            for label in RADIUS_LABELS:
                scores = {name: score(result, label) for name, result in variants.items()}
                changes[label] = {name: contrast(scores[a], scores[b])
                                  for name, (a, b) in CONTRASTS.items()}
            output.append({k: v for k, v in row.items() if k != 'variants'} | {
                'readings': observation['readings'],
                'weather_pass': observation['weather']['passes'],
                'same_prior_six': site == 'Galzignano' and observation['weather']['passes']
                and observation['passes_distance_check'] and date != payload['calibration_date'],
                'prior_calibration_window': date == payload['calibration_date'],
                'changes': changes,
                'artificial_radiance': {name: {'mean': value, 'mc_se': se}
                    for name, (value, se) in ((name, score(v, 'outside_3km')) for name, v in variants.items())}})
    if cases:
        raise ValueError('Extra vertical-profile results without corresponding SQM timestamps.')
    return output


def summarize(rows, label):
    summary = {'n': len(rows)}
    if not rows:
        return summary
    for name in CONTRASTS:
        changes = [row['changes'][label][name] for row in rows]
        values = np.array([p['pct'] for p in changes])
        se = np.array([p['mc_se_pp'] for p in changes])
        summary[name] = {'min_pct': float(values.min()), 'max_pct': float(values.max()),
                        'mean_signed_pct': float(values.mean()),
                        'mean_absolute_pct': float(np.abs(values).mean()),
                        'median_mc_se_pp': float(np.median(se)),
                        'n_beyond_two_mc_se': int(np.count_nonzero(abs(values) > 2 * se))}
    return summary


def write_table(path, rows):
    fields = ['site', 'date_utc', 'sqm_readings', 'era5_weather_pass', 'same_prior_six',
              'geos_cf_version', 'aod_regional', 'aod_near_station',
              'z50_region_m', 'z90_region_m', 'z50_near_station_m', 'z90_near_station_m',
              'source_mask']
    fields += [name + suffix for name in CONTRASTS for suffix in ('_pct', '_mc_se_pp')]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            for label in RADIUS_LABELS:
                changes = row['changes'][label]
                writer.writerow({'site': row['site'], 'date_utc': row['date'],
                    'sqm_readings': row['readings'], 'era5_weather_pass': row['weather_pass'],
                    'same_prior_six': row['same_prior_six'], 'geos_cf_version': row['geos_cf_version'],
                    'aod_regional': row['aod_regional'], 'aod_near_station': row['aod_at_station'],
                    'z50_region_m': row['region_z50_z90_m'][0], 'z90_region_m': row['region_z50_z90_m'][1],
                    'z50_near_station_m': row['station_z50_z90_m'][0],
                    'z90_near_station_m': row['station_z50_z90_m'][1], 'source_mask': label,
                    **{name + '_' + field: value for name, p in changes.items() for field, value in p.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    transport = read_json(args.output / 'transport.json')
    rows = window_rows(args.previous, transport)
    groups = {'six_galzignano': [r for r in rows if r['same_prior_six']],
              'galzignano_clear': [r for r in rows if r['site'] == 'Galzignano' and r['weather_pass']],
              'arcugnano_clear': [r for r in rows if r['site'] == 'Arcugnano' and r['weather_pass']],
              'all': rows}
    write_json(args.output / 'comparison.json', {
        'contrast_order': CONTRASTS, 'ratio_percent': '100 * (numerator/denominator - 1)',
        'uncertainty_scope': 'Independent Monte Carlo estimation only.',
        'transport_sha256': file_hash(args.output / 'transport.json'),
        'selection_sha256': file_hash(args.previous / 'comparison.json'),
        'summaries': {name: {label: summarize(group, label) for label in RADIUS_LABELS}
                      for name, group in groups.items()},
        'windows': rows})
    write_table(args.output / 'nightly-contrasts.csv', rows)
    print({key: f'{v["mean_absolute_pct"]:.2f}%'
           for key, v in summarize(groups['six_galzignano'], 'outside_3km').items() if key != 'n'})


if __name__ == '__main__':
    main()
