"""Summarize unfitted transport and compare the sign of matched SQM changes."""

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json, write_npz
from run_selected_aod_models import HEIGHT_BINS, LABELS, NATIVE_LAT, NATIVE_LON

ROME = ZoneInfo('Europe/Rome')
NATURAL_MAG_LIMITS = (20.99, 22.10)
MIN_OBSERVED_CHANGE_MAG = .05
MIN_EFFECTIVE_SAMPLES = 100
NUMERIC_MARGIN = 2.
MAX_NIGHT_INTERVAL_DAYS = 30


def direction(a, b, label):
    first, second = (row['artificial'][label] for row in (a, b))
    if first['seed'] == second['seed'] and first == second:
        return 0
    if min(first['effective_score_samples'][0], second['effective_score_samples'][0]) < MIN_EFFECTIVE_SAMPLES:
        return None
    difference = second['all_order_radiance'][0] - first['all_order_radiance'][0]
    error = float(np.hypot(first['all_order_standard_error'][0], second['all_order_standard_error'][0]))
    return int(np.sign(difference)) if abs(difference) > NUMERIC_MARGIN * error else 0


def matched_pairs(results):
    groups = defaultdict(list)
    for key, row in results.items():
        local = datetime.fromisoformat(row['utc']).astimezone(ROME)
        source_group = (row['site'], row['map_period'], local.hour)
        if all(label in row['artificial'] for label in LABELS):
            groups[source_group].append((key, row))

    pairs = []
    sky_span = float(np.ptp(10**(-.4*(np.array(NATURAL_MAG_LIMITS)-22))))
    for group in groups.values():
        group.sort(key=lambda item: item[1]['utc'])
        for (ka, a), (kb, b) in combinations(group, 2):
            elapsed = date.fromisoformat(b['night']) - date.fromisoformat(a['night'])
            if not timedelta(0) < elapsed <= timedelta(days=MAX_NIGHT_INTERVAL_DAYS):
                continue
            delta = a['sqm']-b['sqm']  # Magnitude decreases when the sky brightens.
            if abs(delta) < MIN_OBSERVED_CHANGE_MAG:
                continue
            flux = 10**(-.4*(b['sqm']-22))-10**(-.4*(a['sqm']-22))
            pairs.append({'earlier': ka, 'later': kb, 'site': a['site'],
                          'monthly': len(a['map_period']) == 7,
                          'observed_sign': int(np.sign(delta)), 'observed_brightening_mag': delta,
                          'robust_to_natural_v_span_proxy': abs(flux) > sky_span,
                          'sign': {label: direction(a, b, label) for label in LABELS}})
    return pairs


def pair_stats(pairs):
    counts = {}
    for label in LABELS:
        resolved = [p for p in pairs if p['sign'][label] not in (None, 0)]
        counts[label] = {'resolved': len(resolved), 'unresolved': sum(p['sign'][label] == 0 for p in pairs),
                         'low_effective_samples': sum(p['sign'][label] is None for p in pairs),
                         'agree': sum(p['sign'][label] == p['observed_sign'] for p in resolved)}
    both = [p for p in pairs if all(p['sign'][m] not in (None, 0) for m in LABELS[1:])]
    return {'pairs': len(pairs), 'models': counts, 'common_resolved': len(both),
            'common_resolved_agree': {m: sum(p['sign'][m] == p['observed_sign'] for p in both)
                                      for m in LABELS[1:]}}


def transport_stats(results):
    summary = {}
    for label in LABELS:
        sims = [r['artificial'][label] for r in results.values() if label in r['artificial']]
        uncertainty = [100*x['all_order_standard_error'][0]/x['all_order_radiance'][0] for x in sims]
        summary[label] = {'windows': len(sims), 'relative_mc_error_median_pct': float(np.median(uncertainty)),
                          'relative_mc_error_max_pct': max(uncertainty),
                          'effective_lt_100': sum(x['effective_score_samples'][0]<100 for x in sims),
                          'median_outside_10km_pct': float(np.median([100*x['all_order_radiance'][2]/
                                                     x['all_order_radiance'][0] for x in sims]))}
        summary[label]['near_3km_over_20_pct'] = sum(
            x['all_order_radiance'][1] < .8*x['all_order_radiance'][0] for x in sims)
        summary[label]['near_10km_at_least_half'] = sum(
            x['all_order_radiance'][2] <= .5*x['all_order_radiance'][0] for x in sims)
    changes = {}
    full = [r for r in results.values() if len(r['artificial']) == len(LABELS)]
    for a, b in combinations(LABELS, 2):
        delta = []
        resolved = 0
        for row in full:
            first, second = (row['artificial'][m] for m in (a, b))
            v1, v2 = (x['all_order_radiance'][0] for x in (first, second))
            se = np.hypot(first['all_order_standard_error'][0], second['all_order_standard_error'][0])
            delta.append(100*(v2/v1-1))
            if min(x['effective_score_samples'][0] for x in (first, second)) >= MIN_EFFECTIVE_SAMPLES:
                resolved += int(abs(v2-v1) > NUMERIC_MARGIN*se)
        changes[a+' / '+b] = dict(windows=len(delta), median_pct=float(np.median(delta)),
                                  p10_p90_pct=np.quantile(delta, [.1, .9]).tolist(), resolved=resolved)
    return summary, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    transport = args.workspace/'transport.json'
    data = read_json(transport)
    if data['selection_sha256'] != file_hash(args.selection):
        raise ValueError('The selection differs from the one used in transport.')
    selection = read_json(args.selection)['station_nights']
    results = data['results']
    counts, changes = transport_stats(results)
    pairs = matched_pairs(results)
    bright = {key for key, item in selection.items()
              if item['natural_plus_moon_fraction_proxy']['21.35'] <= .5}
    strong = [p for p in pairs if p['earlier'] in bright and p['later'] in bright]
    groups = {'all': pairs, 'strong_artificial': strong,
              'strong_monthly': [p for p in strong if p['monthly']], 'monthly': [p for p in pairs if p['monthly']],
              'annual': [p for p in pairs if not p['monthly']],
              'natural_span': [p for p in pairs if p['robust_to_natural_v_span_proxy']]}
    sites = sorted({r['site'] for r in results.values()})
    full = [r for r in results.values() if len(r['artificial']) == len(LABELS)]
    bright_full = [r for key, r in results.items() if key in bright and len(r['artificial']) == len(LABELS)]
    executions = sum(len({r['artificial'][label]['seed'] for r in results.values()
                           if label in r['artificial']}) for label in LABELS)
    data['comparison'] = {
        'windows': len(results), 'complete_triples': len(full), 'complete_nights': len({r['night'] for r in full}),
        'complete_bright_nights': len({r['night'] for r in bright_full}),
        'unique_runs': executions, 'transport_histories': executions * next(iter(results.values()))['samples'],
        'retained_sqm_readings': sum(r['n_sqm'] for r in full),
        'complete_by_site': {s: sum(r['site'] == s for r in full) for s in sites},
        'monthly_complete_windows': sum(len(r['map_period']) == 7 for r in full),
        'annual_complete_windows': sum(len(r['map_period']) == 4 for r in full),
        'transport': counts, 'changes': changes, 'same_source_sqm_sign': {k: pair_stats(v) for k, v in groups.items()},
        'same_source_sqm_sign_by_site': {s: pair_stats([p for p in pairs if p['site']==s]) for s in sites},
        'pairs': pairs,
        'transport_sha256': file_hash(transport),
        'profile_failures': read_json(args.workspace/'profiles'/'failures.json'),
    }
    paths, arrays = [], []
    for utc in sorted({r['utc'] for r in full}):
        window = datetime.fromisoformat(utc).replace(tzinfo=None).isoformat(timespec='minutes').replace(':', '')
        with np.load(args.workspace/'profiles'/(window+'.npz')) as p:
            paths.append(utc)
            arrays.append(p['areas'])
    write_npz(args.output/'geos-cf-layer-integrals.npz', timestamps_utc=paths, layer_edges_m=HEIGHT_BINS,
              latitude_deg=NATIVE_LAT, longitude_deg=NATIVE_LON, PM25_vertical_integrals=np.stack(arrays))
    data['comparison']['profiles_archive_sha256'] = file_hash(args.output/'geos-cf-layer-integrals.npz')
    write_json(args.output/'results.json', data)
    print({k: v for k, v in data['comparison'].items() if k not in ('pairs', 'profile_failures')}, flush=True)


if __name__ == '__main__':
    main()
