"""Verify and summarize the controlled vertical-profile / angular-boundary study."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.models.boundary_orders import paired_ratio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/city-emission-review-20261113')
    args = parser.parse_args()
    base = args.output
    protocol = json.loads((base/'protocol.json').read_text())
    results, checks = {}, []
    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)
    for name, digest in protocol['code_sha256'].items():
        check('code:'+name, file_hash(ROOT/name) == digest)
    for case in protocol['cases']:
        r = json.loads((base/'cases'/(case['id']+'.json')).read_text())
        results[(case['location'], case['altitude_deg'], case['profile'])] = r
        n = r['settings']['samples']
        mean = np.asarray(r['all_order_radiance'])
        cov = np.asarray(r['mean_covariance'])
        check(case['id']+':identity', r['case'] == case and r['identity']['code'] == protocol['code_sha256'])
        check(case['id']+':sample_budget', n == protocol['samples_per_case'])
        check(case['id']+':seed', r['settings']['seed'] == case['seed'])
        check(case['id']+':no_cutoff', not r['finite_scattering_order_truncation'] and not r['safety_limit_hit'])
        check(case['id']+':outcomes', sum(r['path_outcomes'].values()) == n)
        check(case['id']+':support', sum(r['ground_support_counts'].values()) == r['path_outcomes']['ground'])
        check(case['id']+':finite_positive', np.all(np.isfinite(mean)) and np.all(mean > 0))
        check(case['id']+':bounded', np.all(mean <= np.asarray(r['bounded_scores'])))
        check(case['id']+':order_closure', np.allclose(np.sum(r['category_mean_radiance'], axis=0), mean, rtol=1e-12))
        check(case['id']+':angular_closure', np.allclose(np.sum(r['emission_bin_radiance'], axis=0), mean, rtol=1e-12))
        check(case['id']+':covariance', np.min(np.linalg.eigvalsh(cov)) >= -1e-12*max(1., np.max(cov)))
        check(case['id']+':interval_support', all(x is not None for x in r['normal_95_intervals']))
    check('all_expected_cases', len(results) == len(protocol['cases']) == 24)
    check('independent_profile_seeds', len({c['seed'] for c in protocol['cases']}) == 4)
    rows, angular = [], []
    for (location, altitude, profile), r in results.items():
        ref = results[(location, altitude, 'exponential_1200m')]
        for i, label in enumerate(r['labels']):
            value, baseline = r['all_order_radiance'][i], ref['all_order_radiance'][i]
            ratio = value/baseline
            # Profile seeds are independent. Same-profile identity has zero error.
            se = 0. if profile == 'exponential_1200m' else np.sqrt(
                (r['all_order_standard_error'][i]/baseline)**2+
                (value*ref['all_order_standard_error'][i]/baseline**2)**2)
            rows.append({'location': location, 'altitude_deg': altitude, 'profile': profile,
                'angular_model': label, 'radiance_proxy': value,
                'radiance_standard_error': r['all_order_standard_error'][i],
                'effective_score_samples': r['effective_score_samples'][i],
                'change_vs_exponential_pct': 100*(ratio-1),
                'change_normal_95_lo_pct': float(100*(ratio-1-1.96*se)),
                'change_normal_95_hi_pct': float(100*(ratio-1+1.96*se)),
                'normal_interval_scope': 'Approximate marginal MC only; independent seeds across profiles; no field error.'})
        comparison = paired_ratio(r, 'fitted_0_60_hold_off', 'empirical_hold_off')
        angular.append({'location': location, 'altitude_deg': altitude, 'profile': profile,
            'fitted_over_empirical': comparison,
            'fitted_over_near': r['ratios_to_near']['fitted_0_60_hold_off'],
            'empirical_over_near': r['ratios_to_near']['empirical_hold_off']})
    with (base/'profile-comparisons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    chosen = [r for r in rows if r['angular_model'] == 'fitted_0_60_hold_off']
    shifted = [r for r in chosen if r['profile'] != 'exponential_1200m']
    summary = {'cases_completed': len(results), 'samples_per_case': protocol['samples_per_case'],
        'paths': sum(r['settings']['samples'] for r in results.values()), 'angular_estimates': len(rows),
        'all_order_runtime_s': sum(r['runtime_seconds'] for r in results.values()),
        'peak_rss_mib_process': max(r['peak_rss_kib_process'] for r in results.values())/1024,
        'minimum_effective_score_samples': min(min(r['effective_score_samples']) for r in results.values()),
        'eligible_normal_intervals': sum(x is not None for r in results.values() for x in r['normal_95_intervals']),
        'maximum_order_observed': max(r['max_scattering_order_seen'] for r in results.values()),
        'hybrid_profile_change_pct_range': [min(r['change_vs_exponential_pct'] for r in shifted), max(r['change_vs_exponential_pct'] for r in shifted)],
        'angular_comparisons': angular, 'hybrid_rows': chosen,
        'new_measured_profiles': 0, 'new_independent_field_nights': 0, 'production_solver_changed': False,
        'mc_error_is_not_field_accuracy': True}
    write_json(base/'summary.json', summary)
    write_json(base/'integrity-verification.json', {'checks': checks,
        'passed': sum(r['passed'] for r in checks), 'total': len(checks)})
    titles = {'mixed_0_1000m': 'AOD in 0–1 km', 'elevated_2000_3000m': 'AOD in 2–3 km',
              'half_low_half_elevated': 'Half low / half elevated'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for axis, location in zip(axes, ('ucm', 'north')):
        for j, (profile, title) in enumerate(titles.items()):
            records = sorted((r for r in chosen if r['location'] == location and r['profile'] == profile), key=lambda r:r['altitude_deg'])
            values = np.array([r['change_vs_exponential_pct'] for r in records])
            errors = np.array([r['change_normal_95_hi_pct']-r['change_vs_exponential_pct'] for r in records])
            axis.errorbar(np.arange(3)+(j-1)*.13, values, yerr=errors, marker='o', capsize=3, label=title)
        axis.axhline(0, color='black', lw=.7)
        axis.set(xticks=np.arange(3), xticklabels=['10°','45°','90°'], xlabel='Viewing altitude',
            ylabel='Change vs exponential profile (%)', title='Conditional UCM location' if location == 'ucm' else 'North of Madrid, looking south')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same AOD = 0.10, SSA = 0.9, g = 0.8; all orders at 550 nm\n95% bars: numerical MC only; illustrative profiles, no field validation', fontsize=11)
    fig.savefig(base/'vertical-profile-sensitivity.png', dpi=170)
    plt.close(fig)
    table = ['| Место / высота взгляда | Аэрозоль0–1км | Аэрозоль2–3км | Половина в каждом слое |',
             '|---|---|---|---|']
    for location in ('ucm','north'):
        for altitude in (10,45,90):
            cells = []
            for profile in titles:
                r = next(x for x in chosen if (x['location'], x['altitude_deg'], x['profile']) == (location, altitude, profile))
                half = (r['change_normal_95_hi_pct']-r['change_normal_95_lo_pct'])/2
                cells.append(f"{r['change_vs_exponential_pct']:+.2f}% ±{half:.2f}п.п.")
            table.append('| '+('UCM' if location=='ucm' else 'Север')+f' / {altitude}° | '+' | '.join(cells)+' |')
    (base/'profile-table.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('angular_comparisons','hybrid_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
