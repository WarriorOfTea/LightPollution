"""Collect diagnostic protocols/scores and make scientific comparison plots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113')
    args = parser.parse_args()
    base = args.root
    rows = [read_json(base/f'scattering/aod-{tau:.2f}-alt-{alt}.json')
        for tau in [.02, .10, .30] for alt in [10, 45, 90]]
    columns = ['case_id', 'AOD_550', 'altitude_deg', 'L1', 'L2', 'L3',
        'increase_2_over_1_percent', 'increase_2_over_1_95_halfwidth_pp',
        'increase_3_over_2_percent', 'increase_3_over_2_95_halfwidth_pp', 'MC_first_order_z']
    with (base/'scattering/score-table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            a, b = row['cumulative_two_over_one'], row['cumulative_three_over_two']
            writer.writerow([row['case_id'], row['atmosphere']['aerosol_optical_depth'], row['altitude_deg'],
                *row['exact_order_radiance'], a['relative_increment']*100, a['standard_error']*196,
                b['relative_increment']*100, b['standard_error']*196, row['first_order_z']])
    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for tau, color in zip([.02, .10, .30], ['#126782', '#e49b0f', '#a83232'], strict=True):
        group = [r for r in rows if r['atmosphere']['aerosol_optical_depth'] == tau]
        for ax, key in zip(axes, ['cumulative_two_over_one', 'cumulative_three_over_two'], strict=True):
            ax.errorbar([r['altitude_deg'] for r in group],
                [r[key]['relative_increment']*100 for r in group],
                yerr=[r[key]['standard_error']*196 for r in group],
                marker='o', capsize=4, label=f'AOD={tau:.2f}', color=color)
            ax.set_xlabel('Высота взгляда, градусы')
            ax.set_xticks([10, 45, 90])
            ax.set_ylabel('Добавка к предыдущей сумме, %')
            ax.legend()
    axes[0].set_title('Первый → сумма до второго')
    axes[1].set_title('Сумма до второго → до третьего')
    fig.suptitle('Одинаковая конечная площадка, 550 нм; интервалы ≈95% Монте-Карло')
    fig.savefig(base/'scattering-orders.png', dpi=160)
    plt.close(fig)

    city = read_json(base/'city/results.json')['variants']
    moon = read_json(base/'moon/primary-results.json')['records']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    names = ['Базовая', 'Дневная AOD', 'Дневной g', 'AOD + g']
    axes[0].bar(names, [(r['periods']['2017-11']['fixed_C_model_over_observed_flux']-1)*100 for r in city], color='#126782')
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].set_ylabel('Модель / измерение − 1, % потока')
    axes[0].set_title('Город: 1 контрольная ночь, C фиксирован')
    axes[0].tick_params(axis='x', labelrotation=15)
    for key, label, color in [('default_atmosphere', 'Стандартная атмосфера', '#126782'),
                               ('site_extinction', 'Площадочная экстинкция', '#bd5725')]:
        axes[1].scatter([r['separation_deg'] for r in moon],
            [r['predictions'][key]['residual_mag_model_minus_observed'] for r in moon],
            label=label, color=color, s=28)
    for row in moon:
        low, high = sorted(row['predictions']['site_extinction']['natural_21_22_mag_result_span'])
        axes[1].vlines(row['separation_deg'], low-row['v_mag_arcsec2'], high-row['v_mag_arcsec2'],
            color='#bd5725', alpha=.4, linewidth=1)
    axes[1].axhline(0, color='black', linewidth=.8)
    axes[1].set_ylabel('Модель − измерение, mag/arcsec²')
    axes[1].set_xlabel('Разделение с Луной, градусы')
    axes[1].set_title('Луна: 12 измерений / 3 ночи')
    axes[1].legend(fontsize=9)
    fig.suptitle('Условные полевые проверки; справа линии — сценарий фона 21–22 mag, не доверительный интервал')
    fig.savefig(base/'observed-comparisons.png', dpi=160)
    plt.close(fig)

    stars = read_json(base/'spectra/stellar-continuum.json')['stars']
    stars = [r for r in stars if r['status'] == 'scored_conditional_component']
    fig, ax = plt.subplots(figsize=(10.5, 3.8), constrained_layout=True)
    x = np.arange(len(stars))
    for shift, key, label in [(-.18, 'photopic', 'Фотопический отклик'), (.18, 'scotopic', 'Скотопический отклик')]:
        ax.bar(x+shift, [r['continuum_over_measured_response_minus_one'][key]*100 for r in stars],
            width=.35, label=label)
    ax.set_xticks(x, [r['name'] for r in stars])
    ax.set_ylabel('Континуум / измеренный SED − 1, %')
    ax.axhline(0, color='black', linewidth=.8)
    ax.legend()
    ax.set_title('Спектральное приближение при точных B−V и V; это не оценка воспринимаемого цвета')
    fig.savefig(base/'stellar-spectra.png', dpi=160)
    plt.close(fig)

    catalogue = read_json(ROOT/'artifacts/benchmark-data-audit-20261112/candidate-cases.json')
    statuses = []
    for case in catalogue['cases']:
        state = {'id': case['id'], 'site': case['site'], 'full_scene_validated': False,
            'status': 'not_run', 'metrics': None, 'reason': case['missing']}
        if case['id'] == 'B01':
            state.update(status='conditional_two_night_development_diagnostic', metrics='city/summary.json',
                reason='Daytime atmosphere 5.29–6.46h from night; one prior-inspected check night, monthly source and effective-V bridge.')
        elif case['id'] == 'B04':
            state.update(status='conditional_isolated_moon_component_12_records_3_nights', metrics='moon/summary.json',
                reason='Published angles isolate lunar block; natural background and no per-row cloud QA remain.',
                NASA_required_for_this_isolated_component=False)
        elif case['id'] == 'B12':
            state.update(status='controlled_internal_plus_independent_slab_reference', metrics='scattering/summary.json',
                reason='A new independently discretised slab is used; libRadtran/IPRT comparison remains not run.')
        statuses.append(state)
    write_json(base/'case-status.json', {'catalogue_source': 'benchmark-catalogue0.1', 'cases': statuses,
        'additional_component_tests': ['spectra/summary.json', 'aggregation/summary.json'],
        'full_scene_or_eye_or_real_camera_scores_available': False})
    summaries = {name: read_json(base/name/'summary.json') for name in ['scattering', 'city', 'moon', 'spectra', 'aggregation']}
    official_paths = summaries['scattering']['total_paths_including_tail_repeat_and_timing']+524288
    ledger = {'official_final_MC_paths': official_paths, 'official_final_MC_solver_calls': 19,
        'earlier_full_MC_pass_paths': 28311552,
        'earlier_pass_status': 'Physical values reproduced; timing RSS method rejected because ru_maxrss inherited parent peak.',
        'pilot_MC_paths': 131072, 'benchmark_MC_paths_all_passes_excluding_unit_tests': official_paths+28311552+131072,
        'unit_test_random_paths_not_counted_as_benchmark': True}
    write_json(base/'execution-ledger.json', ledger)
    summary = {'benchmark_version': 'LP-core-diagnostics-1', 'production_version': '0.10.0',
        'git_base_commit': 'e69261f', 'identity_policy': 'Exact scientific/evaluator file hashes are in each protocol; final Git commit contains this artifact.',
        'groups_evaluated': len(summaries), 'measured_photometry_rows': 30, 'distinct_observed_nights': 5,
        'training_photometry_rows': 9, 'check_photometry_rows': 21,
        'all_field_checks_conditional': True, 'measured_stellar_spectra': 7, 'measured_laboratory_lamps': 1,
        'global_accuracy_score': None, 'global_score_unavailable_reason': 'Different quantities, incomplete QA and missing eye/camera truth; no common valid aggregate.',
        'group_summaries': summaries, 'execution': ledger,
        'summarizer_sha256': file_hash(Path(__file__))}
    write_json(base/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['benchmark_version', 'groups_evaluated', 'measured_photometry_rows',
        'distinct_observed_nights', 'global_accuracy_score']}))


if __name__ == '__main__':
    main()
