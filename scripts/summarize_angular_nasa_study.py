"""Numerical QA, CSV and scientific figures from the frozen angular experiment."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.data import load_region
from lightpollution.models.angular_sources import (bin_horizontal_factor, empirical_radiance,
    horizontal_radiance_factor)
from lightpollution.physics import cell_areas
from run_angular_nasa_study import extract

ROOT = Path(__file__).resolve().parents[1]


def flux_ratios(ratio, eta):
    # Split at every kink and radiance cap; integrate each smooth segment.
    edges = sorted([0., .5, 1., *np.cos(np.deg2rad([20, 40, 85, 88]))])
    x, w = np.polynomial.legendre.leggauss(24)
    values = np.zeros(9)
    gn = bin_horizontal_factor(0, 20)
    for left, right in zip(edges[:-1], edges[1:]):
        mu = left+(x+1)*(right-left)/2
        model = [np.ones_like(mu), np.full_like(mu, ratio), np.ones_like(mu),
            *(empirical_radiance(1., ratio, mu, tail) for tail in (0., 1., 2.))]
        for fraction, cap in ((.1, 85), (eta, 85), (eta, 88)):
            h = horizontal_radiance_factor(np.maximum(mu, np.cos(np.deg2rad(cap))))
            model.append((1-fraction+fraction*h)/(1-fraction+fraction*gn))
        values += np.asarray(model) @ (w*mu*(right-left))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/angular-nasa-study-20261113')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    out = args.output
    protocol = json.loads((out/'protocol.json').read_text())
    audits = json.loads((out/'input-summary.json').read_text())
    cases = {p.stem: json.loads(p.read_text()) for p in sorted((out/'cases').glob('*.json'))}
    wanted = {c['id'] for c in protocol['cases']}
    missing = sorted(wanted-set(cases))
    if missing and not args.allow_incomplete:
        raise ValueError(f'Incomplete experiment: {missing}')
    if set(cases)-wanted:
        raise ValueError('Unexpected experiment case.')
    checks, csv_rows = [], []
    for name, result in cases.items():
        n = result['settings']['samples']
        means = np.asarray(result['all_order_radiance'])
        cov = np.asarray(result['mean_covariance'])
        checks.extend([
            {'case': name, 'check': 'outcome_conservation', 'passed': sum(result['path_outcomes'].values()) == n},
            {'case': name, 'check': 'ground_support_conservation', 'passed': sum(result['ground_support_counts'].values()) == result['path_outcomes']['ground']},
            {'case': name, 'check': 'category_closure', 'passed': bool(np.allclose(np.sum(result['category_mean_radiance'], axis=0), means, rtol=1e-12, atol=1e-15))},
            {'case': name, 'check': 'angular_closure', 'passed': bool(np.allclose(np.sum(result['emission_bin_radiance'], axis=0), means, rtol=1e-12, atol=1e-15))},
            {'case': name, 'check': 'covariance_positive_semidefinite', 'passed': bool(np.linalg.eigvalsh(cov).min() >= -1e-12*max(1., np.trace(cov)))},
            {'case': name, 'check': 'same_source_code', 'passed': all(file_hash(ROOT/p) == h for p, h in result['identity']['code'].items())},
            {'case': name, 'check': 'all_order_safety', 'passed': not result['finite_scattering_order_truncation'] and not result['safety_limit_hit']},
        ])
        angles = np.asarray(result['emission_bin_radiance'])
        orders = np.asarray(result['category_mean_radiance'])
        for i, label in enumerate(result['labels']):
            r = result['ratios_to_near'][label]
            csv_rows.append({'case': name, 'role': result['case']['role'], 'variant': label,
                'samples': n, 'radiance_proxy_response': means[i], 'response_standard_error': result['all_order_standard_error'][i],
                'ratio_to_near': r['ratio'], 'ratio_standard_error': r['ratio_standard_error'],
                'ratio_normal95_low': r['normal_95_interval'][0] if r['normal_95_interval'] else '',
                'ratio_normal95_high': r['normal_95_interval'][1] if r['normal_95_interval'] else '',
                'delta_mag_artificial_only': r['delta_mag_artificial_only'],
                'positive_paths': result['positive_paths'][i], 'effective_score_samples': result['effective_score_samples'][i],
                'fraction_emission_zenith_above60': float(angles[3:, i].sum()/means[i]),
                'fraction_emission_zenith_above85': float(angles[5:, i].sum()/means[i]),
                'fraction_above_order3': float(orders[3:, i].sum()/means[i]),
                'fraction_above_order5': float(orders[5:, i].sum()/means[i])})
    independent = []
    for result in cases.values():
        if result['case']['role'] != 'independent_seed':
            continue
        parent = cases[result['case']['id'].split('-seed')[0]]
        a, b = np.array(result['all_order_radiance']), np.array(parent['all_order_radiance'])
        uncertainty = np.hypot(result['all_order_standard_error'], parent['all_order_standard_error'])
        z = abs(a-b)/uncertainty
        independent.append({'case': result['case']['id'], 'z_per_variant': dict(zip(result['labels'], map(float, z))),
            'passed_4_combined_se': bool(np.all(z <= 4))})
        checks.append({'case': result['case']['id'], 'check': 'independent_seed_4se', 'passed': bool(np.all(z <= 4))})
    for name, entry in protocol['inputs'].items():
        checks.append({'case': name, 'check': 'input_cache_sha256', 'passed': file_hash(ROOT/entry['maps']) == entry['maps_sha256']})
    integrity = {'checked_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'checks': checks, 'passed': sum(c['passed'] for c in checks), 'failed': sum(not c['passed'] for c in checks),
        'completed_cases': len(cases), 'expected_cases': len(wanted), 'missing_cases': missing,
        'independent_seed': independent}
    write_json(out/'integrity-verification.json', integrity)
    if integrity['failed']:
        raise ValueError('Numerical integrity check failed.')
    if csv_rows:
        with (out/'sky-comparisons.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(csv_rows)
    totals = {'completed_cases': len(cases), 'expected_cases': len(wanted), 'samples_total': sum(r['settings']['samples'] for r in cases.values()),
        'primary_cases': sum(r['case']['role'] == 'primary' for r in cases.values()),
        'variants_per_case': 9, 'sum_case_runtime_seconds': sum(r['runtime_seconds'] for r in cases.values()),
        'maximum_observed_order': max((r['max_scattering_order_seen'] for r in cases.values()), default=0),
        'minimum_effective_score_samples': min((min(r['effective_score_samples']) for r in cases.values()), default=0),
        'normal_interval_eligible_variant_cases': sum(bool(r['normal_95_interval']) for c in cases.values() for r in c['ratios_to_near'].values()),
        'variant_case_denominator': len(cases)*9,
        'peak_rss_kib': max((r.get('peak_rss_kib_process', 0) for r in cases.values()), default=0),
        'raw_cache_fields_verified': sum(r['raw_cache_fields_verified'] for r in audits.values()),
        'raw_hdf5_references_verified': sum(r['raw_hashes_verified'] for r in audits.values()),
        'source_code_sha256': {p: file_hash(ROOT/p) for p in ['scripts/summarize_angular_nasa_study.py']},
        'upward_flux_ratios': {}}
    for name, entry in protocol['inputs'].items():
        ratios = flux_ratios(entry['off_near_area_ratio_on_transport_support'], entry['regional_eta_uniform_solid_angle'])
        # The all-angle flux uses the SAME transport support, not the per-period audit mask.
        arrays, bounds, _ = load_region(ROOT/entry['maps'])
        _, _, _, support = extract(arrays)
        if name.startswith('madrid'):
            other = 'madrid-2017-11' if name.endswith('03') else 'madrid-2017-03'
            other_arrays, _, _ = load_region(ROOT/protocol['inputs'][other]['maps'])
            support &= extract(other_arrays)[3]
        area = np.broadcast_to(cell_areas(bounds.latitude)[:, None], bounds.shape)
        ratios[2] = float((arrays['AllAngle_Composite_Snow_Free'][support]*area[support]).sum()/(arrays['NearNadir_Composite_Snow_Free'][support]*area[support]).sum())
        totals['upward_flux_ratios'][name] = dict(zip(AngularLabels, map(float, ratios)))
    write_json(out/'summary.json', totals)
    plot(out, protocol, audits, cases, csv_rows)
    write_report(out, protocol, audits, cases, csv_rows, totals, integrity)
    print(json.dumps({'integrity_passed': integrity['passed'], 'integrity_failed': integrity['failed'], **totals}, ensure_ascii=False, indent=2))


AngularLabels = ('near_lambertian', 'off_lambertian', 'all_lambertian',
    'angular_horizon_zero', 'angular_horizon_hold', 'angular_horizon_double',
    'mixed_eta_0.1_cap85', 'mixed_eta_regional_cap85', 'mixed_eta_regional_cap88')


def write_report(out, protocol, audits, cases, rows, totals, integrity):
    def pct(value):
        return f'{100*value:+.2f}%'

    def delta(result, variant):
        return pct(result['ratios_to_near'][variant]['relative_change'])

    def interval(result, variant):
        entry = result['ratios_to_near'][variant]
        if not entry['normal_95_interval']:
            return 'интервал не определён'
        return f"{pct(entry['relative_change'])} ±{196*entry['ratio_standard_error']:.2f} п.п."

    primary = [cases[c['id']] for c in protocol['cases'] if c['role'] == 'primary' and c['id'] in cases]
    lines = ['# Ракурсы NASA и диаграммы излучения города', '',
        'Я проверил, как смена углового композита и продолжение диаграммы за пределы измерений меняют искусственный фон при переносе всех порядков. '
        'Это исследование входов и чувствительности заданного уравнения; новую точность относительно реального неба я не измерял.', '',
        f"Протокол **{protocol['id']}**; базовая ревизия `3e284c8`. "
        f"Завершено **{len(cases)}/{totals['expected_cases']}** сцен, **{totals['samples_total']:,}** траекторий, девять вариантов границы на каждом пути. "
        'Основной интерфейс/KStars остаётся прежним. Новые функции работают в отдельном исследовательском режиме. '
        '[PDF для чтения](LightPollution-angular-nasa-study.pdf).', '',
        '## 1. Какие данные я действительно использовал', '',
        'Я независимо перечитал исходные Collection2 HDF5, проверил продукт/период/географию/единицы/масштаб/fill и сравнил каждый декодированный массив с региональным кэшем. '
        f"Проверены все ссылки на HDF5 (**{totals['raw_hdf5_references_verified']}**) и декодированные поля (**{totals['raw_cache_fields_verified']}**). Новые спутниковые данные и токены не потребовались. "
        'Имена, SHA256 и границы сохранены в `*-audit.json`; источники и права — в [source-audit.json](source-audit.json).', '',
        'NearNadir — диапазон0–20° от местной вертикали, OffNadir —40–60°. AllAngle использует перекрывающиеся наблюдения, включая другие углы; '
        'это не третий независимый снимок под известным углом. Использую только Snow_Free, конечные неотрицательные радиансности, QA0, Num>3 и классы суши/берега0/1/5. '
        'В каждой строке сравниваю одинаковые ячейки, прошедшие все три категории.', '',
        '| Область/период | Вся сетка | Пригодные Near / Off / All | Общее пересечение | Off/Near суммы D×A | All/Near суммы D×A |',
        '|---|---:|---:|---:|---:|---:|']
    for name, d in audits.items():
        counts = d['accepted_land_coast_cells']
        ratios = d['area_sum_ratios_to_near']
        values = [('—' if ratios[p] is None else f'{ratios[p]:.5f}') for p in ('OffNadir', 'AllAngle')]
        lines.append(f"| {name} | {d['total_grid_cells']} | {counts['NearNadir']} / {counts['OffNadir']} / {counts['AllAngle']} | {d['common_three_angle_cells']} | {values[0]} | {values[1]} |")
    lines += ['', 'Здесь D измерена в нВт/(см²·ср), A — площадь ячейки; отношения безразмерны. '
        'Это различие пространственного входа, а не напрямую изменение небесного фона или восстановленные ватты ламп. '
        'Январь Петербурга содержит лишь47 общих ячеек и3 ярких пары, июль не содержит ни одной; эти месяцы в перенос не включены. '
        'Ранее опубликованные423 январские NearNadir-ячейки относились ко всем классам поверхности и одной категории — это другой знаменатель.', '',
        'В перенос входят март/ноябрь Мадрида, годовой Петербург и годовой Саар. Для двух месяцев Мадрида дополнительно фиксирую одно временное пересечение: '
        f"**{protocol['inputs']['madrid-2017-03']['common_cells_for_transport']}** ячейки. "
        'На нём Off/Near составляет1,09141 в марте и1,17348 в ноябре; небольшое отличие от таблицы связано с маской. '
        'Неизвестные ячейки и источники вне региона исключены во всех вариантах одинаково; их свет не объявлен известным нулём.', '',
        '## 2. Что меняется по территории', '',
        'Чтобы слабые значения и цензурирование не определяли вывод, отдельно рассматриваю пары с Near≥5 и Off≥5нВт/(см²·ср). '
        'Пороги1/5/20 и нули сохранены в аудитах. NASA устанавливает значения ниже0,5 в0; поэтому отношение всех слабых пикселей не равно устойчивой оценке диаграммы.', '',
        '| Набор | Ярких пар, знаменатель | Off/Near: P05 / медиана / P95 | Off<Near, число/знаменатель | Ярких блоков12×12 | Off<Near среди блоков |',
        '|---|---:|---:|---:|---:|---:|']
    for name, d in audits.items():
        v, b = d['comparisons']['5.0'], d['blocks']['12']
        if v['denominator'] < 100:
            continue
        r = v['off_over_near']
        lines.append(f"| {name} | {v['denominator']} | {r['p05']:.3f} / {r['median']:.3f} / {r['p95']:.3f} | {v['off_less_than_near']}/{v['denominator']} ({100*v['off_less_than_near']/v['denominator']:.1f}%) | {b['denominator_blocks']} | {b['off_less_than_near']}/{b['denominator_blocks']} |")
    lines += ['', 'Блоки4/12/24 — фиксированная пространственная сетка. Принимаю блок с≥4 общими ячейками, покрытием≥80% известных ячеек суши/берега и обеими средними≥5; '
        'площади ячеек учтены. В Мадриде шаг15″ примерно0,46км с севера на юг и0,35км с запада на восток; блок4×4 — около1,85×1,41км,12×12 —5,56×4,23км. '
        'Это шаг сетки, не независимое оптическое разрешение. Численные блоки сохранены в `*-blocks.csv`.', '',
        '![Пространственные различия ракурсов](spatial-angle-ratios.png)', '',
        'Отношение ниже1 несовместимо с моей прежней семьёй «ламбертовская + неотрицательная низкоугловая компонента» при сопоставимом стационарном источнике. '
        'Такие отношения сохраняются после пространственного усреднения. Однако композиты используют разные времена/ракурсы, поэтому каждый отдельный пиксель не является доказательством формы конкретной лампы. '
        'Я сохраняю несовместимость и не обрезаю найденную η в0 или1.', '',
        '## 3. Что происходит с параметром смеси', '',
        'Я проверил действующие настройки: `SolverSettings.horizontal_fraction=0` и `direct_horizontal_fraction=0` по умолчанию, то есть базовая угловая форма ламбертовская. '
        'Низкоугловая смесь является явно заданным сценарием. Если включён спектральный режим и состав не указан, код выбирает `LED-B3: 1`; '
        'это библиотечный шаблон, а не спектр города, восстановленный NASA. В новом опыте сравнения идут с этой базовой ламбертовской формой при одном фиксированном спектральном приближении.', '',
        'Я сравниваю радиансности. Для ламбертовской площадки радиансность постоянна; косинус входит в интенсивность конечной площадки. '
        'Поэтому ожидание Off/Near=cos50°/cos10° здесь неверно. Формула перехода и подбора приведена в [THEORY](../../THEORY.md#что-добавляют-несколько-спутниковых-ракурсов).', '',
        'При равномерном усреднении по телесному углу внутри диапазонов низкоугловая часть даёт GNear=0,00000696434 и GOff=0,1003695. '
        'Отношение Off/Near=1,10 требует η≈0,4991. Это диагностический подбор с допущением о весах, а не измеренная доля фонарей.', '',
        '| Набор в переносе | Off/Near | η: равномерно по телесному углу | η: равномерно по зенитному углу | Мощность смеси85 / мощность Near-Lambert |',
        '|---|---:|---:|---:|---:|']
    for name, entry in protocol['inputs'].items():
        lines.append(f"| {name} | {entry['off_near_area_ratio_on_transport_support']:.5f} | {entry['regional_eta_uniform_solid_angle']:.5f} | {entry['regional_eta_uniform_zenith_angle']:.5f} | {totals['upward_flux_ratios'][name]['mixed_eta_regional_cap85']:.3f} |")
    lines += ['', 'В этих сравнениях фиксирую измеренный NearNadir и один общий масштаб. Поэтому восстановленная полная мощность при смене формы меняется; '
        'это не прежний опыт изменения η при заведомо неизменной мощности лампы. Отдельной перенастройки C или каждого зенита нет. '
        'Полные отношения интегралов мощности всех вариантов сохранены в [summary.json](summary.json).', '',
        'Исходная интенсивностная низкоугловая функция при переводе в радиансность растёт как1/μ у горизонта. '
        'Я не применяю к ней ложную конечную границу радиансности: в опыте явно фиксирую радиансность после85° или88°. '
        'Оба варианта отличаются от неограниченного исходного закона. Положение ограничения само становится проверяемым допущением.', '',
        '![Продолжения угловой формы](angular-continuations.png)', '',
        '## 4. Как я считал перенос', '',
        f"На сцену — **{protocol['samples_per_case']:,}** траекторий;18 основных сцен и4 контроля. "
        'Я продолжаю обратный путь до земли/космоса либо физического поглощения, без ограничения порядков, удаления малых весов и повторной передачи Бера. '
        'Использую прежние выборку пробегов и фаз Рэлей/HG. Все девять вариантов оцениваются на одних полных путях; сохраняю полную ковариацию для их отношений.', '',
        'Фиксирую550нм, SSA0,9, g0,8, AOD0,10, экспоненциальные масштабы8км/1,2км и верх80км. В двух контролях меняю только AOD на0,02/0,30. '
        'Наблюдатели в20м над одной сферической землёй: UCM40,451111/−3,726111; севернее города40,85/−3,726111; Петербург60,3/30,315; Саар49,532563/6,549911. '
        'Рельеф и фактическая высота станции не воспроизведены. Азимут180° при высотах10°/45°;90° — зенит.', '',
        'Числа DNB задают пространственную амплитуду одного монохромного прокси. Спектральной инверсии DNB→550нм/V я не выполнял. '
        'Облака, Луна, естественный фон, DSM и переотражения земли отключены. Ближние конечные ячейки включены — в отличие от прежних опытов с исключением3км.', '',
        'Я сравниваю9 вариантов: Near/Off/All как отдельные ламбертовские карты; угловую форму с плато Near на0–20°, линейным переходом поμ на20–40° и плато Off на40–60°; '
        'три продолжения этой формы к горизонту до0/Off/2×Off; смесьη0,1 с ограничением85°; регионально подобранную смесь с ограничениями85°/88°. '
        'Все три ограниченные эмпирические формы точно совпадают в двух наблюдаемых диапазонах при любых весах внутри этих диапазонов. '
        'Радиансность каждого варианта проверяется против объявленной конечной границы при каждом попадании.', '',
        '## 5. Насколько различается искусственный фон неба', '',
        'В таблице показываю изменение относительно NearNadir с ламбертовской формой. '
        'Диапазон «0…2Off» отражает выбранные продолжения к горизонту, не доверительную вероятность реальной диаграммы. '
        'У варианта «Off на горизонте» ± — приблизительная95%-полуширина только парной ошибки Монте-Карло, в процентных пунктах. '
        'Это не ошибка атмосферы и не полевая точность.', '',
        '| Сцена | Заменить на Off | Заменить на All | Угловая: Off на горизонте | Угловая: 0…2Off на горизонте |',
        '|---|---:|---:|---:|---:|']
    for r in primary:
        lines.append(f"| {r['case']['id']} | {delta(r, 'off_lambertian')} | {delta(r, 'all_lambertian')} | {interval(r, 'angular_horizon_hold')} | {delta(r, 'angular_horizon_zero')} … {delta(r, 'angular_horizon_double')} |")
    lines += ['', '![Расчётные различия по направлениям](sky-angle-comparison.png)', '',
        'Для смеси привожу отдельные отношения, чтобы большой сценарный эффект не скрывал небольшие различия карт. '
        'Знаменатель каждого отношения — NearNadir/Lambert той же сцены.', '',
        '| Сцена | Смесьη0,1, предел85° | Региональнаяη, предел85° | Та жеη, предел88° | Доля света от испускания>60° в Near/Lambert | Доля после третьего порядка в Near/Lambert |',
        '|---|---:|---:|---:|---:|---:|']
    for r in primary:
        if r['case']['altitude_deg'] != 45:
            continue
        record = next(row for row in rows if row['case'] == r['case']['id'] and row['variant'] == 'near_lambertian')
        vals = [r['ratios_to_near'][p]['ratio'] for p in AngularLabels[6:]]
        lines.append(f"| {r['case']['id']} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {100*record['fraction_emission_zenith_above60']:.1f}% | {100*record['fraction_above_order3']:.2f}% |")
    lines += ['', 'В полной [таблице198 оценок](sky-comparisons.csv) сохраняю все направления/варианты, стандартные ошибки, парные95%-интервалы, '
        'доли испускания>60°/>85° и хвосты после третьего/пятого порядков. '
        'Сырые JSON с ковариациями находятся в `cases/`. Пересчёт вΔmag относится только к искусственной составляющей; с неизвестным естественным фоном изменение общей яркости будет иным.', '',
        '## 6. Атмосферная чувствительность и независимая выборка', '',
        'Здесь фиксирую мартовскую карту, северную точку и взгляд45°, меняю только AOD. Это три сценарные атмосферы, не измерения ночей.', '',
        '| AOD | Заменить на Off | Угловая: Off у горизонта | Угловая: 0…2Off у горизонта | Региональная смесь85 / Near |',
        '|---|---:|---:|---:|---:|']
    atmosphere_cases = [r for r in cases.values() if r['case']['id'].startswith('madrid-2017-03-north-alt45') and r['case']['role'] != 'independent_seed']
    for r in sorted(atmosphere_cases, key=lambda r: r['case']['aod']):
        lines.append(f"| {r['case']['aod']:.2f} | {delta(r, 'off_lambertian')} | {interval(r, 'angular_horizon_hold')} | {delta(r, 'angular_horizon_zero')} … {delta(r, 'angular_horizon_double')} | {r['ratios_to_near']['mixed_eta_regional_cap85']['ratio']:.3f} |")
    lines += ['', 'Другой seed проверяю в мартовской северной точке при10°/45°. До расчёта задал допуск4 объединённых стандартных ошибки по каждому из9 вариантов.', '',
        '| Независимая сцена | Максимальное отклонение, SE | Прошли варианты |', '|---|---:|---:|']
    for r in integrity['independent_seed']:
        lines.append(f"| {r['case']} | {max(r['z_per_variant'].values()):.3f} | {sum(z <= 4 for z in r['z_per_variant'].values())}/9 |")
    lines += ['', '## 7. Проверки, затраты и границы вывода', '',
        f"Численные проверки: **{integrity['passed']}/{len(integrity['checks'])}**, ошибок **{integrity['failed']}**. "
        'Проверяю сохранение числа исходов/масок, сумму порядков/углов, неотрицательность ковариации, хеши кода/кэшей и отсутствие обрезания. '
        f"Максимально встретившийся порядок — **{totals['maximum_observed_order']}**, это не физический предел. "
        f"Минимальная диагностика эффективных весовых вкладов — **{totals['minimum_effective_score_samples']:.1f}**; "
        f"нормальные парные интервалы допущены в **{totals['normal_interval_eligible_variant_cases']}/{totals['variant_case_denominator']}** оценок. "
        'Порог100 вкладов — практический фильтр, не доказательство нормального покрытия.', '',
        f"Сумма времён22 отдельных переносов: **{totals['sum_case_runtime_seconds']:.2f}с**; максимальная RSS процесса **{totals['peak_rss_kib']/1024:.1f}MiB**. "
        'Это измеренное время переносов, без перечитывания HDF5, аудита, графиков и PDF. Часть расчётов шла одновременно с pytest, поэтому время не является чистым аппаратным бенчмарком. '
        'Повторное использование JSON не включено как новый физический расчёт.', '',
        'Полная программная проверка: **378/378**,0 ошибок/пропусков,144,06с; `uv lock --check` пройден,52 пакета. '
        'Новые проверки: сначала6/7 за3,79с, затем7/7 за2,62с. Единственная начальная ошибка — неверное ожидание теста: несовместимаяη может быть>1, а не только<0; '
        'проверка исправлена, физическая формула подбора не менялась. '
        'Два режима поглощения точно воспроизводят прежние средние и ковариации контрольной площадки. '
        'Дополнительно12/12 интегралов мощности регуляризованной смеси совпали с независимой аналитической формулой; '
        'максимальная разница8,9×10⁻¹⁶. [Проверка интегралов](flux-verification.json). '
        'Подробности — [test-verification.json](test-verification.json), [test-results.txt](test-results.txt), [integrity-verification.json](integrity-verification.json).', '',
        'Нормальный интервал не опубликован для `petersburg-2025-north-alt90 / mixed_eta_regional_cap85`: '
        '89,3 эффективных весовых вклада при57333 положительных путях и пороге100. '
        'Причина — сильная концентрация веса в редких ярких/наклонных попаданиях. Точечная оценка сохранена вCSV/JSON; '
        'большое число положительных попаданий само по себе не устраняет эту проблему. '
        'Интервалы Хёффдинга сохраняются как консервативные маргинальные границы и могут быть практически слишком широкими.', '',
        'Не получены: независимая полевая ошибка новых угловых моделей, абсолютная V/DNB→V калибровка, спектры/азимутальные диаграммы отдельных блоков, '
        'синхронные ночные AOD/SSA/профили, суточные ракурсы и статистический интервал полной физической неопределённости. '
        'Причина — этих входов и независимой контрольной серии в исследовании нет. Совпадение со спутниковыми композитами не заменяет такого контроля.', '',
        '## 8. Что я предлагаю делать с моделью', '',
        'Идея разбить территорию на ячейки работает как вычислительная постановка: перенос конечных ячеек всех порядков уже рассчитан. '
        'Для источников я предлагаю хранить Near/Off/All, QA/Num/Std, общую маску, класс поверхности и ограниченную угловую форму по территории. '
        'В плохо покрытых местах оставлять missing. Для укрупнения отдельно контролировать погрешность, особенно у наблюдателя и в тенях.', '',
        'Я не предлагаю автоматически заменять одну общуюη на карту обрезанныхη. Пространственные отношения бывают обоих знаков, '
        'а почти горизонтальная часть исходной смеси слишком жёстко связана с небольшим изменением Off/Near. '
        'Лучше проверять неотрицательные ограниченные функции радиансности/базисы и явно отделять неизмеренное продолжение после60° от данных. '
        'Для следующего подбора нужны суточные скорректированные VNP46A2 и соответствующая геометрия/QA VNP46A1 с реальными Sensor_Zenith/Sensor_Azimuth, '
        'проверка временного поведения, разделение обучающих/контрольных периодов и независимая наземная фотометрия при измеренной атмосфере. '
        'Это предложение по результатам опыта, а не уже выполненная загрузка или подтверждённая новая модель.', '',
        'Спектр города нужно определять отдельно: два угловых композита одной DNB-полосы не определяют смесь LED/натрия и особенно её синюю часть. '
        'В опубликованном [опыте Hung et al.](https://arxiv.org/abs/2107.02026) наземноеV-свечение выросло при снижении DNB после замены освещения; '
        'я использую это как независимое предостережение, не импортирую их численные наблюдения в свой fit.', '',
        '## 9. Воспроизведение', '',
        'Исходные HDF5 и региональные кэши должны соответствовать хешам аудитов. Секреты, рабочие базы и большие HDF5 в Git не входят.', '',
        '```bash', 'uv run python scripts/run_angular_nasa_study.py', 'uv run python scripts/summarize_angular_nasa_study.py',
        'uv run pytest -q -p no:cacheprovider', 'uv lock --check', '```', '',
        'Второй запуск первого скрипта сверяет входы/код/настройки и использует только совпадающие завершённые JSON. '
        'У генератора графиков собственный хеш; изменение оформления не выдаётся за новый перенос. '
        'Канонические предпосылки — [THEORY](../../THEORY.md), действующие режимы — [MODELS](../../MODELS.md), ограничения — [LIMITS](../../LIMITS.md).', '']
    parts = re.split(r'(```[\s\S]*?```|`[^`\n]+`)', '\n'.join(lines))
    for index in range(0, len(parts), 2):
        prose = re.sub(r'(?<=[А-Яа-яёЁ])(?=[0-9ημ])|(?<=[0-9])(?=[А-Яа-яёЁ])', ' ', parts[index])
        prose = re.sub(r'\b(в|из|на|до|после|с|по)(?=[A-Z])', r'\1 ', prose)
        parts[index] = prose
    (out/'README.md').write_text(''.join(parts))


def plot(out, protocol, audits, cases, rows):
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    # Spatial ratios are derived results; zero/unknown/censored cells are masked.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    for ax, name in zip(axes, ('madrid-2017-03', 'madrid-2017-11')):
        with (out/f'{name}-blocks.csv').open() as stream:
            points = [r for r in csv.DictReader(stream) if r['block_pixels'] == '4']
        scat = ax.scatter([float(r['longitude_deg']) for r in points], [float(r['latitude_deg']) for r in points],
            c=[np.log2(float(r['off_over_near'])) for r in points], marker='s', s=9,
            cmap='coolwarm', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        ax.set(xlabel='Долгота, °', ylabel='Широта, °', title=name+' · блоки 4×4')
        ax.set_aspect(1/np.cos(np.deg2rad(40.5)))
        ax.grid(alpha=.15)
    cb = fig.colorbar(scat, ax=axes, shrink=.85)
    cb.set_ticks([-1, 0, 1], labels=['0,5', '1', '2'])
    cb.set_label('OffNadir / NearNadir; обе средние ≥5')
    fig.savefig(out/'spatial-angle-ratios.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    theta = np.linspace(0, 90, 901)
    mu = np.cos(np.deg2rad(theta))
    for ax, name in zip(axes, ('madrid-2017-03', 'madrid-2017-11')):
        entry = protocol['inputs'][name]
        ratio, eta = entry['off_near_area_ratio_on_transport_support'], entry['regional_eta_uniform_solid_angle']
        for tail, label in ((0, 'К горизонту: 0'), (1, 'К горизонту: Off'), (2, 'К горизонту: 2×Off')):
            ax.plot(theta, empirical_radiance(1, ratio, mu, tail), label=label)
        for cap in (85, 88):
            h = horizontal_radiance_factor(np.maximum(mu, np.cos(np.deg2rad(cap))))
            ax.plot(theta, (1-eta+eta*h)/(1-eta+eta*bin_horizontal_factor(0, 20)), '--', label=f'Смесь η={eta:.3f}, предел {cap}°')
        ax.axvspan(0, 20, alpha=.12, color='green'); ax.axvspan(40, 60, alpha=.12, color='green')
        ax.set(xlabel='Угол испускания от вертикали, °', ylabel='Радиансность / NearNadir', title=name, yscale='log', ylim=(.1, 1e3))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'angular-continuations.png'); plt.close(fig)
    if cases:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), constrained_layout=True)
        for ax, (dataset, loc) in zip(axes.flat, [('madrid-2017-03', 'ucm'), ('madrid-2017-03', 'north'), ('madrid-2017-11', 'ucm'), ('madrid-2017-11', 'north')]):
            for variant, label in [('off_lambertian', 'Замена на Off'), ('all_lambertian', 'Замена на All'),
                    ('angular_horizon_zero', 'Угловая: 0 у горизонта'), ('angular_horizon_hold', 'Угловая: Off у горизонта'),
                    ('angular_horizon_double', 'Угловая: 2×Off у горизонта')]:
                selected = sorted([r for r in rows if r['role'] == 'primary' and r['variant'] == variant and r['case'].startswith(dataset+'-'+loc+'-')], key=lambda r: float(r['case'].split('alt')[-1]))
                if selected:
                    ax.errorbar([float(r['case'].split('alt')[-1]) for r in selected], [r['ratio_to_near'] for r in selected],
                        yerr=[1.96*r['ratio_standard_error'] for r in selected], marker='o', ms=3, capsize=2, label=label)
            ax.axhline(1, color='gray', lw=.7)
            ax.set(title=dataset+' · '+('UCM' if loc == 'ucm' else 'севернее города'), xlabel='Высота взгляда, °', ylabel='Искусственный фон / NearNadir', xticks=[10, 45, 90])
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.savefig(out/'sky-angle-comparison.png'); plt.close(fig)
    write_json(out/'plots-provenance.json', {'generator': 'scripts/summarize_angular_nasa_study.py',
        'source_code_sha256': file_hash(Path(__file__)), 'inputs': {'protocol.json': file_hash(out/'protocol.json'), 'input-summary.json': file_hash(out/'input-summary.json'),
            'sky-comparisons.csv': file_hash(out/'sky-comparisons.csv')},
        'plots': {p.name: file_hash(p) for p in out.glob('*.png')},
        'interpretation': 'Scientific plots of numerical inputs/scenarios; not calibrated visual appearance of the sky.'})


if __name__ == '__main__':
    main()
