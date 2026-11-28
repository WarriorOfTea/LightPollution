"""Build the Russian PDF for the azimuthal halo profile experiment."""

import argparse
from pathlib import Path
from statistics import median

import numpy as np

from lightpollution.adapters.storage import read_json
from report_halo_diagnosis import (
    ADJACENCY,
    CITIES,
    INK,
    LOW_ANGLE,
    NASA,
    RED,
    SLATE,
    TEAL,
    TUCSON_ORBIT,
    Report,
    number,
    plt,
)


FILENAME = 'spad-gorodskogo-oreola-po-storonam.pdf'
NOAA = (
    'https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VJ146A1'
)
SECTORS = {
    'toward': ('К спутнику', '#07796F'),
    'away': ('От спутника', '#BC4240'),
    'right': ('Сектор +90°', '#406FA6'),
    'left': ('Сектор −90°', '#90652F'),
}
MODES = {
    'h2': 'H2',
    'core': 'Мощность по ядру',
    'annular': 'Спад по кольцам',
    'radial': 'Спад по четырём сторонам',
    'sides': 'Разность сторон',
    'joint': 'Спад и разность сторон',
    'upper_bound': 'Задано максимальное усиление',
}


def date_name(key, add_city=False):
    city, day, *platform = key.split('/')
    year, month, day = day.split('-')
    return (
        (CITIES[city] + ' · ' if add_city else '')
        + f'{day}.{month}.{year[2:]}'
        + (' · NOAA-20' if platform else ' · NPP')
    )


def rounded(x, places=2, signed=False):
    return number(x, places, signed) if x is not None else '—'


def ground_cell(values, model):
    if model not in values:
        return '—'
    p = values[model]
    return f'{number(p["mae_pct"])} / {number(p["bias_pct"], signed=True)}'


def figures(path, profile_data, summary):
    scenes = profile_data['scenes']
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'text.color': INK,
            'axes.labelcolor': INK,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )

    def finish(fig, name):
        fig.savefig(
            path / (name + '.png'), dpi=190, facecolor='white', bbox_inches='tight'
        )
        plt.close(fig)

    names = ('h2', 'core', 'annular', 'radial', 'sides', 'joint')
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8), layout='constrained')
    sat = [summary['satellite']['joint']['h2']['halo_mae']]
    sat.extend(summary['satellite'][m][m]['halo_mae'] for m in names[1:])
    land = [summary['ground']['all'][m]['mae_pct'] for m in names]
    for axes_id, (ax, values) in enumerate(zip(axes, (sat, land), strict=True)):
        ax.set_yticks(
            range(len(values)), [MODES[m] for m in names] if axes_id == 0 else []
        )
        ax.tick_params(axis='y', length=0)
        ax.invert_yaxis()
        spread = max(values) - min(values)
        ax.set_xlim(min(values) - 0.3 * spread, max(values) + 0.5 * spread)
        ax.axvline(values[0], lw=0.9, color=SLATE, ls='--')
        ax.scatter(
            values,
            range(len(values)),
            c=[SLATE, '#415D79', TEAL, TEAL, TEAL, RED],
            s=34,
        )
        for i, v in enumerate(values):
            ax.text(
                v + 0.10 * spread,
                i,
                number(v, 4 if axes_id == 0 else 2),
                va='center',
                fontsize=8,
            )
        ax.set_xlabel('Спутник · нВт/(см²·ср)' if axes_id == 0 else 'Фотометры · %')
        ax.grid(axis='x', alpha=0.15)
    finish(fig, 'model-comparison')

    keys = ('Berlin/2017-03-29', 'Madrid/2012-02-20', 'Tucson/2019-04-03')
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), layout='constrained')
    for key, ax in zip(keys, axes, strict=True):
        scene = scenes[key]
        p = scene['fit']['joint']['profiles']
        for sector, (label, color) in SECTORS.items():
            points = np.array([row['values'] for row in p if row['sector'] == sector])
            if not len(points):
                continue
            measured = points[:, 0] - points[:, 3]
            ax.plot(
                points[:, 2], measured, 'o-', lw=1.1, ms=4, color=color, label=label
            )
            ax.plot(points[:, 2], points[:, 4], '--', lw=1.1, color=color)
        ax.axhline(0, color=SLATE, lw=0.65)
        ax.set(xlim=(3.5, 40), xlabel='От яркого края, км')
        ax.set_title(
            date_name(key, True).replace(' · NPP', ''), fontsize=9.3, color=INK
        )
        ax.grid(alpha=0.16)
    axes[0].set_ylabel('Радиансность, нВт/(см²·ср)')
    axes[-1].legend(frameon=False, fontsize=6.3, loc='upper right')
    finish(fig, 'directional-profiles')

    pairs = profile_data['paired_views']
    fig, axs = plt.subplots(
        2, 3, figsize=(8.5, 4.45), sharex=True, layout='constrained'
    )
    for (date, pair), ax in zip(sorted(pairs.items()), axs.flat):
        for sector, (label, color) in SECTORS.items():
            p = np.array(
                [row['values'] for row in pair['profiles'] if row['sector'] == sector]
            )
            if not len(p):
                continue
            # Both platforms use the same cells. Axes follow Suomi NPP.
            ax.plot(
                p[:, 2], p[:, 0] - p[:, 1], 'o-', color=color, lw=1, ms=3.5, label=label
            )
            ax.plot(p[:, 2], p[:, 3] - p[:, 4], '--', color=color, lw=0.9)
        ax.axhline(0, color=SLATE, lw=0.65)
        ax.set_title(
            '.'.join(date.split('-')[::-1][:2])
            + ' · '
            + number(pair['separation_minutes'], 0)
            + ' мин',
            fontsize=9.3,
            color=INK,
        )
        ax.set(xlim=(3.5, 40), xlabel='От яркого края, км')
        ax.grid(alpha=0.16)
    for ax in list(axs.flat)[len(pairs) :]:
        ax.axis('off')
    axs[0, 0].set_ylabel('NPP − NOAA-20, нВт/(см²·ср)')
    axs[1, 0].set_ylabel('NPP − NOAA-20, нВт/(см²·ср)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[1, 2].legend(handles, labels, loc='center', frameon=False, fontsize=8)
    finish(fig, 'same-night-differences')

    fig, ax = plt.subplots(figsize=(8.5, 3.0), layout='constrained')
    for i, (key, section) in enumerate(summary['fractions'].items()):
        scale = np.array(list(section['case_fractions'].values())) * 100
        nom = 100 * section['case_fractions']['nominal'][0]
        h2 = 100 * section['reference_fractions'][0]
        ax.vlines(i, scale[:, 0].min(), scale[:, 0].max(), color=TEAL, lw=3, alpha=0.45)
        ax.scatter(i, nom, color=TEAL, s=24, zorder=3)
        ax.scatter(i, h2, color=SLATE, marker='_', s=85, zorder=4)
    ax.set_xticks(
        range(len(scenes)),
        [date_name(key, True) for key in scenes],
        rotation=65,
        ha='right',
        fontsize=6.9,
    )
    ax.set_ylabel('Доля мощности ниже 10°, %')
    ax.grid(axis='y', alpha=0.15)
    finish(fig, 'angular-fractions')


def render(path, profile_data, summary):
    scenes = profile_data['scenes']
    s, g, mg = (summary[k] for k in ('satellite', 'ground', 'matched_ground'))
    tx = lambda mode, field='halo_mae': number(s[mode][mode][field], 4)
    ng = len(summary['rows'])
    pr = Report(path)
    pr.p('LIGHTPOLLUTION  /  СУТОЧНЫЙ VIIRS И ФОТОМЕТРЫ  /  28 НОЯБРЯ 2026', 'eyebrow')
    pr.p('Как убывает городской ореол с разных сторон', 'title')
    pr.p(
        f'Проверил {len(scenes)} пролётов над Берлином, Мадридом и Тусоном. '
        'Отдельно оценил масштаб излучения по ядру города и подгонял форму по спаду ореола.'
    )
    pr.box(
        f'Наземная MAE на {ng} замерах снизилась с <b>{number(g["all"]["h2"]["mae_pct"])}%</b> '
        f'у H2 до <b>{number(g["all"]["core"]["mae_pct"])}%</b> при калибровке по ядру. '
        f'Добавление спада и разности сторон дало <b>{number(g["all"]["joint"]["mae_pct"])}%</b>. '
        f'Спутниковая MAE в этом варианте снизилась с {number(s["joint"]["core"]["halo_mae"], 4)} '
        f'до {tx("joint")} нВт/(см²·ср).'
    )
    pr.figure(
        'model-comparison',
        'Чем левее, тем меньше средняя абсолютная ошибка. '
        'Оси увеличены около фактических значений. Вертикальный пунктир соответствует H2.',
    )
    limit = sum(v['fit']['joint']['at_bound'] for v in scenes.values())
    pr.p(
        f'На {limit}/{len(scenes)} пролётов оценка по спаду и разности сторон достигла границы допустимой формы. '
        'Пять пар Suomi NPP и NOAA‑20 за одну ночь позволили проверить перенос на другой азимут. '
        'Спад измерим, но точная доля света у горизонта в этой постановке слабо определяется.'
    )
    pr.p(
        'В старом опыте угловая подгонка добавляла общую мощность и давала 55,94% на земле. '
        'Здесь масштаб ограничен данными ядра, а изменения формы сохраняют восходящий поток.'
    )
    pr.p(
        'Для восьми прежних измерений Балагера оставляю прежнюю оценку H2. '
        'За проверенные даты там отсутствует чистый спутниковый ореол, пригодный для подгонки.',
        'small',
    )
    pr.sources(
        ('NASA Black Marble C2', NASA),
        ('NOAA‑20 VJ146A1', NOAA),
        ('Асимметрия за границей яркого города', LOW_ANGLE),
    )

    pr.page('01  /  ДАННЫЕ И ГЕОМЕТРИЯ', 'От какого края считаю расстояние')
    pr.p(
        'Взял калиброванные радиансности VIIRS до лунной и атмосферной коррекции '
        'из суточных продуктов VNP46A1 и VJ146A1 Collection 2. Оставил ночную, '
        'ясную или вероятно ясную маску и отсечки Солнца ≤ −18°, Луны ≤ −10°. '
        'Месячные N/O дают карты опорной яркости. Слабые внешние огни, яркие соседние источники '
        'и плавный фон входят в отдельные слагаемые. Расчёт переноса включает все порядки рассеяния.'
    )
    pr.p(
        'Яркий край задаю клетками города с месячным NearNadir от 2 нВт/(см²·ср). '
        'Для остального пространства измеряю расстояние до ближайшей такой клетки. '
        'Беру слабо светящиеся клетки за городской границей вдали от ярких внешних огней. '
        'Использую пять интервалов 4–8, 8–13, 13–19, 19–28 и 28–40 км от края. '
        'Четыре сектора шириной 90° центрированы на направлении к спутнику, противоположной стороне '
        'и двух поперечных направлениях. У теста совпадающих клеток обе сцены используют сектора NPP.'
    )
    pr.table(
        ['Площадка', 'Пролёты', 'Наземные точки', 'Особенность'],
        [
            (
                'Берлин',
                2,
                12,
                'Соседние ночи. Азимуты похожи, углы визирования разные.',
            ),
            (
                'Мадрид',
                3,
                32,
                'Два пролёта декабря в противоположных ракурсах за 25 суток. '
                'В некоторых секторах мало тёмных клеток.',
            ),
            (
                'Тусон',
                10,
                28,
                'Пять пар разных спутников за одну ночь. '
                'Фотометры снимали до переключения фонарей, спутники проходили позже.',
            ),
            ('Балагер', 0, 8, 'В прошлом отборе ореол слишком неполный.'),
        ],
        (0.2, 0.12, 0.16, 0.52),
    )
    pr.p(
        'Наземная модель использует прежнюю атмосферу, апертуру, естественный фон и спектральную свёртку '
        'по приборам. На Тусоне четыре наблюдения сохранены с переносом спутниковой оценки с соседней '
        'даты того же режима. На спутнике базовая длина волны 650 нм. Данные обоих спутников '
        'приводятся к одной схеме переноса, спектральные отличия двух приборов отдельно не гармонизированы.'
    )
    pr.sources(
        ('Руководство Black Marble', NASA),
        ('Суточная радиансность NOAA‑20', NOAA),
        ('Журнал переключения фонарей', TUCSON_ORBIT),
    )

    pr.page('02  /  МОДЕЛЬ И ПОДГОНКА', 'Мощность и угловая доля')
    pr.p(
        'Обозначаю высоту луча над горизонтом α, μ = sin α, O — месячную OffNadir. '
        'Базовая диаграмма H2 ниже 30° имеет L<sub>H2</sub>=O[1+(1−2μ)]. '
        'Пробная форма L(μ)=L<sub>H2</sub>+κO(1−2μ)(1−4μ) в том же диапазоне. '
        'Все углы выше 30° берутся из H2. Это одна физически допустимая семья углов, '
        'в которой меняется доля ниже 10° и 25°, а интеграл ниже 30° остаётся заданным.'
    )
    pr.box(
        'Полная восходящая мощность ячейки F = 2πA∫ μL(μ) dμ. '
        'Добавочный интеграл ∫ μ(1−2μ)(1−4μ) dμ от 0 до 1/2 равен нулю. '
        'Для доли до α &lt; 30° он равен ½μ²(1−2μ)². '
        'Границы κ от −2 до 5+2√6 ≈ 9,90 гарантируют L ≥ 0. '
        'Один и тот же поток можно геометрически перераспределить, сохранив опоры N/O.'
    )
    pr.p(
        'На каждом пролёте оцениваю масштаб s по яркому ядру, одновременно оценивая внешние огни '
        'и остаточный фон по ядру, соседям и дальнему контролю. Дальше s фиксирую. '
        'Спутниковая модель для блока имеет вид y = b + s·(K<sub>H2</sub> + κK<sub>Δ</sub>). '
        'Здесь b включает внешние источники и фон. В 95% ярких блоков ядра крайняя допустимая '
        'поправка к форме меняет городской шаблон не более чем на '
        f'{number(100 * max(x["core_p95_fractional_template_change"] for x in scenes.values()), 1)}% '
        'в каждом пролёте. Поэтому ядро в основном ограничивает масштаб.'
    )
    pr.p(
        'Затем на профилях беру разности соседних интервалов и разности противоположных секторов '
        'на одном расстоянии. Постоянная часть фона сокращается. Для скорости спада делю радиальную '
        'разность на расстояние между средними радиусами. В регрессии вычисляю все разности матрицей D '
        'и минимизирую взвешенную ошибку D(y−b−sK<sub>H2</sub>−sκK<sub>Δ</sub>) '
        'с ковариацией DCD<super>T</super>. Общие интервалы и общие пространственные группы создают '
        'ковариации. Псевдообратная не удваивает вес совпадающей информации.'
    )
    pr.p(
        'Усреднения 4×4 клетки формируют группы около 5–7 км по ширине. '
        'Около трети этих групп отложены. Ковариацию оцениваю из средних внутри групп '
        'и накладываю минимальный разброс 0,02 нВт/(см²·ср) на среднее интервала. '
        'Таблицы включают интервал только при наличии как минимум трёх блоков и двух групп.',
        'small',
    )
    pr.sources(('Расчёт ореола при разных профилях аэрозоля', ADJACENCY))

    pr.page('03  /  ИЗМЕРЕННЫЕ ПРОФИЛИ', 'На каких сторонах спад быстрее')
    pr.figure(
        'directional-profiles',
        'Точки — измерение с вычетом оценённых соседних огней и фона. Пунктир — вклад самого города '
        'в совместной модели. Разреженный профиль Мадрида показывает реальное ограничение маской.',
    )
    direction_rows = []
    for key, scene in scenes.items():
        d = summary['directions'][key]
        direction_rows.append(
            (
                date_name(key, True),
                *[
                    rounded(
                        None
                        if d.get(name, {}).get('decline_nw_per_km') is None
                        else d[name]['decline_nw_per_km'] * 1000,
                        1,
                        True,
                    )
                    for name in SECTORS
                ],
            )
        )
    pr.table(
        ('Ночь и прибор', 'К спутнику', 'От спутника', '+90°', '−90°'),
        direction_rows,
        (0.44, 0.14, 0.14, 0.14, 0.14),
    )
    pr.p(
        'Таблица показывает среднюю скорость убывания по направлению на доступных средних интервалах, '
        'в 10<super>−3</super> нВт/(см²·ср·км). Положительный знак означает спад. Для линейной оценки '
        'нужно от трёх пригодных интервалов. Отрицательный знак показывает рост остатка после '
        'вычета фона и внешних огней. Различия маски и остаточные пригородные источники могут давать '
        'такой наклон, поэтому строки служат наблюдаемым диагностическим признаком.',
        'small',
    )

    pr.page('04  /  ПЯТЬ ПАР ТУСОНА', 'Один город в противоположных ракурсах')
    pr.figure(
        'same-night-differences',
        'Сплошные — разность измерений NPP и NOAA‑20 после вычета двух оценок фона и соседних огней '
        'на общих тёмных клетках. Пунктир — модель той же разности. Радиусы и секторы общие, '
        'направления привязаны к NPP.',
    )
    pair_rows = []
    for day, t in sorted(profile_data['paired_views'].items()):
        a, b = (scenes[key] for key in t['keys'])
        angle = abs((a['azimuth_deg'] - b['azimuth_deg'] + 180) % 360 - 180)
        pair_rows.append(
            (
                '.'.join(day.split('-')[::-1]),
                number(t['separation_minutes'], 1),
                number(angle, 1),
                t['blocks'],
                number(t['kappa']),
                f'{number(t["mae_core"], 4)} → {number(t["mae_gradient"], 4)}',
            )
        )
    pr.table(
        ('Ночь UTC', 'Минут', 'Δ азимут, °', 'Блоков', 'κ', 'MAE разности'),
        pair_rows,
        (0.21, 0.11, 0.14, 0.13, 0.11, 0.30),
    )
    pr.p(
        'Здесь подгоняю двойную разность. Сначала вычитаю радиансности двух приборов '
        'на общих клетках, затем сравниваю, как эта разность меняется с радиусом и стороной. '
        'По одной части пространственных групп оцениваю κ, по остальным проверяю. '
        'На двух ночах оценка достигла верхней границы, ещё на двух — нижней. '
        'Даже в одном режиме ламп результат по двойной разности меняется между соседними ночами.'
    )
    pr.sources(('Измерения режимов Тусона', TUCSON_ORBIT))

    pr.page('05  /  КОНТРОЛЬ НА СПУТНИКЕ', 'Помог ли спад на отложенных данных')
    sat_table = []
    for mode, title in MODES.items():
        q = s['joint']['h2'] if mode == 'h2' else s[mode][mode]
        sat_table.append(
            (
                title,
                q['scenes'],
                number(q['halo_mae'], 4),
                number(q['radial_mae'], 4),
                number(q['sides_mae'], 4),
            )
        )
    pr.table(
        ('Вариант', 'Пролётов', 'MAE ореола', 'MAE спада', 'MAE сторон'),
        sat_table,
        (0.45, 0.12, 0.14, 0.15, 0.14),
    )
    n = s['joint']['joint']
    n_halo = f'{n["halo_n"]:,}'.replace(',', ' ')
    pr.p(
        f'Сравнение использует {n_halo} отложенных блоков радиансности, '
        f'{n["radial_n"]} соседних пар радиусов и {n["sides_n"]} пар противоположных секторов. '
        'Каждому пролёту в среднем даю одинаковый вес. Ореол и разность сторон измерены в нВт/(см²·ср), '
        'скорость спада — в нВт/(см²·ср·км).'
    )
    view = []
    for mode, title in (
        ('other_platform', 'Другой спутник той же ночью'),
        ('another_night', 'Другой день, тот же город'),
    ):
        controls = s[mode]
        old, new = controls['core'], controls[mode]
        view.append(
            (
                title,
                new['scenes'],
                new['halo_n'],
                f'{number(old["halo_mae"], 4)} → {number(new["halo_mae"], 4)}',
                f'{number(old["sides_mae"], 4)} → {number(new["sides_mae"], 4)}',
            )
        )
    pr.table(
        ('Где взял κ', 'Пролётов', 'Блоков', 'MAE ореола', 'MAE сторон'),
        view,
        (0.40, 0.12, 0.13, 0.18, 0.17),
    )
    pr.p(
        'При переносе между спутниками беру κ только с парного пролёта той же ночи, '
        'а масштаб и фон определяю по ядру и контролям целевой сцены. Ореол целевой зоны не участвует '
        'в подборе угла. Для междневного контроля беру Берлин, декабрьский Мадрид и две апрельские '
        'ночи Тусона со схожим режимом. Декабрьские пролёты Мадрида разделены 25 сутками, '
        'поэтому совпадение реальной работы источников там остаётся предположением.'
    )
    pr.p(
        'После включения сторон средняя MAE слабого ореола немного улучшилась. '
        'Проверка именно разности сторон на другом спутнике дала небольшое ухудшение. '
        'Удачное совпадение общего уровня ореола совместимо со слабым знанием угловой доли.'
    )

    pr.page('06  /  ФОТОМЕТРЫ', 'Что изменилось на земле')
    cols = ('h2', 'core', 'radial', 'sides', 'joint', 'old_free_tail')
    ground_rows = [
        (name, g[key]['h2']['n'], *[ground_cell(g[key], mode) for mode in cols])
        for key, name in (
            *CITIES.items(),
            ('all', 'Все'),
            ('same_utc_night', 'Та же ночь'),
        )
    ]
    pr.table(
        (
            'Город',
            'N',
            'H2',
            'Ядро',
            'Спад',
            'Стороны',
            'Спад и стороны',
            'Прежний хвост',
        ),
        ground_rows,
        (0.18, 0.055, 0.119, 0.122, 0.122, 0.125, 0.14, 0.137),
    )
    pr.p(
        'Пары чисел — средняя абсолютная относительная ошибка и среднее отклонение со знаком '
        'в процентах полной линейной яркости. В переносе меняется только искусственное излучение '
        'выбранного города. Данные фотометров используются только для проверки.'
    )
    matched_rows = []
    for mode, title in (
        ('paired_difference', 'Двойная разность двух спутников'),
        ('other_platform', 'Угол с другого спутника той же ночи'),
        ('another_night', 'Угол с другого пролёта другого дня'),
        ('upper_bound', 'Задан верхний предел κ для Тусона'),
    ):
        d = g['Tucson'] if mode == 'upper_bound' else mg[mode]
        matched_rows.append(
            (
                title,
                d[mode]['n'],
                ground_cell(d, 'h2'),
                ground_cell(d, 'core'),
                ground_cell(d, mode),
            )
        )
    pr.table(
        ('Перенос и контроль', 'N', 'H2', 'Ядро', 'С формой'),
        matched_rows,
        (0.43, 0.08, 0.16, 0.16, 0.17),
    )
    pr.p(
        'На Тусоне масштаб по яркому ядру даёт почти весь выигрыш. '
        'Пять оценок только по NOAA‑20 достигают одного и того же верхнего предела, '
        'поэтому совпадение с фотометрами после такого переноса нельзя приписать индивидуальной форме города по ночам. '
        'В Мадриде дальняя яркость остаётся заниженной даже после нового подбора. '
        'В Берлине H2 даёт меньшую среднюю ошибку, чем масштабированный прогноз.'
    )
    pr.p(
        'Здесь выборка по наземным точкам осталась прежней. Восемь замеров Балагера '
        'с прежней ошибкой H2 11,27% по модулю приведены для полноты, в общий парный итог они не входят.',
        'small',
    )

    pr.page('07  /  УСТОЙЧИВОСТЬ', 'Сколько угловой информации осталось')
    pr.figure(
        'angular-fractions',
        'Точка — номинальная оценка доли в шаблоне, короткая серая черта — H2. '
        'Полоса показывает размах между проверенными сценариями воздуха, фона и границы. '
        'Нулевая длина часто означает тот же край допустимой формы во всех сценариях.',
    )
    powers = [p['core']['scale'] for p in scenes.values()]
    res = summary['shape_resolution']
    median_shift = median(x['typical_bin_difference'] for x in res.values())
    gap = s['joint']['core']['halo_mae']
    pair_scales = [
        max(scenes[a]['core']['scale'], scenes[b]['core']['scale'])
        / min(scenes[a]['core']['scale'], scenes[b]['core']['scale'])
        - 1
        for a, b in (q['keys'] for q in profile_data['paired_views'].values())
    ]
    pr.p(
        f'Оценка мощности по ядру составила {number(min(powers))}–{number(max(powers))} '
        'от исходной мощности H2. На парах одной ночи относительное расхождение между двумя '
        f'оценками мощности достигало {number(100 * max(pair_scales), 1)}%. '
        'Это масштаб в принятой полосе и диаграмме, а абсолютный энергетический поток зависит от спектра.'
    )
    pr.p(
        'В пределах всего допустимого углового диапазона типичное изменение среднего интервала '
        f'ореола составило {number(median_shift, 4)} нВт/(см²·ср) по медианам пролётов. '
        f'Для масштаба сравнения средняя ошибка блоков у варианта с ядром — {number(gap, 4)}. '
        'Остаточные пространственные различия заметно больше сигнала углового параметра.'
    )
    pr.p(
        'В проверках менял высоту аэрозоля в 0,65 и 1,8 раза при постоянной AOD, '
        'увеличивал AOD в 1,5 раза, менял асимметрию фазовой функции, исключал слабые внешние огни '
        'и градиент фона, переносил яркий край с порога 2 на 4 нВт/(см²·ср). '
        'Широкие диапазоны на графике показывают чувствительность, а упор на границе ограничивает '
        'ценность даже устойчивых между сценариями чисел.'
    )
    pr.p(
        f'Прямая модель рассчитана по {profile_data["kernels"]} спутниковым ядрам на '
        f'{profile_data["samples_per_kernel"]:,} траекторий для каждого. '
        'Использовал прежние дневные опоры AERONET и сценарий для Мадрида в феврале. '
        'В Тусоне базовое g=0,644 согласовано с наземным исследованием. '
        'Послойная экстинкция на момент пролёта остаётся основной неизвестной атмосферной величиной.',
        'small',
    )
    pr.box(
        'Спад ореола остаётся полезным независимым тестом переноса. Для оценки городской доли '
        'важны более близкие чистые профили, настоящий ночной профиль аэрозоля и независимые угловые опоры. '
        'Раздельный масштаб предотвращает повторение крупного завышения. '
        'Тонкая настройка формы пока не дала устойчивого выигрыша над оценкой мощности по ядру.'
    )
    pr.sources(
        ('Роль высоты и фазовой функции аэрозоля', ADJACENCY),
        ('Работа по угловой информации в ореоле', LOW_ANGLE),
    )
    pr.save(FILENAME, 'Как убывает городской ореол с разных сторон')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles', type=Path)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profiles, summary = read_json(args.profiles), read_json(args.summary)
    figures(args.output, profiles, summary)
    render(args.output, profiles, summary)


if __name__ == '__main__':
    main()
