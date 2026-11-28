"""Render the halo transfer investigation as a Russian PDF and scientific figures."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as RasterImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from lightpollution.adapters.storage import read_json
from run_halo_city_fits import HALO


CITIES = {'Berlin': 'Берлин', 'Madrid': 'Мадрид', 'Tucson': 'Тусон'}
INK, TEAL, RED, SLATE, PALE = '#17314A', '#087D76', '#C34542', '#516A7C', '#EAF4F3'
NASA = 'https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_Collection2.0.pdf'
L1B = 'https://doi.org/10.5067/VIIRS/VNP02DNB.002'
HALO_PAPER = 'https://doi.org/10.1038/s41598-020-64673-2'
LOW_ANGLE = 'https://doi.org/10.1029/2025JD045452'
ADJACENCY = 'https://amt.copernicus.org/articles/16/2531/2023/'
MULTI = 'https://doi.org/10.1029/2023JD039804'
SPECTRAL = 'https://ncc.nesdis.noaa.gov/VIIRS/VIIRSSpectralResponseFunctions.php'
TUCSON_SKY = 'https://doi.org/10.1016/j.jqsrt.2020.107120'
TUCSON_ORBIT = 'https://doi.org/10.1177/1477153520958463'
MULTI_ANGLE = 'https://doi.org/10.1029/2021JD036382'
CALIBRATION = 'https://doi.org/10.1002/jgrd.50873'
ANGLE_ATMOSPHERE = 'https://doi.org/10.1016/j.atmosres.2021.105969'
AIRGLOW = 'https://doi.org/10.1051/0004-6361/201219040'


def number(x, places=2, signed=False):
    return (
        format(x, f'{"+" if signed else ""}.{places}f')
        .replace('.', ',')
        .replace('-', '−')
    )


def draw_figures(output, summary, scenes):
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 9,
            'text.color': INK,
            'axes.labelcolor': INK,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.bottom': True,
            'xtick.color': SLATE,
            'ytick.color': SLATE,
        }
    )

    def finish(figure, name):
        figure.savefig(
            output / (name + '.png'), dpi=185, facecolor='white', bbox_inches='tight'
        )
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), layout='constrained')
    variants = ['h2', 'original', 'conserve', 'weak']
    titles = ['H2', 'Подогнанный хвост', 'Постоянная мощность', 'Слабые соседние огни']
    palette = [SLATE, RED, TEAL, '#AB7236']
    pairs = (
        [summary['satellite'][k]['halo_mae'] for k in variants],
        [summary['ground']['All']['metrics'][k]['mae_pct'] for k in variants],
    )
    for i, (ax, values) in enumerate(zip(axes, pairs, strict=True)):
        ax.barh(np.arange(len(titles)), values, color=palette, height=0.58)
        ax.set_yticks(np.arange(len(titles)), titles if i == 0 else [])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.21)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E1E9F0')
        for j, v in enumerate(values):
            ax.text(
                v + max(values) * 0.025,
                j,
                number(v, 3 if i == 0 else 1),
                va='center',
                fontsize=8,
            )
        ax.set_xlabel(('Спутник · нВт/(см²·ср)', 'Фотометры · % от наблюдения')[i])
    finish(figure, 'comparison')

    alpha = np.linspace(0, 30, 301)
    mu = np.sin(np.deg2rad(alpha))
    tail = 1 - 2 * mu
    berlin = scenes['Berlin/2017-03-29']['fits']
    madrid = scenes['Madrid/2012-02-20']['fits']
    lines = (
        (1 + tail, 'H2', SLATE, '--'),
        (1 + berlin['original']['kappa'] * tail, 'Берлин · 29 марта', RED, '-'),
        (1 + madrid['original']['kappa'] * tail, 'Мадрид · 20 февраля', '#AB7236', '-'),
        (
            1 + berlin['conserve']['kappa'] * tail * (1 - 4 * mu),
            'Постоянная мощность',
            TEAL,
            '-',
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 2.65), layout='constrained')
    for val, title, color, line in lines:
        axes[0].plot(alpha, val, label=title, color=color, linestyle=line, lw=2)
        axes[1].plot(alpha, mu * val, color=color, linestyle=line, lw=2)
    for ax, label in zip(axes, ('Радиансность L/O', 'Сила света I/(AO)'), strict=True):
        ax.set(xlabel='Угол над горизонтом, °', ylabel=label, xlim=(0, 30))
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper right')
    finish(figure, 'angular-power')

    figure, axes = plt.subplots(
        1, 3, figsize=(8.6, 3.0), sharey=True, layout='constrained'
    )
    for (city, name), ax in zip(CITIES.items(), axes, strict=True):
        rows = [r for r in summary['rows'] if r['city'] == city]
        dist = [r['distance_km'] for r in rows]
        for scenario, legend, color, marker in (
            ('h2', 'H2', SLATE, 'o'),
            ('original', 'Подогнанный хвост', RED, '^'),
            ('conserve', 'Постоянная мощность', TEAL, 'x'),
        ):
            errors = [r['relative_pct'][scenario] for r in rows]
            ax.scatter(
                dist, errors, s=22, marker=marker, label=legend, color=color, alpha=0.82
            )
        ax.set_title(f'{name} · {len(rows)}', fontsize=10, color=INK)
        ax.axhline(0, color=INK, lw=0.75)
        ax.grid(axis='y', alpha=0.2)
        ax.set_xlabel('До центра, км')
    axes[0].set_ylabel('Смещение, % от наблюдения')
    axes[-1].legend(loc='upper right', fontsize=6.7, frameon=False)
    finish(figure, 'distance-residuals')

    figure, axes = plt.subplots(
        1, 3, figsize=(8.6, 2.78), sharex=True, sharey=True, layout='constrained'
    )
    methods = [
        ('h2', 'H2'),
        ('original', 'Подгонка амплитуды'),
        ('weak', 'Со слабыми огнями'),
    ]
    all_values = []
    for name, _ in methods:
        for s in scenes.values():
            d = s['fits'][name]['contrast']
            all_values += d['observed'] + d['predicted']
    lo, hi = min(all_values) - 0.02, max(all_values) + 0.02
    for (name, title), ax in zip(methods, axes, strict=True):
        for city, label in CITIES.items():
            obs, pred = [], []
            for key, scene in scenes.items():
                if key.startswith(city + '/'):
                    d = scene['fits'][name]['contrast']
                    obs.extend(d['observed'])
                    pred.extend(d['predicted'])
            color = {'Berlin': TEAL, 'Madrid': RED, 'Tucson': '#396AA1'}[city]
            ax.scatter(obs, pred, s=19, alpha=0.7, color=color, label=label)
        ax.plot([lo, hi], [lo, hi], ls='--', lw=0.8, c=SLATE)
        ax.axvline(0, lw=0.5, c='#CBD9E1')
        ax.axhline(0, lw=0.5, c='#CBD9E1')
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Контраст VIIRS')
        ax.set_title(title, fontsize=9.5, color=INK)
    axes[0].set_ylabel('Контраст модели')
    axes[-1].legend(fontsize=6.9, frameon=False, loc='upper left')
    finish(figure, 'azimuth-contrast')


class Report:
    width = 170 * mm

    def __init__(self, output):
        self.output = output
        self.story = []
        font_dir = Path(matplotlib.get_data_path()) / 'fonts/ttf'
        for name, path in (('Body', 'DejaVuSans.ttf'), ('Bold', 'DejaVuSans-Bold.ttf')):
            pdfmetrics.registerFont(TTFont(name, str(font_dir / path)))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold')
        specs = {
            'eyebrow': ('Bold', 8.1, 10.3, TEAL, 9),
            'title': ('Bold', 27, 31, INK, 12),
            'section': ('Bold', 20.4, 25, INK, 10),
            'subtitle': ('Bold', 11.2, 14, INK, 6),
            'body': ('Body', 10.35, 14.6, INK, 7),
            'small': ('Body', 8.75, 12, SLATE, 7),
            'cell': ('Body', 8.55, 11.4, INK, 0),
            'header': ('Bold', 8.2, 11, '#FFFFFF', 0),
        }
        self.styles = {
            k: ParagraphStyle(
                k,
                fontName=font,
                fontSize=size,
                leading=leading,
                textColor=colors.HexColor(color),
                spaceAfter=space,
            )
            for k, (font, size, leading, color, space) in specs.items()
        }

    def p(self, content, style='body'):
        self.story.append(Paragraph(content, self.styles[style]))

    def page(self, name, title):
        self.story.append(PageBreak())
        self.p(name, 'eyebrow')
        self.p(title, 'section')

    def box(self, content):
        item = Table(
            [[Paragraph(content, self.styles['body'])]], colWidths=[self.width]
        )
        item.setStyle(
            TableStyle(
                [
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(PALE)),
                    ('LINEBEFORE', (0, 0), (0, -1), 2, colors.HexColor(TEAL)),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
                ]
            )
        )
        self.story.extend((item, Spacer(1, 9)))

    def table(self, heads, rows, columns):
        data = [[Paragraph(str(x), self.styles['header']) for x in heads]]
        data += [[Paragraph(str(x), self.styles['cell']) for x in row] for row in rows]
        t = Table(
            data,
            colWidths=[c * self.width for c in columns],
            repeatRows=1,
            hAlign='LEFT',
        )
        t.setStyle(
            TableStyle(
                [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(INK)),
                    (
                        'ROWBACKGROUNDS',
                        (0, 1),
                        (-1, -1),
                        list(map(colors.HexColor, (PALE, '#F8FAFC'))),
                    ),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LINEBELOW', (0, 0), (-1, -1), 0.35, colors.HexColor('#CFDCE6')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]
            )
        )
        self.story.extend((t, Spacer(1, 8)))

    def figure(self, name, caption):
        f = self.output / (name + '.png')
        with RasterImage.open(f) as image:
            w, h = image.size
        self.story.append(Image(str(f), width=self.width, height=self.width * h / w))
        self.p(caption, 'small')

    def sources(self, *pairs):
        self.p(
            '  ·  '.join(
                f'<link href={quoteattr(uri)} color="{TEAL}"><u>{escape(label)}</u></link>'
                for label, uri in pairs
            ),
            'small',
        )

    def save(
        self,
        filename='pochemu-podgonka-oreola-zavysila-yarkost.pdf',
        title='Почему подгонка ореола завысила яркость',
    ):
        doc = SimpleDocTemplate(
            str(self.output / filename),
            pagesize=A4,
            title=title,
            author='LightPollution',
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=23 * mm,
            bottomMargin=17 * mm,
        )

        def footer(canvas, document):
            canvas.setStrokeColor(colors.HexColor('#CCDDE6'))
            canvas.line(19 * mm, 14 * mm, 191 * mm, 14 * mm)
            canvas.setFillColor(colors.HexColor(SLATE))
            canvas.setFont('Body', 7.7)
            canvas.drawString(
                19 * mm, 9 * mm, 'LIGHTPOLLUTION  /  VIIRS И НАЗЕМНАЯ ФОТОМЕТРИЯ'
            )
            canvas.drawRightString(191 * mm, 9 * mm, str(document.page))

        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)


def ground_cell(summary, sample, mode, places=2):
    s = summary['ground'][sample]['metrics'][mode]
    return f'{number(s["mae_pct"], places)} / {number(s["bias_pct"], places, True)}'


def city_date(key):
    city, day = key.split('/')
    d = '.'.join(day.split('-')[::-1])
    return f'{CITIES[city]} · {d}'


def render_results(output, summary, satellite, native):
    scenes = satellite['scenes']
    s, g, compare = (summary[k] for k in ('satellite', 'ground', 'matched'))
    sat = lambda mode: number(s[mode]['halo_mae'], 4)
    ground = lambda sample, mode, places=2: ground_cell(summary, sample, mode, places)
    f = Report(output)
    f.p('LIGHTPOLLUTION  /  РАЗБОР НАБЛЮДАТЕЛЬНОГО ОПЫТА  /  28 НОЯБРЯ 2026', 'eyebrow')
    f.p('Почему подгонка ореола завысила яркость', 'title')
    f.p(
        'Проверил, что действительно увидел спутник и почему выбранная диаграмма дала слишком светлое небо на земле.'
    )
    f.box(
        'Подгонка ниже 30° добавляла городу восходящую мощность. '
        f'Ошибку ореола она снизила с <b>{sat("h2")} до {sat("original")}</b> нВт/(см²·ср). '
        f'Наземная ошибка выросла с <b>{number(g["All"]["metrics"]["h2"]["mae_pct"])}% до '
        f'{number(g["All"]["metrics"]["original"]["mae_pct"])}%</b>. '
        f'При постоянной мощности она составила <b>{number(g["All"]["metrics"]["conserve"]["mae_pct"])}%</b>.'
    )
    f.figure(
        'comparison',
        'Средняя ошибка по модулю. Слева 10 пролётов и 2 118 отложенных усреднений ореола. Справа те же 72 наземных замера в каждом столбце.',
    )
    f.p(
        'В старой задаче один коэффициент должен был объяснить весь избыток света, согласующийся по форме с ореолом. '
        'В эту же часть сигнала попадали слабые внешние огни и погрешности атмосферы. '
        'Отдельный контроль повторного рассеяния уже был заложен в расчёт. '
        'Его выключение при подгонке заставило модель приписать городу ещё больше низкоуглового света.'
    )
    f.p(
        'Сам физический эффект существует. Спутник видит свет города, рассеянный атмосферой за пределами яркой застройки. '
        'По сравнению разных направлений наблюдения можно ограничивать долю излучения около горизонта. '
        'Этот опыт показывает, какие ограничения нужны, чтобы из спутникового совпадения получить наземный прогноз.'
    )
    f.p(
        f'Воспроизвёл {summary["ground_integrations"]} наземных спектральных интегралов и '
        f'{summary["satellite_kernels"]} спутниковых ядер. Для Балагера осталось прежнее сравнение восьми '
        'замеров. В 11 проверенных датах недостаточно чистых данных ореола.',
        'small',
    )
    f.sources(
        ('Наблюдаемая природа ореола', HALO_PAPER),
        ('Угловая инверсия по ореолу', LOW_ANGLE),
    )

    f.page('01  /  СПУТНИКОВЫЕ ДАННЫЕ', 'Как ореол несёт угловую информацию')
    f.p(
        'Если свет доходит от города до слоя воздуха на высоте 1,5 км в 30 км от источника, '
        'в плоском приближении он выходит под углом около 2,9° к горизонту. '
        'Рассеявшись, он может попасть в пиксель над внешним районом, где почти нет ярких ламп. '
        'При косом взгляде свет рассеивается на ближней и дальней сторонах под разными углами. '
        'Из-за выраженного рассеяния вперёд стороны меняются неодинаково.'
    )
    f.box(
        'Зарегистрированный DNB = пространственный отклик прибора × '
        '(прямой свет + одно рассеяние + повторные рассеяния + свет соседей и отражения) '
        '+ остаточный фон и шум. Все оптические слагаемые интегрируются по спектральной полосе прибора.'
    )
    f.p(
        'В суточном VNP46A1 использовал калиброванную радиансность на верхней границе атмосферы '
        'после переноса на сетку 15″. Маска требует ночь, Солнце ≤ −18°, Луну ≤ −10°, '
        'приемлемые флаги облачности, снега и прибора. У десяти принятых пролётов '
        'Солнце в ядре было ниже −30°, Луна ниже −25°. Маска снижает риск облаков, '
        'но не является измерением толщины каждого тонкого слоя.'
    )
    f.p(
        'Радиусы 18–43 км для слабого ореола и 46–65 км для дальнего контроля отсчитываю от центра. '
        'Группы яркого ядра и соседних источников помогают определить фон и его отличие от прямых огней. '
        'Внутри пролёта точность оцениваю на отложенных участках около 7 км. '
        'Пиксели внутри усреднений и соседние участки могут оставаться пространственно связанными.'
    )
    f.p(
        'Месячные NearNadir и OffNadir задают распределение граничной радиансности в двух категориях '
        'зенитного угла, 0–20° и 40–60°. Они дают пространственные опоры, а их веса по ночам и ракурсам '
        'отличаются от суточного пролёта. Ниже 30° над горизонтом начинается область экстраполяции. '
        'При оценке ореола использовал световые шаблоны, в основном один угол и азимут на пролёт '
        'и эффективную длину волны 650 нм.'
    )
    f.p(
        'Один широкий DNB измеряет взвешенный интеграл спектра. В опорной кривой S‑NPP '
        'отклик при 450 нм порядка 2×10<super>−5</super> от пика, при 650 нм около 0,94. '
        'Синюю часть спектра и её угловое распределение надо проверять отдельными полосами. '
        'Эти числа относятся к использованной кривой отклика, усреднённой для одного режима.'
    )
    f.sources(
        ('NASA Black Marble C2', NASA),
        ('NOAA: спектральный отклик VIIRS', SPECTRAL),
        ('Трёхмерный перенос ночного света', ADJACENCY),
    )

    f.page('02  /  ЭНЕРГИЯ', 'Подгонка меняла полную мощность')
    f.p(
        'Пусть α — высота луча над горизонтом, μ = sin α, O — радиансность OffNadir, A — площадь ячейки. '
        'У горизонтальной ячейки сила света I = AμL, а полный восходящий поток F = 2πA∫ μL dμ. '
        'Для осесимметричного источника интеграл идёт от 0 до 1.'
    )
    f.box(
        'При μ &lt; 1/2 прежняя формула L = O[1 + κ(1 − 2μ)]. H1 задаёт κ = 0, H2 задаёт κ = 1. '
        'Поэтому F(κ) − F(H2) = πAO(κ − 1)/12. '
        'Угловые опоры N и O выше 30° при этом остаются фиксированными.'
    )
    powers = []
    for key, scene in scenes.items():
        p, k = scene['angular_power']['original'], scene['fits']['original']['kappa']
        powers.append(
            (
                city_date(key),
                number(k, 1),
                number(k + 1, 1),
                number(p['power_over_h2']),
                number(100 * p['below_25_deg_fraction'], 1),
            )
        )
    f.table(
        ['Пролёт', 'κ', 'L гориз. / O', 'F / F(H2)', 'F ниже 25°, %'],
        powers,
        (0.39, 0.105, 0.185, 0.14, 0.18),
    )
    f.figure(
        'angular-power',
        'Радиансность и сила света в экстраполируемом диапазоне. В Берлине значение горизонта выросло до 27,7 O, а интегральный восходящий поток до 2,92 от H2.',
    )
    f.p(
        'Для перераспределения с постоянной мощностью проверил L = O[1 + κ(1 − 2μ)(1 − 4μ)]. '
        'Интеграл μ(1 − 2μ)(1 − 4μ) от 0 до 1/2 равен нулю. '
        'Диапазон −1…8 следует из неотрицательности радиансности в каждом угле. '
        'Этот вариант сохраняет F(H1) в каждой ячейке и обе угловые опоры. '
        'Абсолютной независимой оценки мощности светильников здесь нет.',
        'small',
    )

    f.page('03  /  РАССЕЯНИЕ И ФОН', 'Откуда появился наземный избыток')
    f.p(
        'Для каждой точки восстановил вклады на тех же случайных траекториях и в исходной полосе прибора. '
        'Наземная разность Δ = (κ − 1) × [прогноз H2 − прогноз H1]. '
        'Равенство численно выполнено на всех 72 замерах.'
    )
    rows = []
    for city, label in (*CITIES.items(), ('All', 'Все')):
        t = g[city]
        once, repeat = t['mean_tail_extra_pct_by_order']
        rows.append(
            (
                label,
                t['n'],
                number(once, 1, True),
                number(repeat, 1, True),
                number(once + repeat, 1, True),
                number(t['mean_natural_pct_of_observed'], 1),
                number(t['metrics']['zero_natural']['bias_pct'], 1, True),
            )
        )
    f.table(
        [
            'Город',
            'N',
            'Одно, п.п.',
            'Повторные, п.п.',
            'Вся добавка, п.п.',
            'Фон, %',
            'Смещение при фоне 0, %',
        ],
        rows,
        (0.16, 0.06, 0.14, 0.165, 0.155, 0.11, 0.21),
    )
    f.p(
        'Первое и последующие рассеяния присутствовали в прямой модели и до прибора, и на земле. '
        'Повторные дали от 17 до 47% спутникового отклика добавленного хвоста '
        'в отложенном ореоле разных пролётов. На земле они дали '
        f'{number(100 * g["All"]["multiple_share_of_added_skyglow"], 1)}% добавки в среднем '
        'по нормированным наблюдениям. Остальная часть пришлась на первое рассеяние.'
    )
    f.p(
        'При переоценке коэффициента только с первым рассеянием на спутнике наземная ошибка '
        f'при полном переносе выросла до {number(g["All"]["metrics"]["single"]["mae_pct"])}%. '
        f'Если одновременно оставить первое рассеяние и на земле, получается '
        f'{ground("All", "single_both")}% по модулю и со знаком. Это контроль усечения переноса.'
    )
    f.p(
        'Среднее условное численное отклонение оцениваю по ковариации H1/H2. '
        f'Медиана стандартной ошибки добавки — {number(g["All"]["median_mc_error_on_delta_points"])} п.п. '
        f'У {g["All"]["tail_larger_than_5mcse_count"]}/72 добавка больше пяти таких стандартных ошибок. '
        'Эта проверка относится к разбросу траекторий при заданных входах.'
    )
    f.p(
        'Мысленно убрал весь принятый наземный естественный фон. В Берлине и Тусоне средняя яркость '
        f'всё равно завышена на {number(g["Berlin"]["metrics"]["zero_natural"]["bias_pct"], 1)}% и '
        f'{number(g["Tucson"]["metrics"]["zero_natural"]["bias_pct"], 1)}%. '
        'Фон остаётся важен на самых тёмных точках и как возможная часть спутниковых остатков. '
        'Причина наблюдаемого добавления между H2 и хвостом точна, потому что в расчёте фон одинаков.'
    )
    f.sources(
        ('Повторные рассеяния и ошибки усечения', MULTI),
        ('Переменный естественный фон ночного неба', AIRGLOW),
    )

    f.page('04  /  ГЕОГРАФИЯ', 'Где завысили и где стало ближе')
    f.figure(
        'distance-residuals',
        'Каждый маркер представляет наземный приборный замер. Шкала одна для трёх городов. В Тусоне разные приборы могут находиться на одном расстоянии.',
    )
    distance_rows = []
    for city, label in CITIES.items():
        for band, name in (('near', '0–15'), ('mid', '15–35'), ('far', '≥35')):
            k = f'{city}/{band}'
            distance_rows.append(
                (
                    label,
                    name,
                    g[k]['n'],
                    ground(k, 'h2', 1),
                    ground(k, 'original', 1),
                    ground(k, 'conserve', 1),
                )
            )
    f.table(
        ['Город', 'Км', 'N', 'H2', 'Подогнанный хвост', 'Постоянная мощность'],
        distance_rows,
        (0.19, 0.115, 0.07, 0.17, 0.245, 0.21),
    )
    f.p(
        'В таблице средняя ошибка по модулю / среднее смещение, в процентах линейной яркости. '
        'У Мадрида прежний H2 сильно занижал дальние точки. Добавленная мощность приблизила их к наблюдениям '
        'и одновременно перевела ближние точки от −25,3% к +23,6%. '
        'На 8 точках, где расчётный вклад самого Мадрида больше половины, ошибка выросла с 26,53% до 27,83%.'
    )
    f.p(
        'Тусон. После исключения двух ночей с более заметными облачными оговорками остаются '
        f'{g["Tucson/less_cloud"]["n"]} замеров. Ошибка H2 и подогнанного хвоста на них '
        f'{number(g["Tucson/less_cloud"]["metrics"]["h2"]["mae_pct"], 1)}% и '
        f'{number(g["Tucson/less_cloud"]["metrics"]["original"]["mae_pct"], 1)}%. '
        'На дальних точках сильнее относительный вклад фона. '
        'Гора Mount Lemmon наблюдает с высоты около 2,8 км. '
        'В расчёте задана высота прибора, а детальная тень гор и отражения поверхности отсутствуют.'
    )
    f.sources(('Геометрия, приборы и опыт Тусона', TUCSON_SKY))

    f.page('05  /  КОНТРОЛЬ ФИЗИКИ', 'Похожие ореолы требуют разного света')
    experiments = (
        ('h2', 'H2, фиксированный κ=1'),
        ('original', 'Прежняя подгонка'),
        ('single', 'Первое рассеяние при подгонке'),
        ('height_low', 'Масштаб высоты ×0,65, та же AOD'),
        ('height_high', 'Масштаб высоты ×1,8, та же AOD'),
        ('aod_low', 'AOD ×0,60'),
        ('aod_high', 'AOD ×1,50'),
        ('g_062', 'Менее направленное рассеяние, g=0,62'),
        ('wave_550', 'Перенос при 550 нм'),
        ('wave_790', 'Перенос при 790 нм'),
        ('pixel_zenith', 'Зенитный угол меняется по пикселям'),
        ('blur_stress', 'Контрольное размытие прибора'),
    )
    f.table(
        [
            'Изменение при спутниковой подгонке',
            'Диапазон κ',
            'MAE ореола',
            'На земле, %',
        ],
        [
            (
                label,
                '…'.join(number(k, 1) for k in s[mode]['min_max_kappa']),
                sat(mode),
                ground('All', mode, 1),
            )
            for mode, label in experiments
        ],
        (0.49, 0.16, 0.15, 0.20),
    )
    f.p(
        'Каждая строка использует те же 10 пролётов и 72 наземных замера, те же классы и разбиение. '
        'Для спутника менял только указанный вход. На землю переносил полученный κ без подгонки. '
        'Случаи 550/790 нм — проверка спектральной чувствительности, со степенью Ангстрема 1,1 от 650 нм. '
        'Это не измерение цвета ореола. Размытие — 98% ядра FWHM 1,1 км и 2% крыла 10 км, '
        'то есть проверка конкретного допущения, без паспортной калибровки хвоста PSF.',
        'small',
    )
    ber = scenes['Berlin/2017-03-28']['fits']
    m = compare['tucson_ground_g']
    f.p(
        'Для Берлина 28 марта при неизменной AOD разные высоты дали '
        f'κ от {number(ber["height_low"]["kappa"], 1)} до {number(ber["height_high"]["kappa"], 1)}. '
        'В Тусоне исходный спутниковый расчёт использовал g=0,8, а наземный — спектральную фазовую функцию '
        'из полевого исследования. Подстановка её значения g≈0,644 при 650 нм изменила наземную MAE '
        f'с {number(m["ground"]["original"]["mae_pct"], 1)}% до '
        f'{number(m["ground"]["tucson_ground_g"]["mae_pct"], 1)}% на 28 точках. '
        'Это выявленный разнобой между двумя моделями атмосферы.'
    )
    f.sources(
        ('Аэрозоль и несимметричный ореол', LOW_ANGLE),
        ('Параметры переноса Тусона', TUCSON_SKY),
    )

    f.page('06  /  ИСТОЧНИКИ И ПРИБОР', 'Что ещё подгонка принимала за город')
    alternatives = (
        ('h2', 'H2'),
        ('original', 'Хвост, свободно добавляющий мощность'),
        ('weak', 'Отдельный вклад слабых внешних огней'),
        ('plane', 'Наклон спутникового фона'),
        ('weak_and_plane', 'Слабые огни и наклон фона'),
        ('conserve', 'Угловое перераспределение при F(H1)'),
    )
    f.table(
        ['Что оценено по спутнику', 'MAE ореола', 'MAE / смещение на земле, %'],
        [(label, sat(mode), ground('All', mode)) for mode, label in alternatives],
        (0.56, 0.17, 0.27),
    )
    f.p(
        'Раньше внешними источниками считались только клетки месячной карты от 2 нВт/(см²·ср). '
        'Проверил отдельную компоненту 0,3–2. В Берлине 29 марта оценка κ упала '
        f'с 26,7 до {number(scenes["Berlin/2017-03-29"]["fits"]["weak"]["kappa"], 1)}. '
        'Месячная радиансность слабых клеток тоже может содержать диффузный свет. '
        'Проверка измеряет чувствительность к карте, долю света реальных пригородных фонарей она не устанавливает.'
    )
    f.p(
        'Из исходного шаблона хвоста в ореоле после вычитания наилучшей линейной '
        'комбинации фона, соседних огней и градиента типично остаётся мало независимой формы. '
        f'На десяти пролётах норма остатка составила '
        f'{number(100 * min(t["tail_norm_after_halo_nuisance_fraction"] for t in scenes.values()), 0)}–'
        f'{number(100 * max(t["tail_norm_after_halo_nuisance_fraction"] for t in scenes.values()), 0)}% '
        'от нормы исходного шаблона. Это показатель сходства компонентов, без статистической интерпретации.'
    )
    nrows = []
    for key, case in native.items():
        r, geo = (
            case['paired_native_centres'][HALO],
            case['same_scan_satellite_position_spread'],
        )
        nrows.append(
            (
                CITIES[key.split('/')[0]],
                r['n'],
                number(r['mae'], 5),
                number(r['correlation'], 4),
                number(geo['as_written']['median_km']),
                number(geo['azimuth_reversed']['median_km'], 1),
            )
        )
    f.table(
        ['Пролёт', 'Пар', 'MAE A1 / L1B', 'r', 'Штатный азимут, км', 'Разворот, км'],
        nrows,
        (0.16, 0.09, 0.18, 0.13, 0.22, 0.22),
    )
    f.p(
        'Сопоставил центры пикселей нативного VIIRS и клеток A1 на двух пролётах с QA. '
        'MAE указана в нВт/(см²·ср). Два правых столбца — медиана разброса восстановленных позиций спутника '
        'внутри строки сканирования. Знак азимута верен. A1 и L1B используют один оптический прибор, '
        'поэтому их согласие ограничивает ошибку сетки, а общее паразитное рассеяние в оптике может сохраняться.'
    )
    f.sources(
        ('NASA VNP02DNB', L1B),
        ('Калибровка VIIRS', CALIBRATION),
        ('Наблюдаемое паразитное размытие и фон', HALO_PAPER),
    )

    f.page('07  /  НЕЗАВИСИМЫЙ КОНТРОЛЬ', 'Работает ли сравнение двух сторон')
    f.figure(
        'azimuth-contrast',
        'Разность радиансностей двух сторон вдоль азимута спутника, в нВт/(см²·ср). 49 отложенных разностей в сопоставленных радиальных и поперечных интервалах, одна точка на пару групп.',
    )
    f.p(
        'Вычитал среднее дальней стороны из средней ближней на одинаковых радиусах. '
        'Постоянный фон сокращается. Асимметрия источников, градиенты атмосферы и неровное покрытие могут оставаться. '
        'Наличие двух пригодных сторон для подгонки проверял отдельно. '
        'У двух декабрьских пролётов Мадрида симметричной поддержки для обучения нет.'
    )
    checks = []
    for method, label in (
        ('contrast_conserve', 'По разности сторон при постоянном F'),
        ('leave_pass_out_original', 'Коэффициент с другой ночи того же месяца'),
        (
            'leave_pass_out_contrast_conserve',
            'Другие ночи, по сторонам и при постоянном F',
        ),
    ):
        m = compare[method]
        old = m['control']
        bs, ns = m['satellite'][old], m['satellite'][method]
        bg, ng = m['ground'][old], m['ground'][method]
        checks.append(
            (
                label,
                ns['passes'],
                ng['n'],
                f'{number(bs["halo_mae"], 4)} → {number(ns["halo_mae"], 4)}',
                f'{number(bg["mae_pct"])} → {number(ng["mae_pct"])}',
            )
        )
    f.table(
        [
            'Подгонка и сравнение с H2 на общих данных',
            'Пролёты',
            'Замеры',
            'MAE ореола',
            'MAE земли, %',
        ],
        checks,
        (0.49, 0.105, 0.105, 0.15, 0.15),
    )
    f.p(
        'При переносе на другой пролёт знак и сумма яркостей ореола не участвовали в переоценке κ. '
        'Для новой сцены подгонялись только соседние источники и фон по участкам вне целевого ореола. '
        'Сравнение с H2 использует такой же отбор. В Тусоне это включает разные известные режимы освещения.'
    )
    f.p(
        'Сохранение потока по абсолютному ореолу упёрлось в границу κ=8 на 10/10 пролётах. '
        'У разницы сторон без ограничения мощности оценки κ от −1 до почти 58. '
        'Небольшой выигрыш наземной MAE на 32 замерах в последней строке сопровождается ухудшением смещения '
        'и ограниченной выборкой. Форму, измеренную по этим данным, пока нельзя считать устойчивой.'
    )
    f.sources(
        ('Метод на противоположных сторонах города', LOW_ANGLE),
        ('Информация из нескольких ракурсов', MULTI_ANGLE),
    )

    f.page('08  /  ВЫВОД И МОДЕЛЬ', 'Что именно можно извлечь из ореола')
    f.p(
        'Ищу долю восходящей мощности ниже 10° и 25° в измеряемой полосе. '
        'Представляю излучение как сумму положительных угловых базисов, нормированных по потоку. '
        'Абсолютный поток и доли я оцениваю отдельно. Если город асимметричен, к диаграмме добавляются '
        'простые азимутальные компоненты, проверяемые при наблюдении с разных направлений.'
    )
    f.box(
        'L(x, μ, φ, λ, t) = Σ a<sub>j</sub>(x,t) s<sub>j</sub>(λ) p<sub>j</sub>(μ,φ). '
        'Каждый базис нормирован условием ∫ μ p<sub>j</sub> dΩ = 1. '
        'Для каждого пролёта переношу L через свою атмосферу, '
        'интегрирую по отклику прибора и учитываю форму его пространственного отклика.'
    )
    f.p(
        'Пусть K<sub>v</sub> — матрица переноса этих базисов к пикселям пролёта v, '
        'b<sub>v</sub> — остаточный фон. Для одного стабильного режима оцениваю общие веса a ≥ 0 '
        'по остаткам r<sub>v</sub> = y<sub>v</sub> − K<sub>v</sub>a − b<sub>v</sub>. '
        'Минимизирую сумму r<sub>v</sub><super>T</super>C<sub>v</sub><super>−1</super>r<sub>v</sub> '
        'с ограничениями на атмосферу из независимых измерений. C<sub>v</sub> учитывает '
        'общие ошибки соседних пикселей. Долю ниже 25° получаю из интегралов базисов по этому диапазону и всей полусфере.'
    )
    f.p(
        'Выделяю яркое ядро для контроля абсолютного масштаба, а ореол использую одновременно по радиусу '
        'и по знаку относительно визирования. При смене азимута положение усиления от рассеяния меняется. '
        'Соседнее поселение остаётся на прежних координатах. Это одно из измеримых различий, '
        'которое позволяет разделить географию источников и влияние атмосферы.'
    )
    f.p(
        'Для ночи нужны AOD и независимое ограничение на высоту и фазовую функцию аэрозоля, '
        'например лидар вместе с AERONET или проверенным реанализом. '
        'Калиброванные полосы синего и красного или данные о спектре ламп нужны для цвета. '
        'Геометрия и яркость должны соответствовать одной фазе работы освещения. '
        'Для Тусона выбирались замеры до полуночи, а пролёты той же ночи были на 1,5–3 часа позже. '
        'Штатная команда фонарям 90→60%, в эксперименте также были 30% и 100%. '
        'В исходной работе переход 90→30% менял полную наземную яркость на несколько процентов. '
        'Процент команды фонарям нельзя применять множителем ко всему городу.'
    )
    f.p(
        'После подгонки варьирую в допустимых пределах фон, распределение аэрозоля, карту слабых источников '
        'и приборный отклик и пересчитываю интервал доли низкоуглового света. '
        'Проверяю на другом азимуте той же фазы освещения и на независимом фотометре. '
        'В этой выборке мощности и доли остаются модельными оценками. '
        'H2 остаётся опорным прогнозом, а энергетически ограниченный вариант показывает, '
        'какая часть результата переживает запрет на добавление света из спутникового остатка.'
    )
    f.sources(
        ('Исходный эксперимент с фонарями', TUCSON_ORBIT),
        ('Наземный опыт Тусона', TUCSON_SKY),
        ('Ореол зависит и от параметров аэрозоля', ANGLE_ATMOSPHERE),
    )

    f.page('09  /  СВЕРКА', 'Разрезы, проверки и границы вывода')
    comparisons = []
    for key, label in (
        ('All', 'Все 72'),
        ('same_night', 'Та же ночь по протоколу'),
        ('Berlin/city_dominant', 'Берлин, преобладает город'),
        ('Madrid/city_dominant', 'Мадрид, преобладает город'),
        ('Tucson/city_dominant', 'Тусон, преобладает город'),
        ('Tucson/less_cloud', 'Тусон, устойчивость к облакам'),
    ):
        comparisons.append(
            (
                label,
                g[key]['n'],
                ground(key, 'h2'),
                ground(key, 'original'),
                ground(key, 'conserve'),
            )
        )
    f.table(
        ['Наземная выборка', 'N', 'H2', 'Подогнанный хвост', 'Постоянная мощность'],
        comparisons,
        (0.34, 0.07, 0.16, 0.22, 0.21),
    )
    f.p(
        'Влияние спектра проверил ещё одним способом. Только у подобранной добавки хвоста заменил '
        'исходную равновесную смесь в полосе DNB крайними архивными шаблонами натриевого света HP1 '
        'и белого LED‑B2 при той же DNB-нормировке. На 72 точках получились '
        f'{ground("All", "original_tail_HP1")} и {ground("All", "original_tail_LED-B2")}% '
        'MAE / смещения. Результат ограничен двумя гипотезами о спектре. '
        'У реальных городов смесь может зависеть от направления и времени.'
    )
    nums = compare['second_seed']['satellite']
    f.p(
        'На трёх контрольных пролётах заменил случайное зерно. MAE совокупности этих же пролётов '
        f'{number(nums["original"]["halo_mae"], 4)} → {number(nums["second_seed"]["halo_mae"], 4)} '
        'нВт/(см²·ср). Это контроль численного разброса при фиксированных физических входах. '
        'Повторные пролёты и приборы на одной точке учитываю как сравнения, а не как статистически независимые эксперименты.'
    )
    f.p(
        'Максимальная высота Солнца на наземных замерах составила '
        f'{number(g["All"]["max_sun_alt_deg"], 1)}°, Луны — '
        f'{number(g["All"]["max_moon_alt_deg"], 1)}°. Берлинский дальний край маршрута близок к порогу '
        'астрономических сумерек. Лунный максимум относится к дальним отсчётам Мадрида. '
        'Это ограничения отдельных точек, при том что основной избыток Берлина и ближнего Тусона '
        'сохраняется даже при удалении всего естественного фона.'
    )
    f.p(
        'В репозитории сохранены точечные результаты, ядра по порядкам в виде агрегированных откликов, '
        'проверка L1B, сценарии атмосферы, погодный архив, хеши и скрипты воспроизведения. '
        'Пригодных спутниковых данных для оценки города Балагер по-прежнему нет. '
        'PSF, рельеф, субпиксельные огни и ночной профиль аэрозоля измерялись здесь лишь косвенно или проверялись сценариями.'
    )
    f.sources(
        ('Black Marble C2', NASA),
        ('Наземные наблюдения Мадрида', 'https://doi.org/10.1016/j.jqsrt.2016.02.029'),
        ('Маршрут Берлина', 'https://doi.org/10.3390/rs12203412'),
    )
    f.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('satellite', type=Path)
    parser.add_argument('native_check', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    b, s, n = (read_json(v) for v in (args.summary, args.satellite, args.native_check))
    args.output.mkdir(parents=True, exist_ok=True)
    draw_figures(args.output, b, s['scenes'])
    render_results(args.output, b, s, n)


if __name__ == '__main__':
    main()
