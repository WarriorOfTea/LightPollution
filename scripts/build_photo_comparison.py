"""Build scientific comparison figures from real photos and computed radiances.

Photograph colours are preserved. The model's colour map encodes a scalar, not
RGB sky colour. No exposure, source power or atmosphere is fitted to the JPEG.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image

from lightpollution.physics import Atmosphere, Observer
from lightpollution.skyview import (DisplayScenario, _draw_dome, horizontal_stars,
                                    load_stars, stellar_extinction)


ROOT = Path('artifacts/photo-comparison')
BACKGROUND = '#080e19'
TEXT = '#e2e8f0'
MUTED = '#acbace'


def gallery():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=BACKGROUND)
    fig.subplots_adjust(left=.035, right=.965, top=.87, bottom=.12, hspace=.28, wspace=.13)
    examples = [
        ('galactic-2012.jpg', 'Звёзды и небольшой световой купол',
         'Galactic · Nicholas A. Tonelli · CC BY 2.0\nКомпактная камера; 20 с, ISO 3200. Остаточные сумерки.'),
        ('sweet-spot-2014.jpg', 'Ясное поле со звёздами и засветкой',
         'Sweet spot · Christian Reimer · CC BY-SA 2.0\n15 с, ISO 6400. Опубликованная точка — наш основной пример.'),
        ('luhasoo-2015.jpg', 'Млечный Путь и цветные купола у горизонта',
         'Luhasoo · Martin Mark · CC BY-SA 4.0\n20 с, ISO 1600. Точная точка камеры не опубликована.'),
        ('ruka-clouds-2013.jpg', 'Контрпример: низкие облака, снег и прожекторы',
         'Orange Skies · Timo Newton-Syms · CC BY-SA 2.0\nРука, 2013. Для проверки ясной атмосферы не подходит.'),
    ]
    for ax, (filename, title, caption) in zip(axes.flat, examples, strict=True):
        ax.set_facecolor(BACKGROUND)
        ax.imshow(Image.open(ROOT/'references'/filename))
        ax.axis('off')
        ax.set_title(title, fontsize=12, color=TEXT, pad=9)
        ax.text(.5, -.04, caption, transform=ax.transAxes, ha='center', va='top',
                fontsize=9, color=MUTED)
    fig.text(.5, .96, 'Реальные фотографии ночного неба', ha='center', fontsize=22)
    fig.text(.5, .925, 'Длинная выдержка и обработка камеры отличаются от наблюдения глазом.',
             ha='center', fontsize=11, color=MUTED)
    fig.text(.5, .048, 'Фотографии не перекрашивались. Ссылки, лицензии и изменения размеров: references/SOURCES.md.',
             ha='center', fontsize=10, color=MUTED)
    fig.text(.5, .023, 'Это подборка наблюдений, а не изображения, сгенерированные физической моделью.',
             ha='center', fontsize=10, color=MUTED)
    fig.savefig(ROOT/'real-sky-examples.png', dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


def comparison():
    case = json.loads((ROOT/'case.json').read_text())
    directory = Path(case['model_directory'])
    with np.load(directory/'sky-grid.npz', allow_pickle=False) as archive:
        az, alt, values = [archive[k] for k in ('azimuth_deg', 'altitude_deg', 'radiance')]
        transfer = json.loads(str(archive['metadata_json']))
    if transfer['observer'] != case['observer'] or transfer['atmosphere'] != case['atmosphere']:
        raise ValueError('Sky calculation and photograph case use different parameters.')
    if transfer['provenance']['region_metadata']['year'] != case['year']:
        raise ValueError('Sky calculation and photograph case use different years.')
    observer = Observer(**case['observer'])
    stars, _ = load_stars()
    star_az, star_alt, astrometry = horizontal_stars(stars, observer, case['utc_assumed'])
    _, _, alternate_astrometry = horizontal_stars(stars, observer, case['alternate_utc'])
    above = star_alt>=0
    extinction = np.full(len(star_alt), np.inf)
    extinction[above] = stellar_extinction(star_alt[above], observer, Atmosphere(**case['atmosphere']))
    field = values/values[-1, 0]
    fig = plt.figure(figsize=(18, 12), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1.05, 1.05],
                            left=.025, right=.98, top=.80, bottom=.14, hspace=.32, wspace=.25)
    photo = fig.add_subplot(grid[:, 0])
    photo.imshow(Image.open(case['reference_image']))
    photo.axis('off')
    photo.set_title('Фотография: Sweet spot', color=TEXT, fontsize=14, pad=12)
    photo.text(.5, -.025, 'Christian Reimer · CC BY-SA 2.0\n15 с · f/3,5 · ISO 6400 · обработанный JPEG',
               transform=photo.transAxes, ha='center', va='top', color=MUTED, fontsize=10)
    polar = fig.add_subplot(grid[0, 1], projection='polar')
    polar.set_theta_zero_location('N')
    polar.set_theta_direction(1)  # Sky seen from below: east on the left.
    image = polar.pcolormesh(np.deg2rad(np.r_[az, 360]), 90-alt,
                            np.column_stack((field, field[:, 0])), shading='auto',
                            norm=LogNorm(vmin=float(min(field.min(), 1)), vmax=float(field.max())), cmap='magma')
    polar.set_facecolor(BACKGROUND)
    polar.set_ylim(0, 90)
    polar.set_thetagrids([0,90,180,270], ['С','В','Ю','З'], color=MUTED)
    polar.set_rgrids([30,60,90], ['60°','30°','0°'], color=MUTED, fontsize=8)
    polar.grid(alpha=.22)
    polar.set_title('Рассчитанная форма засветки\nЦветовая шкала — условное обозначение числа', color=TEXT, fontsize=10, pad=7)
    bar = fig.colorbar(image, ax=polar, fraction=.045, pad=.12, shrink=.8)
    bar.set_label('Искусственный радианс / радианс зенита', color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    profile = fig.add_subplot(grid[0, 2])
    for height in (1, 5, 15, 45):
        row = np.flatnonzero(alt==height)[0]
        profile.plot(np.r_[az,360], np.r_[field[row],field[row,0]], label=f'Высота {height}°')
    profile.set(xlim=(0,360), xlabel='Азимут: север 0°, восток 90°',
                ylabel='Искусственный радианс / радианс зенита',
                title='Засветка зависит от направления и высоты')
    profile.set_facecolor(BACKGROUND)
    profile.set_yscale('log')
    profile.grid(alpha=.2)
    profile.legend(fontsize=9, facecolor=BACKGROUND, labelcolor=TEXT)
    scenarios = []
    for col, magnitude in enumerate(case['display_scenarios_artificial_zenith_mag'], 1):
        ax = fig.add_subplot(grid[1, col])
        scenario = DisplayScenario(magnitude)
        visible, _, _ = _draw_dome(ax, az, alt, field, stars, star_az, star_alt,
                                   extinction, scenario, True, size=900)
        ax.set_title(f'Условный масштаб зенита {magnitude:g} mag/arcsec²\n'
                     'Звёзды и фон монохромны; масштаб не получен из фото',
                     color=TEXT, fontsize=11, pad=12)
        scenarios.append({'artificial_zenith_mag':magnitude, 'visible_catalogue_rows':int(visible.sum())})
    fig.text(.5,.963,'Фотография и модель из её опубликованной точки', ha='center', fontsize=23)
    fig.text(.5,.923,'49,532563° N · 6,549911° E · источники NASA VNP46A4.002, 2014 · ясная модельная атмосфера',
             ha='center', color=MUTED, fontsize=12)
    fig.text(.5,.888,'Кадр не совмещён по звёздам: справа весь купол, а не восстановленное поле камеры.',
             ha='center', color='#f1cd92', fontsize=12)
    fig.text(.5,.082,'Две версии даты: 14 и 22 сентября 2014. Положение звёзд показано для гипотезы 14 сентября, 20:08 UTC.',
             ha='center', color=MUTED, fontsize=11)
    fig.text(.5,.052,'Камера, Млечный Путь, реальные цвета, погода и абсолютная яркость не восстановлены. Процент физической ошибки не определён.',
             ha='center', color=MUTED, fontsize=11)
    fig.text(.5,.023,'Обе нижние панели используют одинаковую экспозицию и правило видимости. Число звёзд — результат условного правила, не измерение зрения.',
             ha='center', color=MUTED, fontsize=10)
    fig.savefig(ROOT/'photo-vs-model.png', dpi=160, facecolor=BACKGROUND)
    plt.close(fig)
    profiles = []
    for height in (1, 5, 15, 45):
        row = np.flatnonzero(alt==height)[0]
        maximum = np.argmax(field[row])
        profiles.append({'altitude_deg':height, 'sampled_peak_azimuth_deg':float(az[maximum]),
                         'peak_artificial_radiance_over_zenith':float(field[row,maximum]),
                         'peak_over_azimuthal_minimum':float(field[row].max()/field[row].min())})
    result = {'observer':case['observer'], 'source_year':case['year'],
              'astrometry_assumed':astrometry, 'astrometry_alternate':alternate_astrometry,
              'photograph_registration_verified':False, 'colour_prediction_implemented':False,
              'physical_accuracy_bound':None, 'display_scenarios':scenarios,
              'profiles':profiles,
              'comparison_scope':'Same published location, different fields of view. '
                                 'No brightness, exposure, colour or atmosphere fitted to the photograph.',
              'output_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ('real-sky-examples.png','photo-vs-model.png')}}
    (ROOT/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':TEXT, 'axes.labelcolor':MUTED,
                         'axes.titlecolor':TEXT, 'xtick.color':MUTED, 'ytick.color':MUTED,
                         'axes.edgecolor':MUTED})
    gallery()
    comparison()


if __name__ == '__main__':
    main()
