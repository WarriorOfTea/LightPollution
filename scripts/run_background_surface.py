"""Reproduce stages 3–4: bounded twilight, airglow geometry and fixed-lamp snow."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.storage import write_json
from lightpollution.application.planner import Planner, compare_results, export_result
from lightpollution.domain.scene import SceneSpec
from lightpollution.models.background import (BackgroundModel, background_config,
    direct_optical_depth, patat_v_magnitude, van_rhijn)
from lightpollution.models.surface import surface_balance, surface_config
from lightpollution.observation import observation_context
from lightpollution.physics import Atmosphere, EARTH_RADIUS_M, Observer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/natural-twilight-snow-20261111'))
    parser.add_argument('--workspace', type=Path, default=Path('data/photometry-spectra-20261111'))
    parser.add_argument('--raw-directory', type=Path, default=Path('data/blackmarble'))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'predeclared-criteria.json').exists():
        write_json(output/'predeclared-criteria.json', {
            'declared_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
            'energy_balance_absolute_tolerance': 1e-12,
            'natural_slant_tau_relative_tolerance': 1e-6,
            'tau_altitudes_deg': [10, 20, 30, 45, 60, 90],
            'sun_depressions_deg': [5, 6, 9, 12, 15],
            'site': '60.3 N, 30.315 E, 20 m above hypothetical common ground',
            'field_validation': 'No independent clear-night snow or twilight observations supplied; no physical accuracy score.'})
    planner = Planner(args.workspace, raw_directory=args.raw_directory)
    base = {'schema_version': 2, 'region': 'petersburg-2025', 'period': '2025',
        'mode': 'undated', 'moon': {'mode': 'none'},
        'observer': {'latitude_deg': 60.3, 'longitude_deg': 30.315, 'height_m': 20},
        'directions': [[0, 90], [0, 50], [90, 45], [180, 45], [270, 45]],
        'background': {}, 'spectrum': {'mix': {'LED-B3': 1}}, 'profile': 'standard'}
    results = {}

    def run(name, data):
        result = planner.run(SceneSpec.from_mapping(data))
        export_result(result, output/name)
        results[name] = result.value
        print(name, round(result.value['runtime']['elapsed_seconds'], 3), 'seconds', flush=True)
        return result.value

    for fraction, name in [(0., 'bare'), (.5, 'half-snow'), (1., 'snow')]:
        data = deepcopy(base)
        data['surface'] = {'snow_fraction': fraction, 'direct_horizontal_fraction': .1}
        run(name, data)
    assert len({v['comparison_basis_id'] for v in results.values()}) == 1
    comparisons = {name: compare_results(results['bare'], results[name]) for name in ['half-snow', 'snow']}
    write_json(output/'snow-comparisons.json', comparisons)
    # An independent exact limit: both lobes Lambertian => city changes by
    # the upward power ratio at every wavelength and direction.
    lambert_bare = deepcopy(base)
    lambert_bare['surface'] = {}
    lb = run('lambert-bare', lambert_bare)
    lambert_bare['surface']['snow_fraction'] = 1
    ls = run('lambert-snow', lambert_bare)
    factor = surface_balance(surface_config({'snow_fraction': 1}))['upward_power_over_snow_free']
    lambert_errors = [abs(b['artificial_relative']/a['artificial_relative']/factor-1)
        for a, b in zip(lb['sky']['directions'], ls['sky']['directions'], strict=True)]
    # A real ephemeris for the white-night site; the empirical brightness
    # comes from Paranal, not a measured Petersburg summer sky.
    summer = deepcopy(base)
    summer.pop('moon')
    summer.update(mode='dated', time='2026-06-21T22:00:00Z', background={'twilight': 'patat2006'})
    white = run('white-night', summer)
    transition = deepcopy(summer)
    transition['time'] = '2026-09-01T01:00:00Z'
    # Find one explicit timestamp in the unmodelled 15–18 degree transition.
    from datetime import timedelta
    start = datetime.fromisoformat('2026-09-01T00:00:00+00:00')
    for minutes in range(0, 180, 5):
        utc = (start+timedelta(minutes=minutes)).isoformat().replace('+00:00', 'Z')
        sun_alt = observation_context(Observer(**base['observer']), utc, 0, 90)['sun']['altitude_deg']
        if -18 < sun_alt < -15:
            transition['time'] = utc
            break
    else:
        raise RuntimeError('No transition test timestamp found; do not silently substitute a dark night.')
    gap = run('deep-transition-unsupported', transition)
    assert all(not r['supported'] and r['total_relative'] is None for r in gap['sky']['directions'])

    observer, atmosphere = Observer(**base['observer']), Atmosphere()
    altitudes = np.asarray([10., 20., 30., 45., 60., 90.])
    tau96 = direct_optical_depth(altitudes, observer, atmosphere)
    r0, r1 = EARTH_RADIUS_M+observer.height_m, EARTH_RADIUS_M+atmosphere.top_m
    projection = r0*np.sin(np.deg2rad(altitudes))
    length = (r1*r1-r0*r0)/(np.sqrt(projection*projection+r1*r1-r0*r0)+projection)
    tau192 = atmosphere.optical_depth_radial(r0*r0, r1*r1, length*length, order=192)
    relative_error = np.abs(tau96/tau192-1)
    assert np.all(relative_error < 1e-6)
    with (output/'natural-geometry.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['altitude_deg', 'van_rhijn', 'tau_96', 'tau_192', 'relative_tau_error'])
        writer.writerows(zip(altitudes, van_rhijn(altitudes, 90000, 20), tau96, tau192, relative_error, strict=True))
    energy_errors = []
    for u in [0, .01, .15, .5, 1]:
        for bare in [.01, .15, 1]:
            for snow in [0, .8, 1]:
                for fraction in [0, .25, .5, .75, 1]:
                    b = surface_balance(surface_config({'direct_upward_fraction': u, 'bare_albedo': bare,
                        'snow_albedo': snow, 'snow_fraction': fraction}))
                    error = abs(b['direct_fraction_of_lamp_power']+b['reflected_fraction_of_lamp_power']+
                                b['absorbed_fraction_of_lamp_power']-1)
                    assert error < 1e-12 and 0 <= b['upward_fraction_of_lamp_power'] <= 1
                    energy_errors.append(error)
    table = [{'sun_depression_deg': depression, 'total_v_mag_arcsec2': float(patat_v_magnitude(depression)),
        'total_over_reference_night': float(10**(-.4*(patat_v_magnitude(depression)-21.6)))}
        for depression in [5, 6, 9, 12, 15]]
    write_json(output/'patat-reproduction.json', {'source': 'Patat et al. 2006 Table 1 V',
        'coefficients': [11.84, 1.518, -.057], 'rms_mag': .18, 'rows': table,
        'local_field_validation': False, 'night_floor': 'Patat 2003 V=21.6 mag/arcsec2'})
    summary = {'completed_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'cases': len(results), 'snow': {name: {'surface': value['observation']['surface_model'],
            'zenith': value['sky']['directions'][0]} for name, value in results.items() if 'surface_model' in value['observation']},
        'white_night': {'utc': summer['time'], 'geometry': white['observation']['geometry'],
            'supported_directions': sum(r['supported'] for r in white['sky']['directions']),
            'total_directions': len(white['sky']['directions']), 'zenith': white['sky']['directions'][0]},
        'deep_transition': {'utc': transition['time'], 'sun_altitude_deg': gap['observation']['geometry']['sun']['altitude_deg'],
                            'supported_directions': 0, 'total_directions': len(gap['sky']['directions'])},
        'numerical_checks': {'energy_balance_cases': len(energy_errors), 'energy_balance_max_absolute_error': max(energy_errors),
            'tau_directions': len(altitudes), 'tau_max_relative_error': float(max(relative_error)),
            'lambert_snow_directions': len(lambert_errors), 'lambert_snow_max_relative_error': max(lambert_errors)},
        'field_validation': {'independent_clear_sky_snow_nights': 0, 'independent_twilight_nights': 0,
            'reason': 'No matched local clear-night observations with measured snow/albedo/atmosphere obtained.',
            'accuracy_metric': None}, 'new_nasa_files_downloaded': 0,
        'snow_maps_loaded': False, 'rights': 'Original computed results; source/library attributions remain in result provenance.'}
    write_json(output/'summary.json', summary)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    depression = np.linspace(5, 15, 150)
    axes[0].plot(depression, patat_v_magnitude(depression), color='#d38920')
    axes[0].axhline(21.6, ls='--', color='#555', label='Ночной сценарий 21,6')
    axes[0].set(xlabel='Солнце под горизонтом, °', ylabel='V mag/arcsec² (меньше — ярче)',
                title='Опубликованный шаблон Paranal')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    alts = np.linspace(10, 90, 100)
    natural = BackgroundModel(background_config({}), observer, atmosphere, 21.6).evaluate(np.zeros_like(alts), alts, None)
    q0 = natural[-1]['natural_relative']
    axes[1].plot(alts, [r['airglow_relative']/q0 for r in natural], label='Airglow + прямое ослабление')
    axes[1].plot(alts, [r['diffuse_residual_relative']/q0 for r in natural], label='Остаточный диффузный фон')
    axes[1].plot(alts, [r['natural_relative']/q0 for r in natural], color='#222', label='Сумма')
    axes[1].set(xlabel='Высота взгляда, °', ylabel='Отношение к ночному зениту', title='Сценарий сферического слоя')
    axes[1].legend(fontsize=8)
    x = np.arange(len(base['directions']))
    for name, label in [('bare', 'Без снега'), ('half-snow', 'Покрытие 50%'), ('snow', 'Покрытие 100%')]:
        ratios = [b['artificial_relative']/a['artificial_relative'] for a, b in
                  zip(results['bare']['sky']['directions'], results[name]['sky']['directions'], strict=True)]
        axes[2].plot(x, ratios, marker='o', label=label)
    axes[2].set_xticks(x, ['Зенит', 'Север 50°', 'Восток 45°', 'Юг 45°', 'Запад 45°'], rotation=25)
    axes[2].set(ylabel='Городской фон / фон без снега', title='NASA: те же лампы, меняется отражение')
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle('Этапы 3–4: численные сценарии, не фотография и не проверенный местный прогноз', fontsize=12)
    fig.tight_layout(rect=(0, .02, 1, .93))
    fig.savefig(output/'background-snow-experiment.png', dpi=160)
    plt.close(fig)
    print(summary['numerical_checks'], flush=True)


if __name__ == '__main__':
    main()
