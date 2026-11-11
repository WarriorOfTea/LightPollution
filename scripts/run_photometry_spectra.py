"""Reproduce stages 1–2 on pinned NASA maps and public spectral/ground data."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from astropy.io.votable import parse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lightpollution.adapters.sources import LocalSourceRepository
from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import read_json, write_json
from lightpollution.application.photometry import fit_dataset
from lightpollution.application.planner import Planner, compare_results, export_result
from lightpollution.domain.scene import SceneSpec
from lightpollution.models.spectra import (band_integral, integrate_spectral_direction,
                                         mix_templates, response_summary)
from lightpollution.physics import Atmosphere, Observer, SolverSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/photometry-spectra-20261111'))
    parser.add_argument('--workspace', type=Path, default=Path('data/photometry-spectra-20261111'))
    parser.add_argument('--raw-directory', type=Path, default=Path('data/blackmarble'))
    parser.add_argument('--spectral-raw', type=Path, default=Path('scratch/spectral-data-20261111'))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    criteria = {'declared_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'spectral_quadrature_relative_tolerance': .01,
        'directions': [[0, 90], [180, 45]], 'lamp_templates': 14, 'responses': 4,
        'scope': '5 nm vs 1 nm integration at these rays only; not physical or full-sky accuracy.'}
    if not (output/'predeclared-criteria.json').exists():
        write_json(output/'predeclared-criteria.json', criteria)
    planner = Planner(args.workspace, raw_directory=args.raw_directory)
    library = load_spectral_library()
    base = {'schema_version': 2, 'region': 'petersburg-2025', 'period': '2025', 'mode': 'undated',
        'moon': {'mode': 'none'}, 'observer': {'latitude_deg': 60.3, 'longitude_deg': 30.315, 'height_m': 20},
        'directions': [[0, 90], [0, 45], [90, 45], [180, 45], [270, 45]], 'profile': 'standard'}
    results = {}
    def run(name, data):
        result = planner.run(SceneSpec.from_mapping(data))
        export_result(result, output/name)
        results[name] = result.value
        print(name, 'completed', round(result.value['runtime']['elapsed_seconds'], 3), 'seconds', flush=True)
        return result.value

    north = run('petersburg-reference', base)
    moved = deepcopy(base)
    moved['observer']['latitude_deg'] = 60.65
    further = run('petersburg-further-north', moved)
    assert north['calibration']['source_coefficient'] == further['calibration']['source_coefficient']
    write_json(output/'spatial-comparison.json', compare_results(north, further))
    madrid = deepcopy(base)
    madrid.update(region='madrid-2017', period='2017-03', directions=[[0, 90]],
                  observer={'latitude_deg': 40.451111, 'longitude_deg': -3.726111, 'height_m': 20})
    march = run('madrid-2017-03', madrid)
    madrid['period'] = '2017-11'
    november = run('madrid-2017-11', madrid)
    selected = read_json(Path('artifacts/seasonal-ground-20261110/astmon-publication-qa.json'))['records']
    measurements = [{'id': row['utc'], 'site_id': 'UCM-AstMon', 'case_id': row['period'], 'utc': row['utc'],
        'night_id': row['night_id'], 'role': 'training' if row['period'] == '2017-03' else 'validation',
        'observed_v_mag_arcsec2': row['mag_arcsec2'], 'cloud_free_confirmed': row['cloud_free_confirmed'],
        'qa_source': row['cloud_qa_source']} for row in selected]
    dataset = {'schema_version': 1, 'band': 'Johnson-V', 'timezone': 'Europe/Madrid',
        'source': 'AstMon, doi:10.5281/zenodo.4633001; Robles et al. 2021 Figure 5/6 clear interval transcription.',
        'rights': 'CC BY 4.0, authors of Zenodo 4633001.',
        'cases': {'2017-03': 'madrid-2017-03/result.json', '2017-11': 'madrid-2017-11/result.json'},
        'samples': measurements,
        'model_height_note': '20 m over common hypothetical ground; geographic 650 m plateau and DEM not reconstructed.'}
    write_json(output/'ground-dataset.json', dataset)
    calibration = fit_dataset(dataset, {'2017-03': march, '2017-11': november})
    write_json(output/'ground-calibration.json', calibration)
    calibrated_scene = deepcopy(madrid)
    calibrated_scene['photometric_calibration'] = calibration
    calibrated = run('madrid-calibrated-validation', calibrated_scene)
    print('Ground control:', calibration['validation']['mae_mag'], 'mag;',
          calibration['validation']['nights'], 'night;', calibration['validation']['samples'], 'samples', flush=True)

    spectral = deepcopy(base)
    spectral.pop('moon')
    spectral.update(mode='dated', time='2026-11-09T21:00:00Z', spectrum={'mix': {'LED-B3': 1.}})
    led = run('spectral-led', spectral)
    spectral['spectrum']['mix'] = {'HP1': 1.}
    sodium = run('spectral-sodium', spectral)
    spectral['spectrum']['mix'] = {'LED-B5': 1.}
    cool = run('spectral-cool-led', spectral)
    assert led['spectral']['source_visible_power_scale_W'] == sodium['spectral']['source_visible_power_scale_W']
    write_json(output/'lamp-comparison.json', {'sodium_vs_reference_LED': compare_results(led, sodium),
                                               'cool_LED_vs_reference_LED': compare_results(led, cool)})

    # Two competing spectral hypotheses fitted to the same V data. A successful
    # V fit must not be misrepresented as identifying the lamp spectrum.
    spectral_fits = {}
    for lamp in ['LED-B3', 'HP1']:
        cases = {}
        for period in ['2017-03', '2017-11']:
            data = deepcopy(madrid)
            data.update(period=period, spectrum={'mix': {lamp: 1.}})
            cases[period] = run('madrid-'+lamp+'-'+period, data)
        dataset_for_spectrum = deepcopy(dataset)
        dataset_for_spectrum['cases'] = {p: 'madrid-'+lamp+'-'+p+'/result.json' for p in cases}
        fitted = fit_dataset(dataset_for_spectrum, cases)
        write_json(output/('ground-dataset-'+lamp+'.json'), dataset_for_spectrum)
        write_json(output/('ground-calibration-'+lamp+'.json'), fitted)
        data['photometric_calibration'] = fitted
        validation_case = run('madrid-'+lamp+'-calibrated-validation', data)
        spectral_fits[lamp] = {'training_nights': fitted['training']['nights'],
            'training_samples': fitted['training']['samples'], 'validation_nights': fitted['validation']['nights'],
            'validation_samples': fitted['validation']['samples'],
            'validation_bias_mag': fitted['validation']['bias_mag'],
            'source_coefficient': fitted['source_coefficient'],
            'validation_zenith': validation_case['spectral']['directions'][0],
            'lamp_inventory_identified': False}
    write_json(output/'competing-spectral-fits.json', spectral_fits)

    catalogue = {int(r['hr']): r for r in csv.DictReader(Path('src/lightpollution/catalogues/bsc5.csv').open())}
    checks = []
    for info in library.provenance['stars']:
        raw = library.measured[info['hr']]
        synthetic_v = float(-2.5*np.log10(band_integral(library.wavelengths, raw,
                                                library.responses['johnson_v'])/library.zero_v))
        bsc_v = float(catalogue[info['hr']]['vmag'])
        checks.append({**info, 'raw_synthetic_V': synthetic_v, 'bsc_V': bsc_v,
                       'residual_synthetic_minus_BSC_mag': synthetic_v-bsc_v,
                       'renormalised_to_BSC_for_this_check': False})
    xp = [r for r in checks if r['source'].startswith('Gaia')]
    residual = np.array([r['residual_synthetic_minus_BSC_mag'] for r in xp])
    stellar_check = {'records': checks, 'gaia_targets': library.provenance['gaia_targets'],
        'gaia_spectra_obtained': len(xp), 'gaia_rejected': library.provenance['gaia_rejections'],
        'gaia_mean_absolute_V_residual_mag': float(np.mean(abs(residual))),
        'gaia_max_absolute_V_residual_mag': float(np.max(abs(residual))),
        'physical_accuracy_certified': False,
        'note': 'Raw absolute XP flux vs BSC, before V normalisation. Generic filter, epochs, bright-source XP and catalogue systematics remain.'}
    write_json(output/'stellar-validation.json', stellar_check)

    # Finer spectral quadrature with original CIE response tables, declared above.
    fine_axis = np.arange(360., 831.)
    fine_templates, fine_responses = {}, {}
    for filename in ['CIE_illum_LEDs.csv', 'CIE_illum_HPs.csv']:
        table = np.loadtxt(args.spectral_raw/filename, delimiter=',')
        metadata = read_json(args.spectral_raw/(filename+'_metadata_v2.json'))
        for j, header in enumerate(metadata['datatableInfo']['columnHeaders'][1:], 1):
            fine_templates[header['title']] = np.interp(fine_axis, table[:, 0], table[:, j], left=0, right=0)
    for name in ['photopic', 'scotopic']:
        table = np.loadtxt(args.spectral_raw/f'CIE_sle_{name}.csv', delimiter=',')
        fine_responses[name] = np.interp(fine_axis, table[:, 0], table[:, 1], left=0, right=0)
    for name in ['v', 'b']:
        table = parse(args.spectral_raw/f'johnson-{name}.xml').get_first_table().array
        fine_responses['johnson_'+name] = np.interp(fine_axis, table['Wavelength']/10,
                                                 table['Transmission'], left=0, right=0)
    repository = LocalSourceRepository(args.workspace, args.raw_directory)
    group = repository.prepare('petersburg-2025', '2025', Observer())
    sources, _ = group.select('2025', 12)
    records = []
    for az, alt in criteria['directions']:
        response = integrate_spectral_direction(Observer(), sources, az, alt, Atmosphere(),
                                                SolverSettings(192, 24), fine_axis, 1.3)
        for name in library.templates:
            fine_mix = mix_templates(fine_axis, fine_templates, {name: 1.})
            coarse_mix = library.mix({name: 1.})
            for band in library.responses:
                fine = float(band_integral(fine_axis, response*fine_mix, fine_responses[band]))
                coarse = float(band_integral(library.wavelengths, response[::5]*coarse_mix,
                                              library.responses[band]))
                difference = abs(coarse-fine)/fine
                records.append({'azimuth_deg': az, 'altitude_deg': alt, 'lamp': name, 'response': band,
                                'fine': fine, 'coarse': coarse, 'relative_difference': difference,
                                'passed_1percent': difference <= .01})
        print('Spectral quadrature ray', az, alt, 'done', flush=True)
    convergence = {'relative_tolerance': .01, 'total': len(records),
        'passed': sum(r['passed_1percent'] for r in records),
        'maximum_relative_difference': max(r['relative_difference'] for r in records),
        'records': records, 'scope': criteria['scope']}
    write_json(output/'spectral-quadrature.json', convergence)

    summary = {'version': '0.3.0', 'real_ground_fit': calibration,
        'spatial_reference_observer': north['request']['observer'],
        'spatial_other_observer': further['request']['observer'],
        'reference_zenith_artificial_V': float(22-2.5*np.log10(north['sky']['directions'][0]['artificial_relative'])),
        'other_zenith_artificial_V': float(22-2.5*np.log10(further['sky']['directions'][0]['artificial_relative'])),
        'common_source_coefficient': north['calibration']['source_coefficient'],
        'spectral_lamp_summary': {name: results[name]['spectral']['directions'][0]
                                  for name in ['spectral-led', 'spectral-sodium', 'spectral-cool-led']},
        'stellar_coverage': led['spectral']['stellar_summary'], 'stellar_validation': stellar_check,
        'competing_spectral_ground_fits': spectral_fits,
        'spectral_quadrature': {k: v for k, v in convergence.items() if k != 'records'},
        'runtime_seconds': {k: v['runtime']['elapsed_seconds'] for k, v in results.items()},
        'new_NASA_downloads': 0, 'new_ground_measurements': 0,
        'physical_error_bound': None, 'independent_spatial_ground_validation_available': False}
    write_json(output/'results-summary.json', summary)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for name, colour in [('LED-B3', '#007d89'), ('LED-B5', '#3567b7'), ('HP1', '#bc7922')]:
        axes[0].plot(library.wavelengths, library.mix({name: 1.}), label=name, color=colour)
    axes[0].set(xlim=(380, 780), xlabel='Длина волны, нм', ylabel='Доля мощности на нм', title='Заданные спектры ламп')
    axes[0].legend()
    for name, title in [('johnson_v', 'Johnson V'), ('photopic', 'Дневной CIE'), ('scotopic', 'Ночной CIE')]:
        axes[1].plot(library.wavelengths, library.responses[name], label=title)
    axes[1].set(xlim=(380, 780), xlabel='Длина волны, нм', ylabel='Нормированный отклик', title='Разные функции чувствительности')
    axes[1].legend()
    axes[2].bar([str(r['hr']) for r in xp], residual, color='#007d89')
    axes[2].axhline(0, color='black', linewidth=.7)
    axes[2].set(xlabel='Номер HR в BSC', ylabel='V из Gaia − V каталога, mag', title='Независимое сравнение до нормировки')
    figure.suptitle('LightPollution 0.3 · спектральный численный эксперимент; изображение неба не моделируется', fontsize=12)
    figure.savefig(output/'spectral-experiment.png', dpi=150)
    plt.close(figure)
    print('Spectral checks:', convergence['passed'], '/', convergence['total'],
          'max relative', convergence['maximum_relative_difference'], flush=True)


if __name__ == '__main__':
    main()
