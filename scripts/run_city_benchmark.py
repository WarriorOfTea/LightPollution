"""Paired Madrid zenith diagnostics: default inputs versus DAYTIME AERONET.

The observations are the previously published clear/moonless windows, so this
is a development check, not a new blind test. AOD and g are changed separately;
no cloud, Moon, spectrum, terrain or snow model is adjusted to absorb residuals.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import io
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.calibration import nightly_values
from lightpollution.data import load_region
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import regional_sources

ROOT = Path(__file__).resolve().parents[1]


def aeronet_rows(path, day):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('AERONET_Site,Date('))
    rows = [r for r in csv.DictReader(io.StringIO('\n'.join(lines[start:]))) if r['Date(dd:mm:yyyy)'] == day]
    return sorted(rows, key=lambda r: r['Time(hh:mm:ss)'])


def daytime_proxy(month, height_scale=1200.):
    aod_path = ROOT/f'scratch/validation-candidates/madrid-2017-{month}-AOD20.txt'
    inv_path = ROOT/f'scratch/validation-candidates/madrid-2017-{month}-ALM20.txt'
    aods, inversions = aeronet_rows(aod_path, f'19:{month}:2017'), aeronet_rows(inv_path, f'19:{month}:2017')
    if not aods or not inversions:
        raise ValueError('The prespecified daytime proxy is unavailable.')
    aod, inv = aods[-1], inversions[-1]
    tau500 = float(aod['AOD_500nm'])
    exponent = float(aod['500-870_Angstrom_Exponent'])
    exact_nm = 1000*float(aod['Exact_Wavelengths_of_AOD(um)_500nm'])
    height = float(aod['Site_Elevation(m)'])
    if tau500 <= 0 or not 0 < exact_nm < 1000 or not 0 <= exponent <= 4:
        raise ValueError('Missing or invalid AOD spectral inputs must not be filled by plausible numbers.')
    tau550 = tau500*(550/exact_nm)**(-exponent)
    tau0 = tau550/(np.exp(-height/height_scale)-np.exp(-80000/height_scale))
    g440, g675 = (float(inv[f'Asymmetry_Factor-Total[{wl}nm]']) for wl in [440, 675])
    if not (-1 < g440 < 1 and -1 < g675 < 1):
        raise ValueError('AERONET g is missing or outside the physical domain.')
    g550 = float(np.interp(np.log(550), np.log([440, 675]), [g440, g675]))
    local = []
    for row in aods:
        tau = float(row['AOD_500nm'])
        alpha = float(row['500-870_Angstrom_Exponent'])
        wave = float(row['Exact_Wavelengths_of_AOD(um)_500nm'])*1000
        if tau > 0 and 0 <= alpha <= 4 and wave > 0:
            local.append(tau*(550/wave)**(-alpha))
    observation = datetime.fromisoformat(f'2017-{month}-19T22:30:00+00:00')
    def instant(row):
        return datetime.strptime(row['Date(dd:mm:yyyy)']+' '+row['Time(hh:mm:ss)'], '%d:%m:%Y %H:%M:%S').replace(tzinfo=timezone.utc)
    return {'selection': 'Last valid daytime record on the observation day, selected without sky residuals.',
        'aod_input': {'path': str(aod_path.relative_to(ROOT)), 'sha256': file_hash(aod_path), 'day_records': len(aods)},
        'inversion_input': {'path': str(inv_path.relative_to(ROOT)), 'sha256': file_hash(inv_path), 'day_records': len(inversions)},
        'aod_utc': instant(aod).isoformat(), 'g_utc': instant(inv).isoformat(),
        'aod_gap_hours_to_22_30_utc': (observation-instant(aod)).total_seconds()/3600,
        'g_gap_hours_to_22_30_utc': (observation-instant(inv)).total_seconds()/3600,
        'site_height_m': height, 'site_latitude_deg': float(aod['Site_Latitude(Degrees)']),
        'site_longitude_deg': float(aod['Site_Longitude(Degrees)']),
        'aod_500_measured': tau500, 'exact_wavelength_nm': exact_nm, 'angstrom_500_870': exponent,
        'aod_550_above_instrument': tau550, 'aod_550_sea_level_scenario': float(tau0),
        'aerosol_scale_height_m_scenario': height_scale, 'g_550_column_proxy': g550,
        'day_aod_550_range_above_instrument': [min(local), max(local)],
        'ssa_available_last_record': float(inv['Single_Scattering_Albedo[440nm]']) > 0,
        'simultaneous_night_measurement': False,
        'vertical_profile_or_local_phase_function_measured': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113/city')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observer = Observer(40.451111, -3.726111, 650.)
    parent = ROOT/'artifacts/seasonal-ground-20261110/astmon-publication-qa.json'
    raw = ROOT/'scratch/data-search-20261110/AstMon_evol.csv'
    imported = read_json(parent)
    if file_hash(raw) != imported['selection']['input_sha256']:
        raise ValueError('The original ground input no longer matches the published QA selection.')
    # Re-read measured values by exact time, rather than scoring a figure or an
    # old model's output as observation. Keep publication cloud QA as a separate source.
    wanted = {r['utc']: r for r in imported['records']}
    selected = []
    with raw.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream, delimiter=';'):
            time = row['Datetime_UTC'].replace(' ', 'T')+'Z'
            if row['Filter'] == 'Johnson-V' and time in wanted:
                record = wanted[time]
                if float(row['Magnitude']) != record['mag_arcsec2'] or float(row['Error']) != record['reported_error_mag']:
                    raise ValueError('Original photometry differs from the selected measurement.')
                selected.append(record)
    if len(selected) != 18:
        raise ValueError('The frozen 9+9 measured-row split must be preserved.')
    nights = nightly_values(selected)
    proxies = {f'2017-{m}': daytime_proxy(m) for m in ['03', '11']}
    protocol = {'id': 'LP-Madrid-atmosphere-proxy-v1',
        'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'observer': asdict(observer), 'training_night': '2017-03-19', 'check_night': '2017-11-19',
        'independent_blind_test': False, 'reason': 'These two windows have already been inspected in previous development.',
        'QA': {'source': 'https://www.mdpi.com/2072-4292/13/8/1511', 'figures': [5, 6],
            'intervals_utc': imported['selection']['publication_windows'], 'raw_cloud_flags_available': False,
            'native_band': 'AstMon Johnson-V', 'sun_below_deg': -18, 'moon_below_deg': -5},
        'source_composite': 'VNP46A3.002 NearNadir Snow_Free, common QA/land support, reference March.',
        'source_exclude_radius_m': 3000, 'source_block_pixels': 12, 'wavelength_nm': 550,
        'band_bridge': 'Effective monochromatic 550nm anchored to V, not integrated local lamp spectra.',
        'natural_V_mag_arcsec2': 21.6, 'all_other_processes_fixed': True,
        'primary_scale_rule': 'Fit one C only to March DEFAULT atmosphere; retain C for both nights and every atmosphere change.',
        'secondary_scale_rule': 'Also report separate common C per variant fitted to March only; never fit November.',
        'cases': ['default', 'daytime_AOD_only', 'daytime_g_only', 'daytime_AOD_and_g'],
        'proxies': proxies, 'nightly_truth': nights,
        'atmosphere': asdict(Atmosphere()), 'numerics': [asdict(SolverSettings(192, 24)), asdict(SolverSettings(384, 48))],
        'source_code_sha256': {p: file_hash(ROOT/p) for p in ['scripts/run_city_benchmark.py',
            'src/lightpollution/physics.py', 'src/lightpollution/sources.py', 'src/lightpollution/calibration.py']},
        'input_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in [parent, raw]},
        'limitations': ['Only one training and one previously inspected check night.',
            'AERONET is daytime, column g is a proxy for local HG and SSA is missing.',
            'Source heights remain spherical-ground; monthly emission is not the instantaneous city.',
            'This does not certify the camera, naked eye, spectrum, terrain or physical absolute scale.']}
    write_json(args.output/'protocol.json', protocol)
    responses, provenance = {}, {}
    for period, proxy in proxies.items():
        cache = ROOT/f'data/seasonal-ground-20261110/madrid-matched-2017/{period}/source-grid.npz'
        arrays, bounds, meta = load_region(cache)
        sources, description = regional_sources(arrays, bounds, observer, block=12,
            normalisation_radiance_area=meta['reference_radiance_area_sum'])
        provenance[period] = {'cache_sha256': file_hash(cache), 'cache_path': str(cache.relative_to(ROOT)),
            'metadata': meta, 'source_selection': description}
        variants = {'default': Atmosphere(),
            'daytime_AOD_only': replace(Atmosphere(), aerosol_optical_depth=proxy['aod_550_sea_level_scenario']),
            'daytime_g_only': replace(Atmosphere(), aerosol_g=proxy['g_550_column_proxy']),
            'daytime_AOD_and_g': replace(Atmosphere(), aerosol_optical_depth=proxy['aod_550_sea_level_scenario'], aerosol_g=proxy['g_550_column_proxy'])}
        responses[period] = {}
        for name, atm in variants.items():
            values = [integrate_direction(observer, sources, 0, 90, atm, settings)
                for settings in [SolverSettings(192, 24), SolverSettings(384, 48)]]
            responses[period][name] = {'raw_radiance_response': values[-1],
                'atmosphere': asdict(atm), 'coarse_response': values[0], 'refinement_relative': abs(values[0]/values[1]-1)}
    truth = {r['period']: r for r in nights}
    natural = 10**(-.4*(21.6-22))
    train, test = '2017-03', '2017-11'
    fixed_c = (truth[train]['brightness_relative']-natural)/responses[train]['default']['raw_radiance_response']
    results = []
    for name in protocol['cases']:
        branch_c = (truth[train]['brightness_relative']-natural)/responses[train][name]['raw_radiance_response']
        row = {'variant': name, 'coefficient_fixed_from_default_training': fixed_c, 'coefficient_branch_training_only': branch_c, 'periods': {}}
        for period in proxies:
            raw_value = responses[period][name]['raw_radiance_response']
            predicted = fixed_c*raw_value+natural
            predicted_branch = branch_c*raw_value+natural
            row['periods'][period] = {'observed_V_mag_arcsec2': truth[period]['mag_arcsec2'],
                'samples': truth[period]['samples'], 'independent_nights': 1,
                'fixed_C_model_over_observed_flux': predicted/truth[period]['brightness_relative'],
                'fixed_C_residual_mag': float(22-2.5*np.log10(predicted)-truth[period]['mag_arcsec2']),
                'training_only_branch_C_model_over_observed_flux': predicted_branch/truth[period]['brightness_relative'],
                'training_only_branch_C_residual_mag': float(22-2.5*np.log10(predicted_branch)-truth[period]['mag_arcsec2']),
                'raw_response_ratio_to_same_period_default': raw_value/responses[period]['default']['raw_radiance_response']}
        results.append(row)
    write_json(args.output/'source-provenance.json', provenance)
    write_json(args.output/'responses.json', responses)
    write_json(args.output/'results.json', {'protocol_sha256': file_hash(args.output/'protocol.json'), 'variants': results})
    summary = {'variants': len(results), 'measured_records': len(selected), 'training_nights': 1, 'check_nights': 1,
        'max_refinement_relative': max(r['refinement_relative'] for p in responses.values() for r in p.values()),
        'check_residual_mag_fixed_C': {r['variant']: r['periods'][test]['fixed_C_residual_mag'] for r in results},
        'check_residual_mag_separate_training_only_C': {r['variant']: r['periods'][test]['training_only_branch_C_residual_mag'] for r in results},
        'minimum_aod_time_gap_hours': min(p['aod_gap_hours_to_22_30_utc'] for p in proxies.values()),
        'maximum_aod_time_gap_hours': max(p['aod_gap_hours_to_22_30_utc'] for p in proxies.values()),
        'nighttime_atmosphere_measured': False, 'absolute_field_accuracy_certified': False}
    write_json(args.output/'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
