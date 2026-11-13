"""Isolate spectral approximations using measured SEDs; no eye/display claims."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.adapters.storage import file_hash, write_json
from lightpollution.models.spectra import band_integral, continuum_from_bv, integrate_spectral_direction
from lightpollution.physics import EARTH_RADIUS_M, Atmosphere, Observer, SolverSettings, Sources, unit_position

ROOT = Path(__file__).resolve().parents[1]


def ratios(axis, value, reference, responses):
    return {name: float(band_integral(axis, value, curve)/band_integral(axis, reference, curve)-1)
        for name, curve in responses.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmark-baseline-20261113/spectra')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library = load_spectral_library()
    lamp_path = ROOT/'scratch/benchmark-research/responses/lspdd-capture-one.body'
    lamp = json.loads(lamp_path.read_text())
    capture = lamp['spectral_captures'][0]
    protocol = {'id': 'LP-measured-SED-component-v1',
        'created_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'library_id': library.identity, 'lamp_source': 'https://api.lspdd.sbcloud.ca/capture/2469',
        'lamp_sha256': file_hash(lamp_path), 'lamp_rights': 'CC BY-NC-ND 4.0; original array not redistributed, derived diagnostics only.',
        'lamp_cct_K': float(capture['cct']), 'library_lamp_candidate': 'LED-B1',
        'lamp_candidate_policy': 'Warm white nominal ~2700K, selected by lamp class before computing spectral differences.',
        'lamp_is_a_regional_street_inventory': False,
        'lamp_normalisation': 'Equal emitted Johnson-V in the COMMON measured/library wavelength interval; NOT DNB.',
        'stellar_test': 'Infer continuum from measured B-V, normalise to measured V, test other spectral responses.',
        'stellar_truth': library.provenance['stars'],
        'star_test_is_conditional_on_perfect_BV_and_V_inputs': True,
        'independent_band_truth': ['CIE photopic', 'CIE scotopic'],
        'not_independent_scores': ['Johnson B/V used to construct continuum', 'Vega used to define library zero points'],
        'unchanged_source_geometry': {'surface_arc_distance_m': 40000., 'observer_height_m': 20,
            'azimuth_deg': 180, 'altitudes_deg': [10, 45, 90], 'atmosphere': 'default', 'order': 1},
        'source_code_sha256': {p: file_hash(ROOT/p) for p in ['scripts/run_spectral_benchmark.py',
            'src/lightpollution/models/spectra.py', 'src/lightpollution/adapters/spectral_library.py']},
        'eye_colour_or_camera_accuracy_certified': False}
    write_json(args.output/'protocol.json', protocol)
    stars = []
    for record in library.provenance['stars']:
        measured = library.measured[record['hr']]
        bv = float(-2.5*np.log10((band_integral(library.wavelengths, measured, library.responses['johnson_b'])/library.zero_b)
            /(band_integral(library.wavelengths, measured, library.responses['johnson_v'])/library.zero_v)))
        approximation = continuum_from_bv(library.wavelengths, bv, library.responses['johnson_b'],
            library.responses['johnson_v'], library.zero_b, library.zero_v)
        if approximation is None:
            stars.append({'hr': record['hr'], 'status': 'outside_continuum_colour_domain', 'derived_BV': bv})
            continue
        shape, temperature = approximation
        shape *= band_integral(library.wavelengths, measured, library.responses['johnson_v'])/band_integral(library.wavelengths, shape, library.responses['johnson_v'])
        stars.append({'hr': record['hr'], 'name': record['name'], 'status': 'scored_conditional_component',
            'source': record['source'], 'derived_BV': bv, 'fitted_continuum_temperature_K': temperature,
            'continuum_over_measured_response_minus_one': ratios(library.wavelengths, shape, measured, library.responses),
            'B_and_V_are_constraints_not_independent_validation': True})
    write_json(args.output/'stellar-continuum.json', {'stars': stars})
    spectrum = np.asarray(capture['spectral_datas'], dtype=float)
    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or np.any(~np.isfinite(spectrum)) or np.any(np.diff(spectrum[:, 0]) <= 0) or np.any(spectrum[:, 1] < 0):
        raise ValueError('Measured lamp must have a finite nonnegative increasing spectral axis.')
    common = (spectrum[:, 0] >= library.wavelengths[0]) & (spectrum[:, 0] <= library.wavelengths[-1])
    axis, measured = spectrum[common, 0], spectrum[common, 1]
    responses = {name: np.interp(axis, library.wavelengths, value) for name, value in library.responses.items()}
    candidate = np.interp(axis, library.wavelengths, library.templates['LED-B1'])
    measured /= band_integral(axis, measured, responses['johnson_v'])
    candidate /= band_integral(axis, candidate, responses['johnson_v'])
    # Transfer is evaluated on the measured axis, in source-independent chunks
    # inside the existing spectral integrator (one source here).
    observer = Observer(0., 0., 20.)
    source = Sources(EARTH_RADIUS_M*unit_position(np.array([-np.rad2deg(40000/EARTH_RADIUS_M)]), np.array([0.])), np.array([1.]))
    directions = []
    for alt in [10., 45., 90.]:
        coarse = integrate_spectral_direction(observer, source, 180., alt, Atmosphere(), SolverSettings(192, 24), axis, 1.3)
        transfer = integrate_spectral_direction(observer, source, 180., alt, Atmosphere(), SolverSettings(384, 48), axis, 1.3)
        value = ratios(axis, candidate*transfer, measured*transfer, responses)
        coarse_value = ratios(axis, candidate*coarse, measured*coarse, responses)
        directions.append({'altitude_deg': alt, 'candidate_over_measured_response_minus_one': value,
            'max_absolute_change_of_ratio_on_refinement': max(abs(value[k]-coarse_value[k]) for k in value)})
    lamp_result = {'common_wavelength_min_max_nm': [float(axis[0]), float(axis[-1])], 'samples': len(axis),
        'candidate': 'LED-B1', 'measured_cct_K': float(capture['cct']),
        'source_candidate_over_measured_response_minus_one': ratios(axis, candidate, measured, responses),
        'sky_responses': directions, 'DNB_response_or_full_lamp_power_tested': False,
        'not_a_field_accuracy_score': True}
    write_json(args.output/'lamp-sensitivity.json', lamp_result)
    valid = [r for r in stars if r['status'] == 'scored_conditional_component']
    summary = {'measured_stellar_SEDs': len(stars), 'scored_continua': len(valid),
        'max_absolute_stellar_photopic_relative_difference': max(abs(r['continuum_over_measured_response_minus_one']['photopic']) for r in valid),
        'max_absolute_stellar_scotopic_relative_difference': max(abs(r['continuum_over_measured_response_minus_one']['scotopic']) for r in valid),
        'lamp_source_scotopic_relative_difference': lamp_result['source_candidate_over_measured_response_minus_one']['scotopic'],
        'lamp_sky_johnson_V_relative_difference_by_altitude': {str(int(r['altitude_deg'])): r['candidate_over_measured_response_minus_one']['johnson_v'] for r in directions},
        'full_sky_or_eye_or_sensor_validation': False}
    write_json(args.output/'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
