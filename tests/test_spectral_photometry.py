from copy import deepcopy
from dataclasses import asdict

from fastapi.testclient import TestClient
import numpy as np
import pytest

from lightpollution.adapters.observation_io import write_observation
from lightpollution.adapters.spectral_library import load_spectral_library
from lightpollution.application.photometry import fit_dataset
from lightpollution.application.planner import Planner, compare_results, export_result
from lightpollution.domain.scene import DomainError, SceneSpec
from lightpollution.domain.calibration import calibration_key
from lightpollution.interfaces.commands import offline_example
from lightpollution.interfaces.web import create_app
from lightpollution.models.shared_photometry import fit_shared_photometry
from lightpollution.models.spectra import (band_integral, continuum_from_bv, integrate_spectral_direction,
    response_summary, stellar_spectral_observation, wavelength_atmosphere)
from lightpollution.physics import Atmosphere, Observer, SolverSettings, integrate_direction
from lightpollution.sources import synthetic_cities


def scene(spectral=False, dated=False):
    value = offline_example()
    value['directions'] = [[0, 90], [180, 45]]
    if not dated:
        value.pop('time')
        value.update(mode='undated', moon={'mode': 'none'})
    if spectral:
        value['spectrum'] = {}
    return value


class PositionalSolver:
    version = 'analytical-test-response'

    def direction(self, observer, sources, az, alt, atmosphere, settings):
        return float(np.sum(sources.power)*(observer.latitude_deg-59)*(1+atmosphere.aerosol_optical_depth))


def samples():
    rows = []
    for night, role, site, x in [('2026-01-01', 'training', 'north', 1.),
                                 ('2026-01-02', 'training', 'north', 3.),
                                 ('2026-01-03', 'validation', 'south', 2.)]:
        rows.append({'night_id': night, 'role': role, 'site_id': site, 'case_id': site,
            'direction': [0, 90], 'raw_response': x,
            'observed_v_mag_arcsec2': 22-2.5*np.log10(2.5*x+.7), 'background_relative': .7,
            'cloud_free_confirmed': True, 'qa_source': 'Controlled synthetic fixture.'})
    return rows


def test_single_coefficient_and_night_weights_with_spatial_holdout():
    rows = samples()
    actual = fit_shared_photometry(rows)
    duplicate = fit_shared_photometry(rows + [deepcopy(rows[0]) for _ in range(20)])
    assert actual['source_coefficient'] == pytest.approx(2.5)
    assert duplicate['source_coefficient'] == pytest.approx(2.5)
    assert actual['validation']['rmse_mag'] < 1e-12
    assert actual['independent_validation_sites'] == ['south']
    assert not actual['field_accuracy_certified'] and actual['uncertainty_interval'] is None


@pytest.mark.parametrize('change', ['same_night', 'unknown_clouds', 'negative_response', 'no_validation'])
def test_ground_fit_rejects_invalid_identification_and_qa(change):
    rows = samples()
    if change == 'same_night':
        rows[-1]['night_id'] = rows[0]['night_id']
    elif change == 'unknown_clouds':
        rows[0]['cloud_free_confirmed'] = None
    elif change == 'negative_response':
        rows[0]['raw_response'] = -1
    else:
        rows = rows[:-1]
    with pytest.raises(ValueError):
        fit_shared_photometry(rows)


def test_common_anchor_and_portable_ground_coefficient(tmp_path):
    planner = Planner(tmp_path, solver=PositionalSolver())
    data = scene()
    first = planner.run(SceneSpec.from_mapping(data)).value
    data['observer']['latitude_deg'] = 60.5
    second = planner.run(SceneSpec.from_mapping(data)).value
    assert first['calibration']['scope'] == second['calibration']['scope']
    assert first['calibration']['source_coefficient'] == second['calibration']['source_coefficient']
    ratio = second['sky']['directions'][0]['artificial_relative']/first['sky']['directions'][0]['artificial_relative']
    assert ratio == pytest.approx(1.5/1.3)
    assert compare_results(first, second)['directions'][0]['right_over_left'] > 1
    target = first['calibration']['source_coefficient']*1.7
    records = []
    for case, role, utc, value in [('a', 'training', '2026-11-09T21:00:00Z', first),
                                   ('b', 'validation', '2026-11-10T21:00:00Z', second)]:
        row = value['sky']['directions'][0]
        records.append({'id': case, 'site_id': case, 'case_id': case, 'role': role, 'utc': utc,
            'observed_v_mag_arcsec2': float(22-2.5*np.log10(target*row['raw_transfer_response']+row['natural_relative'])),
            'cloud_free_confirmed': True, 'qa_source': 'Synthetic fixture; not a field measurement.'})
    dataset = {'schema_version': 1, 'band': 'Johnson-V', 'source': 'synthetic', 'samples': records}
    calibration = fit_dataset(dataset, {'a': first, 'b': second})
    assert calibration['source_coefficient'] == pytest.approx(target)
    assert calibration['validation']['rmse_mag'] < 1e-12
    changed_runtime = deepcopy(first)
    changed_runtime['runtime']['elapsed_seconds'] = 999.
    changed_runtime['cache']['scene_hit'] = True
    assert fit_dataset(dataset, {'a': changed_runtime, 'b': second})['id'] == calibration['id']
    data['photometric_calibration'] = calibration
    def javascript_numbers(value):
        if isinstance(value, dict):
            return {k: javascript_numbers(v) for k, v in value.items()}
        if isinstance(value, list):
            return [javascript_numbers(v) for v in value]
        return int(value) if type(value) is float and value.is_integer() else value
    data['photometric_calibration'] = javascript_numbers(calibration)
    calibrated = planner.run(SceneSpec.from_mapping(data)).value
    assert calibrated['calibration']['origin'] == 'provided_calibration'
    assert calibrated['calibration']['source_coefficient'] == pytest.approx(target)
    assert 'no_independent_field_validation' not in calibrated['observation']['assumptions_and_limitations']
    assert 'ground_validation_limited_to_supplied_nights_and_sites_not_global_accuracy' in calibrated['observation']['assumptions_and_limitations']
    whole_coefficient = deepcopy(calibration)
    whole_coefficient['source_coefficient'] = 4.
    whole_coefficient['id'] = calibration_key({k: v for k, v in whole_coefficient.items() if k != 'id'})
    data['photometric_calibration'] = whole_coefficient
    typed_float = planner.run(SceneSpec.from_mapping(data)).value
    data['photometric_calibration'] = javascript_numbers(whole_coefficient)
    typed_integer = planner.run(SceneSpec.from_mapping(data)).value
    assert typed_float['comparison_basis_id'] == typed_integer['comparison_basis_id']
    data['photometric_calibration'] = javascript_numbers(calibration)
    data['photometric_calibration']['source_coefficient'] *= 2
    with pytest.raises(DomainError, match='повреждена'):
        planner.run(SceneSpec.from_mapping(data))


def test_energy_and_photon_response_have_distinct_units():
    axis = np.array([400., 500., 600.])
    spectrum = np.full(3, 2.)
    response = np.ones(3)
    assert band_integral(axis, spectrum, response) == pytest.approx(400.)
    expected = (600**2-400**2)*1e-9/(6.62607015e-34*299792458)
    assert band_integral(axis, spectrum, response, 'photon') == pytest.approx(expected)
    with pytest.raises(ValueError):
        band_integral(axis, spectrum, response, 'guess')


@pytest.mark.parametrize('wavelength', [400., 550., 750.])
def test_batched_spectrum_matches_independent_scalar_solver(wavelength):
    observer = Observer()
    sources = synthetic_cities(observer)
    atmosphere, settings = Atmosphere(), SolverSettings(32, 8)
    axis = np.array([wavelength, wavelength+1])
    spectral = integrate_spectral_direction(observer, sources, 180, 45, atmosphere, settings, axis, 1.3)
    scalar = integrate_direction(observer, sources, 180, 45,
                                  wavelength_atmosphere(atmosphere, wavelength, 1.3), settings)
    assert spectral[0] == pytest.approx(scalar, rel=3e-13, abs=1e-25)


def test_verified_lamp_library_power_and_distinct_eye_responses():
    library = load_spectral_library()
    assert len(library.templates) == 14 and len(library.measured) >= 3
    assert library.provenance['gaia_targets'] == 9
    for name in library.templates:
        mix = library.mix({name: 1.})
        assert np.trapezoid(mix, library.wavelengths) == pytest.approx(1.)
    blue = response_summary(library.wavelengths, library.mix({'LED-B5': 1.}), library.responses, library.zero_v)
    sodium = response_summary(library.wavelengths, library.mix({'HP1': 1.}), library.responses, library.zero_v)
    assert blue['scotopic_photopic_ratio'] > sodium['scotopic_photopic_ratio']
    assert not np.array_equal(library.responses['johnson_v'], library.responses['photopic'])
    with pytest.raises(DomainError):
        library.mix({'invented-lamp': 1.})


def test_spectral_solver_boundary_is_used_and_validated(tmp_path):
    class SpectralSolver(PositionalSolver):
        version = 'controlled-spectral-contract'
        calls = 0

        def spectrum(self, observer, sources, az, alt, atmosphere, settings, wavelengths, exponent):
            self.calls += 1
            return np.ones(len(wavelengths))*(observer.latitude_deg-59)

    data = scene(spectral=True)
    with pytest.raises(DomainError, match='контракт'):
        Planner(tmp_path/'unsupported', solver=PositionalSolver()).run(SceneSpec.from_mapping(data))
    solver = SpectralSolver()
    first = Planner(tmp_path/'good', solver=solver).run(SceneSpec.from_mapping(data)).value
    assert solver.calls > 0 and first['spectral']['city_only']
    calls = solver.calls
    Planner(tmp_path/'good', solver=solver).run(SceneSpec.from_mapping(data))
    assert solver.calls == calls


def test_stellar_colour_continuum_and_extinction_once(tmp_path):
    library = load_spectral_library()
    value, temperature = continuum_from_bv(library.wavelengths, .65, library.responses['johnson_b'],
                                           library.responses['johnson_v'], library.zero_b, library.zero_v)
    colour = -2.5*np.log10((band_integral(library.wavelengths, value, library.responses['johnson_b'])/library.zero_b)
                           / (band_integral(library.wavelengths, value, library.responses['johnson_v'])/library.zero_v))
    assert colour == pytest.approx(.65, abs=1e-9) and 1500 < temperature < 50000
    rows = [{'hr': hr, 'catalogue_v_mag': mag, 'altitude_deg': 45, 'observed_v_mag': 999}
            for hr, mag in [(99999, 3.), (7001, .03), (123456, 4.)]]
    colours = {99999: {'b_v': '.65'}}
    args = (rows, Observer(), Atmosphere(rayleigh_multiplier=0, aerosol_optical_depth=0),
            library.wavelengths, library.responses, colours, library.measured,
            library.zero_b, library.zero_v, 1.3)
    output, metadata, examples = stellar_spectral_observation(*args)
    assert metadata['available'] == 2 and metadata['unavailable'] == 1
    assert output[0]['observed_v_mag'] == pytest.approx(3.)
    assert output[1]['observed_v_mag'] == pytest.approx(.03)
    assert not output[2]['spectral_available']
    extincted, _, _ = stellar_spectral_observation(rows, Observer(), Atmosphere(), *args[3:])
    assert extincted[0]['observed_v_mag'] > 3
    assert extincted[0]['photopic_illuminance_lux'] < output[0]['photopic_illuminance_lux']
    # CSV columns remain valid when rows have different spectral availability.
    write_observation({'stars': output, 'directions': []}, tmp_path)
    assert 'spectral_available' in (tmp_path/'stars.csv').read_text()


def test_spectral_planner_common_power_mixture_cache_and_export(tmp_path):
    planner = Planner(tmp_path/'work')
    data = scene(spectral=True)
    first = planner.run(SceneSpec.from_mapping(data))
    repeat = planner.run(SceneSpec.from_mapping(data)).value
    assert repeat['cache']['scene_hit']
    changed = deepcopy(data)
    changed['spectrum'] = {'mix': {'HP1': 1.}}
    second = planner.run(SceneSpec.from_mapping(changed)).value
    assert second['cache']['transfer_misses'] == 0  # Same transfer, different source spectrum.
    assert first.value['calibration']['source_coefficient'] == second['calibration']['source_coefficient']
    assert first.value['spectral']['source_visible_power_scale_W'] == second['spectral']['source_visible_power_scale_W']
    assert compare_results(first.value, second)['directions']
    spectral = first.value['spectral']
    library = load_spectral_library()
    for row, direction in zip(spectral['directions'], first.value['sky']['directions'], strict=True):
        integrated = band_integral(library.wavelengths, row['city_radiance_W_m2_sr_nm'], library.responses['johnson_v'])
        assert integrated/spectral['reference_v_radiance_W_m2_sr'] == pytest.approx(direction['artificial_relative'])
    export_result(first, tmp_path/'export')
    assert not spectral['total_sky_spectrum_available']
    assert 'eye_detection' in first.value['unavailable_capabilities']


def test_dated_spectral_catalogue_and_heterogeneous_export(tmp_path):
    result = Planner(tmp_path/'work').run(SceneSpec.from_mapping(scene(spectral=True, dated=True)))
    spectral = result.value['spectral']
    metadata = spectral['stellar_summary']
    assert metadata['available'] + metadata['unavailable'] == len(result.value['stars']['rows'])
    assert metadata['catalogue_continuum'] > 0 and spectral['stellar_examples']
    export_result(result, tmp_path/'export')
    text = (tmp_path/'export/stars.csv').read_text()
    assert 'johnson_v_flux_W_m2' in text and 'spectrum_recipe' in text


@pytest.mark.parametrize('spectrum', [{'mix': {'LED-B3': -1}}, {'mix': {'LED-B3': .4}},
                                      {'angstrom_exponent': 5}, {'stars': 'invent_white'}])
def test_spectrum_request_validation(spectrum):
    data = scene()
    data['spectrum'] = spectrum
    with pytest.raises(DomainError):
        SceneSpec.from_mapping(data)


def test_spectral_http_catalogue_and_job(tmp_path):
    import time
    with TestClient(create_app(tmp_path)) as client:
        assert 'HP1' in client.get('/api/spectral-library').json()['templates']
        data = scene(spectral=True)
        job = client.post('/api/jobs', json=data).json()['id']
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            value = client.get('/api/jobs/'+job).json()
            if value['state'] in ['completed', 'error']:
                break
            time.sleep(.05)
        assert value['state'] == 'completed', value
        result = client.get(f'/api/jobs/{job}/result').json()
        assert result['spectral']['city_only']
        assert client.get(f'/api/jobs/{job}/export/stars.csv').status_code == 200
