from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time
import zipfile
from io import BytesIO

from fastapi.testclient import TestClient
import numpy as np
import pytest

from lightpollution.adapters.sources import LocalSourceRepository
from lightpollution.adapters.storage import read_json, write_json
from lightpollution.application.jobs import JobManager
from lightpollution.application.planner import Planner, compare_results, export_result
from lightpollution.domain.scene import Cancelled, DomainError, SceneSpec
from lightpollution.interfaces.commands import offline_example
from lightpollution.interfaces.web import create_app
from lightpollution.scene import calculate_observation


@pytest.fixture
def undated():
    value = offline_example()
    value.pop('time')
    value.update(mode='undated', moon={'mode': 'none'})
    return value


class CountingSolver:
    version = 'controlled-linear-test-1'

    def __init__(self):
        self.calls = 0

    def direction(self, observer, sources, az, alt, atmosphere, settings):
        self.calls += 1
        return float(np.sum(sources.power) * (1 + atmosphere.aerosol_optical_depth)
                     * (1 + (90 - alt) / 90) * (1 + az / 360))


@pytest.mark.parametrize('index', [0, 1, 2])
def test_frozen_pre_refactor_schema1_outputs(index, tmp_path):
    fixture = json.loads((Path(__file__).parent / 'fixtures/legacy-v1.json').read_text())
    grid = fixture['grid']
    np.savez(tmp_path / 'field.npz', **{k: v for k, v in grid.items() if k != 'metadata'},
             metadata_json=json.dumps(grid['metadata']))
    case = fixture['cases'][index]
    value = calculate_observation(case['scene'], tmp_path)
    for actual, expected in zip(value['directions'], case['expected_directions'], strict=True):
        for key, reference in expected.items():
            if type(reference) is float:
                assert actual[key] == pytest.approx(reference, rel=1e-10, abs=1e-12)
            else:
                assert actual[key] == reference
    assert len(value['stars']) == case['expected_star_count']
    for actual, expected in zip(value['stars'][:3], case['expected_first_stars'], strict=True):
        for key, reference in expected.items():
            if type(reference) is float:
                assert actual[key] == pytest.approx(reference, rel=1e-10, abs=1e-12)
            else:
                assert actual[key] == reference


@pytest.mark.parametrize('change,code', [
    ({'schema_version': 99}, 'unsupported_schema'),
    ({'time': '2026-11-09T21:00:00'}, 'invalid_time'),
    ({'directions': [[0, 9]]}, 'unsupported_altitude'),
    ({'directions': [[0, 90], [180, 90]]}, 'invalid_direction'),
    ({'observer': {'latitude_deg': 60.3, 'longitude_deg': 30.315, 'height_m': 650}}, 'unsupported_height'),
    ({'atmosphere': {'aerosol_optical_depth': -1}}, 'invalid_atmosphere'),
    ({'atmosphere': {'aerosol_g': float('nan')}}, 'invalid_atmosphere'),
    ({'atmosphere': {'reference_wavelength_nm': 600}}, 'unsupported_atmosphere'),
    ({'unknown': 'ignored'}, 'invalid_input'),
])
def test_schema_rejects_ambiguous_requests(change, code):
    value = offline_example()
    value.update(change)
    with pytest.raises(DomainError) as error:
        SceneSpec.from_mapping(value)
    assert error.value.code == code


def test_fixed_source_scale_cache_invalidation_and_comparison(undated, tmp_path):
    solver = CountingSolver()
    planner = Planner(tmp_path, solver=solver)
    first = planner.run(SceneSpec.from_mapping(undated)).to_dict()
    calls = solver.calls
    repeat = planner.run(SceneSpec.from_mapping(undated)).to_dict()
    assert repeat['cache']['scene_hit'] and solver.calls == calls
    changed = deepcopy(undated)
    changed['atmosphere'] = {'aerosol_optical_depth': .2}
    second = planner.run(SceneSpec.from_mapping(changed)).to_dict()
    assert solver.calls > calls
    assert second['calibration']['source_coefficient'] == first['calibration']['source_coefficient']
    assert second['sky']['directions'][0]['artificial_relative'] / first['sky']['directions'][0]['artificial_relative'] == pytest.approx(1.2 / 1.1)
    assert second['sky']['directions'][0]['artificial_relative'] != pytest.approx(10 ** (-.4 * (20.5 - 22)))
    assert compare_results(first, second)['directions'][0]['right_over_left'] > 1
    changed['observer']['longitude_deg'] += .01
    third = planner.run(SceneSpec.from_mapping(changed)).to_dict()
    with pytest.raises(DomainError, match='одной точки'):
        compare_results(first, third)
    assert all(cap not in first['capabilities'] for cap in ['eye_detection', 'camera_electrons'])
    assert 'stellar_geometry' not in first['capabilities']  # No date means no stellar positions.


def test_time_change_reuses_physical_field_and_stellar_outputs_are_separate(tmp_path):
    solver = CountingSolver()
    planner = Planner(tmp_path, solver=solver)
    value = offline_example()
    a = planner.run(SceneSpec.from_mapping(value)).to_dict()
    calls = solver.calls
    value['time'] = '2026-11-10T21:00:00Z'
    b = planner.run(SceneSpec.from_mapping(value)).to_dict()
    assert solver.calls == calls and b['cache']['transfer_misses'] == 0
    assert a['sky']['directions'] != b['sky']['directions']  # Lunar geometry changes.
    assert a['stars']['rows'] != b['stars']['rows']
    assert a['stars']['flux_unit'] != a['sky']['photometry']['diffuse_units']


def test_cancelled_run_resumes_completed_rays_without_publishing_a_scene(undated, tmp_path):
    solver = CountingSolver()
    planner = Planner(tmp_path, solver=solver)
    spec = SceneSpec.from_mapping(undated)
    with pytest.raises(Cancelled):
        planner.run(spec, cancelled=lambda: solver.calls >= 2)
    assert list((tmp_path / 'cache/transfer').glob('*.json'))
    assert not list((tmp_path / 'cache/scenes').glob('*.json'))
    resumed = planner.run(spec).to_dict()
    assert resumed['cache']['transfer_hits'] >= 1
    assert resumed['sky']['directions'][0]['supported']


def test_corrupt_cached_response_is_recomputed(undated, tmp_path):
    solver = CountingSolver()
    planner = Planner(tmp_path, solver=solver)
    spec = SceneSpec.from_mapping(undated)
    first = planner.run(spec).to_dict()
    for path in (tmp_path / 'cache/scenes').glob('*.json'):
        path.unlink()
    path = next((tmp_path / 'cache/transfer').glob('*.json'))
    value = read_json(path)
    value['value']['radiance'] *= 100
    write_json(path, value)  # Checksum intentionally not updated.
    calls = solver.calls
    second = planner.run(spec).to_dict()
    assert solver.calls > calls and second['sky'] == first['sky']


def test_export_roundtrip_and_no_stale_stellar_file(undated, tmp_path):
    result = Planner(tmp_path / 'work', solver=CountingSolver()).run(SceneSpec.from_mapping(undated))
    output = tmp_path / 'export'
    output.mkdir()
    (output / 'stars.csv').write_text('stale star')
    export_result(result, output)
    assert SceneSpec.from_mapping(read_json(output / 'scene.json')).to_dict() == result.value['request']
    assert len((output / 'stars.csv').read_text().splitlines()) == 1
    assert b'\r' not in (output / 'directions.csv').read_bytes()
    assert read_json(output / 'result.json')['sky'] == result.value['sky']


def test_missing_nasa_inputs_are_not_silently_downloaded(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unit test must not contact NASA')
    monkeypatch.setattr('lightpollution.download.requests.get', forbidden)
    repository = LocalSourceRepository(tmp_path / 'work', tmp_path / 'raw')
    from lightpollution.physics import Observer
    with pytest.raises(DomainError) as error:
        repository.prepare('petersburg-2025', '2025', Observer())
    assert error.value.code == 'missing_data'


def test_prepared_months_share_support_and_preserve_source_power(tmp_path, monkeypatch):
    from lightpollution.data import Bounds, save_region
    from lightpollution.physics import Observer
    bounds = Bounds(29, 59, 29.1, 59.1)
    observer = Observer(59.05, 29.05, 20)
    resources = tmp_path / 'resources'
    resources.mkdir()
    entry = {'id': 'fixture', 'kind': 'blackmarble', 'periods': ['2017-03', '2017-11'],
             'reference_period': '2017-03', 'bounds': asdict(bounds)}
    write_json(resources / 'regions.json', [entry])
    paths = {}
    name = 'NearNadir_Composite_Snow_Free'
    for month, amplitude in [(3, 1.), (11, 2.)]:
        label = f'2017-{month:02}'
        arrays = {name: np.full(bounds.shape, amplitude),
                  name + '_Quality': np.zeros(bounds.shape),
                  name + '_Num': np.full(bounds.shape, 5.),
                  'Land_Water_Mask': np.ones(bounds.shape)}
        if month == 11:
            arrays[name + '_Quality'][0, 0] = 1
        paths[label] = tmp_path / (label + '.npz')
        save_region(paths[label], arrays, bounds, {'year': 2017, 'month': month, 'product': 'VNP46A3.002'})
    repository = LocalSourceRepository(tmp_path / 'work', resources=resources)
    monkeypatch.setattr(repository, '_raw_region', lambda e, label, download: paths[label])
    prepared = repository.prepare('fixture', '2017-03', observer)
    a, _ = prepared.select('2017-03')
    b, _ = prepared.select('2017-11')
    assert a.power.sum() == pytest.approx(1.) and b.power.sum() == pytest.approx(2.)
    np.testing.assert_allclose(a.positions_m, b.positions_m)
    changed = repository.prepare('fixture', '2017-11', Observer(59.06, 29.05, 20))
    assert prepared.identity != changed.identity
    prepared.paths['2017-03'].write_bytes(b'corrupt')
    with pytest.raises(DomainError) as error:
        repository.prepare('fixture', '2017-03', observer)
    assert error.value.code == 'corrupt_cache'


@pytest.mark.parametrize('invalid', [[], {'schema_version': 1, 'mode': 'undated', 'artificial': {'mode': 'grid', 'sky_grid': ''}}])
def test_invalid_legacy_documents_remain_validation_errors(invalid):
    with pytest.raises(ValueError):
        calculate_observation(invalid)


def test_job_recovery_and_single_worker_lock(tmp_path):
    folder = tmp_path / 'jobs' / ('a' * 32)
    write_json(folder / 'status.json', {'id': 'a' * 32, 'state': 'running', 'created_at': '2026-11-10T00:00:00Z'})
    manager = JobManager(tmp_path)
    try:
        assert manager.get('a' * 32)['error']['code'] == 'interrupted'
        with pytest.raises(DomainError, match='другим процессом'):
            JobManager(tmp_path)
        with pytest.raises(DomainError):
            manager.get('../secrets')
    finally:
        manager.close()


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get('/api/jobs/' + job_id)
        assert response.status_code == 200
        value = response.json()
        if value['state'] in ['completed', 'cancelled', 'error']:
            return value
        time.sleep(.05)
    pytest.fail('Worker did not finish within the offline-test deadline')


def test_http_worker_export_repeat_compare_and_early_rejection(tmp_path, undated):
    with TestClient(create_app(tmp_path)) as client:
        assert 'LightPollution' in client.get('/').text
        assert client.get('/static/app.js').status_code == 200
        assert client.get('/api/health').json()['status'] == 'ok'
        bad = offline_example()
        bad['time'] = '2026-06-21T12:00:00Z'
        assert client.post('/api/jobs', json=bad).json()['error']['code'] == 'twilight'
        assert client.get('/api/jobs').json() == []
        response = client.post('/api/jobs', json=undated)
        assert response.status_code == 202
        first = response.json()['id']
        assert client.get('/api/health').status_code == 200  # Worker does not block HTTP.
        assert wait_for_job(client, first)['state'] == 'completed'
        a = client.get(f'/api/jobs/{first}/result').json()
        archive = client.get(f'/api/jobs/{first}/export/bundle.zip')
        with zipfile.ZipFile(BytesIO(archive.content)) as zipped:
            assert set(zipped.namelist()) == {'scene.json', 'resolved-scene.json', 'observation.json',
                                              'result.json', 'directions.csv', 'stars.csv'}
            scene = json.loads(zipped.read('scene.json'))
        scene['atmosphere'] = {'aerosol_optical_depth': .2}
        second = client.post('/api/jobs', json=scene).json()['id']
        assert wait_for_job(client, second)['state'] == 'completed'
        b = client.get(f'/api/jobs/{second}/result').json()
        assert a['calibration']['source_coefficient'] == b['calibration']['source_coefficient']
        compared = client.post('/api/compare', json={'left': first, 'right': second})
        assert compared.status_code == 200 and compared.json()['directions']
        assert client.get(f'/api/jobs/{first}/export/worker-error.log').status_code == 404


def test_http_cancel_never_exposes_incomplete_export(tmp_path, undated):
    with TestClient(create_app(tmp_path)) as client:
        job_id = client.post('/api/jobs', json=undated).json()['id']
        client.post(f'/api/jobs/{job_id}/cancel')
        assert wait_for_job(client, job_id)['state'] == 'cancelled'
        assert client.get(f'/api/jobs/{job_id}/result').status_code == 409
        assert client.get(f'/api/jobs/{job_id}/export/result.json').status_code == 409
