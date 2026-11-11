"""One public application use case for CLI, jobs and HTTP."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..adapters.observation_io import write_observation
from ..adapters.sources import LocalSourceRepository, SourceRepository, region_anchor
from ..adapters.storage import content_key, file_hash, read_json, write_json
from ..atmosphere_config import atmosphere_from_mapping
from ..domain.fields import ObservationResources, RunResult, SkyField, SourceScale, StarFluxTable, TransferGrid
from ..domain.calibration import calibration_key
from ..domain.scene import Cancelled, DomainError, SceneSpec
from ..models.observation import calculate_scene
from ..models.transfer import SingleScatteringSolver, TransferSolver
from ..models.spectra import (ARCSEC2_PER_SR, band_integral,
                             response_summary, stellar_spectral_observation)
from ..observation import observation_context
from ..physics import Observer, SolverSettings
from ..skyview import load_stars

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 'shared-regional-photometry-2'
PROFILES = {'quick': (SolverSettings(96, 12), 24),
            'standard': (SolverSettings(192, 24), 12)}


def _code_identity(names):
    return content_key({name: file_hash(ROOT / name) for name in names})


def _read_checked(path):
    try:
        record = read_json(path)
        return record['value'] if record['sha256'] == content_key(record['value']) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_checked(path, value):
    write_json(path, {'sha256': content_key(value), 'value': value})


class Planner:
    def __init__(self, workspace: Path, sources: SourceRepository | None = None,
                 solver: TransferSolver | None = None, raw_directory: Path | None = None):
        self.workspace = Path(workspace)
        self.sources = sources or LocalSourceRepository(workspace, raw_directory)
        self.solver = solver or SingleScatteringSolver()

    def preflight(self, spec: SceneSpec):
        data = spec.to_dict()
        entries = {e['id']: e for e in self.sources.catalogue()}
        entry = entries.get(data['region'])
        if entry is None or data['period'] not in entry['periods']:
            raise DomainError('unknown_dataset', 'Выберите регион и период из каталога.')
        observer, bounds = spec.observer, entry['bounds']
        if not (bounds['south'] <= observer.latitude_deg <= bounds['north']
                and bounds['west'] <= observer.longitude_deg <= bounds['east']):
            raise DomainError('outside_region', 'Точка находится вне подготовленной области.')
        anchor = Observer(**data['anchor_observer']) if 'anchor_observer' in data else region_anchor(entry)
        if not (bounds['south'] <= anchor.latitude_deg <= bounds['north']
                and bounds['west'] <= anchor.longitude_deg <= bounds['east']):
            raise DomainError('outside_region', 'Общая опорная точка находится вне области источников.')
        if 'spectrum' in data:
            if not callable(getattr(self.solver, 'spectrum', None)):
                raise DomainError('unsupported_solver', 'Выбранный решатель не поддерживает спектральный контракт.')
            from ..adapters.spectral_library import load_spectral_library
            library = load_spectral_library()
            library.mix(data['spectrum']['mix'])
            library.mix(data['spectrum']['reference_mix'])
        if data['mode'] == 'dated':
            try:
                context = observation_context(observer, data['time'], *data['directions'][0])
            except (ValueError, OverflowError) as exc:
                raise DomainError('invalid_time', 'Не удалось вычислить эфемериды для указанного времени.') from exc
            if not context['astronomical_darkness']:
                raise DomainError('twilight', 'Солнце выше −18°: сумеречный фон в MVP не реализован.',
                                  {'sun_altitude_deg': context['sun']['altitude_deg']})
        return entry

    def run(self, spec: SceneSpec, progress=None, cancelled=None) -> RunResult:
        started = time.perf_counter()
        progress = progress or (lambda value: None)
        cancelled = cancelled or (lambda: False)
        def check():
            if cancelled():
                raise Cancelled()
        check()
        entry = self.preflight(spec)
        data, resolved = spec.to_dict(), spec.resolved()
        anchor_observer = Observer(**data['anchor_observer']) if 'anchor_observer' in data else region_anchor(entry)
        resolved['anchor_observer'] = asdict(anchor_observer)
        library = None
        spectral_config = data.get('spectrum')
        if spectral_config is not None:
            from ..adapters.spectral_library import load_spectral_library
            library = load_spectral_library()
            mix = library.mix(spectral_config['mix'])
            reference_mix = library.mix(spectral_config['reference_mix'])
        progress({'phase': 'Подготовка источников и QA', 'completed': 0, 'total': None})
        group = self.sources.prepare(data['region'], data['period'], spec.observer,
                                     allow_download=data['download_missing'])
        check()
        transfer_code = _code_identity(['physics.py', 'models/transfer.py', 'models/spectra.py', 'sources.py'])
        scene_code = _code_identity(['models/observation.py', 'photometry.py', 'moonlight.py',
                                     'skyview.py', 'observation.py', 'catalogues/bsc5.csv',
                                     'application/planner.py', 'domain/scene.py', 'domain/calibration.py',
                                     'adapters/spectral_library.py'])
        request_key = content_key({'scene': resolved, 'source_basis': group.identity,
            'transfer_code': transfer_code, 'scene_code': scene_code, 'solver': self.solver.version,
            'application_model': MODEL_VERSION, 'spectral_library': library.identity if library else None})
        scene_path = self.workspace / 'cache/scenes' / (request_key + '.json')
        cached = _read_checked(scene_path)
        if cached is not None:
            check()
            cached['cache'] = {'scene_hit': True, 'transfer_hits': 0, 'transfer_misses': 0}
            cached['runtime'] = {'elapsed_seconds': time.perf_counter() - started}
            progress({'phase': 'Готово, использован сохранённый результат', 'completed': 1, 'total': 1})
            return RunResult(cached)

        settings, block = PROFILES[data['profile']]
        observer, atmosphere = spec.observer, spec.atmosphere
        reference_atmosphere = atmosphere_from_mapping(data['reference_atmosphere'])
        azimuths = np.asarray(sorted({0., 90., 180., 270., *[v[0] for v in data['directions']]}))
        altitudes = np.asarray(sorted({10., 90., *[v[1] for v in data['directions']]}))
        total = 1 + 1 + len(azimuths) * (len(altitudes) - 1)
        stats = {'scene_hit': False, 'transfer_hits': 0, 'transfer_misses': 0}
        completed = 0
        groups = {tuple(asdict(observer).values()): group}
        transport = {'mode': 'spectral_single_scattering' if library else 'grey_550nm',
                     'wavelength_nm': library.wavelengths.tolist() if library else [550.],
                     'angstrom_exponent': spectral_config['angstrom_exponent'] if library else None}
        scope = {'source_basis_id': group.identity, 'transfer_code': transfer_code,
                 'solver_version': self.solver.version, 'transport': transport,
                 'band': 'Generic/Johnson.V energy response' if library else 'effective_V_550nm',
                 'spectral_library': library.identity if library else None,
                 'reference_v_mag_arcsec2': 22., 'height_datum': resolved['height_datum']}

        def response(period, az, alt, atm, numerical, aggregation, target=None):
            nonlocal completed
            check()
            target = target or observer
            target_key = tuple(asdict(target).values())
            if target_key not in groups:
                groups[target_key] = self.sources.prepare(data['region'], period, target,
                                                          allow_download=data['download_missing'])
            selected_group = groups[target_key]
            if selected_group.identity != group.identity:
                raise DomainError('incompatible_scale', 'Источники опорной и текущей точки имеют разные нормировки.')
            direction = (float(az), float(alt)) if alt <= 10 else None
            sources, provenance = selected_group.select(period, aggregation, direction)
            source_digest = hashlib.sha256(sources.positions_m.tobytes() + sources.power.tobytes()).hexdigest()
            identity = content_key({'source_basis': group.identity, 'source_arrays': source_digest,
                'period': period, 'observer': asdict(target), 'atmosphere': asdict(atm),
                'settings': asdict(numerical), 'block': aggregation, 'azimuth': float(az),
                'altitude': float(alt), 'solver': self.solver.version, 'code': transfer_code, 'transport': transport})
            path = self.workspace / 'cache/transfer' / (identity + '.json')
            stored = _read_checked(path)
            try:
                saved = np.asarray(stored.get('radiance'), dtype=float) if isinstance(stored, dict) else np.asarray(np.nan)
            except (TypeError, ValueError):
                saved = np.asarray(np.nan)
            expected_shape = library.wavelengths.shape if library else ()
            if (isinstance(stored, dict) and stored.get('identity') == identity and saved.shape == expected_shape
                    and np.all(np.isfinite(saved)) and np.all(saved >= 0)):
                value = saved if library else float(saved)
                stats['transfer_hits'] += 1
            else:
                if library:
                    value = np.asarray(self.solver.spectrum(target, sources, float(az), float(alt), atm, numerical,
                                        library.wavelengths, spectral_config['angstrom_exponent']), dtype=float)
                    if value.shape != library.wavelengths.shape:
                        raise DomainError('invalid_solver_output', 'Решатель вернул спектр другой размерности.')
                else:
                    value = float(self.solver.direction(target, sources, float(az), float(alt), atm, numerical))
                if np.any(~np.isfinite(value)) or np.any(np.asarray(value) < 0):
                    raise DomainError('invalid_solver_output', 'Решатель вернул недопустимое значение.')
                check()
                _write_checked(path, {'identity': identity, 'radiance': np.asarray(value).tolist()})
                stats['transfer_misses'] += 1
            completed += 1
            progress({'phase': 'Атмосферный перенос', 'completed': completed, 'total': total,
                      'cached_directions': stats['transfer_hits']})
            return value

        reference_values = response(group.reference_period, 0, 90, reference_atmosphere,
                                    *PROFILES['standard'], target=anchor_observer)
        reference = (float(band_integral(library.wavelengths, reference_values*reference_mix,
                                         library.responses['johnson_v'])) if library else reference_values)
        if reference <= 0:
            raise DomainError('invalid_scale', 'Нет положительного отклика опорного зенита.')
        anchor_q = 10 ** (-.4 * (data['anchor_v_mag_arcsec2'] - 22.))
        coefficient = anchor_q / reference
        provided = data.get('photometric_calibration')
        if provided is not None:
            document = {k: v for k, v in provided.items() if k != 'id'}
            if provided['id'] != calibration_key(document) or provided['scope'] != scope:
                raise DomainError('incompatible_calibration', 'Калибровка повреждена или относится к другой области, полосе или модели.')
            coefficient = float(provided['source_coefficient'])
            anchor_q = coefficient * reference
        scale_origin = 'provided_calibration' if provided else 'scenario'
        calibration = {'reference_period': group.reference_period,
            'reference_atmosphere': asdict(reference_atmosphere), 'reference_raw_zenith': reference,
            'anchor_v_mag_arcsec2': float(22-2.5*np.log10(anchor_q)), 'source_coefficient': coefficient,
            'reference_numerical_profile': 'standard', 'field_accuracy_certified': False,
            'scope': scope, 'origin': scale_origin, 'anchor_observer': asdict(anchor_observer),
            'provided_calibration_id': provided['id'] if provided else None,
            'reference_mix': spectral_config['reference_mix'] if library else None}
        scale = SourceScale(group.identity, coefficient, reference, anchor_q, calibration)
        raw = np.zeros((len(altitudes), len(azimuths)))
        spectral_raw = np.zeros((*raw.shape, len(library.wavelengths))) if library else None
        for row, altitude in enumerate(altitudes):
            if altitude == 90:
                values = response(data['period'], 0, 90, atmosphere, settings, block)
                if library:
                    spectral_raw[row, :, :] = values
                    raw[row, :] = band_integral(library.wavelengths, values*mix, library.responses['johnson_v'])
                else:
                    raw[row, :] = values
            else:
                for col, azimuth in enumerate(azimuths):
                    values = response(data['period'], azimuth, altitude, atmosphere, settings, block)
                    if library:
                        spectral_raw[row, col] = values
                        raw[row, col] = band_integral(library.wavelengths, values*mix, library.responses['johnson_v'])
                    else:
                        raw[row, col] = values
        check()
        _, source_provenance = group.select(data['period'], block)
        metadata = {'observer': asdict(observer), 'atmosphere': asdict(atmosphere),
                    'source_basis_id': group.identity, 'provenance': source_provenance,
                    'solver': asdict(settings), 'solver_version': self.solver.version}
        grid = TransferGrid(azimuths, altitudes, raw, metadata,
                            content_key({'metadata': metadata, 'values': raw.tolist()}))
        legacy = {'schema_version': 1, 'mode': data['mode'], 'observer': asdict(observer),
            'directions': data['directions'], 'atmosphere': asdict(atmosphere),
            'natural': {'mode': 'uniform', 'v_mag_arcsec2': data['natural_v_mag_arcsec2'],
                        'source': 'Uniform natural-background scenario; not measured separately.'},
            'artificial': {'mode': 'grid', 'sky_grid': 'current-transfer-field',
                'anchor_v_mag_arcsec2': calibration['anchor_v_mag_arcsec2'], 'scale_origin': scale_origin,
                'source': 'Fixed source coefficient of the reference scene; no independent absolute inversion.',
                'band_bridge': 'effective_550nm_shape_anchored_in_V'}}
        if data['mode'] == 'dated':
            legacy['time'] = data['time']
        else:
            legacy['moon'] = data['moon']
        progress({'phase': 'Луна, звёзды и общая сцена', 'completed': total, 'total': total})
        resources = ObservationResources({'current-transfer-field': grid},
            load_stars() if data['mode'] == 'dated' else None, scale)
        observation = calculate_scene(legacy, resources)
        observation['atmosphere_origin'] = resolved['atmosphere_origin']
        if provided:
            observation['assumptions_and_limitations'] = [
                'ground_validation_limited_to_supplied_nights_and_sites_not_global_accuracy'
                if warning == 'no_independent_field_validation' else warning
                for warning in observation['assumptions_and_limitations']]
        check()
        stars = observation.pop('stars')
        for direction in observation['directions']:
            col = int(np.flatnonzero(azimuths == direction['azimuth_deg'])[0])
            row = int(np.flatnonzero(altitudes == direction['altitude_deg'])[0])
            direction['raw_transfer_response'] = float(raw[row, col])
        spectral_output = None
        if library:
            reference_radiance = library.zero_v * 10**(-.4*22) * ARCSEC2_PER_SR
            spectral_rows = []
            for direction in observation['directions']:
                col = int(np.flatnonzero(azimuths == direction['azimuth_deg'])[0])
                row = int(np.flatnonzero(altitudes == direction['altitude_deg'])[0])
                radiance = spectral_raw[row, col] * mix * coefficient * reference_radiance
                summary = response_summary(library.wavelengths, radiance, library.responses, library.zero_v)
                spectral_rows.append({'azimuth_deg': direction['azimuth_deg'], 'altitude_deg': direction['altitude_deg'],
                    'city_radiance_W_m2_sr_nm': radiance.tolist(), 'city_v_relative': direction['artificial_relative'],
                    'photopic_luminance_cd_m2': summary['photopic'],
                    'scotopic_luminance_cd_m2': summary['scotopic'],
                    'scotopic_photopic_ratio': summary['scotopic_photopic_ratio'],
                    'johnson_b_radiance_W_m2_sr': summary['johnson_b_energy']})
            stars, stellar_metadata, examples = stellar_spectral_observation(stars, observer, atmosphere,
                library.wavelengths, library.responses, library.colours, library.measured,
                library.zero_b, library.zero_v, spectral_config['angstrom_exponent'],
                continuum=spectral_config['stars'] == 'catalogue_continuum')
            observation['stars_metadata']['spectral'] = stellar_metadata
            observation['photometry']['band_bridge'] = 'City: explicit Generic/Johnson.V integration; Moon: empirical KS V; natural: V-only scenario.'
            observation['assumptions_and_limitations'] = [w for w in observation['assumptions_and_limitations']
                if w != 'effective_550nm_shape_anchored_in_V_not_spectral_integration']
            observation['assumptions_and_limitations'] += ['uniform_scenario_lamp_spectrum_not_inverted_from_DNB',
                'no_gas_absorption_or_spectral_SSA_g', 'moon_and_natural_spectra_unavailable']
            spectral_output = {'wavelength_nm': library.wavelengths.tolist(), 'directions': spectral_rows,
                'city_only': True, 'total_sky_spectrum_available': False,
                'source_mix': spectral_config['mix'], 'reference_mix': spectral_config['reference_mix'],
                'source_coefficient_origin': scale_origin, 'library_id': library.identity,
                'source_visible_power_scale_W': coefficient * reference_radiance,
                'stellar_examples': examples, 'stellar_summary': stellar_metadata,
                'reference_v_radiance_W_m2_sr': reference_radiance,
                'inferred_local_lamp_inventory': False, 'display_colour_modelled': False}
        resolved['scale_origin'] = scale_origin
        result = {'schema_version': 1, 'application': 'lightpollution-mvp', 'model_version': MODEL_VERSION,
            'request': data, 'resolved_scene': resolved,
            'sky': SkyField(tuple(observation['directions']), observation['photometry']).to_dict(),
            'stars': StarFluxTable(tuple(stars), observation['stars_metadata']).to_dict(),
            'observation': observation, 'calibration': calibration,
            'comparison_basis_id': content_key({'scope': scope, 'source_coefficient': coefficient,
                'origin': scale_origin, 'provided_id': provided['id'] if provided else None}),
            'data_provenance': {'region': entry, 'QA': group.report, 'source_basis_id': group.identity,
                                'source_period': data['period'], 'geographic_terrain_reconstructed': False},
            'capabilities': ['relative_background'] + (['stellar_geometry', 'relative_stellar_flux']
                                                      if data['mode'] == 'dated' else []),
            'unavailable_capabilities': ['eye_detection', 'camera_electrons', 'spectral_radiance', 'validated_display'],
            'cache': stats, 'runtime': {'elapsed_seconds': time.perf_counter() - started},
            'physical_error_bound': None}
        if library:
            result['spectral'] = spectral_output
            result['sky']['spectral_axis'] = library.wavelengths.tolist()
            result['capabilities'] += ['city_spectral_radiance', 'spectral_band_responses']
            result['unavailable_capabilities'].remove('spectral_radiance')
            result['unavailable_capabilities'].append('total_sky_spectral_radiance')
        _write_checked(scene_path, result)
        progress({'phase': 'Готово', 'completed': total, 'total': total})
        return RunResult(result)


def export_result(result: RunResult, output: Path):
    value = result.to_dict()
    write_json(output / 'scene.json', value['request'])
    write_json(output / 'resolved-scene.json', value['resolved_scene'])
    write_observation({**value['observation'], 'stars': value['stars']['rows']}, output)
    write_json(output / 'result.json', value)


def compare_results(left: dict, right: dict):
    if left.get('comparison_basis_id') != right.get('comparison_basis_id') or not left.get('comparison_basis_id'):
        raise DomainError('incompatible_scale', 'Количественное сравнение требует общей области источников, модели и одной опоры масштаба.')
    first = {(r['azimuth_deg'], r['altitude_deg']): r for r in left['sky']['directions']}
    rows = []
    for b in right['sky']['directions']:
        key = b['azimuth_deg'], b['altitude_deg']
        if key not in first:
            continue
        a = first[key]
        valid = a['supported'] and b['supported'] and a['total_relative'] > 0 and b['total_relative'] > 0
        rows.append({'azimuth_deg': key[0], 'altitude_deg': key[1], 'supported': bool(valid),
            'right_over_left': b['total_relative'] / a['total_relative'] if valid else None,
            'right_minus_left_mag': b['total_v_mag_arcsec2'] - a['total_v_mag_arcsec2'] if valid else None,
            'reason': None if valid else 'unsupported_or_zero_background'})
    if not rows:
        raise DomainError('no_common_directions', 'У сценариев нет общих направлений.')
    return {'directions': rows, 'comparison_basis_id': left['comparison_basis_id'],
            'left_observer': left['request']['observer'], 'right_observer': right['request']['observer'],
            'scale_origin': left['calibration']['origin'], 'physical_accuracy_claimed': False}
