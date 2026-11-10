"""One public application use case for CLI, jobs and HTTP."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..adapters.observation_io import write_observation
from ..adapters.sources import LocalSourceRepository, SourceRepository
from ..adapters.storage import content_key, file_hash, read_json, write_json
from ..atmosphere_config import atmosphere_from_mapping
from ..domain.fields import ObservationResources, RunResult, SkyField, SourceScale, StarFluxTable, TransferGrid
from ..domain.scene import Cancelled, DomainError, SceneSpec
from ..models.observation import calculate_scene
from ..models.transfer import SingleScatteringSolver, TransferSolver
from ..observation import observation_context
from ..physics import SolverSettings
from ..skyview import load_stars

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 'mvp-effective-v-1'
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
        progress({'phase': 'Подготовка источников и QA', 'completed': 0, 'total': None})
        group = self.sources.prepare(data['region'], data['period'], spec.observer,
                                     allow_download=data['download_missing'])
        check()
        transfer_code = _code_identity(['physics.py', 'models/transfer.py', 'sources.py'])
        scene_code = _code_identity(['models/observation.py', 'photometry.py', 'moonlight.py',
                                     'skyview.py', 'observation.py', 'catalogues/bsc5.csv',
                                     'application/planner.py', 'domain/scene.py'])
        request_key = content_key({'scene': resolved, 'source_basis': group.identity,
            'transfer_code': transfer_code, 'scene_code': scene_code, 'solver': self.solver.version,
            'application_model': MODEL_VERSION})
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

        def response(period, az, alt, atm, numerical, aggregation):
            nonlocal completed
            check()
            direction = (float(az), float(alt)) if alt <= 10 else None
            sources, provenance = group.select(period, aggregation, direction)
            source_digest = hashlib.sha256(sources.positions_m.tobytes() + sources.power.tobytes()).hexdigest()
            identity = content_key({'source_basis': group.identity, 'source_arrays': source_digest,
                'period': period, 'observer': asdict(observer), 'atmosphere': asdict(atm),
                'settings': asdict(numerical), 'block': aggregation, 'azimuth': float(az),
                'altitude': float(alt), 'solver': self.solver.version, 'code': transfer_code})
            path = self.workspace / 'cache/transfer' / (identity + '.json')
            stored = _read_checked(path)
            if (isinstance(stored, dict) and stored.get('identity') == identity
                    and isinstance(stored.get('radiance'), (int, float))
                    and np.isfinite(stored['radiance']) and stored['radiance'] >= 0):
                value = stored['radiance']
                stats['transfer_hits'] += 1
            else:
                value = float(self.solver.direction(observer, sources, float(az), float(alt), atm, numerical))
                if not np.isfinite(value) or value < 0:
                    raise DomainError('invalid_solver_output', 'Решатель вернул недопустимое значение.')
                check()
                _write_checked(path, {'identity': identity, 'radiance': value})
                stats['transfer_misses'] += 1
            completed += 1
            progress({'phase': 'Атмосферный перенос', 'completed': completed, 'total': total,
                      'cached_directions': stats['transfer_hits']})
            return value

        reference = response(group.reference_period, 0, 90, reference_atmosphere, *PROFILES['standard'])
        if reference <= 0:
            raise DomainError('invalid_scale', 'Нет положительного отклика опорного зенита.')
        anchor_q = 10 ** (-.4 * (data['anchor_v_mag_arcsec2'] - 22.))
        coefficient = anchor_q / reference
        calibration = {'origin': 'scenario', 'reference_period': group.reference_period,
            'reference_atmosphere': asdict(reference_atmosphere), 'reference_raw_zenith': reference,
            'anchor_v_mag_arcsec2': data['anchor_v_mag_arcsec2'], 'source_coefficient': coefficient,
            'reference_numerical_profile': 'standard', 'field_accuracy_certified': False,
            'scope': {'source_basis_id': group.identity, 'observer': asdict(observer),
                      'band': 'effective V; 550 nm grey shape', 'height_datum': resolved['height_datum']}}
        scale = SourceScale(group.identity, coefficient, reference, anchor_q, calibration)
        raw = np.zeros((len(altitudes), len(azimuths)))
        for row, altitude in enumerate(altitudes):
            if altitude == 90:
                raw[row, :] = response(data['period'], 0, 90, atmosphere, settings, block)
            else:
                for col, azimuth in enumerate(azimuths):
                    raw[row, col] = response(data['period'], azimuth, altitude, atmosphere, settings, block)
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
                'anchor_v_mag_arcsec2': data['anchor_v_mag_arcsec2'], 'scale_origin': 'scenario',
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
        check()
        stars = observation.pop('stars')
        result = {'schema_version': 1, 'application': 'lightpollution-mvp', 'model_version': MODEL_VERSION,
            'request': data, 'resolved_scene': resolved,
            'sky': SkyField(tuple(observation['directions']), observation['photometry']).to_dict(),
            'stars': StarFluxTable(tuple(stars), observation['stars_metadata']).to_dict(),
            'observation': observation, 'calibration': calibration,
            'comparison_basis_id': content_key({'source_basis': group.identity,
                'reference_atmosphere': asdict(reference_atmosphere), 'anchor_q': anchor_q,
                'reference_mag_arcsec2': 22., 'source_coefficient': coefficient,
                'transfer_code': transfer_code, 'solver': self.solver.version, 'model': MODEL_VERSION}),
            'data_provenance': {'region': entry, 'QA': group.report, 'source_basis_id': group.identity,
                                'source_period': data['period'], 'geographic_terrain_reconstructed': False},
            'capabilities': ['relative_background'] + (['stellar_geometry', 'relative_stellar_flux']
                                                      if data['mode'] == 'dated' else []),
            'unavailable_capabilities': ['eye_detection', 'camera_electrons', 'spectral_radiance', 'validated_display'],
            'cache': stats, 'runtime': {'elapsed_seconds': time.perf_counter() - started},
            'physical_error_bound': None}
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
        raise DomainError('incompatible_scale', 'Количественное сравнение требует одной точки, общей области источников и одной опоры масштаба.')
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
            'scale_origin': 'scenario', 'physical_accuracy_claimed': False}
