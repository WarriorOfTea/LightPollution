"""One public application use case for CLI, jobs and HTTP."""

from dataclasses import asdict, replace
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
from ..models.background import BackgroundModel
from ..models.surface import surface_balance, surface_metadata, surface_reference
from ..models.transfer import SingleScatteringSolver, TransferSolver
from ..models.spectra import (ARCSEC2_PER_SR, band_integral,
                             response_summary, stellar_spectral_observation)
from ..observation import observation_context
from ..physics import Observer, SolverSettings
from ..skyview import load_stars

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 'terrain-finite-near-field-kstars-eye-2'
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
            if not context['astronomical_darkness'] and data.get('background', {}).get('twilight') != 'patat2006':
                raise DomainError('twilight', 'Солнце выше −18°: включите ограниченную модель сумерек Patat в настройках фона.',
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
        terrain = None
        solver = self.solver
        observer = spec.observer
        if 'terrain' in data:
            from ..adapters.terrain import load_terrain
            from ..models.topographic_transfer import TopographicSolver
            terrain = load_terrain(data['terrain']['dataset'], self.workspace.parent/'terrain')
            try:
                observer = terrain.observer(observer)
                anchor_observer = terrain.observer(anchor_observer)
            except ValueError as exc:
                raise DomainError('terrain_coverage', str(exc)) from exc
            solver = TopographicSolver(terrain, data['terrain']['visibility_step_m'])
            resolved['observer_msl'] = asdict(observer)
            resolved['anchor_observer_msl'] = asdict(anchor_observer)
            resolved['height_datum'] = 'Input height is AGL above DSM; transfer radial height approximates EGM2008 orthometric MSL on a sphere.'
            resolved['terrain'] = terrain.metadata
            if min(spec.atmosphere.top_m, atmosphere_from_mapping(data['reference_atmosphere']).top_m) <= max(observer.height_m, anchor_observer.height_m):
                raise DomainError('unsupported_atmosphere', 'Верх атмосферы должен находиться выше наблюдателя и опорной точки с учётом DSM.')
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
        transfer_code = _code_identity(['physics.py', 'models/transfer.py', 'models/spectra.py', 'sources.py',
                                       'models/terrain.py', 'models/topographic_transfer.py'])
        scene_code = _code_identity(['models/observation.py', 'photometry.py', 'moonlight.py',
                                     'skyview.py', 'observation.py', 'catalogues/bsc5.csv',
                                     'application/planner.py', 'domain/scene.py', 'domain/calibration.py',
                                     'adapters/spectral_library.py', 'models/background.py', 'models/surface.py',
                                     'models/eye.py', 'models/eye_field.py', 'application/eye.py',
                                     'models/camera.py','models/camera_field.py','application/camera.py',
                                     'models/kstars_horizon.py','adapters/catalogue_identity.py',
                                     'models/catalogue_photometry.py',
                                     'catalogues/bsc5-cross-identifications.csv'])
        request_key = content_key({'scene': resolved, 'source_basis': group.identity,
            'transfer_code': transfer_code, 'scene_code': scene_code, 'solver': solver.version,
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
        surface = data.get('surface')
        atmosphere = spec.atmosphere
        reference_atmosphere = atmosphere_from_mapping(data['reference_atmosphere'])
        azimuths = np.asarray(sorted({0., 90., 180., 270., *[v[0] for v in data['directions']]}))
        altitudes = np.asarray(sorted({10., 90., *[v[1] for v in data['directions']]}))
        if 'eye' in data:
            # A view must not silently rely on the sparse single-ray grid.
            # These extra samples are physical transfers, not interpolated data.
            azimuths=np.unique(np.r_[azimuths,np.arange(0.,360.,45.)])
            altitudes=np.unique(np.r_[altitudes,20.,45.,70.])
        total = 1 + 1 + len(azimuths) * (len(altitudes) - 1)
        if surface and surface['direct_horizontal_fraction'] > 0 and surface['direct_upward_fraction'] > 0:
            total *= 2
        stats = {'scene_hit': False, 'transfer_hits': 0, 'transfer_misses': 0}
        completed = 0
        groups = {tuple(asdict(observer).values()): group}
        transport = {'mode': 'spectral_single_scattering' if library else 'grey_550nm',
                     'wavelength_nm': library.wavelengths.tolist() if library else [550.],
                     'angstrom_exponent': spectral_config['angstrom_exponent'] if library else None}
        scope = {'source_basis_id': group.identity, 'transfer_code': transfer_code,
                 'solver_version': solver.version, 'transport': transport,
                 'band': 'Generic/Johnson.V energy response' if library else 'effective_V_550nm',
                 'spectral_library': library.identity if library else None,
                 'reference_v_mag_arcsec2': 22., 'height_datum': resolved['height_datum']}
        if surface:
            scope['surface_reference'] = {'model': 'fixed_lamp_first_lambertian_ground_reflection_1',
                'code': _code_identity(['models/surface.py']), **surface_reference(surface)}
        if 'near_field' in data:
            scope['near_field'] = data['near_field']
        if terrain:
            scope['terrain'] = {'id': data['terrain']['dataset'],
                'prepared_sha256': terrain.metadata.get('prepared_sha256', content_key(terrain.metadata)),
                'visibility_step_m': data['terrain']['visibility_step_m']}

        def kernel(period, az, alt, atm, numerical, aggregation, target=None):
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
            near = None
            if 'near_field' in data:
                near = {**data['near_field'], 'observer_agl_m': (
                    data.get('anchor_observer', asdict(region_anchor(entry)))['height_m']
                    if target == anchor_observer else data['observer']['height_m'])}
            sources, provenance = (selected_group.select(period, aggregation, direction, finite_cells=near)
                                   if near else selected_group.select(period, aggregation, direction))
            if terrain:
                sources = terrain.lift_sources(sources)
            source_digest = hashlib.sha256(sources.positions_m.tobytes() + sources.power.tobytes()).hexdigest()
            identity = content_key({'source_basis': group.identity, 'source_arrays': source_digest,
                'period': period, 'observer': asdict(target), 'atmosphere': asdict(atm),
                'settings': asdict(numerical), 'block': aggregation, 'azimuth': float(az),
                'altitude': float(alt), 'solver': solver.version, 'code': transfer_code, 'transport': transport,
                'terrain': scope.get('terrain'), 'near_field': data.get('near_field')})
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
                    value = np.asarray(solver.spectrum(target, sources, float(az), float(alt), atm, numerical,
                                        library.wavelengths, spectral_config['angstrom_exponent']), dtype=float)
                    if value.shape != library.wavelengths.shape:
                        raise DomainError('invalid_solver_output', 'Решатель вернул спектр другой размерности.')
                else:
                    value = float(solver.direction(target, sources, float(az), float(alt), atm, numerical))
                if np.any(~np.isfinite(value)) or np.any(np.asarray(value) < 0):
                    raise DomainError('invalid_solver_output', 'Решатель вернул недопустимое значение.')
                check()
                _write_checked(path, {'identity': identity, 'radiance': np.asarray(value).tolist()})
                stats['transfer_misses'] += 1
            completed += 1
            progress({'phase': 'Атмосферный перенос', 'completed': completed, 'total': total,
                      'cached_directions': stats['transfer_hits']})
            return value

        def response(period, az, alt, atm, numerical, aggregation, target=None, *, reference_surface=False):
            if surface is None:
                return kernel(period, az, alt, atm, numerical, aggregation, target), None, None
            balance = surface_balance(surface, reference=reference_surface)
            dm = balance['direct_multiplier_of_reference_upward_power']
            rm = balance['reflected_multiplier_of_reference_upward_power']
            lambert = kernel(period, az, alt, atm, replace(numerical, horizontal_fraction=0.), aggregation, target)
            if surface['direct_horizontal_fraction'] == 0 or dm == 0:
                direct_kernel = lambert
            else:
                direct_kernel = kernel(period, az, alt, atm,
                    replace(numerical, horizontal_fraction=surface['direct_horizontal_fraction']), aggregation, target)
            direct, reflected = dm*direct_kernel, rm*lambert
            return direct+reflected, direct, reflected

        reference_values = response(group.reference_period, 0, 90, reference_atmosphere,
                                    *PROFILES['standard'], target=anchor_observer, reference_surface=True)[0]
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
        direct_raw = np.zeros_like(raw) if surface else None
        reflected_raw = np.zeros_like(raw) if surface else None
        spectral_raw = np.zeros((*raw.shape, len(library.wavelengths))) if library else None
        for row, altitude in enumerate(altitudes):
            if altitude == 90:
                values, direct, reflected = response(data['period'], 0, 90, atmosphere, settings, block)
                if library:
                    spectral_raw[row, :, :] = values
                    raw[row, :] = band_integral(library.wavelengths, values*mix, library.responses['johnson_v'])
                else:
                    raw[row, :] = values
                if surface:
                    direct_raw[row, :] = band_integral(library.wavelengths, direct*mix, library.responses['johnson_v']) if library else direct
                    reflected_raw[row, :] = band_integral(library.wavelengths, reflected*mix, library.responses['johnson_v']) if library else reflected
            else:
                for col, azimuth in enumerate(azimuths):
                    values, direct, reflected = response(data['period'], azimuth, altitude, atmosphere, settings, block)
                    if library:
                        spectral_raw[row, col] = values
                        raw[row, col] = band_integral(library.wavelengths, values*mix, library.responses['johnson_v'])
                    else:
                        raw[row, col] = values
                    if surface:
                        direct_raw[row, col] = band_integral(library.wavelengths, direct*mix, library.responses['johnson_v']) if library else direct
                        reflected_raw[row, col] = band_integral(library.wavelengths, reflected*mix, library.responses['johnson_v']) if library else reflected
        check()
        _, source_provenance = (group.select(data['period'], block, finite_cells={**data['near_field'],
            'observer_agl_m': data['observer']['height_m']}) if 'near_field' in data else group.select(data['period'], block))
        metadata = {'observer': asdict(observer), 'atmosphere': asdict(atmosphere),
                    'source_basis_id': group.identity, 'provenance': source_provenance,
                    'solver': asdict(settings), 'solver_version': solver.version}
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
        background = (BackgroundModel(data['background'], observer, atmosphere, data['natural_v_mag_arcsec2'])
                      if 'background' in data else None)
        sampled = [[float(az),float(alt)] for alt in altitudes for az in azimuths] if 'eye' in data else None
        observation = calculate_scene(legacy, resources, background=background, sample_directions=sampled)
        angular_rows = observation.pop('sampled_directions', None)
        if background:
            observation['background_model'] = background.metadata()
        if surface:
            observation['surface_model'] = surface_metadata(surface)
            observation['assumptions_and_limitations'] += ['snow_fraction_and_albedos_are_scenarios',
                'grey_first_surface_reflection_only_no_repeated_ground_atmosphere_feedback']
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
            if surface:
                direction['artificial_direct_relative'] = float(direct_raw[row, col]*coefficient)
                direction['artificial_reflected_relative'] = float(reflected_raw[row, col]*coefficient)
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
            if surface:
                spectral_output['source_power_scale_reference'] = 'snow_free_upward_power_before_observer_exclusions'
                spectral_output['source_lamp_power_scale_W'] = (coefficient*reference_radiance /
                    surface_balance(surface, reference=True)['snow_free_upward_fraction_of_lamp_power'])
        terrain_output = None
        if terrain:
            azimuth_axis = np.arange(0., 360., 2.)
            try:
                horizon = terrain.horizon(observer, azimuth_axis, data['terrain']['horizon_range_m'],
                                          data['terrain']['visibility_step_m'])
            except ValueError as exc:
                raise DomainError('terrain_coverage', str(exc)) from exc
            def obstruction(az, alt):
                # Exact sampled ray visibility for the actual requested sightline.
                end = observer.position + 1000000*observer.direction(float(az), float(alt))
                return not bool(terrain.visible(observer.position, end, data['terrain']['visibility_step_m']))
            blocked = []
            if stars:
                endpoints = np.asarray([observer.position+1000000*observer.direction(
                    row['azimuth_deg'], row['altitude_deg']) for row in stars])
                visible_stars = terrain.visible(observer.position, endpoints, data['terrain']['visibility_step_m'])
                blocked = [row for row, visible in zip(stars, visible_stars, strict=True) if not visible]
            rejected = {r['hr'] for r in blocked}
            stars = [r for r in stars if r['hr'] not in rejected]
            observation['stars_metadata']['blocked_by_terrain'] = len(blocked)
            observation['stars_metadata']['exported'] = len(stars)
            if library:
                observation['stars_metadata']['spectral_counts_before_terrain_mask'] = dict(stellar_metadata)
                visible_counts = {'measured': 0, 'catalogue_continuum': 0, 'unavailable': 0}
                for star in stars:
                    recipe = star.get('spectrum_recipe', {}).get('kind')
                    kind = ('measured' if recipe == 'measured_reference_shape' else
                            'catalogue_continuum' if recipe == 'B_V_fitted_Planck_continuum' else 'unavailable')
                    visible_counts[kind] += 1
                visible_summary = {**stellar_metadata, **visible_counts,
                    'available': len(stars)-visible_counts['unavailable'], 'total_catalogue_rows': len(stars),
                    'scope': 'exported stars after DSM visibility mask'}
                observation['stars_metadata']['spectral'] = visible_summary
                spectral_output['stellar_summary'] = visible_summary
                spectral_output['stellar_examples'] = [row for row in spectral_output['stellar_examples'] if row['hr'] not in rejected]
            for direction in observation['directions']:
                hidden = obstruction(direction['azimuth_deg'], direction['altitude_deg'])
                direction['terrain_obstructed'] = hidden
                if hidden:
                    direction.setdefault('support_reasons', []).append('terrain_obstructed_unknown_ground_radiance')
                    direction.update(supported=False, status='terrain_obstructed_unknown_ground_radiance',
                                     total_relative=None, total_v_mag_arcsec2=None)
            terrain_output = {'dataset': terrain.metadata, 'horizon_azimuth_deg': azimuth_axis.tolist(),
                'horizon_altitude_deg': horizon.tolist(), 'range_m': data['terrain']['horizon_range_m'],
                'visibility_step_m': data['terrain']['visibility_step_m'],
                'observer_msl': asdict(observer), 'observer_agl_m': data['observer']['height_m'],
                'source_to_scatter_occlusion': True, 'scatter_to_observer_occlusion': True,
                'source_facets_horizontal': True, 'individual_buildings_or_lamps_reconstructed': False,
                'blocked_stars': len(blocked), 'physical_error_bound': None}
            observation['terrain_model'] = terrain_output
            observation['assumptions_and_limitations'].append('DSM_sampled_occlusion_and_spherical_EGM2008_approximation_not_city_mesh')
        origin = data.get('integration', {})
        if 'artificial_horizon' in origin:
            from ..models.kstars_horizon import horizon_visible
            hidden_directions=[r for r in observation['directions'] if not bool(horizon_visible(
                origin['artificial_horizon'],r['azimuth_deg'],r['altitude_deg']))]
        else:
            hidden_directions=observation['directions'] if origin.get('artificial_horizon_visible') is False else []
        for direction in hidden_directions:
            direction.setdefault('support_reasons', []).append('kstars_artificial_horizon_obstructed')
            direction.update(supported=False, status='kstars_artificial_horizon_obstructed',
                             total_relative=None, total_v_mag_arcsec2=None)
        resolved['scale_origin'] = scale_origin
        result = {'schema_version': 1, 'application': 'lightpollution-mvp', 'model_version': MODEL_VERSION,
            'request': data, 'resolved_scene': resolved,
            'sky': SkyField(tuple(observation['directions']), observation['photometry']).to_dict(),
            'stars': StarFluxTable(tuple(stars), observation['stars_metadata']).to_dict(),
            'observation': observation, 'calibration': calibration,
            'comparison_basis_id': content_key({'scope': scope, 'source_coefficient': coefficient,
                'origin': scale_origin, 'provided_id': provided['id'] if provided else None}),
            'data_provenance': {'region': entry, 'QA': {**group.report, **({'local_exclusion_radius_m': 0} if 'near_field' in data else {})},
                                'source_basis_id': group.identity, 'source_period': data['period'],
                                'geographic_terrain_reconstructed': terrain is not None},
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
        if background:
            result['capabilities'] += ['directional_natural_background']
            if background.allows_twilight:
                result['capabilities'].append('bounded_empirical_twilight')
        if surface:
            result['capabilities'].append('fixed_lamp_snow_scenarios')
        if terrain:
            result['terrain'] = terrain_output
            result['capabilities'].append('terrain_height_and_sampled_occlusion')
        if 'near_field' in data:
            result['capabilities'].append('finite_nearby_emitting_cells')
        if 'integration' in data:
            result['integration'] = data['integration']
        if 'eye' in data:
            from .eye import build_eye_result
            assert library is not None and spectral_raw is not None
            radiance = spectral_raw*mix*coefficient*reference_radiance
            photopic = 683*band_integral(library.wavelengths,radiance,library.responses['photopic'])
            scotopic = 1700*band_integral(library.wavelengths,radiance,library.responses['scotopic'])
            supported = np.asarray([r['supported'] for r in angular_rows],dtype=bool)
            if terrain:
                endpoints = np.asarray([observer.position+1000000*observer.direction(*direction) for direction in sampled])
                supported &= terrain.visible(observer.position,endpoints,data['terrain']['visibility_step_m'])
            field = {'azimuth_deg':azimuths.tolist(),'altitude_deg':altitudes.tolist(),
                'city_photopic_luminance_cd_m2':photopic.tolist(),
                'city_scotopic_luminance_cd_m2':scotopic.tolist(),
                'physical_supported':supported.reshape(raw.shape).tolist(),
                'unresolved_v_relative':np.asarray([r['natural_relative']+(r.get('moon_relative') or 0.)+
                    (r.get('twilight_relative') or 0.) for r in angular_rows]).reshape(raw.shape).tolist(),
                'source_basis_id':group.identity,'calibration_origin':scale_origin}
            # Refine the requested view without changing the source coefficient.
            geometry=observation.get('geometry') or {}
            moon=observation.get('moon') or {}
            if (geometry.get('sun',{}).get('altitude_deg',90)<=-18 and moon.get('altitude_deg',-90)<=0
                    and (origin.get('artificial_horizon_visible') is not False or 'artificial_horizon' in origin)):
                from ..models.eye import eye_config
                from ..models.eye_field import build_view_patch
                config=eye_config(data['eye'])
                view={'azimuth_deg':data['directions'][0][0],'altitude_deg':data['directions'][0][1],
                      'horizontal_fov_deg':90.,'width':1800,'height':1100,**config.get('view',{})}
                def eye_ray(az,alt,numerical=settings,aggregation=block):
                    if abs(alt-90)<1e-8:az,alt=0.,90.
                    values=response(data['period'],az,alt,atmosphere,numerical,aggregation)[0]
                    radiance=values*mix*coefficient*reference_radiance
                    p=float(683*band_integral(library.wavelengths,radiance,library.responses['photopic']))
                    s=float(1700*band_integral(library.wavelengths,radiance,library.responses['scotopic']))
                    residual=(background.evaluate(np.asarray(az),np.asarray(alt),geometry)[0]['natural_relative']
                        if background else 10**(-.4*(data['natural_v_mag_arcsec2']-22.)))
                    supported=True
                    if terrain:
                        endpoint=observer.position+1000000*observer.direction(az,alt)
                        supported=bool(terrain.visible(observer.position,endpoint,data['terrain']['visibility_step_m']))
                    progress({'phase':'Уточнение поля глаза; контрольные лучи',
                              'completed':stats['transfer_hits']+stats['transfer_misses'],'total':None})
                    return [p,s,float(residual),supported]
                sampling=config['sampling']
                total=None  # adaptive work is not known before its probes
                field['view_patch']=build_view_patch(view,eye_ray,
                    target=sampling['target_relative_error'],max_grid_size=sampling['max_grid_size'],
                    validation=sampling['validation'],check=check,
                    evaluate_standard=lambda az,alt:eye_ray(az,alt,*PROFILES['standard']))
                field['view_patch']['numerical_profile']=data['profile']
                total=stats['transfer_hits']+stats['transfer_misses']
            reference_star = 1700*band_integral(library.wavelengths,library.measured[7001],
                                              library.responses['scotopic'])*10**(.4*.03)
            result['eye'] = build_eye_result(result,data['eye'],field,float(reference_star))
            result['capabilities'].append('bounded_scotopic_eye_threshold')
            result['unavailable_capabilities'].remove('eye_detection')
            result['unavailable_capabilities'] += ['validated_eye_appearance','eye_colour','adaptation_time_course']
            result['stars']['detected_by_eye'] = result['eye']['summary']['above_threshold']
            result['stars']['metadata']['detected_by_eye_not_modelled'] = False
            observation['stars_metadata']['detected_by_eye_not_modelled'] = False
        if 'camera' in data:
            from .camera import build_camera_result,camera_response
            assert library is not None
            config=data['camera']
            weights,exact_v,residual_factor,_=camera_response(config,library)
            def camera_ray(az,alt):
                values=response(data['period'],az,alt,atmosphere,settings,block)[0]
                spectral_radiance=values*mix*coefficient*reference_radiance
                city=float(band_integral(library.wavelengths,spectral_radiance,weights))
                # Reuse the same Moon/natural/twilight models and support rules.
                local=calculate_scene({**legacy,'directions':[[az,alt]]},resources,background=background)['directions'][0]
                residual=local['natural_relative']+(local.get('moon_relative') or 0.)+(local.get('twilight_relative') or 0.)
                v_city=float(band_integral(library.wavelengths,spectral_radiance,library.responses['johnson_v'])/reference_radiance)
                supported=local['supported']
                if terrain:
                    endpoint=observer.position+1000000*observer.direction(az,alt)
                    supported=bool(supported and terrain.visible(observer.position,endpoint,data['terrain']['visibility_step_m']))
                progress({'phase':'Камера: физический фон по полю','completed':stats['transfer_hits']+stats['transfer_misses'],'total':None})
                return [city,float(residual*reference_radiance*(residual_factor or 0.)),v_city+residual,supported]
            result['camera']=build_camera_result(result,config,camera_ray,check)
            if result['camera']['available']:
                result['capabilities'].append('scenario_linear_camera_electrons')
                result['unavailable_capabilities'].remove('camera_electrons')
            result['unavailable_capabilities']+=['validated_camera_prediction','camera_CFA_colour','camera_temporal_background_integration']
        result['runtime']['elapsed_seconds']=time.perf_counter()-started
        _write_checked(scene_path, result)
        progress({'phase': 'Готово', 'completed': total, 'total': total})
        return RunResult(result)


def export_result(result: RunResult, output: Path):
    value = result.to_dict()
    write_json(output / 'scene.json', value['request'])
    write_json(output / 'resolved-scene.json', value['resolved_scene'])
    write_observation({**value['observation'], 'stars': value['stars']['rows']}, output)
    if 'eye' in value:
        from .eye import export_eye
        from ..interfaces.eye_view import render_eye_view
        value['eye']['display']=render_eye_view(value,output)
        export_eye(value['eye'],output)
    if 'camera' in value:
        from ..interfaces.camera_view import export_camera
        value['camera']['exports']=export_camera(value,output)
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
