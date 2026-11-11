"""Versioned application request. Legacy observation schema 1 remains separate."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json

import numpy as np

from ..atmosphere_config import atmosphere_from_mapping, atmosphere_origin
from ..physics import Atmosphere, Observer
from ..models.background import background_config
from ..models.surface import surface_config


class DomainError(ValueError):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}

    def to_dict(self):
        return {'code': self.code, 'message': self.message, 'details': self.details}


class Cancelled(Exception):
    """Cooperative cancellation at a completed numerical block boundary."""


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise DomainError('invalid_input', f'{name}: требуется конечное число.')
    return float(value)


@dataclass(frozen=True)
class SceneSpec:
    """The private JSON snapshot prevents mutation of an accepted request."""
    _json: str

    @classmethod
    def from_mapping(cls, value: dict):
        if not isinstance(value, dict):
            raise DomainError('invalid_input', 'Сценарий должен быть объектом JSON.')
        data = deepcopy(value)
        allowed = {'schema_version', 'region', 'period', 'mode', 'time', 'observer',
                   'directions', 'atmosphere', 'reference_atmosphere', 'profile',
                   'anchor_v_mag_arcsec2', 'natural_v_mag_arcsec2', 'moon', 'download_missing',
                   'anchor_observer', 'photometric_calibration', 'spectrum', 'background', 'surface',
                   'near_field', 'terrain', 'integration'}
        if set(data) - allowed:
            raise DomainError('invalid_input', 'В сценарии есть неизвестные поля.',
                              {'fields': sorted(set(data) - allowed)})
        if type(data.get('schema_version')) is not int or data['schema_version'] != 2:
            raise DomainError('unsupported_schema', 'Планировщик читает схему 2; observe сохраняет схему 1.')
        if not all(isinstance(data.get(k), str) and data[k] for k in ['region', 'period']):
            raise DomainError('invalid_input', 'Укажите регион и период источников.')
        data.setdefault('mode', 'dated')
        if data['mode'] not in ['dated', 'undated']:
            raise DomainError('invalid_input', 'Режим должен быть dated или undated.')
        if data['mode'] == 'dated':
            if 'moon' in data:
                raise DomainError('invalid_input', 'При заданной дате Луна вычисляется автоматически.')
            try:
                if not isinstance(data.get('time'), str) or 'T' not in data['time']:
                    raise ValueError()
                instant = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))
                if instant.tzinfo is None or instant.utcoffset() is None:
                    raise ValueError()
                data['time'] = instant.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
            except ValueError:
                raise DomainError('invalid_time', 'Нужны дата и время с часовым поясом, например 2026-11-09T21:00:00Z.') from None
        else:
            if 'time' in data or data.get('moon') != {'mode': 'none'}:
                raise DomainError('invalid_input', 'Без даты MVP требует явное moon: {mode: none} и отсутствие time.')
        observer = data.get('observer')
        if not isinstance(observer, dict) or set(observer) != {'latitude_deg', 'longitude_deg', 'height_m'}:
            raise DomainError('invalid_input', 'Укажите широту, долготу и высоту над условной поверхностью.')
        observer = {k: number(v, k) for k, v in observer.items()}
        try:
            Observer(**observer)
        except ValueError as exc:
            raise DomainError('invalid_observer', 'Проверьте координаты и высоту наблюдателя.') from exc
        if not 0 <= observer['height_m'] <= 100:
            raise DomainError('unsupported_height', 'MVP поддерживает 0–100 м над условной общей поверхностью; горный рельеф пока не моделируется.')
        data['observer'] = observer
        if 'anchor_observer' in data:
            anchor = data['anchor_observer']
            if not isinstance(anchor, dict) or set(anchor) != set(observer):
                raise DomainError('invalid_observer', 'Опорная точка требует широту, долготу и высоту.')
            anchor = {k: number(v, k) for k, v in anchor.items()}
            if not (-90 <= anchor['latitude_deg'] <= 90 and -180 <= anchor['longitude_deg'] <= 180
                    and 0 <= anchor['height_m'] <= 100):
                raise DomainError('unsupported_height', 'Проверьте координаты и высоту общей опорной точки.')
            data['anchor_observer'] = anchor
        if 'photometric_calibration' in data:
            calibration = data['photometric_calibration']
            if not isinstance(calibration, dict) or calibration.get('schema_version') != 1:
                raise DomainError('invalid_calibration', 'Нужен документ общей фотометрической калибровки схемы 1.')
            if not isinstance(calibration.get('scope'), dict) or not isinstance(calibration.get('id'), str):
                raise DomainError('invalid_calibration', 'Калибровка должна содержать область применимости и контрольный идентификатор.')
            if number(calibration.get('source_coefficient'), 'source_coefficient') <= 0:
                raise DomainError('invalid_calibration', 'Коэффициент калибровки должен быть положительным.')
        if 'spectrum' in data:
            spectrum = data['spectrum']
            if not isinstance(spectrum, dict) or set(spectrum) - {'mix', 'reference_mix', 'angstrom_exponent', 'stars'}:
                raise DomainError('invalid_spectrum', 'Проверьте параметры спектрального сценария.')
            spectrum.setdefault('mix', {'LED-B3': 1.})
            spectrum.setdefault('reference_mix', {'LED-B3': 1.})
            for key in ['mix', 'reference_mix']:
                mix = spectrum[key]
                if (not isinstance(mix, dict) or not mix or len(mix) > 14
                        or not all(isinstance(k, str) for k in mix)):
                    raise DomainError('invalid_spectrum', 'Смесь должна задавать доли спектральных шаблонов.')
                mix = {k: number(v, k) for k, v in mix.items()}
                if any(v < 0 for v in mix.values()) or not np.isclose(sum(mix.values()), 1., atol=1e-9, rtol=0):
                    raise DomainError('invalid_spectrum', 'Неотрицательные доли мощности смеси должны давать 1.')
                spectrum[key] = mix
            exponent = number(spectrum.get('angstrom_exponent', 1.3), 'angstrom_exponent')
            if not 0 <= exponent <= 4:
                raise DomainError('invalid_spectrum', 'Показатель Ангстрема должен быть в пределах 0–4.')
            spectrum['angstrom_exponent'] = exponent
            spectrum.setdefault('stars', 'catalogue_continuum')
            if spectrum['stars'] not in ['catalogue_continuum', 'measured_only']:
                raise DomainError('invalid_spectrum', 'Спектры звёзд: catalogue_continuum или measured_only.')
        for key, validator in [('background', background_config), ('surface', surface_config)]:
            if key in data:
                try:
                    data[key] = validator(data[key])
                except ValueError as exc:
                    raise DomainError('invalid_' + key, str(exc)) from exc
        if 'near_field' in data:
            near = data['near_field']
            if not isinstance(near, dict) or set(near)-{'radius_m', 'minimum_patch_m', 'quadrature_order'}:
                raise DomainError('invalid_near_field', 'Проверьте настройки конечных ближних ячеек.')
            near = {'radius_m': 3000., 'minimum_patch_m': 2., 'quadrature_order': 3, **near}
            for key, lower, upper in [('radius_m', 1000, 10000), ('minimum_patch_m', 2, 100), ('quadrature_order', 1, 4)]:
                val = number(near[key], key)
                if not lower <= val <= upper or (key=='quadrature_order' and int(val)!=val):
                    raise DomainError('invalid_near_field', f'{key}: вне поддерживаемого диапазона.')
            near['quadrature_order'] = int(near['quadrature_order'])
            if observer['height_m'] < .5:
                raise DomainError('unsupported_height', 'Для конечных ближних источников нужна высота наблюдателя ≥0,5 м над поверхностью.')
            data['near_field'] = near
        if 'terrain' in data:
            terrain = data['terrain']
            if not isinstance(terrain, dict) or set(terrain)-{'dataset', 'visibility_step_m', 'horizon_range_m'}:
                raise DomainError('invalid_terrain', 'Проверьте параметры рельефа.')
            terrain = {'visibility_step_m': 100., 'horizon_range_m': 30000., **terrain}
            if terrain.get('dataset') not in ['demo-flat', 'demo-ridge', 'petersburg-copdem90', 'madrid-copdem90']:
                raise DomainError('invalid_terrain', 'Неизвестный закреплённый набор высот.')
            for key, lower, upper in [('visibility_step_m', 25, 500), ('horizon_range_m', 1000, 50000)]:
                terrain[key] = number(terrain[key], key)
                if not lower <= terrain[key] <= upper:
                    raise DomainError('invalid_terrain', f'{key}: вне поддерживаемого диапазона.')
            data['terrain'] = terrain
        if 'integration' in data:
            origin = data['integration']
            if (not isinstance(origin, dict) or set(origin)-{'provider', 'version', 'snapshot_id',
                    'site_elevation_m', 'focused_object', 'altitude_convention', 'artificial_horizon_visible'}
                    or origin.get('provider') != 'KStars'
                    or not isinstance(origin.get('snapshot_id'), str)
                    or len(origin['snapshot_id']) != 64):
                raise DomainError('invalid_kstars_snapshot', 'Некорректное происхождение сценария KStars.')
            if type(origin.get('artificial_horizon_visible')) not in [bool, type(None)]:
                raise DomainError('invalid_kstars_snapshot', 'Некорректная маска горизонта KStars.')
        directions = data.get('directions', [[0, 90], [0, 45], [90, 45], [180, 45], [270, 45]])
        if not isinstance(directions, list) or not 1 <= len(directions) <= 16:
            raise DomainError('invalid_direction', 'Задайте от 1 до 16 направлений.')
        checked = []
        for direction in directions:
            if not isinstance(direction, (list, tuple)) or len(direction) != 2:
                raise DomainError('invalid_direction', 'Направление — пара [азимут, высота].')
            az, alt = [number(v, 'Направление') for v in direction]
            if not 10 <= alt <= 90:
                raise DomainError('unsupported_altitude', 'Количественный режим MVP поддерживает высоты 10–90°.')
            checked.append([az % 360 if alt != 90 else 0., alt])
        if len({tuple(v) for v in checked}) != len(checked):
            raise DomainError('invalid_direction', 'Направления должны быть различны; зенит один.')
        data['directions'] = checked
        for key in ['atmosphere', 'reference_atmosphere']:
            data.setdefault(key, {})
            try:
                atm = atmosphere_from_mapping(data[key])
            except (TypeError, ValueError) as exc:
                raise DomainError('invalid_atmosphere', 'Проверьте физические параметры атмосферы.', {'field': key}) from exc
            if atm.reference_wavelength_nm != 550 or atm.top_m <= observer['height_m']:
                raise DomainError('unsupported_atmosphere', 'MVP требует 550 нм и верх атмосферы выше наблюдателя.')
        data.setdefault('profile', 'standard')
        if data['profile'] not in ['quick', 'standard']:
            raise DomainError('invalid_profile', 'Профиль расчёта должен быть quick или standard.')
        data.setdefault('download_missing', False)
        if type(data['download_missing']) is not bool:
            raise DomainError('invalid_input', 'download_missing должен быть true или false.')
        for key, default in [('anchor_v_mag_arcsec2', 20.5), ('natural_v_mag_arcsec2', 21.6)]:
            data[key] = number(data.get(key, default), key)
            if not 10 <= data[key] <= 30:
                raise DomainError('invalid_scale', 'Яркость сценария должна быть в пределах 10–30 V mag/arcsec².')
        return cls(json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False))

    def to_dict(self) -> dict:
        return json.loads(self._json)

    @property
    def observer(self):
        return Observer(**self.to_dict()['observer'])

    @property
    def atmosphere(self):
        return atmosphere_from_mapping(self.to_dict()['atmosphere'])

    def resolved(self):
        data = self.to_dict()
        data['atmosphere_origin'] = atmosphere_origin(data['atmosphere'])
        data['atmosphere'] = asdict(self.atmosphere)
        data['reference_atmosphere'] = asdict(atmosphere_from_mapping(data['reference_atmosphere']))
        data['reference_v_mag_arcsec2'] = 22.
        data['height_datum'] = 'height above common hypothetical spherical ground z=0; geographic terrain not reconstructed'
        data['scale_origin'] = 'scenario'
        data['background_origin'] = 'explicit_scenario' if 'background' in data else 'legacy_uniform_scenario'
        data['surface_origin'] = 'explicit_scenario' if 'surface' in data else 'legacy_undecomposed_snow_free_uplight'
        if 'photometric_calibration' in data:
            data['scale_origin'] = 'provided_calibration'
        return data
