"""Versioned application request. Legacy observation schema 1 remains separate."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json

import numpy as np

from ..atmosphere_config import atmosphere_from_mapping, atmosphere_origin
from ..physics import Atmosphere, Observer


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
                   'anchor_v_mag_arcsec2', 'natural_v_mag_arcsec2', 'moon', 'download_missing'}
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
        return data
