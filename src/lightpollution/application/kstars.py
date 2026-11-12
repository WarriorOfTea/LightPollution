"""Validated atomic KStars frames mapped to the existing numerical job contract."""

from copy import deepcopy
from datetime import datetime

from ..adapters.storage import content_key
from ..domain.scene import DomainError, SceneSpec, number


def scene_from_kstars(payload):
    if not isinstance(payload, dict) or set(payload) != {'scene', 'snapshot'}:
        raise DomainError('invalid_kstars_snapshot', 'Нужны scene и единый snapshot KStars.')
    snapshot = deepcopy(payload['snapshot'])
    if (not isinstance(snapshot, dict) or type(snapshot.get('schema_version')) is not int
            or snapshot['schema_version'] != 1
            or set(snapshot)-{'schema_version','kstars_version','utc','altitude_convention',
                              'location','focus','artificial_horizon_visible'}):
        raise DomainError('invalid_kstars_snapshot', 'Нужен snapshot KStars версии 1.')
    if not isinstance(snapshot.get('kstars_version'), str) or not snapshot['kstars_version']:
        raise DomainError('invalid_kstars_snapshot', 'Не указана версия KStars.')
    try:
        instant = datetime.fromisoformat(snapshot['utc'].replace('Z', '+00:00'))
        if instant.tzinfo is None or instant.utcoffset().total_seconds() != 0:
            raise ValueError('Snapshot must use UTC with zero offset.')
        location, focus = snapshot['location'], snapshot['focus']
        latitude = number(location['latitude_deg'], 'latitude_deg')
        longitude = number(location['longitude_deg'], 'longitude_deg')
        elevation = number(location['elevation_m'], 'elevation_m')
        azimuth = number(focus['azimuth_deg'], 'azimuth_deg')
        altitude = number(focus['altitude_deg'], 'altitude_deg')
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DomainError('invalid_kstars_snapshot', 'Проверьте UTC, координаты места и геометрическое направление KStars.') from exc
    if snapshot.get('altitude_convention') != 'geometric_unrefracted':
        raise DomainError('invalid_kstars_snapshot', 'Нужно геометрическое направление, без повторной рефракции.')
    if type(snapshot.get('artificial_horizon_visible')) not in [bool, type(None)]:
        raise DomainError('invalid_kstars_snapshot', 'Маска горизонта должна быть bool или null.')
    if not -500 <= elevation <= 10000:
        raise DomainError('invalid_kstars_snapshot', 'Высота места KStars вне поддерживаемого диапазона.')
    for key in ['ra_jnow_deg','dec_jnow_deg','fov_deg','catalogue_magnitude','star_b_v','star_hd']:
        if focus.get(key) is not None:
            number(focus[key], key)
    if not isinstance(payload['scene'], dict):
        raise DomainError('invalid_kstars_snapshot', 'Базовый сценарий должен быть объектом.')
    scene = deepcopy(payload['scene'])
    if not isinstance(scene.get('observer', {}), dict):
        raise DomainError('invalid_kstars_snapshot', 'observer базового сценария должен быть объектом.')
    agl = scene.get('observer', {}).get('height_m', 2.)
    scene.update(schema_version=2, mode='dated', time=snapshot['utc'],
        observer={'latitude_deg': latitude, 'longitude_deg': longitude, 'height_m': agl},
        directions=[[azimuth, altitude]])
    scene.pop('moon', None)
    if 'camera' in scene:
        camera=scene['camera']
        if not isinstance(camera,dict):raise DomainError('invalid_camera','camera must be an object.')
        camera['view']={**camera.get('view',{}),'azimuth_deg':azimuth,'altitude_deg':altitude}
        if 'catalogue' in camera and camera['catalogue'].get('utc')!=snapshot['utc']:
            raise DomainError('invalid_camera','Каталог камеры не соответствует UTC снимка KStars.')
    # KStars site elevation is NOT the eye's height above ground. Retain it
    # as independent provenance; terrain supplies model MSL where enabled.
    origin = {'provider': 'KStars', 'version': snapshot['kstars_version'],
        'snapshot_id': content_key(snapshot), 'site_elevation_m': elevation,
        'focused_object': str(focus.get('object_name', ''))[:256],
        'altitude_convention': 'geometric_unrefracted',
        'artificial_horizon_visible': snapshot.get('artificial_horizon_visible')}
    scene['integration'] = origin
    return SceneSpec.from_mapping(scene), snapshot
