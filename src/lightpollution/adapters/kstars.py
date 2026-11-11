"""Optional official-session D-Bus client for the native KStars dock contract."""

import json

from ..domain.scene import DomainError


async def native_call(method, signature='', body=None):
    try:
        from dbus_next.aio import MessageBus
        from dbus_next import Message, MessageType
    except ImportError as exc:
        raise DomainError('missing_dependency', 'Для D-Bus нужен uv sync --extra kstars.') from exc
    bus = None
    try:
        bus = await MessageBus().connect()
        response = await bus.call(Message(destination='org.kde.kstars', path='/LightPollution',
            interface='org.lightpollution.KStars', member=method, signature=signature, body=body or []))
        if response.message_type == MessageType.ERROR:
            raise DomainError('kstars_native_unavailable', 'Нет нативной панели LightPollution в текущем KStars; проверьте сборку и D-Bus-сеанс.')
        return response.body[0] if response.body else None
    except DomainError:
        raise
    except (OSError, ValueError) as exc:
        raise DomainError('kstars_connection_failed', 'Не удалось подключиться к D-Bus рабочего стола KStars.') from exc
    finally:
        if bus:
            bus.disconnect()


async def read_native_snapshot():
    value = await native_call('snapshotJSON')
    try:
        return json.loads(value)
    except (ValueError, TypeError) as exc:
        raise DomainError('invalid_kstars_snapshot', 'KStars вернул некорректный JSON.') from exc
