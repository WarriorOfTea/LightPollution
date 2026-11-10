"""Single-process numerical worker, durable states and cooperative cancellation."""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import multiprocessing
import os
from pathlib import Path
import re
import threading
import traceback
import uuid
import zipfile

from ..adapters.storage import atomic_target, read_json, write_json
from ..domain.scene import Cancelled, DomainError, SceneSpec
from .planner import Planner, export_result

TERMINAL = {'completed', 'cancelled', 'error'}
EXPORTS = ('scene.json', 'resolved-scene.json', 'observation.json', 'result.json',
           'directions.csv', 'stars.csv', 'bundle.zip')


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _status(folder: Path, **changes):
    value = read_json(folder / 'status.json')
    value.update(changes)
    value['updated_at'] = timestamp()
    write_json(folder / 'status.json', value)
    return value


def _worker(workspace: str, raw_directory: str | None, job_id: str, owner_pid: int):
    workspace = Path(workspace)
    folder = workspace / 'jobs' / job_id
    def cancelled():
        return (folder / 'cancel').exists() or os.getppid() != owner_pid
    try:
        if cancelled():
            raise Cancelled()
        _status(folder, state='running', started_at=timestamp())
        spec = SceneSpec.from_mapping(read_json(folder / 'request.json'))
        planner = Planner(workspace, raw_directory=Path(raw_directory) if raw_directory else None)
        result = planner.run(spec, progress=lambda value: _status(folder, progress=value), cancelled=cancelled)
        if cancelled():
            raise Cancelled()
        output = folder / 'output'
        export_result(result, output)
        with atomic_target(output / 'bundle.zip') as temporary:
            with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name in EXPORTS:
                    if name != 'bundle.zip':
                        archive.write(output / name, arcname=name)
        if cancelled():
            raise Cancelled()
        _status(folder, state='completed', finished_at=timestamp(), error=None,
                elapsed_seconds=result.value['runtime']['elapsed_seconds'])
    except Cancelled:
        _status(folder, state='cancelled', finished_at=timestamp(),
                progress={'phase': 'Отменено; завершённые лучи сохранены для повтора'})
    except DomainError as exc:
        _status(folder, state='error', error=exc.to_dict(), finished_at=timestamp())
    except Exception:
        (folder / 'worker-error.log').write_text(traceback.format_exc())
        _status(folder, state='error', finished_at=timestamp(), error={
            'code': 'calculation_error', 'message': 'Расчёт не завершён. Подробности сохранены в локальном журнале.'})
    return job_id


class JobManager:
    def __init__(self, workspace: Path, raw_directory: Path | None = None):
        self.workspace = Path(workspace)
        self.raw_directory = Path(raw_directory) if raw_directory else None
        self.folder = self.workspace / 'jobs'
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.workspace / 'worker.lock').open('a+')
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise DomainError('workspace_in_use', 'Эта рабочая папка уже обслуживается другим процессом.') from None
        self._mutex = threading.RLock()
        self._closed = False
        self._futures = {}
        for path in self.folder.glob('*/status.json'):
            value = read_json(path)
            if value['state'] not in TERMINAL:
                (path.parent / 'cancel').touch()
                _status(path.parent, state='error', finished_at=timestamp(), error={
                    'code': 'interrupted', 'message': 'Предыдущий запуск прерван. Повтор использует завершённые блоки кэша.'})
        self._executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise DomainError('unknown_job', 'Задание не найдено.')
        path = self.folder / job_id
        if not (path / 'status.json').exists():
            raise DomainError('unknown_job', 'Задание не найдено.')
        return path

    def get(self, job_id):
        return read_json(self._path(job_id) / 'status.json')

    def list(self):
        values = [read_json(path) for path in self.folder.glob('*/status.json')]
        return sorted(values, key=lambda value: value['created_at'], reverse=True)

    def create(self, spec: SceneSpec):
        Planner(self.workspace, raw_directory=self.raw_directory).preflight(spec)
        with self._mutex:
            if self._closed:
                raise DomainError('closed', 'Сервис расчёта остановлен.')
            if sum(v['state'] not in TERMINAL for v in self.list()) >= 8:
                raise DomainError('queue_full', 'В очереди уже восемь расчётов; дождитесь завершения.')
            job_id = uuid.uuid4().hex
            folder = self.folder / job_id
            value = {'id': job_id, 'state': 'queued', 'created_at': timestamp(),
                'updated_at': timestamp(), 'request': spec.to_dict(),
                'progress': {'phase': 'В очереди', 'completed': 0, 'total': None}, 'error': None}
            write_json(folder / 'request.json', spec.to_dict())
            write_json(folder / 'status.json', value)
            future = self._executor.submit(_worker, str(self.workspace.resolve()),
                str(self.raw_directory.resolve()) if self.raw_directory else None, job_id, os.getpid())
            self._futures[job_id] = future
            def finished(done):
                if done.cancelled():
                    return
                if done.exception() is not None and self.get(job_id)['state'] not in TERMINAL:
                    _status(folder, state='error', error={'code': 'worker_stopped',
                        'message': 'Вычислительный процесс остановился. Расчёт можно повторить.'})
            future.add_done_callback(finished)
            return value

    def cancel(self, job_id):
        with self._mutex:
            folder = self._path(job_id)
            value = self.get(job_id)
            if value['state'] in TERMINAL:
                return value
            (folder / 'cancel').touch()
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                return _status(folder, state='cancelled', finished_at=timestamp())
            return self.get(job_id)

    def result(self, job_id):
        if self.get(job_id)['state'] != 'completed':
            raise DomainError('result_unavailable', 'Расчёт ещё не завершён успешно.')
        return read_json(self._path(job_id) / 'output/result.json')

    def export(self, job_id, name):
        if name not in EXPORTS:
            raise DomainError('unknown_export', 'Этот файл не входит в результат.')
        if self.get(job_id)['state'] != 'completed':
            raise DomainError('result_unavailable', 'Экспорт доступен после успешного завершения.')
        return self._path(job_id) / 'output' / name

    def close(self):
        with self._mutex:
            self._closed = True
            for value in self.list():
                if value['state'] not in TERMINAL:
                    self.cancel(value['id'])
        self._executor.shutdown(wait=True, cancel_futures=True)
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        self._lock_file.close()
