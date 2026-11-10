"""Local server: HTTP handlers delegate to the same use cases as the CLI."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..adapters.sources import LocalSourceRepository
from ..application.jobs import JobManager
from ..application.planner import compare_results
from ..domain.scene import DomainError, SceneSpec

HERE = Path(__file__).resolve().parent


def create_app(workspace: Path = Path('data/mvp'), raw_directory: Path | None = None):
    workspace = Path(workspace)
    sources = LocalSourceRepository(workspace, raw_directory)
    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = JobManager(workspace, raw_directory)
        try:
            yield
        finally:
            app.state.jobs.close()
    app = FastAPI(title='LightPollution', version='0.2.0', lifespan=lifespan)
    app.mount('/static', StaticFiles(directory=HERE / 'static'), name='static')
    templates = Jinja2Templates(directory=HERE / 'templates')

    @app.exception_handler(DomainError)
    async def domain_error(request, exc):
        status = 404 if exc.code in ['unknown_job', 'unknown_export'] else 409 if exc.code in [
            'result_unavailable', 'incompatible_scale', 'queue_full'] else 422
        return JSONResponse({'error': exc.to_dict()}, status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({'error': {'code': 'invalid_request',
            'message': 'Не удалось прочитать запрос. Проверьте формат сценария.'}}, status_code=422)

    @app.get('/')
    def home(request: Request):
        return templates.TemplateResponse(request=request, name='index.html',
                                          context={'regions': sources.catalogue()})

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'application': 'lightpollution-mvp', 'request_schema': 2}

    @app.get('/api/catalogue')
    def catalogue():
        return sources.catalogue()

    @app.get('/api/jobs')
    def jobs():
        return app.state.jobs.list()

    @app.post('/api/jobs', status_code=202)
    def create(payload: dict):
        return app.state.jobs.create(SceneSpec.from_mapping(payload))

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        return app.state.jobs.get(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        return app.state.jobs.cancel(job_id)

    @app.get('/api/jobs/{job_id}/result')
    def result(job_id: str):
        return app.state.jobs.result(job_id)

    @app.get('/api/jobs/{job_id}/export/{name}')
    def export(job_id: str, name: str):
        return FileResponse(app.state.jobs.export(job_id, name), filename=name)

    @app.post('/api/compare')
    def compare(payload: dict):
        if set(payload) != {'left', 'right'}:
            raise DomainError('invalid_input', 'Выберите два завершённых расчёта.')
        return compare_results(app.state.jobs.result(payload['left']), app.state.jobs.result(payload['right']))

    return app
