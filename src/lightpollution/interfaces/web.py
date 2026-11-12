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
    from .. import __version__
    app = FastAPI(title='LightPollution', version=__version__, lifespan=lifespan)
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

    @app.get('/api/spectral-library')
    def spectral_catalogue():
        from ..adapters.spectral_library import load_spectral_library
        return load_spectral_library().catalogue()

    @app.get('/api/kstars/capabilities')
    def kstars_capabilities():
        from ..adapters.terrain import DEFINITIONS
        return {'schema_version': 1, 'backend_version': __version__, 'snapshot_schema': 2,
                'snapshot_schemas_supported':[1,2], 'full_field_artificial_horizon':True,
                'terrain_datasets': [{'id': key, 'prepared': (workspace.parent/'terrain'/key/'manifest.json').exists()}
                                     for key in DEFINITIONS],
                'native_photometric_display_available': False,'camera_linear_FITS_available':True,
                'camera_catalogue_provider':'optional KStars StarComponent snapshot','camera_hardware_control':False}

    @app.post('/api/kstars/jobs', status_code=202)
    def create_kstars(payload: dict):
        from ..application.kstars import scene_from_kstars
        from ..adapters.storage import write_json
        spec, snapshot = scene_from_kstars(payload)
        job = app.state.jobs.create(spec)
        write_json(workspace/'kstars-frames'/(job['id']+'.json'), snapshot)
        return job

    @app.post('/api/photometry/fit')
    def photometry_fit(payload: dict):
        from ..application.photometry import fit_dataset
        if not isinstance(payload.get('cases'), dict):
            raise DomainError('invalid_photometry', 'Укажите cases: имена завершённых заданий.')
        return fit_dataset(payload, {name: app.state.jobs.result(job_id)
                                    for name, job_id in payload['cases'].items()})

    @app.post('/api/eye-profile/export')
    def export_eye_profile(payload: dict):
        from ..models.eye_profiles import profile_document
        try:
            if set(payload)!={'eye'}:raise ValueError('Укажите только eye.')
            return profile_document(payload['eye'])
        except (ValueError,TypeError,KeyError) as exc:
            raise DomainError('invalid_eye_profile',str(exc)) from exc

    @app.post('/api/eye-profile/merge')
    def merge_eye_profile(payload: dict):
        from ..models.eye_profiles import apply_profile
        try:
            if set(payload)!={'eye','profile'}:raise ValueError('Укажите eye и profile.')
            return apply_profile(payload['profile'],payload['eye'])
        except (ValueError,TypeError,KeyError) as exc:
            raise DomainError('invalid_eye_profile',str(exc)) from exc

    @app.post('/api/eye-profile/apply')
    def apply_eye_profile(payload: dict):
        from ..models.eye_profiles import apply_profile
        try:
            if set(payload)!={'scene','profile'}:raise ValueError('Укажите scene и profile.')
            scene=SceneSpec.from_mapping(payload['scene']).to_dict()
            scene['eye']=apply_profile(payload['profile'],scene.get('eye') or {})
            return SceneSpec.from_mapping(scene).to_dict()
        except (ValueError,TypeError,KeyError) as exc:
            raise DomainError('invalid_eye_profile',str(exc)) from exc

    @app.post('/api/camera-profile/export')
    def export_camera_profile(payload: dict):
        from ..models.camera import camera_config
        try:
            if set(payload)!={'camera'}:raise ValueError('Specify camera.')
            return {'schema_version':1,'kind':'camera_profile','profile':camera_config(payload['camera'])['profile']}
        except (ValueError,TypeError,KeyError) as exc:raise DomainError('invalid_camera_profile',str(exc)) from exc

    @app.post('/api/camera-profile/merge')
    def merge_camera_profile(payload: dict):
        from ..models.camera import camera_config
        try:
            if set(payload)!={'camera','profile'}:raise ValueError('Specify camera and profile.')
            document=payload['profile']
            if (not isinstance(document,dict) or set(document)!={'schema_version','kind','profile'}
                    or type(document['schema_version']) is not int or document['schema_version']!=1
                    or document['kind']!='camera_profile'):raise ValueError('Expected camera_profile schema1.')
            return camera_config({**payload['camera'],'profile':document['profile']})
        except (ValueError,TypeError,KeyError) as exc:raise DomainError('invalid_camera_profile',str(exc)) from exc

    @app.post('/api/camera-profile/ekos')
    def ekos_camera_profile(payload: dict):
        from ..adapters.ekos_camera import import_ekos_camera
        try:
            if set(payload)-{'xml','base_profile','read_mode','gain'} or 'xml' not in payload:raise ValueError('Specify XML and optional base_profile/read_mode/gain.')
            return {'schema_version':1,'kind':'camera_profile','profile':import_ekos_camera(**payload)}
        except (ValueError,TypeError,KeyError) as exc:raise DomainError('invalid_camera_profile',str(exc)) from exc

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
