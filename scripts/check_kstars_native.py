"""Run the actual patched KStars in a private D-Bus session and verify jobs.

Start the backend separately. Invoke this script with dbus-run-session.
Only isolated XDG settings are written; the user's KStars profile is untouched.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
from time import monotonic
from zoneinfo import ZoneInfo

from dbus_next.aio import MessageBus
from dbus_next import Message, MessageType
import httpx2
import numpy as np

from lightpollution.adapters.storage import write_json, file_hash
from lightpollution.models.kstars_horizon import horizon_visible


async def verify(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    work = Path('scratch/kstars-native-verification-qt'+str(args.qt_major)).resolve()
    for name in ['config','data/kstars/astrometry','cache','runtime']:
        (work/name).mkdir(parents=True,exist_ok=True)
    (work/'runtime').chmod(0o700)
    (work/'config/kstarsrc').write_text('''[Location]
CityName=LightPollution verification
CountryName=Custom
Latitude=60.3
Longitude=30.315
Elevation=20
TimeZone=0
DST=--
[GUI]
RunStartupWizard=false
WindowWidth=1440
WindowHeight=1000
[General]
KStarsFirstRun=false
[View]
UseAltAz=true
UseRefraction=false
UseAnimatedSlewing=false
IsTracking=false
ShowGround=false
ShowHorizon=true
''')
    env = dict(os.environ)
    root = args.runtime.resolve()
    lib = root/'usr/lib/x86_64-linux-gnu'
    shares=[root/'usr/share',args.data_root.resolve()/'usr/share']
    if args.installation:
        shares=[args.installation.resolve()/'catalogues/share',args.installation.resolve()/'installed/share',*shares]
    env.update(QT_QPA_PLATFORM='offscreen',LIGHTPOLLUTION_URL=args.url,LIGHTPOLLUTION_VERIFICATION='1',
        XDG_CONFIG_HOME=str(work/'config'),XDG_DATA_HOME=str(work/'data'),
        XDG_CACHE_HOME=str(work/'cache'),XDG_RUNTIME_DIR=str(work/'runtime'),
        XDG_DATA_DIRS=':'.join(map(str,shares))+':/usr/local/share:/usr/share',
        LD_LIBRARY_PATH=str(lib)+':'+str(root/'usr/lib')+':'+str(lib/'pulseaudio'),
        QT_PLUGIN_PATH=str(lib/('qt'+str(args.qt_major)+'/plugins')))
    log = (work/'kstars.log').open('w')
    process = subprocess.Popen([str(args.binary.resolve()),'--date','2026-11-11T21:00:00Z'],env=env,stdout=log,stderr=subprocess.STDOUT)
    bus = await MessageBus().connect()
    async def call(member,body=None,signature='',path='/LightPollution',interface='org.lightpollution.KStars'):
        reply=await bus.call(Message(destination='org.kde.kstars',path=path,interface=interface,
            member=member,signature=signature,body=body or []))
        if reply.message_type==MessageType.ERROR:
            raise RuntimeError(f'{member}: {reply.error_name}: {reply.body}')
        return reply.body[0] if reply.body else None
    async def official(member,body,signature):
        return await call(member,body,signature,'/KStars','org.kde.kstars')
    async def snapshot():
        return json.loads(await call('snapshotJSON'))
    async def status():
        return json.loads(await call('statusJSON'))
    async def wait_until(callback,timeout=120):
        deadline=monotonic()+timeout
        while monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError(f'KStars exited {process.returncode}; see {work}/kstars.log')
            try:
                result=await callback()
                if result:return result
            except RuntimeError as exc:
                if 'ServiceUnknown' not in str(exc) and 'UnknownObject' not in str(exc):raise
            await asyncio.sleep(.5)
        raise TimeoutError('Native state did not satisfy the verification condition.')
    try:
        await wait_until(snapshot)
        async def catalogue_ready():return (await status()).get('catalogue_ready',False)
        await wait_until(catalogue_ready)
        await call('stop',path='/KStars/SimClock',interface='org.kde.kstars.SimClock')
        latitude,longitude=(60.3,30.315) if args.eye or args.camera else (59.95,30.3)
        assert await official('setGPSLocation',[longitude,latitude,120.,0.],'dddd')
        # KStars 3.6.2 chooses the nearest city's timezone, overriding tz0
        # when a nearby location exists. Read the actual zone, never assume 0.
        location=json.loads(await official('location',[],''))
        write_json(out/'native-location-API.json',location)
        local=datetime(2026,11,11,21,0,0,tzinfo=timezone.utc)+timedelta(hours=location['tz'])
        await official('setLocalTime',[local.year,local.month,local.day,local.hour,local.minute,local.second],'iiiiii')
        await official('setAltAz',[90.,0.,False],'ddb')
        await asyncio.sleep(2)
        frame=await snapshot()
        assert abs(frame['location']['latitude_deg']-latitude)<1e-6,frame
        assert abs(frame['focus']['altitude_deg']-90)<.05,frame
        assert frame['utc']=='2026-11-11T21:00:00Z',frame
        write_json(out/'native-initial-snapshot.json',frame)
        base={'schema_version':2,'region':'petersburg-2025','period':'2025',
            'observer':{'height_m':2.},'near_field':{},
            'terrain':{'dataset':'petersburg-copdem90','visibility_step_m':100},
            'background':{'twilight':'patat2006'},'profile':'quick'}
        if args.eye:
            base.update(background={},surface={},eye={'view':{'azimuth_deg':180.,
                'altitude_deg':75.,'horizontal_fov_deg':60.,'width':1800,'height':1100}})
        if args.colour:
            if not args.eye:raise ValueError('--colour requires --eye')
            base['eye'].update(appearance={'mode':'experimental_stellar_colour'},
                observer_profile={'profile_id':'native-control-1','origin':'user_supplied'})
        if args.camera:
            base.update(background={},surface={},camera={'exposure_s':30.,'seed':20261112,
                'profile':{'sensor':{'width_px':960,'height_px':640}},
                'sampling':{'validation':'heldout_8','max_grid_size':5}})
        assert await call('setSceneJSON',[json.dumps(base)],'s')
        async with httpx2.AsyncClient(base_url=args.url,timeout=30) as client:
            async def calculate(name):
                await call('calculate')
                async def completed():
                    value=await status()
                    if value['busy']:return None
                    if not value.get('sky'):raise RuntimeError('Native calculation failed: '+value['message'])
                    return value
                result=await wait_until(completed,args.job_timeout)
                assert not result['stale'],result
                response=await client.get('/api/jobs/'+result['job_id']+'/result')
                response.raise_for_status()
                value=response.json()
                assert value['integration']['provider']=='KStars'
                assert value['request']['observer']['height_m']==2.
                assert value['request']['integration']['site_elevation_m']==120.
                assert value['terrain']['observer_msl']['height_m']!=120.
                write_json(out/(name+'-native-status.json'),result)
                write_json(out/(name+'-result.json'),value)
                return result,value
            first,first_value=await calculate('zenith')
            assert await call('savePanelScreenshot',[str(out/'kstars-native-panel.png')],'s')
            if args.eye:
                assert first_value['eye']['display']['available']
                if args.colour:
                    assert first_value['eye']['display']['colour_mode']=='experimental_stellar_colour'
                    assert first_value['eye']['config']['observer_profile']['profile_id']=='native-control-1'
                await call('showEyeView')
                async def preview_ready():
                    return (await status()).get('eye_preview_ready',False)
                await wait_until(preview_ready,30)
                assert await call('saveEyeScreenshot',[str(out/'kstars-native-eye-window.png')],'s')
            if args.camera:
                assert first_value['camera']['exports']['available'],first_value['camera']
                assert first_value['camera']['catalogue']['provider']=='KStars live StarComponent'
                await call('showCameraFITS')
                async def camera_ready():return (await status()).get('camera_FITS_ready',False)
                await wait_until(camera_ready,45)
                assert await call('saveCameraScreenshot',[str(out/'kstars-native-camera-FITS.png')],'s')
                response=await client.get('/api/jobs/'+first['job_id']+'/export/camera.fits');response.raise_for_status()
                (out/'camera.fits').write_bytes(response.content)
                write_json(out/'native-camera-catalogue.json',json.loads(await call('catalogueJSON')))
                if args.qt_major==6:
                    assert first['Qt_version'].startswith('6.') and first['Ekos_exposure_calculator_compiled']
                    await call('showEkosExposure')
                    checked=json.loads(await call('ekosExposureJSON'))
                    bridge=first_value['camera']['exports']['exposure_bridge']
                    for row in checked['rows']:
                        expected=bridge['read_noise_e']**2/(row['sky_rate_e_pixel_s']*((1+.05)**2-1))
                        assert abs(row['exposure_s']/expected-1)<1e-12
                        assert abs(row['sky_e']/(row['sky_rate_e_pixel_s']*row['exposure_s'])-1)<1e-12
                    assert checked['profile_id']==first_value['camera']['profile_id']
                    write_json(out/'native-ekos-verification.json',checked)
                    assert await call('saveEkosScreenshot',[str(out/'kstars-native-ekos.png')],'s')
            await official('setAltAz',[45.,180.,False],'ddb')
            await asyncio.sleep(2)
            assert (await status())['stale']
            second,second_value=await calculate('south45')
            assert abs(second_value['request']['directions'][0][0]-180)<.05
            assert abs(second_value['request']['directions'][0][1]-45)<.05
            assert first_value['integration']['snapshot_id']!=second_value['integration']['snapshot_id']
            assert await call('savePanelScreenshot',[str(out/'kstars-native-south45.png')],'s')
            assert await call('saveWindowScreenshot',[str(out/'kstars-native-window.png')],'s')
            horizon_checks=None
            if args.qt_major==6:
                if args.camera:
                    # Revisit after another aperture / draw cycle: recycled blocks
                    # must not alter previously returned identities or photometry.
                    await official('setAltAz',[90.,0.,False],'ddb');await asyncio.sleep(2)
                    again=json.loads(await call('catalogueJSON'))
                    write_json(out/'native-camera-catalogue-after-pan.json',again)
                    old={r['id']:r for r in first_value['camera']['config']['catalogue']['rows']}
                    new={r['id']:r for r in again['rows']}
                    assert old.keys()==new.keys()
                    for key,row in old.items():
                        assert row['v_mag']==new[key]['v_mag'] and row['b_v']==new[key]['b_v']
                        assert row['astrometry']==new[key]['astrometry']
                    write_json(out/'native-catalogue-repeat-verification.json',{'rows':len(old),
                        'identical_ID_V_B_V_astrometry':True,'after_another_aperture_and_draw_cycle':True})
                    await official('setAltAz',[45.,180.,False],'ddb');await asyncio.sleep(2)
                # Native contour semantics vs independent NumPy ray evaluation.
                regions=[{'name':'wall','ceiling':False,'points':[[160.,45.2],[200.,45.2]]},
                    {'name':'north','ceiling':False,'points':[[350.,20.],[10.,30.]]},
                    {'name':'canopy','ceiling':True,'points':[[165.,60.],[195.,60.]]},
                    {'name':'upper-window','ceiling':False,'points':[[165.,70.],[195.,70.]]}]
                assert await call('setVerificationHorizonJSON',[json.dumps(regions)],'s')
                horizon_frame=await snapshot();write_json(out/'native-horizon-snapshot.json',horizon_frame)
                rng=np.random.default_rng(31415)
                points=np.column_stack([rng.uniform(0,360,2048),rng.uniform(-20,90,2048)]).tolist()
                points.extend([[0.,25.],[180.,45.2],[180.,60.],[180.,70.],[350.,20.],[10.,30.]])
                actual=np.array(json.loads(await call('horizonVisibilityJSON',[json.dumps(points)],'s')))
                expected=horizon_visible(horizon_frame['artificial_horizon'],*np.asarray(points).T)
                write_json(out/'native-horizon-ray-comparison.json',{'points':points,
                    'native_visible':actual.tolist(),'model_visible':expected.tolist()})
                assert np.array_equal(actual,expected)
                horizon_checks={'native_rays':len(points),'matches':int(np.count_nonzero(actual==expected)),
                    'regions':regions,'mask_resampled_to_azimuth_grid':False}
                write_json(out/'native-horizon-verification.json',horizon_checks)
                if args.camera:
                    third,third_value=await calculate('horizon')
                    assert third_value['camera']['exports']['available']
                    pixels=third_value['camera']['exports']
                    assert 0<pixels['supported_pixels']<pixels['field_pixels']
                    write_json(out/'native-horizon-camera-exports.json',pixels)
                    response=await client.get('/api/jobs/'+third['job_id']+'/export/camera-preview.png');response.raise_for_status()
                    (out/'horizon-camera-preview.png').write_bytes(response.content)
                    await call('showCameraFITS')
                    await wait_until(camera_ready,45)
                    assert await call('saveCameraScreenshot',[str(out/'kstars-native-horizon-FITS.png')],'s')
                    response=await client.get('/api/jobs/'+third['job_id']+'/export/camera.fits');response.raise_for_status()
                    (out/'horizon-camera.fits').write_bytes(response.content)
            expected_rejection=None
            if args.qt_major==6 and args.camera:
                await official('setAltAz',[45.,180.,False],'ddb');await asyncio.sleep(2)
                invalid=deepcopy(base);invalid['camera']['sampling']['max_grid_size']=4
                assert await call('setSceneJSON',[json.dumps(invalid)],'s')
                await call('calculate')
                async def rejected():
                    s=await status()
                    return s if not s['busy'] and not s['job_id'] else None
                rejected_status=await wait_until(rejected,45)
                assert rejected_status['stale'] and rejected_status['message']
                expected_rejection={'invalid_sampling_rejected':True,'old_result_remains_stale':True,
                    'message':rejected_status['message']}
                write_json(out/'native-expected-rejection.json',expected_rejection)
            await official('lookTowards',['Vega'],'s')
            await asyncio.sleep(2)
            star_frame=await snapshot()
            assert star_frame['focus'].get('star_hd')==172167,star_frame
            assert star_frame['focus'].get('star_spectral_type'),star_frame
            assert 'star_b_v' in star_frame['focus'],star_frame
            write_json(out/'native-star-catalogue-snapshot.json',star_frame)
            hd_checks=[]
            if args.installation:
                manifest=json.loads((args.installation/'installation.json').read_text())
                for sample in manifest['HD_index']['lookup_samples']:
                    await official('lookTowards',['HD '+str(sample['hd'])],'s');await asyncio.sleep(1)
                    looked=await snapshot()
                    assert looked['focus'].get('star_hd')==sample['hd'],looked
                    assert abs(looked['focus']['catalogue_magnitude']-sample['v_mag'])<1e-4
                    hd_checks.append({'expected':sample,'actual':looked['focus']})
                write_json(out/'native-HD-index-verification.json',hd_checks)
            write_json(out/'native-verification.json',{
                'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
                'kstars_version':frame['kstars_version'],'compiled_binary_sha256':file_hash(args.binary),
                'jobs_completed':3 if args.qt_major==6 and args.camera else 2,
                'jobs_requested':3 if args.qt_major==6 and args.camera else 2,'geometry_round_trip_checked':True,
                'stale_after_map_change_checked':True,'AGL_vs_MSL_checked':True,
                'native_star_B_V_spectral_type_HD_checked':True,
                'native_eye_preview_opened':bool(args.eye),
                'native_experimental_colour_checked':bool(args.colour),
                'native_camera_FITS_Viewer_opened':bool(args.camera),
                'native_live_camera_catalogue_rows':first_value['camera']['catalogue']['rows_received'] if args.camera else None,
                'native_camera_stars_in_frame':first_value['camera']['exports']['stars_in_frame'] if args.camera else None,
                'native_camera_catalogue_stars_fainter_than_6_5':sum(r['v_mag']>6.5 for r in first_value['camera']['config']['catalogue']['rows']) if args.camera else None,
                'Ekos_exposure_calculator_compiled':first.get('Ekos_exposure_calculator_compiled',False),
                'Qt_version':first.get('Qt_version'),'full_horizon':horizon_checks,
                'deep_HD_index_lookups':len(hd_checks),
                'expected_invalid_submission':expected_rejection,
                'screenshots_from_native_Qt_widget':True,'platform':'Qt offscreen / private session D-Bus',
                'build_excludes_INDI_Ekos_device_control':not first.get('Ekos_exposure_calculator_compiled',False),
                'original_user_settings_modified':False,'job_ids':[first['job_id'],second['job_id']]+
                    ([third['job_id']] if args.qt_major==6 and args.camera else []),
                'nearest_city_timezone_read_from_actual_KStars_location_API':location,
                'field_accuracy_validated':False})
            print('Native KStars: jobs, geometry, UTC, stale state, AGL/MSL and screenshots verified.',flush=True)
    finally:
        bus.disconnect()
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--installation',type=Path,help='Use the private setup.py installation manifest.')
    parser.add_argument('--qt-major',type=int,choices=[5,6],default=5)
    parser.add_argument('--data-root',type=Path,default=Path('scratch/kstars-runtime'))
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--eye',action='store_true',help='Verify the bounded eye model and native PNG viewer.')
    parser.add_argument('--colour',action='store_true',help='Check experimental stellar colour and observer-profile round trip.')
    parser.add_argument('--camera',action='store_true',help='Check live KStars star catalogue and linear FITS in the existing FITS Viewer.')
    parser.add_argument('--job-timeout',type=float,default=900.,help='Per-job timeout; real spectral camera rays may take several minutes.')
    parser.add_argument('--output',type=Path,default=Path('artifacts/kstars-terrain-near-20261111'))
    args=parser.parse_args()
    if args.installation:
        record=json.loads((args.installation/'installation.json').read_text())
        args.binary=Path(record['kstars_binary']);args.runtime=Path(record['dependency_root'] or '/')
        args.data_root=args.installation;args.qt_major=6
    if not args.binary or not args.runtime:parser.error('Use --installation or both --binary and --runtime.')
    asyncio.run(verify(args))


if __name__=='__main__':main()
