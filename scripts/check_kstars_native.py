"""Run the actual patched KStars in a private D-Bus session and verify jobs.

Start the backend separately. Invoke this script with dbus-run-session.
Only isolated XDG settings are written; the user's KStars profile is untouched.
"""

import argparse
import asyncio
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

from lightpollution.adapters.storage import write_json, file_hash


async def verify(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    work = Path('scratch/kstars-native-verification').resolve()
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
ShowGround=true
ShowHorizon=true
''')
    env = dict(os.environ)
    root = args.runtime.resolve()
    lib = root/'usr/lib/x86_64-linux-gnu'
    env.update(QT_QPA_PLATFORM='offscreen',LIGHTPOLLUTION_URL=args.url,
        XDG_CONFIG_HOME=str(work/'config'),XDG_DATA_HOME=str(work/'data'),
        XDG_CACHE_HOME=str(work/'cache'),XDG_RUNTIME_DIR=str(work/'runtime'),
        XDG_DATA_DIRS=str(root/'usr/share')+':'+str(args.data_root.resolve()/'usr/share')+':/usr/local/share:/usr/share',
        LD_LIBRARY_PATH=str(lib)+':'+str(lib/'pulseaudio'),
        QT_PLUGIN_PATH=str(lib/'qt5/plugins'))
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
        await call('stop',path='/KStars/SimClock',interface='org.kde.kstars.SimClock')
        latitude,longitude=(60.3,30.315) if args.eye else (59.95,30.3)
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
        assert await call('setSceneJSON',[json.dumps(base)],'s')
        async with httpx2.AsyncClient(base_url=args.url,timeout=30) as client:
            async def calculate(name):
                await call('calculate')
                async def completed():
                    value=await status()
                    if value['busy']:return None
                    if not value.get('sky'):raise RuntimeError('Native calculation failed: '+value['message'])
                    return value
                result=await wait_until(completed,300)
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
            await official('setAltAz',[45.,180.,False],'ddb')
            await asyncio.sleep(2)
            assert (await status())['stale']
            second,second_value=await calculate('south45')
            assert abs(second_value['request']['directions'][0][0]-180)<.05
            assert abs(second_value['request']['directions'][0][1]-45)<.05
            assert first_value['integration']['snapshot_id']!=second_value['integration']['snapshot_id']
            assert await call('savePanelScreenshot',[str(out/'kstars-native-south45.png')],'s')
            assert await call('saveWindowScreenshot',[str(out/'kstars-native-window.png')],'s')
            await official('lookTowards',['Vega'],'s')
            await asyncio.sleep(2)
            star_frame=await snapshot()
            assert star_frame['focus'].get('star_hd')==172167,star_frame
            assert star_frame['focus'].get('star_spectral_type'),star_frame
            assert 'star_b_v' in star_frame['focus'],star_frame
            write_json(out/'native-star-catalogue-snapshot.json',star_frame)
            write_json(out/'native-verification.json',{
                'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
                'kstars_version':frame['kstars_version'],'compiled_binary_sha256':file_hash(args.binary),
                'jobs_completed':2,'jobs_requested':2,'geometry_round_trip_checked':True,
                'stale_after_map_change_checked':True,'AGL_vs_MSL_checked':True,
                'native_star_B_V_spectral_type_HD_checked':True,
                'native_eye_preview_opened':bool(args.eye),
                'native_experimental_colour_checked':bool(args.colour),
                'screenshots_from_native_Qt_widget':True,'platform':'Qt offscreen / private session D-Bus',
                'build_excludes_INDI_Ekos_device_control':True,
                'original_user_settings_modified':False,'job_ids':[first['job_id'],second['job_id']],
                'nearest_city_timezone_read_from_actual_KStars_location_API':location,
                'field_accuracy_validated':False})
            print('Native KStars: 2/2 jobs, geometry, UTC, stale state, AGL/MSL and screenshots verified.',flush=True)
    finally:
        bus.disconnect()
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('scratch/kstars-runtime'))
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--eye',action='store_true',help='Verify the bounded eye model and native PNG viewer.')
    parser.add_argument('--colour',action='store_true',help='Check experimental stellar colour and observer-profile round trip.')
    parser.add_argument('--output',type=Path,default=Path('artifacts/kstars-terrain-near-20261111'))
    asyncio.run(verify(parser.parse_args()))


if __name__=='__main__':main()
