"""Start a private native KStars and its local numerical backend together."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import select
import shutil
from urllib.error import URLError
from urllib.request import urlopen

from setup import native_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installation',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,default=Path('data/eye-model-20261111'))
    parser.add_argument('--raw-directory',type=Path,default=Path('data/blackmarble'))
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--offscreen',action='store_true',help='For a private test display only.')
    args=parser.parse_args();prefix=args.installation.resolve()
    record=json.loads((prefix/'installation.json').read_text())
    if record.get('schema_version')!=1:raise SystemExit('Unsupported installation manifest.')
    if not 1024<=args.port<=65535:parser.error('Use a port in1024..65535.')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',args.port))
    url='http://127.0.0.1:'+str(args.port)
    backend_env=dict(os.environ);project=Path(record['project'])
    backend=[record['python'],'-c','from lightpollution.cli import main; main()',
        'serve','--host','127.0.0.1','--port',str(args.port),'--workspace',str(args.workspace.resolve()),
        '--raw-directory',str(args.raw_directory.resolve())]
    logdir=prefix/'logs';logdir.mkdir(exist_ok=True)
    process=stars=bus=None
    try:
        daemon=shutil.which('dbus-daemon')
        if not daemon:raise RuntimeError('Install dbus-daemon for the private KStars session.')
        with (logdir/'dbus.log').open('a') as buslog:
            bus=subprocess.Popen([daemon,'--session','--nofork','--print-address=1'],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=buslog)
        if not select.select([bus.stdout],[],[],5)[0]:raise TimeoutError('Private D-Bus did not start within5s.')
        address=bus.stdout.readline().decode().strip()
        if not address or bus.poll() is not None:raise RuntimeError('Private D-Bus exited; see logs/dbus.log.')
        backend_env['DBUS_SESSION_BUS_ADDRESS']=address
        session=prefix/'session.json'
        session.write_text(json.dumps({'schema_version':1,'dbus_address':address,'url':url,
            'launcher_pid':os.getpid()},indent=2)+'\n')
        with (logdir/'backend.log').open('a') as log:
            process=subprocess.Popen(backend,cwd=project,env=backend_env,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Backend exited; see '+str(logdir/'backend.log'))
                try:
                    with urlopen(url+'/api/kstars/capabilities',timeout=1) as response:
                        if json.load(response).get('snapshot_schema')!=2:raise RuntimeError('Incompatible backend.')
                    break
                except (URLError,TimeoutError):time.sleep(.2)
            else:raise TimeoutError('Backend did not start within45s.')
            env=native_environment(record.get('dependency_root'));env['LIGHTPOLLUTION_URL']=url
            env['DBUS_SESSION_BUS_ADDRESS']=address
            for key,name in [('XDG_DATA_HOME','user-data'),('XDG_CONFIG_HOME','user-config'),('XDG_CACHE_HOME','user-cache')]:
                path=prefix/name;path.mkdir(exist_ok=True);env[key]=str(path)
            share=[prefix/'catalogues/share',prefix/'installed/share']
            if record.get('dependency_root'):share.append(Path(record['dependency_root'])/'usr/share')
            env['XDG_DATA_DIRS']=':'.join(map(str,share))+':'+env.get('XDG_DATA_DIRS','/usr/local/share:/usr/share')
            if record.get('dependency_root'):
                env['QT_PLUGIN_PATH']=str(Path(record['dependency_root'])/'usr/lib/x86_64-linux-gnu/qt6/plugins')
                env['QML_IMPORT_PATH']=str(Path(record['dependency_root'])/'usr/lib/x86_64-linux-gnu/qt6/qml')
            if args.offscreen:env['QT_QPA_PLATFORM']='offscreen'
            with (logdir/'kstars.log').open('a') as native_log:
                stars=subprocess.Popen([record['kstars_binary']],env=env,stdout=native_log,stderr=subprocess.STDOUT)
                while stars.poll() is None:
                    if process.poll() is not None:raise RuntimeError('Backend stopped; closing this KStars instance.')
                    time.sleep(.5)
                return stars.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        for child in [stars,process,bus]:
            if child and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        if (prefix/'session.json').exists():
            current=json.loads((prefix/'session.json').read_text())
            if current.get('launcher_pid')==os.getpid():(prefix/'session.json').unlink()


if __name__=='__main__':sys.exit(main())
