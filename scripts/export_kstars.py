"""Export a real KStars star raster, optionally using an extracted Linux runtime.

No installation, user configuration changes or shell evaluation are performed.
KStars' --dump interprets the small .kstars script internally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,help='Optional root of locally extracted Ubuntu KStars packages')
    parser.add_argument('--output',type=Path,default=Path('artifacts/petersburg-2025/kstars'))
    parser.add_argument('--latitude',type=float,default=60.3)
    parser.add_argument('--longitude',type=float,default=30.315)
    parser.add_argument('--height',type=float,default=20)
    parser.add_argument('--utc',default='2026-11-09T21:00:00Z')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    work=Path('scratch/kstars-export').resolve();work.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(QT_QPA_PLATFORM='offscreen',XDG_CONFIG_HOME=str(work/'config'),
               XDG_DATA_HOME=str(work/'data'),XDG_CACHE_HOME=str(work/'cache'))
    for directory in ('config','data','cache'):(work/directory).mkdir(exist_ok=True)
    # KStars 3.6 --dump does not create this directory before opening SQLite.
    (work/'data/kstars').mkdir(exist_ok=True)
    (work/'data/kstars/astrometry').mkdir(exist_ok=True)
    if args.runtime:
        runtime=args.runtime.resolve()
        executable=runtime/'usr/bin/kstars'
        lib=runtime/'usr/lib/x86_64-linux-gnu'
        env.update(LD_LIBRARY_PATH=str(lib)+':'+str(lib/'pulseaudio'),
                   QT_PLUGIN_PATH=str(lib/'qt5/plugins'),
                   XDG_DATA_DIRS=str(runtime/'usr/share')+':/usr/local/share:/usr/share')
    else:
        executable=shutil.which('kstars')
        if not executable:raise SystemExit('KStars executable is not installed; the independent render command still works.')
    size=1600;zoom=(size*.46)/(3.141592653589793/2)
    config=f'''[Location]
CityName=Observer
ProvinceName=
CountryName=Custom
Latitude={args.latitude}
Longitude={args.longitude}
Elevation={args.height}
TimeZone=0
DST=--

[GUI]
ShowInfoBoxes=false
ShowStatusBar=false

[View]
UseAltAz=true
UseRefraction=false
UseAnimatedSlewing=false
UseAutoLabel=false
IsTracking=false
Projection=1
SkyRotation=0
ZoomFactor={zoom}
ShowStars=true
StarDensity=5
ShowStarNames=false
ShowStarMagnitudes=false
ShowDeepSky=false
ShowSolarSystem=false
ShowMilkyWay=false
ShowCLines=false
ShowCNames=false
ShowCBounds=false
ShowEquatorialGrid=false
ShowHorizontalGrid=false
ShowHorizon=false
ShowGround=false
ShowFlags=false
ShowEcliptic=false
ShowEquator=false
ShowLocalMeridian=false

[Colors]
ColorSchemeFile=moonless-night.colors
StarColorMode=3
StarColorIntensity=0
'''
    (work/'config/kstarsrc').write_text(config)
    (output/'kstars-settings.ini').write_text(config)
    script=output/'observer.kstars'
    script.write_text('dbus-send --dest=org.kde.kstars /KStars org.kde.kstars.setAltAz double:90 double:180\n')
    version=subprocess.run([str(executable),'--version'],env=env,capture_output=True,text=True,timeout=30)
    command=[str(executable),'--dump',str(output/'kstars-baseline.png'),'--width',str(size),
             '--height',str(size),'--date',args.utc,'--script',str(script)]
    with (output/'kstars-run.log').open('w') as log:
        run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=180)
    if run.returncode or not (output/'kstars-baseline.png').exists():
        raise SystemExit(f'KStars export failed ({run.returncode}); see {output / "kstars-run.log"}')
    metadata={'version':version.stdout.strip(),'utc':args.utc,
              'observer':{'latitude_deg':args.latitude,'longitude_deg':args.longitude,'height_m':args.height},
              'projection':'KStars AzimuthalEquidistant, zenith focus, azimuth=180, rotation=0',
              'size':size,'zoom_pixels_per_radian':zoom,'refraction':False,
              'is_patched_kstars':False,'purpose':'Actual KStars catalogue raster, before skyglow compositing',
              'source':'https://invent.kde.org/education/kstars',
              'documentation':'docs/OBSERVER_VIEW.md'}
    (output/'kstars-summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
