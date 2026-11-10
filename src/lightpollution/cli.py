"""Small reproducible command-line workflow; credentials never enter output files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import requests

from .data import Bounds, audit, load_region, read_region, save_region
from .atmosphere_config import atmosphere_origin
from .download import catalogue, download_composite, write_manifest
from .periods import CompositePeriod
from .physics import Atmosphere, Observer
from .visualization import source_map

from .application.simulation import run_simulation

DEFAULT_ALTITUDES = [1,2,3,5,7.5,10,15,20,30,45,60,75,90]


def make_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Black Marble audit and experimental directional skyglow")
    commands=parser.add_subparsers(dest='command',required=True)
    serve=commands.add_parser('serve',help='Start the local numerical planner')
    serve.add_argument('--workspace',type=Path,default=Path(os.environ.get('LIGHTPOLLUTION_WORKSPACE','data/mvp')))
    serve.add_argument('--raw-directory',type=Path)
    serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--port',type=int,default=8765)
    plan=commands.add_parser('plan',help='Calculate a complete schema-2 scenario in one command')
    plan.add_argument('--scene',type=Path)
    plan.add_argument('--workspace',type=Path,default=Path(os.environ.get('LIGHTPOLLUTION_WORKSPACE','data/mvp')))
    plan.add_argument('--raw-directory',type=Path)
    plan.add_argument('--output',type=Path,default=Path('data/mvp/manual-result'))
    example=commands.add_parser('example',help='Save a small offline schema-2 example')
    example.add_argument('--output',type=Path,required=True)
    datasets=commands.add_parser('datasets',help='Inspect prepared region definitions or download pinned files')
    datasets.add_argument('--workspace',type=Path,default=Path('data/mvp'))
    datasets.add_argument('--raw-directory',type=Path)
    datasets.add_argument('--download',metavar='REGION')
    compare=commands.add_parser('compare',help='Compare two exported result.json files with a common scale')
    compare.add_argument('--left',type=Path,required=True)
    compare.add_argument('--right',type=Path,required=True)
    compare.add_argument('--output',type=Path)
    cat=commands.add_parser('catalogue',help='Inspect public NASA file metadata; no authentication needed')
    cat.add_argument('--year',type=int,default=2025)
    cat.add_argument('--month',type=int,choices=range(1,13),help='Monthly VNP46A3; omit for annual VNP46A4')
    cat.add_argument('--output',type=Path)
    fetch=commands.add_parser('download',help='Download four pinned Petersburg HDF5 tiles')
    fetch.add_argument('--year',type=int,default=2025)
    fetch.add_argument('--month',type=int,choices=range(1,13))
    fetch.add_argument('--directory',type=Path)
    fetch.add_argument('--manifest',type=Path)
    check=commands.add_parser('audit',help='Check and extract the 29–32 E, 59–61 N window')
    check.add_argument('--year',type=int,default=2025)
    check.add_argument('--month',type=int,choices=range(1,13))
    check.add_argument('--directory',type=Path)
    check.add_argument('--output',type=Path)
    check.add_argument('--no-plots',action='store_true',help='Export numerical audit only')
    ground=commands.add_parser('audit-ground',help='Audit a downloaded public native-band reference; no rendering')
    ground.add_argument('--kind',required=True,choices=['astmon','galicia'])
    ground.add_argument('--file',type=Path,required=True)
    ground.add_argument('--output',type=Path,required=True)
    select=commands.add_parser('select-ground',help='Select native V nighttime/Moon-below-horizon candidates; clouds remain unknown')
    select.add_argument('--file',type=Path,required=True)
    select.add_argument('--year',type=int,required=True)
    select.add_argument('--months',type=int,nargs='+',required=True)
    select.add_argument('--latitude',type=float,required=True)
    select.add_argument('--longitude',type=float,required=True)
    select.add_argument('--height',type=float,required=True)
    select.add_argument('--timezone',default='Europe/Madrid')
    select.add_argument('--first-hour',type=int,default=0)
    select.add_argument('--last-hour',type=int,default=4)
    select.add_argument('--output',type=Path,required=True)
    fit=commands.add_parser('fit-ground',help='Fit a city scale on training nights and evaluate disjoint held-out periods')
    fit.add_argument('--selection',type=Path,required=True)
    fit.add_argument('--model-grids',type=Path,nargs='+',required=True)
    fit.add_argument('--training-periods',nargs='+',required=True)
    fit.add_argument('--allow-unverified-clouds',action='store_true')
    fit.add_argument('--output',type=Path,required=True)
    atmosphere=commands.add_parser('atmosphere',help='Export editable baseline atmospheric coefficients, not measured weather')
    atmosphere.add_argument('--output',type=Path,required=True)
    seasonal=commands.add_parser('prepare-seasonal',help='Match monthly QA/coverage and preserve the common source amplitude')
    seasonal.add_argument('--regions',type=Path,nargs='+',required=True)
    seasonal.add_argument('--output',type=Path,required=True)
    seasonal.add_argument('--reference',required=True,help='Reference period, e.g. 2025-01')
    seasonal.add_argument('--latitude',type=float,default=60.30)
    seasonal.add_argument('--longitude',type=float,default=30.315)
    seasonal.add_argument('--height',type=float,default=20)
    seasonal.add_argument('--exclude-radius-m',type=float,default=3000)
    seasonal.add_argument('--minimum-common-fraction',type=float,default=.8)
    seasonal.add_argument('--composite',choices=['NearNadir','OffNadir','AllAngle'],default='NearNadir')
    context=commands.add_parser('context',help='Sun/Moon geometry; not a prediction of lunar sky brightness')
    context.add_argument('--utc',required=True)
    context.add_argument('--latitude',type=float,default=60.30)
    context.add_argument('--longitude',type=float,default=30.315)
    context.add_argument('--height',type=float,default=20)
    context.add_argument('--look-azimuth',type=float,default=180)
    context.add_argument('--look-altitude',type=float,default=35)
    context.add_argument('--output',type=Path,required=True)
    observe=commands.add_parser('observe',help='Numerical dated/undated V-band scene with separate lunar/city/natural components')
    observe.add_argument('--scene',type=Path,required=True,help='Versioned JSON scene; grid paths relative to this file')
    observe.add_argument('--output',type=Path,required=True)
    view=commands.add_parser('render',help='Stars and computed skyglow as seen from the saved observer')
    view.add_argument('--sky',type=Path,default=Path('artifacts/petersburg-2025/observer-sky/sky-grid.npz'))
    view.add_argument('--output',type=Path,default=Path('artifacts/petersburg-2025/observer-view'))
    view.add_argument('--utc',required=True,help='Explicit UTC time, e.g. 2026-11-09T21:00:00Z')
    view.add_argument('--artificial-zenith-mag',type=float,required=True,
                      help='ASSUMED artificial zenith V brightness in mag/arcsec2; not satellite calibration')
    view.add_argument('--natural-mag',type=float,default=21.6)
    view.add_argument('--dark-limiting-mag',type=float,default=6.5)
    view.add_argument('--kstars-directory',type=Path,help='Also composite the computed field on an exported KStars raster')
    view.add_argument('--look-azimuth',type=float,default=180)
    view.add_argument('--look-altitude',type=float,default=35)
    view.add_argument('--horizontal-fov',type=float,default=110)
    for name,help_text in (('simulate','Calculate from a successfully audited NASA region'),
                           ('demo','Synthetic two-city example; not NASA measurements')):
        run=commands.add_parser(name,help=help_text)
        run.add_argument('--output',type=Path,default=Path('artifacts/'+('synthetic-demo' if name=='demo' else 'petersburg-2025/observer-sky')))
        run.add_argument('--region',type=Path,default=Path('artifacts/petersburg-2025/source-grid.npz'))
        run.add_argument('--latitude',type=float,default=60.30)
        run.add_argument('--longitude',type=float,default=30.315)
        run.add_argument('--height',type=float,default=20)
        run.add_argument('--azimuth-step',type=float,default=10)
        run.add_argument('--altitudes',type=float,nargs='+',default=DEFAULT_ALTITUDES if name=='demo' else [0,.5]+DEFAULT_ALTITUDES)
        run.add_argument('--ray-segments',type=int,default=96)
        run.add_argument('--path-quadrature',type=int,default=12)
        run.add_argument('--aod',type=float,help='Override aerosol optical depth in the atmospheric scenario')
        run.add_argument('--atmosphere-file',type=Path,help='JSON atmospheric parameters; remaining parameters use baseline values')
        run.add_argument('--no-plots',action='store_true',help='Numerical outputs only')
        run.add_argument('--horizontal-fraction',type=float,default=0)
        run.add_argument('--block',type=int,default=12)
        run.add_argument('--composite',choices=['NearNadir','OffNadir','AllAngle'])
        run.add_argument('--exclude-radius-m',type=float,default=3000)
        run.add_argument('--include-water',action='store_true')
        run.add_argument('--refine-below-altitude',type=float,default=5,
                         help='Refine source patches near sightlines at/below this altitude')
    return parser



def main():
    parser=make_parser()
    args=parser.parse_args()
    try:
        if args.command in ('plan','example','datasets','serve','compare'):
            from .interfaces.commands import run_command
            run_command(args)
        elif args.command=='catalogue':
            period=CompositePeriod(args.year,args.month)
            manifest=catalogue(args.year) if args.month is None else catalogue(args.year,month=args.month)
            write_manifest(args.output or Path(f'artifacts/nasa-{period.label}-catalogue.json'),manifest)
            print(f"Catalogue only: {len(manifest['files'])} files; not downloaded.")
        elif args.command=='download':
            period=CompositePeriod(args.year,args.month)
            download_composite(args.directory or Path(f'data/blackmarble/{period.label}'),args.year,
                               manifest_path=args.manifest,month=args.month)
        elif args.command=='audit':
            period=CompositePeriod(args.year,args.month)
            args.directory=args.directory or Path(f'data/blackmarble/{period.label}')
            args.output=args.output or Path(f'artifacts/petersburg-{period.label}')
            arrays,attributes=read_region(args.directory,Bounds(),args.year,args.month)
            report=audit(arrays,Bounds())
            report['observation_period']=period.metadata()
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            (args.output/'metadata.json').write_text(json.dumps(attributes,ensure_ascii=False,indent=2)+'\n')
            inputs=[{"filename":value['filename'],"sha256":value['sha256']} for value in attributes.values()]
            save_region(args.output/'source-grid.npz',arrays,Bounds(),
                        {**period.metadata(),"product":period.product+'.002',"inputs":inputs})
            if not args.no_plots:
                source_map(arrays,Bounds(),args.output/'sources.png',year=args.year,month=args.month)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        elif args.command=='select-ground':
            from .calibration import astmon_candidates
            rows,selection=astmon_candidates(args.file,Observer(args.latitude,args.longitude,args.height),
                                             args.year,args.months,args.timezone,args.first_hour,args.last_hour)
            write_manifest(args.output,{'selection':selection,'records':rows})
            print(json.dumps(selection,ensure_ascii=False))
        elif args.command=='fit-ground':
            from .calibration import fit_and_check, model_zenith_predictions, nightly_values
            data=json.loads(args.selection.read_text())
            predictions,provenance=model_zenith_predictions(args.model_grids,Observer(**data['selection']['observer']))
            report=fit_and_check(nightly_values(data['records']),predictions,args.training_periods,
                                allow_unverified_clouds=args.allow_unverified_clouds)
            write_manifest(args.output,{**report,'selection':data['selection'],'model_inputs':provenance})
            print(json.dumps({key:report[key] for key in ['status','reference_city_mag_arcsec2']},ensure_ascii=False))
        elif args.command=='atmosphere':
            write_manifest(args.output,asdict(Atmosphere()))
            print(json.dumps(atmosphere_origin(),ensure_ascii=False))
        elif args.command=='prepare-seasonal':
            from .seasonal import prepare_seasonal
            report=prepare_seasonal(args.regions,args.output,args.reference,
                    Observer(args.latitude,args.longitude,args.height),args.exclude_radius_m,
                    minimum_common_fraction=args.minimum_common_fraction,composite=args.composite)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        elif args.command=='audit-ground':
            from .ground import audit_astmon, audit_galicia
            report=(audit_astmon if args.kind=='astmon' else audit_galicia)(args.file)
            write_manifest(args.output,report)
            print(json.dumps({key:report[key] for key in ('dataset','rows','sha256')},ensure_ascii=False,indent=2))
        elif args.command=='context':
            from .observation import observation_context
            report=observation_context(Observer(args.latitude,args.longitude,args.height),args.utc,
                                       args.look_azimuth,args.look_altitude)
            write_manifest(args.output,report)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        elif args.command=='observe':
            from .scene import calculate_observation, write_observation
            report=calculate_observation(json.loads(args.scene.read_text()),args.scene.resolve().parent)
            write_observation(report,args.output)
            print(json.dumps({'mode':report['mode'],'utc':report['utc'],
                              'directions':len(report['directions']),
                              'supported_directions':sum(row['supported'] for row in report['directions']),
                              'stars':len(report['stars']),'output':str(args.output)},ensure_ascii=False))
        elif args.command=='render':
            from .skyview import DisplayScenario, render_observer
            scenario=DisplayScenario(args.artificial_zenith_mag,args.natural_mag,args.dark_limiting_mag)
            if args.kstars_directory:
                kstars_metadata=json.loads((args.kstars_directory/'kstars-summary.json').read_text())
                if kstars_metadata['utc']!=args.utc:
                    raise ValueError('KStars export and observer renderer timestamps differ.')
            summary=render_observer(args.sky,args.output,args.utc,scenario,
                                    args.look_azimuth,args.look_altitude,args.horizontal_fov)
            if args.kstars_directory:
                from .skyview import composite_kstars
                composite_kstars(args.sky,args.kstars_directory,scenario)
            print(json.dumps({key:summary[key] for key in ('observer','astrometry','display_scenario',
                                                         'catalogue_rows','above_horizon','dark_visible','scenario_visible')},
                             ensure_ascii=False,indent=2))
        else:
            print(json.dumps(run_simulation(args),ensure_ascii=False,indent=2))
    except (ValueError,RuntimeError,OSError,requests.RequestException,KeyError) as exc:
        parser.exit(1,f"Error: {exc}\n")


if __name__=='__main__':
    main()
