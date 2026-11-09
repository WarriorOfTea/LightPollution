"""Small reproducible command-line workflow; credentials never enter output files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import requests

from .data import Bounds, audit, load_region, read_region, save_region
from .download import catalogue, download_annual, write_manifest
from .physics import Atmosphere, Observer, SolverSettings, sky_grid
from .sources import regional_sources, synthetic_cities
from .visualization import source_map, sky_plot

DEFAULT_ALTITUDES = [1,2,3,5,7.5,10,15,20,30,45,60,75,90]


def make_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="Black Marble audit and experimental directional skyglow")
    commands=parser.add_subparsers(dest='command',required=True)
    cat=commands.add_parser('catalogue',help='Inspect public NASA file metadata; no authentication needed')
    cat.add_argument('--year',type=int,default=2025)
    cat.add_argument('--output',type=Path)
    fetch=commands.add_parser('download',help='Download four pinned Petersburg HDF5 tiles')
    fetch.add_argument('--year',type=int,default=2025)
    fetch.add_argument('--directory',type=Path)
    fetch.add_argument('--manifest',type=Path)
    check=commands.add_parser('audit',help='Check and extract the 29–32 E, 59–61 N window')
    check.add_argument('--year',type=int,default=2025)
    check.add_argument('--directory',type=Path)
    check.add_argument('--output',type=Path)
    for name,help_text in (('simulate','Calculate from a successfully audited NASA region'),
                           ('demo','Synthetic two-city example; not NASA measurements')):
        run=commands.add_parser(name,help=help_text)
        run.add_argument('--output',type=Path,default=Path('artifacts/'+('synthetic-demo' if name=='demo' else 'petersburg-2025/sky')))
        run.add_argument('--region',type=Path,default=Path('artifacts/petersburg-2025/source-grid.npz'))
        run.add_argument('--latitude',type=float,default=60.30)
        run.add_argument('--longitude',type=float,default=30.315)
        run.add_argument('--height',type=float,default=20)
        run.add_argument('--azimuth-step',type=float,default=10)
        run.add_argument('--altitudes',type=float,nargs='+',default=DEFAULT_ALTITUDES)
        run.add_argument('--ray-segments',type=int,default=96)
        run.add_argument('--path-quadrature',type=int,default=12)
        run.add_argument('--aod',type=float,default=0.10)
        run.add_argument('--horizontal-fraction',type=float,default=0)
        run.add_argument('--block',type=int,default=12)
        run.add_argument('--exclude-radius-m',type=float,default=3000)
        run.add_argument('--include-water',action='store_true')
    return parser


def run_simulation(args) -> dict:
    if not 0<args.azimuth_step<=90 or not np.isclose(360/args.azimuth_step,round(360/args.azimuth_step)):
        raise ValueError("Azimuth step must divide 360 exactly and be in (0,90].")
    if args.altitudes != sorted(set(args.altitudes)) or not (0 <= min(args.altitudes) <= max(args.altitudes) <= 90):
        raise ValueError("Altitudes must be unique, increasing and inside [0,90].")
    observer=Observer(args.latitude,args.longitude,args.height)
    atmosphere=Atmosphere(aerosol_optical_depth=args.aod)
    settings=SolverSettings(args.ray_segments,args.path_quadrature,args.horizontal_fraction)
    if args.command=='demo':
        sources=synthetic_cities(observer)
        provenance={"input_kind":"synthetic", "actual_nasa_pixels":0,
                    "description":"Two specified test cities: 40 km south and 25 km east; total test power 1.25 W.",
                    "source_power_units":"W (synthetic)","effective_sources":len(sources.power)}
        title='Учебный пример: два заданных города. Это не данные NASA'
    else:
        arrays,bounds,metadata=load_region(args.region)
        sources,provenance=regional_sources(arrays,bounds,observer,args.block,args.exclude_radius_m,include_water=args.include_water)
        provenance.update({"input_kind":"blackmarble", "region_file":str(args.region),"region_metadata":metadata})
        title=f"Петербург, VIIRS {metadata.get('year','')}: условный региональный сценарий"
    azimuths=np.arange(0,360,args.azimuth_step)
    altitudes=np.asarray(args.altitudes)
    started=time.perf_counter()
    radiance=sky_grid(observer,sources,azimuths,altitudes,atmosphere,settings)
    elapsed=time.perf_counter()-started
    location=np.unravel_index(np.argmax(radiance),radiance.shape)
    summary={"provenance":provenance,"observer":asdict(observer),"atmosphere":asdict(atmosphere),
             "solver":asdict(settings),"shape":list(radiance.shape),"elapsed_seconds":elapsed,
             "azimuth_convention":"North=0, East=90, clockwise", "raw_units":"W/(m2 sr)" if args.command=='demo' else "m^-2 sr^-1 per unit total assumed source power",
             "absolute_visible_photometry_calibrated":False,
             "maximum":{"azimuth_deg":float(azimuths[location[1]]),"altitude_deg":float(altitudes[location[0]]),"value":float(radiance[location])},
             "physical_error_bound":None,"model_document":"docs/MODEL.md"}
    for target in (5,15,45):
        if target in altitudes:
            row=int(np.flatnonzero(altitudes==target)[0])
            south,north=radiance[row,np.argmin(abs(azimuths-180))],radiance[row,0]
            summary[f"south_over_north_at_{target}deg"]=float(south/north) if north>0 else None
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'sky-grid.npz',azimuth_deg=azimuths,altitude_deg=altitudes,radiance=radiance,
                        metadata_json=json.dumps(summary,ensure_ascii=False))
    az_mesh,alt_mesh=np.meshgrid(azimuths,altitudes)
    np.savetxt(args.output/'sky-grid.csv',np.column_stack((az_mesh.ravel(),alt_mesh.ravel(),radiance.ravel())),
               delimiter=',',header='azimuth_deg,altitude_deg,radiance',comments='',fmt='%.12g')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    sky_plot(azimuths,altitudes,radiance,args.output/'sky.png',title)
    return summary


def main():
    parser=make_parser()
    args=parser.parse_args()
    try:
        if args.command=='catalogue':
            manifest=catalogue(args.year)
            write_manifest(args.output or Path(f'artifacts/nasa-{args.year}-catalogue.json'),manifest)
            print(f"Catalogue only: {len(manifest['files'])} files; not downloaded.")
        elif args.command=='download':
            download_annual(args.directory or Path(f'data/blackmarble/{args.year}'),args.year,manifest_path=args.manifest)
        elif args.command=='audit':
            args.directory=args.directory or Path(f'data/blackmarble/{args.year}')
            args.output=args.output or Path(f'artifacts/petersburg-{args.year}')
            arrays,attributes=read_region(args.directory,Bounds(),args.year)
            report=audit(arrays,Bounds())
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            (args.output/'metadata.json').write_text(json.dumps(attributes,ensure_ascii=False,indent=2)+'\n')
            inputs=[{"filename":value['filename'],"sha256":value['sha256']} for value in attributes.values()]
            save_region(args.output/'source-grid.npz',arrays,Bounds(),{"year":args.year,"product":"VNP46A4.002", "inputs":inputs})
            source_map(arrays,Bounds(),args.output/'sources.png',year=args.year)
            print(json.dumps(report,ensure_ascii=False,indent=2))
        else:
            print(json.dumps(run_simulation(args),ensure_ascii=False,indent=2))
    except (ValueError,RuntimeError,OSError,requests.RequestException,KeyError) as exc:
        parser.exit(1,f"Error: {exc}\n")


if __name__=='__main__':
    main()
