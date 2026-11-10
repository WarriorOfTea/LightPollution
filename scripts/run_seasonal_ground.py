"""Real monthly QA, amplitude-preserving kernels, native-band ground comparisons.

Requires locally downloaded/pinned HDF5 and AstMon. No token handling or imagery.
Large extracted source maps live in ignored data/, reports and small kernels in Git.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from lightpollution.calibration import astmon_candidates, fit_and_check, model_zenith_predictions, nightly_values
from lightpollution.cli import make_parser, run_simulation
from lightpollution.data import Bounds, audit, read_region, save_region
from lightpollution.download import write_manifest
from lightpollution.periods import CompositePeriod
from lightpollution.physics import Observer
from lightpollution.seasonal import prepare_seasonal


PUBLICATION_WINDOWS=[
    {'start':'2017-03-19T22:00:00Z','end':'2017-03-19T23:00:00Z','figure':5},
    {'start':'2017-11-19T22:00:00Z','end':'2017-11-19T23:00:00Z','figure':6},
]
PAPER='https://www.mdpi.com/2072-4292/13/8/1511'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--astmon',type=Path,default=Path('scratch/data-search-20261110/AstMon_evol.csv'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/seasonal-ground-20261110'))
    parser.add_argument('--cache',type=Path,default=Path('data/seasonal-ground-20261110'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    madrid=Observer(40.451111,-3.726111,650)
    cases=[('petersburg',2025,1,Bounds()),('petersburg',2025,7,Bounds())]
    cases += [('madrid',year,month,Bounds(-5,39,-2,42)) for year,month in [(2015,1),(2015,7),(2017,3),(2017,11)]]
    for site,year,month,bounds in cases:
        p=CompositePeriod(year,month)
        name=f'{site}-{p.label}'
        folder=Path('data/blackmarble')/(p.label if site=='petersburg' else name)
        arrays,attrs=read_region(folder,bounds,year,month)
        report_folder=args.output/name
        report_folder.mkdir(parents=True,exist_ok=True)
        write_manifest(report_folder/'audit.json',audit(arrays,bounds))
        write_manifest(report_folder/'metadata.json',attrs)
        shutil.copyfile(folder/'manifest.json',report_folder/'download-manifest.json')
        save_region(args.cache/name/'source-grid.npz',arrays,bounds,{**p.metadata(),'product':p.product+'.002',
            'inputs':[{'filename':v['filename'],'sha256':v['sha256']} for v in attrs.values()]})
    try:
        prepare_seasonal([args.cache/'petersburg-2025-01/source-grid.npz',args.cache/'petersburg-2025-07/source-grid.npz'],
                         args.cache/'petersburg-matched','2025-01')
    except ValueError as exc:
        write_manifest(args.output/'petersburg-unavailable.json',{'status':'rejected','reason':str(exc)})
    for year,months in [(2015,[1,7]),(2017,[3,11])]:
        matched=args.cache/f'madrid-matched-{year}'
        reference=f'{year}-{months[0]:02}'
        seasonal=prepare_seasonal([args.cache/f'madrid-{year}-{m:02}/source-grid.npz' for m in months],
                                  matched,reference,madrid)
        write_manifest(args.output/f'madrid-seasonal-{year}.json',seasonal)
        grids=[]
        for month in months:
            period=f'{year}-{month:02}'
            target=args.output/f'madrid-model-{period}'
            sim=make_parser().parse_args(['simulate','--region',str(matched/period/'source-grid.npz'),
                '--output',str(target),'--latitude',str(madrid.latitude_deg),'--longitude',str(madrid.longitude_deg),
                '--height',str(madrid.height_m),'--azimuth-step','90','--altitudes','15','45','90',
                '--ray-segments','192','--path-quadrature','24','--block','12','--no-plots'])
            run_simulation(sim)
            grids.append(target/'sky-grid.npz')
        first,last=(0,4) if year==2015 else (23,24)
        records,selection=astmon_candidates(args.astmon,madrid,year,months,first_hour=first,last_hour=last)
        write_manifest(args.output/f'astmon-selection-{year}.json',{'selection':selection,'records':records})
        if year==2017:
            selected=[]
            for record in records:
                for window in PUBLICATION_WINDOWS:
                    if window['start']<=record['utc']<window['end']:
                        selected.append({**record,'cloud_free_confirmed':True,
                            'cloud_qa_source':f"{PAPER}, Figure {window['figure']}: filled clear/moonless markers; conservative interior interval."})
            records=selected
            selection={**selection,'publication_windows':PUBLICATION_WINDOWS,
                       'cloud_flags_in_file':False,'cloud_qa_method':'Manual transcription of author-labelled figures, not an imported CSV flag.',
                       'parent_astronomical_candidates':selection['astronomical_moonless_candidates'],
                       'records_exported':len(records),'certified_cloud_free_candidates':len(records),'unknown_cloud_candidates':0}
            write_manifest(args.output/'astmon-publication-qa.json',{'selection':selection,'records':records})
        predictions,model_inputs=model_zenith_predictions(grids,madrid)
        fit=fit_and_check(nightly_values(records),predictions,[reference],allow_unverified_clouds=year==2015)
        write_manifest(args.output/f'fit-{year}.json',{**fit,'selection':selection,'model_inputs':model_inputs})
        print(json.dumps({'year':year,'common_pixels':seasonal['common_pixels'],
                          'domain':seasonal['reference_land_domain_pixels'],'source_multipliers':[
                          p['source_power_multiplier'] for p in seasonal['periods']],
                          'status':fit['status'],'training_nights':fit['training']['nights'],
                          'held_out_nights':fit['held_out']['nights'],'held_out_mae_mag':fit['held_out']['mae_mag']},ensure_ascii=False))


if __name__=='__main__':
    main()
