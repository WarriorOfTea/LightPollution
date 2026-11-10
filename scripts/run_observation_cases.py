"""Reproduce numerical stage 1–3 cases using the existing audited annual field.

No rendering, new NASA download, eye threshold, or field validation is implied.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import json

from lightpollution.moonlight import scattered_moonlight, nanolambert_to_v_mag
from lightpollution.scene import calculate_observation, write_observation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/observation-core-20261110'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    base={'schema_version':1,'mode':'undated','cloud_free':True,
          'observer':{'latitude_deg':60.3,'longitude_deg':30.315,'height_m':20.},
          'directions':[[az,alt] for alt in [15,45] for az in [0,90,180,270]]+[[0,90],[180,30]],
          'reference_v_mag_arcsec2':22.,
          'natural':{'mode':'uniform','v_mag_arcsec2':21.6,'source':'Assumed isotropic V background; not measured.'},
          'artificial':{'mode':'grid','sky_grid':'../petersburg-2025/observer-sky/sky-grid.npz',
                        'anchor_direction':[0,90],'anchor_v_mag_arcsec2':20.5,
                        'scale_origin':'scenario','source':'Assumed artificial zenith 20.5 V mag/arcsec2; not NASA calibration.',
                        'band_bridge':'effective_550nm_shape_anchored_in_V'},
          'moon':{'mode':'none'}}
    # Resolve independently of the output location and encode a relative input path.
    import os
    grid=Path('artifacts/petersburg-2025/observer-sky/sky-grid.npz').resolve()
    base['artificial']['sky_grid']=os.path.relpath(grid,args.output.resolve())
    cases={'undated-moonless':deepcopy(base)}
    for phase in [30,60,90,120]:
        scene=deepcopy(base)
        scene['moon']={'mode':'manual','azimuth_deg':180.,'altitude_deg':30.,
                       'phase_angle_deg':phase,'distance_km':384400.}
        cases[f'undated-phase-{phase}']=scene
    for name,time in [('dated-2025','2025-01-13T00:00:00+03:00'),
                      ('future-2027','2027-01-20T21:00:00Z')]:
        scene=deepcopy(base)
        scene.pop('moon')
        scene.update(mode='dated',time=time)
        cases[name]=scene
    reports={}
    for name,scene in cases.items():
        (args.output/f'{name}.scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n')
        report=calculate_observation(scene,args.output)
        write_observation(report,args.output/name)
        reports[name]=report
    expected=[[1160,530,437,818],[541,247,204,381],[217,99,82,153],[63,29,24,44]]
    comparisons=[]
    for phase,row in zip([30,60,90,120],expected,strict=True):
        for rho,published in zip([30,60,90,120],row,strict=True):
            predicted=float(scattered_moonlight(phase,30,90-abs(60-rho),rho,.172))
            comparisons.append({'phase_angle_deg':phase,'separation_deg':rho,
                                'published_nL':published,'calculated_nL':predicted,
                                'absolute_difference_nL':abs(predicted-published)})
    specsim=float(nanolambert_to_v_mag(scattered_moonlight(45,20,70,50,.15)))
    invariance=deepcopy(cases['undated-phase-60'])
    invariance['reference_v_mag_arcsec2']=20.
    changed=calculate_observation(invariance,args.output)
    original=reports['undated-phase-60']
    residual=max(abs(a['total_v_mag_arcsec2']-b['total_v_mag_arcsec2']) for a,b in
                 zip(changed['directions'],original['directions'],strict=True) if a['supported'])
    summary={'cases':{name:{'utc':r['utc'],'rays':len(r['directions']),
                          'supported_rays':sum(row['supported'] for row in r['directions']),
                          'stars_exported':len(r['stars']),
                          'moon':r['moon'],'zenith':r['directions'][8]} for name,r in reports.items()},
             'reference_checks':{'source':'Krisciunas-Schaefer 1991 Table 2, excluding rho=5deg',
                                 'comparisons':comparisons,
                                 'within_published_half_nL_rounding':sum(c['absolute_difference_nL']<=.5 for c in comparisons),
                                 'total_comparisons':len(comparisons),
                                 'maximum_absolute_difference_nL':max(c['absolute_difference_nL'] for c in comparisons),
                                 'specsim_documented_mag':19.855,'specsim_calculated_mag':specsim,
                                 'reference_change_max_mag_difference':residual},
             'physical_validation':False,'physical_error_bound':None,
             'new_satellite_files_downloaded':0,'eye_camera_rendering_performed':False}
    assert summary['reference_checks']['within_published_half_nL_rounding']==16
    assert residual<1e-12
    (args.output/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{key:value for key,value in data.items() if key not in ('moon','zenith')}
                      for name,data in summary['cases'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
