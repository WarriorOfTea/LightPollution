"""Real Petersburg eye-threshold scenario and a matched city-off control."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

from lightpollution.adapters.storage import write_json
from lightpollution.application.planner import Planner,export_result
from lightpollution.application.eye import build_eye_result
from lightpollution.domain.scene import SceneSpec
from lightpollution.interfaces.eye_view import render_eye_view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-model-20261111'))
    parser.add_argument('--workspace',type=Path,default=Path('data/eye-model-20261111'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if not (out/'predeclared-criteria.json').exists():
        write_json(out/'predeclared-criteria.json',{
            'declared_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
            'Crumey_dark_reference_mag':6.18,'reference_tolerance_mag':.015,
            'equation54_max_error_mag':.011,'eye_probability_model_fitted':False,
            'stellar_extinction_count':1,'same_view_and_display_scale_city_on_off':True,
            'independent_human_observers':0,
            'image_claim':'model illustration under stated conditions, not a calibrated photograph'})
    scene={'schema_version':2,'region':'petersburg-2025','period':'2025',
        'mode':'dated','time':'2026-11-11T21:00:00Z',
        'observer':{'latitude_deg':60.3,'longitude_deg':30.315,'height_m':2.},
        'directions':[[az,alt] for alt in [20.,45.,70.] for az in [90.,135.,180.,225.,270.]]+[[0.,90.]],
        'terrain':{'dataset':'petersburg-copdem90'},'near_field':{},'surface':{},
        'background':{},'profile':'quick',
        'eye':{'view':{'azimuth_deg':180.,'altitude_deg':75.,'horizontal_fov_deg':60.,'width':1800,'height':1100}}}
    spec=SceneSpec.from_mapping(scene)
    planner=Planner(args.workspace,raw_directory=Path('data/blackmarble'))
    result=planner.run(spec,progress=lambda p:print(p,flush=True))
    export_result(result,out/'city')
    city=result.value
    display=render_eye_view(city,out/'city',title='СЕВЕР ПЕТЕРБУРГА · МОДЕЛЬ НЕВООРУЖЁННОГО ГЛАЗА')
    reference=city['eye']['spectral_bridge']['reference_A0_scotopic_lux_at_observed_V0']
    field=deepcopy(city['eye']['field'])
    field['city_photopic_luminance_cd_m2']=np.zeros_like(field['city_photopic_luminance_cd_m2']).tolist()
    field['city_scotopic_luminance_cd_m2']=np.zeros_like(field['city_scotopic_luminance_cd_m2']).tolist()
    if 'view_patch' in field:
        patch=field['view_patch']
        for key in ['city_photopic_luminance_cd_m2','city_scotopic_luminance_cd_m2']:
            patch[key]=np.zeros_like(patch[key]).tolist()
        patch['counterfactual_city_emission_multiplier']=0.
    field['counterfactual_city_emission_multiplier']=0.
    dark=deepcopy(city)
    dark['eye']=build_eye_result(dark,city['eye']['config'],field,reference)
    dark['counterfactual']={'city_emission_multiplier':0.,'same_stellar_fluxes_and_atmosphere':True,
        'not_a_different_measured_site':True}
    dark_folder=out/'city-off';dark_folder.mkdir(exist_ok=True)
    # This is an eye-only intervention, not another complete radiative-transfer
    # result: do not retain contradictory city-on sky/spectral export tables.
    write_json(dark_folder/'result.json',{'artifact_type':'eye_only_counterfactual',
        'request':dark['request'],'counterfactual':dark['counterfactual'],'eye':dark['eye']})
    from lightpollution.application.eye import export_eye
    export_eye(dark['eye'],dark_folder)
    dark_display=render_eye_view(dark,dark_folder,title='ТА ЖЕ ТОЧКА · УСЛОВНОЕ ОТКЛЮЧЕНИЕ ГОРОДСКОГО СВЕТА')
    # Preserve the physical irradiance, change only the observer/background scenarios.
    sensitivity=[]
    for factor in [1.4,2.,2.4]:
        for ratio in [1.408,2.5]:
            config={**city['eye']['config'],'observer_factor':factor,'unresolved_background_sp_ratio':ratio}
            eye=build_eye_result(city,config,city['eye']['field'],reference)
            sensitivity.append({'observer_factor':factor,'unresolved_background_sp_ratio':ratio,**eye['summary']})
    write_json(out/'sensitivity.json',sensitivity)
    if display['available'] and dark_display['available']:
        a=Image.open(out/'city/eye-view.png').convert('RGB');b=Image.open(out/'city-off/eye-view.png').convert('RGB')
        pair=Image.new('RGB',(a.width,a.height+b.height+14),(11,14,18));pair.paste(a,(0,0));pair.paste(b,(0,a.height+14))
        pair.save(out/'eye-comparison.png')
    write_json(out/'summary.json',{'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'request':spec.to_dict(),'city_thresholds':city['eye']['summary'],'city_display':display,
        'city_off_thresholds':dark['eye']['summary'],'city_off_display':dark_display,
        'runtime':city['runtime'],'cache':city['cache'],'global_restrictions':city['eye']['global_restrictions'],
        'source_basis_id':city['data_provenance']['source_basis_id'],
        'independent_human_observers':0,'field_accuracy':None,
        'moon':city['observation']['moon'],'geometry':city['observation']['geometry'],
        'screen_colour_and_absolute_brightness_validated':False})
    print('DONE',display.get('displayed_stars'),dark_display.get('displayed_stars'),flush=True)


if __name__=='__main__':main()
