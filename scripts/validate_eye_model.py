"""Reproduce published references and audit the saved physical eye example."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lightpollution.adapters.storage import file_hash,read_json,write_json
from lightpollution.models.eye import PHOTOPIC_V0_LUX,REFERENCE_SP,point_threshold,evaluate_detection
from lightpollution.adapters.spectral_library import load_spectral_library


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-model-20261111'))
    args=parser.parse_args();out=args.output
    refs=[]
    for factor,reference in [(1.,6.93),(2.,6.18)]:
        actual=float(-2.5*np.log10(point_threshold(2e-4,factor)/PHOTOPIC_V0_LUX))
        assert abs(actual-reference)<.015
        refs.append({'background_cd_m2':2e-4,'observer_factor':factor,
                     'published_rounded_reference_mag':reference,'actual_mag':actual,
                     'absolute_residual_mag':abs(actual-reference)})
    mu=np.linspace(20.01,21.99,101)
    b=PHOTOPIC_V0_LUX*(180*3600/np.pi)**2*10**(-.4*mu)
    exact=-2.5*np.log10(point_threshold(b,2)/PHOTOPIC_V0_LUX)
    residual=float(np.max(abs(exact-(.3834*mu-1.4400-2.5*np.log10(2)))))
    assert residual<.011
    library=load_spectral_library();wl=library.wavelengths
    # Planck's law, density on the same 5-nm grid as the committed CIE tables.
    wavelength_m=wl*1e-9
    planck=wavelength_m**-5/np.expm1(1.438776877e-2/(wavelength_m*2850))
    rho=float(1700*np.trapezoid(planck*library.responses['scotopic'],wl)/(683*np.trapezoid(planck*library.responses['photopic'],wl)))
    assert abs(rho/REFERENCE_SP-1)<.002
    t=float(point_threshold(2e-4,2))
    full=evaluate_detection(2e-4,2e-4*REFERENCE_SP,2*t*REFERENCE_SP)
    half=evaluate_detection(2e-4,2e-4*REFERENCE_SP,t*REFERENCE_SP)
    assert np.isclose(full['flux_over_threshold'],2) and np.isclose(half['flux_over_threshold'],1)
    city=read_json(out/'city/eye.json');dark=read_json(out/'city-off/eye.json')
    shared=[(a,c) for a,c in zip(city['stars'],dark['stars'],strict=True) if a['supported'] and c['supported']]
    assert all(a['hr']==c['hr'] and np.isclose(a['scotopic_illuminance_lux'],c['scotopic_illuminance_lux'],rtol=1e-12,atol=0)
               and c['flux_over_threshold']>=a['flux_over_threshold'] for a,c in shared)
    summary=read_json(out/'summary.json')
    assert summary['city_display']['view']==summary['city_off_display']['view']
    assert summary['city_off_display']['displayed_stars']>=summary['city_display']['displayed_stars']
    write_json(out/'numerical-validation.json',{
        'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'published_references':refs,'equation54_grid_points':101,
        'equation54_max_residual_mag':residual,'Planck2850K_computed_S_P':rho,
        'Planck2850K_relative_difference_from_rounded_1_408':rho/REFERENCE_SP-1,
        'single_attenuation_flux_ratios':[float(full['flux_over_threshold']),float(half['flux_over_threshold'])],
        'shared_supported_stars_city_on_off':len(shared),'same_stellar_flux_and_monotone_city_off_checked':True,
        'same_view_and_fixed_display_transfer':True,'independent_human_observers':0,
        'independent_visual_detections':0,'field_MAE_mag':None,
        'field_MAE_unavailable_reason':'No matched independent naked-eye observations or local spectral background measurements.',
        'image_hashes':{str(p.relative_to(out)):file_hash(p) for p in [out/'city/eye-view.png',out/'city-off/eye-view.png',out/'eye-comparison.png']}})
    write_json(out/'source-audit.json',{
        'Crumey2014':{'url':'https://arxiv.org/pdf/1405.4209v1','doi':'10.1093/mnras/stu992',
            'downloaded_bytes':1348165,'sha256':'fa6ef183f9402be4d321bff5fa2c112510f89ca683b534e33c63fdb6538e50a4',
            'equations_used':[5,23,26,32,48,53,54],
            'copyrighted_PDF_redistributed':False,'Blackwell_original_table_independently_digitised':False},
        'CIE1912010':{'url':'https://www.cie.co.at/publications/recommended-system-mesopic-photometry-based-visual-performance',
            'used_for':'scope of mesopic visual tasks; not implemented as a star-colour rendering law'},
        'spectral_tables':'Existing committed library; see photometry-spectra-20261111 and docs/DATA_SOURCES.md.',
        'NASA_DSM_and_stellar_provenance':'city/result.json; NASA annual 2025 and Copernicus DSM are not measurements of 2026-11-11 weather/lighting.'})
    print('Eye numerical checks passed; human validation: 0 observers, field accuracy unavailable.')


if __name__=='__main__':main()
