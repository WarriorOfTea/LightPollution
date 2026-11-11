"""Optional Chromium check of spatial inputs and KStars provenance round trip."""

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

from lightpollution.adapters.storage import write_json
from lightpollution.application.kstars import scene_from_kstars


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--output',type=Path,default=Path('artifacts/kstars-terrain-near-20261111'))
    parser.add_argument('--library-directory',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    spec,_=scene_from_kstars({'scene':{'region':'demo','period':'synthetic','profile':'quick',
        'observer':{'height_m':2.},'terrain':{'dataset':'demo-flat'},'near_field':{},'background':{}},
        'snapshot':{'schema_version':1,'kstars_version':'contract fixture, not live execution',
            'utc':'2026-11-11T21:00:00Z','altitude_convention':'geometric_unrefracted',
            'location':{'latitude_deg':60.3,'longitude_deg':30.315,'elevation_m':120.},
            'focus':{'azimuth_deg':0.,'altitude_deg':90.},'artificial_horizon_visible':True}})
    write_json(args.output/'browser-import-scene.json',spec.to_dict())
    env=dict(os.environ)
    if args.library_directory:env['LD_LIBRARY_PATH']=str(args.library_directory.resolve())
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,env=env)
        page=browser.new_page(viewport={'width':1440,'height':1080})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url,wait_until='networkidle')
        page.set_input_files('#import',args.output/'browser-import-scene.json')
        page.wait_for_function('document.querySelector("#terrain-dataset").value === "demo-flat"')
        preserved=page.evaluate('readForm()')
        assert preserved['terrain']==spec.to_dict()['terrain']
        assert preserved['near_field']==spec.to_dict()['near_field']
        assert preserved['integration']==spec.to_dict()['integration']
        page.fill('#longitude','30.32')
        assert 'integration' not in page.evaluate('readForm()')
        page.fill('#longitude','30.315')
        with page.expect_response(lambda r:r.url.endswith('/api/jobs') and r.request.method=='POST') as response:
            page.click('#calculate')
        assert response.value.status==202,response.value.text()
        job=response.value.json()['id']
        page.wait_for_function('active===null && currentResult!==null',timeout=120000)
        assert page.evaluate('currentResult.terrain.observer_agl_m')==2.
        assert page.evaluate('currentResult.request.near_field.minimum_patch_m')==2.
        page.get_by_text('Рельеф и ближние источники',exact=True).click()
        page.screenshot(path=str(args.output/'browser-spatial-controls.png'))
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        write_json(args.output/'browser-verification.json',{'jobs_completed':1,'jobs_requested':1,
            'new_spatial_controls_preserved':True,'KStars_provenance_preserved_without_geometry_edits':True,
            'KStars_provenance_removed_after_manual_geometry_edit':True,
            'javascript_errors':errors,'mobile_width_px':390,'mobile_horizontal_overflow':False,
            'job_id':job,'live_KStars_used_in_this_browser_test':False})
        browser.close()
        print('Browser: spatial inputs, provenance, 1/1 jobs, mobile layout and zero JS errors verified.')


if __name__=='__main__':main()
