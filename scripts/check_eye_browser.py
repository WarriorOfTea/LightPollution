"""Optional Chromium round trip of the eye controls, job and real exports."""

import argparse
import io
import json
import os
from pathlib import Path
import zipfile

from playwright.sync_api import sync_playwright

from lightpollution.adapters.storage import write_json
from lightpollution.domain.scene import SceneSpec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-model-20261111/browser'))
    parser.add_argument('--library-directory',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scene=SceneSpec.from_mapping({'schema_version':2,'region':'demo','period':'synthetic',
        'mode':'dated','time':'2026-11-11T21:00:00Z',
        'observer':{'latitude_deg':60.3,'longitude_deg':30.315,'height_m':2.},
        'directions':[[0.,90.]],'background':{},'profile':'quick',
        'eye':{'observer_factor':2.4,'unresolved_background_sp_ratio':1.408,
            'view':{'azimuth_deg':180.,'altitude_deg':75.,'horizontal_fov_deg':40.,'width':960,'height':640}}})
    write_json(args.output/'import-scene.json',scene.to_dict())
    env=dict(os.environ)
    if args.library_directory:env['LD_LIBRARY_PATH']=str(args.library_directory.resolve())
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,env=env)
        page=browser.new_page(viewport={'width':1440,'height':1080});errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url,wait_until='networkidle')
        page.set_input_files('#import',args.output/'import-scene.json')
        page.wait_for_function('document.querySelector("#eye-enabled").checked')
        assert page.evaluate('readForm().eye')==scene.to_dict()['eye']
        assert page.evaluate('Array.from(document.querySelectorAll("input")).filter(x=>!x.checkValidity()).map(x=>x.id)')==[]
        with page.expect_response(lambda r:r.url.endswith('/api/jobs') and r.request.method=='POST') as response:
            page.click('#calculate')
        assert response.value.status==202,response.value.text()
        job=response.value.json()['id']
        page.wait_for_function('active===null && currentResult!==null',timeout=180000)
        result=page.evaluate('currentResult')
        assert result['eye']['display']['available'],result['eye']['display']
        base=f'{args.url}/api/jobs/{job}/export/'
        for name in ['eye.json','eye-stars.csv','eye-directions.csv','eye-display.json','eye-view.png']:
            downloaded=page.request.get(base+name)
            assert downloaded.status==200,(name,downloaded.status)
            assert len(downloaded.body())>0,name
        png=page.request.get(base+'eye-view.png').body()
        assert png.startswith(b'\x89PNG\r\n\x1a\n')
        bundle=page.request.get(base+'bundle.zip')
        assert bundle.status==200
        with zipfile.ZipFile(io.BytesIO(bundle.body())) as archive:
            names=archive.namelist()
            assert {'eye.json','eye-view.png','eye-stars.csv'}<=set(names)
        page.get_by_text('Невооружённый глаз',exact=True).click()
        page.screenshot(path=str(args.output/'eye-controls.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        write_json(args.output/'verification.json',{'jobs_completed':1,'jobs_requested':1,
            'eye_controls_and_dimensions_round_trip':True,'optional_exports_checked':5,
            'bundle_contains_eye_exports':True,'eye_display_available':True,
            'javascript_errors':errors,'mobile_width_px':390,'mobile_horizontal_overflow':False,
            'job_id':job,'independent_human_validation':False})
        browser.close()
        print('Browser eye: 1/1 jobs, 5/5 exports, ZIP, controls, mobile layout; zero JS errors.')


if __name__=='__main__':main()
