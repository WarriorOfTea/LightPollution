"""Optional real Chromium check of camera scene import, job, FITS and ZIP."""

import argparse
import io
import json
import os
from pathlib import Path
import zipfile

from astropy.io import fits
from playwright.sync_api import sync_playwright

from lightpollution.adapters.storage import write_json
from lightpollution.domain.scene import SceneSpec
from lightpollution.interfaces.commands import offline_example


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:8766')
    parser.add_argument('--output',type=Path,default=Path('artifacts/camera-model-20261112/browser'))
    parser.add_argument('--library-directory',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    scene=offline_example();scene.update(profile='quick',directions=[[180.,50.]],camera={
        'exposure_s':15.,'profile':{'sensor':{'width_px':320,'height_px':240}},
        'view':{'azimuth_deg':180.,'altitude_deg':50.,'rotation_deg':37.},'sampling':{'max_grid_size':3}})
    scene=SceneSpec.from_mapping(scene).to_dict();write_json(args.output/'import-scene.json',scene)
    env=dict(os.environ)
    if args.library_directory:env['LD_LIBRARY_PATH']=str(args.library_directory.resolve())
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,env=env);page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url,wait_until='networkidle');page.set_input_files('#import',args.output/'import-scene.json')
        page.wait_for_function('document.querySelector("#camera-enabled").checked')
        assert page.evaluate('readForm().camera')==scene['camera']
        assert page.evaluate('Array.from(document.querySelectorAll("input")).filter(x=>!x.checkValidity()).map(x=>x.id)')==[]
        with page.expect_response(lambda r:r.url.endswith('/api/jobs') and r.request.method=='POST') as response:page.click('#calculate')
        assert response.value.status==202,response.value.text();job=response.value.json()['id']
        page.wait_for_function('active===null && currentResult!==null',timeout=180000)
        result=page.evaluate('currentResult');assert result['camera']['exports']['available']
        base=f'{args.url}/api/jobs/{job}/export/'
        names=['camera.json','camera.fits','camera-preview.png','camera-stars.csv','camera-profile.json','camera-ekos.json']
        for name in names:
            r=page.request.get(base+name);assert r.status==200 and len(r.body())>0,name
        body=page.request.get(base+'camera.fits').body()
        with fits.open(io.BytesIO(body),checksum=True) as hdus:
            assert hdus[0].header['SIMULATE'] and hdus[0].header['EXPTIME']==15
            assert all(h.verify_checksum()==1 for h in hdus)
        with zipfile.ZipFile(io.BytesIO(page.request.get(base+'bundle.zip').body())) as archive:
            assert set(names)<=set(archive.namelist())
        page.get_by_text('Линейная камера',exact=True).click();page.screenshot(path=str(args.output/'camera-controls.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        write_json(args.output/'verification.json',{'jobs_completed':1,'jobs_requested':1,'camera_scene_round_trip':True,
            'camera_exports_checked':len(names),'camera_exports_requested':len(names),'linear_FITS_checksum_and_exposure_checked':True,
            'bundle_includes_camera':True,'javascript_errors':errors,'mobile_width_px':390,'mobile_horizontal_overflow':False,
            'job_id':job,'ground_camera_validation':False})
        browser.close();print('Camera Chromium: 1/1 jobs, 6/6 exports, FITS checksum, scene round trip, ZIP, no JS errors.')


if __name__=='__main__':main()
