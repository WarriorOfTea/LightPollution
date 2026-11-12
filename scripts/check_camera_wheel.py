"""Check the installed camera package, outside the source checkout."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from astropy.io import fits
import lightpollution
from lightpollution.adapters.ekos_camera import import_ekos_camera
from lightpollution.application.planner import Planner, export_result
from lightpollution.domain.scene import SceneSpec
from lightpollution.interfaces.commands import offline_example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.project.resolve()
    installed=Path(lightpollution.__file__).parent
    assert 'site-packages' in str(installed),installed
    files=[p for p in (root/'src/lightpollution').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    for path in files:
        relative=path.relative_to(root/'src/lightpollution')
        assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((installed/relative).read_bytes()).digest(),relative
    scene=offline_example()
    scene.update(profile='quick',directions=[[0.,90.]],spectrum={},camera={
        'profile':{'sensor':{'width_px':320,'height_px':240}},
        'view':{'azimuth_deg':0.,'altitude_deg':85.},
        'sampling':{'max_grid_size':3}})
    result=Planner(root/'scratch/camera-wheel-workspace').run(SceneSpec.from_mapping(scene))
    output=root/'scratch/camera-wheel-output';export_result(result,output)
    camera=json.loads((output/'result.json').read_text())['camera']
    assert camera['model']=='linear_monochrome_electron_camera_1' and camera['available']
    names=['camera.json','camera.fits','camera-preview.png','camera-stars.csv',
           'camera-profile.json','camera-ekos.json']
    assert all((output/name).is_file() for name in names)
    with fits.open(output/'camera.fits',checksum=True) as hdus:
        assert hdus[0].data.shape==(240,320)
        assert all(hdu.verify_checksum()==1 for hdu in hdus)
        count=len(hdus)
    profile=json.loads((output/'camera-profile.json').read_text())
    assert profile['profile']['origin']=='synthetic'
    record={'version':lightpollution.__version__,'python':sys.version.split()[0],
        'working_directory':str(Path.cwd()),'installed_path':str(installed),
        'package_files_matching_checkout':len(files),'package_files_checked':len(files),
        'distributions_in_isolated_environment':len(list(importlib.metadata.distributions())),
        'offline_camera_scene_completed':True,'camera_exports_checked':len(names),
        'FITS_HDU_checksums_passed':count,'FITS_HDU_checksums_checked':count,
        'camera_profile_adapter_importable':callable(import_ekos_camera)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
