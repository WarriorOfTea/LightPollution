"""Run with the installed wheel's Python from outside the checkout."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import lightpollution
from lightpollution.application.planner import Planner, export_result
from lightpollution.domain.scene import SceneSpec
from lightpollution.interfaces.commands import offline_example


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--colour',action='store_true')
    args=parser.parse_args();root=args.project.resolve()
    installed=Path(lightpollution.__file__).parent
    assert 'site-packages' in str(installed),installed
    files=[p for p in (root/'src/lightpollution').rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    for p in files:
        relative=p.relative_to(root/'src/lightpollution')
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((installed/relative).read_bytes()).digest(),relative
    scene=offline_example()
    scene.update(profile='quick',directions=[[0.,90.]],eye={'view':{
        'altitude_deg':75.,'horizontal_fov_deg':40.,'width':640,'height':480}})
    if args.colour:scene['eye']['appearance']={'mode':'experimental_stellar_colour'}
    result=Planner(root/'scratch/eye-rendering-wheel-workspace').run(SceneSpec.from_mapping(scene))
    output=root/'scratch/eye-rendering-wheel-output'
    export_result(result,output)
    eye=json.loads((output/'result.json').read_text())['eye']
    assert eye['model']=='crumey2014_scotopic_at_eye_1' and eye['display']['available']
    assert eye['display']['display_version']==('experimental_spectral_star_colour_3' if args.colour else 'flux_integrated_achromatic_eye_2')
    names=['eye-view.png','eye-preview.png','eye-mobile.png','eye-view-16bit.png','eye-linear.npz']
    assert all((output/name).is_file() for name in names)
    if args.colour:
        from lightpollution.adapters.colour_library import load_colour_library
        assert load_colour_library().identity and (output/'eye-colour.csv').is_file()
    record={'version':lightpollution.__version__,'python':sys.version.split()[0],
        'working_directory':str(Path.cwd()),'installed_path':str(installed),
        'package_files_matching_checkout':len(files),'package_files_checked':len(files),
        'distributions_in_isolated_environment':len(list(importlib.metadata.distributions())),
        'offline_eye_scene_completed':True,'image_exports_checked':len(names),
        'eye_exports_and_PNG_available':True}
    record['experimental_colour_checked']=args.colour
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
