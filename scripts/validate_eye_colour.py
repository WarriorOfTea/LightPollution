"""Optional independent 16-bit PNG decoder: uv run --with pypng python ..."""

import argparse
import hashlib
import importlib.metadata
import subprocess

import numpy as np
from pathlib import Path
from PIL import Image
import png

from lightpollution.adapters.storage import read_json,write_json
from lightpollution.interfaces.eye_view import encode_field
from lightpollution.models.eye_display import quantize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-colour-20261111'))
    parser.add_argument('--workspace',type=Path,default=Path('data/eye-colour-20261111'))
    parser.add_argument('--check-index',action='store_true',help='Verify staged CIE bytes survive Git line-ending rules.')
    args=parser.parse_args();out=args.output
    w,h,rows,meta=png.Reader(filename=str(out/'eye-view-16bit.png')).read()
    assert meta['bitdepth']==16 and not meta['greyscale'] and meta['planes']==3
    actual=np.asarray(list(rows),dtype=np.uint16).reshape(h,w,3)
    profile=read_json(out/'eye-profile.json')['eye']['display']
    with np.load(args.workspace/'colour/eye-linear.npz') as raw:
        encoded,summary=encode_field(raw,profile);expected=quantize(encoded,16,dither=False)
        supported=raw['supported'];expected[~supported]=0
        assert np.array_equal(actual,expected)
        error=float(abs(actual.astype(float)-65535*encoded)[supported].max())
        assert error<=.5
        pure_background=(raw['stellar_appearance_pixel_illuminance_lux']==0)&supported
    mono_checks={}
    for name in ['eye-view.png','eye-preview.png','eye-mobile.png','eye-view-16bit.png']:
        original=np.asarray(Image.open(Path('artifacts/eye-rendering-20261111/city')/name))
        current=np.asarray(Image.open(args.workspace/'monochrome'/name))
        mono_checks[name]=bool(np.array_equal(original,current))
    assert all(mono_checks.values())
    display=np.asarray(Image.open(out/'eye-view.png'))[80:80+h]
    grey_error=int(np.ptp(display[pure_background].astype(int),axis=-1).max()) if pure_background.any() else None
    assert grey_error==0
    record={'independent_decoder':'pypng '+importlib.metadata.version('pypng'),'RGB_channel_bit_depth':16,
        'image_size':[w,h],'channel_values_matching_expected':int(actual.size),'channel_values_checked':int(actual.size),
        'maximum_rounding_error_code':error,'display_Y_preservation_error':summary['maximum_display_Y_error'],
        'monochrome_pixel_array_equality_against_0_7':mono_checks,
        'pure_background_pixels':int(pure_background.sum()),'pure_background_max_RGB_code_difference':grey_error,
        'pixels_with_any_channel_255':int(np.count_nonzero(np.any(display==255,axis=-1)&supported)),
        'pixels_with_all_channels_255':int(np.count_nonzero(np.all(display==255,axis=-1)&supported)),
        'human_observers':0,'ground_nights':0}
    if args.check_index:
        directory=Path('src/lightpollution/resources/colour')
        expected=read_json(directory/'provenance.json')['sha256']
        for name,checksum in expected.items():
            staged=subprocess.check_output(['git','show',':'+str(directory/name)])
            assert hashlib.sha256(staged).hexdigest()==checksum,name
        record['Git_index_original_CIE_bytes_checked']=len(expected)
        record['Git_index_original_CIE_bytes_matching']=len(expected)
    write_json(out/'encoding-verification.json',record);print(record)


if __name__=='__main__':main()
