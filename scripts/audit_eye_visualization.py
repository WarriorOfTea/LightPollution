"""Audit the saved 0.6 image; do not change physical results or the renderer."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lightpollution.adapters.storage import file_hash, read_json, write_json
from lightpollution.application.eye import sample_eye_field
from lightpollution.interfaces.eye_view import projection, _srgb
from lightpollution.models.eye import REFERENCE_SP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/eye-model-20261111/city'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-visualization-research-20261111'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    names=['result.json','eye-display.json','eye-view.png']
    before={name:file_hash(args.source/name) for name in names}
    result=read_json(args.source/'result.json');display=read_json(args.source/'eye-display.json')
    assert display['display_version']=='monochrome_threshold_illustration_1'
    view=display['view'];w,h=int(view['width']),int(view['height'])
    az,alt,_=projection(view)
    photopic,scotopic,physical=sample_eye_field(result['eye']['field'],az,alt)
    # Reconstruct only the documented background display mapping, without
    # moving any star or recomputing transfer/thresholds.
    encoded=255*_srgb(.0006+.003*np.log1p(scotopic/(2e-4*REFERENCE_SP)))
    background=np.uint8(np.rint(encoded));levels,counts=np.unique(background,return_counts=True)
    field=np.asarray(Image.open(args.source/'eye-view.png').convert('RGB'))[80:80+h,:w]
    assert field.shape==(h,w,3) and display['supported_fraction']==1
    q=np.asarray([s['flux_over_threshold'] for s in display['displayed_star_rows']])
    actual=Image.fromarray(field[:,:,0]);blank=Image.fromarray(background)
    resize=[]
    for width in [w,900,390]:
        height=round(h*width/w)
        a=np.asarray(actual.resize((width,height),Image.Resampling.LANCZOS),float)
        b=np.asarray(blank.resize((width,height),Image.Resampling.LANCZOS),float)
        peaks=[]
        for star in display['displayed_star_rows']:
            x=int((star['x_px']+.5)*width/w);y=int((star['y_px']+.5)*height/h)
            patch=(a-b)[max(0,y-1):min(height,y+2),max(0,x-1):min(width,x+2)]
            peaks.append(float(np.max(patch)))
        resize.append({'width_px':width,'height_px':height,
            'field_max_code':int(a.max()),'stellar_local_peak_excess_code_quantiles':np.quantile(peaks,[0,.5,.9,1]).tolist(),
            'star_positions_with_no_positive_code_in_3x3':sum(p<=0 for p in peaks),
            'star_positions_checked':len(peaks),
            'not_a_human_detection_test':True,'not_a_measurement_of_Telegram_resampling':True})
    grid=result['eye']['field'];axis=np.asarray(grid['azimuth_deg'])
    # A derivative mismatch follows from piecewise bilinear interpolation.
    # It is a numerical diagnostic, not a measured error of the sky model.
    jumps=[]
    for altitude in [45.,70.]:
        for angle in axis:
            eps=.001
            _,s,_=sample_eye_field(grid,np.array([angle-eps,angle,angle+eps])%360,np.full(3,altitude))
            left=(s[1]-s[0])/eps;right=(s[2]-s[1])/eps
            jumps.append({'azimuth_deg':float(angle),'altitude_deg':altitude,
                'left_derivative_cd_m2_per_degree':float(left),
                'right_derivative_cd_m2_per_degree':float(right),
                'slope_jump_cd_m2_per_degree':float(right-left)})
    record={'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'baseline_commit':'f69a8c83f8eef3e026b244bc3ee13761d8cc2546','source_directory':str(args.source),
        'source_sha256':before,'source_files_unchanged':True,'physical_model_recomputed':False,
        'new_eye_or_camera_model_implemented':False,'view':view,'field_pixels':w*h,
        'grid_azimuth_deg':grid['azimuth_deg'],'grid_altitude_deg':grid['altitude_deg'],
        'background_scotopic_cd_m2_range':[float(scotopic.min()),float(scotopic.max())],
        'background_photopic_cd_m2_range':[float(photopic.min()),float(photopic.max())],
        'background_float_display_code_range':[float(encoded.min()),float(encoded.max())],
        'background_quantized_levels':levels.tolist(),'background_pixels_per_level':counts.tolist(),
        'identical_RGB_pixels':int(np.all(field[:,:,0,None]==field,axis=2).sum()),
        'saturated_field_pixels':int(np.any(field==255,axis=2).sum()),'field_max_code':int(field.max()),
        'displayed_stars':len(q),'catalogue_stars_in_view':display['catalogue_stars_in_view'],
        'stellar_flux_over_threshold_quantiles':np.quantile(q,[0,.5,1]).tolist(),
        'resize_diagnostics':resize,'angular_grid_derivative_diagnostics':jumps,
        'independent_held_out_transfer_rays':0,'independent_visual_observers':0,
        'angular_field_accuracy_established':False,'perceptual_brightness_accuracy_established':False}
    assert before=={name:file_hash(args.source/name) for name in names}
    write_json(out/'image-audit.json',record)
    fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    image=axs[0,0].imshow(scotopic,cmap='cividis');fig.colorbar(image,ax=axs[0,0],label='Скотопические кд/м²')
    axs[0,0].set_title('Линейный фон: плавный диапазон света');axs[0,0].set_xlabel('Пиксель поля')
    axs[0,1].imshow(background,vmin=19,vmax=23,cmap='gray')
    axs[0,1].set_title(f'Те же данные после 8-bit: всего {len(levels)} уровня\nКонтраст усилен для диагностики, не вид глазом')
    y=np.arange(h);axs[1,0].plot(y,encoded[:,w//2],label='До округления')
    axs[1,0].step(y,background[:,w//2],where='mid',label='PNG после округления')
    axs[1,0].set_xlabel('Вертикальная координата центрального столбца, px')
    axs[1,0].set_ylabel('Код sRGB 0…255');axs[1,0].legend();axs[1,0].set_title('Происхождение видимых ступеней')
    widths=[r['width_px'] for r in resize];peak=[r['field_max_code'] for r in resize]
    axs[1,1].plot(widths,peak,'o-');axs[1,1].invert_xaxis()
    axs[1,1].set_xlabel('Ширина изображения после уменьшения, px');axs[1,1].set_ylabel('Максимальный код поля')
    axs[1,1].set_title('Яркие точки слабеют при уменьшении\nLanczos в sRGB; не измерение Telegram или зрения')
    fig.suptitle('Аудит сохранённой версии 0.6: физическая модель не изменена',fontsize=15)
    fig.savefig(out/'visualization-diagnostics.png',dpi=130);plt.close(fig)
    print('Audit:',len(levels),'background levels;',len(q),'stars;',record['saturated_field_pixels'],'clipped pixels.')
    print('Resize maxima:',[(r['width_px'],r['field_max_code']) for r in resize])


if __name__=='__main__':main()
