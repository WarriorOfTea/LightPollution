"""Numerical ledger and diagnostics for the frozen 0.7 eye-view case."""

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
from lightpollution.models.eye import REFERENCE_SP
from lightpollution.models.eye_display import (apply_ward1997, quantize, srgb_encode,
    tone_linear, ward1997_curve)
from lightpollution.models.eye_optics import deposit_star, optical_mixture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/eye-rendering-20261111'))
    args=parser.parse_args();out=args.output
    criteria=read_json(out/'predeclared-criteria.json')
    result=read_json(out/'city/result.json')
    display=result['eye']['display'];profile=display['display_profile']
    baseline=Path('artifacts/eye-model-20261111/city')
    before={n:file_hash(baseline/n) for n in ['result.json','eye-display.json','eye-view.png']}
    data=np.load(out/'city/eye-linear.npz',allow_pickle=False)
    background=data['background_scotopic_cd_m2'];support=data['supported']
    encoded=srgb_encode(tone_linear(data['appearance_scotopic_cd_m2'],profile))
    actual16=np.asarray(Image.open(out/'city/eye-view-16bit.png'),dtype=float)
    error16=abs(actual16[support]-65535*encoded[support])
    blank=srgb_encode(tone_linear(background,profile))
    error8=quantize(blank).astype(float)-255*blank
    # Pure stage-1 control: keep the former mapping and physical threshold table.
    legacy=srgb_encode(.0006+.003*np.log1p(background/(2e-4*REFERENCE_SP)))
    legacy_error=quantize(legacy).astype(float)-255*legacy
    flux_rows=[]
    mixture=optical_mixture(6.)
    for width in criteria['resolutions_px']:
        for px,py in criteria['subpixel_phases']:
            for correlation in [0.,.35]:
                buffer=np.zeros((width,width),dtype=float)
                ledger=deposit_star(buffer,width/2+px,width/2+py,1.2345e-8,
                    np.deg2rad(60)/width,mixture,correlation=correlation)
                flux_rows.append({'width_px':width,'subpixel_phase':[px,py],'projection_correlation':correlation,
                    'relative_flux_error':float(abs(buffer.sum()/1.2345e-8-1)),
                    'float32_relative_flux_error':float(abs(buffer.astype(np.float32).sum(dtype=np.float64)/1.2345e-8-1)),**ledger})
    # A synthetic foveal field with isolated known-flux stars. These are already
    # one-degree area means; raw high-resolution PSF peaks are not adaptation.
    omega_degree=(np.pi/180)**2
    fluxes=1e-7*10**(-.4*np.arange(0,8))
    foveal=np.full((40,60),.002)
    for index,e in enumerate(fluxes):foveal[20,5+index*6]+=e/omega_degree
    curve=ward1997_curve(foveal,profile)
    ferwerda=srgb_encode(tone_linear(foveal,profile))
    ward=srgb_encode(apply_ward1997(foveal,curve,profile))
    comparison={'input_basis':'synthetic one-degree foveal scotopic area means; low-light comparator extension',
        'star_fluxes_scotopic_lux':fluxes.tolist(),'solid_angle_per_cell_sr':omega_degree,
        'ferwerda_background_code':float(255*ferwerda[0,0]),
        'ward_background_code':float(255*ward[0,0]),'ward_curve':curve,
        'ferwerda_stellar_cell_codes':[float(255*ferwerda[20,5+i*6]) for i in range(8)],
        'ward_stellar_cell_codes':[float(255*ward[20,5+i*6]) for i in range(8)],
        'selected':'ferwerda1996 fixed profile + explicit shoulder',
        'selection_reason':'fixed adaptation and mapping can be held across city-on/off; no histogram-driven re-exposure',
        'human_preference_or_accuracy_test':False,'full_pcond_features_implemented':False}
    patch=result['eye']['field']['view_patch'];validation=patch['validation']
    numeric=[r['quick_standard_relative_errors_P_S_V'] for r in validation['rows']
             if 'quick_standard_relative_errors_P_S_V' in r]
    summary={'completed_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'stage1_display_only_control':{'threshold_table_reused_without_modification':True,
            'legacy_mapping_dither_max_abs_code_error':float(abs(legacy_error[support]).max()),
            'legacy_mapping_dither_mean_code_error':float(legacy_error[support].mean())},
        'encoding':{'maximum_16bit_code_rounding_error':float(error16.max()),
            'maximum_background_8bit_code_dither_error':float(abs(error8[support]).max()),
            'mean_background_8bit_code_dither_error':float(error8[support].mean()),
            'background_8bit_levels':np.unique(quantize(blank)[support]).tolist(),
            'background_16bit_levels':int(len(np.unique(quantize(blank,16)[support]))),
            'maximum_float_encoded_code':float(255*encoded[support].max()),
            'tone_values_equal_one':int(np.count_nonzero(encoded[support]>=1)),
            'screen_transfer_count':1},
        'flux_validation':{'checks':len(flux_rows),'max_relative_error':max(r['relative_flux_error'] for r in flux_rows),
            'max_float32_relative_flux_error':max(r['float32_relative_flux_error'] for r in flux_rows),
            'target_relative_error':criteria['flux_conservation_relative_error_target'],'rows':flux_rows},
        'optics_fits':[{k:v for k,v in optical_mixture(d).items() if k not in ('sigma_deg','weight')} for d in [2.,4.,6.]],
        'refinement':patch['refinement'],'independent_angular_validation':validation,
        'numerical_profiles':{'rays':len(numeric),'max_relative_difference_P_S_V':np.max(numeric,axis=0).tolist(),
            'fixed_source_coefficient':True,'is_a_physical_accuracy_bound':False},
        'tone_operator_comparison':comparison,
        'city_display':{k:display[k] for k in ['displayed_stars','catalogue_stars_in_view',
            'unavailable_stellar_thresholds_in_view','supported_fraction','field_pixel_max_code','saturated_field_pixels']},
        'baseline_sha256':before,'baseline_unchanged':before=={n:file_hash(baseline/n) for n in before},
        'new_independent_human_observers':0,'new_night_photometry':0,'physical_error_bound':None,
        'perceptual_brightness_or_colour_accuracy_established':False}
    assert summary['encoding']['maximum_16bit_code_rounding_error']<=.500001
    assert summary['encoding']['maximum_background_8bit_code_dither_error']<1
    assert abs(summary['encoding']['mean_background_8bit_code_dither_error'])<.002
    assert summary['flux_validation']['max_relative_error']<=.01
    assert summary['baseline_unchanged']
    assert before==read_json(Path('artifacts/eye-visualization-research-20261111/image-audit.json'))['source_sha256']
    summary['baseline_matches_previous_committed_audit']=True
    write_json(out/'numerical-validation.json',summary)
    write_json(out/'tone-comparison.json',comparison)
    fig,ax=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
    y=np.arange(background.shape[0]);x=background.shape[1]//2
    ax[0,0].plot(y,255*blank[:,x],label='До квантования')
    # Spatial averaging is for diagnosis only; it is not applied to sky radiance.
    dithered=quantize(blank)
    blocks=np.arange(0,len(y)-31,32)
    ax[0,0].plot(blocks+15.5,[dithered[b:b+32,x-16:x+16].mean() for b in blocks],'.',label='8 бит, среднее 32×32')
    ax[0,0].set(xlabel='Вертикальная координата, px',ylabel='Код sRGB',title='Плавный профиль и ограниченный dithering')
    ax[0,0].legend()
    rows=validation['rows']
    err=np.asarray([r['relative_errors_P_S_V'] for r in rows])
    ax[0,1].plot(np.arange(24),100*err[:,0],'o-',label='Город: фотопический')
    ax[0,1].plot(np.arange(24),100*err[:,1],'.-',label='Город: скотопический')
    ax[0,1].axhline(2,color='red',ls='--',label='Цель интерполяции 2%')
    ax[0,1].set(xlabel='Независимый луч',ylabel='Ошибка относительно прямого расчёта, %',title='Геометрическая проверка; не полевая точность')
    ax[0,1].legend(fontsize=8)
    f=np.geomspace(.03,200,400)
    from lightpollution.models.eye_optics import watson_mtf
    predicted=np.exp(-2*np.pi**2*f[:,None]**2*np.asarray(mixture['sigma_deg'])[None,:]**2)@np.asarray(mixture['weight'])
    ax[1,0].semilogx(f,watson_mtf(f,6),label='Watson 2013, D=6 мм')
    ax[1,0].semilogx(f,predicted,'--',label='Положительная смесь Gaussian')
    ax[1,0].set(xlabel='Частота, циклов/градус',ylabel='MTF',title='Оптическое ядро; сумма энергии = 1')
    ax[1,0].legend()
    ax[1,1].plot(range(8),comparison['ferwerda_stellar_cell_codes'],'o-',label='Ferwerda, фиксированный профиль')
    ax[1,1].plot(range(8),comparison['ward_stellar_cell_codes'],'o-',label='Ward 1997, узкий диапазон → linear')
    ax[1,1].set(xlabel='Δm от первой звезды; каждый шаг ослабляет поток в 2,512 раза',ylabel='Код клетки 1°',title='Сравнение переноса, не тест наблюдателей')
    ax[1,1].legend(fontsize=8)
    fig.savefig(out/'numerical-diagnostics.png',dpi=140);plt.close(fig)
    print('Validation:',len(flux_rows),'flux checks; max',summary['flux_validation']['max_relative_error'],
          '; heldout',validation['supported_rays'],'max',validation['max_relative_error'],flush=True)


if __name__=='__main__':main()
