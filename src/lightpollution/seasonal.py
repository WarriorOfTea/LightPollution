"""Matched monthly source support and a fixed cross-month radiance-area scale."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import COMPOSITES, good_mask, load_region, save_region
from .download import sha256, write_manifest
from .periods import CompositePeriod
from .physics import EARTH_RADIUS_M, Observer, cell_areas, unit_position


def prepare_seasonal(region_paths: list[Path], output: Path, reference: str,
                     observer: Observer = Observer(), exclude_radius_m: float = 3000,
                     minimum_common_fraction: float = .8, composite: str = 'NearNadir') -> dict:
    if (len(region_paths)<2 or not np.isfinite(exclude_radius_m) or exclude_radius_m<0
            or not 0<=minimum_common_fraction<=1):
        raise ValueError('Need at least two regions and valid matching thresholds.')
    if composite not in COMPOSITES:
        raise ValueError('Unknown angular composite.')
    regions=[load_region(path) for path in region_paths]
    bounds=regions[0][1]
    periods=[]
    for arrays,b,metadata in regions:
        if b!=bounds:
            raise ValueError('Seasonal regions must have identical bounds/grid.')
        period=CompositePeriod(metadata['year'],metadata.get('month'))
        if period.month is None or metadata.get('product')!='VNP46A3.002':
            raise ValueError('Seasonal matching requires monthly VNP46A3.002 regions.')
        if metadata.get('source_power_multiplier') is not None:
            raise ValueError('Do not apply seasonal matching twice to an already matched region.')
        periods.append(period.label)
    if len(set(periods))!=len(periods) or reference not in periods:
        raise ValueError('Periods must be unique and include the explicit reference.')
    key=composite+'_Composite_Snow_Free'
    ref_index=periods.index(reference)
    lat,lon=np.meshgrid(bounds.latitude,bounds.longitude,indexing='ij')
    distance=2*EARTH_RADIUS_M*np.arcsin(np.clip(np.linalg.norm(
        unit_position(lat,lon)-unit_position(observer.latitude_deg,observer.longitude_deg),axis=-1)/2,0,1))
    domain=np.isin(regions[ref_index][0]['Land_Water_Mask'],(0,1,5)) & (distance>=exclude_radius_m)
    common=domain.copy()
    counts=[]
    for arrays,_,_ in regions:
        mask=good_mask(arrays[key],arrays[key+'_Quality'],arrays[key+'_Num'])
        land=np.isin(arrays['Land_Water_Mask'],(0,1,5))
        counts.append(int(np.count_nonzero(domain & mask & land)))
        common &= mask & land
    denominator=int(domain.sum())
    n=int(common.sum())
    fraction=n/denominator if denominator else 0.
    if not n or fraction<minimum_common_fraction:
        raise ValueError(f'Insufficient common monthly support: {n}/{denominator} eligible land pixels '
                         f'({fraction:.4%}); required {minimum_common_fraction:.1%}. Missing data are not darkness.')
    areas=cell_areas(bounds.latitude)[:,None]
    sums=[float(np.sum(np.where(common,arrays[key],0)*areas)) for arrays,_,_ in regions]
    if any(not np.isfinite(value) or value<=0 for value in sums):
        raise ValueError('A matched month has no positive finite radiance-area sum.')
    reference_sum=sums[ref_index]
    selection={'observer':asdict(observer),'exclude_radius_m':exclude_radius_m,'include_water':False}
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'common-mask.npz',common_mask=common,**asdict(bounds))
    mask_hash=sha256(output/'common-mask.npz')
    summaries=[]
    for path,(arrays,_,metadata),period,value,count in zip(region_paths,regions,periods,sums,counts,strict=True):
        selected={name:np.array(array,copy=True) for name,array in arrays.items()}
        selected[key+'_Quality'][~common]=1
        multiplier=value/reference_sum
        target=output/period/'source-grid.npz'
        meta={**metadata,'source_power_multiplier':multiplier,'seasonal_selection':selection,
              'source_composite':composite,
              'normalisation_reference_period':reference,'common_mask_sha256':mask_hash,
              'radiance_area_sum':value,'reference_radiance_area_sum':reference_sum}
        save_region(target,selected,bounds,meta)
        summaries.append({'period':period,'input':str(path),'input_sha256':sha256(path),
                          'accepted_on_reference_domain':count,'radiance_area_sum':value,
                          'source_power_multiplier':multiplier,'prepared_region':str(target)})
    report={'reference_period':reference,'bounds':asdict(bounds),'selection':selection,'composite':composite,
            'roi_pixels':int(common.size),'reference_land_domain_pixels':denominator,
            'common_pixels':n,'common_fraction_of_reference_land':fraction,
            'minimum_common_fraction':minimum_common_fraction,'common_mask_sha256':mask_hash,
            'radiance_units':'nW/(cm2 sr); sum uses spherical cell area in m2',
            'periods':summaries,'conversion_to_upward_power':'relative proxy, not physical inversion',
            'assumption':'Same DNB-to-V and angular-emission mapping across compared months; unvalidated.',
            'not_modelled':['monthly snow reflection','hourly light switching','measured atmosphere']}
    write_manifest(output/'seasonal.json',report)
    return report
