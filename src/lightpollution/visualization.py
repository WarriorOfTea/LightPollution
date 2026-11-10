"""Scientific plots; display normalisation never changes numerical model output."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np

from .data import COMPOSITES, Bounds, good_mask
from .physics import Observer
from .periods import CompositePeriod


def source_map(arrays: dict, bounds: Bounds, path: Path, observer: Observer = Observer(), year: int = 2025,
               month: int | None = None):
    period=CompositePeriod(year,month)
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout="constrained")
    all_values=[]
    for prefix in COMPOSITES:
        key=prefix+"_Composite_Snow_Free"
        valid=good_mask(arrays[key],arrays[key+"_Quality"],arrays[key+"_Num"])
        all_values.extend(arrays[key][valid & (arrays[key]>0)])
    vmax=max(1,float(np.percentile(all_values,99.5))) if all_values else 1
    cmap=plt.colormaps['magma'].copy()
    cmap.set_bad('#a8adb4')
    for ax,prefix in zip(axes,COMPOSITES,strict=True):
        key=prefix+"_Composite_Snow_Free"
        valid=good_mask(arrays[key],arrays[key+"_Quality"],arrays[key+"_Num"])
        image=ax.imshow(np.ma.masked_where(~valid,arrays[key]),origin="upper",
                        extent=(bounds.west,bounds.east,bounds.south,bounds.north),
                        cmap=cmap,norm=SymLogNorm(linthresh=0.5,vmin=0,vmax=vmax),aspect=1/np.cos(np.deg2rad((bounds.north+bounds.south)/2)))
        ax.plot(30.315,59.94,'c+',markersize=9,label="Центр Петербурга")
        ax.plot(observer.longitude_deg,observer.latitude_deg,'wo',markersize=4,label="Наблюдатель")
        ax.set(title=prefix,xlabel="Долгота, °E",ylabel="Широта, °N")
        ax.legend(fontsize=7,loc='lower right')
        ax.grid(alpha=0.15)
    fig.colorbar(image,ax=axes,shrink=0.75,label='Яркость VIIRS, nW/(cm² sr); шкала symlog')
    fig.suptitle(f"NASA {period.product}.002, {period.label}, без снега. Серый — данные не приняты по качеству",fontsize=12)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def sky_plot(azimuths,altitudes,radiance,path: Path,title: str):
    maximum=float(np.max(radiance))
    if not np.isfinite(maximum) or maximum<=0:
        raise ValueError("A positive, finite sky map is required for a logarithmic figure.")
    relative=radiance/maximum
    fig=plt.figure(figsize=(13,6.2),layout="constrained")
    ax=fig.add_subplot(1,2,1,projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    az_closed=np.r_[azimuths,360.0]
    values=np.column_stack((relative,relative[:,0]))
    theta,radius=np.meshgrid(np.deg2rad(az_closed),90-np.asarray(altitudes))
    low=max(1e-6,min(0.01,float(relative[relative>0].min())))
    im=ax.contourf(theta,radius,np.maximum(values,low),levels=np.geomspace(low,1,31),cmap='inferno',norm=LogNorm(low,1))
    ax.set_ylim(0,90)
    ax.set_facecolor('#b7bdc5')
    ax.set_thetagrids(np.arange(0,360,45),['С','СВ','В','ЮВ','Ю','ЮЗ','З','СЗ'])
    ax.set_yticks([0,30,60,89],['90°','60°','30°','1°'])
    ax.tick_params(labelsize=9)
    ax.set_title("Искусственное свечение: направления неба",pad=22,fontsize=11)
    ticks=np.geomspace(low,1,5)
    bar=fig.colorbar(im,ax=ax,shrink=0.7,ticks=ticks,label='Относительно максимума на этой карте')
    bar.ax.set_yticklabels([f'{tick:.3g}' for tick in ticks])
    profile=fig.add_subplot(1,2,2)
    for target in (5,15,45):
        row=int(np.argmin(abs(np.asarray(altitudes)-target)))
        profile.plot(az_closed,np.r_[relative[row],relative[row,0]],label=f'Высота {altitudes[row]:g}°')
    profile.set(xlim=(0,360),ylim=(0,1.05),xlabel='Азимут, ° (север = 0°, восток = 90°)',ylabel='Яркость / максимум всей карты')
    profile.set_xticks(np.arange(0,361,45))
    profile.grid(alpha=0.25)
    profile.legend()
    profile.set_title('Рассчитанные значения; без сглаживания пиков',fontsize=11)
    fig.suptitle(title,fontsize=13)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)
