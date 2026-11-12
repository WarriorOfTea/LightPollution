"""Observer sky: catalogue astrometry, extinction and an explicit display scenario.

The radiative-transfer field supplies the *shape*. Its conversion to V-band sky
brightness is a user assumption until ground photometry supplies a calibration.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from importlib.resources import files
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .physics import Atmosphere, EARTH_RADIUS_M, Observer


@dataclass(frozen=True)
class DisplayScenario:
    artificial_zenith_mag: float
    natural_mag: float = 21.6
    dark_limiting_mag: float = 6.5

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Display magnitudes must be finite.")
        if not 12 <= self.artificial_zenith_mag <= 30:
            raise ValueError("Artificial zenith scenario must be in [12,30] mag/arcsec2.")


def load_stars() -> tuple[dict[str, np.ndarray], dict]:
    directory = files("lightpollution").joinpath("catalogues")
    with directory.joinpath("bsc5.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    strings = {"name"}
    stars = {key: np.asarray([r[key] for r in rows], dtype=str if key in strings else float)
             for key in rows[0]}
    metadata = json.loads(directory.joinpath("bsc5-provenance.json").read_text())
    return stars, metadata


def horizontal_stars(stars: dict, observer: Observer, timestamp: str, *,
                     allow_moon: bool = False, allow_twilight: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    # Require explicit UTC: an accidentally interpreted local time moves the sky.
    if not timestamp.endswith("Z"):
        raise ValueError("Supply an ISO UTC timestamp ending in Z.")
    instant = Time(timestamp[:-1], scale="utc")
    location = EarthLocation.from_geodetic(observer.longitude_deg*u.deg,
                                          observer.latitude_deg*u.deg, observer.height_m*u.m)
    # Tangential proper-motion propagation on the J2000 sphere. No distance or
    # radial velocity is invented. Perspective acceleration/parallax are omitted.
    ra, dec = np.deg2rad(stars["ra_deg"]), np.deg2rad(stars["dec_deg"])
    vectors = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    east = np.column_stack((-np.sin(ra), np.cos(ra), np.zeros_like(ra)))
    north = np.column_stack((-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)))
    years = (instant.jd-Time("J2000").jd)/365.25
    conversion = np.deg2rad(1/3600)*years
    vectors += conversion*(stars["pm_ra_cosdec_arcsec_yr"][:,None]*east +
                           stars["pm_dec_arcsec_yr"][:,None]*north)
    vectors /= np.linalg.norm(vectors, axis=1)[:,None]
    coordinates = SkyCoord(ra=np.arctan2(vectors[:,1],vectors[:,0])*u.rad,
                           dec=np.arcsin(vectors[:,2])*u.rad, frame=FK5(equinox=Time("J2000")))
    frame = AltAz(obstime=instant, location=location, pressure=0*u.hPa)
    # Reproducible offline transformations, using the lock-pinned IERS tables.
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        horizontal = coordinates.transform_to(frame)
        sun = get_sun(instant).transform_to(frame)
        moon = get_body("moon", instant, location).transform_to(frame)
    metadata = {"utc": timestamp, "sun_altitude_deg": float(sun.alt.deg),
                "moon_altitude_deg": float(moon.alt.deg), "refraction": False,
                "proper_motion": "J2000 tangent-vector propagation; no perspective acceleration or stellar parallax",
                "iers": "Offline bundled tables from lock-pinned astropy-iers-data",
                "earth_geometry": "WGS84 astrometry; spherical Earth in transfer solver"}
    if sun.alt.deg > -18 and not allow_twilight:
        raise ValueError("The Sun is above -18 degrees: twilight is outside this renderer's model.")
    if moon.alt.deg > 0 and not allow_moon:
        raise ValueError("The Moon is above the horizon: lunar sky brightness is not modelled.")
    return horizontal.az.deg, horizontal.alt.deg, metadata


def sample_sky(azimuths, altitudes, radiance, azimuth, altitude):
    """Periodic bilinear interpolation in *linear* radiance, with no overshoot."""
    azimuths, altitudes, radiance = map(np.asarray, (azimuths,altitudes,radiance))
    if radiance.shape != (len(altitudes),len(azimuths)) or len(azimuths)<2 or len(altitudes)<2:
        raise ValueError("Sky grid shape does not match its coordinate arrays.")
    if (azimuths[0] != 0 or np.any(np.diff(azimuths)<=0) or azimuths[-1]>=360
            or np.any(np.diff(altitudes)<=0) or altitudes[0]<0 or altitudes[-1]!=90
            or not np.all(np.isfinite(radiance)) or np.any(radiance<0)):
        raise ValueError("Invalid ordered, finite hemispheric sky grid.")
    az, alt = np.broadcast_arrays(np.mod(azimuth,360), np.asarray(altitude))
    if np.any(~np.isfinite(az)) or np.any(~np.isfinite(alt)) or np.any((alt<altitudes[0]) | (alt>90)):
        raise ValueError("Direction is outside the computed sky field.")
    closed = np.r_[azimuths,360.0]
    col = np.searchsorted(closed,az,side="right")-1
    row = np.clip(np.searchsorted(altitudes,alt,side="right")-1,0,len(altitudes)-2)
    x = (az-closed[col])/(closed[col+1]-closed[col])
    y = (alt-altitudes[row])/(altitudes[row+1]-altitudes[row])
    next_col = (col+1)%len(azimuths)
    return ((1-y)*((1-x)*radiance[row,col]+x*radiance[row,next_col]) +
            y*((1-x)*radiance[row+1,col]+x*radiance[row+1,next_col]))


def background_ratio(relative_to_zenith, scenario: DisplayScenario, artificial=True):
    """Total background in units of the assumed natural V-band sky background."""
    scale = 10**(-0.4*(scenario.artificial_zenith_mag-scenario.natural_mag))
    return 1 + np.asarray(relative_to_zenith)*scale if artificial else np.ones_like(relative_to_zenith)


def limiting_magnitude(background, scenario: DisplayScenario):
    # Background-noise-limited F_threshold ∝ sqrt(B) at fixed angular PSF/SNR.
    # This is an explicit illustration rule, not a validated model of the eye.
    return scenario.dark_limiting_mag-1.25*np.log10(background)


def stellar_extinction(altitudes, observer: Observer, atmosphere: Atmosphere):
    altitudes = np.asarray(altitudes)
    if np.any((altitudes<0)|(altitudes>90)):
        raise ValueError("Extinction requires above-horizon directions.")
    radius = EARTH_RADIUS_M+observer.height_m
    projection = radius*np.sin(np.deg2rad(altitudes))
    length = -projection+np.sqrt(projection**2+(EARTH_RADIUS_M+atmosphere.top_m)**2-radius**2)
    tau = atmosphere.optical_depth_radial(radius**2,(EARTH_RADIUS_M+atmosphere.top_m)**2,
                                          length**2,order=64)
    return 2.5/np.log(10)*tau


def dome_coordinates(azimuth_deg, altitude_deg):
    """View looking up: north up, east LEFT, west right (no terrestrial-map mirror)."""
    radius = (90-np.asarray(altitude_deg))/90
    az = np.deg2rad(azimuth_deg)
    return -radius*np.sin(az), radius*np.cos(az)


def _tone(linear):
    compressed = 1-np.exp(-np.maximum(linear,0))
    return np.where(compressed<=0.0031308,12.92*compressed,1.055*compressed**(1/2.4)-0.055)


def camera_basis(azimuth_deg, altitude_deg):
    if not np.isfinite(azimuth_deg) or not 0<=altitude_deg<=90:
        raise ValueError('Camera needs finite azimuth and altitude in [0,90].')
    az,alt=np.deg2rad([azimuth_deg,altitude_deg])
    forward=np.array([np.cos(alt)*np.sin(az),np.cos(alt)*np.cos(az),np.sin(alt)])
    right=np.array([np.cos(az),-np.sin(az),0.])
    up=np.cross(right,forward)
    return forward,right,up


def _stamp_stars(image,px,py,apparent,keep):
    height,width=image.shape
    for index in np.flatnonzero(keep):
        sigma=1.1
        cx,cy=px[index],py[index]
        x0,x1=max(0,int(cx)-5),min(width,int(cx)+7)
        y0,y1=max(0,int(cy)-5),min(height,int(cy)+7)
        xx,yy=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
        image[y0:y1,x0:x1]+=8.0*10**(-.4*apparent[index])*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma**2))


def _perspective(path,az_grid,alt_grid,field,stars,az,alt,extinction,scenario,observer,timestamp,
                 camera_azimuth,camera_altitude,horizontal_fov):
    if not 20<=horizontal_fov<=140:
        raise ValueError('Perspective horizontal FOV must be in [20,140] degrees.')
    width,height=2000,1125
    forward,right,up=camera_basis(camera_azimuth,camera_altitude)
    limit_x=np.tan(np.deg2rad(horizontal_fov)/2);limit_y=limit_x*height/width
    x,y=np.meshgrid(np.linspace(-limit_x,limit_x,width),np.linspace(limit_y,-limit_y,height))
    rays=forward+x[...,None]*right+y[...,None]*up
    rays/=np.linalg.norm(rays,axis=-1)[...,None]
    pixel_az=np.rad2deg(np.arctan2(rays[...,0],rays[...,1]))%360
    pixel_alt=np.rad2deg(np.arcsin(rays[...,2]))
    relative=sample_sky(az_grid,alt_grid,field,pixel_az,np.maximum(pixel_alt,0))
    background=background_ratio(relative,scenario)
    image=.002*background
    star_bg=background_ratio(sample_sky(az_grid,alt_grid,field,az,np.maximum(alt,0)),scenario)
    apparent=stars['vmag']+extinction
    azr,altr=np.deg2rad(az),np.deg2rad(alt)
    directions=np.column_stack((np.cos(altr)*np.sin(azr),np.cos(altr)*np.cos(azr),np.sin(altr)))
    depth=directions@forward
    sx=np.divide(directions@right,depth,out=np.zeros_like(depth),where=depth>0)
    sy=np.divide(directions@up,depth,out=np.zeros_like(depth),where=depth>0)
    keep=(depth>0)&(alt>=0)&(np.abs(sx)<=limit_x)&(np.abs(sy)<=limit_y)&(apparent<=limiting_magnitude(star_bg,scenario))
    px=(sx+limit_x)/(2*limit_x)*(width-1);py=(limit_y-sy)/(2*limit_y)*(height-1)
    _stamp_stars(image,px,py,apparent,keep)
    rgb=np.repeat(_tone(image)[...,None],3,axis=2)
    rgb[pixel_alt<0]=np.array([7,11,18])/255
    fig,ax=plt.subplots(figsize=(14,9.2),facecolor='#070b12')
    fig.subplots_adjust(left=.015,right=.985,top=.84,bottom=.14)
    ax.imshow(rgb);ax.axis('off')
    fig.text(.5,.95,'Взгляд из точки наблюдения',ha='center',fontsize=21)
    fig.text(.5,.902,f'{observer.latitude_deg:.3f}° N · {observer.longitude_deg:.3f}° E · {timestamp}',ha='center',fontsize=11,color='#b6c4d4')
    fig.text(.5,.862,f'Азимут {camera_azimuth:g}° · высота взгляда {camera_altitude:g}° · поле {horizontal_fov:g}° · {keep.sum()} звёзд по правилу контраста',ha='center',fontsize=10,color='#8da0b8')
    fig.text(.5,.10,f'Сценарий: искусственный фон зенита {scenario.artificial_zenith_mag:g} mag/arcsec² задан вручную.',ha='center',fontsize=11,color='#e2c68f')
    fig.text(.5,.055,'Засветка по источникам NASA и звёзды BSC5. Геометрический горизонт; рельеф и здания не добавлены.',ha='center',fontsize=9,color='#8da0b8')
    fig.savefig(path,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    return {'azimuth_deg':camera_azimuth,'altitude_deg':camera_altitude,'horizontal_fov_deg':horizontal_fov,
            'projection':'rectilinear pinhole','visible_catalogue_rows':int(keep.sum())}


def _draw_dome(ax, az_grid, alt_grid, field, stars, az, alt, extinction, scenario, artificial, size=1200):
    axis = np.linspace(-1.015,1.015,size)
    x,y = np.meshgrid(axis,axis[::-1])
    radius = np.hypot(x,y)
    inside = radius<=1
    pixel_alt = np.maximum(0,90*(1-radius))
    pixel_az = np.rad2deg(np.arctan2(-x,y))%360
    relative = sample_sky(az_grid,alt_grid,field,pixel_az,pixel_alt)
    background = background_ratio(relative,scenario,artificial)
    image = 0.002*background
    star_relative = sample_sky(az_grid,alt_grid,field,az,np.maximum(alt,0))
    star_bg = background_ratio(star_relative,scenario,artificial)
    apparent = stars["vmag"]+extinction
    keep = (alt>=0)&(apparent<=limiting_magnitude(star_bg,scenario))
    sx,sy = dome_coordinates(az,alt)
    px,py = (sx-axis[0])/(axis[-1]-axis[0])*(size-1),(axis[-1]-sy)/(axis[-1]-axis[0])*(size-1)
    # Display PSF: flux follows 10^-0.4m; width is a rendering choice, not seeing.
    _stamp_stars(image,px,py,apparent,keep)
    rgb = np.repeat(_tone(image)[...,None],3,axis=2)
    rgb[~inside] = np.array([7,11,18])/255
    ax.imshow(rgb,extent=(axis[0],axis[-1],axis[0],axis[-1]),origin="upper",interpolation="nearest")
    for angle,label in [(0,"С"),(90,"В"),(180,"Ю"),(270,"З")]:
        tx,ty = dome_coordinates(angle,0)
        ax.text(tx*1.07,ty*1.07,label,color="#bfcbd8",ha="center",va="center",fontsize=11)
    ax.set(xlim=(-1.13,1.13),ylim=(-1.13,1.13),aspect="equal")
    ax.axis("off")
    return keep,star_bg,apparent


def render_observer(sky_path: Path, output: Path, timestamp: str, scenario: DisplayScenario,
                    camera_azimuth=180.,camera_altitude=35.,horizontal_fov=110.) -> dict:
    with np.load(sky_path,allow_pickle=False) as data:
        az_grid,alt_grid,values = data["azimuth_deg"],data["altitude_deg"],data["radiance"]
        transfer = json.loads(str(data["metadata_json"]))
    if transfer.get("provenance",{}).get("input_kind") != "blackmarble":
        raise ValueError("Observer demonstration requires the actual audited NASA source map.")
    if alt_grid[0]!=0 or not np.allclose(values[-1],values[-1,0],rtol=1e-10,atol=0):
        raise ValueError("Observer rendering needs a full 0–90 degree grid with a unique zenith.")
    zenith = float(values[-1,0])
    if not np.isfinite(zenith) or zenith<=0:
        raise ValueError("Positive computed zenith radiance required.")
    field = values/zenith
    observer = Observer(**transfer["observer"])
    atmosphere = Atmosphere(**transfer["atmosphere"])
    source_year = transfer.get("provenance",{}).get("region_metadata",{}).get("year")
    source_label = f"VNP46A4 ({source_year})" if source_year is not None else "VNP46A4 (год не указан)"
    stars,catalogue = load_stars()
    az,alt,astrometry = horizontal_stars(stars,observer,timestamp)
    above = alt>=0
    extinction = np.full(len(alt),np.inf)
    extinction[above] = stellar_extinction(alt[above],observer,atmosphere)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","text.color":"#e2e8f0"})
    fig,axes = plt.subplots(1,2,figsize=(15,8.4),facecolor="#070b12")
    fig.subplots_adjust(left=0.025,right=0.975,bottom=0.14,top=0.83,wspace=0.035)
    dark,_,apparent = _draw_dome(axes[0],az_grid,alt_grid,field,stars,az,alt,extinction,scenario,False)
    visible,bg,_ = _draw_dome(axes[1],az_grid,alt_grid,field,stars,az,alt,extinction,scenario,True)
    axes[0].set_title(f"Без искусственной засветки · {dark.sum()} звёзд",color="#cbd5e1",pad=5,fontsize=13)
    axes[1].set_title(f"Модель засветки · {visible.sum()} звёзд",color="#cbd5e1",pad=5,fontsize=13)
    fig.text(0.5,0.955,"Небо из точки наблюдения",ha="center",fontsize=22,weight="bold")
    fig.text(0.5,0.911,f"{observer.latitude_deg:.3f}° N, {observer.longitude_deg:.3f}° E · {observer.height_m:g} м · {timestamp.replace('T',' ').replace('Z',' UTC')}",ha="center",fontsize=12,color="#b6c4d4")
    fig.text(0.5,0.866,"Весь небесный купол: зенит в центре, горизонт по краю. Вид снизу вверх, восток слева.",ha="center",fontsize=10,color="#8da0b8")
    fig.text(0.5,0.101,f"СЦЕНАРИЙ, НЕ ФОТОГРАФИЯ: искусственный фон в зените {scenario.artificial_zenith_mag:g} mag/arcsec² задан вручную.",ha="center",fontsize=11,color="#e2c68f")
    fig.text(0.5,0.065,f"Форма засветки — перенос света от {source_label}. Звёзды — BSC5/HEASARC; одинаковые экспозиция и атмосфера.",ha="center",fontsize=9,color="#8da0b8")
    fig.text(0.5,0.031,"Число звёзд зависит от условного порога контраста; естественный фон однородный, Луна и Солнце под горизонтом.",ha="center",fontsize=9,color="#8da0b8")
    fig.savefig(output/"observer-sky.png",dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)
    camera=_perspective(output/'observer-perspective.png',az_grid,alt_grid,field,stars,az,alt,extinction,
                         scenario,observer,timestamp,camera_azimuth,camera_altitude,horizontal_fov)
    # Also deliver the scenario by itself, in a large uncluttered view.
    fig,ax = plt.subplots(figsize=(9,10),facecolor="#070b12")
    fig.subplots_adjust(left=.035,right=.965,top=.9,bottom=.14)
    _draw_dome(ax,az_grid,alt_grid,field,stars,az,alt,extinction,scenario,True,size=1600)
    fig.text(.5,.95,"Звёзды и засветка · точка наблюдения",ha="center",fontsize=16)
    fig.text(.5,.91,f"{observer.latitude_deg:.3f}° N · {observer.longitude_deg:.3f}° E · {timestamp.replace('T',' ').replace('Z',' UTC')}",ha="center",fontsize=10,color="#8da0b8")
    fig.text(.5,.09,f"Условный искусственный фон зенита: {scenario.artificial_zenith_mag:g} mag/arcsec².",ha="center",fontsize=10,color="#e2c68f")
    fig.text(.5,.055,"Результат запуска модели; не фотография и не калиброванный прогноз.",ha="center",fontsize=9,color="#8da0b8")
    fig.savefig(output/"observer-view.png",dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)
    table = {"hr": stars["hr"].astype(int),"azimuth_deg":az,"altitude_deg":alt,
             "catalogue_vmag":stars["vmag"],"extinction_mag":extinction,"apparent_vmag":apparent,
             "local_background_natural_units":bg,"threshold_mag":limiting_magnitude(bg,scenario),
             "dark_visible":dark,"scenario_visible":visible}
    with (output/"stars.csv").open("w",newline="") as stream:
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(table)
        writer.writerows(zip(*table.values(),strict=True))
    metadata = {"observer":asdict(observer),"astrometry":astrometry,"display_scenario":asdict(scenario),
                "camera":camera,
                "catalogue":catalogue,"transfer_file":str(sky_path),
                "catalogue_rows":len(az),"above_horizon":int(above.sum()),
                "dark_visible":int(dark.sum()),"scenario_visible":int(visible.sum()),
                "absolute_photometry_calibrated":False,"image_is_photograph":False,
                "visibility_rule":"m_catalogue+extinction <= m_dark-1.25*log10(B_total/B_natural)",
                "tone_mapping":"Fixed monochrome exposure 0.002 per natural background unit; exponential compression then sRGB",
                "projection":"azimuthal equidistant, zenith centre, north top, east left",
                "transfer_summary":transfer,"documentation":"VISUALIZATION.md"}
    (output/"render-summary.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    return metadata


def composite_kstars(sky_path: Path, kstars_directory: Path, scenario: DisplayScenario) -> dict:
    """External layer prototype on an actual KStars export, not an upstream patch.

    KStars glyphs encode magnitude mainly through size; their display intensities
    are not stellar fluxes. This raster compositor therefore makes no star-count
    or human-vision claim. Quantitative selection belongs to render_observer().
    """
    metadata=json.loads((kstars_directory/'kstars-summary.json').read_text())
    with np.load(sky_path,allow_pickle=False) as data:
        az_grid,alt_grid,values=data['azimuth_deg'],data['altitude_deg'],data['radiance']
        transfer=json.loads(str(data['metadata_json']))
    if metadata['observer']!=transfer['observer']:
        raise ValueError('KStars and transfer observer coordinates differ.')
    if metadata['refraction'] or metadata['projection']!='KStars AzimuthalEquidistant, zenith focus, azimuth=180, rotation=0':
        raise ValueError('Unsupported KStars raster projection or refraction.')
    raster=np.asarray(Image.open(kstars_directory/'kstars-baseline.png').convert('RGB'),dtype=float)/255
    height,width=raster.shape[:2]
    if height!=metadata['size'] or width!=metadata['size']:
        raise ValueError('KStars raster size differs from its projection metadata.')
    radius=metadata['zoom_pixels_per_radian']*np.pi/2
    x,y=np.meshgrid((np.arange(width)-width/2)/radius,(height/2-np.arange(height))/radius)
    distance=np.hypot(x,y);inside=distance<=1
    az=np.rad2deg(np.arctan2(-x,y))%360
    alt=np.maximum(0,90*(1-distance))
    relative=sample_sky(az_grid,alt_grid,values/values[-1,0],az,alt)
    background=background_ratio(relative,scenario)
    extinction_altitudes=np.linspace(0,90,1801)
    extinction=np.interp(alt,extinction_altitudes,stellar_extinction(extinction_altitudes,
                         Observer(**transfer['observer']),Atmosphere(**transfer['atmosphere'])))
    # The monochrome KStars export has white stars and green guide lines. The
    # minimum channel rejects those guides without inventing catalogue objects.
    glyph=np.min(raster,axis=2)
    image=_tone(.002*background+.65*glyph*10**(-.4*extinction))
    rgb=np.repeat(image[...,None],3,axis=2)
    rgb[~inside]=np.array([7,11,18])/255
    fig,ax=plt.subplots(figsize=(10,11),facecolor='#070b12')
    fig.subplots_adjust(left=.02,right=.98,top=.90,bottom=.12)
    ax.imshow(rgb,extent=(-width/(2*radius),width/(2*radius),-height/(2*radius),height/(2*radius)))
    for angle,label in [(0,'С'),(90,'В'),(180,'Ю · Петербург'),(270,'З')]:
        tx,ty=dome_coordinates(angle,0)
        ax.text(tx*1.055,ty*1.055,label,ha='center',va='center',color='#bfcbd8',fontsize=11)
    ax.axis('off')
    fig.text(.5,.96,'KStars + модель засветки',ha='center',color='#e2e8f0',fontsize=18)
    observer=metadata['observer']
    fig.text(.5,.92,f"{observer['latitude_deg']:.3f}° N · {observer['longitude_deg']:.3f}° E · {metadata['utc']}",ha='center',color='#8da0b8',fontsize=10)
    fig.text(.5,.08,'Прототип внешнего слоя: звёзды KStars, фон — наша модель переноса.',ha='center',color='#e2e8f0',fontsize=11)
    fig.text(.5,.045,f'Фон зенита {scenario.artificial_zenith_mag:g} mag/arcsec² задан вручную; глифы KStars не являются фотометрией.',ha='center',color='#e2c68f',fontsize=9)
    fig.savefig(kstars_directory/'kstars-with-skyglow.png',dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)
    result={'kstars':metadata,'display_scenario':asdict(scenario),'transfer_file':str(sky_path),
            'upstream_kstars_modified':False,'compositor':'external registered raster layer',
            'stellar_glyph_photometry_calibrated':False,'ground_photometry_calibrated':False,
            'note':'No limiting magnitude or star counts inferred from KStars glyph intensities.'}
    (kstars_directory/'composite-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
