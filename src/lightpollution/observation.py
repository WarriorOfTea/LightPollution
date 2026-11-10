"""Offline Sun/Moon geometry, explicitly separate from lunar radiance prediction."""

from dataclasses import asdict

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, solar_system_ephemeris
from astropy.time import Time
from astropy.utils import iers
import numpy as np

from .physics import Observer


def observation_context(observer: Observer, utc: str, azimuth_deg: float = 180,
                        altitude_deg: float = 35) -> dict:
    if not utc.endswith("Z"):
        raise ValueError("Supply an ISO UTC timestamp ending in Z.")
    observer.direction(azimuth_deg, altitude_deg)
    instant = Time(utc[:-1], format="isot", scale="utc")
    if not instant.isscalar:
        raise ValueError("A single observation time is required.")
    location = EarthLocation.from_geodetic(observer.longitude_deg*u.deg,
                                          observer.latitude_deg*u.deg, observer.height_m*u.m)
    frame = AltAz(obstime=instant, location=location, pressure=0*u.hPa)
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None), \
            solar_system_ephemeris.set("builtin"):
        sun_geocentric = get_body("sun", instant)
        moon_geocentric = get_body("moon", instant)
        sun = get_body("sun", instant, location=location).transform_to(frame)
        moon = get_body("moon", instant, location=location).transform_to(frame)
        # Compute the angle at the Moon from a common geocentric coordinate origin.
        m = moon_geocentric.cartesian.xyz.to_value(u.km)
        s = sun_geocentric.cartesian.xyz.to_value(u.km)
        to_sun, to_earth = s-m, -m
        cosine = np.dot(to_sun, to_earth)/(np.linalg.norm(to_sun)*np.linalg.norm(to_earth))
        phase = np.arccos(np.clip(cosine, -1, 1))
        target = SkyCoord(az=azimuth_deg*u.deg, alt=altitude_deg*u.deg, frame=frame)
        separation = target.separation(moon).deg
    sun_alt = float(sun.alt.deg)
    moon_alt = float(moon.alt.deg)
    return {
        "utc": utc, "observer": asdict(observer),
        "target": {"azimuth_deg": float(azimuth_deg % 360), "altitude_deg": float(altitude_deg)},
        "sun": {"altitude_deg": sun_alt, "azimuth_deg": float(sun.az.deg)},
        "moon": {"altitude_deg": moon_alt, "azimuth_deg": float(moon.az.deg),
                 "topocentric_distance_km": float(moon.distance.to_value(u.km)),
                 "geocentric_phase_angle_deg": float(np.rad2deg(phase)),
                 "geocentric_illuminated_fraction": float((1+np.cos(phase))/2),
                 "target_separation_deg": float(separation)},
        "astronomical_darkness": sun_alt <= -18,
        "moon_center_above_geometric_horizon": moon_alt > 0,
        "legacy_moonless_renderer_applicable": sun_alt <= -18 and moon_alt <= 0,
        "lunar_sky_radiance_modelled": False,
        "method": {"ephemeris": "astropy builtin", "iers": "offline bundled tables",
                   "atmospheric_refraction": False, "terrain_horizon": False,
                   "phase_reference": "geocentric; illuminated fraction is not lunar flux",
                   "limitations": "Geometry only; no scattered moonlight, horizon glow or optics glare."},
    }
