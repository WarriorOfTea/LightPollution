"""Spherical, scalar single-scattering model. See MODELS.md for its derivation.

Distances are metres; scattering/extinction coefficients are m^-1; phase and
emission functions are sr^-1. Source powers can be physical watts or explicitly
normalised weights. No satellite-to-visible-photometry conversion is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def unit_position(latitude_deg, longitude_deg) -> np.ndarray:
    lat, lon = np.broadcast_arrays(np.deg2rad(latitude_deg), np.deg2rad(longitude_deg))
    return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)


@dataclass(frozen=True)
class Observer:
    latitude_deg: float = 60.30
    longitude_deg: float = 30.315
    height_m: float = 20.0

    def __post_init__(self):
        if not (-90 <= self.latitude_deg <= 90 and -180 <= self.longitude_deg <= 180 and 0 <= self.height_m < 80_000):
            raise ValueError("Invalid observer coordinates or spherical-surface height.")

    @property
    def position(self) -> np.ndarray:
        return (EARTH_RADIUS_M + self.height_m) * unit_position(self.latitude_deg, self.longitude_deg)

    def direction(self, azimuth_deg: float, altitude_deg: float) -> np.ndarray:
        """Azimuth is clockwise from north; altitude is above the local horizontal."""
        if not (np.isfinite(azimuth_deg) and 0 <= altitude_deg <= 90):
            raise ValueError("A sky direction needs a finite azimuth and altitude in [0,90].")
        lat, lon = np.deg2rad([self.latitude_deg, self.longitude_deg])
        az, alt = np.deg2rad([azimuth_deg, altitude_deg])
        up = unit_position(self.latitude_deg, self.longitude_deg)
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
        return np.cos(alt)*(np.cos(az)*north + np.sin(az)*east) + np.sin(alt)*up


def cell_areas(latitude_deg: np.ndarray, step_deg: float = 1/240) -> np.ndarray:
    lat = np.deg2rad(latitude_deg)
    step = np.deg2rad(step_deg)
    return EARTH_RADIUS_M**2 * step * (np.sin(lat+step/2)-np.sin(lat-step/2))


def rayleigh_phase(cosine):
    cosine = np.clip(cosine, -1, 1)
    return 3 * (1 + cosine**2) / (16 * np.pi)


def hg_phase(cosine, g: float = 0.8):
    if not -1 < g < 1:
        raise ValueError("Henyey-Greenstein g must be strictly between -1 and 1.")
    return (1-g*g) / (4*np.pi*(1+g*g-2*g*np.clip(cosine,-1,1))**1.5)


def emission(cos_zenith, horizontal_fraction: float = 0.0, horizontal_order: int = 4):
    """A normalised mixture at fixed total upward power, not a measured lamp model."""
    if not 0 <= horizontal_fraction <= 1 or horizontal_order < 0:
        raise ValueError("Invalid upward emission mixture.")
    mu = np.clip(cos_zenith, 0, 1)
    lambert = mu / np.pi
    low_angle = (horizontal_order+1) * (1-mu)**horizontal_order / (2*np.pi)
    return np.where(np.asarray(cos_zenith) >= 0,
                    (1-horizontal_fraction)*lambert + horizontal_fraction*low_angle, 0.0)


@lru_cache(maxsize=16)
def quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 2:
        raise ValueError("At least two path-quadrature nodes are required.")
    x, w = np.polynomial.legendre.leggauss(order)
    return (x+1)/2, w/2


@dataclass(frozen=True)
class Atmosphere:
    reference_wavelength_nm: float = 550.0
    rayleigh_scale_height_m: float = 8000.0
    aerosol_scale_height_m: float = 1200.0
    aerosol_optical_depth: float = 0.10
    aerosol_single_scattering_albedo: float = 0.9
    aerosol_g: float = 0.8
    rayleigh_multiplier: float = 1.0
    top_m: float = 80_000.0

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if not all(np.isfinite(values)):
            raise ValueError("Atmospheric parameters must be finite.")
        if not (self.reference_wavelength_nm > 0 and self.rayleigh_scale_height_m > 0
                and self.aerosol_scale_height_m > 0 and self.aerosol_optical_depth >= 0
                and 0 <= self.aerosol_single_scattering_albedo <= 1
                and -1 < self.aerosol_g < 1 and self.rayleigh_multiplier >= 0 and self.top_m > 0):
            raise ValueError("Invalid atmospheric parameters.")

    @property
    def rayleigh_sea_level(self) -> float:
        # Bruneton reference example, with wavelength expressed in micrometres.
        return self.rayleigh_multiplier * 1.24062e-6 * (self.reference_wavelength_nm/1000)**-4

    @property
    def aerosol_extinction_sea_level(self) -> float:
        return self.aerosol_optical_depth / self.aerosol_scale_height_m

    @property
    def extinction_majorant(self) -> float:
        """A global bound for delta tracking; separate from a surface value."""
        return self.rayleigh_sea_level + self.aerosol_extinction_sea_level

    def coefficients(self, height_m):
        height = np.maximum(height_m, 0)
        molecular = self.rayleigh_sea_level * np.exp(-height/self.rayleigh_scale_height_m)
        aerosol_extinction = self.aerosol_extinction_sea_level * np.exp(-height/self.aerosol_scale_height_m)
        return molecular, aerosol_extinction*self.aerosol_single_scattering_albedo, molecular+aerosol_extinction

    def optical_depth_radial(self, radius0_squared, radius1_squared, length_squared, order: int = 12):
        """Gauss-Legendre integral along a straight *unoccluded* spherical chord."""
        nodes, weights = quadrature(order)
        total = np.zeros(np.broadcast_shapes(np.shape(radius0_squared), np.shape(radius1_squared), np.shape(length_squared)))
        for t, weight in zip(nodes, weights, strict=True):
            r_squared = (1-t)*radius0_squared + t*radius1_squared - t*(1-t)*length_squared
            height = np.sqrt(np.maximum(r_squared, 0)) - EARTH_RADIUS_M
            total += weight*self.coefficients(height)[2]
        return np.sqrt(length_squared)*total

    def optical_depth(self, start: np.ndarray, end: np.ndarray, order: int = 12):
        return self.optical_depth_radial(np.sum(start*start,axis=-1), np.sum(end*end,axis=-1),
                                         np.sum((end-start)**2,axis=-1), order)


@dataclass(frozen=True)
class Sources:
    positions_m: np.ndarray
    power: np.ndarray

    def __post_init__(self):
        if self.positions_m.ndim != 2 or self.positions_m.shape[1] != 3 or self.power.shape != (len(self.positions_m),):
            raise ValueError("Sources need Nx3 positions and N powers.")
        if not (np.all(np.isfinite(self.positions_m)) and np.all(np.isfinite(self.power)) and np.all(self.power >= 0)):
            raise ValueError("Source positions/powers must be finite and powers nonnegative.")
        if not np.allclose(np.linalg.norm(self.positions_m,axis=1), EARTH_RADIUS_M, rtol=0,atol=0.01):
            raise ValueError("This prototype requires sources on the spherical ground surface.")


def source_to_point(sources: Sources, points: np.ndarray):
    displacement = points[:, None, :] - sources.positions_m[None, :, :]
    distance_squared = np.sum(displacement**2, axis=-1)
    if np.any(distance_squared == 0):
        raise ValueError("Scattering point coincides with a point source; use a finite source model.")
    distance = np.sqrt(distance_squared)
    incoming_direction = displacement / distance[..., None]
    mu = np.sum(incoming_direction * (sources.positions_m/EARTH_RADIUS_M)[None,:,:], axis=-1)
    # For emitters on a convex sphere, mu >= 0 is exactly the ground-visibility test.
    return distance_squared, incoming_direction, mu


@dataclass(frozen=True)
class SolverSettings:
    ray_segments: int = 96
    path_quadrature: int = 12
    horizontal_fraction: float = 0.0
    near_ray_scale_m: float = 500.0

    def __post_init__(self):
        if not (self.ray_segments >= 8 and self.path_quadrature >= 2 and self.near_ray_scale_m > 0
                and 0 <= self.horizontal_fraction <= 1):
            raise ValueError("Invalid numerical solver settings.")


def integrate_direction(
    observer: Observer, sources: Sources, azimuth_deg: float, altitude_deg: float,
    atmosphere: Atmosphere = Atmosphere(), settings: SolverSettings = SolverSettings(),
) -> float:
    if not len(sources.power) or not np.any(sources.power):
        return 0.0
    if observer.height_m >= atmosphere.top_m:
        raise ValueError("Observer must be inside the modelled atmosphere.")
    direction = observer.direction(azimuth_deg, altitude_deg)
    origin = observer.position
    projection = np.dot(origin, direction)
    end = -projection + np.sqrt(projection**2 + (EARTH_RADIUS_M+atmosphere.top_m)**2 - np.dot(origin, origin))
    # Concentrate samples near the observer without a source-distance clamp.
    scale = settings.near_ray_scale_m
    edges = scale*np.expm1(np.linspace(0, np.log1p(end/scale), settings.ray_segments+1))
    s, ds = (edges[1:]+edges[:-1])/2, np.diff(edges)
    points = origin[None,:] + s[:,None]*direction[None,:]
    distance_squared, incoming, mu = source_to_point(sources, points)
    point_r_squared = np.sum(points*points,axis=-1)
    tau_source = atmosphere.optical_depth_radial(EARTH_RADIUS_M**2, point_r_squared[:,None],
                                                distance_squared, settings.path_quadrature)
    tau_observer = atmosphere.optical_depth_radial(np.dot(origin,origin), point_r_squared,
                                                  s*s, settings.path_quadrature)
    # Incoming is source->scatter; outgoing propagation is scatter->observer.
    # The sightline itself points observer->scatter and MUST be negated here.
    cosine = -np.sum(incoming*direction[None,None,:], axis=-1)
    br, ba, _ = atmosphere.coefficients(np.sqrt(point_r_squared)-EARTH_RADIUS_M)
    scattering = br[:,None]*rayleigh_phase(cosine) + ba[:,None]*hg_phase(cosine,atmosphere.aerosol_g)
    intensity = sources.power[None,:]*emission(mu,settings.horizontal_fraction)
    integrand = intensity/distance_squared * np.exp(-tau_source) * scattering
    return float(np.sum(np.sum(integrand,axis=1)*np.exp(-tau_observer)*ds))


def sky_grid(observer: Observer, sources: Sources, azimuths: np.ndarray, altitudes: np.ndarray,
             atmosphere: Atmosphere = Atmosphere(), settings: SolverSettings = SolverSettings(),
             source_factory: Callable[[float,float], Sources] | None = None) -> np.ndarray:
    result = np.zeros((len(altitudes),len(azimuths)))
    for row, altitude in enumerate(altitudes):
        if altitude == 90:
            selected = source_factory(0,90) if source_factory else sources
            result[row,:] = integrate_direction(observer,selected,0,90,atmosphere,settings)
        else:
            for col, azimuth in enumerate(azimuths):
                selected = source_factory(float(azimuth),float(altitude)) if source_factory else sources
                result[row,col] = integrate_direction(observer,selected,float(azimuth),float(altitude),atmosphere,settings)
        print(f"Sky altitude {altitude:g} deg done", flush=True)
    return result
