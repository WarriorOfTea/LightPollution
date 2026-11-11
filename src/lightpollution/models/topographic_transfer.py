"""Single scattering with elevated emitters and sampled DSM occlusion on both legs."""

from dataclasses import replace

import numpy as np

from ..physics import EARTH_RADIUS_M, emission, hg_phase, rayleigh_phase
from .spectra import checked_axis


class TopographicSolver:
    version = 'single-scattering-dsm-visibility-1'

    def __init__(self, terrain, step_m=100.):
        self.terrain, self.step_m = terrain, step_m

    def direction(self, observer, sources, azimuth, altitude, atmosphere, settings):
        return float(self.spectrum(observer, sources, azimuth, altitude, atmosphere,
                                   settings, np.array([550.]), 1.3)[0])

    def spectrum(self, observer, sources, azimuth, altitude, atmosphere, settings,
                 wavelengths, angstrom_exponent):
        axis, _ = checked_axis(wavelengths, np.zeros(len(wavelengths))) if len(wavelengths)>1 else (np.asarray(wavelengths), None)
        if not len(sources.power):
            return np.zeros(len(axis))
        origin, direction = observer.position, observer.direction(azimuth, altitude)
        projection = float(np.dot(origin, direction))
        end = -projection+np.sqrt(projection**2+(EARTH_RADIUS_M+atmosphere.top_m)**2-np.dot(origin, origin))
        edges = settings.near_ray_scale_m*np.expm1(np.linspace(
            0, np.log1p(end/settings.near_ray_scale_m), settings.ray_segments+1))
        s, ds = (edges[1:]+edges[:-1])/2, np.diff(edges)
        points = origin+s[:, None]*direction
        r2 = np.sum(points*points, axis=1)
        visible_observer = self.terrain.visible(origin, points, self.step_m)
        rayleigh_atm = replace(atmosphere, aerosol_optical_depth=0.)
        aerosol_atm = replace(atmosphere, rayleigh_multiplier=0.)
        tau_o_r = rayleigh_atm.optical_depth_radial(np.dot(origin, origin), r2, s*s, settings.path_quadrature)
        tau_o_a = aerosol_atm.optical_depth_radial(np.dot(origin, origin), r2, s*s, settings.path_quadrature)
        br, ba, _ = atmosphere.coefficients(np.sqrt(r2)-EARTH_RADIUS_M)
        output = np.zeros(len(axis))
        # Bounded working sets for large finite-cell source arrays.
        for begin in range(0, len(sources.power), 512):
            positions = sources.positions_m[begin:begin+512]
            powers = sources.power[begin:begin+512]
            source_r2 = np.sum(positions*positions, axis=1)
            displacement = points[:, None, :]-positions[None, :, :]
            distance2 = np.sum(displacement*displacement, axis=2)
            if np.any(distance2 <= 0):
                raise ValueError('A scattering node coincides with an emitter.')
            incoming = displacement/np.sqrt(distance2)[:, :, None]
            mu = np.sum(incoming*(positions/np.sqrt(source_r2)[:, None])[None, :, :], axis=2)
            visible = self.terrain.visible(positions[None, :, :], points[:, None, :], self.step_m)
            visible &= visible_observer[:, None] & (mu >= 0)
            tau_r = rayleigh_atm.optical_depth_radial(source_r2[None, :], r2[:, None], distance2, settings.path_quadrature)+tau_o_r[:, None]
            tau_a = aerosol_atm.optical_depth_radial(source_r2[None, :], r2[:, None], distance2, settings.path_quadrature)+tau_o_a[:, None]
            cosine = -np.sum(incoming*direction, axis=2)
            geometric = powers[None, :]*emission(mu, settings.horizontal_fraction)/distance2*ds[:, None]*visible
            r_component = geometric*br[:, None]*rayleigh_phase(cosine)
            a_component = geometric*ba[:, None]*hg_phase(cosine, atmosphere.aerosol_g)
            for i, wavelength in enumerate(axis):
                molecular, aerosol = (550./wavelength)**4, (550./wavelength)**angstrom_exponent
                output[i] += np.sum((r_component*molecular+a_component*aerosol)*np.exp(-tau_r*molecular-tau_a*aerosol))
        return output
