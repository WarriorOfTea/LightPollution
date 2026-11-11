"""Finite height-field terrain and sampled spherical line-of-sight visibility.

Heights are metres over the common spherical mean-sea-level reference.
The raster is an explicit DSM approximation, not a reconstructed city mesh.
"""

from dataclasses import dataclass

import numpy as np

from ..physics import EARTH_RADIUS_M, Observer, unit_position


@dataclass(frozen=True)
class ElevatedSources:
    positions_m: np.ndarray
    power: np.ndarray

    def __post_init__(self):
        if self.positions_m.shape != (len(self.power), 3) or self.power.ndim != 1:
            raise ValueError('Elevated sources need Nx3 positions and N powers.')
        if not np.all(np.isfinite(self.positions_m)) or not np.all(np.isfinite(self.power)) or np.any(self.power < 0):
            raise ValueError('Elevated source arrays must be finite and powers nonnegative.')


@dataclass(frozen=True)
class TerrainGrid:
    latitudes: np.ndarray
    longitudes: np.ndarray
    heights_m: np.ndarray
    metadata: dict

    def __post_init__(self):
        if (self.heights_m.shape != (len(self.latitudes), len(self.longitudes))
                or len(self.latitudes) < 2 or len(self.longitudes) < 2
                or np.any(np.diff(self.latitudes) <= 0) or np.any(np.diff(self.longitudes) <= 0)
                or not all(np.all(np.isfinite(v)) for v in [self.latitudes, self.longitudes, self.heights_m])
                or np.min(self.heights_m) < -500 or np.max(self.heights_m) > 10000):
            raise ValueError('Invalid finite terrain grid or height range.')

    def sample(self, latitude, longitude):
        lat, lon = np.broadcast_arrays(latitude, longitude)
        if (np.any(lat < self.latitudes[0]) or np.any(lat > self.latitudes[-1])
                or np.any(lon < self.longitudes[0]) or np.any(lon > self.longitudes[-1])):
            raise ValueError('Terrain coverage does not contain the required ground path.')
        iy = np.clip(np.searchsorted(self.latitudes, lat, side='right')-1, 0, len(self.latitudes)-2)
        ix = np.clip(np.searchsorted(self.longitudes, lon, side='right')-1, 0, len(self.longitudes)-2)
        y = (lat-self.latitudes[iy])/(self.latitudes[iy+1]-self.latitudes[iy])
        x = (lon-self.longitudes[ix])/(self.longitudes[ix+1]-self.longitudes[ix])
        z = self.heights_m
        return ((1-y)*((1-x)*z[iy, ix]+x*z[iy, ix+1])
                + y*((1-x)*z[iy+1, ix]+x*z[iy+1, ix+1]))

    def lift_sources(self, sources):
        radius = np.linalg.norm(sources.positions_m, axis=1)
        normals = sources.positions_m/radius[:, None]
        lat = np.rad2deg(np.arcsin(normals[:, 2]))
        lon = np.rad2deg(np.arctan2(normals[:, 1], normals[:, 0]))
        height = self.sample(lat, lon)
        # Source facets are infinitesimally above the DSM; no finite lamp
        # mounting height or sloped BRDF is inferred from a satellite pixel.
        return ElevatedSources(normals*(EARTH_RADIUS_M+height+.05)[:, None], sources.power)

    def observer(self, location):
        height = float(self.sample(location.latitude_deg, location.longitude_deg)) + location.height_m
        if height < 0:
            raise ValueError('Observer below mean sea level is outside the current atmospheric profile contract.')
        return Observer(location.latitude_deg, location.longitude_deg, height)

    def visible(self, starts, ends, step_m=100.):
        """Sample only the part of a chord inside the highest terrain sphere.

        Long paths high above all terrain need no samples. Padding work is
        bounded by sorting on sample counts and using small batches.
        No missing ground outside the supplied raster is assumed flat.
        """
        a, b = np.broadcast_arrays(np.asarray(starts, float), np.asarray(ends, float))
        shape = a.shape[:-1]
        a, b = a.reshape(-1, 3), b.reshape(-1, 3)
        if a.shape[-1] != 3 or not np.isfinite(step_m) or step_m <= 0:
            raise ValueError('Invalid visibility segments or step.')
        delta = b-a
        length2 = np.sum(delta*delta, axis=1)
        dot = np.sum(a*delta, axis=1)
        t_min = np.clip(np.divide(-dot, length2, out=np.zeros_like(dot), where=length2>0), 0, 1)
        min_r2 = np.sum((a+t_min[:, None]*delta)**2, axis=1)
        clear = min_r2 >= (EARTH_RADIUS_M+float(np.min(self.heights_m))-.02)**2
        outer = EARTH_RADIUS_M+float(np.max(self.heights_m))+.1
        c = np.sum(a*a, axis=1)-outer*outer
        disc = dot*dot-length2*c
        possible = (disc >= 0) & (length2 > 0) & clear
        denominator = np.where(length2 > 0, length2, 1)
        root = np.sqrt(np.maximum(disc, 0))
        lo = np.maximum(0., (-dot-root)/denominator)
        hi = np.minimum(1., (-dot+root)/denominator)
        possible &= hi >= lo
        counts = np.maximum(1, np.ceil(np.maximum(0, hi-lo)*np.sqrt(length2)/step_m).astype(int))
        indices = np.flatnonzero(possible)
        indices = indices[np.argsort(counts[indices])]
        for begin in range(0, len(indices), 128):
            ix = indices[begin:begin+128]
            n = counts[ix]
            k = np.arange(int(max(n)))
            valid = k[None, :] < n[:, None]
            t = lo[ix, None]+(hi[ix]-lo[ix])[:, None]*(k[None, :]+.5)/n[:, None]
            t = np.minimum(t, hi[ix, None])
            points = a[ix, None, :]+t[:, :, None]*delta[ix, None, :]
            radius = np.linalg.norm(points, axis=2)
            lat = np.rad2deg(np.arcsin(points[:, :, 2]/radius))
            lon = np.rad2deg(np.arctan2(points[:, :, 1], points[:, :, 0]))
            # Do not query padding samples outside the true segment interval.
            terrain = np.zeros(radius.shape)
            terrain[valid] = self.sample(lat[valid], lon[valid])
            blocked = valid & (radius-EARTH_RADIUS_M < terrain-.02)
            clear[ix] = ~np.any(blocked, axis=1)
        return clear.reshape(shape)

    def horizon(self, observer, azimuths, range_m=50000., step_m=100.):
        az = np.deg2rad(np.asarray(azimuths))[:, None]
        angle = np.arange(step_m, range_m+step_m/2, step_m)[None, :]/EARTH_RADIUS_M
        lat0, lon0 = np.deg2rad([observer.latitude_deg, observer.longitude_deg])
        lat = np.arcsin(np.sin(lat0)*np.cos(angle)+np.cos(lat0)*np.sin(angle)*np.cos(az))
        lon = lon0+np.arctan2(np.sin(az)*np.sin(angle)*np.cos(lat0), np.cos(angle)-np.sin(lat0)*np.sin(lat))
        heights = self.sample(np.rad2deg(lat), np.rad2deg(lon))
        radius = EARTH_RADIUS_M+heights
        vertical = radius*np.cos(angle)-(EARTH_RADIUS_M+observer.height_m)
        horizontal = radius*np.sin(angle)
        return np.rad2deg(np.max(np.arctan2(vertical, horizontal), axis=1))
