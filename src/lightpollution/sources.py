"""Explicit source-weight scenarios, with area-preserving aggregation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .data import Bounds, PIXELS_PER_DEGREE, good_mask
from .physics import EARTH_RADIUS_M, Observer, Sources, cell_areas, unit_position


def regional_sources(arrays: dict, bounds: Bounds, observer: Observer, block: int = 12,
                     exclude_radius_m: float = 3000, composite: str = "NearNadir",
                     include_water: bool = False,
                     refinement_direction: tuple[float,float] | None = None,
                     normalisation_radiance_area: float | None = None,
                     finite_cells: dict | None = None) -> tuple[Sources, dict]:
    if block < 1 or exclude_radius_m < 0:
        raise ValueError("Invalid aggregation block or local-source exclusion radius.")
    key = composite+"_Composite_Snow_Free"
    radiance = arrays[key]
    valid = good_mask(radiance,arrays[key+"_Quality"],arrays[key+"_Num"])
    lat, lon = np.meshgrid(bounds.latitude,bounds.longitude,indexing="ij")
    normals = unit_position(lat,lon)
    observer_normal = unit_position(observer.latitude_deg,observer.longitude_deg)
    chord = np.linalg.norm(normals-observer_normal,axis=-1)
    distance = 2*EARTH_RADIUS_M*np.arcsin(np.clip(chord/2,0,1))
    weights = np.where(valid,radiance,0)*cell_areas(bounds.latitude)[:,None]
    total_valid = float(weights.sum())
    if not total_valid > 0:
        raise ValueError("No positive, high-quality radiance in this region.")
    # This SDS uses the VIIRS classes, not MODIS land/water class numbering.
    # C2 guide table 6: 0 land/desert, 1 land, 2 inland water, 3 sea, 5 coast.
    land_or_coast = np.isin(arrays["Land_Water_Mask"],(0,1,5))
    water = np.isin(arrays["Land_Water_Mask"],(2,3))
    unknown_surface = ~land_or_coast & ~water
    eligible = valid & (land_or_coast | water) if include_water else valid & land_or_coast
    outside = distance >= exclude_radius_m
    water_removed = float(weights[valid & water].sum()) if not include_water else 0.0
    unknown_removed = float(weights[valid & unknown_surface].sum())
    local_removed = float(weights[eligible & ~outside].sum())
    weights = np.where(eligible & outside,weights,0)
    total_retained = float(weights.sum())
    if not total_retained > 0:
        raise ValueError("No sources remain after masking.")
    denominator = total_retained if normalisation_radiance_area is None else float(normalisation_radiance_area)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Source normalisation requires a positive finite radiance-area reference.')
    positions, powers, patch_sizes = [], [], []
    ray = observer.direction(*refinement_direction) if refinement_direction is not None else None
    finite_pixel_count = 0

    def add_finite_cell(row, col, power):
        nonlocal finite_pixel_count
        finite_pixel_count += 1
        half = 1/(2*PIXELS_PER_DEGREE)
        lat0, lat1 = np.sin(np.deg2rad([bounds.latitude[row]-half, bounds.latitude[row]+half]))
        lon0, lon1 = np.deg2rad([bounds.longitude[col]-half, bounds.longitude[col]+half])
        order = finite_cells['quadrature_order']
        nodes, weights1 = np.polynomial.legendre.leggauss(order)
        def split(y0, y1, x0, x1, fraction, depth=0):
            latitude = np.rad2deg(np.arcsin((y0+y1)/2))
            longitude = np.rad2deg((x0+x1)/2)
            centre = unit_position(latitude, longitude)
            distance = float(np.linalg.norm(centre-observer_normal)*EARTH_RADIUS_M)
            width = (x1-x0)*np.sqrt(1-((y0+y1)/2)**2)*EARTH_RADIUS_M
            height = (np.arcsin(y1)-np.arcsin(y0))*EARTH_RADIUS_M
            target = max(finite_cells['minimum_patch_m'],
                         .25*max(finite_cells.get('observer_agl_m', observer.height_m), distance))
            if max(width, height) > target and depth < 12:
                ym, xm = (y0+y1)/2, (x0+x1)/2
                for ya, yb in [(y0, ym), (ym, y1)]:
                    for xa, xb in [(x0, xm), (xm, x1)]:
                        split(ya, yb, xa, xb, fraction/4, depth+1)
                return
            ys = (y0+y1)/2+nodes*(y1-y0)/2
            xs = (x0+x1)/2+nodes*(x1-x0)/2
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            pp = unit_position(np.rad2deg(np.arcsin(yy)), np.rad2deg(xx)).reshape(-1, 3)
            ww = (weights1[:, None]*weights1[None, :]/4).reshape(-1)
            positions.extend(EARTH_RADIUS_M*pp)
            powers.extend(power*fraction*ww/denominator)
            patch_sizes.extend([0.]*len(ww))
        split(lat0, lat1, lon0, lon1, 1.)

    def add_patch(row: int, bottom: int, col: int, right: int):
        window = (slice(row,bottom),slice(col,right))
        w = weights[window]
        power = float(w.sum())
        if power <= 0:
            return
        centroid = np.sum(normals[window]*w[...,None],axis=(0,1))
        normal = centroid/np.linalg.norm(centroid)
        patch_radius = float(np.max(np.linalg.norm(normals[window][w>0]-normal,axis=1))*EARTH_RADIUS_M)
        observer_distance = float(np.linalg.norm(normal-observer_normal)*EARTH_RADIUS_M)
        near_ray = False
        if ray is not None:
            displacement = EARTH_RADIUS_M*normal-observer.position
            along = max(0,float(np.dot(displacement,ray)))
            distance_to_ray = float(np.linalg.norm(displacement-along*ray))
            # A block far from the observer can still lie immediately below
            # a low sightline. Resolve it before treating it as one emitter.
            near_ray = distance_to_ray < 4*patch_radius
        # Do not move an excluded ring of pixels into the observer's vicinity
        # by replacing the whole ring section with one large centred emitter.
        if max(bottom-row,right-col)>1 and (observer_distance<3*patch_radius or observer_distance<exclude_radius_m or near_ray):
            rows=sorted(set((row,(row+bottom)//2,bottom)))
            cols=sorted(set((col,(col+right)//2,right)))
            for r0,r1 in zip(rows[:-1],rows[1:],strict=True):
                for c0,c1 in zip(cols[:-1],cols[1:],strict=True):
                    add_patch(r0,r1,c0,c1)
            return
        if (finite_cells is not None and bottom-row == 1 and right-col == 1
                and observer_distance <= finite_cells['radius_m']+500):
            add_finite_cell(row, col, power)
            return
        if finite_cells is not None and observer_distance < finite_cells['radius_m']+patch_radius:
            if max(bottom-row, right-col)>1:
                rows=sorted(set((row,(row+bottom)//2,bottom)))
                cols=sorted(set((col,(col+right)//2,right)))
                for r0,r1 in zip(rows[:-1],rows[1:],strict=True):
                    for c0,c1 in zip(cols[:-1],cols[1:],strict=True):
                        add_patch(r0,r1,c0,c1)
                return
        positions.append(EARTH_RADIUS_M*normal)
        powers.append(power/denominator)
        patch_sizes.append(patch_radius)

    for row in range(0,bounds.shape[0],block):
        for col in range(0,bounds.shape[1],block):
            add_patch(row,min(row+block,bounds.shape[0]),col,min(col+block,bounds.shape[1]))
    sources = Sources(np.asarray(positions),np.asarray(powers))
    report = {
        "composite": composite, "observer": asdict(observer), "block_pixels": block,
        "effective_sources": len(powers), "retained_positive_pixels": int((weights>0).sum()),
        "normalisation": ("Total retained power = 1 arbitrary unit." if normalisation_radiance_area is None
                          else "Fixed regional radiance-area reference before observer exclusions; removed power is not restored."),
        "normalisation_radiance_area": denominator,
        "retained_radiance_area": total_retained,
        "unknown_pixels_are_not_dark_measurements": True,
        "excluded_local_radius_m": exclude_radius_m, "include_water": include_water,
        "weight_denominator_radiance_times_area": total_valid,
        "weight_removed_water_fraction": water_removed/total_valid,
        "weight_removed_unknown_surface_fraction": unknown_removed/total_valid,
        "weight_removed_local_fraction": local_removed/total_valid,
        "retained_weight_fraction": total_retained/total_valid,
        "largest_patch_radius_m": max(patch_sizes),
        "ray_refinement_azimuth_altitude_deg": refinement_direction,
        "nearest_retained_pixel_m": float(distance[weights>0].min()),
        "nearest_effective_source_chord_m": float(np.linalg.norm(sources.positions_m-EARTH_RADIUS_M*observer_normal,axis=1).min()),
        "finite_emitting_pixels": finite_pixel_count,
        "finite_cell_settings": finite_cells,
        "unmodelled_sources_outside_bounds": asdict(bounds),
    }
    return sources,report


def synthetic_cities(observer: Observer = Observer()) -> Sources:
    """Two finite *test* cities; this is never presented as NASA information."""
    # East/north kilometre offsets relative to the observer. Each point is a
    # deliberately specified artificial emitter, so no radiance inversion is used.
    latitudes, longitudes, powers = [], [], []
    for east_km,north_km,total_power,radius_km in ((0,-40,1.0,5.0),(25,0,0.25,2.5)):
        offsets = np.linspace(-radius_km,radius_km,7)
        weights,coordinates = [],[]
        for east in offsets:
            for north in offsets:
                weights.append(np.exp(-(east*east+north*north)/(radius_km/2)**2))
                coordinates.append((east_km+east,north_km+north))
        weights = np.asarray(weights)/sum(weights)*total_power
        for (east,north),power in zip(coordinates,weights,strict=True):
            latitudes.append(observer.latitude_deg+np.rad2deg(north*1000/EARTH_RADIUS_M))
            longitudes.append(observer.longitude_deg+np.rad2deg(east*1000/(EARTH_RADIUS_M*np.cos(np.deg2rad(observer.latitude_deg)))))
            powers.append(power)
    return Sources(EARTH_RADIUS_M*unit_position(np.array(latitudes),np.array(longitudes)),np.array(powers))
