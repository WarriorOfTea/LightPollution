"""Krisciunas & Schaefer (1991), equations 1, 3, 8, 9, 15, 21.

The returned nanoLamberts follow the paper's empirical V-band convention;
they are not a general spectrum-independent conversion to eye luminance.
The first implementation excludes angular separations below 10 degrees.
"""

import numpy as np

REFERENCE_DISTANCE_KM = 384_400.0
KS1991_DOI = "10.1086/132921"


def scattering_airmass(altitude_deg):
    """Equation 3: scattering airmass, NOT the secant or direct-beam airmass."""
    altitude = np.asarray(altitude_deg, dtype=float)
    if np.any(~np.isfinite(altitude)) or np.any((altitude < 0) | (altitude > 90)):
        raise ValueError("Scattering airmass needs altitudes in [0,90] degrees.")
    return (1 - .96 * np.cos(np.deg2rad(altitude))**2)**-.5


def v_mag_to_nanolambert(mag_arcsec2):
    """Paper equation 1; retains the author's rounded coefficients."""
    values = np.asarray(mag_arcsec2, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("V surface brightness must be finite.")
    return 34.08 * np.exp(20.7233 - .92104 * values)


def nanolambert_to_v_mag(brightness):
    values = np.asarray(brightness, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("The logarithmic conversion requires positive finite brightness.")
    return (20.7233 - np.log(values / 34.08)) / .92104


def scattered_moonlight(phase_angle_deg, moon_altitude_deg, target_altitude_deg,
                      separation_deg, extinction_v_mag_airmass,
                      distance_km=REFERENCE_DISTANCE_KM):
    """Lunar-only V background in nL, at separations >=10 deg, no eclipse.

    alpha=0 is full Moon, alpha=180 is new. Distance adds an explicit inverse
    square correction, referenced to 384400 km; the original formula omits it.
    A Moon centre at/below the geometric horizon gives zero by approximation.
    Callers must separately expose this approximation and domain masks.
    """
    alpha, moon_alt, target_alt, rho, k, distance = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in
          (phase_angle_deg, moon_altitude_deg, target_altitude_deg,
           separation_deg, extinction_v_mag_airmass, distance_km)])
    if any(np.any(~np.isfinite(value)) for value in (alpha, moon_alt, target_alt, rho, k, distance)):
        raise ValueError("Moonlight inputs must be finite.")
    if (np.any((alpha < 0) | (alpha > 180)) or np.any((moon_alt < -90) | (moon_alt > 90))
            or np.any((target_alt < 0) | (target_alt > 90))
            or np.any((rho < 10) | (rho > 180)) or np.any(k < 0) or np.any(distance <= 0)):
        raise ValueError("Moonlight requires valid angles, separation >=10 degrees, k>=0 and distance>0.")
    magnitude = -12.73 + .026 * alpha + 4e-9 * alpha**4
    incident = 10**(-.4 * (magnitude + 16.57)) * (REFERENCE_DISTANCE_KM / distance)**2
    scatter = 10**5.36 * (1.06 + np.cos(np.deg2rad(rho))**2) + 10**(6.15 - rho / 40)
    tau_moon = .4 * np.log(10) * k * scattering_airmass(np.maximum(moon_alt, 0))
    tau_target = .4 * np.log(10) * k * scattering_airmass(target_alt)
    brightness = scatter * incident * np.exp(-tau_moon) * (-np.expm1(-tau_target))
    return np.where(moon_alt > 0, brightness, 0.)


def separation_angle(azimuth_deg, altitude_deg, moon_azimuth_deg, moon_altitude_deg):
    """Spherical separation of two apparent geometric horizontal directions."""
    az, alt, maz, malt = np.deg2rad(np.broadcast_arrays(
        azimuth_deg, altitude_deg, moon_azimuth_deg, moon_altitude_deg))
    cosine = np.sin(alt)*np.sin(malt) + np.cos(alt)*np.cos(malt)*np.cos(az-maz)
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))
