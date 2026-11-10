"""Native-band surface brightness. No implicit conversion to watts or eye luminance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhotometricBand:
    name: str
    zero_point_system: str
    response_reference: str

    def __post_init__(self):
        if not all(isinstance(v, str) and v.strip() for v in self.__dict__.values()):
            raise ValueError("Band, zero-point convention and response reference are required.")


@dataclass(frozen=True)
class SurfaceBrightness:
    mag_arcsec2: np.ndarray
    band: PhotometricBand

    def __post_init__(self):
        if not isinstance(self.band, PhotometricBand):
            raise ValueError("Surface brightness requires an explicit photometric band.")
        values = np.array(self.mag_arcsec2, dtype=float, copy=True)
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("Surface brightness magnitudes must be nonempty and finite.")
        values.flags.writeable = False
        object.__setattr__(self, "mag_arcsec2", values)

    def ratio_to(self, reference: SurfaceBrightness) -> np.ndarray:
        if self.band != reference.band:
            raise ValueError("Cannot compare different bands, responses or zero-point conventions.")
        return 10 ** (-0.4 * (self.mag_arcsec2 - reference.mag_arcsec2))


def sum_surface_brightness(*components: SurfaceBrightness) -> SurfaceBrightness:
    """Add independent light components in one band, stably in logarithmic space."""
    if not components or any(c.band != components[0].band for c in components):
        raise ValueError("At least one component is required, all in exactly the same band.")
    logs = np.broadcast_arrays(*[-0.4 * np.log(10) * c.mag_arcsec2 for c in components])
    total = np.logaddexp.reduce(np.stack(logs), axis=0)
    return SurfaceBrightness(-2.5 * total / np.log(10), components[0].band)


KS_V_BAND = PhotometricBand(
    "V", "KS1991 empirical V convention",
    "doi:10.1086/132921; no numerical spectral response imported")


@dataclass(frozen=True)
class RelativeScale:
    """One common surface-brightness reference for ALL diffuse components."""
    band: PhotometricBand = KS_V_BAND
    reference_mag_arcsec2: float = 22.

    def __post_init__(self):
        if not isinstance(self.band, PhotometricBand) or not np.isfinite(self.reference_mag_arcsec2):
            raise ValueError("Relative scale needs a band and finite reference magnitude.")

    def from_magnitudes(self, magnitude: SurfaceBrightness) -> RelativeBrightness:
        reference = SurfaceBrightness(self.reference_mag_arcsec2, self.band)
        return RelativeBrightness(magnitude.ratio_to(reference), self)


@dataclass(frozen=True)
class RelativeBrightness:
    """Linear diffuse brightness B/Bref; zero is a valid absent component."""
    values: np.ndarray
    scale: RelativeScale

    def __post_init__(self):
        values = np.array(self.values, dtype=float, copy=True)
        if (not isinstance(self.scale, RelativeScale) or not values.size
                or np.any(~np.isfinite(values)) or np.any(values < 0)):
            raise ValueError("Relative brightness needs a scale and finite nonnegative values.")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    def __add__(self, other: RelativeBrightness) -> RelativeBrightness:
        if not isinstance(other, RelativeBrightness) or self.scale != other.scale:
            raise ValueError("Components must share exactly the same band and reference scale.")
        return RelativeBrightness(self.values + other.values, self.scale)

    def magnitudes(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return self.scale.reference_mag_arcsec2 - 2.5*np.log10(self.values)
