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
