"""Pure solver boundary: source data in, numeric response out."""

from typing import Protocol

from ..physics import Atmosphere, Observer, SolverSettings, Sources, integrate_direction


class TransferSolver(Protocol):
    version: str

    def direction(self, observer: Observer, sources: Sources, azimuth: float,
                  altitude: float, atmosphere: Atmosphere, settings: SolverSettings) -> float: ...


class SpectralTransferSolver(TransferSolver, Protocol):
    def spectrum(self, observer, sources, azimuth, altitude, atmosphere, settings,
                 wavelengths, angstrom_exponent): ...


class SingleScatteringSolver:
    version = 'single-scattering-sphere-1'

    def direction(self, observer, sources, azimuth, altitude, atmosphere, settings):
        return integrate_direction(observer, sources, azimuth, altitude, atmosphere, settings)

    def spectrum(self, observer, sources, azimuth, altitude, atmosphere, settings,
                 wavelengths, angstrom_exponent):
        from .spectra import integrate_spectral_direction
        return integrate_spectral_direction(observer, sources, azimuth, altitude,
                                             atmosphere, settings, wavelengths, angstrom_exponent)
