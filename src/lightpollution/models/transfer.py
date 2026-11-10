"""Pure solver boundary: source data in, numeric response out."""

from typing import Protocol

from ..physics import Atmosphere, Observer, SolverSettings, Sources, integrate_direction


class TransferSolver(Protocol):
    version: str

    def direction(self, observer: Observer, sources: Sources, azimuth: float,
                  altitude: float, atmosphere: Atmosphere, settings: SolverSettings) -> float: ...


class SingleScatteringSolver:
    version = 'single-scattering-sphere-1'

    def direction(self, observer, sources, azimuth, altitude, atmosphere, settings):
        return integrate_direction(observer, sources, azimuth, altitude, atmosphere, settings)
