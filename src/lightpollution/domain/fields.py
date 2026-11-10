"""Data crossing model boundaries. No filesystem, network or UI dependencies."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TransferGrid:
    azimuths: np.ndarray
    altitudes: np.ndarray
    radiance: np.ndarray
    metadata: dict
    sha256: str


@dataclass(frozen=True)
class SourceScale:
    basis_id: str
    coefficient: float
    anchor_raw: float
    anchor_relative: float
    provenance: dict

    def __post_init__(self):
        if not self.basis_id or not all(np.isfinite(v) and v > 0 for v in
                (self.coefficient, self.anchor_raw, self.anchor_relative)):
            raise ValueError('A fixed source scale needs a positive finite reference.')
        if not np.isclose(self.coefficient, self.anchor_relative / self.anchor_raw, rtol=1e-12, atol=0):
            raise ValueError('Source coefficient must equal reference brightness divided by reference response.')


@dataclass(frozen=True)
class ObservationResources:
    grids: Mapping[str, TransferGrid] = field(default_factory=dict)
    catalogue: tuple[dict, dict] | None = None
    source_scale: SourceScale | None = None


@dataclass(frozen=True)
class SkyField:
    """Diffuse B/Bref, one effective V band; masked values are None."""
    directions: tuple[dict, ...]
    photometry: dict
    capabilities: tuple[str, ...] = ('relative_background',)

    def to_dict(self):
        return {'directions': list(self.directions), 'photometry': self.photometry,
                'capabilities': list(self.capabilities), 'spectral_axis': None}


@dataclass(frozen=True)
class StarFluxTable:
    """Point-source fluxes relative to V=0, not per steradian."""
    rows: tuple[dict, ...]
    metadata: dict

    def to_dict(self):
        return {'rows': list(self.rows), 'metadata': self.metadata,
                'flux_unit': 'F_star/F_Vmag0', 'detected_by_eye': None}


@dataclass(frozen=True)
class RunResult:
    value: dict

    def to_dict(self):
        from copy import deepcopy
        return deepcopy(self.value)
