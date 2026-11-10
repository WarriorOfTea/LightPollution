"""Explicit editable scenario defaults, not a claim about local measured weather."""

from dataclasses import asdict

import numpy as np

from .physics import Atmosphere


def atmosphere_from_mapping(parameters: dict | None = None) -> Atmosphere:
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict) or set(parameters)-set(Atmosphere.__dataclass_fields__):
        raise ValueError("Unknown atmospheric parameter; use the documented physical coefficients.")
    if any(isinstance(v, bool) or not isinstance(v, (float,int)) or not np.isfinite(v)
           for v in parameters.values()):
        raise ValueError("Atmospheric parameters must be finite numbers.")
    return Atmosphere(**parameters)


def atmosphere_origin(parameters: dict | None = None) -> dict:
    parameters = parameters or {}
    return {'kind':'explicit_scenario' if parameters else 'baseline_scenario',
            'user_parameters':sorted(parameters),
            'default_parameters':sorted(set(asdict(Atmosphere()))-set(parameters)),
            'measured_weather':False, 'regional_probability_estimated':False}
