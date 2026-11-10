"""Stable schema-v1 API. See application/legacy and models/observation."""

from .application.legacy import calculate_observation
from .adapters.observation_io import write_observation
from .models.observation import vertical_extinction_v

__all__ = ['calculate_observation', 'write_observation', 'vertical_extinction_v']
