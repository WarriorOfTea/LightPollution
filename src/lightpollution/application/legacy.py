"""Public schema-v1 use case; files are loaded only by its adapter."""

from pathlib import Path

from ..adapters.observation_io import load_observation_resources
from ..models.observation import calculate_scene


def calculate_observation(config: dict, base_directory: Path = Path('.')) -> dict:
    return calculate_scene(config, load_observation_resources(config, base_directory))
