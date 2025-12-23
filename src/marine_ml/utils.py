"""Utility functions for the package."""

import yaml

from marine_ml.constants import PROJECT_ROOT_DIR


def load_params() -> dict:
    """Read hyperparameters from yaml file."""
    with (PROJECT_ROOT_DIR / "params.yaml").open("r") as f:
        return yaml.safe_load(f)
