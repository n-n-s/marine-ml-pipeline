"""Package constants."""

from enum import Enum
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT_DIR / "data"


class DataColumns(str, Enum):
    """Key columns in the marine dataset."""

    wave_height_sig = "wave_height_significant_m"
    wave_height_max = "wave_height_max_m"
    sea_surface_temp = "sea_surface_temperature_degc"
    wave_period_peak = "wave_period_peak_s"
    wave_period_mean = "wave_period_mean_s"
    wave_dir = "wave_direction_deg"
    wave_dir_spread = "directional_wave_spread_deg"

    @classmethod
    def all(cls) -> set[str]:
        """Get a set of all values in the enum."""
        return {e.value for e in cls}
