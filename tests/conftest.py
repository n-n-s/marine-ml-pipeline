import pandas as pd
import pytest


@pytest.fixture
def marine_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "index": pd.Timestamp("2025-11-01 00:00:00+0000", tz="UTC"),
                "wave_height_significant_m": 3.57,
                "wave_height_max_m": 4.93,
                "sea_surface_temperature_degc": 13.9,
                "wave_period_peak_s": 13.33,
                "wave_period_mean_s": 6.452,
                "wave_direction_deg": 239.1,
                "directional_wave_spread_deg": 21.4,
            },
            {
                "index": pd.Timestamp("2025-11-01 00:30:00+0000", tz="UTC"),
                "wave_height_significant_m": 3.5,
                "wave_height_max_m": 7.33,
                "sea_surface_temperature_degc": 13.9,
                "wave_period_peak_s": 10.0,
                "wave_period_mean_s": 6.25,
                "wave_direction_deg": 227.8,
                "directional_wave_spread_deg": 12.8,
            },
            {
                "index": pd.Timestamp("2025-11-01 01:00:00+0000", tz="UTC"),
                "wave_height_significant_m": 3.36,
                "wave_height_max_m": 5.09,
                "sea_surface_temperature_degc": 13.9,
                "wave_period_peak_s": 9.09,
                "wave_period_mean_s": 6.25,
                "wave_direction_deg": 229.2,
                "directional_wave_spread_deg": 18.5,
            },
            {
                "index": pd.Timestamp("2025-11-01 01:30:00+0000", tz="UTC"),
                "wave_height_significant_m": 3.64,
                "wave_height_max_m": 5.65,
                "sea_surface_temperature_degc": 13.9,
                "wave_period_peak_s": 9.09,
                "wave_period_mean_s": 6.452,
                "wave_direction_deg": 226.4,
                "directional_wave_spread_deg": 13.7,
            },
            {
                "index": pd.Timestamp("2025-11-01 02:00:00+0000", tz="UTC"),
                "wave_height_significant_m": 3.34,
                "wave_height_max_m": 5.99,
                "sea_surface_temperature_degc": 13.9,
                "wave_period_peak_s": 12.5,
                "wave_period_mean_s": 6.25,
                "wave_direction_deg": 237.7,
                "directional_wave_spread_deg": 17.7,
            },
        ]
    )
