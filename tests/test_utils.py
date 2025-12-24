from marine_ml.utils import load_params


def test_load_params() -> None:
    actual = load_params()
    expected = {
        "data": {
            "random_state": 42,
            "test_size": 0.2,
            "lag_hours": [1, 3, 6],
            "rolling_window": 6,
            "columns_to_lag_and_window": ["wave_height_significant_m", "wave_direction_deg"],
        },
        "model": {
            "algorithm": "random_forest",
            "max_depth": 5,
            "n_estimators": 100,
            "random_state": 42,
        },
    }
    assert actual == expected
