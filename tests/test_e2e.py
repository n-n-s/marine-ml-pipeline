"""End-to-end test for the complete ML pipeline.

Tests: data preparation → training → model serving → prediction
"""

import json
import logging
import pickle
import shutil
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

from marine_ml.constants import CACHE_DIR, MODEL_DIR, PROJECT_ROOT_DIR
from marine_ml.prepare_data import get_preprocessed_data
from marine_ml.serve import app, model_service
from marine_ml.train import create_model

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def cleanup() -> Generator:
    """Clean up test artifacts before and after test"""
    # Setup: clean test outputs
    test_dirs = [CACHE_DIR, MODEL_DIR, PROJECT_ROOT_DIR / "mlruns"]

    yield

    # Teardown: optionally clean up after test
    # Comment out if you want to inspect artifacts after test
    for dir_path in test_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)


@pytest.fixture(scope="module")
def trained_model(cleanup) -> tuple[RandomForestRegressor, list[str]]:  # noqa: ARG001,ANN001
    """Run the complete pipeline to train a model"""

    # Step 1: Prepare data
    _ = get_preprocessed_data()
    assert (CACHE_DIR / "preprocessed.pkl").exists()  # Verify processed data exists

    # Step 2: Train model (with minimal Optuna trials for speed)
    logger.debug("Training model...")
    _ = create_model()
    # Verify model artifacts exist
    assert (MODEL_DIR / "model.pkl").exists(), "Model file not created"
    assert (MODEL_DIR / "feature_names.json").exists(), "Feature names not created"

    logger.debug("Pipeline completed successfully")
    with (MODEL_DIR / "model.pkl").open("rb") as f:
        model = pickle.load(f)  # noqa: S301
    with (MODEL_DIR / "feature_names.json").open("r") as f:
        feature_names = json.load(f)
    return model, feature_names


@pytest.fixture(scope="module")
def api_client(trained_model: tuple[RandomForestRegressor, list[str]]) -> TestClient:
    """Create a TestClient for the FastAPI app"""

    model_service.model = trained_model[0]
    model_service.feature_names = trained_model[1]
    model_service.model_version = "test-version"

    return TestClient(app)


def test_health(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200  # noqa: PLR2004


def test_reload(api_client: TestClient) -> None:
    response = api_client.post("/reload")
    assert response.status_code == 200  # noqa: PLR2004


def test_complete_pipeline(api_client: TestClient) -> None:
    """Test the complete end-to-end workflow"""

    # Test 1: Health check
    logger.debug("Testing health endpoint...")
    response = api_client.get("/health")
    assert response.status_code == 200  # noqa: PLR2004
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] == 1
    logger.debug("Model version: %s", health_data.get("model_version"))

    # Test 2: Get feature names
    logger.debug("Testing features endpoint...")
    response = api_client.get("/features")
    assert response.status_code == 200  # noqa: PLR2004
    features_data = response.json()
    assert "features" in features_data
    assert len(features_data["features"]) > 0
    feature_names = features_data["features"]
    logger.debug("Number of features required: %s", len(feature_names))

    # Test 3: Make prediction with valid data
    logger.debug("Testing prediction endpoint...")
    sample_features = {
        "directional_wave_spread_deg;75": 40,
        "hour": 7,
        "month": 12,
        "sea_surface_temperature_degc;75": 10.1,
        "wave_direction_deg;75": 270,
        "wave_direction_deg;75_lag_3h": 240,
        "wave_direction_deg;75_lag_6h": 260,
        "wave_direction_deg;75_rolling_mean_6h": 255,
        "wave_direction_deg;75_rolling_std_6h": 40,
        "wave_height_max_m;75": 2.4,
        "wave_height_significant_m;75": 2,
        "wave_height_significant_m;75_lag_3h": 1.8,
        "wave_height_significant_m;75_lag_6h": 1.4,
        "wave_height_significant_m;75_rolling_mean_6h": 1.7,
        "wave_height_significant_m;75_rolling_std_6h": 0.1,
        "wave_period_mean_s;75": 12,
        "wave_period_peak_s;75": 18,
    }
    response = api_client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200  # noqa: PLR2004
    prediction_data = response.json()
    assert "predicted_wave_height" in prediction_data
    assert isinstance(prediction_data["predicted_wave_height"], (int, float))
    expected_min = 3.5
    expected_max = 4.0
    assert prediction_data["predicted_wave_height"] > expected_min
    assert prediction_data["predicted_wave_height"] < expected_max
    logger.debug("Prediction: %.2f m", prediction_data["predicted_wave_height"])
    logger.debug("Model version: %s", prediction_data.get("model_version"))

    # Test 4: Invalid request (missing features)
    logger.debug("Testing error handling...")
    response = api_client.post(
        "/predict",
        json={"features": {"wave_height_source": 2.5}},  # Missing most features
    )
    assert response.status_code in [400, 422, 500]  # Should fail validation

    logger.debug("All end-to-end tests passed")
