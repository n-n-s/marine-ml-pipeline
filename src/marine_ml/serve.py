"""FastAPI service for wave height prediction. Loads model from MLflow Model Registry with fallback to local file."""
# ruff: noqa: BLE001,N806

import json
import logging
import pickle

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from marine_ml.constants import MODEL_DIR

# Configuration
MODEL_NAME = "model-pz-to-lev"
MODEL_STAGE = "Production"

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and managing the ML model."""

    def __init__(self) -> None:
        """Initialise class."""
        self.model = None
        self.model_version = None
        self.feature_names = None
        self._load_model()

    def _load_model(self) -> bool:
        """Load model from MLflow Registry or local file."""
        if not self._load_from_registry():
            logger.info("Falling back to local model file...")
            if not self._load_from_file():
                logger.warning("Could not load model from MLflow or file")
                return False
        return True

    def _load_from_registry(self) -> bool:
        """Load model from MLflow Model Registry."""
        try:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

            logger.info("Loading model from MLflow Registry: %s", model_uri)
            self.model = mlflow.pyfunc.load_model(model_uri)

            # Get model version info
            client = mlflow.tracking.MlflowClient()
            model_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            if model_versions:
                self.model_version = model_versions[0].version
                logger.info("Loaded model version %s from '%s' stage", self.model_version, MODEL_STAGE)

            # Try to get feature names from model metadata
            try:
                self.feature_names = self.model.metadata.get_input_schema().input_names()
            except Exception:
                logger.warning("Could not load feature names from model metadata")

        except Exception:
            logger.exception("Error loading from MLflow Registry")
            return False
        else:
            return True

    def _load_from_file(self) -> bool:
        """Load model from local file."""
        try:
            with (MODEL_DIR / "model.pkl").open("rb") as f:
                self.model = pickle.load(f)  # noqa: S301

            with (MODEL_DIR / "feature_names.json").open("r") as f:
                self.feature_names = json.load(f)

            logger.info("Loaded model from local file: %s", MODEL_DIR / "model.pkl")
        except Exception:
            logger.exception("Error loading from file")
            return False
        else:
            return True

    def reload(self) -> bool:
        """Reload model."""
        return self._load_model()

    def predict(self, features: dict[str, float]) -> float:
        """Make prediction."""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if self.feature_names:
            # Validate all required features present
            missing = [f for f in self.feature_names if f not in features]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")

            # Create DataFrame with correct feature order
            feature_values = [features[f] for f in self.feature_names]
            X = pd.DataFrame([feature_values], columns=self.feature_names)
        else:
            # Fallback: use input order
            X = pd.DataFrame([features])

        prediction = self.model.predict(X)[0]
        return float(prediction)

    @property
    def is_loaded(self) -> bool:
        """Whether model is loaded."""
        return self.model is not None

    @property
    def version_info(self) -> dict[str, str]:
        """Model version."""
        return {
            "model_version": str(self.model_version) if self.model_version else "local-file",
            "model_stage": MODEL_STAGE if self.model_version else "local",
        }


# Create singleton instance
model_service = ModelService()


class PredictionInput(BaseModel):
    """Input features for prediction."""

    features: dict[str, float] = Field(
        ...,
        description="Dictionary of feature names to values",
        example={
            "wave_height_significant_m;75": 2.0,
            "wave_height_max_m;75": 2.4,
            "sea_surface_temperature_degc;75": 10.1,
            "wave_period_peak_s;75": 18.0,
            "wave_period_mean_s;75": 12.0,
            "wave_direction_deg;75": 270.0,
            "directional_wave_spread_deg;75": 40.0,
            "wave_height_significant_m;75_lag_3h": 1.8,
            "wave_height_significant_m;75_lag_6h": 1.4,
            "wave_direction_deg;75_lag_3h": 240.0,
            "wave_direction_deg;75_lag_6h": 260.0,
            "wave_height_significant_m;75_rolling_mean_6h": 1.7,
            "wave_height_significant_m;75_rolling_std_6h": 0.1,
            "wave_direction_deg;75_rolling_mean_6h": 255.0,
            "wave_direction_deg;75_rolling_std_6h": 40.0,
            "hour": 7,
            "month": 12,
        },
    )


class PredictionOutput(BaseModel):
    """Prediction output."""

    predicted_wave_height: float = Field(..., description="Predicted wave height in meters")
    model_version: str = Field(..., description="Model version used for prediction")
    model_stage: str = Field(..., description="Model stage (Production/Staging)")


# FastAPI app
app = FastAPI(
    title="Wave Height Prediction API",
    description="Predict wave height at target buoy using source buoy conditions",
    version="0.0.1",
)


@app.get("/")
def root() -> dict[str, str | bool | None]:
    """Info of the API."""
    version_info = model_service.version_info
    return {
        "message": "Wave Height Prediction API",
        "status": "running",
        "model_loaded": model_service.is_loaded,
        "model_name": MODEL_NAME if model_service.is_loaded else None,
        "model_version": version_info["model_version"],
        "model_stage": version_info["model_stage"],
    }


@app.get("/health")
def health() -> dict[str, str | int]:
    """Health check endpoint."""
    version_info = model_service.version_info
    return {
        "status": "healthy" if model_service.is_loaded else "unhealthy",
        "model_loaded": model_service.is_loaded,
        "model_version": version_info["model_version"],
        "n_features": len(model_service.feature_names) if model_service.feature_names else 0,
    }


@app.get("/features")
def get_features() -> dict[str, list[str]]:
    """Return required feature names."""
    if model_service.feature_names is None:
        raise HTTPException(status_code=503, detail="Feature names not available")
    return {"features": model_service.feature_names}


@app.post("/predict")
def predict(input_data: PredictionInput) -> PredictionOutput:
    """Predict wave height."""
    try:
        # Ensure month & hour are required dtype for MLflow
        for int_feat in ["month", "hour"]:
            if not isinstance(input_data.features[int_feat], np.int32):
                input_data.features[int_feat] = np.int32(input_data.features[int_feat])  # ty: ignore[invalid-assignment]
        prediction = model_service.predict(input_data.features)
        version_info = model_service.version_info

        return PredictionOutput(
            predicted_wave_height=prediction,
            model_version=version_info["model_version"],
            model_stage=version_info["model_stage"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904


@app.post("/reload")
def reload_model() -> dict[str, str]:
    """Reload model (useful for updating to new version)."""
    if model_service.reload():
        version_info = model_service.version_info
        return {
            "status": "success",
            "message": f"Reloaded model version {version_info['model_version']}",
            "model_version": version_info["model_version"],
        }
    raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
