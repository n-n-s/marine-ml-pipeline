# marine-ml-pipeline
ML pipeline for wave data

[![Test](https://github.com/n-n-s/marine-ml-pipeline/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/n-n-s/marine-ml-pipeline/actions/workflows/ci.yaml)

## Simple ML Model
Defines a simple regression model to predict the wave height of one buoy based on the data from another.

> [!NOTE]
> The model is essentially a placeholder, to facilitate the ML pipeline build.

## Running the Pipeline

### Manual Execution

1. Activate virtual env, e.g. `./.venv/bin/activate`
2. `uv run poe run`
3. Navigate to http://localhost:8000/docs and try out the `/predict` endpoint
4. Stop serving the API in the terminal (ctrl+c)
5. Stop mlflow: `uv run poe mlflow-stop`
