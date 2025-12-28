# marine-ml-pipeline
ML pipeline for wave data

## Simple ML Model
Defines a simple regression model to predict the wave height of one buoy based on the data from another.

> [!NOTE]
> The model is essentially a placeholder, to facilitate the ML pipeline build.

## Running the Pipeline

### Manual Execution (for development)

1. Activate virtual env, e.g. `./.venv/bin/activate`
2. `uv run poe run`
3. Navigate to http://localhost:8000/docs and try out the `/predict` endpoint
4. Stop serving the API in the terminal (ctrl+c)
5. Stop mlflow: `uv run poe mlflow-stop`
