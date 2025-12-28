# marine-ml-pipeline
Machine learning pipeline for predicting wave height at coastal monitoring locations.
Demonstrates end-to-end MLOps practices.

[![Test](https://github.com/n-n-s/marine-ml-pipeline/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/n-n-s/marine-ml-pipeline/actions/workflows/ci.yaml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

## Overview

This project predicts wave height at one buoy location using oceanographic and meteorological data from another
location. While the ML problem itself is intentionally straightforward, the focus is on demonstrating ML engineering
practices and the transition from exploratory research to deployable systems.

## What this project demonstrates

- **Model Experiment Tracking**: Comprehensive logging with `mlflow`
- **Model Hyperparameter Optimisation**: Automated tuning with `optuna`
- **Model Registry**: Version-controlled model deployment via `mlflow`
- **Model serving via API Deployment**: Production-ready REST API with `FastAPI`

## Data Source

Real-time coastal monitoring data from the **UK National Network of Regional Coastal Monitoring Programmes**,
made freely available under the terms of the
[Open Government Licence](https://coastalmonitoring.org/CCO_OGL.pdf). 

Detailed information: [data/README.md](data/README.md).

## Quickstart

### Prerequisites

[uv](https://docs.astral.sh/uv/) package manager

### Installation

- clone the repository
- `cd` into the project directory and run `uv sync`

### Running the Pipeline

#### Using a Task Runner

As the pipeline is straightforward, a simple `poethepoet` task is configured.

1. Activate virtual env, e.g. `./.venv/bin/activate`
2. `uv run poe run`
3. Navigate to http://localhost:8000/docs and try out the `/predict` endpoint
4. Stop serving the API in the terminal (ctrl+c)
5. Stop mlflow: `uv run poe mlflow-stop`

#### Manual Execution

Start the MLflow UI, prepare data and train model with hyperparameter optimisation by
running:

```bash
# Start mlflow ui
nohup uv run mlflow ui > mlflow.log 2>&1 &

# Preprocess data
uv run python src/marine_ml/prepare_data.py

# Train model with hyperparameter optimisation
uv run python src/marine_ml/train.py
```

You may view experiments in MLflow UI at http://localhost:5000 to see:
- All training runs with hyperparameters
- Optuna trial comparisons
- Model performance metrics
- Model registry and versions

Start the API server:

```bash
# Start the API server
uv run python src/marine_ml/serve.py
```

You may trial the available endpoints in the interactive API documentation at http://localhost:8000/docs :

- GET /health - Service health check
- GET /features - List required input features
- POST /predict - Make wave height predictions
- POST /reload - Reload model from registry (zero-downtime updates)

Finally, stop the mlflow service using `pkill -f 'mlflow'`.


## Design Decisions

### Why This Architecture?

- **Task runner for orchestration:** sufficient for this simple pipeline.
- **MLflow Model Registry:** Enables version-controlled deployments, A/B testing capability, and instant rollbacks - critical for production ML systems.
- **Optuna integration:** Systematic hyperparameter optimisation rather than manual tuning, with full experiment tracking.
- **FastAPI over MLflow serving:** Provides production-ready features (custom validation, error handling, health checks) not available in MLflow's basic serving.

Current architecture is using:

- **Local** MLflow tracking and model registry
- **Single-instance** API serving
- **File-based** model storage

To scale, architecture would extend to a cloud platform, e.g. Azure.
This would also enable monitoring of the model, e.g. for drift detection, in addition to automated model retraining
triggers.

### R&D to Deployment Journey
This project demonstrates the transition from exploratory work to deployed systems:

- Exploration (`notebooks/`): Initial data analysis and prototyping
- Refactoring (`src/marine_ml`): Modular, testable, production code
- Monitoring (`MLflow`): Experiment tracking and model registry
- Deployment (`FastAPI`): Production-ready inference service

### Configuration

Hyperparameters and settings are defined in `params.yaml` for flexibility.

## Testing

Test suite includes data validation, model performance checks, and full pipeline integration tests.
Run the test suite using `uv run poe test`.

## Tech Stack

- **Python 3.12**
- **MLflow** - Experiment tracking and model registry
- **Optuna** - Hyperparameter optimization
- **FastAPI** - REST API serving
- **scikit-learn** - Model training
- **pytest** - Testing
- **uv** - Dependency management
