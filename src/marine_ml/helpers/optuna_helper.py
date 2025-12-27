"""Helper functions for Optuna studies."""
# ruff: noqa: N803

import json
import logging
import pickle
from collections.abc import Callable
from functools import partial

import mlflow
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from marine_ml.constants import PROJECT_ROOT_DIR
from marine_ml.plots import evaluation_plots
from marine_ml.utils import load_params

optuna.logging.set_verbosity(optuna.logging.ERROR)  # override Optuna's default logging to reduce verbosity

logger = logging.getLogger(__name__)


def champion_callback(study: Study, frozen_trial: FrozenTrial) -> None:
    """Log only when a new trial iteration improves upon existing best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.

    """
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            msg = (
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
            logger.info(msg)
        else:
            msg = f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            logger.info(msg)


def create_objective(
    *, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, params: dict
) -> Callable:
    """Define Optuna objective function."""
    optuna_params = params["training"]["optuna_params"]

    def objective(trial: Trial) -> float:
        """Optuna objective function."""
        trial_params = {}
        for param_name, param_config in optuna_params.items():
            if param_config["type"] == "int":
                trial_params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
            elif param_config["type"] == "float":
                trial_params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"])

        with mlflow.start_run(nested=True):
            model = RandomForestRegressor(
                **trial_params,
                n_jobs=-1,
                random_state=params["model"]["random_state"],
            )
            model.fit(X=X_train, y=y_train)
            preds = model.predict(X_test)
            error = mean_absolute_error(y_test, preds)

            mlflow.log_params(trial_params)
            mlflow.log_metric("mae", error)

        return error

    return objective


def train_with_optuna_rf(  # noqa: PLR0913
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_id: str,
    run_name: str,
    show_plots: bool = True,
) -> str:
    """Run Optuna study to optimise model, logging to MLflow.

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param experiment_id: MLflow experiment ID
    :param run_name: MLflow run name
    :return: MLflow model_uri
    """
    params = load_params()
    n_trials = params["training"]["n_trials"]

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        logger.info("Hyperparameter optimsation with Optuna")
        study = optuna.create_study(direction="minimize", study_name="wave-height-optimisation")

        # Create a partial function with the data pre-filled
        objective_with_data = partial(
            create_objective(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, params=params)
        )

        study.optimize(objective_with_data, n_trials=n_trials, callbacks=[champion_callback], show_progress_bar=True)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mae", study.best_value)

        mlflow.set_tags(
            tags={
                "project": "PZ from Lev Wave Buoy Project",
                "optimizer_engine": "optuna",
                "model_family": "RandomForest",
                "feature_set_version": 1,
            }
        )

        # Fit the model with best params before logging
        model = RandomForestRegressor(**study.best_params, n_jobs=-1)
        model.fit(X_train, y_train)

        logger.info("Evaluating model")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mlflow.log_metrics({"train_mae": train_mae, "test_mae": test_mae, "train_r2": train_r2, "test_r2": test_r2})

        model_name = "model-pz-to-lev"
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            metadata={"model_data_version": 1},
            registered_model_name=model_name,
            signature=infer_signature(X_train, y_train),
        )

        # Visualisation
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        y_pred = model.predict(X_test)
        fig = evaluation_plots(
            y_test=y_test,
            y_pred=y_pred,
            test_r2=r2_score(y_test, y_pred),
            feature_importance=feature_importance,
            show=show_plots,
        )
        mlflow.log_figure(figure=fig, artifact_file="evaluation_plots.png")

        # Get the run and register model to "Production" stage
        client = mlflow.tracking.MlflowClient()
        # Find the latest model version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = str(max([int(mv.version) for mv in model_versions]))
        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True,  # Archive old production models
        )

        logger.info("Model registered as '%s' version %s", model_name, latest_version)
        logger.info("Promoted to 'Production' stage.")

        # Also save locally as backup
        local_model_output = PROJECT_ROOT_DIR / "models"
        local_model_output.mkdir(exist_ok=True, parents=True)

        with (local_model_output / "model.pkl").open("wb") as f:
            pickle.dump(model, f)
        with (local_model_output / "feature_names.json").open("w") as f:
            json.dump(X_train.columns.tolist(), f, indent=4)  # ty: ignore[invalid-argument-type]

        return mlflow.get_artifact_uri(model_name)
