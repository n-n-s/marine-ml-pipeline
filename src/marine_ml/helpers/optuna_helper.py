"""Helper functions for Optuna studies."""

# ruff: noqa: N803
import logging
from functools import partial

import mlflow
import optuna
import pandas as pd
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from marine_ml.plots import evaluation_plots

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


def objective(
    trial: Trial, *, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """Optuna objective function."""
    with mlflow.start_run(nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        bst = RandomForestRegressor(
            **params,
            n_jobs=-1,
            random_state=42,
        )
        bst.fit(X=X_train, y=y_train)
        preds = bst.predict(X_test)
        error = mean_absolute_error(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("mae", error)

    return error


def run_optuna_rf(  # noqa: PLR0913
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_id: str,
    run_name: str,
    n_trials: int = 10,
) -> str:
    """Run Optuna study to optimise model, logging to MLflow.

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param experiment_id: MLflow experiment ID
    :param run_name: MLflow run name
    :param n_trials: Optuna n_trials
    :return: MLflow model_uri
    """
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        study = optuna.create_study(direction="minimize")

        # Create a partial function with the data pre-filled
        objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        study.optimize(objective_with_data, n_trials=n_trials, callbacks=[champion_callback])

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

        artifact_path = "model"
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
            metadata={"model_data_version": 1},
        )

        # Visualisation
        feature_importance = pd.DataFrame(
            {"feature": X_train.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        y_pred = model.predict(X_test)
        fig = evaluation_plots(
            y_test=y_test, y_pred=y_pred, test_r2=r2_score(y_test, y_pred), feature_importance=feature_importance
        )
        mlflow.log_figure(figure=fig, artifact_file="evaluation_plots.png")

        return mlflow.get_artifact_uri(artifact_path)
