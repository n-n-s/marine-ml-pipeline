"""Model training functions."""

# ruff: noqa: N806
import logging

from marine_ml.helpers.optuna_helper import train_with_optuna
from marine_ml.prepare_data import get_preprocessed_data, split_data

logger = logging.getLogger(__name__)


def create_model() -> str:
    """Define ml model.

    :return: MLflow model_uri
    """
    X, y = get_preprocessed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    return train_with_optuna(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, show_plots=False)


if __name__ == "__main__":
    create_model()
