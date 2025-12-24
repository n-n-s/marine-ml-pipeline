"""Model training functions."""

# ruff: noqa: N806, N803
import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from marine_ml.constants import DataColumns
from marine_ml.data_loading import get_marine_data
from marine_ml.utils import load_params

logger = logging.getLogger(__name__)


def get_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Get X, y data."""
    _df = get_marine_data().dropna()
    # Prepare features and target
    feat_cols = _df.columns[_df.columns.str.contains("75")]  # Penzance buoy id
    X = _df[feat_cols]
    y = _df[f"{DataColumns.wave_height_sig.value};107"]  # Porthleven buoy id
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data for train/test.

    :param X: Features dataframe
    :param y: Target series
    :return: X_train, X_test, y_train, y_test
    """
    params = load_params()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"]
    )
    return X_train, X_test, y_train, y_test


def create_model() -> RandomForestRegressor:
    """Define ml model."""
    params = load_params()
    return RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["model"]["random_state"],
        n_jobs=-1,  # Use all CPU cores
    )
