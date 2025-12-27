"""Model training functions."""

# ruff: noqa: N806, N803
import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from marine_ml.constants import DataColumns
from marine_ml.data_loading import get_marine_data
from marine_ml.preprocess import preprocess_data
from marine_ml.utils import load_params

logger = logging.getLogger(__name__)


def get_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Get X, y data."""
    params = load_params()

    feature_buoy = 75  # Penzance buoy id
    target_buoy = 107  # Porthleven buoy id

    _df = get_marine_data()

    # Feature engineering
    feat_cols = _df.columns[_df.columns.str.contains(f";{feature_buoy}")]
    df_feat = preprocess_data(
        _df[feat_cols],
        lags=params["data"]["lag_hours"],
        window=params["data"]["rolling_window"],
        columns=[f"{c};{feature_buoy}" for c in params["data"]["columns_to_lag_and_window"]],
    )

    # Target
    target_col = f"{DataColumns.wave_height_sig.value};{target_buoy}"
    target = _df[[target_col]]

    df_all = pd.concat([df_feat, target], axis=1).dropna()

    X = df_all[df_feat.columns]
    y = df_all[target_col]

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
    """Define ml model without optuna."""
    params = load_params()
    return RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["model"]["random_state"],
        n_jobs=-1,  # Use all CPU cores
    )
