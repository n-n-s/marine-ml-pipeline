"""Data preprocessing."""

# ruff: noqa: N806,N803,S301
import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from marine_ml.constants import CACHE_DIR, DataColumns
from marine_ml.data_loading import get_marine_data
from marine_ml.preprocess import preprocess_data
from marine_ml.utils import load_params

logger = logging.getLogger(__name__)


def get_preprocessed_data() -> tuple[pd.DataFrame, pd.Series]:
    """Get X, y data with engineered features."""
    cache_fp = CACHE_DIR / "preprocessed.pkl"
    if cache_fp.exists():
        logger.info("Loading preprocessed from cache.")
        with cache_fp.open("rb") as f:
            return pickle.load(f)

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

    # Write to cache
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    with cache_fp.open("wb") as f:
        pickle.dump((X, y), f)

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


if __name__ == "__main__":
    X, y = get_preprocessed_data()
