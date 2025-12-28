"""Functions to preprocess the data ready for model training."""

import pandas as pd


def _create_lag_features(df: pd.DataFrame, *, lags: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    """Create lagged features for time series."""
    columns = columns or list(df.columns)
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def _create_rolling_features(df: pd.DataFrame, *, window: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Create rolling statistics."""
    columns = columns or list(df.columns)
    for col in columns:
        df[f"{col}_rolling_mean_{window}h"] = df[col].rolling(window).mean()
        df[f"{col}_rolling_std_{window}h"] = df[col].rolling(window).std()
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df["hour"] = df.index.hour  # ty: ignore[possibly-missing-attribute]
    df["month"] = df.index.month  # ty: ignore[possibly-missing-attribute]
    return df


def preprocess_data(
    df: pd.DataFrame, *, lags: list[int], window: int, columns: list[str] | None = None
) -> pd.DataFrame:
    """Add feature engineered columns to dataframe.

    :param df: dataframe for which to add feature engineered columns to
    :param lags: list of hours to add lag columns for
    :param window: window amount for which to perform rolling mean and std
    :param columns: the columns in df for which to perform the lags and window calculation on
    :return: df with additional feature engineering columns
    """
    return (
        df.pipe(_create_lag_features, lags=lags, columns=columns)
        .pipe(_create_rolling_features, window=window, columns=columns)
        .pipe(_add_temporal_features)
    )
