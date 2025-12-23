"""Simple model to predict significant wave height of one buoy based on another buoy's data."""

# ruff: noqa: W293, N806, N803
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from marine_ml.constants import PROJECT_ROOT_DIR, DataColumns
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


def example_run() -> None:  # noqa: PLR0915
    """Train and predict an example."""
    level = logger.getEffectiveLevel()
    logging.basicConfig(level=logging.INFO)  # temporarily set log level for this function
    model = create_model()
    X, y = get_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    log_msg = f"""Model performance...
    
    Training Set:
      RMSE: {train_rmse:.4f}
      MAE:  {train_mae:.4f}
      R^2:   {train_r2:.4f}
      
    Test Set:
      RMSE: {test_rmse:.4f}
      MAE:  {test_mae:.4f}
      R^2:   {test_r2:.4f}
    """
    logger.info(log_msg)

    # Feature importance
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    log_msg = f"Feature Importance...\n{feature_importance}"
    logger.info(log_msg)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Predicted vs Actual (Test Set)
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
    axes[0, 0].set_xlabel("Actual Wave Height")
    axes[0, 0].set_ylabel("Predicted Wave Height")
    axes[0, 0].set_title(f"Predicted vs Actual (Test Set)\nR^2 = {test_r2:.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(visible=True, alpha=0.3)

    # 2. Residuals plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.5, edgecolors="k", linewidth=0.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Predicted Wave Height")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].grid(visible=True, alpha=0.3)

    # 3. Feature importance bar plot
    axes[1, 0].barh(feature_importance["feature"], feature_importance["importance"])
    axes[1, 0].set_xlabel("Importance")
    axes[1, 0].set_title("Feature Importance")
    axes[1, 0].grid(visible=True, alpha=0.3, axis="x")

    # 4. Distribution of predictions vs actual
    axes[1, 1].hist(y_test, bins=30, alpha=0.5, label="Actual", edgecolor="black")
    axes[1, 1].hist(y_test_pred, bins=30, alpha=0.5, label="Predicted", edgecolor="black")
    axes[1, 1].set_xlabel("Wave Height")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution: Actual vs Predicted")
    axes[1, 1].legend()
    axes[1, 1].grid(visible=True, alpha=0.3)

    fig.tight_layout()

    output_file = PROJECT_ROOT_DIR / "output/model_evaluation.png"
    output_file.parent.mkdir(exist_ok=True, parents=True)

    fig.savefig(output_file)
    logger.info("Visualization saved to output folder.")

    logging.basicConfig(level=level)  # reset log level to original


if __name__ == "__main__":
    example_run()
