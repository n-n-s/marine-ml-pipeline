"""Useful plots."""

import matplotlib.pyplot as plt
import pandas as pd


def evaluation_plots(
    *,
    y_test: pd.Series,
    y_pred: pd.Series,
    test_r2: float,
    feature_importance: pd.DataFrame,
) -> plt.Figure:
    """Subplots of evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Predicted vs Actual (Test Set)
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
    axes[0, 0].set_xlabel("Actual Wave Height")
    axes[0, 0].set_ylabel("Predicted Wave Height")
    axes[0, 0].set_title(f"Predicted vs Actual (Test Set)\nR^2 = {test_r2:.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(visible=True, alpha=0.3)

    # 2. Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidth=0.5)
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
    axes[1, 1].hist(y_pred, bins=30, alpha=0.5, label="Predicted", edgecolor="black")
    axes[1, 1].set_xlabel("Wave Height")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution: Actual vs Predicted")
    axes[1, 1].legend()
    axes[1, 1].grid(visible=True, alpha=0.3)

    fig.tight_layout()

    return fig
