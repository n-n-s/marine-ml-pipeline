"""Useful plots."""

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from windrose import WindroseAxes  # noqa: F401

from marine_ml.constants import DataColumns


def evaluation_plots(
    *, y_test: pd.Series, y_pred: pd.Series, test_r2: float, feature_importance: pd.DataFrame | None, show: bool = False
) -> plt.Figure:
    """Subplots of evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Predicted vs Actual (Test Set)
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, edgecolors="black", linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect prediction")
    axes[0, 0].set_xlabel("Actual Wave Height")
    axes[0, 0].set_ylabel("Predicted Wave Height")
    axes[0, 0].set_title(f"Predicted vs Actual (Test Set)\nR^2 = {test_r2:.3f}")
    axes[0, 0].legend()
    axes[0, 0].grid(visible=True, alpha=0.3)

    # 2. Residuals plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolors="black", linewidth=0.5)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("Predicted Wave Height")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].grid(visible=True, alpha=0.3)

    # 3. Feature importance bar plot (handle model that do not surface feature importance)
    fi_title = "Feature Importance"
    if feature_importance is not None:
        axes[1, 0].barh(
            feature_importance["feature"],
            feature_importance["importance"],
            alpha=0.7,
            edgecolor="black",
            color="steelblue",
        )
        axes[1, 0].set_xlabel("Importance")
        axes[1, 0].grid(visible=True, alpha=0.3, axis="x")
    else:
        fi_title += " - Unavailable for model"
    axes[1, 0].set_title(fi_title)

    # 4. Residuals distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
    axes[1, 1].axvline(x=0, color="r", linestyle="--", lw=2, label="Zero residual")
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Residuals Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(visible=True, alpha=0.3)

    fig.tight_layout()

    if show:
        fig.show()
    else:
        plt.close(fig)

    return fig


def plot_sig_wave_height(df: pd.DataFrame) -> plt.Figure:
    """Plot significant wave height of two buoys."""
    x_col = f"{DataColumns.wave_height_sig.value};75"
    y_col = f"{DataColumns.wave_height_sig.value};107"

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    _df = df.copy()[[x_col, y_col]].dropna()

    # Time series plot
    _df.plot(ax=ax[0])
    ax[0].grid(visible=True, alpha=0.3)

    # Scatter plot
    x = _df[x_col].to_numpy()
    y = _df[y_col].to_numpy()
    slope, intercept, r_value, _p_value, _std_err = stats.linregress(x, y)
    line = slope * x + intercept
    ax[1].scatter(x, y, color="blue", label="Data points", s=1, alpha=0.7)
    ax[1].plot(x, line, color="red", label="Best fit")
    equation = f"y = {slope:.3f}x + {intercept:.3f}"
    r_squared = f"R^2 = {r_value**2:.3f}"
    ax[1].text(
        0.05,
        0.95,
        equation,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.5},
    )
    ax[1].text(
        0.05,
        0.88,
        r_squared,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.5},
    )
    ax[1].grid(visible=True, alpha=0.3)
    ax[1].set_xlabel(x_col)
    ax[1].set_ylabel(y_col)

    fig.tight_layout()
    plt.close(fig)

    return fig


def plot_wave_direction_roses(df: pd.DataFrame) -> plt.Figure:
    """Plot wave direction roses for two buoys using the windrose library."""
    dir_col_75 = f"{DataColumns.wave_dir.value};75"
    dir_col_107 = f"{DataColumns.wave_dir.value};107"
    height_col_75 = f"{DataColumns.wave_height_sig.value};75"
    height_col_107 = f"{DataColumns.wave_height_sig.value};107"

    fig = plt.figure(figsize=(16, 7))

    # Buoy 75
    ax1 = fig.add_subplot(121, projection="windrose")
    ax1.bar(
        df[dir_col_75],
        df[height_col_75],
        normed=True,  # Show as percentages
        opening=0.8,  # Bar width
        edgecolor="white",
        bins=[0, 1, 2, 3, 4, 100],  # Wave height bins
        cmap=plt.cm.YlOrRd,  # ty: ignore[unresolved-attribute]
    )
    ax1.set_title("Wave Rose - Buoy 75", pad=20, fontsize=14, fontweight="bold")
    ax1.set_legend(title="Wave Height (m)", bbox_to_anchor=(1.1, 1.0))  # ty: ignore[unresolved-attribute]

    # Buoy 107
    ax2 = fig.add_subplot(122, projection="windrose")
    ax2.bar(
        df[dir_col_107],
        df[height_col_107],
        normed=True,
        opening=0.8,
        edgecolor="white",
        bins=[0, 1, 2, 3, 4, 100],
        cmap=plt.cm.YlOrRd,  # ty: ignore[unresolved-attribute]
    )
    ax2.set_title("Wave Rose - Buoy 107", pad=20, fontsize=14, fontweight="bold")
    ax2.set_legend(title="Wave Height (m)", bbox_to_anchor=(1.1, 1.0))  # ty: ignore[unresolved-attribute]

    fig.tight_layout()
    plt.close(fig)

    return fig
