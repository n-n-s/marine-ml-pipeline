"""Wave conditions plots."""

import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from windrose import WindroseAxes  # noqa: F401

from marine_ml.constants import DataColumns


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
