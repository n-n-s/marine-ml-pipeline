"""Mapping plots."""

from typing import NamedTuple, no_type_check

from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class Location(NamedTuple):
    """Represents a buoy location for plotting."""

    name: str
    latitude: float
    longitude: float


def _calculate_extent(locations: list[Location], padding_factor: float = 0.1, min_padding: float = 1.5) -> list[float]:
    """Calculate map extent dynamically based on given locations."""
    lats = [loc.latitude for loc in locations]
    lons = [loc.longitude for loc in locations]

    lon_range = max(lons) - min(lons)
    lat_range = max(lats) - min(lats)

    lon_pad = max(lon_range * padding_factor, min_padding)
    lat_pad = max(lat_range * padding_factor, min_padding)

    return [min(lons) - lon_pad, max(lons) + lon_pad, min(lats) - lat_pad, max(lats) + lat_pad]


def _calculate_context_extent(main_extent: list[float], scale_factor: float = 3.0) -> list[float]:
    """Calculate context map extent based on main map extent.

    :param main_extent: [lon_min, lon_max, lat_min, lat_max]
    :param scale_factor: How much larger the context should be (default 3x)
    :return: Context extent clamped to valid lat/lon ranges
    """
    center_lon = (main_extent[0] + main_extent[1]) / 2
    center_lat = (main_extent[2] + main_extent[3]) / 2

    width = (main_extent[1] - main_extent[0]) * scale_factor
    height = (main_extent[3] - main_extent[2]) * scale_factor

    context_extent = [center_lon - width / 2, center_lon + width / 2, center_lat - height / 2, center_lat + height / 2]

    # Clamp to valid lat/lon ranges
    context_extent[0] = max(context_extent[0], -180)
    context_extent[1] = min(context_extent[1], 180)
    context_extent[2] = max(context_extent[2], -90)
    context_extent[3] = min(context_extent[3], 90)

    return context_extent


@no_type_check
def plot_buoys_on_map(
    location_1: Location,
    location_2: Location,
    *,
    add_context_map: bool = True,
    context_scale_factor: float = 3.0,
) -> plt.Figure:
    """Plot two locations on a map with coastlines and features.

    :param location_1: First location to plot
    :param location_2: Second location to plot
    :param add_context_map: Whether to add a mini context map (default True)
    :param context_scale_factor: How much larger the context map should be (default 3x)
    :return: figure object
    """
    # Create figure and main axis with map projection
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add map features to main map
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

    # Plot the locations
    ax.plot(
        location_1.longitude,
        location_1.latitude,
        "ro",
        markersize=8,
        transform=ccrs.PlateCarree(),
        label=location_1.name,
        markeredgecolor="darkred",
        markeredgewidth=2,
    )
    ax.plot(
        location_2.longitude,
        location_2.latitude,
        "bs",
        markersize=8,
        transform=ccrs.PlateCarree(),
        label=location_2.name,
        markeredgecolor="darkblue",
        markeredgewidth=2,
    )

    # Set extent dynamically for main map
    extent = _calculate_extent([location_1, location_2])
    ax.set_extent(extent)

    # Add gridlines to main map
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Buoy Locations")

    # Add context mini-map
    if add_context_map:
        # Create inset axis in lower left corner
        # Position: [left, bottom, width, height] in figure coordinates
        ax_inset = fig.add_axes((0.23, 0.115, 0.25, 0.25), projection=ccrs.PlateCarree())

        # Add simple features to context map
        ax_inset.add_feature(cfeature.LAND, facecolor="tan", edgecolor="none")
        ax_inset.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.3)

        # Calculate and set context extent dynamically
        context_extent = _calculate_context_extent(extent, scale_factor=context_scale_factor)
        ax_inset.set_extent(context_extent, crs=ccrs.PlateCarree())

        # Draw rectangle showing main map extent
        rect = Rectangle(
            (extent[0], extent[2]),  # (lon_min, lat_min)
            extent[1] - extent[0],  # width
            extent[3] - extent[2],  # height
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax_inset.add_patch(rect)

        # Mark the locations on context map
        ax_inset.plot(location_1.longitude, location_1.latitude, "ro", markersize=4, transform=ccrs.PlateCarree())
        ax_inset.plot(location_2.longitude, location_2.latitude, "bs", markersize=4, transform=ccrs.PlateCarree())

        # Add border to inset
        ax_inset.spines["geo"].set_edgecolor("black")
        ax_inset.spines["geo"].set_linewidth(1.5)

    return fig
