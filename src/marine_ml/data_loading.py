"""Module for loading marine data."""

import pandas as pd

from marine_ml.constants import CACHE_DIR, DATA_DIR, DataColumns


def _construct_filename(month: int, year: int) -> str:
    min_acceptable_month = 1
    max_acceptable_month = 12
    if month > max_acceptable_month or month < min_acceptable_month:
        msg = "Invalid month"
        raise ValueError(msg)
    return f"{year}{month:02d}_marine.parquet"


def _load_marine_data(*, site_id: int, month: int, year: int) -> pd.DataFrame:
    """Load data from the wave buoy."""
    fp = DATA_DIR / f"siteid_{site_id}" / _construct_filename(month, year)
    return pd.read_parquet(fp)


def get_marine_data() -> pd.DataFrame:
    """Construct a time series of buoy data with column names suffix with site_id."""
    cache_fp = CACHE_DIR / "marine_data.parquet"
    if cache_fp.exists():
        return pd.read_parquet(cache_fp)

    dataframes = []
    for site_id in [75, 107]:  # 75 = Penzance, 107 = Porthleven
        df_list = [_load_marine_data(site_id=site_id, month=m + 1, year=2025) for m in range(12)]
        dataframes.append(pd.concat(df_list).add_suffix(f";{site_id}"))

    _df = pd.concat(dataframes, axis=1)
    cols_to_keeps = [c for c in _df.columns if any(sub in c for sub in DataColumns.all())]
    data = _df[cols_to_keeps]

    # Write to cache
    cache_fp.parent.mkdir(exist_ok=True, parents=True)
    data.to_parquet(cache_fp)

    return data
