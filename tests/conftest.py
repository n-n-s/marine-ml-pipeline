import pandas as pd
import pytest

from marine_ml.data_loading import get_marine_data


@pytest.fixture
def marine_data() -> pd.DataFrame:
    return get_marine_data()
