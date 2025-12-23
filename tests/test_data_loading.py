import pytest

from marine_ml.data_loading import _construct_filename


class TestConstructFilename:
    @staticmethod
    def test_valid() -> None:
        month = 12
        year = 2000
        actual = _construct_filename(month, year)
        assert actual == f"{year}{month}_marine.parquet"

    @staticmethod
    @pytest.mark.parametrize("month", [0, 13])
    def test_invalid_month_raise_error(month: int) -> None:
        with pytest.raises(ValueError, match="Invalid month"):
            _construct_filename(month, 2000)
