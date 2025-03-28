import numpy as np
import polars as pl
import pytest

from chrono_features.features.autocorrelation import Autocorrelation
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


@pytest.fixture
def sample_dataset() -> TSDataset:
    """Creates a test dataset with time series."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "timestamp": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "value": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 4.0, 3.0, 7.0, 4.0, 10.0],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_autocorrelation_lag1_expanding(sample_dataset: TSDataset) -> None:
    """Tests autocorrelation with lag=1 on an expanding window."""
    autocorr = Autocorrelation(
        columns="value",
        window_types=WindowType.EXPANDING(),
        lag=1,
        out_column_names="autocorr_lag1_expanding",
    )

    result = autocorr.transform(sample_dataset)
    result_values = result.data["autocorr_lag1_expanding"].to_numpy()

    # Expected values for expanding window with lag=1
    expected = np.array(
        [
            # [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
            np.nan,  # first element - nan (not enough data)
            np.nan,  # correlation between [1] and [3] (not enough data)
            1.0,  # correlation between [1, 3] and [3, 5]
            -0.32732683535398865,  # correlation between [1, 3, 5] and [3, 5, 2]
            -0.3779644730092272,  # correlation between [1, 3, 5, 2] and [3, 5, 2, 4]
            0,  # correlation between [1, 3, 5, 2, 4] and [3, 5, 2, 4, 6]
            # [1.0, 4.0, 3.0, 7.0, 4.0, 10.0]
            np.nan,  # first element of new time series
            np.nan,  # correlation between [5] and [4] (not enough data)
            -1.0,  # correlation between [1, 4] and [4, 3]
            -0.05241424183609577,  # correlation between [1, 4, 3] and [4, 3, 7]
            -0.19245008972987526,  # correlation between [1, 4, 3, 7] and [4, 3, 7, 4]
            -0.05603766997559098,  # correlation between [1, 4, 3, 7, 4] and [4, 3, 7, 4, 10]
        ],
    )

    np.testing.assert_array_almost_equal(result_values, expected, decimal=5)


def test_autocorrelation_lag2_rolling(sample_dataset: TSDataset) -> None:
    """Tests autocorrelation with lag=2 on a rolling window of size 5."""
    autocorr = Autocorrelation(
        columns="value",
        window_types=WindowType.ROLLING(size=5),
        lag=2,
        out_column_names="autocorr_lag2_rolling",
    )

    result = autocorr.transform(sample_dataset)
    result_values = result.data["autocorr_lag2_rolling"].to_numpy()

    # Expected values for rolling window of size 4 with lag=2
    expected = np.array(
        [
            np.nan,  # not enough data
            np.nan,  # not enough data
            np.nan,  # not enough data
            np.nan,  # not enough data
            -0.32732683535398865,  # correlation between [1, 3, 5], [5, 2, 4] in window [1, 3, 5, 2, 4]
            -0.32732683535398865,  # correlation between [3, 5, 2] and [2, 4, 6] in window [3, 5, 2, 4, 6]
            np.nan,  # not enough data
            np.nan,  # not enough data
            np.nan,  # not enough data
            np.nan,  # not enough data
            0.8910421112136309,  # correlation between [1, 4, 3], [3, 7, 4] in window [1, 4, 3, 7, 4]
            0.9607689228305227,  # correlation between [4, 3, 7], [7, 4, 10] in window [4, 3, 7, 4, 10]
        ],
    )

    np.testing.assert_array_almost_equal(result_values, expected, decimal=5)


def test_autocorrelation_multiple_columns(sample_dataset: TSDataset) -> None:
    """Tests autocorrelation on multiple columns."""
    # Add a second column to the dataset
    sample_dataset.add_feature("value2", [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1])

    autocorr = Autocorrelation(
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
        lag=1,
        out_column_names=["autocorr_value", "autocorr_value2"],
    )

    result = autocorr.transform(sample_dataset)

    # Check first column (same as in first test)
    result_values1 = result.data["autocorr_value"].to_numpy()
    expected1 = np.array(
        [
            # [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
            np.nan,  # first element - nan (not enough data)
            np.nan,  # correlation between [1] and [3] (not enough data)
            1.0,  # correlation between [1, 3] and [3, 5]
            -0.32732683535398865,  # correlation between [1, 3, 5] and [3, 5, 2]
            -0.3779644730092272,  # correlation between [1, 3, 5, 2] and [3, 5, 2, 4]
            0,  # correlation between [1, 3, 5, 2, 4] and [3, 5, 2, 4, 6]
            # [1.0, 4.0, 3.0, 7.0, 4.0, 10.0]
            np.nan,  # first element of new time series
            np.nan,  # correlation between [5] and [4] (not enough data)
            -1.0,  # correlation between [1, 4] and [4, 3]
            -0.05241424183609577,  # correlation between [1, 4, 3] and [4, 3, 7]
            -0.19245008972987526,  # correlation between [1, 4, 3, 7] and [4, 3, 7, 4]
            -0.05603766997559098,  # correlation between [1, 4, 3, 7, 4] and [4, 3, 7, 4, 10]
        ],
    )
    np.testing.assert_array_almost_equal(result_values1, expected1, decimal=5)

    # Check second column (should be all NaN since values are constant)
    result_values2 = result.data["autocorr_value2"].to_numpy()
    expected2 = np.array([np.nan, np.nan, 1, 1, 1, 1, np.nan, np.nan, 1, 1, 1, 1])
    np.testing.assert_array_equal(result_values2, expected2)


def test_autocorrelation_dynamic_window(sample_dataset: TSDataset) -> None:
    """Tests autocorrelation with a dynamic window."""
    # Add a column with window lengths
    sample_dataset.add_feature("window_len", [1, 2, 2, 3, 3, 4, 1, 2, 2, 3, 3, 4])

    autocorr = Autocorrelation(
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
        lag=1,
        out_column_names="autocorr_dynamic",
    )

    result = autocorr.transform(sample_dataset)
    result_values = result.data["autocorr_dynamic"].to_numpy()

    # Expected values for dynamic window
    expected = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            -1,
            -1,
            -0.32733,
            np.nan,
            np.nan,
            np.nan,
            -1,
            -1,
            -0.72058,
        ],
    )

    np.testing.assert_array_almost_equal(result_values, expected, decimal=5)


def test_autocorrelation_invalid_lag():
    """Tests handling of invalid lag (should be >= 1)."""
    with pytest.raises(ValueError):
        Autocorrelation(
            columns="value",
            window_types=WindowType.EXPANDING,
            lag=0,
            out_column_names="autocorr_invalid",
        )


def test_autocorrelation_with_negative_lag():
    """Tests that autocorrelation raises an error with negative lag values."""
    with pytest.raises(ValueError, match="Lag must be greater than 0"):
        Autocorrelation(
            columns="value",
            window_types=WindowType.EXPANDING(),
            lag=-1,
            out_column_names="autocorr_negative_lag",
        )
