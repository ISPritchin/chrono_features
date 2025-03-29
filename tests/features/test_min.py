import numpy as np
import polars as pl
import pytest

from chrono_features.features.min import Min
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


@pytest.fixture
def sample_dataset() -> TSDataset:
    """Creates a test dataset with time series."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "timestamp": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "value": [5.0, 3.0, 1.0, 4.0, 2.0, 6.0, 10.0, 4.0, 7.0, 3.0, 8.0, 1.0],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_min_expanding(sample_dataset: TSDataset) -> None:
    """Tests minimum calculation with expanding window."""
    min_feature = Min(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="min_expanding",
    )

    result = min_feature.transform(sample_dataset)
    result_values = result.data["min_expanding"].to_numpy()

    # Expected values for expanding window
    expected = np.array(
        [
            # First time series [5.0, 3.0, 1.0, 4.0, 2.0, 6.0]
            5.0,  # min of [5.0]
            3.0,  # min of [5.0, 3.0]
            1.0,  # min of [5.0, 3.0, 1.0]
            1.0,  # min of [5.0, 3.0, 1.0, 4.0]
            1.0,  # min of [5.0, 3.0, 1.0, 4.0, 2.0]
            1.0,  # min of [5.0, 3.0, 1.0, 4.0, 2.0, 6.0]
            # Second time series [10.0, 4.0, 7.0, 3.0, 8.0, 1.0]
            10.0,  # min of [10.0]
            4.0,  # min of [10.0, 4.0]
            4.0,  # min of [10.0, 4.0, 7.0]
            3.0,  # min of [10.0, 4.0, 7.0, 3.0]
            3.0,  # min of [10.0, 4.0, 7.0, 3.0, 8.0]
            1.0,  # min of [10.0, 4.0, 7.0, 3.0, 8.0, 1.0]
        ],
    )

    np.testing.assert_array_almost_equal(result_values, expected)


def test_min_rolling(sample_dataset: TSDataset) -> None:
    """Tests minimum calculation with rolling window of size 3."""
    min_feature = Min(
        columns="value",
        window_types=WindowType.ROLLING(size=3, only_full_window=False),
        out_column_names="min_rolling",
    )

    result = min_feature.transform(sample_dataset)
    result_values = result.data["min_rolling"].to_numpy()

    # Expected values for rolling window of size 3
    expected = np.array(
        [
            # First time series [5.0, 3.0, 1.0, 4.0, 2.0, 6.0]
            5.0,  # min of [5.0]
            3.0,  # min of [5.0, 3.0]
            1.0,  # min of [5.0, 3.0, 1.0]
            1.0,  # min of [3.0, 1.0, 4.0]
            1.0,  # min of [1.0, 4.0, 2.0]
            2.0,  # min of [4.0, 2.0, 6.0]
            # Second time series [10.0, 4.0, 7.0, 3.0, 8.0, 1.0]
            10.0,  # min of [10.0]
            4.0,  # min of [10.0, 4.0]
            4.0,  # min of [10.0, 4.0, 7.0]
            3.0,  # min of [4.0, 7.0, 3.0]
            3.0,  # min of [7.0, 3.0, 8.0]
            1.0,  # min of [3.0, 8.0, 1.0]
        ],
    )

    np.testing.assert_array_almost_equal(result_values, expected)


def test_min_dynamic(sample_dataset: TSDataset) -> None:
    """Tests minimum calculation with dynamic window."""
    # Add a column with window lengths
    sample_dataset.add_feature("window_len", [1, 2, 3, 2, 3, 4, 1, 2, 3, 2, 3, 4])

    min_feature = Min(
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
        out_column_names="min_dynamic",
    )

    result = min_feature.transform(sample_dataset)
    result_values = result.data["min_dynamic"].to_numpy()

    # Expected values for dynamic window
    expected = np.array(
        [
            # First time series with window lengths [1, 2, 3, 2, 3, 4]
            5.0,  # min of [5.0] (window length 1)
            3.0,  # min of [5.0, 3.0] (window length 2)
            1.0,  # min of [5.0, 3.0, 1.0] (window length 3)
            1.0,  # min of [3.0, 1.0] (window length 2)
            1.0,  # min of [1.0, 4.0, 2.0] (window length 3)
            1.0,  # min of [3.0, 1.0, 4.0, 2.0] (window length 4)
            # Second time series with window lengths [1, 2, 3, 2, 3, 4]
            10.0,  # min of [10.0] (window length 1)
            4.0,  # min of [10.0, 4.0] (window length 2)
            4.0,  # min of [10.0, 4.0, 7.0] (window length 3)
            3.0,  # min of [7.0, 3.0] (window length 2)
            3.0,  # min of [7.0, 3.0, 8.0] (window length 3)
            1.0,  # min of [3.0, 8.0, 1.0, 0.0] (window length 4)
        ],
    )

    np.testing.assert_array_almost_equal(result_values, expected)


def test_min_multiple_columns(sample_dataset: TSDataset) -> None:
    """Tests minimum calculation on multiple columns."""
    # Add a second column to the dataset
    sample_dataset.add_feature("value2", [10, 8, 6, 4, 2, 0, 0, 2, 4, 6, 8, 10])

    min_feature = Min(
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
        out_column_names=["min_value", "min_value2"],
    )

    result = min_feature.transform(sample_dataset)

    # Check first column
    result_values1 = result.data["min_value"].to_numpy()
    expected1 = np.array(
        [
            5.0,
            3.0,
            1.0,
            1.0,
            1.0,
            1.0,  # First time series
            10.0,
            4.0,
            4.0,
            3.0,
            3.0,
            1.0,  # Second time series
        ],
    )
    np.testing.assert_array_almost_equal(result_values1, expected1)

    # Check second column
    result_values2 = result.data["min_value2"].to_numpy()
    expected2 = np.array(
        [
            10,
            8,
            6,
            4,
            2,
            0,  # First time series
            0,
            0,
            0,
            0,
            0,
            0,  # Second time series
        ],
    )
    np.testing.assert_array_almost_equal(result_values2, expected2)


def test_min_zero_length_window(sample_dataset: TSDataset) -> None:
    """Tests minimum calculation with zero-length windows."""
    # Add a column with some zero window lengths
    sample_dataset.add_feature("window_len", [0, 1, 2, 0, 3, 4, 0, 1, 2, 0, 3, 4])

    min_feature = Min(
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
        out_column_names="min_dynamic",
    )

    result = min_feature.transform(sample_dataset)
    result_values = result.data["min_dynamic"].to_numpy()
    # For zero-length windows, we expect NaN values
    expected_values = np.array(
        [
            np.nan,  # window length 0
            3,  # window length 1
            1,  # window length 2
            np.nan,  # window length 0
            1,  # window length 3
            1,  # window length 4
            np.nan,  # window length 0
            4,  # window length 1
            4,  # window length 2
            np.nan,  # window length 0
            3,  # window length 3
            1,  # window length 4
        ],
    )

    # Use assert_array_equal to handle NaN values correctly
    assert np.allclose(result_values, expected_values, equal_nan=True)
