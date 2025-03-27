import numpy as np
import polars as pl
import pytest

from chrono_features.features import SimpleMovingAverage, WeightedMovingAverage
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def sample_dataset():
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0],
            "timestamp": range(9),
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_weighted_moving_average_basic(sample_dataset):
    # Test with linear weights [1, 2, 3]
    wma = WeightedMovingAverage(columns="value", window_size=3, weights=[1, 2, 3])
    result = wma.transform(sample_dataset)

    expected_values = [
        (1 * 3) / 3,  # window [1] with weights [3]
        (1 * 2 + 2 * 3) / 5,  # window [1, 2] with weights [2, 3] (last 2 of [1,2,3])
        (1 * 1 + 2 * 2 + 3 * 3) / 6,  # window [1, 2, 3]
        (2 * 1 + 3 * 2 + 4 * 3) / 6,  # window [2, 3, 4]
        (3 * 1 + 4 * 2 + 5 * 3) / 6,  # window [3, 4, 5]
        (10 * 3) / 3,  # new series, window [10] with weights [3]
        (10 * 2 + 20 * 3) / 5,  # window [10, 20] with weights [2, 3]
        (10 * 1 + 20 * 2 + 30 * 3) / 6,  # window [10, 20, 30]
        (20 * 1 + 30 * 2 + 40 * 3) / 6,  # window [20, 30, 40]
    ]

    result_values = result.data["value_weighted_moving_average_rolling_3"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True, rtol=1e-5)


def test_weighted_moving_average_only_full_window(sample_dataset):
    # Test with only_full_window=True
    wma = WeightedMovingAverage(columns="value", window_size=3, weights=[1, 2, 3], only_full_window=True)
    result = wma.transform(sample_dataset)

    expected_values = [
        np.nan,  # window too small
        np.nan,  # window too small
        (1 * 1 + 2 * 2 + 3 * 3) / 6,  # first full window
        (2 * 1 + 3 * 2 + 4 * 3) / 6,
        (3 * 1 + 4 * 2 + 5 * 3) / 6,
        np.nan,  # new series
        np.nan,  # window too small
        (10 * 1 + 20 * 2 + 30 * 3) / 6,  # first full window in second series
        (20 * 1 + 30 * 2 + 40 * 3) / 6,
    ]

    result_values = result.data["value_weighted_moving_average_rolling_3"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True, rtol=1e-5)


def test_weighted_moving_average_custom_weights(sample_dataset):
    # Test with custom weights [0.1, 0.3, 0.6]
    wma = WeightedMovingAverage(columns="value", window_size=3, weights=[0.1, 0.3, 0.6], only_full_window=True)
    result = wma.transform(sample_dataset)

    expected_values = [
        np.nan,
        np.nan,
        (1 * 0.1 + 2 * 0.3 + 3 * 0.6) / 1.0,
        (2 * 0.1 + 3 * 0.3 + 4 * 0.6) / 1.0,
        (3 * 0.1 + 4 * 0.3 + 5 * 0.6) / 1.0,
        np.nan,
        np.nan,
        (10 * 0.1 + 20 * 0.3 + 30 * 0.6) / 1.0,
        (20 * 0.1 + 30 * 0.3 + 40 * 0.6) / 1.0,
    ]

    result_values = result.data["value_weighted_moving_average_rolling_3"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True, rtol=1e-5)


def test_weighted_moving_average_invalid_weights():
    # Test with invalid weights
    with pytest.raises(ValueError):
        WeightedMovingAverage(columns="value", window_size=3, weights=[1, 2])  # length mismatch


def test_weighted_moving_average_custom_output_names(sample_dataset):
    # Test with custom output column names
    wma = WeightedMovingAverage(columns="value", window_size=3, out_column_names="custom_wma", weights=[1, 2, 3])
    result = wma.transform(sample_dataset)

    assert "custom_wma" in result.data.columns
    assert len(result.data["custom_wma"]) == len(sample_dataset.data)


def test_weighted_moving_average_falls_back_to_simple(sample_dataset):
    # Test that without weights it uses simple moving average
    wma = WeightedMovingAverage(columns="value", window_size=3, weights=[1, 1, 1])
    result = wma.transform(sample_dataset)

    # Calculate expected simple moving average
    sma = SimpleMovingAverage(columns="value", window_size=3)
    expected = sma.transform(sample_dataset)

    # Should produce same results as simple moving average
    result_values = result.data["value_weighted_moving_average_rolling_3"].to_numpy()
    expected_values = expected.data["value_simple_moving_average_rolling_3"].to_numpy()

    np.testing.assert_array_equal(result_values, expected_values)
