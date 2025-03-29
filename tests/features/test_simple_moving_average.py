import numpy as np
import polars as pl
import pytest

from chrono_features.features.simple_moving_average import SimpleMovingAverage
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50, 5, 15, 25, 35, 45],
            "other_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_simple_moving_average_with_custom_output_name():
    """Test SimpleMovingAverage with custom output column name."""
    sma = SimpleMovingAverage(columns="value", window_size=3, out_column_names="custom_sma")

    # Check that output column name is correctly set
    assert sma.out_column_names == ["custom_sma"]


def test_simple_moving_average_with_multiple_columns():
    """Test SimpleMovingAverage with multiple input columns."""
    sma = SimpleMovingAverage(columns=["value", "other_value"], window_size=2)

    # Check that columns are correctly set
    assert sma.columns == ["value", "other_value"]


def test_simple_moving_average_with_multiple_output_names():
    """Test SimpleMovingAverage with multiple output column names."""
    sma = SimpleMovingAverage(
        columns=["value", "other_value"],
        window_size=2,
        out_column_names=["value_sma", "other_value_sma"],
    )

    # Check that output column names are correctly set
    assert sma.out_column_names == ["value_sma", "other_value_sma"]


def test_simple_moving_average_invalid_window_size():
    """Test SimpleMovingAverage with invalid window size."""
    # Test with zero window size
    with pytest.raises(ValueError):
        SimpleMovingAverage(columns="value", window_size=0)

    # Test with negative window size
    with pytest.raises(ValueError):
        SimpleMovingAverage(columns="value", window_size=-1)


def test_simple_moving_average_calculation(sample_dataset):
    """Test SimpleMovingAverage calculation results."""
    # Create SMA with window size 3
    sma = SimpleMovingAverage(columns="value", window_size=3, only_full_window=True)

    # Apply transformation
    result = sma.transform(sample_dataset)

    # Expected values for window size 3
    # For id=1: [NaN, NaN, (10+20+30)/3, (20+30+40)/3, (30+40+50)/3]
    # For id=2: [NaN, NaN, (5+15+25)/3, (15+25+35)/3, (25+35+45)/3]
    expected_values = [np.nan, np.nan, 20.0, 30.0, 40.0, np.nan, np.nan, 15.0, 25.0, 35.0]  # For id=1  # For id=2

    # Check results
    result_values = result.data["value_simple_moving_average_3"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True)


def test_simple_moving_average_with_full_window_calculation(sample_dataset):
    """Test SimpleMovingAverage calculation with only_full_window=True."""
    # Create SMA with window size 3 and only_full_window=True
    sma = SimpleMovingAverage(columns="value", window_size=3, only_full_window=True)

    # Apply transformation
    result = sma.transform(sample_dataset)

    # Expected values for window size 3 with only_full_window=True
    # For id=1: [NaN, NaN, (10+20+30)/3, (20+30+40)/3, (30+40+50)/3]
    # For id=2: [NaN, NaN, (5+15+25)/3, (15+25+35)/3, (25+35+45)/3]
    # Note: With only_full_window=True, the first two values for each id should be NaN
    expected_values = [np.nan, np.nan, 20.0, 30.0, 40.0, np.nan, np.nan, 15.0, 25.0, 35.0]  # For id=1  # For id=2

    # Check results
    result_values = result.data["value_simple_moving_average_3"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True)


def test_simple_moving_average_window_size_one(sample_dataset):
    """Test SimpleMovingAverage with window size 1."""
    # Create SMA with window size 1
    sma = SimpleMovingAverage(columns="value", window_size=1)

    # Apply transformation
    result = sma.transform(sample_dataset)

    # With window size 1, SMA should be the same as the original values
    expected_values = [10, 20, 30, 40, 50, 5, 15, 25, 35, 45]

    # Check results
    result_values = result.data["value_simple_moving_average_1"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values)


def test_simple_moving_average_window_size_larger_than_data(sample_dataset):
    """Test SimpleMovingAverage with window size larger than available data."""
    # Create SMA with window size 10 (larger than data for each id)
    sma = SimpleMovingAverage(columns="value", window_size=10, only_full_window=True)

    # Apply transformation
    result = sma.transform(sample_dataset)

    # Expected values: all NaN because window size is larger than data for each id
    expected_values = np.full(10, np.nan)

    # Check results
    result_values = result.data["value_simple_moving_average_10"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True)


def test_simple_moving_average_with_missing_values():
    """Test SimpleMovingAverage with missing values in the data."""
    # Create dataset with missing values
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1],
            "timestamp": [1, 2, 3, 4, 5],
            "value": [10.0, np.nan, 30.0, 40.0, 50.0],
        },
    )
    dataset = TSDataset(data, id_column_name="id", ts_column_name="timestamp")

    # Create SMA with window size 3
    sma = SimpleMovingAverage(columns="value", window_size=3, only_full_window=False)

    # Apply transformation
    result = sma.transform(dataset)

    # Expected values with None in the data
    # [NaN, NaN, NaN, NaN, (30+40+50)/3]
    expected_values = [10.0, np.nan, np.nan, np.nan, 40.0]

    # Check results
    result_values = result.data["value_simple_moving_average_3"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True)


def test_simple_moving_average_with_multiple_columns_calculation(sample_dataset):
    """Test SimpleMovingAverage calculation with multiple columns."""
    # Create SMA with window size 2 for multiple columns
    sma = SimpleMovingAverage(columns=["value", "other_value"], window_size=2, only_full_window=True)

    # Apply transformation
    result = sma.transform(sample_dataset)

    # Expected values for "value" with window size 2
    # For id=1: [NaN, (10+20)/2, (20+30)/2, (30+40)/2, (40+50)/2]
    # For id=2: [NaN, (5+15)/2, (15+25)/2, (25+35)/2, (35+45)/2]
    expected_value_sma = [np.nan, 15.0, 25.0, 35.0, 45.0, np.nan, 10.0, 20.0, 30.0, 40.0]  # For id=1  # For id=2

    # Expected values for "other_value" with window size 2
    # For id=1: [NaN, (1+2)/2, (2+3)/2, (3+4)/2, (4+5)/2]
    # For id=2: [NaN, (6+7)/2, (7+8)/2, (8+9)/2, (9+10)/2]
    expected_other_value_sma = [np.nan, 1.5, 2.5, 3.5, 4.5, np.nan, 6.5, 7.5, 8.5, 9.5]  # For id=1  # For id=2

    # Check results
    value_sma = result.data["value_simple_moving_average_2"].to_numpy()
    other_value_sma = result.data["other_value_simple_moving_average_2"].to_numpy()

    np.testing.assert_allclose(value_sma, expected_value_sma, equal_nan=True)
    np.testing.assert_allclose(other_value_sma, expected_other_value_sma, equal_nan=True)
