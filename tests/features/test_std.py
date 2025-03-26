import numpy as np
import polars as pl
import pytest

from chrono_features.features.std import Std
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


@pytest.fixture
def sample_dataset():
    """Test dataset with two time series (id=1 and id=2)"""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1, 2, 3, 4, 6, 8],  # Values chosen for easy std calculation
        }
    )
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


def test_std_expanding(sample_dataset):
    """Test expanding window standard deviation calculation"""
    transformer = Std(columns="value", window_types=WindowType.EXPANDING())
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1: [NaN (n=1), std(1,2)=0.5, std(1,2,3)=0.816496580927726]
    # For id=2: [NaN (n=1), std(4,6)=1, std(4,6,8)=1.632993161855452]
    expected_values = np.array([np.nan, 0.5, 0.816496580927726, np.nan, 1, 1.632993161855452])

    result_values = transformed_dataset.data["value_std_expanding"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, rtol=1e-7, equal_nan=True)


def test_std_rolling(sample_dataset):
    """Test rolling window standard deviation calculation"""
    transformer = Std(columns="value", window_types=WindowType.ROLLING(size=2))
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1: [NaN (window not full), std(1,2)=0.5, std(2,3)=0.5]
    # For id=2: [NaN (window not full), std(4,6)=1, std(6,8)=1]
    expected_values = np.array([np.nan, 0.5, 0.5, np.nan, 1, 1])

    result_values = transformed_dataset.data["value_std_rolling_2"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, rtol=1e-7, equal_nan=True)


def test_std_dynamic(sample_dataset):
    """Test dynamic window standard deviation calculation"""
    # Add window length column (alternating between 1 and 2)
    sample_dataset.add_feature("window_len", [1, 2, 1, 2, 1, 2])

    transformer = Std(columns="value", window_types=WindowType.DYNAMIC(len_column_name="window_len"))
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For windows of size 1: NaN (std undefined for single value)
    # For windows of size 2: std of current and previous value
    expected_values = np.array([np.nan, 0.5, np.nan, 0.5, np.nan, 1])

    result_values = transformed_dataset.data["value_std_dynamic_based_on_window_len"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, rtol=1e-7, equal_nan=True)


def test_std_multiple_columns(sample_dataset):
    """Test standard deviation calculation for multiple columns"""
    sample_dataset.add_feature("value2", [10, 12, 14, 20, 24, 28])

    transformer = Std(columns=["value", "value2"], window_types=WindowType.EXPANDING())
    transformed_dataset = transformer.transform(sample_dataset)

    # Check results for 'value' column (same as first test)
    expected_values_value = np.array([np.nan, 0.5, 0.816496580927726, np.nan, 1, 1.632993161855452])
    result_values_value = transformed_dataset.data["value_std_expanding"].to_numpy()
    np.testing.assert_allclose(result_values_value, expected_values_value, rtol=1e-7, equal_nan=True)

    # Check results for 'value2' column:
    # For id=1: [NaN, std(10,12)=1, std(10,12,14)=1.632993161855452]
    # For id=2: [NaN, std(20,24)=2, std(20,24,28)=3.265986323710904]
    expected_values_value2 = np.array([np.nan, 1, 1.632993161855452, np.nan, 2, 3.265986323710904])
    result_values_value2 = transformed_dataset.data["value2_std_expanding"].to_numpy()
    np.testing.assert_allclose(result_values_value2, expected_values_value2, rtol=1e-7, equal_nan=True)


def test_std_custom_out_column_names(sample_dataset):
    """Test custom output column names"""
    transformer = Std(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_std",
    )
    transformed_dataset = transformer.transform(sample_dataset)

    expected_values = np.array([np.nan, 0.5, 0.816496580927726, np.nan, 1, 1.632993161855452])
    result_values = transformed_dataset.data["custom_std"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, rtol=1e-7, equal_nan=True)


def test_std_single_value_windows():
    """Test behavior with windows containing single values"""
    data = pl.DataFrame(data={"id": [1, 1, 2, 2], "timestamp": [1, 2, 1, 2], "value": [1, 2, 3, 4]})
    dataset = TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")

    transformer = Std(columns="value", window_types=WindowType.ROLLING(size=1))
    transformed_dataset = transformer.transform(dataset)

    # Standard deviation of single value is NaN
    expected_values = np.array([np.nan, np.nan, np.nan, np.nan])
    result_values = transformed_dataset.data["value_std_rolling_1"].to_numpy()
    np.testing.assert_allclose(result_values, expected_values, equal_nan=True)
