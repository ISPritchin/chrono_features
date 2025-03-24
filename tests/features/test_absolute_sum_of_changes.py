import pytest
import numpy as np
import polars as pl

from chrono_features.features.absolute_sum_of_changes import AbsoluteSumOfChanges
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


@pytest.fixture
def sample_dataset():
    """Test dataset fixture with sample time series data."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1, 3, 6, 4, 5, 8],  # Changes: +2, +3, +1, +3
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_absolute_sum_of_changes_expanding(sample_dataset):
    """Test expanding window calculation of absolute sum of changes."""
    transformer = AbsoluteSumOfChanges(
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1: [NaN, |3-1|=2, |6-3| + 2 = 3+2=5]
    # For id=2: [NaN, |5-4|=1, |8-5| + 1 = 3+1=4]
    expected_values = np.array([np.nan, 2, 5, np.nan, 1, 4])

    result_values = transformed_dataset.data["value_sum_expanding"].to_numpy()
    assert np.allclose(result_values, expected_values, equal_nan=True)


def test_absolute_sum_of_changes_rolling(sample_dataset):
    """Test rolling window calculation of absolute sum of changes."""
    transformer = AbsoluteSumOfChanges(
        columns="value",
        window_types=WindowType.ROLLING(size=2),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1: [NaN, |3-1|=2, |6-3|=3]
    # For id=2: [NaN, |5-4|=1, |8-5|=3]
    expected_values = np.array([np.nan, 2, 3, np.nan, 1, 3])

    result_values = transformed_dataset.data["value_sum_rolling_2"].to_numpy()
    assert np.allclose(result_values, expected_values, equal_nan=True)


def test_absolute_sum_of_changes_dynamic(sample_dataset):
    """Test dynamic window calculation of absolute sum of changes."""
    # Add window length column
    sample_dataset.add_feature("window_len", [1, 2, 2, 1, 2, 2])

    transformer = AbsoluteSumOfChanges(
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1:
    #   - window_len=1: NaN (not enough data)
    #   - window_len=2: |3-1|=2
    #   - window_len=2: |6-3|=3
    # For id=2:
    #   - window_len=1: NaN
    #   - window_len=2: |5-4|=1
    #   - window_len=2: |8-5|=3
    expected_values = np.array([np.nan, 2, 3, np.nan, 1, 3])

    result_values = transformed_dataset.data["value_sum_dynamic_based_on_window_len"].to_numpy()
    assert np.allclose(result_values, expected_values, equal_nan=True)


def test_absolute_sum_of_changes_multiple_columns(sample_dataset):
    """Test calculation for multiple columns."""
    sample_dataset.add_feature("value2", [10, 13, 16, 20, 22, 25])  # Changes: +3, +3, +2, +3

    transformer = AbsoluteSumOfChanges(
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Check results for 'value' column (same as first test)
    expected_values_value = np.array([np.nan, 2, 5, np.nan, 1, 4])
    result_values_value = transformed_dataset.data["value_sum_expanding"].to_numpy()
    assert np.allclose(result_values_value, expected_values_value, equal_nan=True)

    # Check results for 'value2' column:
    # For id=1: [NaN, |13-10|=3, |16-13| + 3 = 3+3=6]
    # For id=2: [NaN, |22-20|=2, |25-22| + 2 = 3+2=5]
    expected_values_value2 = np.array([np.nan, 3, 6, np.nan, 2, 5])
    result_values_value2 = transformed_dataset.data["value2_sum_expanding"].to_numpy()
    assert np.allclose(result_values_value2, expected_values_value2, equal_nan=True)


def test_absolute_sum_of_changes_custom_out_column_names(sample_dataset):
    """Test support for custom output column names."""
    transformer = AbsoluteSumOfChanges(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_abs_sum",
    )
    transformed_dataset = transformer.transform(sample_dataset)

    expected_values = np.array([np.nan, 2, 5, np.nan, 1, 4])
    result_values = transformed_dataset.data["custom_abs_sum"].to_numpy()
    assert np.allclose(result_values, expected_values, equal_nan=True)
