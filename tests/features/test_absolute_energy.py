import numpy as np
import polars as pl
import pytest

from chrono_features.features.absolute_energy import AbsoluteEnergy
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


@pytest.fixture
def sample_dataset() -> TSDataset:
    """Test dataset fixture with sample time series data."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1, 2, 3, 4, 5, 6],  # Values to calculate energy from
        },
    )
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


def test_absolute_energy_expanding(sample_dataset: TSDataset) -> None:
    """Test expanding window calculation of absolute energy."""
    transformer = AbsoluteEnergy(
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results (sum of squares):
    # For id=1: [1²=1, 1²+2²=5, 1²+2²+3²=14]
    # For id=2: [4²=16, 4²+5²=41, 4²+5²+6²=77]
    expected_values = np.array([1, 5, 14, 16, 41, 77])

    result_values = transformed_dataset.data["value_absolute_energy_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_absolute_energy_rolling(sample_dataset: TSDataset) -> None:
    """Test rolling window calculation of absolute energy."""
    transformer = AbsoluteEnergy(
        columns="value",
        window_types=WindowType.ROLLING(size=2),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1: [NaN, 1²+2²=5, 2²+3²=13]
    # For id=2: [NaN, 4²+5²=41, 5²+6²=61]
    expected_values = np.array([np.nan, 5, 13, np.nan, 41, 61])

    result_values = transformed_dataset.data["value_absolute_energy_rolling_2"].to_numpy()
    assert np.allclose(result_values, expected_values, equal_nan=True)


def test_absolute_energy_dynamic(sample_dataset: TSDataset) -> None:
    """Test dynamic window calculation of absolute energy."""
    # Add window length column
    sample_dataset.add_feature("window_len", [1, 2, 2, 1, 2, 2])

    transformer = AbsoluteEnergy(
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Expected results:
    # For id=1:
    #   - window_len=1: 1²=1
    #   - window_len=2: 1²+2²=5
    #   - window_len=2: 2²+3²=13
    # For id=2:
    #   - window_len=1: 4²=16
    #   - window_len=2: 4²+5²=41
    #   - window_len=2: 5²+6²=61
    expected_values = np.array([1, 5, 13, 16, 41, 61])

    result_values = transformed_dataset.data["value_absolute_energy_dynamic_based_on_window_len"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_absolute_energy_multiple_columns(sample_dataset: TSDataset) -> None:
    """Test calculation for multiple columns."""
    sample_dataset.add_feature("value2", [10, 20, 30, 40, 50, 60])

    transformer = AbsoluteEnergy(
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = transformer.transform(sample_dataset)

    # Check results for 'value' column (same as first test)
    expected_values_value = np.array([1, 5, 14, 16, 41, 77])
    result_values_value = transformed_dataset.data["value_absolute_energy_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values_value, expected_values_value)

    # Check results for 'value2' column:
    # For id=1: [10²=100, 10²+20²=500, 10²+20²+30²=1400]
    # For id=2: [40²=1600, 40²+50²=4100, 40²+50²+60²=7700]
    expected_values_value2 = np.array([100, 500, 1400, 1600, 4100, 7700])
    result_values_value2 = transformed_dataset.data["value2_absolute_energy_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values_value2, expected_values_value2)


def test_absolute_energy_custom_out_column_names(sample_dataset: TSDataset) -> None:
    """Test support for custom output column names."""
    transformer = AbsoluteEnergy(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_energy",
    )
    transformed_dataset = transformer.transform(sample_dataset)

    expected_values = np.array([1, 5, 14, 16, 41, 77])
    result_values = transformed_dataset.data["custom_energy"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)
