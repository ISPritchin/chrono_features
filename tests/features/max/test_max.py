import numpy as np
import polars as pl
import pytest

from chrono_features import WindowType
from chrono_features.features.max import Max
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def sample_dataset() -> TSDataset:
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_max_expanding_with_optimization(sample_dataset: TSDataset) -> None:
    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values = np.array([1, 2, 3, 4, 5, 6])
    result_values = transformed_dataset.data["value_max_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_max_expanding_without_optimization(sample_dataset: TSDataset) -> None:
    max_transformer = Max(
        use_optimization=False,
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values = np.array([1, 2, 3, 4, 5, 6])
    result_values = transformed_dataset.data["value_max_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_max_rolling_only_full_window_with_optimization(sample_dataset: TSDataset) -> None:
    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=WindowType.ROLLING(size=2, only_full_window=True),
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values = np.array([np.nan, 2, 3, np.nan, 5, 6])
    result_values = transformed_dataset.data["value_max_rolling_2"].to_numpy()
    assert np.allclose(result_values, expected_values, equal_nan=True)


def test_max_multiple_window_types(sample_dataset: TSDataset) -> None:
    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2, only_full_window=True)],
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values_expanding = np.array([1, 2, 3, 4, 5, 6])
    expected_values_rolling = np.array([np.nan, 2, 3, np.nan, 5, 6])

    result_values_expanding = transformed_dataset.data["value_max_expanding"].to_numpy()
    result_values_rolling = transformed_dataset.data["value_max_rolling_2"].to_numpy()

    np.testing.assert_array_equal(result_values_expanding, expected_values_expanding)
    assert np.allclose(result_values_rolling, expected_values_rolling, equal_nan=True)


def test_max_rolling_all_windows_with_optimization(sample_dataset: TSDataset) -> None:
    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=WindowType.ROLLING(size=2, only_full_window=False),
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values = np.array([1, 2, 3, 4, 5, 6])
    result_values = transformed_dataset.data["value_max_rolling_2"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_max_dynamic_with_optimization(sample_dataset: TSDataset) -> None:
    sample_dataset.add_feature("window_len", [1, 2, 1, 1, 2, 1])
    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values = np.array([1, 2, 3, 4, 5, 6])
    result_values = transformed_dataset.data["value_max_dynamic_based_on_window_len"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_max_multiple_columns_with_optimization(sample_dataset: TSDataset) -> None:
    sample_dataset.add_feature("value2", [10, 20, 30, 40, 50, 60])
    max_transformer = Max(
        use_optimization=True,
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values_value = np.array([1, 2, 3, 4, 5, 6])
    expected_values_value2 = np.array([10, 20, 30, 40, 50, 60])

    result_values_value = transformed_dataset.data["value_max_expanding"].to_numpy()
    result_values_value2 = transformed_dataset.data["value2_max_expanding"].to_numpy()

    np.testing.assert_array_equal(result_values_value, expected_values_value)
    np.testing.assert_array_equal(result_values_value2, expected_values_value2)


def test_max_custom_out_column_names(sample_dataset: TSDataset) -> None:
    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_max",
    )
    transformed_dataset = max_transformer.transform(sample_dataset)

    expected_values = np.array([1, 2, 3, 4, 5, 6])
    result_values = transformed_dataset.data["custom_max"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_max_with_nan_values():
    data_with_nan = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1.0, np.nan, 3, 4, 5, np.nan],
        }
    )
    dataset_with_nan = TSDataset(data_with_nan, id_column_name="id", ts_column_name="timestamp")

    max_transformer = Max(
        use_optimization=True,
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = max_transformer.transform(dataset_with_nan)

    expected_values = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan])
    result_values = transformed_dataset.data["value_max_expanding"].to_numpy()

    assert np.allclose(result_values, expected_values, equal_nan=True)
