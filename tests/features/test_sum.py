import numpy as np
import polars as pl
import pytest

from chrono_features.features._base import WindowType
from chrono_features.features.sum import Sum
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def sample_dataset() -> TSDataset:
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1, 2, 3, 4, 5, 6],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_sum_expanding_with_optimization(sample_dataset: TSDataset) -> None:
    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 6, 4, 9, 15])
    result_values = transformed_dataset.data["value_sum_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_expanding_without_optimization(sample_dataset: TSDataset) -> None:
    sum_transformer = Sum(
        use_prefix_sum_optimization=False,
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 6, 4, 9, 15])
    result_values = transformed_dataset.data["value_sum_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_rolling_only_full_window_with_optimization(sample_dataset: TSDataset) -> None:
    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns="value",
        window_types=WindowType.ROLLING(size=2, only_full_window=True),
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([np.nan, 3, 5, np.nan, 9, 11])
    result_values = transformed_dataset.data["value_sum_rolling_2"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_rolling_all_windows_with_optimization(sample_dataset: TSDataset) -> None:
    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns="value",
        window_types=WindowType.ROLLING(size=2, only_full_window=False),
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 5, 4, 9, 11])
    result_values = transformed_dataset.data["value_sum_rolling_2"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_dynamic_with_optimization(sample_dataset: TSDataset) -> None:
    sample_dataset.add_feature("window_len", [1, 2, 1, 1, 2, 1])
    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns="value",
        window_types=WindowType.DYNAMIC(len_column_name="window_len"),
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 3, 4, 9, 6])
    result_values = transformed_dataset.data["value_sum_dynamic_based_on_window_len"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_multiple_columns_with_optimization(sample_dataset: TSDataset) -> None:
    sample_dataset.add_feature("value2", [10, 20, 30, 40, 50, 60])
    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns=["value", "value2"],
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values_value = np.array([1, 3, 6, 4, 9, 15])
    expected_values_value2 = np.array([10, 30, 60, 40, 90, 150])

    result_values_value = transformed_dataset.data["value_sum_expanding"].to_numpy()
    result_values_value2 = transformed_dataset.data["value2_sum_expanding"].to_numpy()

    np.testing.assert_array_equal(result_values_value, expected_values_value)
    np.testing.assert_array_equal(result_values_value2, expected_values_value2)


def test_sum_custom_out_column_names(sample_dataset: TSDataset) -> None:
    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_sum",
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 6, 4, 9, 15])
    result_values = transformed_dataset.data["custom_sum"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


@pytest.mark.parametrize("use_prefix_sum_optimization", [True, False])
def test_sum_multiple_window_types(sample_dataset: TSDataset, *, use_prefix_sum_optimization: bool) -> None:
    sum_transformer = Sum(
        use_prefix_sum_optimization=use_prefix_sum_optimization,
        columns="value",
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2, only_full_window=True)],
    )
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values_expanding = np.array([1, 3, 6, 4, 9, 15])
    expected_values_rolling = np.array([np.nan, 3, 5, np.nan, 9, 11])

    result_values_expanding = transformed_dataset.data["value_sum_expanding"].to_numpy()
    result_values_rolling = transformed_dataset.data["value_sum_rolling_2"].to_numpy()

    np.testing.assert_array_equal(result_values_expanding, expected_values_expanding)
    np.testing.assert_array_equal(result_values_rolling, expected_values_rolling)


def test_sum_with_nan_values():
    data_with_nan = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1.0, np.nan, 3, 4, 5, np.nan],
        },
    )
    dataset_with_nan = TSDataset(data_with_nan, id_column_name="id", ts_column_name="timestamp")

    sum_transformer = Sum(
        use_prefix_sum_optimization=True,
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = sum_transformer.transform(dataset_with_nan)

    expected_values = np.array([1.0, np.nan, np.nan, 4, 9, np.nan])
    result_values = transformed_dataset.data["value_sum_expanding"].to_numpy()

    assert np.allclose(result_values, expected_values, equal_nan=True)
