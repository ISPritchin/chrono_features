import pytest
import polars as pl
import numpy as np
from chrono_features.features._base import WindowType
from chrono_features.features.sum import Sum
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def sample_dataset():
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "timestamp": [1, 2, 3, 1, 2, 3],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_sum_expanding(sample_dataset):
    sum_transformer = Sum(columns="value", window_type=WindowType.EXPANDING())
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 6, 4, 9, 15])
    result_values = transformed_dataset.data["value_sum_expanding"].to_numpy()
    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_rolling_only_full_window(sample_dataset):
    sum_transformer = Sum(columns="value", window_type=WindowType.ROLLING(size=2))
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([np.nan, 3, 5, np.nan, 9, 11])
    result_values = transformed_dataset.data["value_sum_rolling_2"].to_numpy()

    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_rolling_all_windows(sample_dataset):
    sum_transformer = Sum(columns="value", window_type=WindowType.ROLLING(size=2, only_full_window=False))
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 5, 4, 9, 11])
    result_values = transformed_dataset.data["value_sum_rolling_2"].to_numpy()

    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_dynamic(sample_dataset):
    sample_dataset.add_feature("window_len", [1, 2, 1, 1, 2, 1])
    sum_transformer = Sum(columns="value", window_type=WindowType.DYNAMIC(len_column_name="window_len"))
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values = np.array([1, 3, 3, 4, 9, 6])
    result_values = transformed_dataset.data["value_sum_dynamic_based_on_window_len"].to_numpy()

    np.testing.assert_array_equal(result_values, expected_values)


def test_sum_multiple_columns(sample_dataset):
    # Добавим еще одну колонку для тестирования множественных колонок
    sample_dataset.add_feature("value2", [10, 20, 30, 40, 50, 60])
    sum_transformer = Sum(columns=["value", "value2"], window_type=WindowType.EXPANDING())
    transformed_dataset = sum_transformer.transform(sample_dataset)

    expected_values_value = np.array([1, 3, 6, 4, 9, 15])
    expected_values_value2 = np.array([10, 30, 60, 40, 90, 150])

    result_values_value = transformed_dataset.data["value_sum_expanding"].to_numpy()
    result_values_value2 = transformed_dataset.data["value2_sum_expanding"].to_numpy()

    np.testing.assert_array_equal(result_values_value, expected_values_value)
    np.testing.assert_array_equal(result_values_value2, expected_values_value2)
