import numpy as np
import polars as pl
import pytest

from chrono_features.features.median import Median, MedianWithoutOptimization  # Import the new class
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowBase, WindowType


# Fixtures
@pytest.fixture
def sample_ts_dataset() -> TSDataset:
    """Fixture for creating a test TSDataset."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "value": [10, 20, 30, 40, 50, 60],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def sample_window_types() -> dict[str, WindowBase]:
    """Fixture for providing different window types."""
    return {
        "expanding": WindowType.EXPANDING(),
        "rolling_full": WindowType.ROLLING(size=2, only_full_window=True),
        "rolling_partial": WindowType.ROLLING(size=2, only_full_window=False),
        "dynamic": WindowType.DYNAMIC(len_column_name="dynamic_len"),
    }


class TestMedianNumbaLevel:
    """Tests for the Median class."""

    def test_initialization(self, sample_window_types: dict[str, WindowBase]) -> None:
        """Test Median initialization with default and custom out_column_name."""
        # Test with default out_column_name
        median_default = MedianWithoutOptimization(columns="value", window_types=sample_window_types["expanding"])
        assert median_default.out_column_names == ["value_median_expanding"]

        # Test with custom out_column_name
        median_custom = MedianWithoutOptimization(
            columns="value",
            window_types=sample_window_types["rolling_full"],
            out_column_names="custom_median",
        )
        assert median_custom.out_column_names == ["custom_median"]

    def test_numba_func(self) -> None:
        """Test that _numba_func correctly calculates the median of an array."""
        median_calculator = MedianWithoutOptimization(columns="value", window_types=WindowType.EXPANDING())
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        expected_median = np.median(test_array)
        result = median_calculator._numba_func(test_array)
        assert np.isclose(result, expected_median)

    def test_transform_expanding_window(self, sample_ts_dataset: TSDataset) -> None:
        """Test transform with expanding window."""
        median_calculator = Median(columns="value", window_types=WindowType.EXPANDING())
        transformed_dataset = median_calculator.transform(sample_ts_dataset)

        # Verify new column was added
        assert "value_median_expanding" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([10.0, 15.0, 20.0, 40.0, 45.0, 60.0], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_expanding"].to_numpy()
        assert np.allclose(result_medians, expected_medians)

    def test_transform_rolling_window_full(self, sample_ts_dataset: TSDataset) -> None:
        """Test transform with rolling window (only_full_window=True)."""
        median_calculator = Median(
            columns="value",
            window_types=WindowType.ROLLING(size=2, only_full_window=True),
        )
        transformed_dataset = median_calculator.transform(sample_ts_dataset)

        # Verify new column was added
        assert "value_median_rolling_2" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([np.nan, 15.0, 25.0, np.nan, 45.0, np.nan], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_rolling_2"].to_numpy()
        assert np.allclose(result_medians, expected_medians, equal_nan=True)

    def test_transform_rolling_window_partial(self, sample_ts_dataset: TSDataset) -> None:
        """Test transform with rolling window (only_full_window=False)."""
        median_calculator = Median(
            columns="value",
            window_types=WindowType.ROLLING(size=2, only_full_window=False),
        )
        transformed_dataset = median_calculator.transform(sample_ts_dataset)

        # Verify new column was added
        assert "value_median_rolling_2" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([10.0, 15.0, 25.0, 40.0, 45.0, 60.0], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_rolling_2"].to_numpy()
        assert np.allclose(result_medians, expected_medians)

    def test_transform_dynamic_window(self, sample_ts_dataset: TSDataset) -> None:
        """Test transform with dynamic window."""
        # Add dynamic_len column
        sample_ts_dataset.data = sample_ts_dataset.data.with_columns(pl.Series("dynamic_len", [1, 2, 1, 1, 2, 1]))
        median_calculator = Median(
            columns="value",
            window_types=WindowType.DYNAMIC(len_column_name="dynamic_len"),
        )
        transformed_dataset = median_calculator.transform(sample_ts_dataset)

        # Verify new column was added
        assert "value_median_dynamic_based_on_dynamic_len" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([10.0, 15.0, 30.0, 40.0, 45.0, 60.0], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_dynamic_based_on_dynamic_len"].to_numpy()
        assert np.allclose(result_medians, expected_medians)


# Other fixtures remain unchanged, only replacing Mean with Median
@pytest.fixture
def empty_ts_dataset() -> TSDataset:
    """Fixture for creating an empty TSDataset."""
    data = pl.DataFrame({"id": [], "timestamp": [], "value": []})
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def single_row_ts_dataset() -> TSDataset:
    """Fixture for creating a TSDataset with a single row."""
    data = pl.DataFrame({"id": [1], "timestamp": [1], "value": [10]})
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def ts_dataset_with_nan() -> TSDataset:
    """Fixture for creating a TSDataset with NaN values."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "value": [10.0, np.nan, 30.0, 40.0, 50.0, np.nan],
        },
    )
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def ts_dataset_multiple_columns() -> TSDataset:
    """Fixture for creating a TSDataset with multiple columns."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "value1": [10, 20, 30, 40, 50, 60],
            "value2": [100, 200, 300, 400, 500, 600],
        },
    )
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


class TestMedian:
    """Tests for the Median class."""

    def test_transform_dataset_with_nan(self, ts_dataset_with_nan: TSDataset) -> None:
        """Test transform with dataset containing NaN values."""
        median_calculator = Median(columns="value", window_types=WindowType.EXPANDING())
        transformed_dataset = median_calculator.transform(ts_dataset_with_nan)

        # Verify new column was added
        assert "value_median_expanding" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([10.0, np.nan, np.nan, 40.0, 45.0, np.nan], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_expanding"].to_numpy()
        assert np.allclose(result_medians, expected_medians, equal_nan=True)

    def test_transform_single_row_dataset(self, single_row_ts_dataset: TSDataset) -> None:
        """Test transform with dataset containing a single row."""
        median_calculator = Median(columns="value", window_types=WindowType.EXPANDING())
        transformed_dataset = median_calculator.transform(single_row_ts_dataset)

        # Verify new column was added
        assert "value_median_expanding" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([10.0], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_expanding"].to_numpy()
        assert np.allclose(result_medians, expected_medians)

    def test_transform_multiple_columns(self, ts_dataset_multiple_columns: TSDataset) -> None:
        """Test transform with dataset containing multiple columns."""
        median_calculator = MedianWithoutOptimization(columns=["value1", "value2"], window_types=WindowType.EXPANDING())
        transformed_dataset = median_calculator.transform(ts_dataset_multiple_columns)

        # Verify new columns were added
        for column in ["value1_median_expanding", "value2_median_expanding"]:
            assert column in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = {
            "value1_median_expanding": np.array([10.0, 15.0, 20.0, 40.0, 45.0, 60.0], dtype=np.float32),
            "value2_median_expanding": np.array([100.0, 150.0, 200.0, 400.0, 450.0, 600.0], dtype=np.float32),
        }

        for column, expected in expected_medians.items():
            result_medians = transformed_dataset.data[column].to_numpy()
            assert np.allclose(result_medians, expected)

    def test_transform_large_window_size(self, sample_ts_dataset: TSDataset) -> None:
        """Test transform with window size exceeding available data."""
        median_calculator = Median(
            columns="value",
            window_types=WindowType.ROLLING(size=10, only_full_window=True),
        )
        transformed_dataset = median_calculator.transform(sample_ts_dataset)

        # Verify new column was added
        assert "value_median_rolling_10" in transformed_dataset.data.columns

        # Verify calculated values
        expected_medians = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
        result_medians = transformed_dataset.data["value_median_rolling_10"].to_numpy()
        assert np.allclose(result_medians, expected_medians, equal_nan=True)
