import numpy as np
import polars as pl
import pytest

from chrono_features.features.mean import Mean
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowBase, WindowType


# Fixtures
@pytest.fixture
def sample_ts_dataset() -> TSDataset:
    """Fixture to create a sample TSDataset for testing."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "value": [10, 20, 30, 40, 50, 60],
        }
    )
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def sample_window_types() -> dict[str, WindowBase]:
    """Fixture to provide various window types for testing."""
    return {
        "expanding": WindowType.EXPANDING(),
        "rolling_full": WindowType.ROLLING(size=2, only_full_window=True),
        "rolling_partial": WindowType.ROLLING(size=2, only_full_window=False),
        "dynamic": WindowType.DYNAMIC(len_column_name="dynamic_len"),
    }


class TestMeanNumbaLevel:
    """Tests for the Mean class."""

    def test_initialization(self, sample_window_types: dict[str, WindowBase]) -> None:
        """Test that Mean initializes correctly with default and custom out_column_name."""
        # Test with default out_column_name
        mean_default = Mean(columns="value", window_types=sample_window_types["expanding"])
        assert mean_default.out_column_names == ["value_mean_expanding"]

        # Test with custom out_column_name
        mean_custom = Mean(
            columns="value",
            window_types=sample_window_types["rolling_full"],
            out_column_names="custom_mean",
        )
        assert mean_custom.out_column_names == ["custom_mean"]

    def test_numba_func(self) -> None:
        """Test that _numba_func correctly calculates the mean of an array."""
        mean_calculator = Mean(columns="value", window_types=WindowType.EXPANDING())
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        expected_mean = np.mean(test_array)
        result = mean_calculator._numba_func(test_array)
        assert np.isclose(result, expected_mean)

    def test_transform_expanding_window(self, sample_ts_dataset: TSDataset) -> None:
        """Test the transform method with an expanding window."""
        mean_calculator = Mean(columns="value", window_types=WindowType.EXPANDING())
        transformed_dataset = mean_calculator.transform(sample_ts_dataset)

        # Check that the new column is added
        assert "value_mean_expanding" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([10.0, 15.0, 20.0, 40.0, 45.0, 60.0], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_expanding"].to_numpy()
        assert np.allclose(result_means, expected_means)

    def test_transform_rolling_window_full(self, sample_ts_dataset: TSDataset) -> None:
        """Test the transform method with a rolling window (only_full_window=True)."""
        mean_calculator = Mean(
            columns="value",
            window_types=WindowType.ROLLING(size=2, only_full_window=True),
        )
        transformed_dataset = mean_calculator.transform(sample_ts_dataset)

        # Check that the new column is added
        assert "value_mean_rolling_2" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([np.nan, 15.0, 25.0, np.nan, 45.0, np.nan], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_rolling_2"].to_numpy()
        assert np.allclose(result_means, expected_means, equal_nan=True)

    def test_transform_rolling_window_partial(self, sample_ts_dataset: TSDataset) -> None:
        """Test the transform method with a rolling window (only_full_window=False)."""
        mean_calculator = Mean(
            columns="value",
            window_types=WindowType.ROLLING(size=2, only_full_window=False),
        )
        transformed_dataset = mean_calculator.transform(sample_ts_dataset)

        # Check that the new column is added
        assert "value_mean_rolling_2" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([10.0, 15.0, 25.0, 40.0, 45.0, 60.0], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_rolling_2"].to_numpy()
        assert np.allclose(result_means, expected_means)

    def test_transform_dynamic_window(self, sample_ts_dataset: TSDataset) -> None:
        """Test the transform method with a dynamic window."""
        # Add a dynamic length column to the dataset
        sample_ts_dataset.data = sample_ts_dataset.data.with_columns(pl.Series("dynamic_len", [1, 2, 1, 1, 2, 1]))
        mean_calculator = Mean(
            columns="value",
            window_types=WindowType.DYNAMIC(len_column_name="dynamic_len"),
        )
        transformed_dataset = mean_calculator.transform(sample_ts_dataset)

        # Check that the new column is added
        assert "value_mean_dynamic_based_on_dynamic_len" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([10.0, 15.0, 30.0, 40.0, 45.0, 60.0], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_dynamic_based_on_dynamic_len"].to_numpy()
        assert np.allclose(result_means, expected_means)


@pytest.fixture
def empty_ts_dataset() -> TSDataset:
    """Fixture to create an empty TSDataset for testing."""
    data = pl.DataFrame({"id": [], "timestamp": [], "value": []})
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def single_row_ts_dataset() -> TSDataset:
    """Fixture to create a TSDataset with a single row for testing."""
    data = pl.DataFrame({"id": [1], "timestamp": [1], "value": [10]})
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def ts_dataset_with_nan() -> TSDataset:
    """Fixture to create a TSDataset with NaN values for testing."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "value": [10.0, np.nan, 30.0, 40.0, 50.0, np.nan],
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def ts_dataset_multiple_columns() -> TSDataset:
    """Fixture to create a TSDataset with multiple columns for testing."""
    data = pl.DataFrame(
        data={
            "id": [1, 1, 1, 2, 2, 3],
            "timestamp": [1, 2, 3, 1, 2, 1],
            "value1": [10, 20, 30, 40, 50, 60],
            "value2": [100, 200, 300, 400, 500, 600],
        }
    )
    return TSDataset(data=data, id_column_name="id", ts_column_name="timestamp")


# Tests for the Mean class
class TestMean:
    """Tests for the Mean class."""

    def test_transform_dataset_with_nan(self, ts_dataset_with_nan: TSDataset) -> None:
        """Test the transform method with a dataset containing NaN values."""
        mean_calculator = Mean(columns="value", window_types=WindowType.EXPANDING())
        transformed_dataset = mean_calculator.transform(ts_dataset_with_nan)

        # Check that the new column is added
        assert "value_mean_expanding" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([10.0, np.nan, np.nan, 40.0, 45.0, np.nan], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_expanding"].to_numpy()

        assert np.allclose(result_means, expected_means, equal_nan=True)

    def test_transform_single_row_dataset(self, single_row_ts_dataset: TSDataset) -> None:
        """Test the transform method with a dataset containing a single row."""
        mean_calculator = Mean(columns="value", window_types=WindowType.EXPANDING())
        transformed_dataset = mean_calculator.transform(single_row_ts_dataset)

        # Check that the new column is added
        assert "value_mean_expanding" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([10.0], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_expanding"].to_numpy()
        assert np.allclose(result_means, expected_means)

    def test_transform_multiple_columns(self, ts_dataset_multiple_columns: TSDataset) -> None:
        """Test the transform method with a dataset containing multiple columns."""
        mean_calculator = Mean(columns=["value1", "value2"], window_types=WindowType.EXPANDING())
        transformed_dataset = mean_calculator.transform(ts_dataset_multiple_columns)

        # Check that the new columns are added
        for column in ["value1_mean_expanding", "value2_mean_expanding"]:
            assert column in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = {
            "value1_mean_expanding": np.array([10.0, 15.0, 20.0, 40.0, 45.0, 60.0], dtype=np.float32),
            "value2_mean_expanding": np.array([100.0, 150.0, 200.0, 400.0, 450.0, 600.0], dtype=np.float32),
        }

        for column, expected in expected_means.items():
            result_means = transformed_dataset.data[column].to_numpy()
            assert np.allclose(result_means, expected)

    def test_transform_large_window_size(self, sample_ts_dataset: TSDataset) -> None:
        """Test the transform method with a window size larger than the available data."""
        mean_calculator = Mean(
            columns="value",
            window_types=WindowType.ROLLING(size=10, only_full_window=True),
        )
        transformed_dataset = mean_calculator.transform(dataset=sample_ts_dataset)

        # Check that the new column is added
        assert "value_mean_rolling_10" in transformed_dataset.data.columns

        # Check the calculated values
        expected_means = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
        result_means = transformed_dataset.data["value_mean_rolling_10"].to_numpy()
        assert np.allclose(result_means, expected_means, equal_nan=True)
