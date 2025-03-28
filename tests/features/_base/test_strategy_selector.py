import numpy as np
import polars as pl
import pytest

from chrono_features.features._base import StrategySelector
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowBase, WindowType


# Create mock implementations for testing
class AddOneImplementation:
    def __init__(self, *, columns, window_types, out_column_names=None):
        self.columns = columns if isinstance(columns, list) else [columns]
        self.window_types = window_types if isinstance(window_types, list) else [window_types]
        self.out_column_names = out_column_names

    def transform(self, dataset):
        result = dataset.clone()
        for column in self.columns:
            for window_type in self.window_types:
                # Add a simple feature that just adds 1 to the original column
                if self.out_column_names is None:
                    feature_name = window_type.add_suffix_to_feature(column)
                else:
                    feature_name = self.out_column_names[0]
                result.add_feature(feature_name, result.data[column] + 1)
        return result


class MultiplyByTwoImplementation:
    def __init__(self, *, columns, window_types, out_column_names=None):
        self.columns = columns if isinstance(columns, list) else [columns]
        self.window_types = window_types if isinstance(window_types, list) else [window_types]
        self.out_column_names = out_column_names

    def transform(self, dataset):
        result = dataset.clone()
        for column in self.columns:
            for window_type in self.window_types:
                # Add a simple feature that just multiplies the original column by 2
                if self.out_column_names is None:
                    feature_name = window_type.add_suffix_to_feature(column)
                else:
                    feature_name = self.out_column_names[0]
                result.add_feature(feature_name, result.data[column] * 2)
        return result


class ConcreteStrategySelector(StrategySelector):
    """Concrete implementation of StrategySelector for testing."""

    def __init__(
        self,
        *,
        columns,
        window_types,
        out_column_names=None,
        use_impl2_for_rolling=False,
    ):
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )
        self.use_impl2_for_rolling = use_impl2_for_rolling

    def _select_implementation_type(self, window_type: WindowBase) -> type:
        if isinstance(window_type, WindowType.ROLLING) and self.use_impl2_for_rolling:
            return MultiplyByTwoImplementation
        return AddOneImplementation


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "timestamp": [1, 2, 3, 1, 2],
            "value": [10, 20, 30, 40, 50],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_strategy_selector_init():
    """Test initialization of StrategySelector with different parameter types."""
    # Test with string column
    selector = ConcreteStrategySelector(
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    assert selector.columns == ["value"]
    assert len(selector.window_types) == 1
    assert isinstance(selector.window_types[0], WindowType.EXPANDING)
    assert selector.out_column_names is None

    # Test with list of columns
    selector = ConcreteStrategySelector(
        columns=["value", "other_value"],
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)],
        out_column_names=["value_out", "other_out"],
    )
    assert selector.columns == ["value", "other_value"]
    assert len(selector.window_types) == 2
    assert selector.out_column_names == ["value_out", "other_out"]

    # Test with string out_column_name
    selector = ConcreteStrategySelector(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_name",
    )
    assert selector.out_column_names == ["custom_name"]


def test_strategy_selector_transform(sample_dataset):
    """Test the transform method of StrategySelector."""
    # Test with a single implementation type
    selector = ConcreteStrategySelector(
        columns="value",
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)],
    )

    result = selector.transform(sample_dataset)

    # Check that the original columns are preserved
    assert "id" in result.data.columns
    assert "timestamp" in result.data.columns
    assert "value" in result.data.columns

    # Check that new columns were added with the expected values
    assert "value_expanding" in result.data.columns
    assert "value_rolling_2" in result.data.columns

    # Check the values of the new columns
    np.testing.assert_array_equal(
        result.data["value_expanding"].to_numpy(),
        sample_dataset.data["value"].to_numpy() + 1,
    )
    np.testing.assert_array_equal(
        result.data["value_rolling_2"].to_numpy(),
        sample_dataset.data["value"].to_numpy() + 1,
    )


def test_strategy_selector_multiple_implementations(sample_dataset):
    """Test StrategySelector with multiple implementation types."""
    selector = ConcreteStrategySelector(
        columns="value",
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)],
        use_impl2_for_rolling=True,  # Use Implementation2 for ROLLING windows
    )

    result = selector.transform(sample_dataset)

    # Check that new columns were added with the expected values
    assert "value_expanding" in result.data.columns
    assert "value_rolling_2" in result.data.columns

    # Check the values of the new columns
    np.testing.assert_array_equal(
        result.data["value_expanding"].to_numpy(),
        sample_dataset.data["value"].to_numpy() + 1,
    )
    np.testing.assert_array_equal(
        result.data["value_rolling_2"].to_numpy(),
        sample_dataset.data["value"].to_numpy() * 2,
    )


def test_strategy_selector_multiple_columns(sample_dataset):
    """Test StrategySelector with multiple columns."""
    # Add another column to the dataset
    sample_dataset.add_feature("another_value", sample_dataset.data["value"] * 10)

    selector = ConcreteStrategySelector(
        columns=["value", "another_value"],
        window_types=WindowType.EXPANDING(),
    )

    result = selector.transform(sample_dataset)

    # Check that new columns were added for both input columns
    assert "value_expanding" in result.data.columns
    assert "another_value_expanding" in result.data.columns

    # Check the values of the new columns
    np.testing.assert_array_equal(
        result.data["value_expanding"].to_numpy(),
        sample_dataset.data["value"].to_numpy() + 1,
    )
    np.testing.assert_array_equal(
        result.data["another_value_expanding"].to_numpy(),
        sample_dataset.data["another_value"].to_numpy() + 1,
    )


def test_strategy_selector_multiple_columns_and_mixed_windows(sample_dataset):
    """Test StrategySelector with multiple columns and mixed window types."""
    # Add another column to the dataset
    sample_dataset.add_feature("another_value", sample_dataset.data["value"] * 10)

    # Add a column to use as length reference for dynamic window
    sample_dataset.add_feature("window_length", pl.Series([1, 2, 3, 1, 2]))

    # Create a selector with multiple columns and mixed window types
    selector = ConcreteStrategySelector(
        columns=["value", "another_value"],
        window_types=[
            WindowType.EXPANDING(),
            WindowType.ROLLING(size=2),
            WindowType.DYNAMIC(len_column_name="window_length"),
        ],
        use_impl2_for_rolling=True,  # Use MultiplyByTwoImplementation for ROLLING windows
    )

    result = selector.transform(sample_dataset)

    # Check that all expected columns were added
    # For "value" column
    assert "value_expanding" in result.data.columns
    assert "value_rolling_2" in result.data.columns
    assert "value_dynamic_based_on_window_length" in result.data.columns

    # For "another_value" column
    assert "another_value_expanding" in result.data.columns
    assert "another_value_rolling_2" in result.data.columns
    assert "another_value_dynamic_based_on_window_length" in result.data.columns

    # Check the values of the new columns
    # For "value" column
    np.testing.assert_array_equal(
        result.data["value_expanding"].to_numpy(),
        sample_dataset.data["value"].to_numpy() + 1,
    )
    np.testing.assert_array_equal(
        result.data["value_rolling_2"].to_numpy(),
        sample_dataset.data["value"].to_numpy() * 2,
    )
    np.testing.assert_array_equal(
        result.data["value_dynamic_based_on_window_length"].to_numpy(),
        sample_dataset.data["value"].to_numpy() + 1,
    )

    # For "another_value" column
    np.testing.assert_array_equal(
        result.data["another_value_expanding"].to_numpy(),
        sample_dataset.data["another_value"].to_numpy() + 1,
    )
    np.testing.assert_array_equal(
        result.data["another_value_rolling_2"].to_numpy(),
        sample_dataset.data["another_value"].to_numpy() * 2,
    )
    np.testing.assert_array_equal(
        result.data["another_value_dynamic_based_on_window_length"].to_numpy(),
        sample_dataset.data["another_value"].to_numpy() + 1,
    )


def test_strategy_selector_custom_column_names(sample_dataset):
    """Test StrategySelector with custom output column names."""
    # This test verifies that out_column_names is correctly passed to implementations
    # In a real implementation, we would expect the output column to be named based on out_column_names
    # But our test implementations ignore out_column_names

    selector = ConcreteStrategySelector(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="custom_feature",
    )

    result = selector.transform(sample_dataset)

    # The implementation ignores out_column_names, so we still get the default name
    assert "custom_feature" in result.data.columns
