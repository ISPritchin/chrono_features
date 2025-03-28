import pytest
import numpy as np
import polars as pl

from chrono_features.features._base import FeatureGenerator
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


# Create a concrete implementation of FeatureGenerator for testing
class ConcreteFeatureGenerator(FeatureGenerator):
    def transform_for_window_type(self, dataset, column, window_type):  # noqa: ARG002
        # Simple implementation that returns the original column values
        return dataset.data[column].to_numpy()


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


def test_feature_generator_init_with_valid_params():
    """Test initialization of FeatureGenerator with valid parameters."""
    # Test with string column and single window type
    generator = ConcreteFeatureGenerator(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="value_expanded",
    )
    assert generator.columns == ["value"]
    assert len(generator.window_types) == 1
    assert isinstance(generator.window_types[0], WindowType.EXPANDING)
    assert generator.out_column_names == ["value_expanded"]

    # Test with list of columns and multiple window types
    generator = ConcreteFeatureGenerator(
        columns=["value", "other_value"],
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)],
        out_column_names=["value_expanded", "value_rolling", "other_expanded", "other_rolling"],
    )
    assert generator.columns == ["value", "other_value"]
    assert len(generator.window_types) == 2
    assert generator.out_column_names == ["value_expanded", "value_rolling", "other_expanded", "other_rolling"]


def test_feature_generator_init_with_auto_column_names():
    """Test that out_column_names is generated automatically when not provided."""
    # This would be tested in subclasses that implement auto-naming


def test_feature_generator_init_with_invalid_columns():
    """Test initialization with invalid columns parameter."""
    # Test with empty list
    with pytest.raises(ValueError):
        ConcreteFeatureGenerator(columns=[], window_types=WindowType.EXPANDING(), out_column_names="value_expanded")

    # Test with non-string, non-list
    with pytest.raises(ValueError):
        ConcreteFeatureGenerator(
            columns=123,
            window_types=WindowType.EXPANDING(),
            out_column_names="value_expanded",  # Not a string or list
        )


def test_feature_generator_init_with_invalid_window_types():
    """Test initialization with invalid window_types parameter."""
    # Test with empty list
    with pytest.raises(ValueError):
        ConcreteFeatureGenerator(columns="value", window_types=[], out_column_names="value_expanded")

    # Test with non-WindowBase, non-list
    with pytest.raises(ValueError):
        ConcreteFeatureGenerator(
            columns="value",
            window_types="not_a_window_type",  # Not a WindowBase or list
            out_column_names="value_expanded",
        )


def test_feature_generator_init_with_mismatched_out_column_names():
    """Test initialization with mismatched number of out_column_names."""
    # Test with too few out_column_names
    with pytest.raises(ValueError):
        ConcreteFeatureGenerator(
            columns=["value", "other_value"],
            window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)],
            out_column_names=["value_expanded"],  # Should have 4 names
        )

    # Test with too many out_column_names
    with pytest.raises(ValueError):
        ConcreteFeatureGenerator(
            columns="value",
            window_types=WindowType.EXPANDING(),
            out_column_names=["value_expanded", "extra_name"],  # Should have 1 name
        )


def test_feature_generator_transform_with_missing_column(sample_dataset):
    """Test transform with a column that doesn't exist in the dataset."""
    generator = ConcreteFeatureGenerator(
        columns="non_existent_column",
        window_types=WindowType.EXPANDING(),
        out_column_names="feature",
    )

    with pytest.raises(ValueError, match="Column 'non_existent_column' not found in the dataset"):
        generator.transform(sample_dataset)


def test_feature_generator_transform_with_valid_inputs(sample_dataset):
    """Test transform with valid inputs."""
    generator = ConcreteFeatureGenerator(
        columns="value",
        window_types=WindowType.EXPANDING(),
        out_column_names="value_expanded",
    )

    result = generator.transform(sample_dataset)

    # Check that the original columns are preserved
    assert "id" in result.data.columns
    assert "timestamp" in result.data.columns
    assert "value" in result.data.columns

    # Check that the new column was added
    assert "value_expanded" in result.data.columns

    # In our concrete implementation, the values should be the same as the original
    np.testing.assert_array_equal(result.data["value_expanded"].to_numpy(), sample_dataset.data["value"].to_numpy())


def test_feature_generator_transform_with_multiple_columns_and_windows(sample_dataset):
    """Test transform with multiple columns and window types."""
    # Add another column to the dataset
    sample_dataset.add_feature("another_value", sample_dataset.data["value"] * 2)

    generator = ConcreteFeatureGenerator(
        columns=["value", "another_value"],
        window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)],
        out_column_names=["value_expanded", "value_rolling", "another_expanded", "another_rolling"],
    )

    result = generator.transform(sample_dataset)

    # Check that all new columns were added
    assert "value_expanded" in result.data.columns
    assert "value_rolling" in result.data.columns
    assert "another_expanded" in result.data.columns
    assert "another_rolling" in result.data.columns
