import numpy as np
import polars as pl
import pandas as pd
import pytest

from chrono_features import WindowType
from chrono_features.features import Max, Median, Sum, WeightedMovingAverage
from chrono_features.features.sum import SumWithPrefixSumOptimization
from chrono_features.features.mean import WeightedMean
from chrono_features.transformation_pipeline import TransformationPipeline
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def sample_dataset():
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0],
            "timestamp": range(9),
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_empty_pipeline(sample_dataset):
    """Test pipeline with no transformations returns unchanged dataset"""
    pipeline = TransformationPipeline([], verbose=False)
    result = pipeline.fit_transform(sample_dataset)

    assert result.data.equals(sample_dataset.data)
    assert len(pipeline.get_transformation_names()) == 0


def test_single_median_transformation(sample_dataset):
    """Test pipeline with single Median transformation"""
    median = Median(columns="value", window_types=WindowType.ROLLING(size=3))
    pipeline = TransformationPipeline([median], verbose=False)
    original_columns = set(sample_dataset.data.columns)

    result = pipeline.fit_transform(sample_dataset)

    assert len(pipeline.get_transformation_names()) == 1
    assert "Median" in pipeline.get_transformation_names()[0]
    assert original_columns.issubset(set(result.data.columns))
    assert "value_median_rolling_3" in result.data.columns
    assert "Median" in pipeline.describe()


def test_sum_with_prefix_optimization(sample_dataset):
    """Test Sum transformation with prefix sum optimization"""
    sum_transform = Sum(columns="value", window_types=WindowType.EXPANDING(), use_prefix_sum_optimization=True)
    pipeline = TransformationPipeline([sum_transform], verbose=False)

    result = pipeline.fit_transform(sample_dataset)

    assert "value_sum_expanding" in result.data.columns
    # Check sum calculation correctness
    expected_values = [1, 3, 6, 10, 15, 10, 30, 60, 100]
    assert np.allclose(result.data["value_sum_expanding"].fill_nan(-1).to_numpy(), expected_values, equal_nan=True)


def test_weighted_moving_average(sample_dataset):
    """Test WeightedMovingAverage transformation"""
    weights = np.array([0.1, 0.3, 0.6])  # Weights for WMA
    wma = WeightedMovingAverage(columns="value", window_size=3, weights=weights)
    pipeline = TransformationPipeline([wma], verbose=False)

    result = pipeline.fit_transform(sample_dataset)

    assert "value_weighted_moving_average_rolling_3" in result.data.columns
    # Check first valid value (third element)
    expected = (0.1 * 1 + 0.3 * 2 + 0.6 * 3) / (0.1 + 0.3 + 0.6)
    assert np.isclose(result.data["value_weighted_moving_average_rolling_3"][2], expected)


def test_multiple_different_transformations(sample_dataset):
    """Test pipeline with multiple different aggregation types"""
    transformations = [
        Median(columns="value", window_types=WindowType.ROLLING(size=2)),
        Sum(columns="value", window_types=WindowType.EXPANDING()),
        WeightedMovingAverage(columns="value", window_size=3, weights=[0.2, 0.3, 0.5]),
    ]
    pipeline = TransformationPipeline(transformations, verbose=False)

    result = pipeline.fit_transform(sample_dataset)

    expected_columns = {"value_median_rolling_2", "value_sum_expanding", "value_weighted_moving_average_rolling_3"}
    assert expected_columns.issubset(set(result.data.columns))

    # Check transformation order
    assert pipeline.get_transformation_names() == ["Median", "SumWithPrefixSumOptimization", "WeightedMean"]


@pytest.fixture
def empty_dataset():
    """Fixture for empty dataset"""
    data = pl.DataFrame(
        {
            "id": [],
            "value": [],
            "timestamp": [],
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


@pytest.fixture
def multi_column_dataset():
    """Fixture for dataset with multiple columns"""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "price": [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
            "volume": [100, 200, 300, 150, 250, 350],
            "timestamp": range(6),
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def test_empty_dataset(empty_dataset):
    """Test pipeline with empty dataset"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Median(columns="value", window_types=WindowType.ROLLING(size=2)),
            Sum(columns="value", window_types=WindowType.EXPANDING()),
        ],
        verbose=False,
    )

    # Act
    result = pipeline.fit_transform(empty_dataset)

    # Assert
    assert len(result.data) == 0
    assert "value_median_rolling_2" in result.data.columns
    assert "value_sum_expanding" in result.data.columns


def test_multi_column_dataset(multi_column_dataset):
    """Test pipeline with dataset containing multiple columns"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Median(columns="price", window_types=WindowType.ROLLING(size=2)),
            Sum(columns="volume", window_types=WindowType.EXPANDING()),
            WeightedMovingAverage(columns="price", window_size=2, weights=[0.4, 0.6]),
        ],
        verbose=False,
    )

    # Act
    result = pipeline.fit_transform(multi_column_dataset)

    # Assert
    # Check original columns preserved
    assert "price" in result.data.columns
    assert "volume" in result.data.columns

    # Check new columns added
    assert "price_median_rolling_2" in result.data.columns
    assert "volume_sum_expanding" in result.data.columns
    assert "price_weighted_moving_average_rolling_2" in result.data.columns

    # Verify no unexpected columns
    assert len(result.data.columns) == 7  # 4 original + 3 new

    # Check median values
    expected_medians = [np.nan, 15.0, 25.0, np.nan, 20.0, 30.0]
    assert np.array_equal(
        result.data["price_median_rolling_2"].fill_nan(-999).to_numpy(),
        np.where(np.isnan(expected_medians), -999, expected_medians),
        equal_nan=True,
    )


def test_selective_column_transformation(multi_column_dataset):
    """Test applying transformations to specific columns only"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Sum(columns="price", window_types=WindowType.EXPANDING()),
            Median(columns="volume", window_types=WindowType.ROLLING(size=2)),
        ],
        verbose=False,
    )

    # Act
    result = pipeline.fit_transform(multi_column_dataset)

    # Assert
    # Check only expected columns transformed
    assert "price_sum_expanding" in result.data.columns
    assert "volume_median_rolling_2" in result.data.columns

    # Check no unexpected transformations
    assert "volume_sum_expanding" not in result.data.columns
    assert "price_median_rolling_2" not in result.data.columns


def test_mixed_window_types(multi_column_dataset):
    """Test mixing different window types in transformations"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Sum(columns="volume", window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)]),
            Median(columns="price", window_types=WindowType.ROLLING(size=3)),
        ],
        verbose=False,
    )

    # Act
    result = pipeline.fit_transform(multi_column_dataset)

    # Assert
    # Check all expected columns created
    assert "volume_sum_expanding" in result.data.columns
    assert "volume_sum_rolling_2" in result.data.columns
    assert "price_median_rolling_3" in result.data.columns

    # Check expanding sum values
    expected_expanding = [100, 300, 600, 150, 400, 750]
    assert np.array_equal(result.data["volume_sum_expanding"].to_numpy(), expected_expanding)

    # Check rolling sum values
    expected_rolling = [np.nan, 300, 500, np.nan, 400, 600]
    assert np.array_equal(
        result.data["volume_sum_rolling_2"].fill_nan(-999).to_numpy(),
        np.where(np.isnan(expected_rolling), -999, expected_rolling),
        equal_nan=True,
    )


def test_pipeline_addition_operator():
    """Test combining pipelines using + operator"""
    # Arrange
    pipeline1 = TransformationPipeline([Median(columns="price", window_types=WindowType.ROLLING(size=2))])

    pipeline2 = TransformationPipeline([Sum(columns="volume", window_types=WindowType.EXPANDING())])

    single_transform = WeightedMovingAverage(columns="price", window_size=3, weights=[0.1, 0.3, 0.6])

    # Act
    combined_pipeline = pipeline1 + pipeline2
    combined_with_single = pipeline1 + single_transform

    # Assert
    assert len(combined_pipeline.transformations) == 2
    assert isinstance(combined_pipeline.transformations[0], Median)
    assert isinstance(combined_pipeline.transformations[1], SumWithPrefixSumOptimization)

    assert len(combined_with_single.transformations) == 2
    assert isinstance(combined_with_single.transformations[0], Median)
    assert isinstance(combined_with_single.transformations[1], WeightedMean)


def test_multiple_pipeline_chaining(multi_column_dataset):
    """Test chaining multiple pipeline executions"""
    # Arrange
    pipeline1 = TransformationPipeline([Sum(columns="volume", window_types=WindowType.EXPANDING())])

    pipeline2 = TransformationPipeline([Median(columns="price", window_types=WindowType.ROLLING(size=2))])

    # Act
    result1 = pipeline1.fit_transform(multi_column_dataset)
    result2 = pipeline2.fit_transform(result1)

    # Assert
    assert "volume_sum_expanding" in result2.data.columns
    assert "price_median_rolling_2" in result2.data.columns
    assert len(result2.data.columns) == 6  # 4 original + 2 new


def test_self_addition():
    """Test pipeline + pipeline returns new independent instance"""
    # Arrange
    pipeline = TransformationPipeline([Median(columns="price", window_types=WindowType.ROLLING(size=2))])

    # Act
    combined = pipeline + pipeline

    # Assert
    assert len(combined.transformations) == 2
    assert combined is not pipeline
    assert combined.transformations[0] is not pipeline.transformations[0]  # deepcopy check


def test_invalid_transformation_type():
    """Test passing invalid transformation type"""
    with pytest.raises(TypeError) as excinfo:
        TransformationPipeline(["not_a_transformation"])  # incorrect type

    assert "must be a FeatureGenerator" in str(excinfo.value)


def test_invalid_dataset_type():
    """Test passing invalid dataset type"""
    # Arrange
    pipeline = TransformationPipeline([Sum(columns="price", window_types=WindowType.EXPANDING())])

    # Act/Assert
    with pytest.raises(TypeError) as excinfo:
        pipeline.fit_transform("not_a_dataset")

    assert "Unsupported input type" in str(excinfo.value)


def test_conflicting_column_names(multi_column_dataset):
    """Test column name conflicts"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Sum(columns="price", window_types=WindowType.EXPANDING(), out_column_names="custom_name"),
            Median(columns="volume", window_types=WindowType.EXPANDING(), out_column_names="custom_name"),
        ],
    )

    # Act/Assert
    with pytest.raises(ValueError) as excinfo:
        pipeline.fit_transform(multi_column_dataset)

    assert "already exists" in str(excinfo.value).lower()


def test_nonexistent_column(multi_column_dataset):
    """Test transformation with non-existent column"""
    # Arrange
    pipeline = TransformationPipeline([Sum(columns="nonexistent", window_types=WindowType.EXPANDING())])

    # Act/Assert
    with pytest.raises(ValueError) as excinfo:
        pipeline.fit_transform(multi_column_dataset)

    assert "not found in the dataset" in str(excinfo.value)


def test_invalid_window_params():
    """Test invalid window parameters"""
    with pytest.raises(ValueError) as excinfo:
        Sum(columns="price", window_types=WindowType.ROLLING(size=-1))  # negative size

    assert "window size" in str(excinfo.value).lower()


def test_get_transformation_names():
    """Test get_transformation_names() returns correct names in order"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Median(columns="price", window_types=WindowType.ROLLING(size=2)),
            Sum(columns="volume", window_types=WindowType.EXPANDING()),
            WeightedMovingAverage(columns="price", window_size=3, weights=[0.1, 0.3, 0.6]),
        ],
    )

    # Act
    names = pipeline.get_transformation_names()

    # Assert
    assert names == ["Median", "SumWithPrefixSumOptimization", "WeightedMean"]
    assert len(names) == 3


def test_clone_creates_independent_copy():
    """Test clone() creates truly independent copy"""
    # Arrange
    original = TransformationPipeline([Sum(columns="price", window_types=WindowType.EXPANDING())])

    # Act
    cloned = original.clone()
    cloned.transformations.append(Median(columns="volume", window_types=WindowType.ROLLING(size=2)))

    # Assert
    assert len(original.transformations) == 1
    assert len(cloned.transformations) == 2
    assert original is not cloned
    assert original.transformations[0] is not cloned.transformations[0]


def test_verbose_output(capsys, multi_column_dataset):
    """Test verbose mode prints transformation progress"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Sum(columns="price", window_types=WindowType.EXPANDING()),
            Median(columns="volume", window_types=WindowType.ROLLING(size=2)),
        ],
        verbose=True,
    )

    # Act
    pipeline.fit_transform(multi_column_dataset)
    captured = capsys.readouterr()

    # Assert
    output = captured.out
    assert "Applying transformation 1/2: SumWithPrefixSumOptimization..." in output
    assert "Applying transformation 2/2: Median..." in output
    assert "Added columns: ['price_sum_expanding']" in output
    assert "Dataset shape:" in output


def test_silent_mode(capsys, multi_column_dataset):
    """Test non-verbose mode produces no output"""
    # Arrange
    pipeline = TransformationPipeline([Sum(columns="price", window_types=WindowType.EXPANDING())], verbose=False)

    # Act
    pipeline.fit_transform(multi_column_dataset)
    captured = capsys.readouterr()

    # Assert
    assert captured.out == ""


def test_complex_integration(multi_column_dataset):
    """Test complex pipeline with multiple transformation types"""
    # Arrange
    pipeline = TransformationPipeline(
        [
            Sum(columns=["price", "volume"], window_types=[WindowType.EXPANDING(), WindowType.ROLLING(size=2)]),
            Median(columns="price", window_types=WindowType.ROLLING(size=3)),
            WeightedMovingAverage(columns="volume", window_size=2, weights=[0.4, 0.6]),
        ],
        verbose=False,
    )

    # Act
    result = pipeline.fit_transform(multi_column_dataset)

    # Assert
    # Check all expected columns
    expected_columns = {
        "price_sum_expanding",
        "price_sum_rolling_2",
        "volume_sum_expanding",
        "volume_sum_rolling_2",
        "price_median_rolling_3",
        "volume_weighted_moving_average_rolling_2",
    }
    assert expected_columns.issubset(set(result.data.columns))


def test_large_dataset_performance():
    """Test pipeline performance with large dataset"""
    # Arrange
    np.random.seed(42)
    large_data = pl.DataFrame(
        {
            "id": np.repeat(np.arange(100), 1000),
            "value": np.random.normal(0, 1, 100000),
            "timestamp": np.tile(np.arange(1000), 100),
        },
    )
    large_dataset = TSDataset(large_data, id_column_name="id", ts_column_name="timestamp")

    pipeline = TransformationPipeline(
        [
            Sum(columns="value", window_types=WindowType.EXPANDING()),
            Median(columns="value", window_types=WindowType.ROLLING(size=50)),
        ],
        verbose=False,
    )

    # Act/Assert (primarily check that it executes without errors)
    result = pipeline.fit_transform(large_dataset)

    # Check calculation correctness for first 5 elements of first series
    first_series = result.data.filter(pl.col("id") == 0)
    assert np.allclose(
        first_series["value_sum_expanding"][:5].to_numpy(),
        np.cumsum(large_data["value"][:5].to_numpy()),
    )


# Add this new test class after all the existing tests
class TestUnifiedFitTransform:
    """Tests for the unified fit_transform method that handles different input types."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "timestamp": [1, 2, 3, 1, 2, 3],
                "value": [10, 20, 30, 40, 50, 60],
            },
        )

    @pytest.fixture
    def sample_transformers(self):
        """Create sample transformers for testing."""
        return [
            Max(columns="value", window_types=WindowType.EXPANDING()),
            Sum(columns="value", window_types=WindowType.ROLLING(size=2)),
        ]

    @pytest.fixture
    def sample_pipeline(self, sample_transformers):
        """Create a sample pipeline for testing."""
        return TransformationPipeline(sample_transformers)

    def test_fit_transform_with_tsdataset(self, sample_data, sample_pipeline):
        """Test fit_transform with a TSDataset input."""
        # Create a TSDataset
        dataset = TSDataset(sample_data, id_column_name="id", ts_column_name="timestamp")

        # Apply the pipeline
        result = sample_pipeline.fit_transform(dataset)

        # Check that the result is a TSDataset
        assert isinstance(result, TSDataset)

        # Check that the expected columns were added
        assert "value_max_expanding" in result.data.columns
        assert "value_sum_rolling_2" in result.data.columns

        # Check some values
        max_values = result.data["value_max_expanding"].to_numpy()
        expected_max = np.array([10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(max_values, expected_max)

    def test_fit_transform_with_polars_df(self, sample_data, sample_pipeline):
        """Test fit_transform with a polars DataFrame input."""
        # Apply the pipeline directly to a polars DataFrame
        result = sample_pipeline.fit_transform(sample_data, id_column_name="id", ts_column_name="timestamp")

        # Check that the result is a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that the expected columns were added
        assert "value_max_expanding" in result.columns
        assert "value_sum_rolling_2" in result.columns

        # Check some values
        max_values = result["value_max_expanding"].to_numpy()
        expected_max = np.array([10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(max_values, expected_max)

    def test_fit_transform_with_pandas_df(self, sample_data, sample_pipeline):
        """Test fit_transform with a pandas DataFrame input."""
        # Convert to pandas DataFrame
        pandas_df = sample_data.to_pandas()

        # Apply the pipeline directly to a pandas DataFrame
        result = sample_pipeline.fit_transform(pandas_df, id_column_name="id", ts_column_name="timestamp")

        # Check that the result is a pandas DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that the expected columns were added
        assert "value_max_expanding" in result.columns
        assert "value_sum_rolling_2" in result.columns

        # Check some values
        max_values = result["value_max_expanding"].to_numpy()
        expected_max = np.array([10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(max_values, expected_max)

    def test_fit_transform_missing_column_names(self, sample_data, sample_pipeline):
        """Test fit_transform with missing column names."""
        # Try to apply the pipeline without column names
        with pytest.raises(TypeError, match="id_column_name and ts_column_name must be provided"):
            sample_pipeline.fit_transform(sample_data)

    def test_fit_transform_unsupported_type(self, sample_pipeline):
        """Test fit_transform with an unsupported input type."""
        # Try to apply the pipeline to a list
        with pytest.raises(TypeError, match="Unsupported input type"):
            sample_pipeline.fit_transform([1, 2, 3])
