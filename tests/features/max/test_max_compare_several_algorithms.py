import numpy as np
import pytest

from chrono_features import TSDataset, WindowType
from chrono_features.features.max import MaxWithOptimization, MaxWithoutOptimization
from tests.utils.compare_multiple_implementations import compare_multiple_implementations
from tests.utils.performance import create_dataset

# Set a fixed random seed for reproducible tests
np.random.seed(42)


@pytest.fixture
def medium_dataset() -> TSDataset:
    """Create a dataset with 300 time series, each with 20 points."""
    return create_dataset(n_ids=300, n_timestamps=20)


def test_expanding_window_optimization_comparison(medium_dataset: TSDataset) -> None:
    """Test that different implementations of expanding window max produce identical results."""
    implementations = [
        (MaxWithOptimization, "optimized"),
        (MaxWithoutOptimization, "non_optimized"),
    ]

    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.EXPANDING(),
    )


def test_rolling_window_optimization_comparison(medium_dataset: TSDataset) -> None:
    """Test that different implementations of rolling window max produce identical results."""
    implementations = [
        (MaxWithOptimization, "optimized"),
        (MaxWithoutOptimization, "non_optimized"),
    ]

    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.ROLLING(size=5, only_full_window=False),
    )

    implementations = [
        (MaxWithOptimization, "optimized"),
        (MaxWithoutOptimization, "non_optimized"),
    ]

    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.ROLLING(size=5, only_full_window=True),
    )


def test_dynamic_window_optimization_comparison(medium_dataset: TSDataset) -> None:
    """Test that different implementations of dynamic window max produce identical results."""
    # Add a dynamic window length column with values between 1 and 5
    window_lengths = np.random.randint(1, 6, size=len(medium_dataset.data))
    medium_dataset.add_feature("window_len", window_lengths)

    implementations = [
        (MaxWithOptimization, "optimized"),
        (MaxWithoutOptimization, "non_optimized"),
    ]

    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.DYNAMIC(len_column_name="window_len"),
    )
