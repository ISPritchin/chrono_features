# ruff: noqa: ANN401, T201

from typing import Any

import numpy as np
import pytest

from chrono_features import TSDataset, WindowType
from chrono_features.features.absolute_sum_of_changes import (
    AbsoluteSumOfChangesWithOptimization,
    AbsoluteSumOfChangesWithoutOptimization,
)
from chrono_features.features.max import MaxWithOptimization, MaxWithoutOptimization
from chrono_features.features.min import MinWithOptimization, MinWithoutOptimization
from chrono_features.features.sum import SumWithPrefixSumOptimization, SumWithoutOptimization
from tests.utils.compare_multiple_implementations import compare_multiple_implementations
from tests.utils.performance import create_dataset

# Set a fixed random seed for reproducible tests
np.random.seed(42)


@pytest.fixture
def medium_dataset() -> TSDataset:
    """Create a dataset with 300 time series, each with 20 points."""
    return create_dataset(n_ids=300, n_timestamps=20)


def run_optimization_comparison_tests(
    medium_dataset: TSDataset,
    optimized_implementation: Any,
    non_optimized_implementation: Any,
    feature_name: str,
) -> None:
    """Run comparison tests for different window types.

    Args:
        medium_dataset: Dataset to use for testing
        optimized_implementation: Optimized implementation class
        non_optimized_implementation: Non-optimized implementation class
        feature_name: Name of the feature being tested
    """
    # Test expanding window
    implementations = [
        (optimized_implementation, "optimized"),
        (non_optimized_implementation, "non_optimized"),
    ]

    print(f"Testing expanding window for {feature_name}...")
    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.EXPANDING(),
    )

    # Test rolling window with only_full_window=False
    print(f"Testing rolling window (partial) for {feature_name}...")
    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.ROLLING(size=5, only_full_window=False),
    )

    # Test rolling window with only_full_window=True
    print(f"Testing rolling window (full) for {feature_name}...")
    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.ROLLING(size=5, only_full_window=True),
    )

    # Test dynamic window
    # Add a dynamic window length column with values between 1 and 5
    window_lengths = np.random.randint(1, 6, size=len(medium_dataset.data))
    medium_dataset.add_feature("window_len", window_lengths)

    print(f"Testing dynamic window for {feature_name}...")
    compare_multiple_implementations(
        medium_dataset,
        implementations,
        WindowType.DYNAMIC(len_column_name="window_len"),
    )


@pytest.mark.parametrize(
    ("optimized_implementation", "non_optimized_implementation", "feature_name"),
    [
        (AbsoluteSumOfChangesWithOptimization, AbsoluteSumOfChangesWithoutOptimization, "AbsoluteSumOfChanges"),
        (MaxWithOptimization, MaxWithoutOptimization, "Max"),
        (MinWithOptimization, MinWithoutOptimization, "Min"),
        (SumWithPrefixSumOptimization, SumWithoutOptimization, "Sum"),
    ],
)
def test_optimization_comparison(
    medium_dataset: TSDataset,
    optimized_implementation: Any,
    non_optimized_implementation: Any,
    feature_name: str,
) -> None:
    """Test that different implementations of the same feature produce identical results.

    This test function compares optimized and non-optimized implementations of time series features
    to ensure they produce the same results across different window types (expanding, rolling, dynamic).

    Args:
        medium_dataset: A TSDataset instance containing multiple time series for testing
        optimized_implementation: The optimized implementation class of the feature
        non_optimized_implementation: The non-optimized implementation class of the feature
        feature_name: String identifier for the feature being tested

    Returns:
        None

    Raises:
        AssertionError: If the results from optimized and non-optimized implementations differ
    """
    run_optimization_comparison_tests(
        medium_dataset=medium_dataset,
        optimized_implementation=optimized_implementation,
        non_optimized_implementation=non_optimized_implementation,
        feature_name=feature_name,
    )
