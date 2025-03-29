import numpy as np
import pytest
from pathlib import Path

from chrono_features import WindowType
from chrono_features.features.absolute_sum_of_changes import (
    AbsoluteSumOfChangesWithOptimization,
    AbsoluteSumOfChangesWithoutOptimization,
    AbsoluteSumOfChanges,
)
from chrono_features.features.max import MaxWithOptimization, MaxWithoutOptimization, Max
from chrono_features.features.min import MinWithOptimization, MinWithoutOptimization, Min
from chrono_features.features.sum import SumWithPrefixSumOptimization, SumWithoutOptimization, Sum
from tests.utils.performance import create_dataset_with_dynamic_windows, compare_performance

# Set a fixed random seed for reproducible tests
np.random.seed(42)

# Common output file for all tests
OUTPUT_FILE = str(Path(__file__).absolute().parent / "performance_results.xlsx")

# Common datasets for all tests
DATASETS = [
    (create_dataset_with_dynamic_windows(n_ids=5, n_timestamps=100, max_window_size=10), "Small Dataset"),
    (create_dataset_with_dynamic_windows(n_ids=50, n_timestamps=1000, max_window_size=50), "Medium Dataset"),
    (create_dataset_with_dynamic_windows(n_ids=500, n_timestamps=10000, max_window_size=100), "Large Dataset"),
]

# Common window types for all tests
WINDOW_TYPES = [
    WindowType.EXPANDING(),
    WindowType.ROLLING(size=10, only_full_window=True),
    WindowType.ROLLING(size=100, only_full_window=True),
    WindowType.ROLLING(size=1000, only_full_window=True),
    WindowType.DYNAMIC(len_column_name="dynamic_len"),
]


@pytest.mark.performance
def test_absolute_sum_of_changes_performance() -> None:
    """Compare performance of AbsoluteSumOfChanges implementations across various window types."""
    implementations = [
        (AbsoluteSumOfChangesWithOptimization, "optimized"),
        (AbsoluteSumOfChangesWithoutOptimization, "non_optimized"),
        (AbsoluteSumOfChanges, "strategy_selector"),
    ]

    # Run performance comparison
    compare_performance(
        datasets=DATASETS,
        implementations=implementations,
        window_types=WINDOW_TYPES,
        output_file=OUTPUT_FILE,
    )


@pytest.mark.performance
def test_max_performance() -> None:
    """Compare performance of Max implementations across various window types."""
    implementations = [
        (MaxWithOptimization, "optimized"),
        (MaxWithoutOptimization, "non_optimized"),
        (Max, "strategy_selector"),
    ]

    # Run performance comparison
    compare_performance(
        datasets=DATASETS,
        implementations=implementations,
        window_types=WINDOW_TYPES,
        output_file=OUTPUT_FILE,
    )


@pytest.mark.performance
def test_min_performance() -> None:
    """Compare performance of Min implementations across various window types."""
    implementations = [
        (MinWithOptimization, "optimized"),
        (MinWithoutOptimization, "non_optimized"),
        (Min, "strategy_selector"),
    ]

    # Run performance comparison
    compare_performance(
        datasets=DATASETS,
        implementations=implementations,
        window_types=WINDOW_TYPES,
        output_file=OUTPUT_FILE,
    )


@pytest.mark.performance
def test_sum_performance() -> None:
    """Compare performance of Sum implementations across various window types."""
    implementations = [
        (SumWithPrefixSumOptimization, "optimized"),
        (SumWithoutOptimization, "non_optimized"),
        (Sum, "strategy_selector"),
    ]

    # Run performance comparison
    compare_performance(
        datasets=DATASETS,
        implementations=implementations,
        window_types=WINDOW_TYPES,
        output_file=OUTPUT_FILE,
    )
