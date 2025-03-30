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
from chrono_features.features.mean import MeanWithPrefixSumOptimization, MeanWithoutOptimization, Mean
from chrono_features.features.median import MedianWithOptimization, MedianWithoutOptimization, Median
from chrono_features.features.min import MinWithOptimization, MinWithoutOptimization, Min
from chrono_features.features.sum import SumWithPrefixSumOptimization, SumWithoutOptimization, Sum
from tests.utils.performance import create_dataset_with_dynamic_windows, compare_performance

# Set a fixed random seed for reproducible tests
np.random.seed(42)

# Common output file for all tests
OUTPUT_FILE = str(Path(__file__).absolute().parent / "performance_results.xlsx")

# Common datasets for all tests (ordered from smallest to largest)
DATASETS = [
    (create_dataset_with_dynamic_windows(n_ids=5, n_timestamps=100, max_window_size=10), "D1"),
    (create_dataset_with_dynamic_windows(n_ids=50, n_timestamps=200, max_window_size=50), "D2"),
    (create_dataset_with_dynamic_windows(n_ids=500, n_timestamps=300, max_window_size=100), "D3"),
    (create_dataset_with_dynamic_windows(n_ids=5000, n_timestamps=300, max_window_size=100), "D4"),
    (create_dataset_with_dynamic_windows(n_ids=50000, n_timestamps=300, max_window_size=100), "D5"),
]

# Common window types for all tests
WINDOW_TYPES = [
    WindowType.EXPANDING(),
    WindowType.ROLLING(size=10, only_full_window=True),
    WindowType.ROLLING(size=30, only_full_window=True),
    WindowType.ROLLING(size=50, only_full_window=True),
    WindowType.DYNAMIC(len_column_name="dynamic_len"),
]

TIME_THRESHOLD = 2


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
        time_threshold_seconds=TIME_THRESHOLD,
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
        time_threshold_seconds=TIME_THRESHOLD,
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
        time_threshold_seconds=TIME_THRESHOLD,
    )


@pytest.mark.performance
def test_mean_performance() -> None:
    """Compare performance of Mean implementations across various window types."""
    implementations = [
        (MeanWithPrefixSumOptimization, "optimized"),
        (MeanWithoutOptimization, "non_optimized"),
        (Mean, "strategy_selector"),
    ]

    # Run performance comparison
    compare_performance(
        datasets=DATASETS,
        implementations=implementations,
        window_types=WINDOW_TYPES,
        output_file=OUTPUT_FILE,
        time_threshold_seconds=TIME_THRESHOLD,
    )


@pytest.mark.performance
def test_median_performance() -> None:
    """Compare performance of Median implementations across various window types."""
    implementations = [
        (MedianWithOptimization, "optimized"),
        (MedianWithoutOptimization, "non_optimized"),
        (Median, "strategy_selector"),
    ]

    # Run performance comparison
    compare_performance(
        datasets=DATASETS,
        implementations=implementations,
        window_types=WINDOW_TYPES,
        output_file=OUTPUT_FILE,
        time_threshold_seconds=TIME_THRESHOLD,
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
        time_threshold_seconds=TIME_THRESHOLD,
    )
