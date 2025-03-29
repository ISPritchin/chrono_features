from pathlib import Path

import pytest

from chrono_features import WindowType
from chrono_features.features.sum import SumWithPrefixSumOptimization, SumWithoutOptimization, Sum
from tests.utils.performance import create_dataset_with_dynamic_windows, compare_performance


@pytest.mark.performance
def test_performance_comparison() -> None:
    """Test performance of Sum feature with different window types and optimization settings across dataset sizes."""
    # Define datasets to test
    datasets = {
        "medium": create_dataset_with_dynamic_windows(n_ids=50, n_timestamps=10000, max_window_size=100),
        "large": create_dataset_with_dynamic_windows(n_ids=500, n_timestamps=10000, max_window_size=100),
    }

    # Define window types to test
    window_types = [
        WindowType.EXPANDING(),
        WindowType.ROLLING(size=10),
        WindowType.ROLLING(size=100),
        WindowType.ROLLING(size=1000),
        WindowType.DYNAMIC(len_column_name="dynamic_len"),
    ]

    # Define implementations to test
    implementations = [
        (SumWithPrefixSumOptimization, "optimized"),
        (SumWithoutOptimization, "non_optimized"),
        (Sum, "strategy_selector"),
    ]

    # Output file path
    output_file = str(Path(__file__).absolute().parent / "performance_results.xlsx")

    # Run the performance comparison for each dataset
    for dataset_name, dataset in datasets.items():
        # Run the performance comparison
        compare_performance(
            dataset=dataset,
            implementations=implementations,
            window_types=window_types,
            column_name="value",
            output_file=output_file,
            dataset_name=dataset_name,
        )
