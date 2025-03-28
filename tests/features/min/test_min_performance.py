from pathlib import Path

import pytest

from chrono_features import WindowType
from chrono_features.features import Min
from tests.utils.performance import create_dataset_with_dynamic_windows, performance_comparison


@pytest.mark.performance
def test_performance_comparison() -> None:
    """Test performance of Min feature with different window types and optimization settings across dataset sizes."""
    # Define datasets to test
    datasets = {
        "medium": create_dataset_with_dynamic_windows(n_ids=50, n_timestamps=10000, max_window_size=100),
        "large": create_dataset_with_dynamic_windows(n_ids=500, n_timestamps=10000, max_window_size=100),
    }

    # Define window types to test
    window_types = [
        (WindowType.EXPANDING(), "expanding"),
        (WindowType.ROLLING(size=10), "rolling_10"),
        (WindowType.ROLLING(size=100), "rolling_100"),
        (WindowType.ROLLING(size=1000), "rolling_1000"),
        (WindowType.DYNAMIC(len_column_name="dynamic_len"), "dynamic"),
    ]

    # Create transformer instances using list comprehension
    transformers = [
        Min(
            columns="value",
            window_types=window_type,
            out_column_names=[f"min_{name}"],
        )
        for window_type, name in window_types
    ]

    # Run the performance comparison
    performance_comparison(
        datasets=datasets,
        transformers=transformers,
        output_xlsx_file_path=Path(__file__).absolute().parent / "performance_results.xlsx",
    )
