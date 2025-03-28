from pathlib import Path

import pytest

from chrono_features import WindowType
from chrono_features.features import Max
from tests.utils.performance import create_dataset, performance_comparison


@pytest.mark.performance
def test_performance_comparison() -> None:
    """Test performance of Max feature with different window types and optimization settings across dataset sizes."""
    # Define datasets to test
    datasets = {
        "medium": create_dataset(n_ids=50, n_timestamps=10000),
        "large": create_dataset(n_ids=500, n_timestamps=10000),
    }

    # Create transformer instances directly
    transformers = [
        Max(
            columns="value",
            window_types=WindowType.EXPANDING(),
            out_column_names=["max_expanding"],
        ),
        Max(
            columns="value",
            window_types=WindowType.ROLLING(size=10),
            out_column_names=["max_rolling_10"],
        ),
        Max(
            columns="value",
            window_types=WindowType.ROLLING(size=100),
            out_column_names=["max_rolling_100"],
        ),
        Max(
            columns="value",
            window_types=WindowType.ROLLING(size=1000),
            out_column_names=["max_rolling_1000"],
        ),
    ]

    # Run the performance comparison
    performance_comparison(
        datasets=datasets,
        transformers=transformers,
        output_xlsx_file_path=Path(__file__).absolute().parent / "performance_results.xlsx",
    )
