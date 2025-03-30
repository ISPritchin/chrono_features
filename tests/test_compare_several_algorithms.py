# ruff: noqa: ANN401, T201

from typing import Any

import numpy as np
import pytest
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TaskProgressColumn

from chrono_features import TSDataset, WindowType
from chrono_features.features.absolute_sum_of_changes import (
    AbsoluteSumOfChangesWithOptimization,
    AbsoluteSumOfChangesWithoutOptimization,
)
from chrono_features.features.max import MaxWithOptimization, MaxWithoutOptimization
from chrono_features.features.mean import MeanWithPrefixSumOptimization, MeanWithoutOptimization
from chrono_features.features.median import MedianWithOptimization, MedianWithoutOptimization
from chrono_features.features.min import MinWithOptimization, MinWithoutOptimization
from chrono_features.features.sum import SumWithPrefixSumOptimization, SumWithoutOptimization
from tests.utils.compare_multiple_implementations import compare_multiple_implementations
from tests.utils.performance import create_dataset

# Set a fixed random seed for reproducible tests
np.random.seed(42)


@pytest.fixture
def medium_dataset() -> TSDataset:
    """Create a dataset with 300 time series, each with 20 points."""
    return create_dataset(n_ids=500, n_timestamps=30)


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
    # Test implementations
    implementations = [
        (optimized_implementation, "optimized"),
        (non_optimized_implementation, "non_optimized"),
    ]

    # Define window types to test
    window_types = [
        ("expanding", WindowType.EXPANDING()),
        ("rolling_partial", WindowType.ROLLING(size=10, only_full_window=False)),
        ("rolling_full", WindowType.ROLLING(size=10, only_full_window=True)),
    ]

    # Create console for rich output
    console = Console()

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
    ) as progress:
        # Add main task for feature
        feature_task = progress.add_task(
            f"[yellow]Testing {feature_name}",
            total=len(window_types) + 1,
        )  # +1 for dynamic window

        # Test each window type
        for window_name, window_type in window_types:
            # Update progress description
            progress.update(feature_task, description=f"[yellow]Testing {feature_name} with {window_name} window")

            # Run comparison
            compare_multiple_implementations(
                medium_dataset,
                implementations,
                window_type,
            )

            # Update progress
            progress.advance(feature_task)

        # Test dynamic window
        progress.update(feature_task, description=f"[yellow]Testing {feature_name} with dynamic window")

        # Add a dynamic window length column with values between 1 and 5
        window_lengths = np.random.randint(1, 6, size=len(medium_dataset.data))
        medium_dataset.add_feature("window_len", window_lengths)

        # Run comparison for dynamic window
        compare_multiple_implementations(
            medium_dataset,
            implementations,
            WindowType.DYNAMIC(len_column_name="window_len"),
        )

        # Complete the task
        progress.advance(feature_task)

    # Print completion message
    console.print(f"[green]✓ Completed all tests for {feature_name}")


@pytest.mark.parametrize(
    ("optimized_implementation", "non_optimized_implementation", "feature_name"),
    [
        pytest.param(
            AbsoluteSumOfChangesWithOptimization,
            AbsoluteSumOfChangesWithoutOptimization,
            "AbsoluteSumOfChanges",
            id="AbsoluteSumOfChanges",
        ),
        pytest.param(MaxWithOptimization, MaxWithoutOptimization, "Max", id="Max"),
        pytest.param(MeanWithPrefixSumOptimization, MeanWithoutOptimization, "Mean", id="Mean"),
        pytest.param(MedianWithOptimization, MedianWithoutOptimization, "Median", id="Median"),
        pytest.param(MinWithOptimization, MinWithoutOptimization, "Min", id="Min"),
        pytest.param(SumWithPrefixSumOptimization, SumWithoutOptimization, "Sum", id="Sum"),
    ],
)
def test_optimization_comparison(
    medium_dataset: TSDataset,
    optimized_implementation: Any,
    non_optimized_implementation: Any,
    feature_name: str,
) -> None:
    """Test that different implementations of the same feature produce identical results."""
    run_optimization_comparison_tests(
        medium_dataset=medium_dataset,
        optimized_implementation=optimized_implementation,
        non_optimized_implementation=non_optimized_implementation,
        feature_name=feature_name,
    )
