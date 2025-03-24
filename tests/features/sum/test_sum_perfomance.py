import time
import pytest
import polars as pl
import numpy as np
from chrono_features.features._base import WindowType
from chrono_features.features.sum import Sum
from chrono_features.ts_dataset import TSDataset


@pytest.fixture
def large_dataset(n_ids=50, n_timestamps=1000000):
    # Create a large dataset for performance testing
    ids = np.repeat(range(n_ids), n_timestamps)
    timestamps = np.tile(np.arange(1, n_timestamps + 1), n_ids)
    values = np.random.rand(n_ids * n_timestamps)  # Random values
    data = pl.DataFrame(
        {
            "id": ids,
            "timestamp": timestamps,
            "value": values,
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


@pytest.mark.performance
def test_performance_comparison(large_dataset):
    # window_size = 50
    # Test the method with optimization
    # Test the method without optimization
    sum_transformer_without_opt = Sum(
        columns="value",
        use_prefix_sum_optimization=False,
        window_types=WindowType.EXPANDING(),
        out_column_names=["s2"],
    )

    start_time = time.time()
    transformed_dataset_without_opt = sum_transformer_without_opt.transform(large_dataset)
    time_without_opt = time.time() - start_time

    sum_transformer_with_opt = Sum(
        columns="value",
        use_prefix_sum_optimization=True,
        window_types=WindowType.EXPANDING(),
        out_column_names=["s1"],
    )

    start_time = time.time()
    transformed_dataset_with_opt = sum_transformer_with_opt.transform(large_dataset)
    time_with_opt = time.time() - start_time

    # Verify that the results match
    result_with_opt = transformed_dataset_with_opt.data["s1"].to_numpy()
    result_without_opt = transformed_dataset_without_opt.data["s2"].to_numpy()

    np.testing.assert_array_almost_equal(result_with_opt, result_without_opt, decimal=3)

    # Print execution times
    print(f"\nExecution time with optimization: {time_with_opt:.4f} seconds")
    print(f"Execution time without optimization: {time_without_opt:.4f} seconds")

    # Compare performance
    if time_with_opt < time_without_opt:
        print("Optimization is faster.")
    else:
        print("Optimization did not provide a speed advantage.")
