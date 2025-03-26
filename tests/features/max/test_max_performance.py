import time

import numpy as np
import polars as pl
import pytest

from chrono_features import TSDataset, WindowType
from chrono_features.features import Max


@pytest.fixture
def large_dataset(n_ids=50, n_timestamps=10000) -> TSDataset:
    ids = np.repeat(range(n_ids), n_timestamps)
    timestamps = np.tile(np.arange(1, n_timestamps + 1), n_ids)
    values = np.random.rand(n_ids * n_timestamps)
    data = pl.DataFrame(
        {
            "id": ids,
            "timestamp": timestamps,
            "value": values,
        }
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


@pytest.mark.performance
def test_performance_comparison(large_dataset: TSDataset) -> None:
    max_transformer_without_opt = Max(
        columns="value",
        use_optimization=False,
        window_types=WindowType.EXPANDING(),
        out_column_names=["m2"],
    )

    start_time = time.time()
    transformed_dataset_without_opt = max_transformer_without_opt.transform(large_dataset)
    time_without_opt = time.time() - start_time

    max_transformer_with_opt = Max(
        columns="value",
        use_optimization=True,
        window_types=WindowType.EXPANDING(),
        out_column_names=["m1"],
    )

    start_time = time.time()
    transformed_dataset_with_opt = max_transformer_with_opt.transform(large_dataset)
    time_with_opt = time.time() - start_time

    result_with_opt = transformed_dataset_with_opt.data["m1"].to_numpy()
    result_without_opt = transformed_dataset_without_opt.data["m2"].to_numpy()

    np.testing.assert_array_almost_equal(result_with_opt, result_without_opt, decimal=3)

    print(f"\nExecution time with optimization: {time_with_opt:.4f} seconds")
    print(f"Execution time without optimization: {time_without_opt:.4f} seconds")

    if time_with_opt < time_without_opt:
        print("Optimization is faster.")
    else:
        print("Optimization did not provide a speed advantage.")
