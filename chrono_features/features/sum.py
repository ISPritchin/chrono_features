import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTS,
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
)
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowBase, WindowType


@numba.njit
def process_expanding(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Efficiently calculate cumulative sum for expanding windows using Numba optimization.

    This function maintains a running sum for each time series, resetting when a new
    time series begins (indicated by lens[i] == 1).

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.

    Returns:
        np.ndarray: Cumulative sum values for each expanding window.
    """
    result = np.empty(len(feature), dtype=np.float64)
    for i in range(len(lens)):
        current_len = lens[i]
        if current_len == 1:
            cumulative_sum = feature[i]
        else:
            cumulative_sum += feature[i]
        result[i] = cumulative_sum

    return result


@numba.njit
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Calculate sum for dynamic windows using prefix sum optimization.

    This function uses a prefix sum array to efficiently calculate sums for
    variable-sized windows within each time series.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Sum values for each dynamic window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    buffer_size = ts_lens.max() + 1
    prefix_sum_array = np.empty(buffer_size, dtype=np.float64)

    end = 0
    for i in range(len(ts_lens)):
        current_len = ts_lens[i]
        start = end
        end = start + current_len

        prefix_sum_array[0] = 0
        for j in range(start, end):
            prefix_sum_array[j - start + 1] = prefix_sum_array[j - start] + feature[j]

        for j in range(start, end):
            v = j - start + 1
            start_window = v - lens[j]
            if lens[j] == 0:
                result[j] = np.nan
            else:
                result[j] = prefix_sum_array[v] - prefix_sum_array[start_window]

    return result


@numba.njit
def process_rolling(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Optimized processing for rolling windows.

    For rolling windows, this implementation uses the same algorithm as dynamic windows
    since both can be efficiently calculated using prefix sums.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Sum values for each rolling window.
    """
    return process_dynamic(feature, lens, ts_lens)


class SumWithPrefixSumOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    """Sum feature generator with optimized implementation using prefix sums.

    This class provides efficient sum calculations for different window types
    by using specialized prefix sum algorithms that avoid redundant calculations.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[str] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the optimized sum feature generator.

        Args:
            columns: Columns to calculate sum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="sum",
        )

    @staticmethod
    def process_all_ts(
        feature: np.ndarray,
        ts_lens: np.ndarray,
        lens: np.ndarray,
        window_type: WindowBase,
    ) -> np.ndarray:
        """Process all time series using the appropriate method based on window type.

        Selects the optimal algorithm for each window type to maximize performance.

        Args:
            feature: Array of feature values.
            ts_lens: Lengths of individual time series.
            lens: Array of window lengths for each point.
            window_type: Type of window to use.

        Returns:
            np.ndarray: Sum values for each window.

        Raises:
            ValueError: If an unsupported window type is provided.
        """
        if isinstance(window_type, WindowType.EXPANDING):
            return process_expanding(
                feature=feature,
                lens=lens,
            )
        if isinstance(window_type, WindowType.ROLLING):
            return process_rolling(feature=feature, lens=lens, ts_lens=ts_lens)
        if isinstance(window_type, WindowType.DYNAMIC):
            return process_dynamic(feature=feature, lens=lens, ts_lens=ts_lens)

        msg = "Unsupported window type"
        raise ValueError(msg)


class SumWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Sum feature generator using the standard implementation.

    This class uses the base class's window processing logic with a simple sum function,
    which is less optimized but more straightforward than the prefix sum approach.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "sum",
    ) -> None:
        """Initialize the standard sum feature generator.

        Args:
            columns: Columns to calculate sum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
            func_name: Name of the function for output column naming.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the sum of the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray: Sum value.
        """
        return np.sum(xs)

    def transform_for_window_type(self, dataset: TSDataset, column: "str", window_type: WindowBase) -> np.ndarray:
        """Transform data for a specific window type.

        For expanding windows, this method uses the optimized implementation
        even when the standard implementation is selected.

        Args:
            dataset: Input time series dataset.
            column: Column to transform.
            window_type: Window type to use.

        Returns:
            np.ndarray: Transformed feature values.
        """
        if not isinstance(window_type, WindowType.EXPANDING):
            return super().transform_for_window_type(dataset=dataset, column=column, window_type=window_type)

        return SumWithPrefixSumOptimization(
            columns=column,
            window_types=window_type,
            out_column_names=None,
        ).transform_for_window_type(dataset=dataset, column=column, window_type=window_type)


class Sum:
    """Factory class for creating sum feature generators.

    Provides a unified interface to create either optimized or standard implementations
    based on the user's preference.

    Examples:
        Basic usage with a single column and expanding window:

        >>> from chrono_features.features import Sum
        >>> from chrono_features.window_type import WindowType
        >>> from chrono_features.ts_dataset import TSDataset
        >>> import polars as pl
        >>>
        >>> # Create sample data
        >>> data = pl.DataFrame({
        ...     "id": [1, 1, 1, 2, 2],
        ...     "timestamp": [1, 2, 3, 1, 2],
        ...     "value": [10, 20, 30, 40, 50]
        ... })
        >>> dataset = TSDataset(data, id_column="id", timestamp_column="timestamp")
        >>>
        >>> # Create a sum feature generator for column 'value' with expanding window
        >>> sum_generator = Sum(
        ...     columns='value',
        ...     window_types=WindowType.EXPANDING(),
        ...     out_column_names='value_sum'
        ... )
        >>>
        >>> # Apply to dataset
        >>> transformed_dataset = sum_generator.transform(dataset)
        >>> print(transformed_dataset.data["value_sum"])
        shape: (5,)
        Series: 'value_sum' [f64]
        [
            10.0
            30.0
            60.0
            40.0
            90.0
        ]

        Using optimization for better performance with large window_size or expanding window:

        >>> # Create an optimized sum generator
        >>> optimized_sum = Sum(
        ...     columns='value',
        ...     window_types=WindowType.ROLLING(size=500),
        ...     out_column_names='value_sum_2',
        ...     use_prefix_sum_optimization=True
        ... )
    """

    def __new__(
        cls,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        use_prefix_sum_optimization: bool = False,
    ) -> SumWithPrefixSumOptimization | SumWithoutOptimization:
        """Create a sum feature generator.

        Args:
            columns: Columns to calculate sum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
            use_prefix_sum_optimization: Whether to use the optimized implementation.

        Returns:
            Either SumWithPrefixSumOptimization or SumWithoutOptimization based on the
            use_prefix_sum_optimization flag.
        """
        if use_prefix_sum_optimization or isinstance(window_types, WindowType.EXPANDING):
            return SumWithPrefixSumOptimization(
                columns=columns,
                window_types=window_types,
                out_column_names=out_column_names,
            )
        return SumWithoutOptimization(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )
