import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTS,
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
    StrategySelector,
)
from chrono_features.window_type import WindowBase, WindowType


@numba.njit  # pragma: no cover
def process_expanding(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Efficiently calculate cumulative absolute sum of changes for expanding windows.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.

    Returns:
        np.ndarray: Absolute sum of changes values for each expanding window.
    """
    result = np.empty(len(feature), dtype=np.float64)
    cumulative_sum = 0.0
    prev_value = np.nan

    for i in range(len(lens)):
        current_len = lens[i]
        if current_len == 1:
            # Reset for new time series
            result[i] = np.nan
            cumulative_sum = 0.0
            prev_value = feature[i]
        else:
            # Add absolute difference to cumulative sum
            current_value = feature[i]
            abs_diff = abs(current_value - prev_value)
            cumulative_sum += abs_diff
            result[i] = cumulative_sum
            prev_value = current_value

    return result


@numba.njit  # pragma: no cover
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Calculate absolute sum of changes for dynamic windows using prefix sum optimization.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Absolute sum of changes values for each dynamic window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    # Create a buffer for absolute differences
    buffer_size = ts_lens.max()
    abs_diffs = np.empty(buffer_size, dtype=np.float64)
    prefix_sum_array = np.empty(buffer_size + 1, dtype=np.float64)

    end = 0
    for i in range(len(ts_lens)):
        current_len = ts_lens[i]
        start = end
        end = start + current_len

        # Calculate absolute differences for this time series
        for j in range(start + 1, end):
            abs_diffs[j - start] = abs(feature[j] - feature[j - 1])

        # Create prefix sum array of absolute differences
        prefix_sum_array[0] = 0
        for j in range(current_len):
            prefix_sum_array[j] = prefix_sum_array[j - 1] + abs_diffs[j]

        # Calculate results for each point
        for j in range(start, end):
            if lens[j] <= 1:
                result[j] = np.nan  # Need at least 2 points for changes
                continue

            index_of_end_of_window = j - start
            # Determine window boundaries
            window_size = min(lens[j], index_of_end_of_window + 1)
            if window_size <= 1:
                result[j] = np.nan  # No changes with just one point
                continue

            # Calculate sum of absolute differences in the window
            end_idx = index_of_end_of_window
            start_idx = max(0, index_of_end_of_window - window_size + 1)

            result[j] = prefix_sum_array[end_idx] - prefix_sum_array[start_idx]

    return result


@numba.njit  # pragma: no cover
def process_rolling(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Optimized processing for rolling windows.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Absolute sum of changes values for each rolling window.
    """
    return process_dynamic(feature, lens, ts_lens)


class AbsoluteSumOfChangesWithOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    """Absolute sum of changes feature generator with optimized implementation.

    This class provides efficient calculations for different window types
    by using specialized algorithms that avoid redundant calculations.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the optimized absolute sum of changes feature generator.

        Args:
            columns: Columns to calculate absolute sum of changes for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="absolute_sum_of_changes",
        )

    @staticmethod
    def process_all_ts(
        feature: np.ndarray,
        ts_lens: np.ndarray,
        lens: np.ndarray,
        window_type: WindowBase,
    ) -> np.ndarray:
        """Process all time series using the appropriate method based on window type.

        Args:
            feature: Array of feature values.
            ts_lens: Lengths of individual time series.
            lens: Array of window lengths for each point.
            window_type: Type of window to use.

        Returns:
            np.ndarray: Absolute sum of changes values for each window.

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


class AbsoluteSumOfChangesWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Absolute sum of changes feature generator for time series data.

    Calculates the sum of absolute differences between consecutive values within
    specified windows. This feature measures the total amount of change or volatility
    in a time series segment.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "absolute_sum_of_changes",
    ) -> None:
        """Initialize the absolute sum of changes feature generator.

        Args:
            columns: Columns to calculate absolute sum of changes for.
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
    @numba.njit  # pragma: no cover
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the absolute sum of changes of the input array.

        Args:
            xs (np.ndarray): The input window.

        Returns:
            np.ndarray: The absolute sum of changes or NaN if the window has 1 or fewer elements.
        """
        if len(xs) <= 1:
            return np.nan

        return np.sum(np.abs(xs[1:] - xs[:-1]))


class AbsoluteSumOfChanges(StrategySelector):
    """Factory class for creating absolute sum of changes feature generators.

    Provides a unified interface that dynamically selects the optimal implementation
    based on the window type and user preference.

    Examples:
        >>> from chrono_features.features import AbsoluteSumOfChanges
        >>> from chrono_features.window_type import WindowType
        >>> from chrono_features.ts_dataset import TSDataset
        >>> import polars as pl
        >>>
        >>> # Create sample data
        >>> data = pl.DataFrame({
        ...     "id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        ...     "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4],
        ...     "value": [10, 12, 9, 11, 14, 20, 22, 19, 21]
        ... })
        >>> dataset = TSDataset(data, id_column="id", timestamp_column="timestamp")
        >>>
        >>> # Create absolute sum of changes with rolling window of size 3
        >>> asc = AbsoluteSumOfChanges(
        ...     columns='value',
        ...     window_types=WindowType.ROLLING(size=3),
        ...     out_column_names='value_asc'
        ... )
        >>>
        >>> # Apply to dataset
        >>> transformed_dataset = asc.transform(dataset)
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        use_optimization: bool = False,
    ) -> None:
        """Initialize the absolute sum of changes feature generator with dynamic strategy selection.

        Args:
            columns: Columns to calculate absolute sum of changes for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
            use_optimization: Whether to use the optimized implementation
                for non-expanding windows. Expanding windows always use optimization.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )
        self.use_optimization = use_optimization

    def _select_implementation_type(self, window_type: WindowBase) -> type:
        """Select the appropriate implementation class based on window type and optimization flag.

        Args:
            window_type: The window type to process.

        Returns:
            The appropriate implementation class.
        """
        # Use optimized implementation in these cases
        min_window_size_for_optimization = 50
        if any(
            [
                isinstance(window_type, WindowType.EXPANDING),
                isinstance(window_type, WindowType.ROLLING)
                and (self.use_optimization or window_type.size > min_window_size_for_optimization),
                isinstance(window_type, WindowType.DYNAMIC) and self.use_optimization,
            ],
        ):
            return AbsoluteSumOfChangesWithOptimization

        # Default to standard implementation
        return AbsoluteSumOfChangesWithoutOptimization
