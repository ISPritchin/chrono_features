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
    """Process expanding window minimum calculation with Numba optimization.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.

    Returns:
        np.ndarray: Minimum values for each expanding window.
    """
    result = np.empty(len(feature), dtype=np.float64)
    current_min = np.inf
    for i in range(len(lens)):
        if lens[i] == 1:  # Start of a new client/time series
            current_min = feature[i]
        else:
            if feature[i] < current_min or np.isnan(feature[i]):
                current_min = feature[i]
            current_min = min(current_min, feature[i])
        result[i] = current_min
    return result


@numba.njit  # pragma: no cover
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Process dynamic window minimum calculation with Numba optimization.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Minimum values for each dynamic window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    for i in range(len(ts_lens)):
        start = ts_lens[:i].sum()
        end = start + ts_lens[i]

        for j in range(start, end):
            potential_start = j - lens[j] + 1
            window_start = potential_start if potential_start >= start else start
            if lens[j]:
                result[j] = np.min(feature[window_start : j + 1])
            else:
                result[j] = np.nan
    return result


@numba.njit  # pragma: no cover
def process_rolling(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Process rolling window minimum calculation with Numba optimization.

    Uses a sliding window approach that's more efficient than the dynamic window
    implementation for fixed-size windows.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Minimum values for each rolling window.
    """
    result = np.empty(len(feature), dtype=np.float64)
    window_size_threshold = 3

    ts_start = 0
    for ts_len in ts_lens:
        ts_end = ts_start + ts_len

        # Process each time series separately
        for i in range(ts_start, ts_end):
            window_size = lens[i]

            if window_size == 0:
                result[i] = np.nan
                continue

            # Start of window
            start_idx = max(ts_start, i - window_size + 1)

            # For small windows, just use direct min calculation
            if window_size <= window_size_threshold:
                result[i] = np.min(feature[start_idx : i + 1])
                continue

            # For the first point in each window size, calculate min directly
            if i == ts_start or lens[i] != lens[i - 1]:
                result[i] = np.min(feature[start_idx : i + 1])
                continue

            # For subsequent points, use sliding window optimization:
            # If the new value is smaller than previous min, it becomes the new min
            if feature[i] <= result[i - 1]:
                result[i] = feature[i]
            # If the value leaving the window was the min, recalculate
            elif start_idx > ts_start and feature[start_idx - 1] == result[i - 1]:
                result[i] = np.min(feature[start_idx : i + 1])
            # Otherwise, keep the previous min
            else:
                result[i] = result[i - 1]

        ts_start = ts_end

    return result


class MinWithOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    """Minimum feature generator with optimized implementation for different window types.

    Uses specialized algorithms for each window type to improve performance.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the optimized minimum feature generator.

        Args:
            columns: Columns to calculate minimum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="min",
        )

    @staticmethod
    def process_all_ts(
        feature: np.ndarray,
        ts_lens: np.ndarray,
        lens: np.ndarray,
        window_type: WindowBase,
    ) -> np.ndarray:
        """Process all time series with the appropriate algorithm based on window type.

        Args:
            feature: Array of feature values.
            ts_lens: Lengths of individual time series.
            lens: Array of window lengths for each point.
            window_type: Type of window to use.

        Returns:
            np.ndarray: Minimum values for each window.

        Raises:
            ValueError: If an unsupported window type is provided.
        """
        if isinstance(window_type, WindowType.EXPANDING):
            return process_expanding(feature=feature, lens=lens)
        if isinstance(window_type, WindowType.ROLLING):
            return process_rolling(feature=feature, lens=lens, ts_lens=ts_lens)
        if isinstance(window_type, WindowType.DYNAMIC):
            return process_dynamic(feature=feature, lens=lens, ts_lens=ts_lens)
        msg = "Unsupported window type"
        raise ValueError(msg)


class MinWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Minimum feature generator using the standard implementation.

    Uses the base class's window processing logic with a simple min function.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the standard minimum feature generator.

        Args:
            columns: Columns to calculate minimum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="min",
        )

    @staticmethod
    @numba.njit  # pragma: no cover
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the minimum value in the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray : Minimum value.
        """
        return np.min(xs)


class Min(StrategySelector):
    """Minimum feature generator with dynamic strategy selection.

    Selects between optimized and standard implementations based on window type.
    """

    def _select_implementation_type(self, window_type: WindowBase) -> type:
        """Select the appropriate implementation class based on window type.

        Args:
            window_type: The window type to process.

        Returns:
            The appropriate implementation class.
        """
        min_window_size_for_optimization = 50
        if isinstance(window_type, WindowType.EXPANDING) or (
            isinstance(window_type, WindowType.ROLLING) and (window_type.size > min_window_size_for_optimization),
        ):
            return MinWithOptimization
        return MinWithoutOptimization
