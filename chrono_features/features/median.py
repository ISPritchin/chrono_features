# ruff: noqa: C901, PLR0912

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
    """Efficiently calculate median for expanding windows.

    This implementation maintains a sorted array for each time series and updates it
    as new values are added, which is more efficient than resorting the entire array.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.

    Returns:
        np.ndarray: Median values for each expanding window.
    """
    result = np.empty(len(feature), dtype=np.float64)
    buffer = np.empty(lens.max(), dtype=np.float64)

    current_len = 0
    for i in range(len(lens)):
        if lens[i] == 1:  # New time series starts
            buffer[0] = feature[i]
            current_len = 1
        else:
            # Insert new value into sorted buffer using insertion sort
            j = current_len - 1
            while j >= 0 and buffer[j] > feature[i]:
                buffer[j + 1] = buffer[j]
                j -= 1
            buffer[j + 1] = feature[i]
            current_len += 1

        # Calculate median from sorted buffer
        if current_len % 2 == 0:  # Even number of elements
            result[i] = (buffer[current_len // 2 - 1] + buffer[current_len // 2]) / 2
        else:  # Odd number of elements
            result[i] = buffer[current_len // 2]

    return result


@numba.njit  # pragma: no cover
def process_rolling(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray, window_size: int) -> np.ndarray:
    """Optimized processing for rolling windows median calculation.

    This implementation uses a sliding window approach with a sorted buffer,
    efficiently adding new elements and removing old ones as the window moves.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Median values for each rolling window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    end = 0
    for i in range(len(ts_lens)):
        current_len = ts_lens[i]
        start = end
        end = start + current_len

        # For each time series, maintain a sorted buffer
        buffer = np.empty(window_size, dtype=np.float64)
        buffer_size = 0

        # For each point in the time series
        for j in range(start, end):
            if lens[j] == 0:
                result[j] = np.nan
                continue

            # Calculate window boundaries
            window_start = max(j - lens[j] + 1, start)
            window_end = j + 1

            # If this is the first point or after a gap, initialize buffer
            if buffer_size == 0 or j == start:
                # Fill and sort buffer
                window_values = feature[window_start:window_end].copy()
                window_values.sort()
                buffer_size = len(window_values)
                buffer[:buffer_size] = window_values
            else:
                # Remove oldest value if window is full and sliding forward
                if j - window_size >= start and buffer_size == window_size:
                    old_value = feature[j - window_size]
                    # Find and remove old value from buffer
                    idx = 0
                    while idx < buffer_size and buffer[idx] != old_value:
                        idx += 1

                    if idx < buffer_size:  # Found the value
                        # Shift elements to remove the old value
                        for k in range(idx, buffer_size - 1):
                            buffer[k] = buffer[k + 1]
                        buffer_size -= 1

                # Insert new value into sorted buffer
                new_value = feature[j]
                idx = buffer_size - 1
                while idx >= 0 and buffer[idx] > new_value:
                    if idx + 1 < window_size:  # Prevent buffer overflow
                        buffer[idx + 1] = buffer[idx]
                    idx -= 1

                if idx + 1 < window_size:  # Prevent buffer overflow
                    buffer[idx + 1] = new_value
                    buffer_size = min(buffer_size + 1, window_size)

            # Calculate median from sorted buffer
            if buffer_size % 2 == 0:  # Even number of elements
                result[j] = (buffer[buffer_size // 2 - 1] + buffer[buffer_size // 2]) / 2
            else:  # Odd number of elements
                result[j] = buffer[buffer_size // 2]

    return result


@numba.njit  # pragma: no cover
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Optimized processing for dynamic windows median calculation.

    This implementation handles variable-sized windows efficiently by
    sorting only the necessary elements for each window.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of dynamic window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Median values for each dynamic window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    end = 0
    for i in range(len(ts_lens)):
        current_len = ts_lens[i]
        start = end
        end = start + current_len

        for j in range(start, end):
            if lens[j] == 0:
                result[j] = np.nan
                continue

            # Calculate window boundaries based on dynamic window size
            window_start = max(j - lens[j] + 1, start)
            window_end = j + 1

            # Sort the window values and find median
            window_values = feature[window_start:window_end].copy()
            window_values.sort()

            if len(window_values) % 2 == 0:  # Even number of elements
                result[j] = (window_values[len(window_values) // 2 - 1] + window_values[len(window_values) // 2]) / 2
            else:  # Odd number of elements
                result[j] = window_values[len(window_values) // 2]

    return result


class MedianWithOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    """Median feature generator with optimized implementation.

    This class provides more efficient median calculations for different window types
    by using specialized algorithms that avoid redundant sorting operations.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[str] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the optimized median feature generator.

        Args:
            columns: Columns to calculate median for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="median",
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
            np.ndarray: Median values for each window.

        Raises:
            ValueError: If an unsupported window type is provided.
        """
        if isinstance(window_type, WindowType.EXPANDING):
            return process_expanding(
                feature=feature,
                lens=lens,
            )
        if isinstance(window_type, WindowType.ROLLING):
            return process_rolling(feature=feature, lens=lens, ts_lens=ts_lens, window_size=window_type.size)
        if isinstance(window_type, WindowType.DYNAMIC):
            return process_dynamic(feature=feature, lens=lens, ts_lens=ts_lens)

        msg = "Unsupported window type"
        raise ValueError(msg)


class MedianWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Median feature generator using the standard implementation.

    This class uses the base class's window processing logic with a simple median function.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "median",
    ) -> None:
        """Initialize the standard median feature generator.

        Args:
            columns: Columns to calculate median for.
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
        """Calculate the median value of the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray: Median value.
        """
        return np.median(xs)


class Median(StrategySelector):
    """Factory class for creating median feature generators with dynamic strategy selection.

    Provides a unified interface that dynamically selects the optimal implementation
    based on the window type and size.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the median feature generator with dynamic strategy selection.

        Args:
            columns: Columns to calculate median for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
            use_optimization: Whether to use the optimized implementation.
                For large windows, optimization is recommended.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )

    def _select_implementation_type(self, window_type: WindowBase) -> type:
        """Select the appropriate implementation class based on window type and optimization flag.

        Args:
            window_type: The window type to process.

        Returns:
            The appropriate implementation class.
        """
        # Use optimized implementation in these cases
        min_window_size_for_optimization = 10
        if any(
            [
                isinstance(window_type, WindowType.EXPANDING),
                isinstance(window_type, WindowType.ROLLING) and window_type.size > min_window_size_for_optimization,
            ],
        ):
            return MedianWithOptimization

        # Default to standard implementation
        return MedianWithoutOptimization
