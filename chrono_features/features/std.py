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
def process_expanding_std(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Efficiently calculate standard deviation for expanding windows.

    This implementation uses a running algorithm to compute standard deviation
    without recalculating the entire window each time.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.

    Returns:
        np.ndarray: Standard deviation values for each expanding window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    # Running sum and sum of squares for efficient calculation
    sum_x = 0.0
    sum_x2 = 0.0
    count = 0

    for i in range(len(lens)):
        if lens[i] == 1:  # New time series starts
            sum_x = feature[i]
            sum_x2 = feature[i] * feature[i]
            count = 1
            result[i] = 0.0  # Standard deviation of a single value is 0
        else:
            # Update running sums
            sum_x += feature[i]
            sum_x2 += feature[i] * feature[i]
            count += 1

            # Calculate standard deviation using the formula:
            # std = sqrt((sum_x2 - (sum_x)^2/n) / n)
            if count > 1:
                variance = (sum_x2 - (sum_x * sum_x) / count) / count
                # Handle numerical precision issues
                variance = max(variance, 0)
                result[i] = np.sqrt(variance)
            else:
                result[i] = 0.0

    return result


@numba.njit  # pragma: no cover
def process_rolling_std(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray, window_size: int) -> np.ndarray:
    """Optimized processing for rolling windows standard deviation calculation.

    This implementation uses a sliding window approach with running sums,
    efficiently adding new elements and removing old ones as the window moves.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Standard deviation values for each rolling window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    end = 0
    for i in range(len(ts_lens)):
        current_len = ts_lens[i]
        start = end
        end = start + current_len

        # For each time series, maintain running sums
        sum_x = 0.0
        sum_x2 = 0.0
        window_values = np.empty(window_size, dtype=np.float64)
        window_size_actual = 0

        # For each point in the time series
        for j in range(start, end):
            if lens[j] == 0:
                result[j] = np.nan
                continue

            # Calculate window boundaries
            window_start = max(j - lens[j] + 1, start)
            window_end = j + 1

            # If this is the first point or after a gap, initialize sums
            if j == start or window_size_actual == 0:
                sum_x = 0.0
                sum_x2 = 0.0
                window_size_actual = 0

                for k in range(window_start, window_end):
                    window_values[window_size_actual] = feature[k]
                    sum_x += feature[k]
                    sum_x2 += feature[k] * feature[k]
                    window_size_actual += 1
            else:
                # Remove oldest value if window is full and sliding forward
                if j - window_size >= start and window_size_actual == window_size:
                    old_value = feature[j - window_size]
                    sum_x -= old_value
                    sum_x2 -= old_value * old_value

                    # Shift values in the buffer
                    for k in range(window_size_actual - 1):
                        window_values[k] = window_values[k + 1]

                    window_size_actual -= 1

                # Add new value
                new_value = feature[j]
                window_values[window_size_actual] = new_value
                sum_x += new_value
                sum_x2 += new_value * new_value
                window_size_actual += 1

            # Calculate standard deviation
            if window_size_actual > 1:
                variance = (sum_x2 - (sum_x * sum_x) / window_size_actual) / window_size_actual
                # Handle numerical precision issues
                variance = max(variance, 0)
                result[j] = np.sqrt(variance)
            else:
                result[j] = 0.0

    return result


@numba.njit  # pragma: no cover
def process_dynamic_std(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Optimized processing for dynamic windows standard deviation calculation.

    This implementation handles variable-sized windows efficiently by
    calculating standard deviation directly for each window.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of dynamic window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Standard deviation values for each dynamic window.
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

            # Calculate standard deviation directly
            window_values = feature[window_start:window_end]

            if len(window_values) > 1:
                result[j] = np.std(window_values)
            else:
                result[j] = 0.0

    return result


class StdWithOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    """Standard deviation feature generator with optimized implementation.

    This class provides more efficient standard deviation calculations for different window types
    by using specialized algorithms that avoid redundant calculations.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[str] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the optimized standard deviation feature generator.

        Args:
            columns: Columns to calculate standard deviation for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="std",
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
            np.ndarray: Standard deviation values for each window.

        Raises:
            ValueError: If an unsupported window type is provided.
        """
        if isinstance(window_type, WindowType.EXPANDING):
            return process_expanding_std(
                feature=feature,
                lens=lens,
            )
        if isinstance(window_type, WindowType.ROLLING):
            return process_rolling_std(feature=feature, lens=lens, ts_lens=ts_lens, window_size=window_type.size)
        if isinstance(window_type, WindowType.DYNAMIC):
            return process_dynamic_std(feature=feature, lens=lens, ts_lens=ts_lens)

        msg = "Unsupported window type"
        raise ValueError(msg)


class StdWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Standard deviation feature generator using the standard implementation.

    This class uses the base class's window processing logic with a simple standard deviation function.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "std",
    ) -> None:
        """Initialize the standard standard deviation feature generator.

        Args:
            columns: Columns to calculate standard deviation for.
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
        """Calculate the standard deviation of the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray: Standard deviation value or NaN if array has 1 or fewer elements.
        """
        if len(xs) == 0:
            return np.nan

        return np.std(xs)


class Std(StrategySelector):
    """Factory class for creating standard deviation feature generators with dynamic strategy selection.

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
        """Initialize the standard deviation feature generator with dynamic strategy selection.

        Args:
            columns: Columns to calculate standard deviation for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
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
                isinstance(window_type, WindowType.DYNAMIC),
            ],
        ):
            return StdWithOptimization

        # Default to standard implementation
        return StdWithoutOptimization
