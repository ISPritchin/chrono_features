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
    """Process expanding window maximum calculation with Numba optimization.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.

    Returns:
        np.ndarray: Maximum values for each expanding window.
    """
    result = np.empty(len(feature), dtype=np.float64)
    current_max = -np.inf
    for i in range(len(lens)):
        if lens[i] == 1:  # Start of a new client/time series
            current_max = feature[i]
        else:
            if feature[i] > current_max or np.isnan(feature[i]):
                current_max = feature[i]
            current_max = max(current_max, feature[i])
        result[i] = current_max
    return result


@numba.njit  # pragma: no cover
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Process dynamic window maximum calculation with Numba optimization.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Maximum values for each dynamic window.
    """
    result = np.empty(len(feature), dtype=np.float64)

    for i in range(len(ts_lens)):
        start = ts_lens[:i].sum()
        end = start + ts_lens[i]

        for j in range(start, end):
            potential_start = j - lens[j] + 1
            window_start = potential_start if potential_start >= start else start
            if lens[j]:
                result[j] = np.max(feature[window_start : j + 1])
            else:
                result[j] = np.nan
    return result


@numba.njit  # pragma: no cover
def process_rolling(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """Process rolling window maximum calculation with Numba optimization.

    For rolling windows, we use the same implementation as dynamic windows.

    Args:
        feature (np.ndarray): Array of feature values.
        lens (np.ndarray): Array of window lengths for each point.
        ts_lens (np.ndarray): Lengths of individual time series.

    Returns:
        np.ndarray: Maximum values for each rolling window.
    """
    return process_dynamic(feature, lens, ts_lens)


class MaxWithOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    """Maximum feature generator with optimized implementation for different window types.

    Uses specialized algorithms for each window type to improve performance.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the optimized maximum feature generator.

        Args:
            columns: Columns to calculate maximum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="max",
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
            np.ndarray: Maximum values for each window.

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


class MaxWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Maximum feature generator using the standard implementation.

    Uses the base class's window processing logic with a simple max function.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the standard maximum feature generator.

        Args:
            columns: Columns to calculate maximum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="max",
        )

    @staticmethod
    @numba.njit  # pragma: no cover
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the maximum value in the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray : Maximum value.
        """
        return np.max(xs)


class Max(StrategySelector):
    """Maximum feature generator with dynamic strategy selection.

    Selects between optimized and standard implementations based on window type.
    """

    def _select_implementation_type(self, window_type: WindowBase) -> type:
        """Select the appropriate implementation class based on window type.

        Args:
            window_type: The window type to process.

        Returns:
            The appropriate implementation class.
        """
        if isinstance(window_type, WindowType.EXPANDING):
            return MaxWithOptimization
        return MaxWithoutOptimization
