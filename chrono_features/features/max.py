import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTS,
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
)
from chrono_features.window_type import WindowBase, WindowType


@numba.njit
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


@numba.njit
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


@numba.njit
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
        window_types: list[str] | WindowType,
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
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "max",
    ) -> None:
        """Initialize the standard maximum feature generator.

        Args:
            columns: Columns to calculate maximum for.
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
        """Calculate the maximum value in the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray : Maximum value.
        """
        return np.max(xs)


class Max:
    """Factory class for creating maximum feature generators.

    Provides a unified interface to create either optimized or standard implementations.
    """

    def __new__(
        cls,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        use_optimization: bool = False,
    ) -> MaxWithOptimization | MaxWithoutOptimization:
        """Create a maximum feature generator.

        Args:
            columns: Columns to calculate maximum for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
            use_optimization: Whether to use the optimized implementation.

        Returns:
            Either MaxWithOptimization or MaxWithoutOptimization based on the use_optimization flag.
        """
        if use_optimization or isinstance(window_types, WindowType.EXPANDING):
            return MaxWithOptimization(
                columns=columns,
                window_types=window_types,
                out_column_names=out_column_names,
            )
        return MaxWithoutOptimization(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )
