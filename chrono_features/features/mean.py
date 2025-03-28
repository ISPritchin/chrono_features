from collections.abc import Iterable

import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class Mean(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Mean feature generator for time series data.

    Calculates the arithmetic mean of values within specified windows.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "mean",
    ) -> None:
        """Initialize the mean feature generator.

        Args:
            columns: Columns to calculate mean for.
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
        """Calculate the arithmetic mean of the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray: Mean value.
        """
        return np.mean(xs)


class SimpleMovingAverage:
    """Factory class for creating simple moving average feature generators.

    Creates a Mean feature generator with a rolling window of specified size.
    """

    def __new__(
        cls,
        *,
        columns: str | list[str],
        window_size: int,
        out_column_names: str | list[str] | None = None,
        only_full_window: bool = False,
    ) -> Mean:
        """Create a simple moving average feature generator.

        Args:
            columns: Columns to calculate moving average for.
            window_size: Size of the rolling window.
            out_column_names: Names for output columns.
            only_full_window: Whether to calculate only for full windows.

        Returns:
            Mean: A Mean feature generator configured for simple moving average.

        Raises:
            ValueError: If window_size is less than or equal to 0.
        """
        if window_size <= 0:
            raise ValueError

        return Mean(
            columns=columns,
            window_types=WindowType.ROLLING(
                size=window_size,
                only_full_window=only_full_window,
            ),
            out_column_names=out_column_names,
            func_name="simple_moving_average",
        )


class WeightedMean(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Weighted mean feature generator for time series data.

    Calculates the weighted arithmetic mean of values within specified windows.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        weights: np.ndarray | None = None,
        out_column_names: list[str] | str | None = None,
        func_name: str = "weighted_mean",
    ) -> None:
        """Initialize the weighted mean feature generator.

        Args:
            columns: Columns to calculate weighted mean for.
            window_types: Types of windows to use.
            weights: Array of weights to apply to window values.
            out_column_names: Names for output columns.
            func_name: Name of the function for output column naming.

        Raises:
            ValueError: If weights is None.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )

        if weights is None:
            msg = "Weights cannot be None"
            raise ValueError(msg)

        self.weights = weights
        self.numba_kwargs = {"weights": weights}

    @staticmethod
    @numba.njit  # pragma: no cover
    def apply_func_to_full_window(
        feature: np.ndarray,
        func: callable,
        lens: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Apply a weighted function to sliding or expanding windows of a feature array.

        Args:
            feature: Input feature array.
            func: Numba-compiled function to apply.
            lens: Array of window lengths for each point.
            weights: Array of weights to apply to window values.

        Returns:
            np.ndarray: Result of applying the weighted function to each window.
        """
        result = np.empty(len(feature), dtype=np.float32)
        max_window_size = len(weights)

        for i in numba.prange(len(result)):
            if lens[i]:
                window_size = lens[i]
                window_weights = weights[max_window_size - window_size :]
                window_data = feature[i + 1 - window_size : i + 1]
                result[i] = func(window_data, window_weights)
            else:
                result[i] = np.nan
        return result

    @staticmethod
    @numba.njit  # pragma: no cover
    def _numba_func(xs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Calculate the weighted mean of the input array.

        Args:
            xs: Input array.
            weights: Array of weights to apply.

        Returns:
            np.ndarray: Weighted mean value.
        """
        return np.sum(xs * weights) / np.sum(weights)


class WeightedMovingAverage:
    """Factory class for creating weighted moving average feature generators.

    Creates a WeightedMean feature generator with a rolling window of specified size.
    """

    def __new__(
        cls,
        *,
        columns: str | list[str],
        window_size: int,
        weights: np.ndarray | list[float],
        out_column_names: str | list[str] | None = None,
        only_full_window: bool = False,
    ) -> Mean | WeightedMean:
        """Create a weighted moving average feature generator.

        Args:
            columns: Columns to calculate weighted moving average for.
            window_size: Size of the rolling window.
            weights: Weights to apply to window values.
            out_column_names: Names for output columns.
            only_full_window: Whether to calculate only for full windows.

        Returns:
            WeightedMean: A WeightedMean feature generator configured for weighted moving average.

        Raises:
            ValueError: If weights length doesn't match window_size or if weights is not iterable.
        """
        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float32)

        if not isinstance(weights, Iterable):
            return ValueError

        if len(weights) != window_size:
            msg = f"Length of weights must match window_size. Got {len(weights)}, expected {window_size}"
            raise ValueError(msg)

        return WeightedMean(
            columns=columns,
            window_types=WindowType.ROLLING(
                size=window_size,
                only_full_window=only_full_window,
            ),
            weights=weights,
            out_column_names=out_column_names,
            func_name="weighted_moving_average",
        )
