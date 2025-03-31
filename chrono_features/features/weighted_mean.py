import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


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
            numba_kwargs={"weights": weights},
        )

        if weights is None:
            msg = "Weights cannot be None"
            raise ValueError(msg)

    @staticmethod
    @numba.njit  # pragma: no cover
    def apply_func_to_full_window(
        feature: np.ndarray,
        func: callable,
        lens: np.ndarray,
        weights: np.ndarray,
        n_out_features: int = 1,
    ) -> np.ndarray:
        """Apply a weighted function to sliding or expanding windows of a feature array.

        Args:
            feature: Input feature array.
            func: Numba-compiled function to apply.
            lens: Array of window lengths for each point.
            weights: Array of weights to apply to window values.
            n_out_features: Number of output features. Defaults to 1.

        Returns:
            np.ndarray: Result of applying the weighted function to each window.
        """
        result = np.empty((len(feature), n_out_features), dtype=np.float32)
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
