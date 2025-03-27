import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class Autocorrelation(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Autocorrelation feature generator for time series data.

    Calculates the autocorrelation of values within specified windows at a given lag.
    Autocorrelation measures the correlation between a time series and a lagged version
    of itself.

    Examples:
        >>> from chrono_features.features import Autocorrelation
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
        >>> # Create autocorrelation with lag=1 and rolling window of size 4
        >>> autocorr = Autocorrelation(
        ...     columns='value',
        ...     window_types=WindowType.ROLLING(size=4),
        ...     lag=1,
        ...     out_column_names='value_autocorr_lag1'
        ... )
        >>>
        >>> # Apply to dataset
        >>> transformed_dataset = autocorr.transform(dataset)
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        lag: int,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the autocorrelation feature generator.

        Args:
            columns: Columns to calculate autocorrelation for.
            window_types: Types of windows to use.
            lag: The lag value for autocorrelation calculation.
            out_column_names: Names for output columns.

        Raises:
            ValueError: If lag is less than or equal to 0.
        """
        func_name = f"autocorrelation_lag_{lag}"
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )
        if lag <= 0:
            msg = "Lag must be greater than 0"
            raise ValueError(msg)

        self.numba_kwargs = {"lag": lag}

    @staticmethod
    @numba.njit
    def apply_func_to_full_window(
        feature: np.ndarray,
        func: callable,
        lens: np.ndarray,
        lag: int,
    ) -> np.ndarray:
        """Applies a function to sliding or expanding windows of a feature array.

        Args:
            feature (np.ndarray): The input feature array.
            func (Callable): The Numba-compiled function to apply.
            lens (np.ndarray): Array of window lengths for each point.
            lag (int): The lag value for autocorrelation calculation.

        Returns:
            np.ndarray: The result of applying the function to each window.
        """
        result = np.empty(len(feature), dtype=np.float32)
        for i in numba.prange(len(result)):
            if lens[i]:
                result[i] = func(feature[i + 1 - lens[i] : i + 1], lag)
            else:
                result[i] = np.nan
        return result

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray, lag: int) -> np.ndarray:
        """Calculate the autocorrelation of the input array at the specified lag.

        Args:
            xs (np.ndarray): The input window.
            lag (int): The lag value for autocorrelation calculation.

        Returns:
            np.ndarray: The autocorrelation value or NaN if the window is too small.
        """
        if len(xs) <= lag + 1:
            return np.nan

        return autocorrelation(xs, lag=lag)


@numba.njit
def corr(x: np.array, y: np.array) -> float:
    """Calculate the Pearson correlation coefficient between two arrays.

    This is a Numba-optimized implementation that avoids using numpy's built-in
    correlation functions for better performance.

    Args:
        x (np.array): First input array.
        y (np.array): Second input array.

    Returns:
        float: Pearson correlation coefficient between x and y.
               Returns 0 if either array has zero standard deviation.
    """
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    covariance = 0
    variance_x = 0
    variance_y = 0
    for i in range(n):
        covariance += (x[i] - mean_x) * (y[i] - mean_y)
        variance_x += (x[i] - mean_x) ** 2
        variance_y += (y[i] - mean_y) ** 2

    std_dev_x = np.sqrt(variance_x / n)
    std_dev_y = np.sqrt(variance_y / n)
    if std_dev_x == 0 or std_dev_y == 0:
        return 0

    return covariance / (n * std_dev_x * std_dev_y)


@numba.njit
def autocorrelation(x: np.array, lag: int = 12) -> float:
    """Calculate the autocorrelation of an array at the specified lag.

    Autocorrelation measures the correlation between a time series and a lagged version
    of itself.

    Args:
        x (np.array): Input time series array.
        lag (int, optional): The lag value. Defaults to 12.

    Returns:
        float: Autocorrelation value at the specified lag.
    """
    return corr(x[lag:], x[:-lag])
