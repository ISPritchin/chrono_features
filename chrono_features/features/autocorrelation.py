import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class Autocorrelation(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        lag: int,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        func_name = f"autocorrelation_lag_{lag}"
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )
        if lag <= 0:
            raise ValueError

        self.numba_kwargs = {"lag": lag}

    @staticmethod
    @numba.njit
    def apply_func_to_full_window(
        feature: np.ndarray,
        func: callable,
        lens: np.ndarray,
        lag: int,
    ) -> np.ndarray:
        """
        Applies a function to sliding or expanding windows of a feature array.

        Args:
            feature (np.ndarray): The input feature array.
            func (Callable): The Numba-compiled function to apply.
            lens (np.ndarray): Array of window lengths for each point.

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

    @numba.njit
    def _numba_func(xs: np.ndarray, lag: int) -> np.ndarray:
        """
        Abstract method defining the Numba-compiled function to apply to each window.

        Args:
            xs (np.ndarray): The input window.

        Returns:
            np.ndarray: The result of applying the function to the window.
        """
        if len(xs) <= lag + 1:
            return np.nan

        return autocorrelation(xs, lag=lag)


@numba.njit
def corr(x: np.array, y: np.array) -> float:
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    covariance = 0
    variance_x = 0
    variance_y = 0
    for i in range(n):
        covariance += (x[i] - mean_x) * (y[i] - mean_x)
        variance_x += (x[i] - mean_x) ** 2
        variance_y += (y[i] - mean_y) ** 2

    std_dev_x = np.sqrt(variance_x / n)
    std_dev_y = np.sqrt(variance_y / n)
    if std_dev_x == 0 or std_dev_y == 0:
        return 0

    return covariance / (n * std_dev_x * std_dev_y)


@numba.njit
def autocorrelation(x: np.array, lag: int = 12) -> float:
    return corr(x[lag:], x[:-lag])
