from typing import Iterable
import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class Mean(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name="mean",
    ):
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        return np.mean(xs)


class SimpleMovingAverage:
    def __new__(
        self,
        columns: str | list[str],
        *,
        window_size: int,
        out_column_names: str | list[str] | None = None,
        only_full_window: bool = False,
    ) -> Mean:
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
    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        weights: np.ndarray | None = None,
        out_column_names: list[str] | str | None = None,
        func_name: str = "weighted_mean",
    ):
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )

        if weights is None:
            raise ValueError("Weights cannot be None")

        self.weights = weights
        self.numba_kwargs = {"weights": weights}

    @staticmethod
    @numba.njit
    def apply_func_to_full_window(
        feature: np.ndarray,
        func: callable,
        lens: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
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
    @numba.njit
    def _numba_func(xs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.sum(xs * weights) / np.sum(weights)


class WeightedMovingAverage:
    def __new__(
        cls,
        columns: str | list[str],
        *,
        window_size: int,
        weights: np.ndarray | list[float],
        out_column_names: str | list[str] | None = None,
        only_full_window: bool = False,
    ) -> Mean | WeightedMean:
        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float32)

        if not isinstance(weights, Iterable):
            return ValueError

        if len(weights) != window_size:
            raise ValueError(f"Length of weights must match window_size. Got {len(weights)}, expected {window_size}")

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
