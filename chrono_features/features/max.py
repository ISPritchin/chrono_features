import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTS,
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
)
from chrono_features.window_type import WindowBase, WindowType


@numba.njit
def process_expanding(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    result = np.empty(len(feature), dtype=np.float64)
    current_max = -np.inf
    for i in range(len(lens)):
        if lens[i] == 1:  # Начало нового клиента
            current_max = feature[i]
        else:
            if feature[i] > current_max or np.isnan(feature[i]):
                current_max = feature[i]
            current_max = max(current_max, feature[i])
        result[i] = current_max
    return result


@numba.njit
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
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
    return process_dynamic(feature, lens, ts_lens)


class MaxWithOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[str] | WindowType,
        out_column_names: list[str] | str | None = None,
    ):
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
        if isinstance(window_type, WindowType.EXPANDING):
            return process_expanding(feature=feature, lens=lens)
        if isinstance(window_type, WindowType.ROLLING):
            return process_rolling(feature=feature, lens=lens, ts_lens=ts_lens)
        if isinstance(window_type, WindowType.DYNAMIC):
            return process_dynamic(feature=feature, lens=lens, ts_lens=ts_lens)
        raise ValueError


class MaxWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name="max",
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
        return np.max(xs)


class Max:
    def __new__(
        cls,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        use_optimization: bool = False,
    ) -> MaxWithOptimization | MaxWithoutOptimization:
        if use_optimization:
            return MaxWithOptimization(
                columns=columns,
                window_types=window_types,
                out_column_names=out_column_names,
            )
        else:
            return MaxWithoutOptimization(
                columns=columns,
                window_types=window_types,
                out_column_names=out_column_names,
            )
