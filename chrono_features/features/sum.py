import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTS,
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
)
from chrono_features.window_type import WindowType, WindowBase


@numba.njit
def process_expanding(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    result = np.empty(len(feature), dtype=np.float64)
    for i in range(len(lens)):
        current_len = lens[i]
        if current_len == 1:
            cumulative_sum = feature[i]
        else:
            cumulative_sum += feature[i]
        result[i] = cumulative_sum

    return result


@numba.njit
def process_dynamic(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    result = np.empty(len(feature), dtype=np.float64)

    buffer_size = ts_lens.max() + 1
    prefix_sum_array = np.empty(buffer_size, dtype=np.float64)

    end = 0
    for i in range(len(ts_lens)):
        current_len = ts_lens[i]
        start = end
        end = start + current_len

        prefix_sum_array[0] = 0
        for j in range(start, end):
            prefix_sum_array[j - start + 1] = prefix_sum_array[j - start] + feature[j]

        for j in range(start, end):
            v = j - start + 1
            start_window = v - lens[j]
            if lens[j] == 0:
                result[j] = np.nan
            else:
                result[j] = prefix_sum_array[v] - prefix_sum_array[start_window]

    return result


@numba.njit
def process_rolling(feature: np.ndarray, lens: np.ndarray, ts_lens: np.ndarray) -> np.ndarray:
    """
    Optimized processing for rolling windows.
    If not implemented in a subclass, falls back to process_dynamic.
    """
    return process_dynamic(feature, lens, ts_lens)


class SumWithPrefixSumOptimization(_FromNumbaFuncWithoutCalculatedForEachTS):
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
            func_name="sum",
        )

    @staticmethod
    def process_all_ts(
        feature: np.ndarray,
        ts_lens: np.ndarray,
        lens: np.ndarray,
        window_type: WindowBase,  # int representation of WindowTypeEnum
    ) -> np.ndarray:
        """
        Process all time series using the appropriate method based on window type.

        Args:
            feature (np.ndarray): The feature array.
            ts_lens (np.ndarray): The lengths of each time series.
            lens (np.ndarray): The window lengths for each point.
            window_type (int): The type of window (int representation of WindowTypeEnum).
            window_size (int): The size of the rolling window (if applicable).
            only_full_window (bool): Whether to use only full windows (if applicable).

        Returns:
            np.ndarray: The result array.
        """
        # Выбор метода в зависимости от типа окна
        if isinstance(window_type, WindowType.EXPANDING):
            return process_expanding(
                feature=feature,
                lens=lens,
            )
        if isinstance(window_type, WindowType.ROLLING):
            return process_rolling(feature=feature, lens=lens, ts_lens=ts_lens)
        if isinstance(window_type, WindowType.DYNAMIC):
            # Для dynamic и других типов окон используем универсальный метод
            return process_dynamic(feature=feature, lens=lens, ts_lens=ts_lens)

        raise ValueError


class SumWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name="sum",
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
        return np.sum(xs)

    def transform_for_window_type(self, dataset, column, window_type):
        if not isinstance(window_type, WindowType.EXPANDING):
            return super().transform_for_window_type(dataset=dataset, column=column, window_type=window_type)
        else:
            return SumWithPrefixSumOptimization(
                columns=column, window_types=window_type, out_column_names=None
            ).transform_for_window_type(dataset=dataset, column=column, window_type=window_type)


class Sum:
    def __new__(
        cls,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        use_prefix_sum_optimization: bool = False,
    ) -> SumWithPrefixSumOptimization | SumWithoutOptimization:
        if use_prefix_sum_optimization:
            return SumWithPrefixSumOptimization(
                columns=columns,
                window_types=window_types,
                out_column_names=out_column_names,
            )
        else:
            return SumWithoutOptimization(
                columns=columns,
                window_types=window_types,
                out_column_names=out_column_names,
            )
