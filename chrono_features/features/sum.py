import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTS
from chrono_features.window_type import WindowType


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
def process_dynamic(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    print(feature)
    print(lens)

    prefix_sum_array = np.empty(len(feature) + 1, dtype=np.float64)
    prefix_sum_array[0] = 0
    for i in range(len(feature)):
        prefix_sum_array[i + 1] = prefix_sum_array[i] + feature[i]

    result = np.empty(len(feature), dtype=np.float64)
    for i in range(len(result)):
        end = i + 1
        start = end - lens[i]
        if lens[i] == 0:
            result[i] = np.nan
        else:
            result[i] = prefix_sum_array[end] - prefix_sum_array[start]

    return result


@numba.njit
def process_rolling(
    feature: np.ndarray,
    lens: np.ndarray,
) -> np.ndarray:
    """
    Optimized processing for rolling windows.
    If not implemented in a subclass, falls back to process_dynamic.
    """
    return process_dynamic(feature, lens)


class Sum(_FromNumbaFuncWithoutCalculatedForEachTS):
    def __init__(
        self,
        columns: list[str] | str,
        window_type: WindowType,
        out_column_names: list[str] | str | None = None,
    ):
        super().__init__(columns, window_type, out_column_names, func_name="sum")

    @staticmethod
    @numba.njit
    def process_all_ts(
        feature: np.ndarray,
        ts_lens: np.ndarray,
        lens: np.ndarray,
        window_type: int,  # int representation of WindowTypeEnum
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
        res = np.empty(len(feature), dtype=np.float64)
        end = 0
        for ts_len in ts_lens:
            start = end
            end += ts_len
            window_data = feature[start:end]
            window_lens = lens[start:end]

            # Выбор метода в зависимости от типа окна
            if window_type == 1:
                res[start:end] = process_rolling(
                    feature=window_data,
                    lens=window_lens,
                )
            elif window_type == 0:
                res[start:end] = process_expanding(
                    feature=window_data,
                    lens=window_lens,
                )
            else:
                # Для dynamic и других типов окон используем универсальный метод
                res[start:end] = process_dynamic(
                    feature=window_data,
                    lens=window_lens,
                )
        return res
