from abc import ABC, abstractmethod
from typing import Callable

import numba
import numpy as np
import polars as pl

from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType, WindowBase


@numba.jit(nopython=True)
def _calculate_expanding_window_length(ids: np.ndarray) -> np.ndarray:
    lens = np.empty(len(ids), dtype=np.int32)
    lens[0] = 1
    for i in range(1, len(lens)):
        if ids[i] == ids[i - 1]:
            lens[i] = lens[i - 1] + 1
        else:
            lens[i] = 1
    return lens


@numba.jit(nopython=True)
def _calculate_rolling_window_length(ids: np.ndarray, window_size: int, only_full_window: bool) -> np.ndarray:
    lens = np.empty(len(ids), dtype=np.int32)
    lens[0] = 1
    for i in range(1, len(lens)):
        if ids[i] == ids[i - 1]:
            lens[i] = lens[i - 1] + 1
        else:
            lens[i] = 1

    if only_full_window:
        for i in range(len(lens)):
            if lens[i] >= window_size:
                lens[i] = window_size
            else:
                lens[i] = 0
    else:
        for i in range(len(lens)):
            if lens[i] > window_size:
                lens[i] = window_size

    return lens


def calculate_window_lengths(dataset: TSDataset, window_type: WindowBase) -> np.ndarray:
    """
    Calculate window lengths for each point in the dataset.

    Args:
        dataset (TSDataset): The input dataset.
        window_type (WindowBase): The type of window (expanding, rolling, etc.).
        id_column_name (str): The name of the column containing IDs.

    Returns:
        np.ndarray: Array of window lengths for each point.
    """
    if isinstance(window_type, WindowType.DYNAMIC):
        return dataset.data[window_type.len_column_name].to_numpy()

    # Get the IDs and convert them to a hash for consistent comparison
    ids = dataset._get_numeric_id_column_values()

    # Calculate window lengths based on the window type
    if isinstance(window_type, WindowType.EXPANDING):
        return _calculate_expanding_window_length(ids)
    elif isinstance(window_type, WindowType.ROLLING):
        return _calculate_rolling_window_length(ids, window_type.size, window_type.only_full_window)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")


@numba.njit
def calculate_length_for_each_time_series(ids: np.ndarray) -> np.ndarray:
    ts_lens = np.empty(len(ids), dtype=np.int32)
    current_id_index = 0
    current_len = 1
    for i in range(1, len(ids)):
        print(ts_lens)
        if ids[i] != ids[i - 1]:
            ts_lens[current_id_index] = current_len
            current_id_index += 1
            current_len = 1
        else:
            current_len += 1
    ts_lens[current_id_index] = current_len
    print(ts_lens)
    return ts_lens[: current_id_index + 1]


class FeatureGenerator(ABC):
    def __init__(self, columns: list[str] | str, window_type: WindowType, out_column_name=None):
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list) or not len(columns):
            raise ValueError

        if not isinstance(window_type, WindowBase):
            raise ValueError

        self.columns = columns
        self.window_type = window_type
        self.out_column_name = out_column_name

    @abstractmethod
    def transform(self, dataset: TSDataset) -> np.ndarray:
        raise NotImplementedError


class _FromNumbaFuncWithoutCalculatedForEachTSPoint(FeatureGenerator):
    """
    Base class for feature generators that use Numba-optimized functions.
    Applies a Numba-compiled function to sliding or expanding windows of data.
    """

    def __init__(
        self,
        columns: list[str] | str,
        window_type: WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = None,
    ):
        """
        Initializes the feature generator.

        Args:
            columns (Union[List[str], str]): The columns to apply the function to.
            window_type (WindowType): The type of window (expanding, rolling, etc.).
            out_column_names (Union[List[str], str, None]): The names of the output columns.
            func_name (str): The name of the function (used to generate output column names if not provided).
        """
        super().__init__(columns, window_type, out_column_names)

        # Convert columns to a list if a single string is provided
        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

        # Generate out_column_names if not provided
        if out_column_names is None:
            if func_name is None:
                raise ValueError("func_name must be provided if out_column_names is None")
            self.out_column_names = [f"{column}_{func_name}_{window_type.suffix}" for column in self.columns]
        elif isinstance(out_column_names, str):
            self.out_column_names = [out_column_names]
        else:
            self.out_column_names = out_column_names

    @staticmethod
    @numba.njit
    def apply_func_to_full_window(feature: np.ndarray, func: Callable, lens: np.ndarray) -> np.ndarray:
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
                result[i] = func(feature[i + 1 - lens[i] : i + 1])
            else:
                result[i] = np.nan
        return result

    @abstractmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """
        Abstract method defining the Numba-compiled function to apply to each window.

        Args:
            xs (np.ndarray): The input window.

        Returns:
            np.ndarray: The result of applying the function to the window.
        """
        raise NotImplementedError

    def transform(self, dataset: TSDataset) -> TSDataset:
        """
        Applies the Numba-compiled function to each column in the dataset.

        Args:
            dataset (TSDataset): The input dataset.

        Returns:
            TSDataset: The transformed dataset with new columns added.
        """
        if not self.columns:
            raise ValueError("No columns specified for transformation.")

        if len(self.columns) != len(self.out_column_names):
            raise ValueError("The number of columns and output column names must match.")

        for column, out_column_name in zip(self.columns, self.out_column_names):
            if column not in dataset.data.columns:
                raise ValueError(f"Column '{column}' not found in the dataset.")

            # Calculate window lengths
            lens = calculate_window_lengths(
                dataset=dataset,
                window_type=self.window_type,
            )
            print(lens)

            # Apply the function to the feature array
            feature_array = dataset.data[column].to_numpy()
            result_array = self.apply_func_to_full_window(feature_array, self._numba_func, lens)

            # Add the result as a new column to the dataset
            dataset.data = dataset.data.with_columns(pl.Series(result_array).alias(out_column_name))

        return dataset


@numba.njit
def process_dynamic(feature: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """
    Process one time series using the dynamic window approach.
    This method must be implemented by subclasses.
    """
    raise NotImplementedError
