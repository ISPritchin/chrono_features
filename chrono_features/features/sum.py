import numba
import numpy as np
import polars as pl

from chrono_features.features._base import (
    FeatureGenerator,
    calculate_length_for_each_time_series,
    calculate_window_lengths,
)
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType, WindowTypeEnum


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


class Sum(FeatureGenerator):
    def __init__(
        self,
        columns: list[str] | str,
        window_type: WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "sum",
    ):
        """
        Initializes the feature generator.

        Args:
            columns (Union[List[str], str]): The columns to apply the function to.
            window_type (WindowType): The type of window (expanding, rolling, etc.).
            out_column_names (Union[List[str], str, None]): The names of the output columns.
            func_name (str): The name of the function (used to generate output column names if not provided).

        Raises:
            ValueError: If an unsupported window type is provided.
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

        # Map WindowType to WindowTypeEnum
        if isinstance(window_type, WindowType.ROLLING):
            self.window_type_enum = WindowTypeEnum.ROLLING
            self.window_size = window_type.size
            self.only_full_window = window_type.only_full_window
        elif isinstance(window_type, WindowType.EXPANDING):
            self.window_type_enum = WindowTypeEnum.EXPANDING
        elif isinstance(window_type, WindowType.DYNAMIC):
            self.window_type_enum = WindowTypeEnum.DYNAMIC
        else:
            raise ValueError(f"Unsupported window type: {window_type}")

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

            # Apply the function to the feature array
            feature_array = dataset.data[column].to_numpy()
            ts_lens = calculate_length_for_each_time_series(dataset._get_numeric_id_column_values())
            print(ts_lens)

            result_array = self.process_all_ts(
                feature=feature_array,
                ts_lens=ts_lens,
                lens=lens,
                window_type=self.window_type_enum.value,  # Pass int representation
            )

            # Add the result as a new column to the dataset
            dataset.data = dataset.data.with_columns(pl.Series(result_array).alias(out_column_name))

        return dataset
