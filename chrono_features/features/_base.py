from abc import ABC, abstractmethod
from itertools import product
from typing import Callable, NoReturn, Optional

import numba
import numpy as np
import polars as pl

from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowBase, WindowType


@numba.jit(nopython=True)  # pragma: no cover
def _calculate_expanding_window_length(ids: np.ndarray) -> np.ndarray:
    lens = np.empty(len(ids), dtype=np.int32)
    lens[0] = 1
    for i in range(1, len(lens)):
        if ids[i] == ids[i - 1]:
            lens[i] = lens[i - 1] + 1
        else:
            lens[i] = 1
    return lens


@numba.njit  # pragma: no cover
def _calculate_rolling_window_length(*, ids: np.ndarray, window_size: int, only_full_window: bool) -> np.ndarray:
    lens = np.zeros(len(ids), dtype=np.int32)
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
            lens[i] = min(lens[i], window_size)

    return lens


@numba.njit  # pragma: no cover
def _calculate_dynamic_window_length(ids: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Calculate window lengths for dynamic windows, ensuring they don't exceed available data.

    Args:
        ids: Array of time series IDs.
        lens: Array of requested window lengths.

    Returns:
        Array of adjusted window lengths.
    """
    result = np.zeros_like(lens, dtype=np.int32)
    current_len = 1

    for i in range(len(ids)):
        if i == 0 or ids[i] != ids[i - 1]:
            current_len = 1
        else:
            current_len += 1

        # Limit window length to available points in current time series
        result[i] = min(lens[i], current_len)

    return result


def calculate_window_lengths(dataset: TSDataset, window_type: WindowBase) -> np.ndarray:
    """Calculate window lengths for each point in the dataset.

    Args:
        dataset (TSDataset): The input dataset.
        window_type (WindowBase): The type of window (expanding, rolling, etc.).

    Returns:
        np.ndarray: Array of window lengths for each point.
    """
    if isinstance(window_type, WindowType.DYNAMIC):
        # Get the dynamic window lengths
        lens = dataset.data[window_type.len_column_name].to_numpy()

        # Get the IDs to determine time series boundaries
        ids = dataset.get_numeric_id_column_values()

        # Use the Numba-optimized function to calculate adjusted window lengths
        return _calculate_dynamic_window_length(ids, lens)

    # Get the IDs and convert them to a hash for consistent comparison
    ids = dataset.get_numeric_id_column_values()

    # Calculate window lengths based on the window type
    if isinstance(window_type, WindowType.EXPANDING):
        res = _calculate_expanding_window_length(ids)
    elif isinstance(window_type, WindowType.ROLLING):
        res = _calculate_rolling_window_length(
            ids=ids,
            window_size=window_type.size,
            only_full_window=window_type.only_full_window,
        )
    else:
        msg = f"Unsupported window type: {window_type}"
        raise TypeError(msg)

    return res


@numba.njit  # pragma: no cover
def calculate_length_for_each_time_series(ids: np.ndarray) -> np.ndarray:
    ts_lens = np.empty(len(ids), dtype=np.int32)
    current_id_index = 0
    current_len = 1
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            ts_lens[current_id_index] = current_len
            current_id_index += 1
            current_len = 1
        else:
            current_len += 1
    ts_lens[current_id_index] = current_len
    return ts_lens[: current_id_index + 1]


class AbstractGenerator(ABC):
    @abstractmethod
    def transform(self, dataset: TSDataset) -> TSDataset:
        raise NotImplementedError


class FeatureGenerator(AbstractGenerator):
    def __init__(
        self,
        columns: list[str] | str,
        window_types: WindowType,
        out_column_names: str | list[str] | None = None,
    ) -> None:
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list) or not len(columns):
            raise ValueError

        if isinstance(window_types, WindowBase):
            window_types = [window_types]
        if not isinstance(window_types, list) or not len(window_types):
            raise ValueError

        if isinstance(out_column_names, str):
            out_column_names = [out_column_names]
        if out_column_names is not None and len(columns) * len(window_types) != len(out_column_names):
            raise ValueError

        self.columns = columns
        self.window_types = window_types
        self.out_column_names = out_column_names

    def transform(self, dataset: TSDataset) -> np.ndarray:
        if not self.columns:
            msg = "No columns specified for transformation."
            raise ValueError(msg)

        if len(self.columns) * len(self.window_types) != len(self.out_column_names):
            msg = "The number of columns and output column names must match."
            raise ValueError(msg)

        dataset_copy = dataset.clone()
        for (column, window_type), out_column_name in zip(
            product(self.columns, self.window_types),
            self.out_column_names,
            strict=True,
        ):
            if column not in dataset.data.columns:
                msg = f"Column '{column}' not found in the dataset."
                raise ValueError(msg)

            result_array = self.transform_for_window_type(
                dataset=dataset,
                column=column,
                window_type=window_type,
            )

            # Add the result as a new column to the dataset
            dataset_copy.add_feature(values=pl.Series(result_array), name=out_column_name)

        return dataset_copy

    @abstractmethod
    def transform_for_window_type(
        self,
        dataset: TSDataset,
        column: str,
        window_type: WindowBase,
    ) -> NoReturn:
        raise NotImplementedError


class _FromNumbaFuncWithoutCalculatedForEachTSPoint(FeatureGenerator):
    """Base class for feature generators that use Numba-optimized functions.

    Applies a Numba-compiled function to sliding or expanding windows of data.
    """

    def __init__(
        self,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: Optional[str] = None,
    ) -> None:
        """Initializes the feature generator.

        Args:
            columns (Union[List[str], str]): The columns to apply the function to.
            window_types (Union[List[WindowType], WindowType]): The type of windows (expanding, rolling, etc.).
            out_column_names (Union[List[str], str, None]): The names of the output columns.
            func_name (str): The name of the function (used to generate output column names if not provided).
        """
        super().__init__(columns, window_types, out_column_names)

        # Generate out_column_names if not provided
        if self.out_column_names is None:
            if func_name is None:
                msg = "func_name must be provided if out_column_names is None"
                raise ValueError(msg)
            self.out_column_names = [
                f"{column}_{func_name}_{window_type.suffix}"
                for column in self.columns
                for window_type in self.window_types
            ]

        self.numba_kwargs = {}

    @staticmethod
    @numba.njit  # pragma: no cover
    def apply_func_to_full_window(
        feature: np.ndarray,
        func: Callable,
        lens: np.ndarray,
    ) -> np.ndarray:
        """Applies a function to sliding or expanding windows of a feature array.

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
    @numba.njit  # pragma: no cover
    def _numba_func(self: np.ndarray) -> np.ndarray:
        """Abstract method defining the Numba-compiled function to apply to each window.

        Args:
            xs (np.ndarray): The input window.

        Returns:
            np.ndarray: The result of applying the function to the window.
        """
        raise NotImplementedError

    def transform_for_window_type(
        self,
        dataset: TSDataset,
        column: str,
        window_type: WindowBase,
    ) -> np.ndarray:
        lens = calculate_window_lengths(
            dataset=dataset,
            window_type=window_type,
        )

        # Apply the function to the feature array
        feature_array = dataset.data[column].to_numpy()
        return self.apply_func_to_full_window(
            feature=feature_array,
            func=self._numba_func,
            lens=lens,
            **self.numba_kwargs,
        )


class _FromNumbaFuncWithoutCalculatedForEachTS(FeatureGenerator):
    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "sum",
    ) -> None:
        """Initializes the feature generator.

        Args:
            columns (Union[List[str], str]): The columns to apply the function to.
            window_types (Union[List[WindowType], WindowType]): The type of windows (expanding, rolling, etc.).
            out_column_names (Union[List[str], str, None]): The names of the output columns.
            func_name (str): The name of the function (used to generate output column names if not provided).

        Raises:
            ValueError: If an unsupported window type is provided.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )

        if out_column_names is None:
            if func_name is None:
                msg = "func_name must be provided if out_column_names is None"
                raise ValueError(msg)
            self.out_column_names = [
                f"{column}_{func_name}_{window_type.suffix}"
                for column in self.columns
                for window_type in self.window_types
            ]
        elif isinstance(out_column_names, str):
            self.out_column_names = [out_column_names]
        else:
            self.out_column_names = out_column_names

    def transform_for_window_type(self, dataset: TSDataset, column: str, window_type: WindowBase) -> np.ndarray:
        lens = calculate_window_lengths(
            dataset=dataset,
            window_type=window_type,
        )

        # Apply the function to the feature array
        feature_array = dataset.data[column].to_numpy()

        ts_lens = calculate_length_for_each_time_series(dataset.get_numeric_id_column_values())

        return self.process_all_ts(
            feature=feature_array,
            lens=lens,
            ts_lens=ts_lens,
            window_type=window_type,  # Pass int representation
        )

    @staticmethod
    @numba.njit  # pragma: no cover
    def process_all_ts(
        feature: np.ndarray,
        lens: np.ndarray,
        ts_lens: np.ndarray,
        window_type: WindowBase,
    ) -> np.array:
        raise NotImplementedError


class StrategySelector(AbstractGenerator, ABC):
    """Abstract base class for feature generators with dynamic strategy selection.

    This class provides a framework for implementing feature generators that can
    dynamically select between different implementation strategies based on window type
    and other parameters.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the strategy selector with common parameters.

        Args:
            columns: Columns to calculate features for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
        """
        if isinstance(columns, str):
            self.columns = [columns]
        else:
            self.columns = columns

        self.window_types = window_types if isinstance(window_types, list) else [window_types]

        if out_column_names is not None:
            if isinstance(out_column_names, str):
                self.out_column_names = [out_column_names]
            else:
                self.out_column_names = out_column_names
        else:
            self.out_column_names = None

        # Initialize metadata dictionary to store implementation information
        self.implementation_metadata = {}

    @abstractmethod
    def _select_implementation_type(self, window_type: WindowBase) -> type:
        """Select the appropriate implementation class based on window type.

        Args:
            window_type: The window type to process.

        Returns:
            The appropriate implementation class.
        """

    def transform(self, dataset: TSDataset) -> TSDataset:  # noqa: C901
        """Transform the dataset using the appropriate implementation for each window type.

        This implementation first groups window types by their optimal implementation,
        then calls each implementation only once with all relevant window types.

        Args:
            dataset: The dataset to transform.

        Returns:
            The transformed dataset.
        """
        result = dataset.clone()

        # Initialize metadata storage in the result dataset if not present
        if not hasattr(result, "_feature_metadata"):
            result._feature_metadata = {}  # noqa: SLF001

        # Group window types by implementation for each column
        for column in self.columns:
            # Dictionary to group window types by implementation class
            implementation_groups = {}

            # Group window types by implementation class
            for i, window_type in enumerate(self.window_types):
                impl_class = self._select_implementation_type(window_type)

                if impl_class not in implementation_groups:
                    implementation_groups[impl_class] = {
                        "window_types": [],
                        "window_indices": [],
                    }

                implementation_groups[impl_class]["window_types"].append(window_type)
                implementation_groups[impl_class]["window_indices"].append(i)

            # Process each implementation group
            for impl_class, group_data in implementation_groups.items():
                window_types = group_data["window_types"]
                window_indices = group_data["window_indices"]

                # Determine the appropriate out_column_names for this implementation
                local_out_column_names = None
                if self.out_column_names is not None:
                    # Calculate the indices for out_column_names
                    # For each column and window type combination
                    col_idx = self.columns.index(column)
                    out_indices = []
                    for window_idx in window_indices:
                        # Calculate the index in the flattened list of out_column_names
                        out_idx = col_idx * len(self.window_types) + window_idx
                        out_indices.append(out_idx)

                    # Extract the relevant out_column_names
                    local_out_column_names = [self.out_column_names[i] for i in out_indices]

                # Create the transformer instance with the grouped windows
                transformer = impl_class(
                    columns=column,
                    window_types=window_types,
                    out_column_names=local_out_column_names,
                )

                # Transform using the implementation
                transformed_result = transformer.transform(dataset)

                # Store metadata about which implementation was used for each column/window combination
                for i, window_type in enumerate(window_types):
                    if local_out_column_names:
                        out_col_name = local_out_column_names[i]

                        # Create a key for the metadata dictionary
                        metadata_key = (column, str(window_type))

                        # Store implementation information
                        implementation_info = {
                            "implementation_class": impl_class.__name__,
                            "column": column,
                            "window_type": str(window_type),
                        }

                        # Store in both the class instance and the result dataset
                        self.implementation_metadata[metadata_key] = implementation_info
                        result._feature_metadata[out_col_name] = implementation_info  # noqa: SLF001

                # Add all new columns from the transformed result to our result
                for col_name in transformed_result.data.columns:
                    # Skip the original columns
                    if col_name in dataset.data.columns:
                        continue
                    result.add_feature(col_name, transformed_result.data[col_name])

        return result
