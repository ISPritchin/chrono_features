from copy import deepcopy
from typing import Self

import numba
import numpy as np
import polars as pl


class TSDataset:
    """A class to handle time series datasets.

    This class provides functionality to manage and manipulate time series data,
    including sorting, adding features, and checking monotonicity of blocks.

    Attributes:
        id_column_name (str): The name of the column containing identifiers.
        ts_column_name (str): The name of the column containing timestamps.
        data (pl.DataFrame): The underlying DataFrame containing the time series data.
    """

    def __init__(self, data: pl.DataFrame, *, ts_column_name: str, id_column_name: str) -> None:
        """Initializes the TSDataset with the given DataFrame and column names.

        Args:
            data (pl.DataFrame): The DataFrame containing the time series data.
            ts_column_name (str): The name of the column containing timestamps.
            id_column_name (str): The name of the column containing identifiers.
        """
        self.id_column_name = id_column_name
        self.ts_column_name = ts_column_name
        self.data = data
        self._sort()

    def head(self, n: int | None = None) -> pl.DataFrame:
        """Returns the first n rows of the dataset.

        Args:
            n (int): The number of rows to return.

        Returns:
            pl.DataFrame: The first n rows of the dataset.
        """
        return self.data.head(n=n)

    def tail(self, n: int | None = None) -> pl.DataFrame:
        """Returns the last n rows of the dataset.

        Args:
            n (int): The number of rows to return.

        Returns:
            pl.DataFrame: The last n rows of the dataset.
        """
        return self.data.tail(n=n)

    @staticmethod
    @numba.njit  # pragma: no cover
    def is_monotonic_blocks(values: np.ndarray) -> bool:
        """Checks if the given array is monotonic with blocks of equal values.

        Args:
            values (np.array): The array to check.

        Returns:
            bool: True if the array is monotonic with blocks, False otherwise.

        Examples:
            >>> TSDataset.is_monotonic_blocks(np.array([1, 1, 2, 2, 3, 3]))
            True
            >>> TSDataset.is_monotonic_blocks(np.array([3, 2, 1]))
            True
            >>> TSDataset.is_monotonic_blocks(np.array([1, 2, 2, 1]))
            False
            >>> TSDataset.is_monotonic_blocks(np.array([2, 2, 1, 1]))
            True
        """
        n_unique = len(np.unique(values))
        n_blocks = 1
        for i in numba.prange(1, len(values)):
            n_blocks += values[i] != values[i - 1]
        return n_unique == n_blocks

    def _sort(self) -> None:
        """Sorts the dataset by the identifier and timestamp columns.

        If the identifier column is not numeric, it is hashed before sorting.
        """
        if not self.data[self.id_column_name].dtype.is_numeric():
            id_values = self.data[self.id_column_name].hash()
        else:
            id_values = self.data[self.id_column_name]

        is_sorted_id_column = TSDataset.is_monotonic_blocks(
            id_values.to_numpy(),
        )

        is_sorted_ts_columns = self.data.filter(
            pl.col(self.id_column_name) == pl.col(self.id_column_name).shift(1),
            pl.col(self.ts_column_name) < pl.col(self.ts_column_name).shift(1),
        ).is_empty()  # If no such rows exist, the order is maintained

        if not is_sorted_id_column or not is_sorted_ts_columns:
            self.data = self.data.sort([self.id_column_name, self.ts_column_name])

    def add_feature(self, name: str, values: np.ndarray) -> None:
        """Adds a new feature (column) to the dataset.

        Args:
            name (str): The name of the new feature.
            values: The values for the new feature.

        Raises:
            ValueError: If a column with the given name already exists or
                if the length of values does not match the dataset length.
        """
        if len(values) != len(self.data):
            msg = "Length of values does not match the dataset length."
            raise ValueError(msg)

        if name in self.data.columns:
            msg = f"Column '{name}' already exists in the dataset"
            raise ValueError(msg)

        self.data = self.data.with_columns(pl.Series(values).alias(name))

    def get_numeric_id_column_values(self) -> np.ndarray:
        """Return the values of the identifier column.

        If the identifier column is not numeric, it is hashed before returning.

        Returns:
            np.ndarray: The values of the identifier column.
        """
        if not self.data.schema[self.id_column_name].is_numeric():
            return self.data[self.id_column_name].hash().to_numpy()

        return self.data[self.id_column_name].to_numpy()

    def clone(self) -> Self:
        """Returns a deep copy of the dataset."""
        return deepcopy(self)

    def write_csv(self, path: str) -> None:
        """Saves the dataset to a CSV file.

        Args:
            path (str): Path to save the CSV file
        """
        self.data.write_csv(path)
