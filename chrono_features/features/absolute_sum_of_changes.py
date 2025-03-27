import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class AbsoluteSumOfChangesWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Absolute sum of changes feature generator for time series data.

    Calculates the sum of absolute differences between consecutive values within
    specified windows. This feature measures the total amount of change or volatility
    in a time series segment.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_type: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "absolute_sum_of_changes",
    ) -> None:
        """Initialize the absolute sum of changes feature generator.

        Args:
            columns: Columns to calculate absolute sum of changes for.
            window_type: Types of windows to use.
            out_column_names: Names for output columns.
            func_name: Name of the function for output column naming.
        """
        super().__init__(
            columns=columns,
            window_types=window_type,
            out_column_names=out_column_names,
            func_name=func_name,
        )

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the absolute sum of changes of the input array.

        Args:
            xs (np.ndarray): The input window.

        Returns:
            np.ndarray: The absolute sum of changes or NaN if the window has 1 or fewer elements.
        """
        if len(xs) <= 1:
            return np.nan

        return np.sum(np.abs(xs[1:] - xs[:-1]))


class AbsoluteSumOfChanges:
    """Factory class for creating absolute sum of changes feature generators.

    Provides a unified interface to create absolute sum of changes implementations.

    Examples:
        >>> from chrono_features.features import AbsoluteSumOfChanges
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
        >>> # Create absolute sum of changes with rolling window of size 3
        >>> asc = AbsoluteSumOfChanges(
        ...     columns='value',
        ...     window_types=WindowType.ROLLING(size=3),
        ...     out_column_names='value_asc'
        ... )
        >>>
        >>> # Apply to dataset
        >>> transformed_dataset = asc.transform(dataset)
    """

    def __new__(
        cls,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> AbsoluteSumOfChangesWithoutOptimization:
        """Create an absolute sum of changes feature generator.

        Args:
            columns: Columns to calculate absolute sum of changes for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.

        Returns:
            AbsoluteSumOfChangesWithoutOptimization: An absolute sum of changes feature generator.
        """
        return AbsoluteSumOfChangesWithoutOptimization(
            columns=columns,
            window_type=window_types,
            out_column_names=out_column_names,
        )
