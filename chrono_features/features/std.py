import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class StdWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Standard deviation feature generator for time series data.

    Calculates the standard deviation of values within specified windows.
    """

    def __init__(
        self,
        columns: list[str] | str,
        window_type: WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "std",
    ) -> None:
        """Initialize the standard deviation feature generator.

        Args:
            columns: Columns to calculate standard deviation for.
            window_type: Type of window to use.
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
        """Calculate the standard deviation of the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray: Standard deviation value or NaN if array has 1 or fewer elements.
        """
        if len(xs) <= 1:
            return np.nan

        return np.std(xs)


class Std:
    """Factory class for creating standard deviation feature generators.

    Provides a unified interface to create standard deviation implementations.
    """

    def __new__(
        cls,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> StdWithoutOptimization:
        """Create a standard deviation feature generator.

        Args:
            columns: Columns to calculate standard deviation for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.

        Returns:
            StdWithoutOptimization: A standard deviation feature generator.
        """
        return StdWithoutOptimization(columns=columns, window_type=window_types, out_column_names=out_column_names)
