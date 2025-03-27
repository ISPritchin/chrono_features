import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class Median(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Median feature generator for time series data.

    Calculates the median value of data points within specified windows.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "median",
    ) -> None:
        """Initialize the median feature generator.

        Args:
            columns: Columns to calculate median for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.
            func_name: Name of the function for output column naming.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the median value of the input array.

        Args:
            xs: Input array.

        Returns:
            np.ndarray: Median value.
        """
        return np.median(xs)
