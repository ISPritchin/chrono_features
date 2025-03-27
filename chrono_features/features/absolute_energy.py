import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features.window_type import WindowType


class AbsoluteEnergyWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Absolute energy feature generator for time series data.

    Calculates the sum of squared values within specified windows. This feature
    measures the overall energy or magnitude of a time series segment, giving
    higher weight to larger values.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name: str = "absolute_energy",
    ) -> None:
        """Initialize the absolute energy feature generator.

        Args:
            columns: Columns to calculate absolute energy for.
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
        """Calculate the absolute energy of the input array.

        Args:
            xs (np.ndarray): The input window.

        Returns:
            np.ndarray: The sum of squared values in the window.
        """
        return (xs * xs).sum()


class AbsoluteEnergy:
    """Factory class for creating absolute energy feature generators.

    Provides a unified interface to create absolute energy implementations.

    Examples:
        >>> from chrono_features.features import AbsoluteEnergy
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
        >>> # Create absolute energy with expanding window
        >>> energy = AbsoluteEnergy(
        ...     columns='value',
        ...     window_types=WindowType.EXPANDING(),
        ...     out_column_names='value_energy'
        ... )
        >>>
        >>> # Apply to dataset
        >>> transformed_dataset = energy.transform(dataset)
    """

    def __new__(
        cls,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> AbsoluteEnergyWithoutOptimization:
        """Create an absolute energy feature generator.

        Args:
            columns: Columns to calculate absolute energy for.
            window_types: Types of windows to use.
            out_column_names: Names for output columns.

        Returns:
            AbsoluteEnergyWithoutOptimization: An absolute energy feature generator.
        """
        return AbsoluteEnergyWithoutOptimization(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )
