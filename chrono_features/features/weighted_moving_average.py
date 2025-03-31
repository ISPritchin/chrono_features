# ruff: noqa: EM101, TRY003

from collections.abc import Iterable

import numpy as np

from chrono_features.features._base import AbstractGenerator
from chrono_features.features.weighted_mean import WeightedMean
from chrono_features.ts_dataset import TSDataset
from chrono_features.window_type import WindowType


class WeightedMovingAverage(AbstractGenerator):
    """Weighted moving average feature generator for time series data.

    Calculates the weighted moving average of values within a rolling window.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_size: int,
        weights: np.ndarray | list[float],
        out_column_names: list[str] | str | None = None,
        only_full_window: bool = False,
    ) -> None:
        """Initialize the weighted moving average feature generator.

        Args:
            columns: Columns to calculate weighted moving average for.
            window_size: Size of the rolling window.
            weights: Weights to apply to window values.
            out_column_names: Names for output columns.
            only_full_window: Whether to calculate only for full windows.

        Raises:
            ValueError: If weights length doesn't match window_size or if weights is not iterable.
        """
        super().__init__(columns=columns, out_column_names=out_column_names)
        self.window_size = window_size
        self.only_full_window = only_full_window

        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float32)

        if not isinstance(weights, Iterable):
            raise TypeError("Weights must be iterable")

        if len(weights) != window_size:
            msg = f"Length of weights must match window_size. Got {len(weights)}, expected {window_size}"
            raise ValueError(msg)

        self.weights = weights

        if out_column_names is None:
            if isinstance(columns, str):
                wma_out_column_names = f"{columns}_weighted_moving_average_{window_size}"
            else:
                wma_out_column_names = [f"{col}_weighted_moving_average_{window_size}" for col in columns]
        else:
            wma_out_column_names = out_column_names

        self.weighted_mean_transformer = WeightedMean(
            columns=columns,
            window_types=WindowType.ROLLING(size=window_size, only_full_window=only_full_window),
            weights=weights,
            out_column_names=wma_out_column_names,
        )

    def transform(self, dataset: TSDataset) -> TSDataset:
        """Apply the weighted moving average transformation to the dataset.

        Args:
            dataset: Dataset to transform.

        Returns:
            TSDataset: Transformed dataset.
        """
        return self.weighted_mean_transformer.transform(dataset)
