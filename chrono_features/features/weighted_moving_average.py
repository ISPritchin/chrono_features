from collections.abc import Iterable

import numpy as np

from chrono_features.features.weighted_mean import WeightedMean
from chrono_features.window_type import WindowType


class WeightedMovingAverage:
    """Factory class for creating weighted moving average feature generators.

    Creates a WeightedMean feature generator with a rolling window of specified size.
    """

    def __new__(
        cls,
        *,
        columns: str | list[str],
        window_size: int,
        weights: np.ndarray | list[float],
        out_column_names: str | list[str] | None = None,
        only_full_window: bool = False,
    ) -> WeightedMean:
        """Create a weighted moving average feature generator.

        Args:
            columns: Columns to calculate weighted moving average for.
            window_size: Size of the rolling window.
            weights: Weights to apply to window values.
            out_column_names: Names for output columns.
            only_full_window: Whether to calculate only for full windows.

        Returns:
            WeightedMean: A WeightedMean feature generator configured for weighted moving average.

        Raises:
            ValueError: If weights length doesn't match window_size or if weights is not iterable.
        """
        if isinstance(weights, list):
            weights = np.array(weights, dtype=np.float32)

        if not isinstance(weights, Iterable):
            return ValueError

        if len(weights) != window_size:
            msg = f"Length of weights must match window_size. Got {len(weights)}, expected {window_size}"
            raise ValueError(msg)

        return WeightedMean(
            columns=columns,
            window_types=WindowType.ROLLING(
                size=window_size,
                only_full_window=only_full_window,
            ),
            weights=weights,
            out_column_names=out_column_names,
            func_name="weighted_moving_average",
        )
