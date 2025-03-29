from chrono_features.features.mean import Mean
from chrono_features.window_type import WindowType


class SimpleMovingAverage:
    """Factory class for creating simple moving average feature generators.

    Creates a Mean feature generator with a rolling window of specified size.
    """

    def __new__(
        cls,
        *,
        columns: str | list[str],
        window_size: int,
        out_column_names: str | list[str] | None = None,
        only_full_window: bool = False,
    ) -> Mean:
        """Create a simple moving average feature generator.

        Args:
            columns: Columns to calculate moving average for.
            window_size: Size of the rolling window.
            out_column_names: Names for output columns.
            only_full_window: Whether to calculate only for full windows.

        Returns:
            Mean: A Mean feature generator configured for simple moving average.

        Raises:
            ValueError: If window_size is less than or equal to 0.
        """
        if window_size <= 0:
            raise ValueError

        return Mean(
            columns=columns,
            window_types=WindowType.ROLLING(
                size=window_size,
                only_full_window=only_full_window,
            ),
            out_column_names=out_column_names,
            func_name="simple_moving_average",
        )
