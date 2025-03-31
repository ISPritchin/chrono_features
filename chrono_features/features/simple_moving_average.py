from chrono_features.features._base import AbstractGenerator
from chrono_features import TSDataset
from chrono_features.features.mean import Mean
from chrono_features.window_type import WindowType


class SimpleMovingAverage(AbstractGenerator):
    """Simple moving average feature generator for time series data.

    Calculates the simple moving average of values within a rolling window.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_size: int,
        only_full_window: bool = False,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the simple moving average feature generator.

        Args:
            columns: Columns to calculate simple moving average for.
            window_size: Size of the rolling window.
            only_full_window: Whether to calculate only for full windows.
            out_column_names: Names for output columns.
        """
        super().__init__(columns=columns, out_column_names=out_column_names)
        self.window_size = window_size
        self.only_full_window = only_full_window

        # Если имена выходных колонок не указаны, создаем их в формате "column_sma_window_size"
        if out_column_names is None:
            if isinstance(columns, str):
                sma_out_column_names = f"{columns}_simple_moving_average_{window_size}"
            else:
                sma_out_column_names = [f"{col}_simple_moving_average_{window_size}" for col in columns]
        else:
            sma_out_column_names = out_column_names

        self.mean_transformer = Mean(
            columns=columns,
            window_types=WindowType.ROLLING(size=window_size, only_full_window=only_full_window),
            out_column_names=sma_out_column_names,
        )

    def transform(self, dataset: TSDataset) -> TSDataset:
        """Apply the simple moving average transformation to the dataset.

        Args:
            dataset: Dataset to transform.

        Returns:
            TSDataset: Transformed dataset.
        """
        return self.mean_transformer.transform(dataset)
