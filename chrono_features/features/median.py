import numba
import numpy as np

from chrono_features.features._base import _FromNumbaFunc
from chrono_features.window_type import WindowType


class Median(_FromNumbaFunc):
    def __init__(
        self,
        columns: list[str] | str,
        window_type: WindowType,
        out_column_names: list[str] | str | None = None,
    ):
        super().__init__(columns, window_type, out_column_names, func_name="median")

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        return np.median(xs)
