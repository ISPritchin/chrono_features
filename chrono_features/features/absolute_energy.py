import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
)
from chrono_features.window_type import WindowType


class AbsoluteEnergyWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    def __init__(
        self,
        columns: list[str] | str,
        window_type: WindowType,
        out_column_names: list[str] | str | None = None,
    ):
        super().__init__(columns, window_type, out_column_names, func_name="absolute_energy")

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        return (xs * xs).sum()


class AbsoluteEnergy:
    def __new__(
        cls,
        columns: list[str] | str,
        window_types: WindowType,
        out_column_names: list[str] | str | None = None,
    ):
        return AbsoluteEnergyWithoutOptimization(columns, window_types, out_column_names)
