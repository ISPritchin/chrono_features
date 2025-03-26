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
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name="absolute_energy",
    ):
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name=func_name,
        )

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        return (xs * xs).sum()


class AbsoluteEnergy:
    def __new__(
        cls,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> AbsoluteEnergyWithoutOptimization:
        return AbsoluteEnergyWithoutOptimization(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
        )
