import numba
import numpy as np

from chrono_features.features._base import (
    _FromNumbaFuncWithoutCalculatedForEachTSPoint,
)
from chrono_features.window_type import WindowType


class AbsoluteSumOfChangesWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    def __init__(
        self,
        columns: list[str] | str,
        window_type: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
        func_name="absolute_sum_of_changes",
    ):
        super().__init__(
            columns=columns,
            window_types=window_type,
            out_column_names=out_column_names,
            func_name=func_name,
        )

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        if len(xs) <= 1:
            return np.nan

        return np.sum(np.abs((xs[1:] - xs[:-1])))


class AbsoluteSumOfChanges:
    def __new__(
        cls,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> AbsoluteSumOfChangesWithoutOptimization:
        return AbsoluteSumOfChangesWithoutOptimization(
            columns=columns, window_type=window_types, out_column_names=out_column_names
        )
