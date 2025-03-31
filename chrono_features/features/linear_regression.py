"""Linear regression feature generator."""

import numpy as np
import numba

from chrono_features.features._base import _FromNumbaFuncWithoutCalculatedForEachTSPoint
from chrono_features import WindowType


class LinearRegressionWithoutOptimization(_FromNumbaFuncWithoutCalculatedForEachTSPoint):
    """Linear regression feature generator without optimization.

    This class calculates slope, intercept, and r² of a linear regression line
    for values within a specified window. These statistics help identify and evaluate
    trends in time series data.
    """

    def __init__(
        self,
        *,
        columns: list[str] | str,
        window_types: list[WindowType] | WindowType,
        out_column_names: list[str] | str | None = None,
    ) -> None:
        """Initialize the linear regression feature generator.

        Args:
            columns: Columns to calculate the linear regression coefficients for.
            window_types: Types of windows to use.
            coefficients: Which coefficients to extract. Can be "slope", "intercept",
                         "r_squared", or any combination. Default is all coefficients.
            out_column_names: Names for output columns. If provided, should match the number
                              of selected coefficients per input column.
        """
        super().__init__(
            columns=columns,
            window_types=window_types,
            out_column_names=out_column_names,
            func_name="linear_regression",
        )

        self.out_column_names = [
            f"{column}_linear_regression_{func}_{window_type.suffix}"
            for column in self.columns
            for window_type in self.window_types
            for func in ["slope", "intercept", "r_squared"]
        ]

        self.coefficients = ["slope", "intercept", "r_squared"]
        self.n_out_features = 3

    @staticmethod
    @numba.njit
    def _numba_func(xs: np.ndarray) -> np.ndarray:
        """Calculate the linear regression statistics for the input window.

        Args:
            xs: The input window.

        Returns:
            np.ndarray: Array with slope, intercept, and r² or NaNs if the window is too small.
        """
        n = len(xs)
        n_min_points_for_meaningful_regression = 3
        if n < n_min_points_for_meaningful_regression:  # Need at least 3 points for meaningful regression
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        # Create x values (indices)
        x = np.arange(n, dtype=np.float32)

        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(xs)

        # Calculate slope: sum((x_i - x_mean) * (y_i - y_mean)) / sum((x_i - x_mean)^2)
        numerator = np.float32(0.0)
        denominator = np.float32(0.0)

        for i in range(n):
            x_diff = x[i] - x_mean
            y_diff = xs[i] - y_mean
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff

        if denominator == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        # Calculate slope
        slope = numerator / denominator

        # Calculate intercept: y_mean - slope * x_mean
        intercept = y_mean - slope * x_mean

        # Calculate r²
        ss_total = np.float32(0.0)
        ss_residual = np.float32(0.0)

        for i in range(n):
            y_pred = intercept + slope * x[i]
            ss_total += (xs[i] - y_mean) ** 2
            ss_residual += (xs[i] - y_pred) ** 2

        if ss_total == 0:
            return np.array([slope, intercept, np.nan], dtype=np.float32)

        r_squared = np.float32(1.0) - (ss_residual / ss_total)

        # Return only slope, intercept, and r²
        return np.array([slope, intercept, r_squared], dtype=np.float32)
