import numpy as np
import polars as pl
from scipy import stats

from chrono_features.features._base import WindowType
from chrono_features.features.linear_regression import LinearRegressionWithoutOptimization
from chrono_features.ts_dataset import TSDataset


def test_linear_regression_vs_scipy():
    """Test that our linear regression implementation matches scipy's results."""
    # Generate 10 time series of length 30
    np.random.seed(42)
    num_series = 10
    series_length = 30

    # Create data with different trends for each series
    data = {"id": [], "timestamp": [], "value": []}

    for series_id in range(num_series):
        # Generate a time series with a linear trend plus some noise
        slope = np.random.uniform(-2, 2)  # Random slope
        intercept = np.random.uniform(-10, 10)  # Random intercept
        noise_level = np.random.uniform(0.1, 2.0)  # Random noise level

        for t in range(series_length):
            # Linear trend with noise
            value = intercept + slope * t + np.random.normal(0, noise_level)

            data["id"].append(series_id)
            data["timestamp"].append(t)
            data["value"].append(value)

    # Create dataset
    data = pl.DataFrame(data)
    dataset = TSDataset(data, id_column_name="id", ts_column_name="timestamp")

    # Apply our linear regression transformer with expanding window
    lr_transformer = LinearRegressionWithoutOptimization(
        columns="value",
        window_types=WindowType.EXPANDING(),
    )
    transformed_dataset = lr_transformer.transform(dataset)

    # Extract results
    our_slopes = transformed_dataset.data["value_linear_regression_slope_expanding"].to_numpy()
    our_intercepts = transformed_dataset.data["value_linear_regression_intercept_expanding"].to_numpy()
    our_r_squared = transformed_dataset.data["value_linear_regression_r_squared_expanding"].to_numpy()

    # Calculate the same statistics using scipy for comparison
    scipy_slopes = []
    scipy_intercepts = []
    scipy_r_squared = []

    # Group by id to process each series separately
    for series_id in range(num_series):
        series_mask = np.array(data["id"]) == series_id
        series_timestamps = np.array(data["timestamp"])[series_mask]
        series_values = np.array(data["value"])[series_mask]

        # For each point in the series, calculate regression on all previous points (expanding window)
        for i in range(len(series_timestamps)):
            if i <= 1:  # Need at least 2 points for regression
                scipy_slopes.append(np.nan)
                scipy_intercepts.append(np.nan)
                scipy_r_squared.append(np.nan)
                continue

            # Get all points up to and including current point
            x = series_timestamps[: i + 1]
            y = series_values[: i + 1]

            # Calculate linear regression using scipy
            slope, intercept, r_value, _, _ = stats.linregress(x, y)

            scipy_slopes.append(slope)
            scipy_intercepts.append(intercept)
            scipy_r_squared.append(r_value**2)  # r_value is correlation coefficient, square it to get r²

    # Convert to numpy arrays
    scipy_slopes = np.array(scipy_slopes)
    scipy_intercepts = np.array(scipy_intercepts)
    scipy_r_squared = np.array(scipy_r_squared)

    # Compare results, allowing for small numerical differences
    # Filter out NaN values for comparison
    valid_indices = ~np.isnan(our_slopes) & ~np.isnan(scipy_slopes)

    # Check slopes
    np.testing.assert_allclose(our_slopes[valid_indices], scipy_slopes[valid_indices], rtol=1e-5, atol=1e-5)

    # Check intercepts
    np.testing.assert_allclose(our_intercepts[valid_indices], scipy_intercepts[valid_indices], rtol=1e-5, atol=1e-5)

    # Check r²
    np.testing.assert_allclose(our_r_squared[valid_indices], scipy_r_squared[valid_indices], rtol=1e-5, atol=1e-5)
