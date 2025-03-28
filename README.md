# Chrono Features

A Python library for efficient time series feature generation with support for various window types and optimized calculations.

## Installation

```bash
pip install chrono-features
```

## Overview
Chrono Features is a library designed to simplify the process of generating features from time series data. It provides:

- Support for multiple window types (expanding, rolling, dynamic)
- Optimized calculations for better performance
- A consistent API for all feature generators
- Integration with polars DataFrames

```python
import polars as pl
from chrono_features import TSDataset, WindowType
from chrono_features.features import Max, Median, Sum, Std

# Create a sample dataset
data = pl.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2],
        "timestamp": [1, 2, 3, 1, 2, 3],
        "value": [1, 2, 3, 4, 5, 6],
    }
)

# Create a TSDataset
dataset = TSDataset(data, id_column_name="id", ts_column_name="timestamp")

# Create a feature transformer
max_transformer = Max(
    columns="value",
    window_types=WindowType.EXPANDING(),
)

# Apply the transformation
transformed_dataset = max_transformer.transform(dataset)

# View the result
print(transformed_dataset.data)
```
```output
shape: (6, 4)
┌─────┬───────────┬───────┬─────────────────────┐
│ id  ┆ timestamp ┆ value ┆ value_max_expanding │
│ --- ┆ ---       ┆ ---   ┆ ---                 │
│ i64 ┆ i64       ┆ i64   ┆ f64                 │
╞═════╪═══════════╪═══════╪═════════════════════╡
│ 1   ┆ 1         ┆ 1     ┆ 1.0                 │
│ 1   ┆ 2         ┆ 2     ┆ 2.0                 │
│ 1   ┆ 3         ┆ 3     ┆ 3.0                 │
│ 2   ┆ 1         ┆ 4     ┆ 4.0                 │
│ 2   ┆ 2         ┆ 5     ┆ 5.0                 │
│ 2   ┆ 3         ┆ 6     ┆ 6.0                 │
└─────┴───────────┴───────┴─────────────────────┘
```

## Supported Transformers

The following table lists all available transformers in the `chrono_features.features` package:

| Transformer | Description | Parameters |
| ----------- | ----------- | ---------- |
| `Max` | Calculates maximum value in window | `columns`, `window_types` |
| `Min` | Calculates the minimum value in window| `columns`, `window_types`|
| `Sum` | Calculates sum of values in window | `columns`, `window_types`, `use_prefix_sum_optimization` |
| `Mean` | Calculates average of values in window | `columns`, `window_types` |
| `Std` | Calculates standard deviation in window | `columns`, `window_types` |
| `Median` | Calculates median value in window | `columns`, `window_types` |
| `SimpleMovingAverage` | Calculates simple moving average | `columns`, `window_size`, `only_full_window` |
| `WeightedMovingAverage` | Calculates weighted moving average | `columns`, `window_size`, `weights`, `only_full_window` |

## Core Concepts
### TSDataset
The `TSDataset` class is a wrapper around a polars DataFrame that provides additional functionality for time series data

```python
import polars as pl

from chrono_features import TSDataset


df = pl.DataFrame({
    "id": [1, 1, 1, 2, 2, 2],
    "timestamp": [1, 2, 3, 1, 2, 3],
    "value": [5, 3, 4, 10, 2, 3],
})

# Create a TSDataset from a polars DataFrame
dataset = TSDataset(
    data=df,
    id_column_name="id",  # Column containing entity identifiers
    ts_column_name="timestamp"  # Column containing timestamps
)

# Add a new feature
dataset.add_feature("new_feature", [1, 2, 3, 4, 5, 6])
```


### Window Types
The library supports different types of windows for feature calculation. Window types determine how data points are grouped for feature calculation:

```python
from chrono_features import WindowType

# Expanding window (includes all previous values)
expanding_window = WindowType.EXPANDING()
# For each timestamp, includes all data points from the beginning up to the current timestamp
# Example: For timestamps [1, 2, 3], windows would be [1], [1, 2], [1, 2, 3]

# Rolling window (includes only the last N values)
rolling_window = WindowType.ROLLING(size=10)  # Window of size 10
# For each timestamp, includes at most N previous data points
# Example with size=2: For timestamps [1, 2, 3], windows would be [1], [1, 2], [2, 3]

# Rolling window with only full windows
rolling_window_full = WindowType.ROLLING(size=10, only_full_window=True)
# Only calculates features when the window has exactly N data points
# Example with size=2: For timestamps [1, 2, 3], windows would be [NaN], [1, 2], [2, 3]

# Dynamic window (window size varies based on a column)
dynamic_window = WindowType.DYNAMIC(len_column_name="window_len")
# Window size is determined by values in the specified column
# Example: If window_len column has values [1, 2, 1], windows would include
# the last 1, 2, and 1 data points respectively
```

#### Window Type Combinations
You can use multiple window types for a single feature generator:

```python
from chrono_features import WindowType
from chrono_features.features import Max

# Using multiple window types in a single transformer
max_transformer = Max(
    columns="value",
    window_types=[
        WindowType.EXPANDING(),
        WindowType.ROLLING(size=5),
        WindowType.ROLLING(size=10),
    ]
)
# This will create three output columns:
# - value_max_expanding
# - value_max_rolling_5
# - value_max_rolling_10
```

#### Window Type Behavior by ID
Windows are calculated separately for each unique ID in your dataset:

```python
# For a dataset with:
# id=1, timestamp=[1, 2, 3], value=[10, 20, 30]
# id=2, timestamp=[1, 2, 3], value=[40, 50, 60]

# With WindowType.EXPANDING():
# For id=1: windows are [10], [10, 20], [10, 20, 30]
# For id=2: windows are [40], [40, 50], [40, 50, 60]

# With WindowType.ROLLING(size=2):
# For id=1: windows are [10], [10, 20], [20, 30]
# For id=2: windows are [40], [40, 50], [50, 60]
```

### Transformation Pipeline
You can combine multiple transformers into a pipeline for more efficient processing:

```python
import pandas as pd
import polars as pl

from chrono_features import TSDataset, WindowType
from chrono_features.features import Sum, Median, Max
from chrono_features.transformation_pipeline import TransformationPipeline


dataset = TSDataset(
    data=pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "timestamp": [1, 2, 3, 1, 2, 3],
        "value": [1, 2, 3, 4, 5, 6],
    }),
    id_column_name="id",
    ts_column_name="timestamp"
)

# Create a pipeline with multiple transformers
pipeline = TransformationPipeline(
    [
        Sum(columns="value", window_types=WindowType.EXPANDING()),
        Median(columns="value", window_types=WindowType.ROLLING(size=10)),
        Max(columns="value", window_types=WindowType.EXPANDING()),
    ],
    verbose=True  # Print progress information
)

# Apply the pipeline to a TSDataset
transformed_dataset = pipeline.fit_transform(dataset)

# Or apply directly to a polars DataFrame
pl_df = pl.DataFrame({
    "id": [1, 1, 1, 2, 2, 2],
    "timestamp": [1, 2, 3, 1, 2, 3],
    "value": [1, 2, 3, 4, 5, 6],
})

# Transform the polars DataFrame directly
transformed_pl_df = pipeline.fit_transform(
    pl_df,
    id_column_name="id",
    ts_column_name="timestamp"
)

# Or apply to a pandas DataFrame
pd_df = pd.DataFrame({
    "id": [1, 1, 1, 2, 2, 2],
    "timestamp": [1, 2, 3, 1, 2, 3],
    "value": [1, 2, 3, 4, 5, 6],
})

# Transform the pandas DataFrame directly
transformed_pd_df = pipeline.fit_transform(
    pd_df,
    id_column_name="id",
    ts_column_name="timestamp"
)
```

## Examples
### Calculating Multiple Features

```python
import polars as pl
from chrono_features import TSDataset, WindowType
from chrono_features.features import Max, Median, Sum, Std
from chrono_features.transformation_pipeline import TransformationPipeline

# Create a sample dataset
data = pl.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2],
        "timestamp": [1, 2, 3, 1, 2, 3],
        "price": [10, 12, 15, 20, 18, 22],
        "volume": [100, 120, 150, 200, 180, 220],
    }
)

# Create a TSDataset
dataset = TSDataset(data, id_column_name="id", ts_column_name="timestamp")

# Create transformers for different columns
max_price = Max(columns="price", window_types=WindowType.EXPANDING())
sum_volume = Sum(columns="volume", window_types=WindowType.EXPANDING())
median_price = Median(columns="price", window_types=WindowType.ROLLING(size=2))
std_volume = Std(columns="volume", window_types=WindowType.ROLLING(size=2))

# Create a pipeline with multiple transformers
pipeline = TransformationPipeline(
    [
        max_price,
        sum_volume,
        median_price,
        std_volume,
    ],
    verbose=True  # Print progress information
)

# Apply the pipeline
transformed_dataset = pipeline.fit_transform(dataset)

# View the result
print(transformed_dataset.data)
```

```output
Applying transformation 1/4: MaxWithOptimization...
  Added columns: ['price_max_expanding']
  Dataset shape: 6 rows,                         5 columns
Applying transformation 2/4: Sum...
  Added columns: ['price_max_expanding', 'volume_sum_expanding']
  Dataset shape: 6 rows,                         6 columns
Applying transformation 3/4: Median...
  Added columns: ['price_max_expanding', 'price_median_rolling_2', 'volume_sum_expanding']
  Dataset shape: 6 rows,                         7 columns
Applying transformation 4/4: StdWithoutOptimization...
  Added columns: ['price_max_expanding', 'price_median_rolling_2', 'volume_std_rolling_2', 'volume_sum_expanding']
  Dataset shape: 6 rows,                         8 columns
shape: (6, 8)
┌─────┬───────────┬───────┬────────┬─────────────────────┬──────────────────────┬────────────────────────┬──────────────────────┐
│ id  ┆ timestamp ┆ price ┆ volume ┆ price_max_expanding ┆ volume_sum_expanding ┆ price_median_rolling_2 ┆ volume_std_rolling_2 │
│ --- ┆ ---       ┆ ---   ┆ ---    ┆ ---                 ┆ ---                  ┆ ---                    ┆ ---                  │
│ i64 ┆ i64       ┆ i64   ┆ i64    ┆ f64                 ┆ f64                  ┆ f32                    ┆ f32                  │
╞═════╪═══════════╪═══════╪════════╪═════════════════════╪══════════════════════╪══════════════════════╡
│ 1   ┆ 1         ┆ 10    ┆ 100    ┆ 10.0                ┆ 100.0                ┆ NaN                    ┆ NaN                  │
│ 1   ┆ 2         ┆ 12    ┆ 120    ┆ 12.0                ┆ 220.0                ┆ 11.0                   ┆ 10.0                 │
│ 1   ┆ 3         ┆ 15    ┆ 150    ┆ 15.0                ┆ 370.0                ┆ 13.5                   ┆ 15.0                 │
│ 2   ┆ 1         ┆ 20    ┆ 200    ┆ 20.0                ┆ 200.0                ┆ NaN                    ┆ NaN                  │
│ 2   ┆ 2         ┆ 18    ┆ 180    ┆ 20.0                ┆ 380.0                ┆ 19.0                   ┆ 10.0                 │
│ 2   ┆ 3         ┆ 22    ┆ 220    ┆ 22.0                ┆ 600.0                ┆ 20.0                   ┆ 20.0                 │
└─────┴───────────┴───────┴────────┴─────────────────────┴──────────────────────┴────────────────────────┴──────────────────────┘
```

### Using Dynamic Windows
```python
import polars as pl
from chrono_features import TSDataset, WindowType
from chrono_features.features import Max

# Create a sample dataset
data = pl.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2],
        "timestamp": [1, 2, 3, 1, 2, 3],
        "value": [1, 2, 3, 4, 5, 6],
        "window_len": [1, 2, 3, 1, 2, 3],  # Dynamic window lengths
    }
)

# Create a TSDataset
dataset = TSDataset(data, id_column_name="id", ts_column_name="timestamp")

# Create a transformer with dynamic window
max_transformer = Max(
    columns="value",
    window_types=WindowType.DYNAMIC(len_column_name="window_len"),
)

# Apply the transformation
transformed_dataset = max_transformer.transform(dataset)

# View the result
print(transformed_dataset.data)
```

```output
shape: (6, 5)
┌─────┬───────────┬───────┬────────────┬─────────────────────────────────┐
│ id  ┆ timestamp ┆ value ┆ window_len ┆ value_max_dynamic_based_on_win… │
│ --- ┆ ---       ┆ ---   ┆ ---        ┆ ---                             │
│ i64 ┆ i64       ┆ i64   ┆ i64        ┆ f32                             │
╞═════╪═══════════╪═══════╪════════════╪═════════════════════════════════╡
│ 1   ┆ 1         ┆ 1     ┆ 1          ┆ 1.0                             │
│ 1   ┆ 2         ┆ 2     ┆ 2          ┆ 2.0                             │
│ 1   ┆ 3         ┆ 3     ┆ 3          ┆ 3.0                             │
│ 2   ┆ 1         ┆ 4     ┆ 1          ┆ 4.0                             │
│ 2   ┆ 2         ┆ 5     ┆ 2          ┆ 5.0                             │
│ 2   ┆ 3         ┆ 6     ┆ 3          ┆ 6.0                             │
└─────┴───────────┴───────┴────────────┴─────────────────────────────────┘
```


## Optimization

The library implements several optimizations to improve performance when calculating features over time series data.

### Rolling Window Optimization for `Max` and `Min`

For rolling window calculations like `Max` and `Min`, a sliding window approach is used to avoid redundant calculations:

1. **Small Windows**: For windows of size 3 or smaller, direct calculation is used as it's already efficient.

2. **Sliding Window Technique**: For larger windows, the algorithm uses previous results when possible:
   - For `Max`:
     - If the new value is larger than the previous maximum, it becomes the new maximum
     - If the value leaving the window was the previous maximum, recalculate the maximum
     - Otherwise, keep the previous maximum
   - For `Min`:
     - If the new value is smaller than the previous minimum, it becomes the new minimum
     - If the value leaving the window was the previous minimum, recalculate the minimum
     - Otherwise, keep the previous minimum

This optimization significantly reduces computation time for large rolling windows, especially when the maximum/minimum values don't change frequently.

### Prefix Sum Optimization for `Sum`

For calculating sums over windows, the library uses a prefix sum (cumulative sum) optimization to avoid redundant addition operations:

1. **How Prefix Sums Work**: 
   - A prefix sum array stores cumulative sums of the original array
   - To find the sum of any window, subtract the prefix sum at the start of the window from the prefix sum at the end

2. **Example**:
   - Original array: `[3, 1, 4, 1, 5, 9]`
   - Prefix sum array: `[0, 3, 4, 8, 9, 14, 23]` (starting with 0)
   - To find sum of elements from index 2 to 4: `prefix_sum[5] - prefix_sum[2] = 14 - 4 = 10`

3. **Performance Improvement**:
   - Standard approach: O(n) operations per window (where n is window size)
   - Prefix sum approach: O(1) operations per window, regardless of window size
   - For a dataset with m windows, complexity improves from O(m×n) to O(m+n)

4. **When It's Used**:
   - Used  when `use_prefix_sum_optimization=True` and (`window size > 50` for rolling windows or for dynamic windows with any sizes)

This optimization is particularly effective for large windows or when many window calculations are needed.

## License
This project is licensed under the terms of the LICENSE file (MIT License) included in the repository.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.