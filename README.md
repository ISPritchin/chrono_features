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

## Core Concepts
### TSDataset
The `TSDataset` class is a wrapper around a polars DataFrame that provides additional functionality for time series data

```python
from chrono_features import TSDataset

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

from chrono_features import WindowType
from chrono_features.features import Sum, Median, Max
from chrono_features.transformation_pipeline import TransformationPipeline

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
print(dataset.data)
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

## License
This project is licensed under the terms of the LICENSE file (MIT License) included in the repository.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.