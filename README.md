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
The library supports different types of windows for feature calculation:

```python
from chrono_features import WindowType

# Expanding window (includes all previous values)
expanding_window = WindowType.EXPANDING()

# Rolling window (includes only the last N values)
rolling_window = WindowType.ROLLING(size=10)  # Window of size 10
rolling_window_full = WindowType.ROLLING(size=10, only_full_window=True)  # Only calculate when window is full

# Dynamic window (window size varies based on a column)
dynamic_window = WindowType.DYNAMIC(len_column_name="window_len")  # Window size from 'window_len' column
```

### Feature Generators
The library includes various feature generators:

```python
# Calculate the maximum value in each window
from chrono_features.features import Max

max_transformer = Max(
    columns="value",  # Column to calculate max for
    window_types=WindowType.EXPANDING(),  # Window type
    use_optimization=True,  # Use optimized calculation
    out_column_names="custom_max"  # Custom output column name (optional)
)

# Calculate the median value in each window
from chrono_features.features import Median

median_transformer = Median(
    columns="value",
    window_types=WindowType.ROLLING(size=10)
)

# Calculate the sum of values in each window
from chrono_features.features import Sum

sum_transformer = Sum(
    columns="value",
    window_types=WindowType.EXPANDING(),
    use_prefix_sum_optimization=True  # Use prefix sum optimization
)

# Calculate the standard deviation in each window
from chrono_features.features import Std

std_transformer = Std(
    columns="value",
    window_types=WindowType.EXPANDING()
)

# Calculate the absolute energy (sum of squares) in each window
from chrono_features.features.absolute_energy import AbsoluteEnergy

energy_transformer = AbsoluteEnergy(
    columns="value",
    window_types=WindowType.EXPANDING()
)

# Calculate the absolute sum of changes in each window
from chrono_features.features.absolute_sum_of_changes import AbsoluteSumOfChanges

changes_transformer = AbsoluteSumOfChanges(
    columns="value",
    window_types=WindowType.EXPANDING()
)

# Calculate the autocorrelation in each window
from chrono_features.features.autocorrelation import Autocorrelation

autocorr_transformer = Autocorrelation(
    columns="value",
    window_types=WindowType.EXPANDING(),
    lag=1  # Lag for autocorrelation
)
```

### Transformation Pipeline
You can combine multiple transformers into a pipeline:

```python
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

# Apply the pipeline
transformed_dataset = pipeline.fit_transform(dataset)
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