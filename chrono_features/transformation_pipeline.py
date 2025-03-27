# ruff: noqa: T201

from copy import deepcopy
from typing import Union

import polars as pl
import pandas as pd

from chrono_features.features._base import FeatureGenerator
from chrono_features.ts_dataset import TSDataset


class TransformationPipeline:
    """Pipeline for applying sequential transformations to time series data.

    The pipeline applies each transformation in the order they are provided, maintaining
    the state of the dataset through each step.

    Attributes:
        transformations: List of FeatureGenerator objects to apply.
        verbose: Whether to print progress information during transformation.

    Examples:
        >>> from chrono_features.ts_dataset import TSDataset
        >>> from chrono_features.features import Mean, Std
        >>> from chrono_features.window_type import WindowType
        >>> import polars as pl

        >>> # Create sample data
        >>> data = pl.DataFrame({
        ...     "id": [1, 1, 1, 2, 2],
        ...     "timestamp": [1, 2, 3, 1, 2],
        ...     "value": [10, 20, 30, 40, 50]
        ... })
        >>> dataset = TSDataset(data, id_column_name="id", ts_column_name="timestamp")

        >>> # Create feature generators
        >>> mean_feature = Mean(
        ...     columns="value",
        ...     window_types=WindowType.ROLLING(size=2)
        ... )
        >>> std_feature = Std(
        ...     columns="value",
        ...     window_types=WindowType.ROLLING(size=2)
        ... )

        >>> # Create and apply pipeline
        >>> pipeline = TransformationPipeline([mean_feature, std_feature])
        >>> transformed_dataset = pipeline.fit_transform(dataset)

        >>> # Or directly with a polars DataFrame
        >>> transformed_df = pipeline.fit_transform_df(
        ...     data,
        ...     id_column_name="id",
        ...     ts_column_name="timestamp"
        ... )
    """

    def __init__(self, transformations: list[FeatureGenerator], *, verbose: bool = False) -> None:
        """Initializes the transformation pipeline.

        Args:
            transformations: List of transformation objects to apply sequentially.
            verbose: If True, prints progress information during transformation.
        """
        self.transformations = transformations
        self.verbose = verbose
        self._validate_transformations()

    def _validate_transformations(self) -> None:
        """Validates that all pipeline steps are proper FeatureGenerator instances."""
        for i, trans in enumerate(self.transformations):
            if not isinstance(trans, FeatureGenerator):
                msg = f"Transformation #{i+1} must be a FeatureGenerator, got {type(trans)}"
                raise TypeError(msg)

    def fit_transform(
        self,
        data: TSDataset | pl.DataFrame | pd.DataFrame,
        id_column_name: str | None = None,
        ts_column_name: str | None = None,
    ) -> TSDataset | pl.DataFrame | pd.DataFrame:
        """Applies all transformations sequentially to the input data.

        This method intelligently handles different input types:
        - TSDataset: Used directly
        - polars.DataFrame: Converted to TSDataset using provided column names
        - pandas.DataFrame: Converted to polars, then to TSDataset, and result converted back to pandas

        Args:
            data: Input data (TSDataset, polars.DataFrame, or pandas.DataFrame)
            id_column_name: Name of the column containing entity identifiers (required for DataFrame inputs)
            ts_column_name: Name of the column containing timestamps (required for DataFrame inputs)

        Returns:
            Transformed data in the same format as the input

        Raises:
            TypeError: If input type is not supported or required parameters are missing
        """
        import pandas as pd

        # Case 1: Input is already a TSDataset
        if isinstance(data, TSDataset):
            current_dataset = data.clone()

            for i, transformation in enumerate(self.transformations):
                if self.verbose:
                    trans_name = transformation.__class__.__name__
                    print(f"Applying transformation {i+1}/{len(self.transformations)}: {trans_name}...")

                current_dataset = transformation.transform(current_dataset)

                if self.verbose:
                    new_cols = sorted(set(current_dataset.data.columns) - set(data.data.columns))
                    print(f"  Added columns: {list(new_cols)}")
                    print(
                        f"  Dataset shape: {len(current_dataset.data)} rows, \
                        {len(current_dataset.data.columns)} columns",
                    )

            return current_dataset

        # Case 2: Input is a polars DataFrame
        if isinstance(data, pl.DataFrame):
            if id_column_name is None or ts_column_name is None:
                msg = "When input is a DataFrame, id_column_name and ts_column_name must be provided"
                raise TypeError(msg)

            return self.fit_transform_polars(data, id_column_name, ts_column_name)

        # Case 3: Input is a pandas DataFrame
        if pd and isinstance(data, pd.DataFrame):
            if id_column_name is None or ts_column_name is None:
                msg = "When input is a DataFrame, id_column_name and ts_column_name must be provided"
                raise TypeError(msg)

            return self.fit_transform_pandas(data, id_column_name, ts_column_name)

        msg = f"Unsupported input type: {type(data)}. Expected TSDataset, polars.DataFrame, or pandas.DataFrame"
        raise TypeError(msg)

    def fit_transform_polars(self, df: pl.DataFrame, id_column_name: str, ts_column_name: str) -> pl.DataFrame:
        """Applies all transformations sequentially to a polars DataFrame.

        This method creates a TSDataset from the input DataFrame and applies
        all transformations, then returns the transformed DataFrame.

        Args:
            df: Input polars DataFrame to transform.
            id_column_name: Name of the column containing entity identifiers.
            ts_column_name: Name of the column containing timestamps.

        Returns:
            Transformed polars DataFrame after applying all transformations.

        Raises:
            TypeError: If input is not a polars DataFrame.
        """
        if not isinstance(df, pl.DataFrame):
            msg = "Input must be a polars DataFrame"
            raise TypeError(msg)

        # Create a TSDataset from the DataFrame
        dataset = TSDataset(df, id_column_name=id_column_name, ts_column_name=ts_column_name)

        # Apply transformations
        transformed_dataset = self.fit_transform(dataset)

        # Return the transformed DataFrame
        return transformed_dataset.data

    def fit_transform_pandas(self, df: pd.DataFrame, id_column_name: str, ts_column_name: str) -> pd.DataFrame:
        """Applies all transformations sequentially to a pandas DataFrame.

        This method converts the pandas DataFrame to polars, creates a TSDataset,
        applies all transformations, and then converts back to pandas.

        Args:
            df: Input pandas DataFrame to transform.
            id_column_name: Name of the column containing entity identifiers.
            ts_column_name: Name of the column containing timestamps.

        Returns:
            Transformed pandas DataFrame after applying all transformations.

        Raises:
            TypeError: If input is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            msg = "Input must be a pandas DataFrame"
            raise TypeError(msg)

        # Convert pandas DataFrame to polars
        pl_df = pl.from_pandas(df)

        # Use the polars method and convert back to pandas
        result_pl = self.fit_transform_polars(pl_df, id_column_name, ts_column_name)

        # Convert back to pandas and return
        return result_pl.to_pandas()

    def __add__(self, other: Union["TransformationPipeline", FeatureGenerator]) -> "TransformationPipeline":
        """Combines pipelines or adds a transformation using + operator.

        Args:
            other: Another pipeline or single transformation to add.

        Returns:
            New combined TransformationPipeline instance.

        Raises:
            TypeError: If other is not a pipeline or transformation.
        """
        if isinstance(other, FeatureGenerator):
            return TransformationPipeline([*self.transformations, deepcopy(other)])
        if isinstance(other, TransformationPipeline):
            return TransformationPipeline(deepcopy(self.transformations) + deepcopy(other.transformations))
        msg = f"Cannot add {type(other)} to TransformationPipeline"
        raise TypeError(msg)

    def get_transformation_names(self) -> list[str]:
        """Returns names of all transformations in the pipeline.

        Returns:
            List of transformation class names.
        """
        return [t.__class__.__name__ for t in self.transformations]

    def describe(self) -> str:
        """Generates a textual description of the pipeline.

        Returns:
            Multi-line string describing the pipeline steps and parameters.
        """
        desc = ["Transformation Pipeline with steps:"]
        for i, trans in enumerate(self.transformations):
            params = {k: v for k, v in trans.__dict__.items() if not k.startswith("_") and k not in ["numba_kwargs"]}
            desc.append(f"{i+1}. {trans.__class__.__name__}: {params}")
        return "\n".join(desc)

    def clone(self) -> "TransformationPipeline":
        """Creates a deep copy of the pipeline.

        Returns:
            New TransformationPipeline instance with copied transformations.
        """
        import copy

        return TransformationPipeline(transformations=copy.deepcopy(self.transformations), verbose=self.verbose)
