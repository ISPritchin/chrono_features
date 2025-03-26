from copy import deepcopy
from typing import List, Union

from chrono_features.features._base import FeatureGenerator
from chrono_features.ts_dataset import TSDataset


class TransformationPipeline:
    """Pipeline for applying sequential transformations to time series data.

    The pipeline applies each transformation in the order they are provided, maintaining
    the state of the dataset through each step.

    Attributes:
        transformations: List of FeatureGenerator objects to apply.
        verbose: Whether to print progress information during transformation.
    """

    def __init__(self, transformations: List[FeatureGenerator], verbose: bool = True):
        """Initializes the transformation pipeline.

        Args:
            transformations: List of transformation objects to apply sequentially.
            verbose: If True, prints progress information during transformation.
        """
        self.transformations = transformations
        self.verbose = verbose
        self._validate_transformations()

    def _validate_transformations(self):
        """Validates that all pipeline steps are proper FeatureGenerator instances."""
        for i, trans in enumerate(self.transformations):
            if not isinstance(trans, FeatureGenerator):
                raise TypeError(f"Transformation #{i+1} must be a FeatureGenerator, got {type(trans)}")

    def fit_transform(self, dataset: TSDataset) -> TSDataset:
        """Applies all transformations sequentially to the input dataset.

        Args:
            dataset: Input TSDataset to transform.

        Returns:
            Transformed TSDataset after applying all transformations.

        Raises:
            TypeError: If input is not a TSDataset instance.
        """
        if not isinstance(dataset, TSDataset):
            raise TypeError("Input must be a TSDataset object")

        current_dataset = dataset.clone()

        for i, transformation in enumerate(self.transformations):
            if self.verbose:
                trans_name = transformation.__class__.__name__
                print(f"Applying transformation {i+1}/{len(self.transformations)}: {trans_name}...")

            current_dataset = transformation.transform(current_dataset)

            if self.verbose:
                new_cols = set(current_dataset.data.columns) - set(dataset.data.columns)
                print(f"  Added columns: {list(new_cols)}")
                print(f"  Dataset shape: {len(current_dataset.data)} rows, {len(current_dataset.data.columns)} columns")

        return current_dataset

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
            return TransformationPipeline(self.transformations + [deepcopy(other)])
        elif isinstance(other, TransformationPipeline):
            return TransformationPipeline(deepcopy(self.transformations) + deepcopy(other.transformations))
        else:
            raise TypeError(f"Cannot add {type(other)} to TransformationPipeline")

    def get_transformation_names(self) -> List[str]:
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
