import numpy as np
import pytest

from chrono_features import TSDataset, WindowType


def compare_multiple_implementations(
    dataset: TSDataset,
    implementations: list,
    window_type: WindowType,
    column: str = "value",
):
    """Compare multiple implementations against each other.

    Args:
        dataset: The dataset to transform
        implementations: List of (implementation_class, name) tuples
        window_type: The window type to use
        column: The column to process
    """
    results = {}

    # Apply each implementation
    for impl_class, name in implementations:
        transformer = impl_class(
            columns=column,
            window_types=window_type,
            out_column_names=f"{name}_result",
        )
        transformed = transformer.transform(dataset.clone())
        results[name] = transformed.data[f"{name}_result"].to_numpy()

    # Compare all implementations against each other
    for i, (name1, values1) in enumerate(results.items()):
        for name2, values2 in list(results.items())[i + 1 :]:
            is_close = np.allclose(values1, values2, rtol=1e-5, equal_nan=True)
            if not is_close:
                mask = ~np.isclose(values1, values2, rtol=1e-5, equal_nan=True)
                if np.any(mask):
                    diff_indices = np.where(mask)[0]
                    max_diff_idx = diff_indices[np.argmax(np.abs(values1[mask] - values2[mask]))]

                    # Get window type information
                    window_info = str(window_type)
                    if hasattr(window_type, "size"):
                        window_info += f" (size={window_type.size})"
                    elif hasattr(window_type, "len_column_name"):
                        window_info += f" (len_column={window_type.len_column_name})"

                    # Calculate percentage of different values
                    total_values = len(values1)
                    diff_count = np.sum(mask)
                    diff_percent = (diff_count / total_values) * 100

                    error_msg = (
                        f"Results from {name1} and {name2} don't match for {window_info}.\n"
                        f"Number of different values: {diff_count} ({diff_percent:.2f}% of total)\n"
                        f"Max difference at index {max_diff_idx}:\n"
                        f"  {name1}: {values1[max_diff_idx]}\n"
                        f"  {name2}: {values2[max_diff_idx]}\n"
                        f"  Absolute difference: {abs(values1[max_diff_idx] - values2[max_diff_idx])}"
                    )
                    pytest.fail(error_msg)
