# ruff: noqa: C901, PLR0912, PLR0915, ISC003

import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl
import psutil
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from chrono_features import TSDataset, WindowType
from chrono_features.features._base import FeatureGenerator


def performance_comparison(
    datasets: dict[str, TSDataset],
    transformers: list[FeatureGenerator],
    output_xlsx_file_path: str | Path,
) -> None:
    """
    Run performance comparison for multiple transformers across different datasets.

    Args:
        datasets: Dictionary mapping dataset names to TSDataset instances
        transformers: List of transformer instances to test
        output_xlsx_file_path: Path to the output Excel file
    """
    # Create output directory if it doesn't exist
    Path(output_xlsx_file_path.parent).mkdir(exist_ok=True, parents=True)

    try:
        system_info = {
            "OS": platform.system() + " " + platform.release(),
            "Python Version": platform.python_version(),
            "CPU": platform.processor(),
            "CPU Cores": psutil.cpu_count(logical=False),
            "CPU Threads": psutil.cpu_count(logical=True),
            "Total RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
            "Available RAM (GB)": round(psutil.virtual_memory().available / (1024**3), 2),
        }
    except ImportError:
        system_info = {
            "OS": platform.system() + " " + platform.release(),
            "Python Version": platform.python_version(),
            "CPU": platform.processor(),
        }

    # Store all results
    all_results = []
    all_dataset_infos = {}

    # Process each dataset
    for dataset_name, dataset in datasets.items():
        # Collect dataset information
        n_ids = dataset.data[dataset.id_column_name].n_unique()
        n_timestamps_per_id = len(dataset.data) // n_ids
        total_rows = len(dataset.data)
        memory_usage_mb = dataset.data.estimated_size() / (1024 * 1024)

        dataset_info = {
            "Dataset name": dataset_name,
            "Number of unique IDs": n_ids,
            "Timestamps per ID": n_timestamps_per_id,
            "Total rows": total_rows,
            "Memory usage (MB)": round(memory_usage_mb, 2),
            "Test date": time.strftime("%Y-%m-%d %H:%M:%S"),
            **system_info,
        }

        # Store dataset info for later use
        all_dataset_infos[dataset_name] = dataset_info

        # Process each transformer
        for transformer in transformers:
            # Get transformer name
            transformer_name = transformer.__class__.__name__

            # Extract transformer parameters
            transformer_params = {}
            for attr_name in dir(transformer):
                # Skip private attributes, methods, and callables
                if attr_name.startswith("_") or callable(getattr(transformer, attr_name)):
                    continue

                # Get important parameters
                if attr_name in ["columns", "window_types", "use_optimization", "out_column_names"]:
                    transformer_params[attr_name] = getattr(transformer, attr_name)

            # Format parameters as string
            params_str = ", ".join([f"{k}={v}" for k, v in transformer_params.items()])

            # Run the transformation and measure time
            start_time = time.time()
            transformed_dataset = transformer.transform(dataset)
            execution_time = time.time() - start_time

            # Get output column name
            if hasattr(transformer, "out_column_names") and transformer.out_column_names:
                out_col = transformer.out_column_names[0]
            else:
                out_col = f"result_{transformer_name.lower()}"

            # Print results

            # Store result
            result = {
                "dataset_size": dataset_name,
                "dataset_info": dataset_info,
                "transformer": transformer_name,
                "transformer_params": params_str,
                "execution_time": execution_time,
                "output_column": out_col,
                "result": transformed_dataset.data[out_col].to_numpy(),
            }

            all_results.append(result)

    # Create combined dataset info
    combined_info = {
        "Test type": "Performance comparison",
        "Datasets tested": ", ".join(datasets.keys()),
        "Transformers tested": ", ".join({r["transformer"] for r in all_results}),
        "Test date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "System": system_info.get("OS", "Unknown"),
        "Python Version": system_info.get("Python Version", "Unknown"),
        "CPU": system_info.get("CPU", "Unknown"),
        "CPU Cores/Threads": f"{system_info.get('CPU Cores', 'Unknown')}/{system_info.get('CPU Threads', 'Unknown')}",
        "Total RAM (GB)": system_info.get("Total RAM (GB)", "Unknown"),
    }

    save_results_to_excel(all_results, combined_info, output_xlsx_file_path)


def create_dataset(n_ids: int, n_timestamps: int) -> TSDataset:
    """Create a dataset with specified number of IDs and timestamps per ID."""
    ids = np.repeat(range(n_ids), n_timestamps)
    timestamps = np.tile(np.arange(1, n_timestamps + 1), n_ids)
    values = np.random.rand(n_ids * n_timestamps)
    data = pl.DataFrame(
        {
            "id": ids,
            "timestamp": timestamps,
            "value": values,
        },
    )
    return TSDataset(data, id_column_name="id", ts_column_name="timestamp")


def run_performance_test(
    dataset: TSDataset,
    transformer_factory: Callable[[bool, WindowType, str], FeatureGenerator],
    window_type: WindowType,
    window_desc: str,
    *,
    use_optimization: bool,
) -> dict:
    """
    Run a single performance test and return timing results.

    Args:
        dataset: The dataset to transform
        transformer_factory: A function that creates a transformer instance
        window_type: The window type to use
        window_desc: A string description of the window type
        use_optimization: Whether to use optimization

    Returns:
        Dict with test results
    """
    # Create transformer using the factory function
    transformer = transformer_factory(use_optimization, window_type, window_desc)

    # Get output column name from transformer
    if hasattr(transformer, "out_column_names") and transformer.out_column_names:
        out_col = transformer.out_column_names[0]
    else:
        out_col = f"result_{window_desc}_{'opt' if use_optimization else 'no_opt'}"

    # Run the transformation and measure time
    start_time = time.time()
    transformed_dataset = transformer.transform(dataset)
    execution_time = time.time() - start_time

    return {
        "window_type": window_desc,
        "window_params": str(window_type),
        "optimization": "Yes" if use_optimization else "No",
        "execution_time": execution_time,
        "output_column": out_col,
        "result": transformed_dataset.data[out_col].to_numpy(),
    }


def save_results_to_excel(results: list[dict], dataset_info: dict, filename: str) -> None:
    """Save performance test results to an Excel file."""
    # Create dataset info dataframe
    info_data = [[key, value] for key, value in dataset_info.items()]
    info_df = pd.DataFrame(info_data, columns=["Metric", "Value"])

    # Create performance results dataframe
    perf_data = []

    # Simplify results to just include dataset, transformer, and time
    for result in results:
        # Format transformer parameters with each parameter on a new line
        params_str = result.get("transformer_params", "")
        formatted_params = params_str.replace(", ", "\n")

        perf_data.append(
            {
                "Dataset Name": result.get("dataset_size", "unknown"),
                "Dataset Parameters": f"IDs: {result['dataset_info'].get('Number of unique IDs', 'N/A')}, "
                + f"Timestamps: {result['dataset_info'].get('Timestamps per ID', 'N/A')}, "
                + f"Total Rows: {result['dataset_info'].get('Total rows', 'N/A')}",
                "Transformer Name": result.get("transformer", "unknown"),
                "Transformer Parameters": formatted_params,
                "Time (s)": result["execution_time"],
            },
        )

    # Create DataFrame and sort by Dataset Name and then group by similar Transformer Parameters
    perf_df = pd.DataFrame(perf_data)

    # Sort by Dataset Name first
    perf_df = perf_df.sort_values(by=["Dataset Name", "Transformer Parameters"])

    # Create Excel workbook
    wb = Workbook()

    # Dataset info sheet
    info_sheet = wb.active
    info_sheet.title = "Test Information"

    # Add title
    info_sheet["A1"] = "TEST INFORMATION"
    info_sheet["A1"].font = Font(bold=True, size=14)
    info_sheet.merge_cells("A1:B1")

    # Add data
    for r_idx, row in enumerate(dataframe_to_rows(info_df, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            info_sheet.cell(row=r_idx, column=c_idx, value=value)

    # Auto-adjust column widths for info sheet
    for column in info_sheet.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                cell_length = len(str(cell.value))
                max_length = max(max_length, cell_length)
        adjusted_width = max_length + 2  # Add padding
        info_sheet.column_dimensions[column_letter].width = adjusted_width

    # Create datasets comparison sheet
    datasets_sheet = wb.create_sheet(title="Datasets Comparison")

    # Extract dataset-specific information from results
    dataset_comparison_data = []
    dataset_sizes = {r.get("dataset_size", "unknown") for r in results}

    for dataset_size in dataset_sizes:
        # Find any result for this dataset to extract dataset info
        dataset_result = next((r for r in results if r.get("dataset_size") == dataset_size), None)
        if dataset_result and "dataset_info" in dataset_result:
            dataset_info = dataset_result["dataset_info"]
            dataset_comparison_data.append(
                {
                    "Dataset": dataset_size,
                    "IDs": dataset_info.get("Number of unique IDs", "N/A"),
                    "Timestamps per ID": dataset_info.get("Timestamps per ID", "N/A"),
                    "Total Rows": dataset_info.get("Total rows", "N/A"),
                    "Memory (MB)": dataset_info.get("Memory usage (MB)", "N/A"),
                },
            )

    # Add title to datasets sheet
    datasets_sheet["A1"] = "DATASETS COMPARISON"
    datasets_sheet["A1"].font = Font(bold=True, size=14)
    datasets_sheet.merge_cells("A1:E1")

    # Create DataFrame for datasets comparison
    datasets_df = pd.DataFrame(dataset_comparison_data)

    # Add datasets comparison data
    for r_idx, row in enumerate(dataframe_to_rows(datasets_df, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            datasets_sheet.cell(row=r_idx, column=c_idx, value=value)

    # Auto-adjust column widths for datasets sheet
    for column in datasets_sheet.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                cell_length = len(str(cell.value))
                max_length = max(max_length, cell_length)
        adjusted_width = max_length + 2  # Add padding
        datasets_sheet.column_dimensions[column_letter].width = adjusted_width

    # Performance results sheet
    perf_sheet = wb.create_sheet(title="Performance Results")

    # Add title
    perf_sheet["A1"] = "PERFORMANCE RESULTS"
    perf_sheet["A1"].font = Font(bold=True, size=14)
    perf_sheet.merge_cells("A1:E1")

    # Add data
    for r_idx, row in enumerate(dataframe_to_rows(perf_df, index=False, header=True), 2):
        for c_idx, value in enumerate(row, 1):
            cell = perf_sheet.cell(row=r_idx, column=c_idx, value=value)

            # Format time with 6 decimal places
            if c_idx == 5 and r_idx > 2:
                cell.value = f"{value:.6f}"

            # Set text alignment for transformer parameters to allow line breaks
            if c_idx == 4 and r_idx > 2:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

    # Auto-adjust column widths for performance sheet
    column_max_lengths = {}
    for column in perf_sheet.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                # For multiline content, check each line
                if isinstance(cell.value, str) and "\n" in cell.value:
                    lines = cell.value.split("\n")
                    for line in lines:
                        max_length = max(max_length, len(line))
                else:
                    cell_length = len(str(cell.value))
                    max_length = max(max_length, cell_length)

        # Set minimum and maximum widths
        adjusted_width = min(max(max_length + 2, 15), 60)  # Min 15, Max 60
        column_max_lengths[column_letter] = max_length
        perf_sheet.column_dimensions[column_letter].width = adjusted_width

    # Auto-adjust row heights for the parameter cells
    for row in range(3, perf_sheet.max_row + 1):
        # Get the transformer parameters cell in column D (4)
        param_cell = perf_sheet.cell(row=row, column=4)
        if param_cell.value and isinstance(param_cell.value, str):
            # Count number of lines
            num_lines = param_cell.value.count("\n") + 1
            # Set row height based on number of lines (approximately 15 points per line)
            row_height = max(20, min(num_lines * 15, 150))  # Min 20, Max 150
            perf_sheet.row_dimensions[row].height = row_height

    # Save workbook
    wb.save(filename)
