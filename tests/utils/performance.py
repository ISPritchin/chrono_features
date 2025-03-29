# ruff: noqa: C901, PLR0912, PLR0915, ISC003, T201
import time
import platform
import psutil
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

from chrono_features import TSDataset, WindowType
from chrono_features.features._base import FeatureGenerator


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


def create_dataset_with_dynamic_windows(n_ids, n_timestamps, max_window_size=1000):
    """Create a dataset with an additional column for dynamic window lengths.

    Args:
        n_ids: Number of unique IDs in the dataset.
        n_timestamps: Number of timestamps per ID.
        max_window_size: Maximum value for dynamic window length (default: 1000).

    Returns:
        TSDataset with added dynamic window length column.
    """
    dataset = create_dataset(n_ids=n_ids, n_timestamps=n_timestamps)

    # Add a single dynamic window length column
    total_rows = len(dataset.data)

    # Values from 0 to max_window_size
    dynamic_len = np.random.randint(0, max_window_size + 1, size=total_rows)
    dataset.add_feature("dynamic_len", dynamic_len)

    return dataset


def compare_performance(
    dataset: TSDataset,
    implementations: list[tuple[type[FeatureGenerator], str]],
    window_types: list[WindowType],
    column_name: str = "value",  # Renamed from 'column' to 'column_name' to avoid redefining argument
    output_file: str | None = None,
    dataset_name: str = "Default",
) -> pd.DataFrame:
    """Compare performance of multiple implementations across different window types."""
    results = []

    # Process each window type
    for window_type in window_types:
        window_info = str(window_type)

        # Apply each implementation
        for impl_class, name in implementations:
            # Skip if any previous implementation for this window exceeded max time

            # Create transformer
            transformer = impl_class(
                columns=column_name,  # Use column_name instead of column
                window_types=window_type,
                out_column_names=f"{name}_result",
            )

            # Measure execution time
            start_time = time.time()
            try:
                _ = transformer.transform(dataset.clone())
                execution_time = time.time() - start_time

                # Create result dictionary with all required fields
                dataset_size = (
                    f"{len(dataset.data)} rows "  # Split long line
                    f"({dataset.data[dataset.id_column_name].n_unique()} IDs)"
                )
                result = {
                    "dataset_size": dataset_size,
                    "window_type": window_info,
                    f"{name}_time": execution_time,
                    "transformer": name,
                    "implementation_class": impl_class.__name__,
                    "execution_time": execution_time,
                    "dataset_info": {
                        "Number of unique IDs": dataset.data[dataset.id_column_name].n_unique(),
                        "Timestamps per ID": len(dataset.data) // dataset.data[dataset.id_column_name].n_unique(),
                        "Total rows": len(dataset.data),
                    },
                }

                results.append(result)

            except Exception as e:  # noqa: BLE001
                print(f"Error with {name} on {window_type}: {e!s}")
                # Still add a result with the error
                dataset_size = (
                    f"{len(dataset.data)} rows "  # Split long line
                    f"({dataset.data[dataset.id_column_name].n_unique()} IDs)"
                )
                results.append(
                    {
                        "dataset_size": dataset_size,
                        "window_type": window_info,
                        f"{name}_time": f"error: {e!s}",
                        "transformer": name,
                        "implementation_class": impl_class.__name__,
                        "execution_time": f"error: {e!s}",
                        "dataset_info": {
                            "Number of unique IDs": dataset.data[dataset.id_column_name].n_unique(),
                            "Timestamps per ID": len(dataset.data) // dataset.data[dataset.id_column_name].n_unique(),
                            "Total rows": len(dataset.data),
                        },
                    },
                )

    results_df = pd.DataFrame(results)

    # Save to Excel file if requested
    if output_file:
        # Check if file exists to determine if we need to create a new workbook or append to existing
        if Path(output_file).exists():  # Use Path.exists() instead of os.path.exists()
            # Load existing workbook
            wb = load_workbook(output_file)

            # Remove unwanted sheets if they exist
            sheets_to_remove = ["Test Information", "Datasets Comparison", "Performance Results"]
            for sheet_name in sheets_to_remove:
                if sheet_name in wb.sheetnames:
                    del wb[sheet_name]
        else:
            # Create a new workbook
            wb = Workbook()
            # Remove default sheet
            if "Sheet" in wb.sheetnames:
                del wb["Sheet"]

        # Create or get sheet for this dataset
        sheet_name = f"{dataset_name} Results"
        if sheet_name in wb.sheetnames:
            # Instead of clearing cells, delete the sheet and create a new one
            del wb[sheet_name]
            sheet = wb.create_sheet(title=sheet_name)
        else:
            sheet = wb.create_sheet(title=sheet_name)

        # Add title
        sheet["A1"] = f"PERFORMANCE RESULTS - {dataset_name.upper()} DATASET"
        sheet["A1"].font = Font(bold=True, size=14)
        sheet.merge_cells("A1:F1")

        # Add dataset description
        row = 3
        sheet[f"A{row}"] = "Dataset Information"
        sheet[f"A{row}"].font = Font(bold=True)
        sheet.merge_cells(f"A{row}:F{row}")
        row += 1

        # Add dataset details
        dataset_info = {
            "Number of unique IDs": dataset.data[dataset.id_column_name].n_unique(),
            "Timestamps per ID": len(dataset.data) // dataset.data[dataset.id_column_name].n_unique(),
            "Total rows": len(dataset.data),
            "Test date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add system info if available
        dataset_info.update(
            {
                "OS": platform.system() + " " + platform.release(),
                "Python Version": platform.python_version(),
                "CPU": platform.processor(),
                "CPU Cores": psutil.cpu_count(logical=False),
                "CPU Threads": psutil.cpu_count(logical=True),
                "Total RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
            },
        )

        # Write dataset info
        for key, value in dataset_info.items():
            sheet[f"A{row}"] = key
            sheet[f"B{row}"] = str(value)
            sheet[f"A{row}"].font = Font(italic=True)
            row += 1

        # Add empty row
        row += 1

        # Add results header
        sheet[f"A{row}"] = "Performance Results"
        sheet[f"A{row}"].font = Font(bold=True)
        sheet.merge_cells(f"A{row}:C{row}")
        row += 2

        # Prepare data for Excel - create a more readable format
        for window_type in window_types:
            window_str = str(window_type)
            window_results = [r for r in results if r["window_type"] == window_str]

            if window_results:
                # Add window type as a header row
                sheet[f"A{row}"] = window_str
                sheet[f"A{row}"].font = Font(bold=True)
                sheet.merge_cells(f"A{row}:C{row}")
                row += 1

                # Add column headers
                sheet[f"B{row}"] = "Implementation"
                sheet[f"C{row}"] = "Time (s)"
                sheet[f"B{row}"].font = Font(bold=True)
                sheet[f"C{row}"].font = Font(bold=True)
                row += 1

                # Add results for each implementation
                for _, name in implementations:  # Use _ for unused impl_class variable
                    # Find result for this implementation
                    impl_result = next((r for r in window_results if r["transformer"] == name), None)

                    if impl_result:
                        time_value = impl_result.get(f"{name}_time", "N/A")
                        time_str = f"{time_value:.6f}" if isinstance(time_value, (int, float)) else str(time_value)

                        sheet[f"B{row}"] = f"{name} ({impl_result['implementation_class']})"
                        sheet[f"C{row}"] = time_str

                        # Highlight implementation classes
                        if (
                            "WithOptimization" in impl_result["implementation_class"]
                            or "Optimized" in impl_result["implementation_class"]
                        ):
                            sheet[f"B{row}"].font = Font(color="006100")  # Dark green
                        elif (
                            "WithoutOptimization" in impl_result["implementation_class"]
                            or "Standard" in impl_result["implementation_class"]
                        ):
                            sheet[f"B{row}"].font = Font(color="9C0006")  # Dark red

                        row += 1

                # Add empty row between window types
                row += 1

        # Auto-adjust column widths
        for column in sheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                if cell.value:
                    cell_length = len(str(cell.value))
                    max_length = max(max_length, cell_length)

            # Set minimum and maximum widths
            adjusted_width = min(max(max_length + 2, 15), 60)  # Min 15, Max 60
            sheet.column_dimensions[column_letter].width = adjusted_width

        # Save workbook
        wb.save(output_file)
        print(f"Results for {dataset_name} dataset saved to Excel file: {output_file}")

    return results_df
