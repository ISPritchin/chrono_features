# ruff: noqa: T201, BLE001, ISC002, S102

"""Utilities for testing code examples in markdown files."""

import ast
import io
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import pytest


def extract_python_examples(markdown_path: Path) -> list[str]:
    """Extract Python code examples from a markdown file."""
    with markdown_path.open() as f:
        content = f.read()

    # Find all Python code blocks
    extracted_examples = re.findall(r"```python\n(.*?)```", content, re.DOTALL)
    return extracted_examples


def extract_output_blocks(markdown_path: Path) -> dict[int, str]:
    """Extract output blocks that follow Python examples in a markdown file."""
    with markdown_path.open() as f:
        content = f.read()

    # Find all Python code blocks and their positions
    code_matches = list(re.finditer(r"```python\n(.*?)```", content, re.DOTALL))

    # Find all output blocks and their positions
    output_matches = list(re.finditer(r"```output\n(.*?)```", content, re.DOTALL))

    # Match each code block with the nearest following output block
    examples_outputs = {}

    for i, code_match in enumerate(code_matches):
        code_end = code_match.end()
        example_index = i

        # Find the nearest output block after the current code block
        for output_match in output_matches:
            output_start = output_match.start()

            # If output block comes after code block
            if output_start > code_end:
                # Check there are no other code blocks in between
                next_code_start = code_matches[i + 1].start() if i + 1 < len(code_matches) else float("inf")
                if output_start < next_code_start:
                    examples_outputs[example_index] = output_match.group(1)
                break

    return examples_outputs


def get_example_indices_with_line_numbers(markdown_path: Path) -> list[tuple[int, int]]:
    """Get the indices of Python examples in a markdown file with their line numbers."""
    with markdown_path.open() as f:
        content = f.read()

    code_matches = list(re.finditer(r"```python\n(.*?)```", content, re.DOTALL))

    result = []
    for i, match in enumerate(code_matches):
        line_number = content[: match.start()].count("\n") + 1
        result.append((i, line_number))

    return result


def check_code_examples_in_markdown_file(markdown_path: Path):
    """Check all Python code examples in a markdown file.

    Args:
        markdown_path: Path to the markdown file to check
    """
    examples_with_lines = get_example_indices_with_line_numbers(markdown_path)
    examples = extract_python_examples(markdown_path)
    output_blocks = extract_output_blocks(markdown_path)

    file_path_str = str(markdown_path.absolute())

    for example_index, line_number in examples_with_lines:
        example = examples[example_index]
        expected_output = output_blocks.get(example_index)

        # Skip examples that are clearly just comments
        if example.strip().startswith("#") and len(example.strip().split("\n")) < 3:
            continue

        # Check if the code is syntactically correct
        try:
            ast.parse(example)
        except SyntaxError as e:
            error_message = f"Example #{example_index + 1} ({file_path_str}:{line_number}) has syntax error: {e}\n\n\
                Code block:\n{'-' * 40}\n{example}\n{'-' * 40}"
            pytest.fail(error_message)

        # Execute the example and capture output
        try:
            # Create local namespace for execution
            local_vars = {}

            # Capture output
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                # Execute code
                exec(example, globals(), local_vars)

            # Get captured output
            actual_output = captured_output.getvalue().strip()

            # If example contains print() but has no output block, test should fail
            if "print(" in example and actual_output and expected_output is None:
                # Print clean output block to console for easy copying
                print(
                    f"\n\nCLEAN OUTPUT FOR EXAMPLE #{example_index + 1} \
                    ({file_path_str}:{line_number}):\n```output\n{actual_output}\n```\n",
                    file=sys.stderr,
                )

                error_message = (
                    f"Example #{example_index + 1} ({file_path_str}:{line_number}) contains print() "
                    "statements but has no output block.\n\n"
                    f"Please copy the CLEAN OUTPUT above and add it after the example in the markdown file."
                )
                pytest.fail(error_message)

            # If expected output is specified, compare it with actual output
            if expected_output is not None:
                expected_output = expected_output.strip()

                # Normalize whitespace and line breaks for more accurate comparison
                actual_output_normalized = re.sub(r"\s+", " ", actual_output).strip()
                expected_output_normalized = re.sub(r"\s+", " ", expected_output).strip()

                if actual_output_normalized != expected_output_normalized:
                    # Print clean output block to console for easy copying
                    print(
                        f"\n\nCLEAN OUTPUT FOR EXAMPLE #{example_index + 1} \
                        ({file_path_str}:{line_number}):\n```output\n{actual_output}\n```\n",
                        file=sys.stderr,
                    )

                    error_message = (
                        f"Example #{example_index + 1} ({file_path_str}:{line_number}) \
                            output doesn't match expected output.\n\n"
                        f"Please copy the CLEAN OUTPUT above and update the output block in the markdown file."
                    )
                    pytest.fail(error_message)

        except Exception:
            tb = traceback.format_exc()
            error_message = (
                f"Example #{example_index + 1} ({file_path_str}:{line_number}) failed with error:\n\n"
                f"{tb}\n\n"
                f"Code block:\n{'-' * 40}\n{example}\n{'-' * 40}"
            )
            pytest.fail(error_message)
