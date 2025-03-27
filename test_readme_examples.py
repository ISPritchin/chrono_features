# ruff: noqa: BLE001, S102

from pathlib import Path

from tests.utils.markdown_code_tester import check_code_examples_in_markdown_file


def test_readme() -> None:
    """Test all Python code examples in the README.md file."""
    readme_path = Path(__file__).parent / "README.md"
    check_code_examples_in_markdown_file(readme_path)
