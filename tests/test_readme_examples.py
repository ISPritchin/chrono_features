# ruff: noqa: BLE001, S102

import re
from pathlib import Path
import pytest
import ast


def extract_python_examples(readme_path: str | Path):
    readme_path = Path(readme_path)
    """Extract Python code examples from README.md."""
    with readme_path.open() as f:
        content = f.read()

    # Находим все блоки кода Python
    return re.findall(r"```python\n(.*?)```", content, re.DOTALL)


@pytest.fixture
def readme_path():
    """Return the absolute path to the README.md file."""
    current_dir = Path(__file__).parent
    repo_root = current_dir.parent
    return repo_root / "README.md"


@pytest.mark.parametrize("example_index", range(20))  # Проверяем до 20 примеров
def test_example(readme_path, example_index):
    """Test each example individually."""
    examples = extract_python_examples(readme_path)

    if example_index >= len(examples):
        pytest.skip(f"Only {len(examples)} examples available")

    example = examples[example_index]

    # Пропускаем примеры, которые явно являются комментариями
    if example.strip().startswith("#") and len(example.strip().split("\n")) < 3:
        pytest.skip("Example is just a comment")

    # Проверяем, является ли код синтаксически правильным
    try:
        ast.parse(example)
    except SyntaxError as e:
        error_message = f"Example #{example_index + 1} has syntax error: {e}\n\n\
            Code block:\n{'-' * 40}\n{example}\n{'-' * 40}"
        pytest.fail(error_message)

    # Выполняем пример
    try:
        # Создаем локальное пространство имен для выполнения
        local_vars = {}

        # Выполняем код
        exec(example, globals(), local_vars)

        # Если дошли до этой точки, значит пример выполнился без ошибок
        assert True
    except Exception as e:
        error_message = f"Example #{example_index + 1} failed with error: {e}\n\n \
            Code block:\n{'-' * 40}\n{example}\n{'-' * 40}"
        pytest.fail(error_message)
