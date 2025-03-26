import sys

import libcst as cst
from libcst.metadata import PositionProvider


class FuncDefinitionFormatter(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Проверяем, есть ли type hints у параметров
        has_typehint = any(param.annotation is not None for param in updated_node.params.params)

        if not has_typehint:
            return updated_node

        # Форматируем параметры (каждый с новой строки)
        new_params = []
        for param in updated_node.params.params:
            new_params.append(
                param.with_changes(
                    comma=cst.Comma(
                        whitespace_after=cst.ParenthesizedWhitespace(
                            indent=True,
                            last_line=cst.SimpleWhitespace("    "),
                        )
                    )
                )
            )

        # Обновляем ноду с новыми параметрами
        return updated_node.with_changes(
            params=updated_node.params.with_changes(
                params=new_params,
                lpar=[cst.LeftParen(whitespace_after=cst.SimpleWhitespace("\n    "))],
                rpar=[cst.RightParen(whitespace_before=cst.SimpleWhitespace("\n    "))],
            )
        )


def process_file(file_path: str) -> bool:
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    try:
        module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)
        transformer = FuncDefinitionFormatter()
        modified_module = wrapper.visit(transformer)

        if modified_module.code != code:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(modified_module.code)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    changed = False
    for file_path in sys.argv[1:]:
        if file_path.endswith(".py"):
            changed |= process_file(file_path)
    sys.exit(1 if changed else 0)
