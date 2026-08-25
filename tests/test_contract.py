"""Pin all modules, public call signatures, class fields and class methods."""

import ast
import json
from pathlib import Path


def public_api(root):
    result = {}
    for path in sorted(root.glob('*.py')):
        tree = ast.parse(path.read_text())
        entries = {}
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith('_'):
                entries[node.name] = (
                    ast.unparse(node.args)
                    + ' -> '
                    + (ast.unparse(node.returns) if node.returns else '')
                )
            elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                fields = {}
                for child in node.body:
                    if isinstance(child, ast.AnnAssign):
                        fields[ast.unparse(child.target)] = (
                            ast.unparse(child.annotation)
                            + ' = '
                            + (ast.unparse(child.value) if child.value else '')
                        )
                    elif isinstance(child, ast.FunctionDef) and (
                        not child.name.startswith('_') or child.name == '__init__'
                    ):
                        fields[child.name] = (
                            ast.unparse(child.args)
                            + ' -> '
                            + (ast.unparse(child.returns) if child.returns else '')
                        )
                entries[node.name] = fields
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (
                        not target.id.startswith('_') or target.id in ('__all__', '__version__')
                    ):
                        entries[target.id] = ast.unparse(node.value)
        result[path.name] = entries
    return result


def test_public_api_golden():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'tests/contracts/public-api.json').read_text())
    assert public_api(root / 'src/speechloom') == expected
