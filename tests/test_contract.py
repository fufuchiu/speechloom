"""Freeze public signatures, decorators, inheritance and data fields."""

import ast
import json
from pathlib import Path


def callable_contract(node):
    return {
        'signature': ast.unparse(node.args),
        'returns': ast.unparse(node.returns) if node.returns else None,
        'decorators': [ast.unparse(value) for value in node.decorator_list],
        'async': isinstance(node, ast.AsyncFunctionDef),
    }


def public_api(root):
    result = {}
    for path in sorted(root.glob('*.py')):
        entries = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith('_'):
                entries[node.name] = callable_contract(node)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                fields, methods, instance_fields = {}, {}, set()
                for child in node.body:
                    if isinstance(child, ast.AnnAssign):
                        fields[ast.unparse(child.target)] = {
                            'type': ast.unparse(child.annotation),
                            'default': ast.unparse(child.value) if child.value else None,
                        }
                    elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not child.name.startswith('_') or child.name == '__init__':
                            methods[child.name] = callable_contract(child)
                        for expression in ast.walk(child):
                            if isinstance(expression, ast.Attribute) and isinstance(
                                expression.ctx, ast.Store
                            ):
                                if (
                                    isinstance(expression.value, ast.Name)
                                    and expression.value.id == 'self'
                                    and not expression.attr.startswith('_')
                                ):
                                    instance_fields.add(expression.attr)
                entries[node.name] = {
                    'bases': [ast.unparse(value) for value in node.bases],
                    'decorators': [ast.unparse(value) for value in node.decorator_list],
                    'fields': fields,
                    'methods': methods,
                    'instance_fields': sorted(instance_fields),
                }
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
