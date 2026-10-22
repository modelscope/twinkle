from __future__ import annotations

import ast
from pathlib import Path

_SERVER = Path(__file__).resolve().parents[3] / 'src' / 'twinkle' / 'server'
_TWINKLE_HANDLERS = (
    _SERVER / 'gateway' / 'twinkle_handlers.py',
    _SERVER / 'model' / 'twinkle_handlers.py',
    _SERVER / 'sampler' / 'twinkle_handlers.py',
    _SERVER / 'processor' / 'twinkle_handlers.py',
)


def _status_code(call: ast.Call) -> int | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, int):
        return call.args[0].value
    for keyword in call.keywords:
        if keyword.arg == 'status_code' and isinstance(keyword.value, ast.Constant):
            return keyword.value.value if isinstance(keyword.value.value, int) else None
    return None


def test_twinkle_handlers_have_no_http_exception_bypass_except_iterator_410() -> None:
    violations: list[str] = []
    allowed_410 = 0
    for path in _TWINKLE_HANDLERS:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            func = node.exc.func
            if not isinstance(func, ast.Name) or func.id != 'HTTPException':
                continue
            status_code = _status_code(node.exc)
            if path.parent.name == 'processor' and status_code == 410:
                allowed_410 += 1
            else:
                violations.append(f'{path.relative_to(_SERVER)}:{node.lineno} status={status_code}')

    assert allowed_410 == 1
    assert violations == []
