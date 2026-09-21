# Copyright (c) ModelScope Contributors. All rights reserved.
"""Executable client/server package-boundary contracts."""
from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).parents[3] / 'src'
_ALLOWED_SERVER_IMPORTS = (
    'twinkle.protocol.types',
    'twinkle.protocol.headers',
    'twinkle.protocol.json_utils',
    'twinkle.protocol.serialize',
)


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_client_does_not_import_server():
    offenders = []
    for path in (_ROOT / 'twinkle_client').rglob('*.py'):
        for module in _imports(path):
            if module == 'twinkle.server' or module.startswith('twinkle.server.'):
                offenders.append(f'{path.relative_to(_ROOT)} -> {module}')
    assert offenders == []


def test_server_only_imports_shared_client_contracts():
    offenders = []
    for path in (_ROOT / 'twinkle' / 'server').rglob('*.py'):
        for module in _imports(path):
            if module == 'twinkle_client' or module.startswith('twinkle_client.'):
                if not any(module == allowed or module.startswith(f'{allowed}.')
                           for allowed in _ALLOWED_SERVER_IMPORTS):
                    offenders.append(f'{path.relative_to(_ROOT)} -> {module}')
    assert offenders == []
