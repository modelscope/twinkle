# Copyright (c) ModelScope Contributors. All rights reserved.
"""Static check: model backend calls map the tenant adapter name (F002 / P002).

``run_submit`` hands each ``backend_call`` a *tenant-scoped* adapter name
(``owner_id-<name>``). Before that reaches a backend model method it must be
translated by ``ModelManagement.resolve_model_adapter_name`` — which returns
``''`` in full-parameter mode (the empty-string default optimizer group) and the
name unchanged in LoRA mode. The inline forward/backward/step endpoints did this;
the ``*_from_data_plane`` endpoints originally passed the raw tenant name, so a
full-mode data-plane request drove the wrong optimizer group.

This guard fails if any ``self.call_backend(...)`` in ``model/twinkle_handlers.py``
passes ``adapter_name`` as the bare local (i.e. unmapped). Passing the raw name
*positionally* (as ``add_adapter_to_model`` does when creating a tenant adapter)
is intentionally not a keyword and is therefore not flagged.
"""
from __future__ import annotations

import ast
import pathlib

import twinkle

_HANDLERS = pathlib.Path(twinkle.__file__).resolve().parent / 'server' / 'model' / 'twinkle_handlers.py'


def _is_call_backend(func: ast.AST) -> bool:
    """True for ``self.call_backend`` (Attribute ``call_backend`` on Name ``self``)."""
    return (isinstance(func, ast.Attribute) and func.attr == 'call_backend' and isinstance(func.value, ast.Name)
            and func.value.id == 'self')


def _is_mapped(value: ast.AST) -> bool:
    """True when the ``adapter_name`` value is wrapped by ``self.resolve_model_adapter_name(...)``."""
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == 'resolve_model_adapter_name')


def _unmapped_adapter_name_calls(tree: ast.AST) -> list[tuple[int, str]]:
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_call_backend(node.func):
            continue
        for kw in node.keywords:
            if kw.arg != 'adapter_name':
                continue
            # Bare local ``adapter_name`` is the tenant-scoped name and must be mapped.
            if isinstance(kw.value, ast.Name) and kw.value.id == 'adapter_name' and not _is_mapped(kw.value):
                offenders.append((node.lineno, ast.unparse(kw.value)))
    return offenders


def test_model_backend_calls_map_adapter_name():
    tree = ast.parse(_HANDLERS.read_text(), filename=str(_HANDLERS))
    offenders = _unmapped_adapter_name_calls(tree)
    assert not offenders, ('These self.call_backend(...) sites pass the raw tenant adapter_name instead of '
                           f'self.resolve_model_adapter_name(adapter_name): {offenders}')


def test_checker_detects_unmapped_adapter_name():
    source = ('async def route(self, body, adapter_name, token):\n'
              '    await self.call_backend(self.model.forward, inputs=[], adapter_name=adapter_name)\n')
    assert _unmapped_adapter_name_calls(ast.parse(source)) == [(2, 'adapter_name')]


def test_checker_allows_mapped_and_positional():
    source = (
        'async def route(self, body, adapter_name, token):\n'
        '    await self.call_backend(self.model.forward, adapter_name=self.resolve_model_adapter_name(adapter_name))\n'
        '    await self.call_backend(self.model.add_adapter_to_model, adapter_name, config)\n')
    assert _unmapped_adapter_name_calls(ast.parse(source)) == []
