# Copyright (c) ModelScope Contributors. All rights reserved.
"""Static check: no direct Blocking_Backend_Call in ``src/twinkle/server/**``.

Spec: T3.7 / R2#7 / R2#8 / Property 2.

Asserts that no module under ``src/twinkle/server`` invokes ``self.model.<m>(...)``
or ``self.sampler.<m>(...)`` directly -- every such call must go through
``call_backend`` (the Blocking_Call_Boundary). The scan range is the directory
(not a file list), so a newly added handler file cannot silently escape it.

Allowed bypasses are read from the shared ``backend_call_exemptions`` module, which
this spec and ``server-request-lifecycle`` are meant to consume unchanged. This spec
can only verify that *its* check reads that file; the two-spec binding holds once
the lifecycle check is wired to the same file.
"""
from __future__ import annotations

import ast
import pathlib

import twinkle
from tests.server.static.backend_call_exemptions import BACKEND_CALL_EXEMPTIONS

_SERVER_ROOT = pathlib.Path(twinkle.__file__).resolve().parent / 'server'


def _is_self_backend_call(node: ast.Call) -> str | None:
    """Return 'model'/'sampler' if node is a direct self.model/self.sampler.<m>() call."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    owner = func.value  # the object the method is called on
    if (isinstance(owner, ast.Attribute) and owner.attr in ('model', 'sampler')
            and isinstance(owner.value, ast.Name) and owner.value.id == 'self'):
        return owner.attr
    return None


class _Collector(ast.NodeVisitor):

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.func_stack: list[str] = []
        self.offenders: list[tuple[str, str, int, str]] = []

    def _visit_func(self, node: ast.AST) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Call(self, node: ast.Call) -> None:
        owner = _is_self_backend_call(node)
        if owner is not None:
            enclosing = self.func_stack[-1] if self.func_stack else '<module>'
            if (self.relpath, enclosing) not in BACKEND_CALL_EXEMPTIONS:
                self.offenders.append((self.relpath, enclosing, node.lineno, f'self.{owner}.{node.func.attr}'))
        self.generic_visit(node)


def test_no_direct_backend_call_in_server():
    offenders: list[tuple[str, str, int, str]] = []
    for path in _SERVER_ROOT.rglob('*.py'):
        relpath = str(path.relative_to(_SERVER_ROOT))
        collector = _Collector(relpath)
        collector.visit(ast.parse(path.read_text(), filename=str(path)))
        offenders.extend(collector.offenders)

    assert not offenders, (
        'Direct backend calls must go through call_backend (or be listed in '
        f'backend_call_exemptions): {offenders}')


def test_exemptions_are_read_from_shared_file():
    # The shared file is the single source of allowed bypasses (R2#8).
    assert ('sampler/twinkle_handlers.py', '_stream_generator') in BACKEND_CALL_EXEMPTIONS
