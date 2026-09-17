# Copyright (c) ModelScope Contributors. All rights reserved.
"""Static check: no direct Blocking_Backend_Call in ``src/twinkle/server/**``.

Spec: T3.7 / R2#7 / R2#8 / Property 2.

Asserts that backend methods are invoked only through ``call_backend``. The check
covers direct calls, aliases created with ``getattr``, and methods passed to generic
thread executors. The scan range is the server directory rather than a file list.

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


def _backend_method(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Attribute):
        return None
    owner = node.value
    if isinstance(owner, ast.Attribute) and owner.attr in ('model', 'sampler'):
        return owner.attr
    return None


def _getattr_backend_method(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'getattr':
        return None
    if not node.args:
        return None
    owner = node.args[0]
    if isinstance(owner, ast.Attribute) and owner.attr in ('model', 'sampler'):
        return owner.attr
    return None


def _is_backend_derived(node: ast.AST, backend_names: set[str]) -> bool:
    """True if *node* is (transitively) the backend or a value bound from it.

    Matches ``self.model`` / ``self.sampler`` and any attribute/subscript chain
    rooted at them, plus locals recorded in ``backend_names`` (e.g. from
    ``actors = self.sampler._actors`` then ``actor = actors[0]``). This catches the
    "bind the private actor list to a local, then call ``.remote()``" bypass that
    plain attribute-name matching cannot see.
    """
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr in ('model', 'sampler'):
                return True
            node = node.value
        else:
            node = node.value
    return isinstance(node, ast.Name) and node.id in backend_names


class _Collector(ast.NodeVisitor):

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.func_stack: list[str] = []
        self.backend_aliases: set[str] = set()
        self.backend_derived: set[str] = set()
        self.offenders: list[tuple[str, str, int, str]] = []

    def _visit_func(self, node: ast.AST) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Assign(self, node: ast.Assign) -> None:
        if _getattr_backend_method(node.value) is not None:
            self.backend_aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
        # Track locals bound (transitively) to the private backend actor list, so a
        # later ``.remote()`` on them is still counted as a backend call.
        if _is_backend_derived(node.value, self.backend_derived):
            self.backend_derived.update(target.id for target in node.targets if isinstance(target, ast.Name))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        owner = _backend_method(node.func)
        label = ast.unparse(node.func) if owner is not None else None
        if isinstance(node.func, ast.Name) and node.func.id in self.backend_aliases:
            owner = 'alias'
            label = node.func.id
        if isinstance(node.func, ast.Attribute) and node.func.attr in ('to_thread', 'run_in_executor') and node.args:
            escaped_owner = _backend_method(node.args[0]) or _getattr_backend_method(node.args[0])
            if escaped_owner is not None:
                owner = escaped_owner
                label = f'{ast.unparse(node.func)}({ast.unparse(node.args[0])})'
        # ``<actor>.<method>.remote(...)`` where <actor> traces back to the private
        # backend actor list bypasses call_backend just like a direct call.
        if (owner is None and isinstance(node.func, ast.Attribute) and node.func.attr == 'remote'
                and _is_backend_derived(node.func.value, self.backend_derived)):
            owner = 'remote'
            label = ast.unparse(node.func)
        if owner is not None:
            enclosing = self.func_stack[-1] if self.func_stack else '<module>'
            if (self.relpath, enclosing) not in BACKEND_CALL_EXEMPTIONS:
                self.offenders.append((self.relpath, enclosing, node.lineno, label or owner))
        self.generic_visit(node)


def test_no_direct_backend_call_in_server():
    offenders: list[tuple[str, str, int, str]] = []
    for path in _SERVER_ROOT.rglob('*.py'):
        relpath = str(path.relative_to(_SERVER_ROOT))
        collector = _Collector(relpath)
        collector.visit(ast.parse(path.read_text(), filename=str(path)))
        offenders.extend(collector.offenders)

    assert not offenders, ('Direct backend calls must go through call_backend (or be listed in '
                           f'backend_call_exemptions): {offenders}')


def test_exemptions_are_read_from_shared_file():
    assert ('sampler/twinkle_handlers.py', '_stream_queue') in BACKEND_CALL_EXEMPTIONS


def test_checker_detects_indirect_backend_calls():
    source = """
async def route(self):
    await asyncio.to_thread(self.model.save)
    unload = getattr(self.sampler, 'unload_adapter_paths')
    unload([])
"""
    collector = _Collector('example.py')
    collector.visit(ast.parse(source))
    assert len(collector.offenders) == 2


def test_checker_allows_call_backend():
    source = """
async def route(self):
    unload = getattr(self.sampler, 'unload_adapter_paths')
    await self.call_backend(unload, [])
"""
    collector = _Collector('example.py')
    collector.visit(ast.parse(source))
    assert collector.offenders == []


def test_checker_detects_aliased_remote_backend_call():
    # The sample_stream bypass shape: bind the private actor list to a local, then
    # call .remote() on an element. The guard must see this through the aliasing.
    source = """
async def sample_stream(self):
    actors = self.sampler._actors
    actor = actors[0]
    actor.sample_stream_to_queue.remote(q)
"""
    collector = _Collector('example.py')
    collector.visit(ast.parse(source))
    assert len(collector.offenders) == 1
    assert collector.offenders[0][1] == 'sample_stream'


def test_checker_ignores_remote_on_unrelated_object():
    # A .remote() on a handle not derived from self.model/self.sampler is not a
    # backend-boundary bypass and must not be flagged.
    source = """
async def route(self):
    handle = get_some_actor()
    handle.do.remote(1)
"""
    collector = _Collector('example.py')
    collector.visit(ast.parse(source))
    assert collector.offenders == []
