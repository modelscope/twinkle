# Copyright (c) ModelScope Contributors. All rights reserved.
"""Architecture check (F007 / P006): no module imports its own package root.

A module under ``twinkle.server`` must not import one of its own *ancestor*
sub-packages (e.g. ``checkpoint/tinker.py`` importing ``twinkle.server.checkpoint``).
Such an import is a package-initialisation-order dependency: it re-enters the
ancestor's ``__init__`` while that ``__init__`` is still importing the child,
which is the module-level cycle this check forbids. Importing a *sibling* module
directly (``from .paths import ...``) or the top-level ``twinkle`` public API is
fine and excluded.

``import-linter``'s ``forbidden`` contract cannot express this (it treats the
source module as part of the forbidden package and reports the contract as kept),
so the check is written directly against grimp's static graph. grimp parses files
without importing the target package, so this stays safe to run in CI.
"""
from __future__ import annotations

import pathlib
import pytest

grimp = pytest.importorskip('grimp', reason='grimp is required for the package-root import architecture check')

import twinkle  # noqa: E402

_PACKAGE = 'twinkle.server'
_SRC = str(pathlib.Path(twinkle.__file__).resolve().parent.parent)


def _package_root_imports() -> list[dict]:
    """Return every import where a module imports one of its own ancestor sub-packages."""
    import sys
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    graph = grimp.build_graph('twinkle', include_external_packages=False)
    offenders: list[dict] = []
    for module in sorted(graph.modules):
        if not module.startswith(_PACKAGE + '.'):
            continue
        parts = module.split('.')
        ancestors = {'.'.join(parts[:i]) for i in range(1, len(parts))}
        ancestors = {a for a in ancestors if a.startswith(_PACKAGE + '.')}
        for imported in graph.find_modules_directly_imported_by(module):
            if imported in ancestors:
                for detail in graph.get_import_details(importer=module, imported=imported):
                    offenders.append({
                        'importer': module,
                        'imported': imported,
                        'line_number': detail.get('line_number'),
                        'line_contents': (detail.get('line_contents') or '').strip(),
                    })
    return offenders


def test_no_module_imports_its_own_package_root():
    offenders = _package_root_imports()
    assert not offenders, (
        'These modules import one of their own ancestor sub-packages (a package-init cycle); '
        'import the sibling module directly instead:\n'
        + '\n'.join(f"  {o['importer']} -> {o['imported']} (L{o['line_number']}): {o['line_contents']}"
                    for o in offenders))
