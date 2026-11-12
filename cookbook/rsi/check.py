# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reject a check script that only its own author can pass.

Handed to the challenger as ``brittle_check_fn``: it is python-specific, and the
challenger is not. The reason travels back to the model through the same retry
path a failing assertion uses, because the defect is the same kind.
"""
import ast
from typing import Optional

__all__ = ['brittle_check_reason']

# Two rules the check-script prompt already states -- no equality on a script's
# source text, no byte count or checksum on a binary -- were broken by 9 and 8 of
# 41 measured tasks respectively, so stating them a third time is not the fix. A
# check that pins the exact source of a .py rejects every equivalent solution,
# and one that pins a .png's byte count rejects every matplotlib version; both
# make a task nobody but the author can pass.
_SIZE_OR_HASH_NAMES = ('getsize', 'st_size', 'sha256', 'sha1', 'md5', 'hexdigest',
                       'digest')
# What makes a string python rather than data. Checked instead of "is it long and
# multi-line", because the contents of a csv or a json file are legitimately
# asserted verbatim -- the statement handed those to the solver -- while the text
# of a script never is.
_LOOKS_LIKE_PYTHON = ('import ', 'def ', 'print(', 'with open(', 'if __name__')


def brittle_check_reason(script: str) -> Optional[str]:
    """Why this check script would reject a correct solution, or None.

    Returned text goes back to the model through the same retry path a failing
    assertion uses, because the defect is the same kind: an assertion that does
    not hold for solutions other than the one in front of it.

    Read off the syntax tree rather than matched as text. Both defects survive
    patterns easily: source equality reads the file into a name first
    (``c = f.read()``, then ``assert c == '...'``) so nothing sits between
    ``open()`` and ``==``, and a size check can put the call either around the
    name (``getsize("a.png")``) or after it.

    Python throughout -- the tree, the marker words, the stdlib names below. There
    is no language-neutral version of this: another language keeps the two rules
    but rewrites the whole body, which is why the challenger takes it as
    ``brittle_check_fn`` rather than calling it directly.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # Unparseable means it cannot run either, so let the sandbox report it.
        return None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare)
                and any(isinstance(o, ast.Eq) for o in node.ops)):
            continue
        for side in [node.left] + list(node.comparators):
            if not (isinstance(side, ast.Constant) and isinstance(side.value, str)):
                continue
            if any(m in side.value for m in _LOOKS_LIKE_PYTHON):
                return ('AssertionError: this check compares a file against the '
                        'full text of a python script with ==, which only the '
                        'exact script you wrote can pass. Assert what running '
                        'that script produces instead.')
    # A byte count or a checksum compared for equality. Not restricted to
    # binary suffixes: the prompt says "NEVER check a file size in bytes" about
    # any file, and keying on a suffix list let
    # ``getsize('data.mat') == 264`` through. Only equality against a literal is
    # a defect -- ``getsize(f) > 0`` is a fine way to say "not empty".
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare)
                and any(isinstance(o, ast.Eq) for o in node.ops)):
            continue
        sides = [node.left] + list(node.comparators)
        has_literal = any(isinstance(s, ast.Constant)
                          and isinstance(s.value, (int, float, str))
                          and not isinstance(s.value, bool) for s in sides)
        if not has_literal:
            continue
        for side in sides:
            names = {n.attr for n in ast.walk(side) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(side) if isinstance(n, ast.Name)}
            hit = names & set(_SIZE_OR_HASH_NAMES)
            if hit:
                what = ('a checksum' if hit - {'getsize', 'st_size'}
                        else 'a byte count')
                return (f'AssertionError: this check pins {what} of a file, and '
                        'correct solutions differ there. Assert what can be read '
                        'out of the file instead -- its structure, or the values '
                        'inside it.')
    # Comparing raw bytes of a file: same defect, different spelling.
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare)
                and any(isinstance(o, ast.Eq) for o in node.ops)):
            continue
        for side in [node.left] + list(node.comparators):
            if isinstance(side, ast.Constant) and isinstance(side.value, bytes):
                return ('AssertionError: this check compares the raw bytes of a '
                        'file, and correct solutions differ there. Assert what '
                        'can be read out of it instead.')
    return None
