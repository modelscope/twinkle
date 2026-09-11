# Copyright (c) ModelScope Contributors. All rights reserved.
"""Describing a workspace to a model, by listing it from inside.

A caller that has to write a check against what an episode left behind needs to
be told what is in there. The listing is produced by a script run *in* the
environment rather than off this process's filesystem, so a local directory and a
directory inside a microVM describe themselves the same way -- which is what lets
one check script be written against either.

File bodies go out byte for byte and the trailing-newline count is stated: a
listing that tidies up is not ground truth, and a check written against a tidied
listing fails on the very state it was written from.
"""
from typing import Any, Sequence, Tuple

# Written to stdout in one write: the caller reads the whole stream, and a partial
# line would read as a truncated file body.
_SNAPSHOT_SCRIPT = '''
import os, sys
root, max_files, per_file, budget, skip = {root!r}, {max_files}, {per_file}, {budget}, {skip!r}
rows = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in skip]
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        try:
            rows.append((os.path.relpath(path, root), os.path.getsize(path), path))
        except OSError:
            pass
rows.sort()
rows = rows[:max_files]
lines = ['%s %d' % (rel, size) for rel, size, _ in rows]
for rel, _, path in rows:
    if budget <= 0:
        break
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read(per_file + 1)
    except (OSError, UnicodeDecodeError):
        continue  # binary or unreadable: the listing already names it
    if '\\x00' in text:
        continue
    body = text[:per_file]
    budget -= len(body)
    trailing = len(body) - len(body.rstrip('\\n'))
    if len(text) > len(body):
        suffix = ' (first %d bytes)' % per_file
    elif trailing == 0:
        suffix = ' (no newline at end)'
    else:
        suffix = ' (ends with %d newline character(s))' % trailing
    # One trailing newline is dropped because the join puts it back. What must not
    # happen is stripping them all: the header states the count, and a body shown
    # shorter than the size column contradicts it.
    lines += ['', '--- %s%s ---' % (rel, suffix), body[:-1] if body.endswith('\\n') else body]
sys.stdout.write('\\n'.join(lines).strip())
'''


def list_workspace(env: Any, *, max_files: int, per_file: int, budget: int,
                   skip: Sequence[str]) -> Tuple[str, str]:
    """The environment's workspace as ``(listing, error)``; both empty when it has none.

    Takes anything with a ``workspace`` and a ``run_script``, so it is as usable
    from an environment written outside this package as from one written in it.

    Args:
        env: the environment to look inside.
        max_files: how many files to name, shortest path first.
        per_file: how many bytes of each file body to show.
        budget: total bytes of file bodies, across all of them.
        skip: directory names not to walk into.

    Returns:
        ``(listing, error)``. The two are kept apart because a listing that says
        "empty" when it means "I could not look" produces tasks whose only true
        assertion is that nothing happened.
    """
    root = getattr(env, 'workspace', None)
    if not root:
        return '', ''
    exit_code, output = env.run_script(
        _SNAPSHOT_SCRIPT.format(root=root, max_files=max_files, per_file=per_file, budget=budget, skip=tuple(skip)))
    if exit_code != 0:
        # Reported, not raised: the caller rejects the episode on an error string,
        # and a listing that failed says nothing about the episode.
        return '', f'could not list {root}: {output}'
    return output.strip(), ''
