# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check generated verifiers against author, empty and mutated workspaces."""
from __future__ import annotations

import ast
import json
import os
from typing import Dict, Any, List, Optional, Tuple


def declared_derived_paths(check_script: str) -> List[str]:
    try:
        tree = ast.parse(check_script)
    except SyntaxError as error:
        raise ValueError(f'invalid verifier syntax: {error}') from error
    value = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == 'DERIVED_OUTPUTS' for target in targets):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError) as error:
                    raise ValueError('DERIVED_OUTPUTS must be a literal list of paths') from error
                break
    if not isinstance(value, (list, tuple)) or not value or len(value) > 16:
        raise ValueError('DERIVED_OUTPUTS must list between 1 and 16 paths')
    paths = []
    for path in value:
        if not isinstance(path, str) or not path or os.path.isabs(path):
            raise ValueError('DERIVED_OUTPUTS paths must be non-empty relative strings')
        normalized = os.path.normpath(path)
        if normalized in {'.', '..'} or normalized.startswith('../'):
            raise ValueError(f'DERIVED_OUTPUTS path escapes the workspace: {path!r}')
        paths.append(normalized)
    if len(set(paths)) != len(paths):
        raise ValueError('DERIVED_OUTPUTS paths must be unique')
    return paths


def verify_check(env, check_script: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Require a verifier to reject an empty and at least one mutated workspace."""
    try:
        derived_paths = declared_derived_paths(check_script)
    except ValueError as error:
        return (None, str(error))
    program = r'''
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile

root = __ROOT__
check_source = __CHECK__
derived_paths = __DERIVED__
backup = tempfile.mkdtemp(prefix='rsi-verifier-')
state = os.path.join(backup, 'state')
checker = os.path.join(backup, 'check.py')
os.makedirs(state)
with open(checker, 'w', encoding='utf-8') as handle:
    handle.write(check_source)

skip = {'.ms_agent', '__pycache__', '.ipynb_checkpoints', '.git'}

def clear():
    os.makedirs(root, exist_ok=True)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def copy_contents(source, destination):
    for name in os.listdir(source):
        src = os.path.join(source, name)
        dst = os.path.join(destination, name)
        if os.path.isdir(src) and not os.path.islink(src):
            shutil.copytree(src, dst, symlinks=True)
        elif os.path.islink(src):
            os.symlink(os.readlink(src), dst)
        else:
            shutil.copy2(src, dst)

def restore():
    clear()
    copy_contents(state, root)

def run_check():
    completed = subprocess.run(
        [sys.executable, checker], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    return completed.returncode, completed.stdout[-1000:]

result = {}
backup_complete = False
try:
    copy_contents(root, state)
    backup_complete = True
    author_exit, author_output = run_check()
    result['author_state_passed'] = author_exit == 0
    result['author_state_exit'] = author_exit
    result['author_state_output'] = author_output
    restore()

    profile = []
    for dirpath, dirnames, filenames in os.walk(state):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, state)
            with open(path, 'rb') as handle:
                body = handle.read()
            profile.append({
                'path': rel,
                'size': len(body),
                'sha256': hashlib.sha256(body).hexdigest(),
                'media_type': mimetypes.guess_type(rel)[0] or 'application/octet-stream',
            })
    profile.sort(key=lambda item: item['path'])
    result['artifact_profile'] = profile
    result['snapshot_sha256'] = hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    profile_by_path = {item['path']: item for item in profile}
    missing_derived = [path for path in derived_paths if path not in profile_by_path]
    result['derived_paths'] = derived_paths
    result['missing_derived_paths'] = missing_derived

    clear()
    empty_exit, empty_output = run_check()
    result['empty_workspace_failed'] = empty_exit != 0
    result['empty_workspace_exit'] = empty_exit
    result['empty_workspace_output'] = empty_output

    mutation_checks = []
    for derived_path in derived_paths:
        artifact = profile_by_path.get(derived_path)
        if artifact is None:
            continue
        restore()
        path = os.path.join(root, artifact['path'])
        with open(path, 'rb') as handle:
            body = bytearray(handle.read())
        if body:
            body[len(body) // 2] ^= 1
        else:
            body.extend(b'RSI_MUTATION')
        with open(path, 'wb') as handle:
            handle.write(body)
        exit_code, output = run_check()
        mutation_checks.append({
            'path': artifact['path'],
            'verifier_failed': exit_code != 0,
            'exit': exit_code,
            'output': output,
        })
        if exit_code != 0:
            break
    result['mutation_checks'] = mutation_checks
    result['mutation_sensitive_path'] = next(
        (item['path'] for item in mutation_checks if item['verifier_failed']), None)
    result['passed'] = bool(
        result['author_state_passed']
        and result['empty_workspace_failed']
        and not result['missing_derived_paths']
        and result['mutation_sensitive_path']
    )
finally:
    if backup_complete:
        restore()
    shutil.rmtree(backup, ignore_errors=True)

print('__RSI_VERIFY__' + json.dumps(result, ensure_ascii=False, sort_keys=True))
if not result['passed']:
    raise SystemExit(1)
'''.replace('__DERIVED__', repr(derived_paths)).replace('__ROOT__',
                                                        repr(env.workspace)).replace('__CHECK__', repr(check_script))
    code, payload = env.run_script(program, 'python', timeout=120 * (len(derived_paths) + 2) + 30)
    marker = '__RSI_VERIFY__'
    position = payload.rfind(marker)
    if position < 0:
        return (None, f'verifier meta-check produced no result (exit {code}): {payload[-500:]}')
    try:
        result = json.loads(payload[position + len(marker):].splitlines()[0])
    except json.JSONDecodeError as error:
        return (None, f'verifier meta-check returned invalid JSON: {error}')
    if code != 0 or not result.get('passed'):
        return result, ('check must pass on the author workspace, fail on an empty workspace, '
                        'and fail after mutating at least one artifact; '
                        f"author_passed={result.get('author_state_passed')}, "
                        f"empty_failed={result.get('empty_workspace_failed')}, "
                        f"missing_derived={result.get('missing_derived_paths')}, "
                        f"mutation_sensitive_path={result.get('mutation_sensitive_path')}")
    return (result, '')
