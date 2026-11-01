# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the lock utility with real files, including independent processes."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / 'src/twinkle/utils/parallel.py'


def _load_parallel():
    # This utility does not require the training dependencies imported by utils.
    spec = importlib.util.spec_from_file_location('twinkle_parallel_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(cwd, script, env):
    loader = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('twinkle_parallel_test', {str(MODULE_PATH)!r})
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)
"""
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(loader) + textwrap.dedent(script)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='Requires enforced POSIX permissions')
@pytest.mark.parametrize('custom_home', [False, True])
def test_locks_work_from_read_only_directory(tmp_path, custom_home):
    cwd = tmp_path / 'read_only'
    cwd.mkdir()
    env = os.environ.copy()
    env.pop('TWINKLE_HOME', None)
    # A fresh interpreter honors TMPDIR when choosing the default location.
    env['TMPDIR'] = str(tmp_path)
    if custom_home:
        env['TWINKLE_HOME'] = str(tmp_path / 'runtime')
    env['TWINKLE_SESSION_ID'] = tmp_path.name
    cwd.chmod(0o555)
    try:
        result = _run(cwd, """
with parallel.processing_lock('dataset/name'):
    pass
assert parallel.try_claim_once('read-only-notification') is True
assert parallel.try_claim_once('read-only-notification') is False
""", env)
        assert result.returncode == 0, result.stderr
        assert not (cwd / '.locks').exists()
    finally:
        cwd.chmod(0o755)


def test_import_does_not_create_lock_directory(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    monkeypatch.setenv('TWINKLE_HOME', str(runtime))
    monkeypatch.chdir(tmp_path)
    _load_parallel()
    assert not runtime.exists()
    assert not (tmp_path / '.locks').exists()


def test_claim_is_shared_across_working_directories(tmp_path):
    env = os.environ.copy()
    env['TWINKLE_HOME'] = str(tmp_path / 'runtime')
    env['TWINKLE_SESSION_ID'] = 'first-session'
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    script = "print(parallel.try_claim_once('same-key', payload='payload', namespace='notify'))"
    result = _run(first, script, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'True'
    result = _run(second, script, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'False'
    env['TWINKLE_SESSION_ID'] = 'next-session'
    result = _run(second, script, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'True'


def test_processing_lock_waits_across_working_directories(tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_HOME', str(tmp_path / 'runtime'))
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    parallel = _load_parallel()
    child = None
    with parallel.processing_lock('same-dataset'):
        loader = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('twinkle_parallel_test', {str(MODULE_PATH)!r})
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)
print('ready', flush=True)
with parallel.processing_lock('same-dataset'):
    print('entered', flush=True)
"""
        child = subprocess.Popen(
            [sys.executable, '-c', textwrap.dedent(loader)], cwd=second,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            # Wait for module import before checking that the child blocks.
            import selectors
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=10), 'Child did not finish importing'
            assert child.stdout.readline().strip() == 'ready'
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert not selector.select(timeout=0.25), 'Child bypassed the held lock'
        except BaseException:
            child.kill()
            child.communicate()
            raise
    try:
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        assert stdout.strip() == 'entered'
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_processing_lock_releases_after_exception(tmp_path, monkeypatch):
    monkeypatch.setenv('TWINKLE_HOME', str(tmp_path / 'runtime'))
    monkeypatch.chdir(tmp_path)
    parallel = _load_parallel()
    with pytest.raises(ValueError):
        with parallel.processing_lock('dataset/name'):
            raise ValueError('processing failed')
    lock = parallel.FileLock(os.path.join(parallel._LOCK_DIR, 'dataset_name.lock'))
    assert parallel.acquire_lock(lock, blocking=False)
    parallel.release_lock(lock)
