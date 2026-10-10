# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import inspect
import os
import re
from contextlib import contextmanager
from datasets.utils.filelock import FileLock

_LOCK_DIR = '.locks'
os.makedirs(_LOCK_DIR, exist_ok=True)


def _sanitize_lock_name(name: str) -> str:
    r"""Sanitize lock file name for cross-platform compatibility.

    Windows does not allow : / \ * ? " < > | in file names.
    """
    # Replace problematic characters with underscores
    return re.sub(r'[:/\\*?"<>|]', '_', name)


def acquire_lock(lock: FileLock, blocking: bool):
    try:
        if 'blocking' in inspect.signature(lock.acquire).parameters:
            lock.acquire(blocking=blocking)
        else:
            lock.acquire(timeout=(0 if not blocking else None))
        return True
    except TimeoutError:
        return False


def release_lock(lock: FileLock):
    lock.release(force=True)


def _get_session_token() -> str:
    """Return a stable token shared by all ranks in the same training run."""
    return os.environ.get('TWINKLE_SESSION_ID') or str(os.getppid())


def try_claim_once(key: str, *, payload: str = '', namespace: str = 'claim') -> bool:
    """Atomically claim a one-shot slot identified by ``key`` (single-winner).

    Stale claims left by a prior session (identified by a session token stored
    inside the sentinel file) are automatically evicted on first access, so
    no manual cleanup or import-time wipe is needed.

    Session token: ``TWINKLE_SESSION_ID`` env if set, else ``os.getppid()``
    (all torchrun ranks share the same parent; for ray, set the env in driver
    and workers inherit via ``RuntimeEnv``).

    Falls back to ``True`` on any filesystem error — callers should treat
    this as best-effort idempotency, never as a correctness barrier.
    """
    try:
        session = _get_session_token()
        digest = hashlib.md5(_sanitize_lock_name(key).encode('utf-8')).hexdigest()[:16]
        os.makedirs(_LOCK_DIR, exist_ok=True)
        path = os.path.join(_LOCK_DIR, f'{namespace}_{digest}.once')
        return _try_create_claim(path, session, payload)
    except Exception:  # noqa: BLE001
        return True


def _try_create_claim(path: str, session: str, payload: str) -> bool:
    # At most one retry after evicting a stale claim.
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f'{session}\n{payload}'.encode())
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            try:
                with open(path, encoding='utf-8') as f:
                    stored = f.readline().strip()
                if stored == session:
                    return False  # same session, genuine loser
                os.unlink(path)  # stale from prior run → evict
            except FileNotFoundError:
                continue  # another process evicted, retry
            except Exception:  # noqa: BLE001
                return False
    return True


class PosixFileLock:
    """POSIX advisory file lock with persistent fd for repeated acquire/release.

    Fork-safe: reopens its fd lazily when used from a child process so each
    worker owns its own descriptor.
    """

    def __init__(self, path: str):
        import fcntl
        self._path = path
        self._fcntl = fcntl
        self._fd = open(path, 'w')
        self._pid = os.getpid()

    def _ensure_fd(self):
        # After fork, child must reopen so it doesn't share parent's fd state.
        pid = os.getpid()
        if pid != self._pid:
            self._fd = open(self._path, 'w')
            self._pid = pid

    def acquire(self):
        self._ensure_fd()
        self._fcntl.flock(self._fd, self._fcntl.LOCK_EX)

    def release(self):
        self._fcntl.flock(self._fd, self._fcntl.LOCK_UN)

    def close(self):
        self._fd.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def __getstate__(self):
        return {'_path': self._path}

    def __setstate__(self, state):
        import fcntl
        self._path = state['_path']
        self._fcntl = fcntl
        self._fd = open(self._path, 'w')
        self._pid = os.getpid()


@contextmanager
def processing_lock(lock_file: str):
    """A file lock to prevent parallel operations to one file.

    This lock is specially designed for the scenario that one writing and multiple reading, for example:
    1. Download model
    2. Preprocess a dataset and generate cache files

    Firstly, it will try to acquire the lock, only one process will win and do the writing,
        other processes fall to `acquire_lock(lock, True)`

    After the writing process finishes the job, other processes will acquire and
        release immediately to do parallel reading.

    Args:
        lock_file: The lock file.
    Returns:

    """
    lock_name = _sanitize_lock_name(lock_file)
    lock: FileLock = FileLock(os.path.join(_LOCK_DIR, f'{lock_name}.lock'))  # noqa

    if acquire_lock(lock, False):
        try:
            yield
        finally:
            release_lock(lock)
    else:
        acquire_lock(lock, True)
        release_lock(lock)
        yield
