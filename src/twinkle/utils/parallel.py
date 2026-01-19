# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from contextlib import contextmanager

from datasets.utils._filelock import FileLock

shutil.rmtree('.locks', ignore_errors=True)
os.makedirs('.locks', exist_ok=True)

def acquire_lock(lock, blocking):
    try:
        lock.acquire(blocking=blocking)
        return True
    except Exception:
        return False


def release_lock(lock):
    lock.release()


def acquire_lock(lock, blocking):
    try:
        lock.acquire(blocking=blocking)
        return True
    except Exception:
        return False


def release_lock(lock):
    lock.release()


@contextmanager
def processing_lock(lock_file: str, timeout: float = 600.0):
    """Acquire a file lock for distributed-safe processing.
    
    Args:
        lock_file: Name of the lock file (will be sanitized).
        timeout: Maximum time to wait for lock acquisition in seconds.
    
    In distributed training, only rank 0 should process data while
    other ranks wait. This lock ensures that.
    """
    # Sanitize lock file name
    safe_name = lock_file.replace('/', '_').replace(':', '_').replace(' ', '_')
    lock_path = os.path.join(_locks_dir, f"{safe_name}.lock")
    lock = FileLock(lock_path, timeout=timeout)

    if acquire_lock(lock, False):
        try:
            yield
        finally:
            release_lock(lock)
    else:
        acquire_lock(lock, True)
        release_lock(lock)
        yield