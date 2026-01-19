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


@contextmanager
def processing_lock(lock_file: str):
    lock = FileLock(os.path.join('.locks', f"{lock_file}.lock"))

    if acquire_lock(lock, False):
        try:
            yield
        finally:
            release_lock(lock)
    else:
        acquire_lock(lock, True)
        release_lock(lock)
        yield