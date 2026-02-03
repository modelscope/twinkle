# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from contextlib import contextmanager

from datasets.utils.filelock import FileLock

os.makedirs('.locks', exist_ok=True)


def acquire_lock(lock: FileLock, blocking: bool):
    try:
        lock.acquire(blocking=blocking)
        return True
    except TimeoutError:
        return False


def release_lock(lock: FileLock):
    lock.release(force=True)


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
    lock: FileLock = FileLock(os.path.join('.locks', f"{lock_file}.lock")) # noqa

    if acquire_lock(lock, False):
        try:
            yield
        finally:
            release_lock(lock)
    else:
        acquire_lock(lock, True)
        release_lock(lock)
        yield