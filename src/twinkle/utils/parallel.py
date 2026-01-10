import os
import shutil
from contextlib import contextmanager

from datasets.utils._filelock import FileLock

shutil.rmtree('.locks', ignore_errors=True)
os.makedirs('.locks', exist_ok=True)


@contextmanager
def processing_lock(lock_file: str):
    lock = FileLock(os.path.join('.locks', f"{lock_file}.lock"))

    if lock.acquire(blocking=False):
        try:
            yield
        finally:
            lock.release()
    else:
        lock.acquire(blocking=True)
        lock.release()
        yield