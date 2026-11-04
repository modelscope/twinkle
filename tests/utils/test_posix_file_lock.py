import multiprocessing
import os
import pytest

from twinkle.utils.parallel import PosixFileLock


def _hold_lock(lock_path: str, acquired, release) -> None:
    with PosixFileLock(lock_path):
        acquired.set()
        if not release.wait(timeout=30):
            raise TimeoutError("test did not release file lock")


def _acquire_lock(lock_path: str, started, acquired) -> None:
    started.set()
    with PosixFileLock(lock_path):
        acquired.set()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file lock requires fcntl")
def test_posix_file_lock_serializes_processes(tmp_path):
    """A held advisory lock excludes an independent process until released."""
    context = multiprocessing.get_context("spawn")
    lock_path = str(tmp_path / "process.lock")
    first_acquired = context.Event()
    release_first = context.Event()
    second_started = context.Event()
    second_acquired = context.Event()
    first = context.Process(target=_hold_lock, args=(lock_path, first_acquired, release_first))
    second = context.Process(target=_acquire_lock, args=(lock_path, second_started, second_acquired))

    try:
        first.start()
        assert first_acquired.wait(timeout=30)

        second.start()
        assert second_started.wait(timeout=30)
        assert not second_acquired.wait(timeout=0.2)

        release_first.set()
        assert second_acquired.wait(timeout=30)
    finally:
        release_first.set()
        for process in (first, second):
            if process.pid is None:
                continue
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)

    assert first.exitcode == 0
    assert second.exitcode == 0
