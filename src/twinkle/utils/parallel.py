# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import inspect
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from datasets.utils.filelock import FileLock
from datetime import timedelta

_LOCK_DIR = '.locks'
os.makedirs(_LOCK_DIR, exist_ok=True)

# Coordination store: derived from MASTER_PORT so every rank of a group agrees on it without
# talking to each other first. A collision with another job's MASTER_PORT surfaces as a bind
# error at store creation, not as silent misbehaviour.
_COORD_PORT_OFFSET = int(os.environ.get('TWINKLE_COORD_PORT_OFFSET', 10091))
_COORD_TIMEOUT = timedelta(seconds=int(os.environ.get('TWINKLE_COORD_TIMEOUT', 21600)))
_store = None
_store_failed = False
# One sequence per lock name: the same key is locked repeatedly over a run (encode, then check,
# ...) and each round needs its own flags, or round 2 finds round 1's flag set and every rank
# charges in at once.
_seq = defaultdict(int)


class LockPeerError(RuntimeError):
    """The rank that was supposed to do the work failed, so waiting for it is pointless."""


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


def _coord_ranks():
    """Rank/world-size of the group sharing one coordination store."""
    return int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))


def _is_local_master() -> bool:
    """Whether this rank writes for its node.

    Ray sets LOCAL_RANK to 0 for every worker (each actor owns its process), so under ray every
    rank reports True and the node tier collapses into 'global master first, then everyone' --
    which is the right degradation when the node layout is not knowable from the env.
    """
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def _node_index() -> int:
    """Index of this rank's node. torchrun exports GROUP_RANK; derive it otherwise."""
    group_rank = os.environ.get('GROUP_RANK')
    if group_rank is not None:
        return int(group_rank)
    rank, _ = _coord_ranks()
    return rank // (int(os.environ.get('LOCAL_WORLD_SIZE', 1)) or 1)


def _get_store():
    """Join the coordination store, or return None to fall back to the file lock.

    Rank 0 serves it and every other rank connects. This deliberately avoids torch.distributed's
    default process group: the store must be usable long before any backend is initialized, and a
    NCCL barrier here would put dataset preprocessing under the collective watchdog timeout.

    Note the store lives inside rank 0's process, so rank 0 has to outlive every other user of it.
    """
    global _store, _store_failed
    if _store is not None or _store_failed:
        return _store
    rank, _ = _coord_ranks()
    try:
        from torch.distributed import TCPStore
        # world_size=None keeps this off the startup path: nobody blocks waiting for peers, and a
        # client simply retries until rank 0 has its server up.
        _store = TCPStore(
            os.environ.get('MASTER_ADDR', '127.0.0.1'),
            int(os.environ.get('MASTER_PORT', 29500)) + _COORD_PORT_OFFSET,
            world_size=None,
            is_master=(rank == 0),
            timeout=_COORD_TIMEOUT,
            wait_for_workers=False)
    except Exception:  # noqa: BLE001
        # Port taken, torch too old, no network -- degrade to the file lock rather than fail a run
        # over a coordination detail.
        _store_failed = True
    return _store


def _use_store() -> bool:
    """Whether ranks can be ordered through the store instead of the file lock."""
    _, world_size = _coord_ranks()
    return world_size > 1 and 'MASTER_ADDR' in os.environ and _get_store() is not None


def _wait_for(store, flag):
    store.wait([flag], _COORD_TIMEOUT)
    if store.get(flag) != b'1':
        raise LockPeerError(f'The peer responsible for {flag} failed; refusing to read after it')


@contextmanager
def _ordered_by_store(key: str, sticky: bool):
    """Run the body on every rank, but ordered: global master, then node masters, then the rest."""
    store = _get_store()
    name = _sanitize_lock_name(key)
    if sticky:
        # The key already identifies the result (a repo id), and the work is idempotent, so one
        # flag serves the whole run: a rank arriving after the flag is set just proceeds.
        prefix = f'twinkle/lock/{name}'
    else:
        prefix = f'twinkle/lock/{name}/{_seq[name]}'
        _seq[name] += 1
    global_flag, node_flag = f'{prefix}/global', f'{prefix}/node{_node_index()}'
    rank, _ = _coord_ranks()
    is_global_master, is_local_master = rank == 0, _is_local_master()

    if not is_global_master:
        try:
            # Node masters queue behind the global master, the rest behind their own node master.
            _wait_for(store, global_flag if is_local_master else node_flag)
        except BaseException:
            if is_local_master:
                # Pass the failure down, instead of leaving our own node waiting on a flag that
                # will never be set.
                store.set(node_flag, b'0')
            raise

    ok = False
    try:
        yield
        ok = True
    finally:
        if is_local_master:
            # Publish even on failure: a crashed writer must not hang the readers.
            for flag in ([global_flag, node_flag] if is_global_master else [node_flag]):
                store.set(flag, b'1' if ok else b'0')


@contextmanager
def processing_lock(lock_file: str, sticky: bool = False):
    """Serialize one writer against many readers for the same resource.

    Two mechanisms, same semantics -- the body runs on every rank, it just stops running
    concurrently:

    1. With several ranks sharing a coordination store, ranks are ordered global master -> node
       masters -> everyone else. Preferred, because it needs no filesystem locking, which is what
       breaks on network storage.
    2. Otherwise (single process, or a lone ray actor) a file lock, where whichever process wins
       writes and the rest wait and then read.

    The store path assumes every rank of the group enters this the same number of times with the
    same key. That holds when the call sites are symmetric -- all ranks running the same script,
    or all actors of an ``execute='all'`` group -- and is why ``sticky`` exists for the cases
    where it does not.

    Args:
        lock_file: Identifies the resource being written.
        sticky: Treat the key as naming the result rather than the round, so a rank that arrives
            after the work is done proceeds instead of waiting for a fresh flag. Use it for
            idempotent, content-addressed work such as a download; leave it off for work that
            repeats under the same key, such as dataset preprocessing.
    """
    if _use_store():
        with _ordered_by_store(lock_file, sticky):
            yield
        return

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
