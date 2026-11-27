# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint storage media: local disk, OSS and S3, behind local paths.

The medium layer never serializes. It hands out LOCAL paths that the
model-side serialization code (safetensors / torch / ``save_pretrained`` /
DeepSpeed / mcore dist_checkpointing, all path APIs) uses unchanged:

- Local paths pass through unchanged; every primitive here is the os/shutil
  call the code used directly.
- ``oss://`` / ``s3://`` reads materialize into a persistent local cache
  (``to_local``; mmap-backed tensors must outlive the read). A cached object
  whose remote version changed re-downloads, so a bucket path overwritten
  after an earlier run never serves the old bytes.
- ``oss://`` / ``s3://`` writes serialize into a private staging directory
  whose flush uploads every staged file, with the checkpoint's completion
  markers LAST, so a partially uploaded directory never looks complete to
  a resume. ``write_dir`` is the single-writer ``with`` form; the multi-rank
  ``save`` path uses ``write_lease`` + ``commit`` (flush, barrier, markers,
  rotation) and defers an async save's seal to a join on the training loop.

Remote filesystems go through fsspec: ``ossfs`` for ``oss://`` (install
``ossfs``), ``s3fs`` for ``s3://`` (install ``s3fs``). Both
are imported lazily on first use, so this module never disturbs the base
import chain. Credentials come from the environment variables the
filesystems themselves read; they are never logged.
"""
import hashlib
import os
import shutil
import tempfile
from abc import ABC
from contextlib import contextmanager
from typing import Callable, Iterator, NamedTuple, Optional, Sequence

__all__ = ['OSSStorage', 'S3Storage', 'Storage', 'to_local', 'write_dir', 'local_run_dir', 'is_remote_url']

_REMOTE_SCHEMES = ('oss://', 's3://')


def is_remote_url(path) -> bool:
    """Whether ``path`` names an object-storage location rather than a local path."""
    return isinstance(path, str) and path.startswith(_REMOTE_SCHEMES)


class StorageEntry(NamedTuple):
    """One directory listing entry; ``path`` is the full path/URL, ``mtime_ns`` orders rotation."""

    name: str
    path: str
    is_dir: bool
    mtime_ns: int


class WriteLease:
    """A local directory to serialize into, flushed by ``flush_files`` / ``upload_markers``.

    ``path`` is what the serialization code writes to: the real target for a
    local destination, a private staging copy for a remote one. ``markers``
    are lease-relative files a resume treats as the checkpoint's completion
    markers; they upload LAST (``upload_markers``), after every rank's
    ``flush_files`` landed, so a partially uploaded directory never looks
    complete.
    """

    def __init__(self, path: str, target: str, markers: Sequence[str] = (),
                 storage: Optional['Storage'] = None):
        self.path = path
        self.target = target
        self.markers = tuple(markers)
        #: the storage that created the lease; every flush below no-ops on its ``remote`` flag
        self.storage = storage
        #: zero-arg callable the owner may set, run after a multi-rank flush seals
        #: (the training save uses it for save_total_limit rotation); None runs nothing
        self.rotate: Optional[Callable[[], None]] = None

    def flush_files(self) -> None:
        """Upload the staged non-marker files (no-op locally)."""
        if self.storage.remote:
            self.storage._upload_lease(self, markers=False)

    def upload_markers(self) -> None:
        """Upload the completion markers last, sealing the checkpoint (no-op locally)."""
        if self.storage.remote:
            self.storage._upload_lease(self, markers=True)

    def seal(self) -> None:
        """Seal a save: remote flush ordering first, then rotation (both media).

        The remote machinery (barrier, marker upload, staging cleanup) is shared
        by the synchronous ``commit`` and the deferred async seal -- the background
        tail cannot touch the training collectives, so both callers hold the
        barrier here, on a thread that may. Every participating rank must call it,
        or the barrier hangs. ``rotate`` then runs for local and remote alike:
        the training save sets it, so save_total_limit rotation rides the seal
        for sync and async saves uniformly.
        """
        if self.storage.remote:
            import torch.distributed as dist

            try:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                self.upload_markers()
            except BaseException:
                self.abort()
                raise
            self.discard()
        self.rotate and self.rotate()

    def commit(self) -> None:
        """Flush this rank's files, then seal: barrier, markers, cleanup, rotation.

        The synchronous multi-rank save's single verb (every rank calls it; the
        barrier inside seal orders all files before the markers). Local leases
        skip the remote machinery; rotation still runs through ``seal``.
        """
        self.flush_files()
        self.seal()

    def discard(self) -> None:
        """Delete the staging directory after a successful flush (no-op locally)."""
        if self.storage.remote:
            shutil.rmtree(self.path, ignore_errors=True)

    def abort(self) -> None:
        """Best-effort cleanup after a failed flush: the uploaded half and the staged one."""
        if not self.storage.remote:
            return
        try:
            self.storage.delete_tree(self.target)
        except Exception:  # noqa: BLE001 -- cleanup must not mask the flush error
            pass
        self.discard()


class Storage(ABC):
    """Medium primitives for checkpoint directories; serialization stays with the model."""

    remote: bool = False

    @staticmethod
    def from_url(path: str) -> 'Storage':
        """Dispatch on the path's scheme: ``oss://`` / ``s3://`` remote, anything else local."""
        if is_remote_url(path):
            return OSSStorage() if path.startswith('oss://') else S3Storage()
        return LocalDiskStorage()

    # -- write side ----------------------------------------------------------------
    def write_lease(self, path: str, markers: Sequence[str] = ()) -> WriteLease:
        """A local directory to serialize ``path`` into; see :class:`WriteLease`."""
        raise NotImplementedError

    # -- read side -----------------------------------------------------------------
    @contextmanager
    def read_dir(self, path: str, require: Sequence[str] = ()) -> Iterator[str]:
        """Yield a local directory holding the path's files.

        ``require`` names lease-relative objects that must exist first: the
        completion markers whose absence means an incomplete checkpoint.
        """
        raise NotImplementedError

    def download_file(self, path: str) -> str:
        """A local path holding the single file's current bytes."""
        raise NotImplementedError

    # -- control-plane primitives (rotation, resume probes) ---------------------------
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def is_dir(self, path: str) -> bool:
        raise NotImplementedError

    def iterdir(self, path: str) -> list:
        raise NotImplementedError

    def delete_tree(self, path: str) -> None:
        raise NotImplementedError

    def same_dir(self, entry_path: str, other_path: str) -> bool:
        """Whether two directory paths name the same directory (rotation's current-save guard)."""
        raise NotImplementedError

    def copy_into_local(self, source: str, local_target: str) -> None:
        """Copy ``source`` (full path, any medium) into a local target; a missing source is a no-op."""
        raise NotImplementedError


class LocalDiskStorage(Storage):
    """Native pass-through: every primitive is the os/shutil call the code used directly."""

    def write_lease(self, path: str, markers: Sequence[str] = ()) -> WriteLease:
        # The target path IS the lease; the caller's own makedirs calls create it,
        # exactly as before this abstraction existed.
        return WriteLease(path, path, storage=self)

    @contextmanager
    def read_dir(self, path: str, require: Sequence[str] = ()) -> Iterator[str]:
        for marker in require:
            if not os.path.exists(os.path.join(path, marker)):
                raise FileNotFoundError(
                    f'{os.path.join(path, marker)} is missing: the checkpoint is incomplete '
                    '(a save still in flight, or a partial write its markers never sealed).')
        yield path

    def download_file(self, path: str) -> str:
        return path

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def is_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def iterdir(self, path: str) -> list:
        entries = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) or entry.is_file(follow_symlinks=False):
                    entries.append(StorageEntry(
                        name=entry.name,
                        path=entry.path,
                        is_dir=entry.is_dir(follow_symlinks=False),
                        mtime_ns=entry.stat(follow_symlinks=False).st_mtime_ns,
                    ))
        return entries

    def delete_tree(self, path: str) -> None:
        shutil.rmtree(path)

    def same_dir(self, entry_path: str, other_path: str) -> bool:
        return os.path.realpath(entry_path) == os.path.realpath(other_path)

    def copy_into_local(self, source: str, local_target: str) -> None:
        if os.path.isfile(source) and os.path.realpath(source) != os.path.realpath(local_target):
            shutil.copy2(source, local_target)


def to_local(path: str, require: Sequence[str] = ()) -> str:
    """Resolve an ``oss://`` / ``s3://`` path into its persistent local cache directory.

    Local paths pass through unchanged. The cache persists (mmap-backed tensors must
    outlive the read; re-reads skip the download), so the returned path stays valid
    after this call returns.
    """
    storage = Storage.from_url(path)
    with storage.read_dir(path, require=require) as local:
        return local


def local_run_dir(path: str) -> str:
    """The local directory for run-level logs (metrics, summaries, eval/rollout dumps) under ``path``.

    Local paths pass through unchanged. An ``oss://`` / ``s3://`` output dir is a
    checkpoint store; its run logs, which are appended in place, stay on local disk
    under the remote cache home instead of a bogus ``./oss:`` tree.
    """
    if not is_remote_url(path):
        return path
    return Storage.from_url(path)._cache_dir(path, leaf='runlevel')


@contextmanager
def write_dir(path: str, markers: Sequence[str] = ()) -> Iterator[str]:
    """``with write_dir(target) as local:`` serialize into ``local``; it lands at ``target``.

    The single-writer form: ``local`` is the real target locally (zero change;
    the directory is created here, like the ``makedirs`` the local callers used
    to run themselves), a staging copy remotely (uploaded on exit, markers last,
    staging deleted). For master-only sidecars. The multi-rank training ``save``
    cannot use this: its marker upload must wait on a barrier only the training
    loop can hold, so it uses ``Storage.write_lease`` + ``commit`` instead.
    """
    storage = Storage.from_url(path)
    lease = storage.write_lease(path, markers=markers)
    os.makedirs(lease.path, exist_ok=True)
    try:
        yield lease.path
        lease.flush_files()
        lease.upload_markers()
    finally:
        # Staging only, never ``abort``: the target is usually shared (a sidecar into a
        # checkpoint other writers sealed, args.json into the whole output dir), so a
        # failed sidecar must not delete what it was added to.
        lease.discard()


class _FsspecStorage(Storage):
    """The shared fsspec implementation; subclasses only name the protocol."""

    remote = True
    #: scheme prefix, e.g. ``'oss://'``; the fsspec protocol name is the scheme without ``://``
    scheme: str = ''
    #: the distribution providing the filesystem, for the missing-dependency message
    package: str = ''

    @property
    def _fs(self):
        # Lazy: nothing imports ossfs/s3fs until a remote URL is actually used.
        import fsspec

        try:
            return fsspec.filesystem(self.scheme[:-3])
        except ImportError as e:
            raise ImportError(
                f'{self.scheme} paths need the {self.package} filesystem; '
                f'install it (e.g. `pip install {self.package}`) '
                'and retry.'
            ) from e

    def _fs_path(self, path: str) -> str:
        return path[len(self.scheme):]

    def _url(self, fs_path: str) -> str:
        return self.scheme + fs_path

    # -- write side --------------------------------------------------------------
    def write_lease(self, path: str, markers: Sequence[str] = ()) -> WriteLease:
        staging = tempfile.mkdtemp(prefix='twinkle-remote-save-')
        return WriteLease(staging, path, markers=markers, storage=self)

    def _upload_lease(self, lease: WriteLease, *, markers: bool) -> None:
        # Every rank uploads whatever its own staging holds: the markers live on the
        # rank that writes them (the master), so a rank without them uploads none.
        if not os.path.isdir(lease.path):
            return
        selected = []
        for root, _, files in os.walk(lease.path):
            for name in files:
                local = os.path.join(root, name)
                rel = os.path.relpath(local, lease.path)
                if (rel in lease.markers) != markers:
                    continue
                selected.append((local, rel))
        if not selected:
            return
        fs = self._fs
        remote_root = self._fs_path(lease.target)
        for local, rel in selected:
            fs.put(local, f'{remote_root}/{rel}')

    # -- read side ---------------------------------------------------------------
    def _cache_dir(self, path: str, *, leaf: str) -> str:
        # One cache per URL, shared by every process on the node: downloads land through
        # temp file + os.replace under a per-URL file lock, so ranks never see torn files
        # and the first one downloads while the rest reuse it.
        root = os.environ.get('TWINKLE_REMOTE_CACHE_DIR') or os.path.expanduser('~/.cache/twinkle-remote-cache')
        key = f'{self.scheme[:-3]}-{hashlib.sha1(path.encode()).hexdigest()}'
        cache = os.path.join(root, key, leaf)
        os.makedirs(cache, exist_ok=True)
        return cache

    @contextmanager
    def read_dir(self, path: str, require: Sequence[str] = ()) -> Iterator[str]:
        from filelock import FileLock

        fs_path = self._fs_path(path).rstrip('/')
        fs = self._fs
        for marker in require:
            if not fs.exists(f'{fs_path}/{marker}'):
                raise FileNotFoundError(
                    f'{os.path.join(path, marker)} is missing: the checkpoint is incomplete '
                    '(a save still in flight, or a partial upload its markers never sealed).')
        cache = self._cache_dir(path, leaf='tree')
        with FileLock(f'{cache}.lock'):
            if not fs.isdir(fs_path):
                # A single object (a media file, one jsonl) materializes as one file, so
                # the yielded local path replaces the URL itself.
                if not fs.exists(fs_path):
                    raise FileNotFoundError(f'{path}: no such object.')
                local = self._materialize(fs, fs_path, fs.info(fs_path), cache, os.path.basename(fs_path))
            else:
                wanted = set()
                for obj, info in (fs.find(fs_path, detail=True) or {}).items():
                    rel = obj.rstrip('/')[len(fs_path):].lstrip('/')
                    if rel and info.get('type') != 'directory':
                        wanted.add(rel)
                        self._materialize(fs, obj, info, cache, rel)
                # Drop what the remote no longer holds (a re-saved checkpoint with fewer
                # shards, a dataset dir that lost a file): loaders glob the whole dir.
                for root, _, files in os.walk(cache):
                    for name in files:
                        local_file = os.path.join(root, name)
                        if os.path.relpath(local_file, cache) not in wanted:
                            os.unlink(local_file)
                local = cache
        yield local

    @staticmethod
    def _object_stamp(info: dict) -> str:
        """One object's remote version (etag / mtime / size), for cache invalidation."""
        stamp = info.get('ETag') or info.get('etag') or info.get('LastModified') or info.get('mtime') or 0
        if hasattr(stamp, 'timestamp'):
            stamp = stamp.timestamp()
        return f'{stamp}:{info.get("size")}'

    def _materialize(self, fs, obj: str, info: dict, cache: str, rel: str) -> str:
        """Download ``obj`` to ``cache/rel`` unless the cached copy matches the remote version.

        The version stamp lives in a sibling ``{cache}.stamps`` tree (never beside the
        data: the cache dir is handed to loaders and hub pushes as the checkpoint itself),
        so a bucket path overwritten after an earlier run re-downloads instead of serving
        the old bytes. The temp file sits there too and lands via ``os.replace``: no torn
        file is ever visible, and mmap readers keep their old inode.
        """
        target = os.path.join(cache, rel)
        stamp_file = os.path.join(f'{cache}.stamps', rel)
        stamp = self._object_stamp(info)
        if os.path.exists(target) and os.path.exists(stamp_file):
            with open(stamp_file) as f:
                if f.read() == stamp:
                    return target
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.makedirs(os.path.dirname(stamp_file), exist_ok=True)
        tmp = f'{stamp_file}.{os.getpid()}.part'
        try:
            fs.get(obj, tmp)
            os.replace(tmp, target)
            with open(tmp, 'w') as f:
                f.write(stamp)
            os.replace(tmp, stamp_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target

    def download_file(self, path: str) -> str:
        from filelock import FileLock

        fs = self._fs
        fs_path = self._fs_path(path).rstrip('/')
        cache = self._cache_dir(path, leaf='files')
        with FileLock(f'{cache}.lock'):
            return self._materialize(fs, fs_path, fs.info(fs_path), cache, os.path.basename(fs_path))

    # -- control-plane primitives ---------------------------------------------------
    def exists(self, path: str) -> bool:
        return self._fs.exists(self._fs_path(path))

    def is_dir(self, path: str) -> bool:
        return self._fs.isdir(self._fs_path(path))

    def iterdir(self, path: str) -> list:
        fs_path = self._fs_path(path)
        entries = []
        for info in self._fs.ls(fs_path, detail=True):
            raw = info['name'].rstrip('/')
            name = raw.rsplit('/', 1)[-1]
            # fsspec listing names carry no scheme; re-attach ours so callers can
            # feed entry.path back into this storage's path-taking primitives.
            full = raw if raw.startswith(self.scheme) else self._url(raw)
            is_dir = info.get('type') == 'directory'
            stamp = info.get('LastModified') or info.get('mtime') or 0
            if hasattr(stamp, 'timestamp'):
                stamp = stamp.timestamp()
            entries.append(StorageEntry(name=name, path=full, is_dir=is_dir, mtime_ns=int(stamp * 1e9)))
        return entries

    def delete_tree(self, path: str) -> None:
        self._fs.rm(self._fs_path(path), recursive=True)

    def same_dir(self, entry_path: str, other_path: str) -> bool:
        return entry_path.rstrip('/') == other_path.rstrip('/')

    def copy_into_local(self, source: str, local_target: str) -> None:
        if self.exists(source):
            os.makedirs(os.path.dirname(local_target), exist_ok=True)
            with open(local_target, 'wb') as f:
                f.write(self._fs.cat_file(self._fs_path(source)))


class OSSStorage(_FsspecStorage):
    scheme = 'oss://'
    package = 'ossfs'


class S3Storage(_FsspecStorage):
    scheme = 's3://'
    package = 's3fs'
