# Copyright (c) ModelScope Contributors. All rights reserved.
"""Parallelize torch DCP ``FileSystemReader.read_data`` for faster mcore checkpoint resume.

torch's stock ``FileSystemReader.read_data`` (``torch.distributed.checkpoint``) reads the load plan
strictly single-threaded -- two nested loops that ``torch.load`` each shard item one by one. Restoring
a large mcore optimizer/RNG checkpoint (the ``iter_*`` directory written by ``dist_checkpointing`` with
the default ``torch_dist`` backend) is therefore I/O-bound and shows no progress.

This patch replaces ``read_data`` with a thread-pool version: it splits ``plan.items`` round-robin
across ``MCORE_READER_MAX_WORKERS`` (default 16) workers and shows a ``Loading:`` progress bar. The
result is identical -- each worker calls the original ``read_data`` on a disjoint shard (disjoint
target tensors, independent read streams), and the same completed ``Future`` is returned. Mirrors
legacy swift's ``_patch_torch_FileSystemReader`` (swift/megatron/init.py).

Usage (persistent, like other global patches)::

    from twinkle.patch import apply_patch
    from twinkle.patch.torch_dcp_reader import TorchDCPParallelReaderPatch
    apply_patch(None, TorchDCPParallelReaderPatch())
"""
import os
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_twinkle_origin_dcp_read_data'
_SLICE_MARKER = '_twinkle_origin_dcp_slice_file'


class TorchDCPParallelReaderPatch(Patch):
    """Parallel + progress-bar ``FileSystemReader.read_data`` for faster DCP checkpoint load. Reversible."""

    def __call__(self, module=None, *args, **kwargs):
        from torch.distributed.checkpoint.filesystem import FileSystemReader
        if hasattr(FileSystemReader, _MARKER):
            return module

        import concurrent.futures
        from contextlib import contextmanager
        from copy import copy
        from torch.futures import Future
        from tqdm import tqdm

        origin_read_data = FileSystemReader.read_data
        origin_slice_file = FileSystemReader._slice_file
        max_workers = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

        @contextmanager
        def _patch_slice_file(prog_bar):
            # Advance the bar once per file slice actually read, then restore the original method.
            def _slice_file(self, *a, **k):
                prog_bar.update()
                return origin_slice_file(self, *a, **k)

            FileSystemReader._slice_file = _slice_file
            try:
                yield
            finally:
                FileSystemReader._slice_file = origin_slice_file

        def read_data(self, plan, planner):

            def _worker(plan_shard):
                origin_read_data(self, plan_shard, planner)

            items = plan.items
            n = min(max_workers, len(items)) or 1
            prog_bar = tqdm(total=len(items), dynamic_ncols=True, desc='Loading: ')
            with _patch_slice_file(prog_bar):
                with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                    futures = []
                    for i in range(n):
                        plan_shard = copy(plan)
                        plan_shard.items = items[i::n]  # round-robin, balances bytes across workers
                        futures.append(pool.submit(_worker, plan_shard))
                    concurrent.futures.wait(futures)
            prog_bar.close()
            fut: Future = Future()
            fut.set_result(None)
            return fut

        setattr(FileSystemReader, _MARKER, origin_read_data)
        setattr(FileSystemReader, _SLICE_MARKER, origin_slice_file)
        FileSystemReader.read_data = read_data
        logger.info('Patched torch DCP FileSystemReader.read_data for parallel checkpoint loading '
                    f'(max_workers={max_workers}).')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        from torch.distributed.checkpoint.filesystem import FileSystemReader
        origin = getattr(FileSystemReader, _MARKER, None)
        if origin is not None:
            FileSystemReader.read_data = origin
            delattr(FileSystemReader, _MARKER)
        if hasattr(FileSystemReader, _SLICE_MARKER):
            delattr(FileSystemReader, _SLICE_MARKER)
        return module
