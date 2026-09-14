"""Submission/collection adapter for asynchronous reward computation."""
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any, List

from .data import RewardItem, RewardResult, reorder_by_id, split_items
from .metrics import RewardLoopMetrics
from .worker import RewardLoopWorker


@dataclass
class BatchHandle:
    items: List[RewardItem]
    futures: List[Any]
    chunks: List[List[RewardItem]]
    collected: bool = False


class AsyncRewardPipeline:
    def __init__(self, workers=None, num_workers=1, mode="async", backlog=2, on_backlog_full="block",
                 on_error="raise", worker_kwargs=None):
        if mode not in ("async", "sync"):
            raise ValueError("mode must be 'async' or 'sync'")
        if on_backlog_full not in ("block", "drop_oldest"):
            raise ValueError("on_backlog_full must be 'block' or 'drop_oldest'")
        if on_error not in ("raise", "zero"):
            raise ValueError("on_error must be 'raise' or 'zero'")
        if num_workers < 1 or backlog < 1:
            raise ValueError("num_workers and backlog must be positive")
        self.mode, self.backlog, self.on_backlog_full, self.on_error = mode, backlog, on_backlog_full, on_error
        self.workers = workers or [RewardLoopWorker(**(worker_kwargs or {})) for _ in range(num_workers)]
        if not self.workers:
            raise ValueError("at least one worker is required")
        self.pending: List[BatchHandle] = []
        self.metrics = RewardLoopMetrics()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_workers))
        self._lock = threading.RLock()

    @classmethod
    def from_args(cls, args, **kwargs):
        worker_kwargs = vars(args).copy() if hasattr(args, "__dataclass_fields__") else dict(args)
        return cls(num_workers=worker_kwargs.pop("num_workers", 1), worker_kwargs=worker_kwargs, **kwargs)

    @staticmethod
    def _drain_future(future):
        if isinstance(future, concurrent.futures.Future):
            if not future.done():
                future.cancel()
            try:
                future.result()
            except BaseException:
                pass
        else:
            try:
                import ray
                ray.cancel(future, force=True)
            except Exception:
                pass

    def _discard(self, handle: BatchHandle):
        for future in handle.futures:
            self._drain_future(future)
        handle.collected = True
        if handle in self.pending:
            self.pending.remove(handle)

    def submit(self, batch: List[RewardItem]) -> BatchHandle:
        if not batch:
            return BatchHandle([], [], [])
        with self._lock:
            while len(self.pending) >= self.backlog:
                if self.on_backlog_full == "drop_oldest":
                    self._discard(self.pending[0])
                else:
                    self.collect(self.pending[0])
            chunks = split_items(batch, len(self.workers))
            futures = []
            started = time.monotonic()
            for worker, chunk in zip(self.workers, chunks):
                if hasattr(worker, "compute_score_batch") and hasattr(worker.compute_score_batch, "remote"):
                    futures.append(worker.compute_score_batch.remote(chunk))
                else:
                    futures.append(self._executor.submit(worker.compute_score_batch, chunk))
            handle = BatchHandle(batch, futures, chunks)
            self.pending.append(handle)
            self.metrics.submitted_batches += 1
            self.metrics.submit_time += time.monotonic() - started
            self.metrics.max_backlog = max(self.metrics.max_backlog, len(self.pending))
        if self.mode == "sync":
            self.collect(handle)
        return handle

    def collect(self, handle: BatchHandle):
        if handle is None:
            return []
        with self._lock:
            if handle.collected:
                raise RuntimeError("reward batch was already collected or discarded")
        started = time.monotonic()
        results = []
        errors = []
        for index, future in enumerate(handle.futures):
            try:
                if isinstance(future, concurrent.futures.Future):
                    values = future.result()
                else:
                    import ray
                    values = ray.get(future)
                results.extend(values)
            except BaseException as exc:
                errors.append((index, exc))
                if self.on_error == "raise":
                    for remaining in handle.futures[index + 1:]:
                        self._drain_future(remaining)
                    with self._lock:
                        handle.collected = True
                        if handle in self.pending:
                            self.pending.remove(handle)
                    raise
                results.extend(RewardResult(item.item_id, 0.0, {"error": str(exc)})
                              for item in handle.chunks[index])
        if errors and self.on_error == "zero":
            # Successful chunks remain intact; failed chunks have already received zeros.
            pass
        with self._lock:
            handle.collected = True
            if handle in self.pending:
                self.pending.remove(handle)
        self.metrics.collect_wait_time += time.monotonic() - started
        self.metrics.collected_batches += 1
        by_id = reorder_by_id(results)
        return [by_id[item.item_id] for item in handle.items]

    def close(self):
        with self._lock:
            for handle in list(self.pending):
                self._discard(handle)
        self._executor.shutdown(wait=True, cancel_futures=True)
