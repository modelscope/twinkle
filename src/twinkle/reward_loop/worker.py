import asyncio
import importlib
import threading
from typing import Optional

from .batching import MicroBatcher
from .data import RewardItem
from .default_score import compute_score as default_compute_score
from .reward_manager import get_reward_manager_cls

try:
    from twinkle.infra import remote_class
except ImportError:  # pragma: no cover
    remote_class = lambda **kwargs: (lambda cls: cls)


@remote_class(execute="all")
class RewardLoopWorker:
    def __init__(self, manager_name="naive", compute_score=None, custom_reward_function_path=None,
                 custom_reward_function_name="compute_score", unknown_rewards="warn", reward_kwargs=None,
                 max_rpm=None, max_tpm=None, max_concurrent=1, timeout=300.0,
                 micro_batch_size=0, micro_batch_timeout_ms=0.0, **kwargs):
        if compute_score is None and custom_reward_function_path:
            module = importlib.import_module(custom_reward_function_path)
            compute_score = getattr(module, custom_reward_function_name)
        self.unknown_rewards = unknown_rewards
        self.compute_score = compute_score or default_compute_score
        manager_cls = get_reward_manager_cls(manager_name)
        options = dict(reward_kwargs or {})
        if manager_name == "rate_limited":
            options.update(max_rpm=max_rpm, max_tpm=max_tpm, max_concurrent=max_concurrent, timeout=timeout)
        self.manager = manager_cls(compute_score=self.compute_score, **options)
        # Managers build asyncio primitives in __init__ (the base class semaphore,
        # the token bucket locks), and a primitive binds itself to the first loop
        # it is awaited on. Serving every batch from one long-lived loop keeps them
        # reusable across batches; a per-batch asyncio.run() breaks on the second.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True,
                                             name="RewardLoopWorker-EventLoop")
        self._loop_thread.start()
        # Optional: merge items across concurrent compute_score_batch calls into
        # larger manager calls (see batching.py). 0 / 1 leaves the worker exactly
        # as it was, one manager call per compute_score_batch.
        self._batcher = None
        if micro_batch_size and micro_batch_size > 1:
            self._batcher = MicroBatcher(self.manager.call_score_batch, int(micro_batch_size),
                                         float(micro_batch_timeout_ms), loop=self._loop)
        # How many calls must be able to overlap for a chunk to fill up. The
        # dispatcher sizes its thread pool from this (AsyncRewardPipeline.__init__):
        # one thread per call, so a narrower pool would dribble items in one at a
        # time and no chunk would ever reach micro_batch_size.
        self.max_inflight_calls = max(1, int(micro_batch_size or 0))
        self._inflight = set()
        self._inflight_lock = threading.Lock()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def compute_score_batch(self, items):
        if self._batcher is not None:
            return self._batcher.submit(items).result()
        future = asyncio.run_coroutine_threadsafe(self.manager.run_batch(items), self._loop)
        with self._inflight_lock:
            self._inflight.add(future)
        future.add_done_callback(self._release_inflight)
        return future.result()

    def _release_inflight(self, future):
        with self._inflight_lock:
            self._inflight.discard(future)

    def close(self):
        """Stop the worker's event loop and release its thread pool. Idempotent."""
        if self._batcher is not None:
            self._batcher.close()
        # Callers blocked in compute_score_batch must be released here, not by the
        # loop: stopping it would leave a just-woken coroutine un-run, and the
        # caller would wait forever for a result that can no longer arrive.
        with self._inflight_lock:
            waiting, self._inflight = list(self._inflight), set()
        for future in waiting:
            try:
                if not future.done():
                    future.set_exception(RuntimeError('RewardLoopWorker closed with the call in flight'))
            except Exception:  # pragma: no cover - raced with completion
                pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        if not self._loop.is_closed():
            # Also shuts down the default executor that call_score() runs in.
            self._loop.close()
