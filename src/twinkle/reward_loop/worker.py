import asyncio
import importlib
import threading
from typing import Optional

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
                 max_rpm=None, max_tpm=None, max_concurrent=1, timeout=300.0, **kwargs):
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

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def compute_score_batch(self, items):
        future = asyncio.run_coroutine_threadsafe(self.manager.run_batch(items), self._loop)
        return future.result()

    def close(self):
        """Stop the worker's event loop and release its thread pool. Idempotent."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        if not self._loop.is_closed():
            # Also shuts down the default executor that call_score() runs in.
            self._loop.close()
