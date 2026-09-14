import asyncio
import time
from collections.abc import Callable
from typing import Optional

from .base import RewardManagerBase
from .registry import register


class AsyncTokenBucket:
    def __init__(self, rate=None, capacity=None):
        self.rate = float(rate or 0)
        self.capacity = float(capacity if capacity is not None else (rate or 1))
        if self.rate < 0 or self.capacity <= 0:
            raise ValueError("token bucket rate must be non-negative and capacity positive")
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount=1):
        amount = float(amount)
        if amount < 0:
            raise ValueError("token amount must be non-negative")
        if not self.rate or amount == 0:
            return
        if amount > self.capacity:
            raise ValueError(f"token request {amount:g} exceeds bucket capacity {self.capacity:g}")
        while True:
            async with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                wait = (amount - self.tokens) / self.rate
            await asyncio.sleep(wait)


@register("rate_limited")
class RateLimitedRewardManager(RewardManagerBase):
    def __init__(self, *args, max_rpm=None, max_tpm=None, max_concurrent=1, timeout=300.0,
                 token_counter: Optional[Callable] = None, fallback_on_error=False, **kwargs):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        self.rpm = AsyncTokenBucket((max_rpm or 0) / 60.0, capacity=max_rpm or 1)
        self.tpm = AsyncTokenBucket((max_tpm or 0) / 60.0, capacity=max_tpm or 1)
        self.timeout = timeout
        self.token_counter = token_counter or (lambda item: len(item.solution_str))
        self.fallback_on_error = fallback_on_error

    async def run_single(self, item):
        try:
            await self.rpm.acquire()
            await self.tpm.acquire(self.token_counter(item))
            return await asyncio.wait_for(super().run_single(item), self.timeout)
        except (asyncio.TimeoutError, ValueError) as exc:
            if not self.fallback_on_error:
                raise
            return self._zero_result(item, exc)

    @staticmethod
    def _zero_result(item, exc):
        from ..data import RewardResult
        return RewardResult(item.item_id, 0.0, {"error": str(exc)})
