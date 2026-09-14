"""Base reward manager implementation."""
import asyncio
import inspect
import math
from abc import ABC
from collections.abc import Mapping
from typing import Any, Callable, Optional

from ..data import RewardItem, RewardResult


class RewardManagerBase(ABC):
    def __init__(self, compute_score: Optional[Callable] = None, max_concurrent: Optional[int] = None, **kwargs):
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.compute_score = compute_score
        self.kwargs = kwargs
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def call_score(self, item: RewardItem) -> Any:
        if self.compute_score is None:
            raise ValueError(f"compute_score is required for item {item.item_id!r}")
        args = (item.data_source, item.solution_str, item.ground_truth, item.extra_info)
        if inspect.iscoroutinefunction(self.compute_score):
            return await self.compute_score(*args)
        loop = asyncio.get_running_loop()
        value = await loop.run_in_executor(None, lambda: self.compute_score(*args))
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def normalize_score(value: Any):
        if isinstance(value, tuple):
            if len(value) != 2 or not isinstance(value[1], Mapping):
                raise TypeError("tuple score must be (number, mapping)")
            score, extra = value
            return RewardManagerBase._finite_score(score), dict(extra)
        if isinstance(value, Mapping):
            if "score" not in value and "reward" not in value:
                raise TypeError("mapping score must contain 'score' or 'reward'")
            score = value.get("score", value.get("reward"))
            extra = value.get("extra_info", {})
            if not isinstance(extra, Mapping):
                raise TypeError("score extra_info must be a mapping")
            return RewardManagerBase._finite_score(score), dict(extra)
        return RewardManagerBase._finite_score(value), {}

    @staticmethod
    def _finite_score(value: Any) -> float:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("reward score must be finite")
        return score

    async def run_single(self, item: RewardItem) -> RewardResult:
        async def run():
            score, extra = self.normalize_score(await self.call_score(item))
            return RewardResult(item.item_id, score, extra)
        if self._semaphore is None:
            return await run()
        async with self._semaphore:
            return await run()

    async def run_batch(self, items):
        tasks = [asyncio.create_task(self.run_single(item)) for item in items]
        if not tasks:
            return []
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
