# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reusable lifecycle for task challengers."""
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

from twinkle.data_format import Trajectory
from twinkle.utils import get_logger
from twinkle_agentic.envs import Env

logger = get_logger()

__all__ = ['Challenger']


def _parallel(fn: Callable[[int], Any], count: int) -> List[Any]:
    """Run ``fn`` over ``range(count)`` concurrently, preserving order."""
    if count <= 1:
        return [fn(i) for i in range(count)]
    out: List[Any] = [None] * count
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(fn, i): i for i in range(count)}
        for future, i in futures.items():
            out[i] = future.result()
    return out


class Challenger(ABC):
    """Common batching and environment lifecycle for task challengers.

    Subclasses define how a round builds its prompt, explores it, and measures
    candidate difficulty. One environment is owned by one concurrent job for the
    complete lifetime of that job.
    """

    def __init__(
            self,
            *,
            envs: Sequence[Env],
            num_challenger_rollouts: int = 8,
            num_solver_rollouts: int = 8,
            pass_band: Tuple[float, float] = (1.0, 7.0),
            max_empty_rounds: int = 0,
    ):
        if not envs:
            raise ValueError('envs is empty: a challenger needs a workspace to act in and grade')
        if num_challenger_rollouts < 1:
            raise ValueError(f'num_challenger_rollouts must be >= 1, got '
                             f'{num_challenger_rollouts}')
        if num_solver_rollouts < 0:
            raise ValueError(f'num_solver_rollouts must be >= 0, got {num_solver_rollouts}')
        if max_empty_rounds < 0:
            raise ValueError(f'max_empty_rounds must be >= 0, got {max_empty_rounds}')
        if num_solver_rollouts:
            if len(pass_band) != 2:
                raise ValueError(f'pass_band is (low, high) in attempt counts, got {pass_band}')
            low, high = pass_band
            if not 0 <= low <= high <= num_solver_rollouts:
                raise ValueError(f'pass_band must satisfy 0 <= low <= high <= num_solver_rollouts, got '
                                 f'{pass_band} against num_solver_rollouts={num_solver_rollouts}')
        self.envs = list(envs)
        self.num_challenger_rollouts = num_challenger_rollouts
        self.num_solver_rollouts = num_solver_rollouts
        self.pass_band = pass_band
        self.max_empty_rounds = max_empty_rounds
        self.n_proposed = 0
        self.n_kept = 0

    @property
    def n_slots(self) -> int:
        """How many jobs may run at once: one per environment."""
        return len(self.envs)

    def env(self, slot: int = 0) -> Env:
        """Return the current environment for ``slot``."""
        return self.envs[slot]

    @abstractmethod
    def _build_challenge_prompt(self) -> Optional[Trajectory]:
        """Build one round's shared prompt, or return None when exhausted."""

    @abstractmethod
    def _explore(self, prompt: Trajectory) -> List[Trajectory]:
        """Generate and validate candidates from one shared prompt."""

    @abstractmethod
    def _filter_difficulty(self, tasks: List[Trajectory]) -> List[Trajectory]:
        """Measure candidate difficulty and return the accepted tasks."""

    def __call__(self, batch_size: int, total: Optional[int] = None) -> Iterator[List[Trajectory]]:
        """Yield finished tasks in batches."""
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        pending: List[Trajectory] = []
        produced = 0
        empty_rounds = 0
        while total is None or produced < total:
            want = batch_size if total is None else min(batch_size, total - produced)
            while len(pending) < want:
                kept = self._round()
                if kept is None:
                    if pending:
                        yield pending
                    return
                if kept:
                    empty_rounds = 0
                    pending.extend(kept)
                    continue
                empty_rounds += 1
                if self.max_empty_rounds and empty_rounds >= self.max_empty_rounds:
                    logger.warning(f'[{type(self).__name__}] stopped after {empty_rounds} '
                                   'consecutive rounds without a usable task')
                    if pending:
                        yield pending
                    return
            yield pending[:want]
            produced += want
            pending = pending[want:]

    def _round(self) -> Optional[List[Trajectory]]:
        """Run one proposal group; None means the source is exhausted."""
        prompt = self._build_challenge_prompt()
        if prompt is None:
            return None
        verified = self._explore(prompt)
        kept = self._filter_difficulty(verified)
        self.n_proposed += self.num_challenger_rollouts
        self.n_kept += len(kept)
        logger.info(f'[{type(self).__name__}] {self.num_challenger_rollouts} episodes, '
                    f'{len(verified)} verified, {len(kept)} in band '
                    f'(cumulative {self.n_kept}/{self.n_proposed})')
        return kept
