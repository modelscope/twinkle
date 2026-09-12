# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reusable lifecycle for task challengers."""
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from twinkle.data_format import Trajectory
from twinkle.utils import get_logger
from twinkle_agentic.envs import Env, EnvLeases

logger = get_logger()

__all__ = ['ChallengeBatch', 'Challenger']


@dataclass
class ChallengeBatch:
    """One batch of trainable episodes, split by which side produced them.

    A side is a list of groups, one group being what a single advantage is taken
    over: every member carries its own ``rewards``, and the advantage is that reward
    against the rest of its group. Handed out grouped rather than flat so that the
    consumer does not have to reconstruct the grouping to score anything.
    """

    challenger: List[List[Trajectory]] = field(default_factory=list)
    solver: List[List[Trajectory]] = field(default_factory=list)

    def __len__(self) -> int:
        """Trajectories, not groups: what a batch costs to train on."""
        return sum(len(group) for group in self.challenger) + sum(len(group) for group in self.solver)


class Challenger(ABC):
    """Common batching and environment lifecycle for task challengers.

    Work is a stream of jobs over a pool of environments, not a sequence of
    rounds. A job leases one environment for its whole life and gives it back the
    moment it ends, to whichever job is next in line -- so proposing and solving
    run at the same time, on the same environments, and the sampler and the API
    are never idle waiting for the slowest member of a round.

    A subclass says what a unit of work is: :meth:`_launch` submits its jobs with
    :meth:`_submit`, and whatever bookkeeping ties those jobs together calls
    :meth:`_complete` once, with the groups that unit earned. Nothing here waits
    on a job, and a job must not wait on another job -- the workers are the
    environments, so a job that blocks holds one hostage.

    The single barrier is the batch. The caller's next move after taking one is an
    optimizer step, and a job that straddled that step would have been sampled
    under weights that no longer exist, so a batch drains before it is handed
    over. Nothing inside a batch drains.
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
        self.envs = EnvLeases(envs)
        self.num_challenger_rollouts = num_challenger_rollouts
        self.num_solver_rollouts = num_solver_rollouts
        self.pass_band = pass_band
        self.max_empty_rounds = max_empty_rounds
        self.n_proposed = 0
        self.n_kept = 0
        # One worker per environment, which is what makes a lease never block: a
        # worker runs one job, a job holds one environment, so a running job
        # always has one to hold.
        self._workers = ThreadPoolExecutor(max_workers=len(self.envs), thread_name_prefix='challenger')
        self._finished: 'queue.Queue[Tuple[List[List[Trajectory]], List[List[Trajectory]]]]' = queue.Queue()
        self._counter = threading.Lock()
        self._jobs = 0

    @abstractmethod
    def _launch(self) -> bool:
        """Submit the jobs of one more unit of work; False when exhausted.

        Must submit at least one job when it returns True, since the caller reads
        "nothing running" as "nothing more is coming".
        """

    def _submit(self, job: Callable[[Env], None]) -> None:
        """Run ``job`` on an environment of its own, as soon as one is free."""
        with self._counter:
            self._jobs += 1
        self._workers.submit(self._run, job)

    def _run(self, job: Callable[[Env], None]) -> None:
        # The count has to fall however this ends, the lease included: a job still
        # counted as running is a batch that never finishes waiting for it.
        try:
            with self.envs.lease() as env:
                job(env)
        except Exception as exc:  # noqa: BLE001 -- one bad job must not end a run
            logger.warning(f'[{type(self).__name__}] job failed: {type(exc).__name__}: {exc}')
        finally:
            with self._counter:
                self._jobs -= 1

    def _complete(self, challenger: List[List[Trajectory]], solver: List[List[Trajectory]]) -> None:
        """Hand one unit's trainable groups to whoever is filling a batch.

        Called from the job that finished the unit, so it must not block: it drops
        the groups in a queue and returns to the pool.
        """
        proposing = [group for group in challenger if self._has_spread(group)]
        solving = [group for group in solver if self._has_spread(group)]
        flat = len(challenger) - len(proposing) + len(solver) - len(solving)
        if flat:
            logger.info(f'[{type(self).__name__}] dropped {flat} groups whose rewards were all equal')
        self._finished.put((proposing, solving))

    @staticmethod
    def _has_spread(group: List[Trajectory]) -> bool:
        """True when a group's rewards differ.

        One reward repeated is one reward minus itself: every advantage in the group
        is zero and the whole group is a forward pass spent on no gradient.
        """
        if len(group) < 2:
            return False
        first = float(group[0].get('rewards') or 0.0)
        return any(abs(float(member.get('rewards') or 0.0) - first) > 1e-9 for member in group[1:])

    def _quotas(self, batch_size: int, solver_ratio: float) -> Tuple[int, int]:
        """Groups per batch on each side, from a trajectory count and a solver share.

        Rounded to whole groups, and never below one group for a side that is asked
        for at all -- so the ratio is honoured to the nearest group and a batch can
        come out larger than ``batch_size``. Both numbers are fixed here, once, so
        every batch of a run has the same shape.
        """
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}')
        if not 0.0 <= solver_ratio <= 1.0:
            raise ValueError(f'solver_ratio is the solving side\'s share of a batch, got {solver_ratio}')
        solver_groups = 0
        if solver_ratio > 0.0 and self.num_solver_rollouts:
            solver_groups = max(1, round(batch_size * solver_ratio / self.num_solver_rollouts))
        challenger_groups = 0
        if solver_ratio < 1.0:
            rest = batch_size - solver_groups * self.num_solver_rollouts
            challenger_groups = max(1, round(rest / self.num_challenger_rollouts))
        if not challenger_groups and not solver_groups:
            raise ValueError(f'solver_ratio={solver_ratio} with num_solver_rollouts='
                             f'{self.num_solver_rollouts} leaves a batch with nothing in it')
        return challenger_groups, solver_groups

    def __call__(self,
                 batch_size: int,
                 total: Optional[int] = None,
                 solver_ratio: float = 0.5) -> Iterator[ChallengeBatch]:
        """Yield batches of a fixed shape: so many proposing groups, so many solving ones.

        ``batch_size`` counts trajectories and ``solver_ratio`` is the solving side's
        share of them; both are turned into whole group counts up front.

        Units of work are launched until both quotas are filled. A unit yields one
        proposing group and at most one solving group, so the proposing side fills
        first and its surplus is dropped -- nothing is held for the next batch,
        where it would be scored against weights that have already moved. A run
        that stops making progress is caught by ``max_empty_rounds``, counted over
        units that added nothing to the batch.

        ``total`` bounds the trajectories yielded overall, to the nearest batch.
        """
        want_challenger, want_solver = self._quotas(batch_size, solver_ratio)
        logger.info(f'[{type(self).__name__}] batch shape: '
                    f'{want_challenger} x {self.num_challenger_rollouts} proposing + '
                    f'{want_solver} x {self.num_solver_rollouts} solving trajectories; '
                    f'{len(self.envs)} environments')
        produced = 0
        while total is None or produced < total:
            batch = self._fill(want_challenger, want_solver)
            if batch is None:
                return
            produced += len(batch)
            yield batch

    def _fill(self, want_challenger: int, want_solver: int) -> Optional[ChallengeBatch]:
        """One batch; None when the source ran out with nothing left to hand over."""
        pending = ChallengeBatch()
        dropped = 0
        empty = 0
        exhausted = False
        while len(pending.challenger) < want_challenger or len(pending.solver) < want_solver:
            # Refill to capacity first: the pool is the throttle, and a unit's own
            # jobs multiply once it starts, so keeping the workers busy is enough
            # to keep proposing and solving overlapped without counting either.
            while not exhausted and self._jobs < len(self.envs):
                if not self._launch():
                    exhausted = True
            unit = self._next()
            if unit is not None:
                added, over = self._absorb(pending, unit, want_challenger, want_solver)
                dropped += over
                empty = 0 if added else empty + 1
            elif exhausted:
                break
            else:
                empty += 1
            if self.max_empty_rounds and empty >= self.max_empty_rounds:
                logger.warning(f'[{type(self).__name__}] stopped after {empty} consecutive '
                               'units of work that added nothing to the batch')
                break
        dropped += self._drain(pending, want_challenger, want_solver)
        if dropped:
            logger.info(f'[{type(self).__name__}] dropped {dropped} groups the batch had no room for')
        return pending if len(pending) else None

    def _next(self) -> Optional[Tuple[List[List[Trajectory]], List[List[Trajectory]]]]:
        """The next finished unit, or None once nothing is still running."""
        while self._jobs:
            try:
                return self._finished.get(timeout=1.0)
            except queue.Empty:
                continue
        # The last job may have completed its unit between the check and here.
        try:
            return self._finished.get_nowait()
        except queue.Empty:
            return None

    def _absorb(self, pending: ChallengeBatch, unit: Tuple[List[List[Trajectory]], List[List[Trajectory]]],
                want_challenger: int, want_solver: int) -> Tuple[int, int]:
        """Take what a unit earned into the batch; returns ``(added, dropped)``."""
        challenger, solver = unit
        added = 0
        dropped = 0
        for into, groups, want in ((pending.challenger, challenger, want_challenger), (pending.solver, solver,
                                                                                       want_solver)):
            for group in groups:
                if len(into) >= want:
                    dropped += 1
                    continue
                into.append(group)
                added += 1
        return added, dropped

    def _drain(self, pending: ChallengeBatch, want_challenger: int, want_solver: int) -> int:
        """Wait out every job still running, taking what it earned if there is room."""
        dropped = 0
        while True:
            unit = self._next()
            if unit is None:
                return dropped
            dropped += self._absorb(pending, unit, want_challenger, want_solver)[1]

    def close(self) -> None:
        self._workers.shutdown(wait=True)
        self.envs.close()
