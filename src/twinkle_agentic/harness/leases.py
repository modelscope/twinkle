# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lending one agent harness per episode, from a fixed-size pool.

A harness carries the agent framework's per-episode state -- ms-agent's memory
tools, runtime hooks, a skill runtime. That state does not reset itself between
episodes, and sharing one harness across trajectories leaks it from one into the
next: measured on ms-agent, ``DefaultMemory`` guards itself with an
``asyncio.Lock`` that does not synchronise across the fresh event loops
``run_sync`` opens per call, so two episodes on one harness corrupt each other's
memory. So each job gets its own, the same way each job gets its own
:class:`~twinkle_agentic.envs.base.Env`.

This is the harness half of :class:`~twinkle_agentic.envs.base.EnvLeases`: the
pool bounds how many run at once (give it as many as there are workers and a
lease never blocks), and the cleaning happens on the way *in*. A harness cannot
be scrubbed back to new in place -- the leak is exactly the state that will not
clear -- so cleaning here means building a fresh one and dropping the last.
"""
from typing import Callable

from twinkle_agentic.utils.leases import Leases
from .base import AgentHarness


class HarnessLeases(Leases[AgentHarness]):
    """Hand each episode a harness with none of the last episode left in it.

    Built from a factory rather than a ready list: a lease returns a brand-new
    harness and discards the one handed back, because an agent framework's memory
    is what leaks across episodes and it is not something that can be reset. The
    pool holds ``size`` slots, so at most ``size`` harnesses are alive at once.
    """

    def __init__(self, factory: Callable[[], AgentHarness], size: int):
        if size < 1:
            raise ValueError(f'HarnessLeases size must be >= 1, got {size}')
        self._factory = factory
        super().__init__([factory() for _ in range(size)])

    def _prepare(self, harness: AgentHarness) -> AgentHarness:
        """Drop the harness that was handed back and lend a fresh one."""
        close = getattr(harness, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 -- a dead harness must not block the next job
                pass
        return self._factory()


__all__ = ['HarnessLeases']
