# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property-based test for the shared Cleanup_Starter (R11.2, R11.3).

# Feature: server-structural-refactor, Property 1: Cleanup_Starter is idempotent
and never raises to its caller — for any deployment instance using the shared
LazyCleanupMixin and any sequence of one or more ``_ensure_state_cleanup_started``
invocations (including when the underlying ``start_cleanup_task`` reports
already-running or raises), the underlying start is awaited at most once across
invocations that observe an unstarted instance, no invocation propagates an
exception, and the instance ends in the started state.
"""
from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twinkle.server.app_scaffold import LazyCleanupMixin
from twinkle.server.gateway.app import GatewayServer
from twinkle.server.model.app import ModelManagement
from twinkle.server.processor.app import ProcessorManagement
from twinkle.server.sampler.app import SamplerManagement


def test_all_four_deployments_use_the_shared_mixin() -> None:
    for cls in (GatewayServer, ModelManagement, SamplerManagement, ProcessorManagement):
        assert issubclass(cls, LazyCleanupMixin), f'{cls.__name__} does not use LazyCleanupMixin'
        # And no class redefines the method — it resolves to the mixin's.
        assert cls._ensure_state_cleanup_started is LazyCleanupMixin._ensure_state_cleanup_started, \
            f'{cls.__name__} shadows the shared _ensure_state_cleanup_started'


class _FakeState:
    """Mock ServerState whose start_cleanup_task outcome is configurable."""

    def __init__(self, outcome: str) -> None:
        self.outcome = outcome  # 'ok' | 'already' | 'raise'
        self.start_calls = 0

    async def start_cleanup_task(self) -> bool:
        self.start_calls += 1
        if self.outcome == 'raise':
            raise RuntimeError('boom')
        if self.outcome == 'already':
            return False
        return True


class _Dep(LazyCleanupMixin):
    """Minimal carrier of the mixin for property testing (no Ray/event loop)."""

    def __init__(self, state: _FakeState) -> None:
        self.state = state


@settings(max_examples=200)
@given(
    n=st.integers(min_value=1, max_value=20),
    outcome=st.sampled_from(['ok', 'already', 'raise']),
)
def test_property_1_cleanup_starter_idempotent_and_never_raises(n: int, outcome: str) -> None:

    async def run() -> None:
        state = _FakeState(outcome)
        dep = _Dep(state)
        for _ in range(n):
            # Must never propagate an exception to the caller.
            await dep._ensure_state_cleanup_started()
        # At most one effective start across all invocations (the per-instance
        # flag short-circuits after the first attempt, even when it raised).
        assert state.start_calls <= 1
        # The instance ends in the started state.
        assert dep._state_cleanup_started is True

    asyncio.run(run())
