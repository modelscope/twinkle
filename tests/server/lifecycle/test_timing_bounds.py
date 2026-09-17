# Copyright (c) ModelScope Contributors. All rights reserved.
"""Timing guards for the lifecycle constants.

- Property 2 (R1#6): a single Submit_Endpoint's server-side duration is bounded by the
  Inline_Fast_Path window + 1s and is INDEPENDENT of how long the task itself runs. This
  is the spec's core benefit claim and was previously the only property with no automated
  guard.
- R2#4 / R2#6 / D4: the retrieve poll interval is a single shared declaration, is strictly
  inside the Long_Poll_Window, and is deliberately FIXED (see the measurement recorded in
  ``poll_config`` and in the test below).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

ray = pytest.importorskip('ray')

from twinkle.server.lifecycle.poll_config import long_poll_window, retrieve_poll_interval  # noqa: E402
from twinkle.server.state import ServerState                              # noqa: E402
from twinkle.server.utils.task_queue.config import TaskQueueConfig        # noqa: E402
from twinkle.server.utils.task_queue.mixin import TaskQueueMixin          # noqa: E402

_WINDOW = 0.05


class _Harness(TaskQueueMixin):
    def __init__(self) -> None:
        self.state = ServerState()
        self.replica_id = 'test-replica'
        self._init_task_queue(
            TaskQueueConfig(enabled=False, inline_fast_path_timeout=_WINDOW), deployment_name='test')


@pytest.mark.asyncio
async def test_submit_duration_is_bounded_and_task_duration_independent():
    """Property 2: submit returns on the window, not on task completion."""
    h = _Harness()

    async def fast():
        return None

    async def slow():
        await asyncio.sleep(5.0)
        return {'loss': 1.0}

    try:
        # Warm up first: the very first submit pays for creating the detached state actor
        # and starting the ComputeWorker (~10s on a cold Ray). That cost is not part of the
        # submit path this test is about, and timing it made the test pass or fail depending
        # on whether an earlier test in the directory had already warmed the backend.
        await h.submit_and_peek(lambda: fast(), task_type='warmup')

        started = time.monotonic()
        env = await h.submit_and_peek(lambda: slow(), task_type='forward_backward')
        elapsed = time.monotonic() - started

        # Bounded by the window + 1s even though the task needs 5s (R1#6).
        assert elapsed < _WINDOW + 1.0, f'submit took {elapsed:.3f}s, expected < {_WINDOW + 1.0}s'
        # 5s task cannot have finished, so the envelope must be non-terminal.
        assert env.status not in ('completed', 'failed', 'cancelled'), env.status
        assert env.result is None and env.error is None
        assert env.request_id
    finally:
        await h.shutdown_task_queue()


def test_poll_interval_satisfies_the_constant_chain():
    interval, window = retrieve_poll_interval(), long_poll_window()
    assert interval > 0, 'a non-positive interval would busy-spin the state backend'
    assert interval < window, 'a single poll step must not exceed the whole window'


def test_both_retrieve_endpoints_share_one_interval_declaration():
    """R2#6: one declaration point, and no endpoint reading os.environ on its own.

    Also pins the measured decision: a FIXED interval, not exponential backoff. The
    backoff variant was implemented, measured on real PPU hardware, and reverted --
    control-plane ops (0.00-0.09s) never reach retrieve because the 50ms inline window
    absorbs them, while ``forward_backward`` lands at 0.52-0.65s where a 0.05->1.0s
    doubling schedule checks at 0.80 instead of the fixed schedule's 0.55, costing ~22%
    per step. If backoff is ever reintroduced, it needs a ceiling around 0.2s and an
    explicit decision to pay 2.5x the poll rate.
    """
    import twinkle.server.gateway.tinker_handlers as tinker_h
    import twinkle.server.gateway.twinkle_handlers as twinkle_h
    from twinkle.server.lifecycle import poll_config

    assert not hasattr(poll_config, 'initial_poll_interval'), 'backoff was reverted by measurement'
    assert not hasattr(poll_config, 'max_poll_interval'), 'backoff was reverted by measurement'

    for module in (tinker_h, twinkle_h):
        source = Path(module.__file__).read_text(encoding='utf-8')
        assert 'TWINKLE_POLL_INTERVAL' not in source, f'{module.__name__} reads the env var directly'
        assert 'TWINKLE_LONG_POLL_TIMEOUT' not in source, f'{module.__name__} reads the env var directly'


def test_gateway_guard_warns_once_per_value_not_once_per_request(monkeypatch):
    """D5 guard must be audible but not spam.

    ``long_poll_window()`` runs on the hot path of both retrieve endpoints, not only at
    startup, so an unguarded warning would repeat on every retrieve request (~2/s during
    training). It must fire for a misconfigured value, stay silent on repeats of that same
    value, and speak up again when the value changes to a different offending one.

    Counts calls on a stub logger rather than using ``caplog``: ``get_logger()`` sets
    ``propagate = False``, so records never reach the root handler caplog installs.
    """
    from twinkle.server.lifecycle import poll_config

    class _Counter:
        def __init__(self):
            self.warnings = 0

        def warning(self, *_args, **_kwargs):
            self.warnings += 1

    counter = _Counter()
    monkeypatch.setattr(poll_config, 'logger', counter)
    monkeypatch.setattr(poll_config, '_warned_window', None, raising=False)

    def warnings_while(value: str, calls: int) -> int:
        monkeypatch.setenv('TWINKLE_LONG_POLL_TIMEOUT', value)
        before = counter.warnings
        for _ in range(calls):
            assert poll_config.long_poll_window() == float(value)
        return counter.warnings - before

    assert warnings_while('90', calls=5) == 1, 'an over-the-gateway-limit window must warn exactly once'
    assert warnings_while('90', calls=5) == 0, 'repeats of the same value must stay silent'
    assert warnings_while('120', calls=3) == 1, 'a different offending value must warn again'
    assert warnings_while('30', calls=5) == 0, 'a safe window must never warn'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
