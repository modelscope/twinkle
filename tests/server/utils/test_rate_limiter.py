# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import pytest

from twinkle.server.task_queue.rate_limiter import RateLimiter


class _RecordingGauge:

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, str]]] = []

    def set(self, value: int, *, tags: dict[str, str]) -> None:
        self.calls.append((value, tags))


@pytest.mark.asyncio
async def test_zero_rps_disables_request_limit() -> None:
    limiter = RateLimiter(rps_limit=0, tps_limit=100)

    assert await limiter.check_and_record('token', 1) == (True, None)
    assert await limiter.check_and_record('token', 1) == (True, None)


@pytest.mark.asyncio
async def test_zero_tps_disables_token_limit() -> None:
    limiter = RateLimiter(rps_limit=100, tps_limit=0)

    assert await limiter.check_and_record('token', 1_000_000) == (True, None)
    assert await limiter.check_and_record('token', 1_000_000) == (True, None)


@pytest.mark.asyncio
async def test_active_token_metric_distinguishes_replicas() -> None:
    gauge = _RecordingGauge()
    first = RateLimiter(
        rps_limit=100,
        tps_limit=100,
        active_tokens_gauge=gauge,
        deployment_name='model',
        replica_id='replica-a',
    )
    second = RateLimiter(
        rps_limit=100,
        tps_limit=100,
        active_tokens_gauge=gauge,
        deployment_name='model',
        replica_id='replica-b',
    )

    assert await first.check_and_record('token-a', 1) == (True, None)
    assert await second.check_and_record('token-b', 1) == (True, None)

    assert gauge.calls == [
        (1, {'deployment': 'model', 'replica': 'replica-a'}),
        (1, {'deployment': 'model', 'replica': 'replica-b'}),
    ]
