# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end proof of the Inline_Fast_Path + Client_Future_Layer seam.

A minimal harness drives the *real* ``submit_and_peek`` against a real compute
worker and a real (memory) ServerState -- no HTTP, no GPU. The resulting envelope
is round-tripped through model_dump/model_validate (simulating the wire) and fed to
the real client ``resolve``, so this covers the exact submit -> client path.
"""
from __future__ import annotations

import pytest

ray = pytest.importorskip('ray')

from twinkle.server.state import ServerState                              # noqa: E402
from twinkle.server.task_queue.config import TaskQueueConfig        # noqa: E402
from twinkle.server.task_queue.mixin import TaskQueueMixin          # noqa: E402
from twinkle_client import _future                                        # noqa: E402
from twinkle_client.exceptions import TaskFailedError                     # noqa: E402
from twinkle.protocol.types.lifecycle import TaskEnvelope                   # noqa: E402


class _Harness(TaskQueueMixin):
    """Real task queue + real state, with a window generous enough to be deterministic."""

    def __init__(self) -> None:
        self.state = ServerState()
        self.replica_id = 'test-replica'
        # enabled=False skips rate limiting; a 5s window makes "task finishes inside
        # the window" deterministic for a trivial in-process coroutine.
        self._init_task_queue(
            TaskQueueConfig(enabled=False, inline_fast_path_timeout=5.0), deployment_name='test')


def _across_the_wire(env: TaskEnvelope) -> TaskEnvelope:
    return TaskEnvelope.model_validate(env.model_dump(mode='json'))


@pytest.mark.asyncio
async def test_window_completed_task_is_single_round_trip(monkeypatch):
    """A task terminal within the window makes the client issue zero retrieves."""
    h = _Harness()

    async def _ok():
        return None

    try:
        env = await h.submit_and_peek(lambda: _ok(), task_type='step')
        assert env.status == 'completed'
        assert env.result is None

        monkeypatch.setattr(_future, '_post_retrieve',
                            lambda _r: pytest.fail('completed submit must not poll retrieve'))
        assert _future.resolve(_across_the_wire(env), model_cls=None) is None
    finally:
        await h.shutdown_task_queue()


@pytest.mark.asyncio
async def test_window_failed_task_surfaces_payload_as_taskfailed(monkeypatch):
    """A failure inside the window reaches the client via the submit
    response and is raised as TaskFailedError with its payload intact."""
    h = _Harness()

    async def _boom():
        raise ValueError('kaboom')

    try:
        env = await h.submit_and_peek(lambda: _boom(), task_type='step')
        assert env.status == 'failed'
        assert env.error is not None
        assert 'kaboom' in env.error.error

        monkeypatch.setattr(_future, '_post_retrieve',
                            lambda _r: pytest.fail('failed submit must not poll retrieve'))
        with pytest.raises(TaskFailedError) as exc:
            _future.resolve(_across_the_wire(env), model_cls=None)
        assert 'kaboom' in exc.value.error
        assert exc.value.category == 'server'
        assert exc.value.error_code == 500
    finally:
        await h.shutdown_task_queue()
