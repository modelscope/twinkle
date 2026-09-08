from __future__ import annotations

import asyncio

import pytest

from twinkle_client.async_rl import Worker, WorkerPipeline


class _FunctionWorker(Worker):

    def __init__(self, name, function):
        super().__init__(name)
        self.function = function

    async def run(self) -> None:
        await self.function()


def test_worker_pipeline_runs_roles_concurrently() -> None:
    producer_started = asyncio.Event()
    consumer_started = asyncio.Event()

    async def producer():
        producer_started.set()
        await consumer_started.wait()

    async def consumer():
        consumer_started.set()
        await producer_started.wait()

    asyncio.run(WorkerPipeline((
        _FunctionWorker('producer', producer),
        _FunctionWorker('consumer', consumer),
    )).run())


def test_worker_pipeline_cancels_peer_when_one_role_fails() -> None:
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    async def peer():
        waiting.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def failure():
        await waiting.wait()
        raise RuntimeError('role failed')

    with pytest.raises(RuntimeError, match='role failed'):
        asyncio.run(WorkerPipeline((
            _FunctionWorker('peer', peer),
            _FunctionWorker('failure', failure),
        )).run())
    assert cancelled.is_set()


def test_worker_pipeline_rejects_duplicate_role_names() -> None:
    async def noop():
        return None

    with pytest.raises(ValueError, match='unique'):
        WorkerPipeline((
            _FunctionWorker('same', noop),
            _FunctionWorker('same', noop),
        ))
