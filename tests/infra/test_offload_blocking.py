# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect88: offload_blocking keeps a blocking sync remote method off the actor's event loop.

Ray makes a class with any ``async def`` an asyncio actor and runs its sync methods ON the
default concurrency-group loop (async_compat.sync_to_async). A long-blocking sync body there
(a weight receive waiting on a transport) monopolizes the loop, so resume_generation in the
weight sync's finally queues forever behind it. remote_function(offload_blocking=True) routes
the call to the method's awaitable companion, which runs the body in a worker thread.

Reverse-controls (flag-off / non-flagged) MUST starve: they prove the offload, not ray default
behavior, is what frees the loop.
"""
from __future__ import annotations

import concurrent.futures
import time
from types import SimpleNamespace

import pytest

ray = pytest.importorskip('ray')

import twinkle.infra as infra  # noqa: E402
from twinkle.infra import remote_function  # noqa: E402
from twinkle.infra.continuous_work import _WORKER_ASYNC_PREFIX  # noqa: E402


def _block_forever():
    return concurrent.futures.Future().result()


class _Component:
    async def _async_marker(self):
        return 1

    @remote_function(dispatch='all', lazy_collect=True, offload_blocking=True)
    def receive_offload(self):
        return _block_forever()

    @remote_function(dispatch='all', lazy_collect=True, offload_blocking=False)
    def receive_noflag(self):
        return _block_forever()

    @remote_function(dispatch='all', lazy_collect=False, offload_blocking=True)
    def receive_value(self):
        return 42

    @remote_function(dispatch='all', lazy_collect=False, offload_blocking=True)
    def receive_raise(self):
        raise ValueError('transport broke')

    @remote_function(dispatch='all', collect='none', lazy_collect=False, timeout=5)
    def resume_generation(self):
        return 'resumed'


for _name in dir(_Component):
    _companion = getattr(getattr(_Component, _name, None), '_worker_async_companion', None)
    if _companion is not None:
        setattr(_Component, _WORKER_ASYNC_PREFIX + _name, _companion)

_Actor = ray.remote(max_concurrency=256)(_Component)


@pytest.fixture(scope='module', autouse=True)
def _ray_and_ray_mode():
    ray.init(ignore_reinit_error=True, num_cpus=4, include_dashboard=False, logging_level='ERROR')
    prev = infra._mode
    infra._mode = 'ray'
    try:
        yield
    finally:
        infra._mode = prev


def _handle():
    return SimpleNamespace(_actors=[_Actor.remote()], _max_concurrency=None)


def _resume_starved(h):
    try:
        _Component.resume_generation(h)
        return False
    except ray.exceptions.GetTimeoutError:
        return True


def test_offload_frees_loop():
    h = _handle()
    _Component.receive_offload(h)  # orphan blocks a worker thread, must not block the loop
    time.sleep(2)
    try:
        assert not _resume_starved(h), 'resume starved despite offload_blocking'
    finally:
        ray.kill(h._actors[0], no_restart=True)


def test_reverse_flag_off_starves():
    h = _handle()
    _Component.receive_noflag(h)
    time.sleep(2)
    try:
        assert _resume_starved(h), 'flag-off receive did not starve (control invalid)'
    finally:
        ray.kill(h._actors[0], no_restart=True)


def test_offload_outcome_fidelity():
    h = _handle()
    try:
        assert _Component.receive_value(h) == 42
        with pytest.raises(ValueError):
            _Component.receive_raise(h)
    finally:
        ray.kill(h._actors[0], no_restart=True)


def test_local_mode_unchanged():
    prev = infra._mode
    infra._mode = 'local'
    try:
        out = _Component().receive_value()
        assert out == 42 and not hasattr(out, '__await__')
    finally:
        infra._mode = prev