# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from types import ModuleType, SimpleNamespace

import pytest

from twinkle.infra.continuous_work import _sticky_bind, _sticky_lookup
from twinkle.infra.worker_pool import WorkerPool


@pytest.mark.parametrize('shutdown_fails', [False, True])
def test_close_workers_before_pool_configuration(monkeypatch, shutdown_fails):
    """Sampler construction can fail before admission or sticky state exists."""
    ray = ModuleType('ray')
    failure = RuntimeError('Sampler initialization failed')
    shutdown_ref = object()
    actor = SimpleNamespace(
        _actor_id=SimpleNamespace(hex=lambda: 'initializing-worker'),
        shutdown=SimpleNamespace(remote=lambda: shutdown_ref),
    )
    killed = []

    def get(ref):
        assert ref is shutdown_ref
        if shutdown_fails:
            raise failure

    ray.get = get
    ray.kill = lambda actor, no_restart: killed.append((actor, no_restart))
    monkeypatch.setitem(sys.modules, 'ray', ray)
    pool = WorkerPool()
    pool._actors = (actor,)

    if shutdown_fails:
        with pytest.raises(RuntimeError, match='Sampler initialization failed') as caught:
            pool.close_workers()
        assert caught.value is failure
    else:
        pool.close_workers()

    assert pool._actors == ()
    assert killed == [(actor, True)]
    assert not hasattr(pool, '_sticky_owners_state')


def test_retiring_worker_removes_only_its_handlers_sticky_bindings():
    pool = WorkerPool()
    other_pool = WorkerPool()
    _sticky_bind(pool, 'retired-request', 'retired-worker')
    _sticky_bind(pool, 'live-request', 'live-worker')
    _sticky_bind(other_pool, 'other-request', 'retired-worker')

    pool._drop_worker_bookkeeping({'retired-worker'})

    assert _sticky_lookup(pool, 'retired-request') is None
    assert _sticky_lookup(pool, 'live-request') == 'live-worker'
    assert _sticky_lookup(other_pool, 'other-request') == 'retired-worker'
