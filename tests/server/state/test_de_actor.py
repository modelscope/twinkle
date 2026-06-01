# Copyright (c) ModelScope Contributors. All rights reserved.
"""Phase 0d — De-Actor ServerState tests (R19).

Covers:
- # Feature: server-config-observability-refactor, Property 25: State operation equivalence under direct-backend access
- de-Actor wiring: ``get_server_state`` returns a direct-bound ``ServerState``
  and never creates a detached Ray Actor (R19.1, R19.2).
- in-process MemoryBackend works without Redis (R19.6).
"""
from __future__ import annotations

from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from twinkle.server.state import (
    PersistenceConfig,
    ReplicaRegistry,
    ServerState,
    get_server_state,
    reset_server_state_cache,
)
from twinkle.server.state.backend.memory_backend import MemoryBackend


# ---------- 4.6: de-Actor wiring + in-process persistence ------------------ #


def _ray_attr_used(obj_path: str) -> bool:
    """Return True if ``obj_path`` (e.g. ``ray.remote``) is referenced in source."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / 'src' / 'twinkle' / 'server' / 'state' / 'server_state.py'
    return obj_path in src.read_text()


def test_no_detached_actor_in_source() -> None:
    """The state module must not call ``ray.remote(...)`` or use ``lifetime='detached'``.

    Static check: searching the file is enough — the dynamic check below also
    confirms ``ray.remote`` is never invoked when ``get_server_state`` runs.
    """
    assert not _ray_attr_used('ray.remote('), (
        'state/server_state.py still references ray.remote(...) — '
        'detached actor must not be created (R19.1).'
    )
    assert not _ray_attr_used("lifetime='detached'"), (
        "state/server_state.py still uses lifetime='detached' (R19.1)."
    )


def test_get_server_state_does_not_call_ray_remote() -> None:
    reset_server_state_cache()
    import ray

    with mock.patch.object(ray, 'remote') as remote_spy, \
            mock.patch.object(ray, 'get_actor', side_effect=ValueError) as get_actor_spy:
        state = get_server_state(actor_name='unit', backend=MemoryBackend())
    assert isinstance(state, ServerState)
    assert remote_spy.call_count == 0, 'ray.remote was called — detached actor created'
    # ray.get_actor may not be called at all under direct-backend access.
    # Either way, the contract is that no remote actor is built.
    _ = get_actor_spy


def test_get_server_state_caches_per_process() -> None:
    reset_server_state_cache()
    a = get_server_state(actor_name='cache-a', backend=MemoryBackend())
    b = get_server_state(actor_name='cache-a')
    assert a is b


def test_get_server_state_separate_keys_yield_separate_instances() -> None:
    reset_server_state_cache()
    a = get_server_state(actor_name='k1', backend=MemoryBackend())
    b = get_server_state(actor_name='k2', backend=MemoryBackend())
    assert a is not b


def test_in_process_persistence_no_redis_required() -> None:
    """``PersistenceConfig`` defaults to memory mode and ``ServerState`` works
    without an external Redis (R19.6)."""
    reset_server_state_cache()
    cfg = PersistenceConfig()  # mode == 'memory'
    state = get_server_state(actor_name='no-redis', persistence_config=cfg)
    assert isinstance(state, ServerState)


# ---------- 4.5: state-operation equivalence under direct-backend ---------- #


_OP_STRATEGY = st.lists(
    st.one_of(
        # ('register_replica', replica_id, max_loras)
        st.tuples(
            st.just('register_replica'),
            st.sampled_from(['r1', 'r2', 'r3']),
            st.integers(min_value=1, max_value=4),
        ),
        # ('add_model', model_id, token, replica_id)
        st.tuples(
            st.just('add_model'),
            st.text(min_size=1, max_size=4, alphabet='abcdefg'),
            st.sampled_from(['t1', 't2']),
            st.sampled_from(['r1', 'r2', None]),
        ),
        # ('config_set', key, value)
        st.tuples(
            st.just('config_set'),
            st.sampled_from(['k1', 'k2', 'k3']),
            st.integers(min_value=0, max_value=99),
        ),
    ),
    min_size=0,
    max_size=12,
)


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(ops=_OP_STRATEGY)
@pytest.mark.asyncio
async def test_property_25_state_operation_equivalence(ops: list[tuple]) -> None:
    """Two ``ServerState`` instances driven by the same op stream agree.

    Two instances bound to one shared backend must agree on every read after
    the same sequence of writes — this is the equivalence the actor used to
    enforce, now provided by the shared backend itself (R19.3).
    """
    backend = MemoryBackend()
    a = ServerState(backend=backend)
    b = ServerState(backend=backend)
    seen_models: set[str] = set()
    for op in ops:
        kind = op[0]
        if kind == 'register_replica':
            _, rid, mx = op
            await a.register_replica(rid, mx)
        elif kind == 'add_model':
            _, mid, token, rid = op
            if mid in seen_models:
                continue
            seen_models.add(mid)
            await a.register_model({'base_model': 'x'}, token=token, model_id=mid, replica_id=rid)
        elif kind == 'config_set':
            _, k, v = op
            await b.add_config(k, v)

    # Both instances see the same persisted view.
    assert await a.get_capacity_info() == await b.get_capacity_info()
    for k in ('k1', 'k2', 'k3'):
        assert await a.get_config(k) == await b.get_config(k)


# ---------- ReplicaRegistry direct ---------------------------------------- #


@pytest.mark.asyncio
async def test_replica_registry_round_trip() -> None:
    backend = MemoryBackend()
    reg = ReplicaRegistry(backend)
    await reg.register('r1', 4)
    await reg.register('r2', 7)
    assert await reg.get_max_loras('r1') == 4
    assert await reg.get_max_loras('r2') == 7
    assert await reg.get_max_loras('unknown') is None
    all_ = await reg.get_all()
    assert all_ == {'r1': 4, 'r2': 7}
    await reg.unregister('r1')
    assert await reg.get_max_loras('r1') is None


@pytest.mark.asyncio
async def test_cross_instance_visibility_in_process() -> None:
    """Two ``ServerState`` instances on one shared MemoryBackend see the same writes (R19.4 in-process)."""
    backend = MemoryBackend()
    a = ServerState(backend=backend)
    b = ServerState(backend=backend)
    await a.register_replica('r1', 3)
    await a.register_model({'base_model': 'x'}, token='t1', model_id='m1', replica_id='r1')
    info_b = await b.get_capacity_info()
    assert info_b == {'max_loras': 3, 'used_loras': 1, 'free_loras': 2}
    avail_b = await b.get_available_replica_ids(['r1'])
    assert avail_b == ['r1']
