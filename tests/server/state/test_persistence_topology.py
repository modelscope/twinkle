# Copyright (c) ModelScope Contributors. All rights reserved.
"""Persistence topology tests for the actor-wrapped RayActorBackend.

``RayActorBackend`` owns a detached Ray actor on purpose so all Ray Serve
workers share one consistent in-memory store. These tests assert that
``get_server_state`` returns a process-local cached instance and that two
``ServerState`` instances on the same backend agree under the same op stream.

Also pins in-process RayActorBackend behaviour — no external Redis is required;
sharing happens through the detached actor.
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from twinkle.server.state import (PersistenceConfig, ReplicaRegistry, ServerState, get_server_state,
                                  reset_server_state_cache)
from twinkle.server.state.backend.memory_backend import RayActorBackend


def test_get_server_state_caches_per_process() -> None:
    reset_server_state_cache()
    a = get_server_state(actor_name='cache-a', backend=RayActorBackend())
    b = get_server_state(actor_name='cache-a')
    assert a is b


def test_get_server_state_separate_keys_yield_separate_instances() -> None:
    reset_server_state_cache()
    a = get_server_state(actor_name='k1', backend=RayActorBackend())
    b = get_server_state(actor_name='k2', backend=RayActorBackend())
    assert a is not b


def test_in_process_persistence_no_redis_required() -> None:
    """``PersistenceConfig`` defaults to memory mode and ``ServerState`` works
    without an external Redis.

    The mode now reaches the shared ``_StateActor`` over Ray, but from the
    caller's perspective it remains "no external service needed".
    """
    reset_server_state_cache()
    cfg = PersistenceConfig()  # mode == 'memory'
    state = get_server_state(actor_name='no-redis', persistence_config=cfg)
    assert isinstance(state, ServerState)


# ---------- 4.5: state-operation equivalence under shared backend --------- #

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


@settings(
    max_examples=100,
    deadline=None,  # actor RPC adds variable latency vs the old in-process dict
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(ops=_OP_STRATEGY)
@pytest.mark.asyncio
async def test_property_25_state_operation_equivalence(ops: list[tuple]) -> None:
    """Two ``ServerState`` instances driven by the same op stream agree.

    Two instances bound to the same backend (i.e. forwarding to the same
    detached actor) must agree on every read after the same sequence of
    writes — the shared backend itself is the coordination point.
    """
    backend = RayActorBackend()
    # Hypothesis reuses the same function scope across all examples, so the
    # autouse conftest fixture only clears the actor once. Reset between
    # examples here to keep this test independent (per-token model limit
    # would otherwise trip after ~30 accumulated models).
    await backend.close()
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
    backend = RayActorBackend()
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
async def test_two_states_share_backend_in_process() -> None:
    """Two ``ServerState`` instances on one shared RayActorBackend see each other's writes.

    Sharing is via the detached ``_StateActor`` rather than an in-process dict.
    """
    backend = RayActorBackend()
    a = ServerState(backend=backend)
    b = ServerState(backend=backend)
    await a.register_replica('r1', 3)
    await a.register_model({'base_model': 'x'}, token='t1', model_id='m1', replica_id='r1')
    info_b = await b.get_capacity_info()
    assert info_b == {'max_loras': 3, 'used_loras': 1, 'free_loras': 2}
    avail_b = await b.get_available_replica_ids(['r1'])
    assert avail_b == ['r1']
