# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-token model-limit race regression tests (Requirement 18).

Pins the atomic-counter fix: ``ModelManager.add`` enforces the per-token model
limit through ``StateBackend.update_atomic`` instead of a separate
count-then-add sequence, so N concurrent adds with the same token against a
limit of L register at most L models (the prior race let two concurrent adds
both observe ``L - 1`` and both succeed).
"""
from __future__ import annotations

import asyncio
import pytest

from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend


@pytest.mark.asyncio
async def test_concurrent_adds_same_token_respect_limit() -> None:
    backend = RayActorBackend()
    limit = 5
    state = ServerState(backend=backend, per_token_model_limit=limit)

    n = 25
    results: list[bool] = []

    async def try_add(i: int) -> None:
        try:
            await state.register_model({'base_model': 'b'}, token='tok', model_id=f'm{i}')
            results.append(True)
        except RuntimeError:
            results.append(False)

    await asyncio.gather(*(try_add(i) for i in range(n)))

    accepted = results.count(True)
    assert accepted <= limit, f'{accepted} models registered for one token, exceeds limit {limit}'
    # The persisted model records must agree with the accept count.
    models = await state._model_mgr.get_all()
    tok_models = [m for m in models.values() if m.token == 'tok']
    assert len(tok_models) == accepted


@pytest.mark.asyncio
async def test_remove_frees_token_slot() -> None:
    backend = RayActorBackend()
    state = ServerState(backend=backend, per_token_model_limit=1)

    await state.register_model({'base_model': 'b'}, token='tok', model_id='m1')
    with pytest.raises(RuntimeError):
        await state.register_model({'base_model': 'b'}, token='tok', model_id='m2')

    # Removing the first model frees the single slot.
    assert await state.unload_model('m1') is True
    # Now another add for the same token succeeds.
    await state.register_model({'base_model': 'b'}, token='tok', model_id='m3')
    models = await state._model_mgr.get_all()
    assert {mid for mid in models} == {'m3'}


@pytest.mark.asyncio
async def test_rebuild_indexes_recovers_counter_from_records() -> None:
    """The per-token counter is rebuilt from the persisted records on start, so
    a stale/cleared counter is corrected rather than blocking or over-counting."""
    backend = RayActorBackend()
    state = ServerState(backend=backend, per_token_model_limit=3)

    await state.register_model({'base_model': 'b'}, token='tok', model_id='m1')
    await state.register_model({'base_model': 'b'}, token='tok', model_id='m2')

    # Simulate a stale counter (e.g. after a crash) then rebuild from records.
    await state._model_mgr.rebuild_indexes()

    # Counter now reflects exactly the 2 persisted records; a 3rd add is allowed,
    # a 4th is rejected.
    await state.register_model({'base_model': 'b'}, token='tok', model_id='m3')
    with pytest.raises(RuntimeError):
        await state.register_model({'base_model': 'b'}, token='tok', model_id='m4')
