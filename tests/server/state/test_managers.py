"""Tests for state managers using RayActorBackend as integration backend."""
from __future__ import annotations

import asyncio
import pytest
import time
from datetime import datetime, timezone
from unittest import mock

from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend
from twinkle.server.state.future_manager import FutureManager
from twinkle.server.state.model_manager import ModelManager
from twinkle.server.state.models import FutureRecord, ModelRecord, SamplingSessionRecord, SessionRecord
from twinkle.server.state.sampling_manager import SamplingSessionManager
from twinkle.server.state.session_manager import SessionManager

# ============================================================
# SessionManager Tests
# ============================================================


class TestSessionManager:

    @pytest.fixture
    def backend(self):
        return RayActorBackend()

    @pytest.fixture
    def manager(self, backend):
        return SessionManager(backend=backend, expiration_timeout=300.0)

    @pytest.mark.asyncio
    async def test_add_and_get(self, manager):
        record = SessionRecord(tags=['test'], sdk_version='1.0')
        await manager.add('sess1', record)
        result = await manager.get('sess1')
        assert result is not None
        assert result.tags == ['test']
        assert result.sdk_version == '1.0'

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, manager):
        result = await manager.get('nonexistent')
        assert result is None

    @pytest.mark.asyncio
    async def test_remove(self, manager):
        record = SessionRecord()
        await manager.add('sess1', record)
        removed = await manager.remove('sess1')
        assert removed is True
        assert await manager.get('sess1') is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, manager):
        removed = await manager.remove('nonexistent')
        assert removed is False

    @pytest.mark.asyncio
    async def test_count(self, manager):
        await manager.add('s1', SessionRecord())
        await manager.add('s2', SessionRecord())
        assert await manager.count() == 2

    @pytest.mark.asyncio
    async def test_touch_updates_heartbeat(self, manager):
        record = SessionRecord(last_heartbeat=1000.0)
        await manager.add('sess1', record)
        before = time.time()
        result = await manager.touch('sess1')
        after = time.time()
        assert result is True
        updated = await manager.get('sess1')
        assert before <= updated.last_heartbeat <= after

    @pytest.mark.asyncio
    async def test_touch_nonexistent(self, manager):
        result = await manager.touch('nonexistent')
        assert result is False

    @pytest.mark.asyncio
    async def test_get_last_heartbeat(self, manager):
        record = SessionRecord(last_heartbeat=12345.0)
        await manager.add('sess1', record)
        hb = await manager.get_last_heartbeat('sess1')
        assert hb == 12345.0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, manager):
        now = time.time()
        # Old session
        old_record = SessionRecord(last_heartbeat=now - 1000)
        await manager.add('old_sess', old_record)
        # Recent session
        new_record = SessionRecord(last_heartbeat=now)
        await manager.add('new_sess', new_record)

        cutoff = now - 500
        removed_count = await manager.cleanup_expired(cutoff)
        assert removed_count == 1
        assert await manager.get('old_sess') is None
        assert await manager.get('new_sess') is not None

    @pytest.mark.asyncio
    async def test_cleanup_expired_uses_created_at_fallback(self, manager):
        """When last_heartbeat is 0, should use created_at for expiry check."""
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        record = SessionRecord(last_heartbeat=0.0, created_at=old_time)
        await manager.add('old_sess', record)

        cutoff = time.time() - 100
        removed_count = await manager.cleanup_expired(cutoff)
        assert removed_count == 1


# ============================================================
# ModelManager Tests
# ============================================================


class TestModelManager:

    @pytest.fixture
    def backend(self):
        return RayActorBackend()

    @pytest.fixture
    def manager(self, backend):
        return ModelManager(backend=backend, expiration_timeout=300.0, per_token_model_limit=3)

    @pytest.mark.asyncio
    async def test_add_and_get(self, manager):
        record = ModelRecord(token='tok1', session_id='sess1', base_model='qwen')
        await manager.add('model1', record)
        result = await manager.get('model1')
        assert result is not None
        assert result.token == 'tok1'
        assert result.base_model == 'qwen'

    @pytest.mark.asyncio
    async def test_remove(self, manager):
        record = ModelRecord(token='tok1')
        await manager.add('model1', record)
        removed = await manager.remove('model1')
        assert removed is True
        assert await manager.get('model1') is None

    @pytest.mark.asyncio
    async def test_token_limit_enforced(self, manager):
        """Adding more models than per_token_model_limit should raise RuntimeError."""
        for i in range(3):
            await manager.add(f'm{i}', ModelRecord(token='tok1'))

        with pytest.raises(RuntimeError, match='Model limit exceeded'):
            await manager.add('m3', ModelRecord(token='tok1'))

    @pytest.mark.asyncio
    async def test_token_limit_per_token(self, manager):
        """Limit is per-token, different tokens have separate limits."""
        for i in range(3):
            await manager.add(f'a{i}', ModelRecord(token='tokenA'))
        # Different token should work
        await manager.add('b0', ModelRecord(token='tokenB'))
        assert await manager.get('b0') is not None

    @pytest.mark.asyncio
    async def test_replica_registration(self, manager):
        await manager.register_replica('replica1', max_loras=5)
        info = await manager.get_capacity_info()
        assert info['max_loras'] == 5
        assert info['used_loras'] == 0
        assert info['free_loras'] == 5

    @pytest.mark.asyncio
    async def test_capacity_info_after_add(self, manager):
        await manager.register_replica('r1', max_loras=3)
        record = ModelRecord(token='tok1', replica_id='r1')
        await manager.add('m1', record)
        info = await manager.get_capacity_info()
        assert info['used_loras'] == 1
        assert info['free_loras'] == 2

    @pytest.mark.asyncio
    async def test_indexes_derived_from_backend(self, manager):
        """Per-token / per-replica counts are derived from the backend."""
        record1 = ModelRecord(token='tok1', replica_id='r1')
        record2 = ModelRecord(token='tok1', replica_id='r2')
        await manager.add('m1', record1)
        await manager.add('m2', record2)

        await manager.register_replica('r1', max_loras=5)
        await manager.register_replica('r2', max_loras=5)

        # Backend-derived availability reflects all persisted records.
        avail = await manager.get_available_replica_ids(['r1', 'r2'])
        assert avail == ['r1', 'r2']

        # Per-token count enforces the limit using the persisted records.
        count = await manager._count_models_for_token('tok1')
        assert count == 2

    @pytest.mark.asyncio
    async def test_cascade_cleanup_by_session(self, manager):
        """Models owned by expired sessions should be cleaned up."""
        now = time.time()
        record = ModelRecord(
            token='tok1',
            session_id='expired_sess',
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        await manager.add('m1', record)

        # Cleanup with cascade
        removed = await manager.cleanup_expired(
            cutoff_time=now - 10000,  # cutoff is old, so age-based wouldn't trigger
            expired_session_ids=['expired_sess'],
        )
        assert removed == 1
        assert await manager.get('m1') is None

    @pytest.mark.asyncio
    async def test_get_available_replica_ids(self, manager):
        await manager.register_replica('r1', max_loras=2)
        await manager.register_replica('r2', max_loras=1)
        # Fill r2
        await manager.add('m1', ModelRecord(token='t', replica_id='r2'))

        available = await manager.get_available_replica_ids(['r1', 'r2', 'r3_unknown'])
        # r1 has capacity, r2 is full, r3 unknown (conservative include)
        assert 'r1' in available
        assert 'r2' not in available
        assert 'r3_unknown' in available


# ============================================================
# Model Limit Race (merged from test_model_limit_race)
# ============================================================


class TestModelLimitRace:

    @pytest.mark.asyncio
    async def test_concurrent_adds_same_token_respect_limit(self):
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
        models = await state._model_mgr.get_all()
        tok_models = [m for m in models.values() if m.token == 'tok']
        assert len(tok_models) == accepted

    @pytest.mark.asyncio
    async def test_remove_frees_token_slot(self):
        backend = RayActorBackend()
        state = ServerState(backend=backend, per_token_model_limit=1)

        await state.register_model({'base_model': 'b'}, token='tok', model_id='m1')
        with pytest.raises(RuntimeError):
            await state.register_model({'base_model': 'b'}, token='tok', model_id='m2')

        assert await state.unload_model('m1') is True
        await state.register_model({'base_model': 'b'}, token='tok', model_id='m3')
        models = await state._model_mgr.get_all()
        assert {mid for mid in models} == {'m3'}

    @pytest.mark.asyncio
    async def test_rebuild_indexes_recovers_counter_from_records(self):
        backend = RayActorBackend()
        state = ServerState(backend=backend, per_token_model_limit=3)

        await state.register_model({'base_model': 'b'}, token='tok', model_id='m1')
        await state.register_model({'base_model': 'b'}, token='tok', model_id='m2')

        await state._model_mgr.rebuild_indexes()

        await state.register_model({'base_model': 'b'}, token='tok', model_id='m3')
        with pytest.raises(RuntimeError):
            await state.register_model({'base_model': 'b'}, token='tok', model_id='m4')


# ============================================================
# SamplingSessionManager Tests
# ============================================================


class TestSamplingSessionManager:

    @pytest.fixture
    def backend(self):
        return RayActorBackend()

    @pytest.fixture
    def manager(self, backend):
        return SamplingSessionManager(backend=backend, expiration_timeout=300.0)

    @pytest.mark.asyncio
    async def test_add_and_get(self, manager):
        record = SamplingSessionRecord(session_id='sess1', base_model='qwen')
        await manager.add('samp1', record)
        result = await manager.get('samp1')
        assert result is not None
        assert result.session_id == 'sess1'
        assert result.base_model == 'qwen'

    @pytest.mark.asyncio
    async def test_cleanup_expired_by_age(self, manager):
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        record = SamplingSessionRecord(session_id='sess1', created_at=old_time)
        await manager.add('samp_old', record)

        # Recent
        record2 = SamplingSessionRecord(session_id='sess2')
        await manager.add('samp_new', record2)

        cutoff = time.time() - 100
        removed = await manager.cleanup_expired(cutoff)
        assert removed == 1
        assert await manager.get('samp_old') is None
        assert await manager.get('samp_new') is not None

    @pytest.mark.asyncio
    async def test_cleanup_expired_cascade(self, manager):
        """Sampling sessions should be cleaned when their parent session expires."""
        record = SamplingSessionRecord(session_id='expired_sess')
        await manager.add('samp1', record)

        removed = await manager.cleanup_expired(
            cutoff_time=0.0,  # Won't catch by age
            expired_session_ids=['expired_sess'],
        )
        assert removed == 1
        assert await manager.get('samp1') is None


# ============================================================
# FutureManager Tests
# ============================================================


class TestFutureManager:

    @pytest.fixture
    def backend(self):
        return RayActorBackend()

    @pytest.fixture
    def manager(self, backend):
        return FutureManager(backend=backend, expiration_timeout=300.0)

    @pytest.mark.asyncio
    async def test_store_status_creates_new(self, manager):
        await manager.store_status(
            request_id='req1',
            status='pending',
            model_id='model1',
        )
        result = await manager.get('req1')
        assert result is not None
        assert result.status == 'pending'
        assert result.model_id == 'model1'

    @pytest.mark.asyncio
    async def test_store_status_updates_existing(self, manager):
        await manager.store_status(request_id='req1', status='pending', model_id='m1')
        await manager.store_status(request_id='req1', status='completed', model_id='m1', result={'output': 'done'})
        result = await manager.get('req1')
        assert result.status == 'completed'
        assert result.result == {'output': 'done'}

    @pytest.mark.asyncio
    async def test_store_status_with_pydantic_result(self, manager):
        """Pydantic models should be serialized via model_dump."""
        from pydantic import BaseModel

        class MockResult(BaseModel):
            score: float = 0.95

        await manager.store_status(request_id='req1', status='completed', model_id='m1', result=MockResult())
        result = await manager.get('req1')
        assert result.result == {'score': 0.95}

    @pytest.mark.asyncio
    async def test_store_status_preserves_reason(self, manager):
        await manager.store_status(request_id='req1', status='rate_limited', model_id=None, reason='Too many requests')
        result = await manager.get('req1')
        assert result.reason == 'Too many requests'

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, manager):
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
        old_record = FutureRecord(status='completed', created_at=old_time, updated_at=old_time)
        await manager.add('old_req', old_record)

        new_record = FutureRecord(status='pending')
        await manager.add('new_req', new_record)

        cutoff = time.time() - 100
        removed = await manager.cleanup_expired(cutoff)
        assert removed == 1
        assert await manager.get('old_req') is None
        assert await manager.get('new_req') is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, manager):
        result = await manager.get('nonexistent')
        assert result is None

    @pytest.mark.asyncio
    async def test_store_status_queue_state(self, manager):
        await manager.store_status(
            request_id='req1',
            status='queued',
            model_id='m1',
            queue_state='paused_rate_limit',
            queue_state_reason='Rate limit hit')
        result = await manager.get('req1')
        assert result.queue_state == 'paused_rate_limit'
        assert result.queue_state_reason == 'Rate limit hit'


# ============================================================
# Cascade Cleanup Consistency (merged from test_cleanup_cascade_consistency)
# ============================================================


class TestCascadeCleanup:

    @pytest.mark.asyncio
    async def test_cascade_set_matches_removed_sessions(self):
        backend = RayActorBackend()
        state = ServerState(backend=backend, expiration_timeout=0.0)
        sid = await state.create_session({'session_id': 's-expired'})
        await state.register_model({'base_model': 'b'}, token='t1', model_id='m1', session_id=sid)
        time.sleep(0.01)
        stats = await state.cleanup_expired_resources()
        assert stats['sessions'] == 1
        assert stats['models'] == 1
        assert await state.get_session_last_heartbeat('s-expired') is None
        assert await state.get_model_metadata('m1') is None

    @pytest.mark.asyncio
    async def test_touch_between_scans_cannot_orphan_children(self):
        backend = RayActorBackend()
        state = ServerState(backend=backend, expiration_timeout=0.0)
        sid = await state.create_session({'session_id': 's-race'})
        await state.register_model({'base_model': 'b'}, token='t1', model_id='m-race', session_id=sid)
        time.sleep(0.01)

        session_mgr = state._session_mgr
        real_get_all = session_mgr.get_all
        touched = {'done': False}

        async def get_all_then_touch():
            records = await real_get_all()
            if not touched['done']:
                touched['done'] = True
                await session_mgr.touch(sid)
            return records

        with mock.patch.object(session_mgr, 'get_all', side_effect=get_all_then_touch):
            await state.cleanup_expired_resources()

        session_alive = await state.get_session_last_heartbeat('s-race') is not None
        model_alive = await state.get_model_metadata('m-race') is not None
        assert session_alive == model_alive
