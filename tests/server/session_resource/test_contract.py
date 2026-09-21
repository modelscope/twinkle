from __future__ import annotations

import asyncio
from contextlib import suppress
from unittest import mock

import pytest

from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.session_resource.adapter import AdapterManagerMixin
from twinkle.server.session_resource.base import SessionResourceMixin
from twinkle.server.session_resource.processor import ProcessorManagerMixin


class _State:

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])

    async def get_session_last_heartbeat(self, session_id: str) -> float | None:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


class _ResourceManager(SessionResourceMixin):

    def __init__(self, state: _State, timeout: float = 10.0) -> None:
        self.state = state
        self.expired: list[str] = []
        self._init_resource_manager(resource_timeout=timeout)

    async def _on_resource_expired(self, resource_id: str) -> None:
        self.expired.append(resource_id)


class _MissingBaseHook(SessionResourceMixin):
    pass


class _MissingAdapterHook(AdapterManagerMixin):
    pass


class _MissingProcessorHook(ProcessorManagerMixin):
    pass


def test_all_resource_expiry_hooks_are_abstract() -> None:
    for cls in (_MissingBaseHook, _MissingAdapterHook, _MissingProcessorHook):
        with pytest.raises(TypeError):
            cls()


def test_registration_requires_session_id() -> None:
    manager = _ResourceManager(_State())
    with pytest.raises(RequestRejectedError):
        manager.register_resource('r1', 'token', '')


@pytest.mark.asyncio
async def test_liveness_failure_has_hard_upper_bound_and_recovery_refreshes() -> None:
    state = _State([RuntimeError('down'), 109.0, RuntimeError('down'), RuntimeError('down')])
    manager = _ResourceManager(state, timeout=10.0)
    with mock.patch('twinkle.server.session_resource.base.time.time', return_value=100.0):
        manager.register_resource('r1', 'token', 'session')
    record = manager.get_resource_info('r1')

    with mock.patch('twinkle.server.session_resource.base.time.time', return_value=105.0):
        assert await manager._is_session_alive('session', 'r1', record) is True
    with mock.patch('twinkle.server.session_resource.base.time.time', return_value=110.0):
        assert await manager._is_session_alive('session', 'r1', record) is True
    assert manager.get_resource_info('r1')['last_liveness_confirmed_at'] == 110.0
    with mock.patch('twinkle.server.session_resource.base.time.time', return_value=115.0):
        assert await manager._is_session_alive('session', 'r1', record) is True
    with mock.patch('twinkle.server.session_resource.base.time.time', return_value=120.0):
        assert await manager._is_session_alive('session', 'r1', record) is False


@pytest.mark.asyncio
async def test_countdown_restart_preserves_confirmation_time() -> None:
    manager = _ResourceManager(_State([100.0]))
    with mock.patch('twinkle.server.session_resource.base.time.time', return_value=100.0):
        manager.register_resource('r1', 'token', 'session')
    confirmed_at = manager.get_resource_info('r1')['last_liveness_confirmed_at']

    manager._ensure_countdown_started()
    first_task = manager._countdown_task
    manager.stop_resource_countdown()
    with suppress(asyncio.CancelledError):
        await first_task

    manager._ensure_countdown_started()
    second_task = manager._countdown_task
    assert second_task is not first_task
    assert manager.get_resource_info('r1')['last_liveness_confirmed_at'] == confirmed_at
    manager.stop_resource_countdown()
    with suppress(asyncio.CancelledError):
        await second_task


def test_worker_restart_does_not_restore_local_resources() -> None:
    old = _ResourceManager(_State())
    old.register_resource('r1', 'token', 'session')
    restarted = _ResourceManager(_State())
    assert restarted.get_resource_info('r1') is None
