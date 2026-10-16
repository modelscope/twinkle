from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twinkle.server.model.app import MODEL_SELECTOR, ModelManagement


class _CapacityState:

    def __init__(self) -> None:
        self.capacities: dict[str, int] = {}

    async def register_replica(self, replica_id: str, max_loras: int) -> None:
        self.capacities[replica_id] = max_loras

    async def unregister_replica(self, replica_id: str) -> None:
        self.capacities.pop(replica_id, None)

    async def get_capacity_info(self) -> dict[str, int]:
        max_loras = sum(self.capacities.values())
        return {'max_loras': max_loras, 'used_loras': 0, 'free_loras': max_loras}


def _make_lifecycle_manager(state: _CapacityState, replica_id: str, max_loras: int) -> ModelManagement:
    manager = ModelManagement.__new__(ModelManagement)
    manager.state = state
    manager.replica_id = replica_id
    manager.max_loras = max_loras
    manager._replica_registered = False
    manager.data_plane = SimpleNamespace(close=AsyncMock())
    return manager


@pytest.mark.asyncio
async def test_replica_lifecycle_updates_shared_capacity() -> None:
    state = _CapacityState()
    first = _make_lifecycle_manager(state, 'replica-1', 3)
    second = _make_lifecycle_manager(state, 'replica-2', 3)

    await first._register_replica_on_startup()
    assert await state.get_capacity_info() == {'max_loras': 3, 'used_loras': 0, 'free_loras': 3}

    await second._register_replica_on_startup()
    assert await state.get_capacity_info() == {'max_loras': 6, 'used_loras': 0, 'free_loras': 6}

    await second.shutdown()
    assert await state.get_capacity_info() == {'max_loras': 3, 'used_loras': 0, 'free_loras': 3}


@pytest.mark.asyncio
async def test_async_constructor_registers_replica_before_ready() -> None:
    state = SimpleNamespace(register_replica=AsyncMock())
    replica_context = SimpleNamespace(replica_id=SimpleNamespace(unique_id='replica-1'))
    manager = ModelManagement.__new__(ModelManagement)

    with patch('twinkle.server.model.app.DeviceGroup', return_value=SimpleNamespace(name='group')), \
         patch('twinkle.server.model.app.init_twinkle_runtime', return_value=None), \
         patch('twinkle.server.model.app.serve.get_replica_context', return_value=replica_context), \
         patch.object(MODEL_SELECTOR, 'construct', return_value=MagicMock()), \
         patch('twinkle.server.model.app.get_server_state', return_value=state), \
         patch('twinkle.server.data_plane.DataPlaneProxy', return_value=MagicMock()), \
         patch.object(ModelManagement, '_init_task_queue'), \
         patch.object(ModelManagement, '_init_adapter_manager'):
        await ModelManagement.__init__(
            manager,
            model_id='model',
            nproc_per_node=1,
            device_group={'name': 'group'},
            device_mesh={},
            backend='mock',
            max_loras=3,
        )

    state.register_replica.assert_awaited_once_with('replica-1', 3)
    assert manager._replica_registered is True
