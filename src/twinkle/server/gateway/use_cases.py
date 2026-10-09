# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gateway use cases that do real assembly or control flow.

What is *not* here is the point: ``create_session`` / ``touch_session`` were one-line
forwards to ``state``, so "is it in this file?" told a reader nothing. Now the file holds
only the long-poll loop and the five checkpoint use cases, which wire up
``create_*_manager(token, client_type)`` -- i.e. things a handler cannot express in one
line. The four handlers that call ``self.state`` directly (``get_capacity_info``,
``cancel_future``, ``get_cleanup_stats``, ``get_model_metadata``) deliberately stay
direct: wrapping them for symmetry would add forwarding, not structure.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from twinkle.server.checkpoint import create_checkpoint_manager, create_training_run_manager
from twinkle.server.lifecycle.poll_config import long_poll_window, retrieve_poll_interval

_TERMINAL_STATUSES = frozenset({'completed', 'failed', 'cancelled'})


@dataclass(frozen=True, slots=True)
class FuturePollResult:
    record: dict[str, Any] | None
    timed_out: bool


async def poll_future(state: Any, request_id: str) -> FuturePollResult:
    """Long-poll one canonical future record without constructing wire responses."""
    deadline = asyncio.get_running_loop().time() + long_poll_window()
    interval = retrieve_poll_interval()
    record = None
    while True:
        record = await state.get_future(request_id)
        if record is not None and record.get('status') in _TERMINAL_STATUSES:
            return FuturePollResult(record=record, timed_out=False)
        if asyncio.get_running_loop().time() >= deadline:
            return FuturePollResult(record=record, timed_out=True)
        await asyncio.sleep(interval)


def list_training_runs(token: str, client_type: str, *, limit: int, offset: int) -> Any:
    return create_training_run_manager(token, client_type=client_type).list_runs(limit=limit, offset=offset)


def get_training_run(token: str, client_type: str, run_id: str, *, check_permission: bool = False) -> Any | None:
    manager = create_training_run_manager(token, client_type=client_type)
    if check_permission and hasattr(manager, 'get_with_permission'):
        return manager.get_with_permission(run_id)
    return manager.get(run_id)


def list_checkpoints(token: str, client_type: str, run_id: str) -> Any | None:
    return create_checkpoint_manager(token, client_type=client_type).list_checkpoints(run_id)


def delete_checkpoint(token: str, client_type: str, run_id: str, checkpoint_id: str) -> bool:
    return create_checkpoint_manager(token, client_type=client_type).delete(run_id, checkpoint_id)


def get_weights_info(token: str, client_type: str, path: str) -> Any | None:
    return create_checkpoint_manager(token, client_type=client_type).get_weights_info(path)
