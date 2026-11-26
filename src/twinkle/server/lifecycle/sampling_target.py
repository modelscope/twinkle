# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resolve Tinker sampling selectors before routing or changing weights."""
from __future__ import annotations
from typing import TYPE_CHECKING

from twinkle.server.exceptions import RequestRejectedError, ResourceNotFoundError

if TYPE_CHECKING:
    from twinkle.server.state import ServerState


async def resolve_tinker_sampling_target(
    state: ServerState,
    *,
    base_model: str | None,
    model_path: str | None,
    sampling_session_id: str | None,
) -> tuple[str | None, str | None]:
    if not sampling_session_id:
        return base_model, model_path
    session = await state.get_sampling_session(sampling_session_id)
    if session is None:
        raise ResourceNotFoundError('Unknown sampling session.')
    session_model = session.get('base_model')
    session_path = session.get('model_path')
    if base_model and session_model and base_model != session_model:
        raise RequestRejectedError('base_model conflicts with the sampling session.')
    if model_path and session_path and model_path != session_path:
        raise RequestRejectedError('model_path conflicts with the sampling session.')
    return session_model or base_model, session_path or model_path
