# Copyright (c) ModelScope Contributors. All rights reserved.
"""Request / response models for the twinkle processor endpoints.

The processor surface is a generic RPC bridge: ``create`` names a class to build and
``call`` names a method to invoke, both with caller-supplied arguments whose names are
only known to the target. Those arguments therefore live in declared passthrough
dicts (``init_kwargs`` / ``call_kwargs``) rather than being spread over the top level,
which is what lets the envelope itself be strict -- a misspelt ``processor_id`` or
``function`` now fails instead of being silently treated as an argument.

No ``target``: the callable is resolved from ``processor_type`` + ``class_type`` (or a
live instance plus ``function``), not from a single sibling field, so the passthrough
keys are forwarded unchecked. Claiming otherwise would need a second resolution path
that guesses.

Class names are prefixed with ``Processor`` to avoid collisions when importing from
``twinkle.protocol.types`` alongside ``model.py``.
"""
from __future__ import annotations

from pydantic import JsonValue
from typing import Any, Dict

from .base import ResponseModel, StrictRequest, passthrough


class ProcessorCreateRequest(StrictRequest):
    processor_type: str
    class_type: str
    init_kwargs: dict[str, JsonValue] = passthrough()


class ProcessorHeartbeatRequest(StrictRequest):
    processor_id: str


class ProcessorCallRequest(StrictRequest):
    processor_id: str
    function: str
    call_kwargs: dict[str, JsonValue] = passthrough()


class ProcessorCreateResponse(ResponseModel):
    """Response body for the /create endpoint."""
    processor_id: str


class ProcessorHeartbeatResponse(ResponseModel):
    """Response body for the /heartbeat endpoint."""
    status: str = 'ok'


class ProcessorCallResponse(ResponseModel):
    """Response body for the /call endpoint."""
    result: Any
