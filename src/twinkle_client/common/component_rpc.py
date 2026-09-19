# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared transport-bound plumbing for processor-backed component clients."""
from __future__ import annotations

from typing import Any

from twinkle_client._request_builder import build_request
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport
from twinkle_client.types.processor import (ProcessorCallRequest, ProcessorCallResponse, ProcessorCreateRequest,
                                            ProcessorCreateResponse)

_UNSET = object()


def processor_base_url(transport: ClientTransport) -> str:
    return f'{transport.context.base_url}/processor/twinkle'


def create_remote_component(
    processor_type: str,
    class_type: str,
    *,
    transport: ClientTransport | None = None,
    **init_kwargs: Any,
) -> str:
    """Create a server-side component using one captured transport."""
    resolved = capture_transport(transport)
    body = build_request(ProcessorCreateRequest, processor_type=processor_type, class_type=class_type, **init_kwargs)
    response = resolved.post_model(f'{processor_base_url(resolved)}/create', body)
    return ProcessorCreateResponse(**response.json()).processor_id


def call_remote_component(
    processor_id: str,
    function: str,
    http_timeout: Any = _UNSET,
    /,
    *,
    transport: ClientTransport | None = None,
    **call_kwargs: Any,
) -> Any:
    """Invoke one server-side component using its owner's transport."""
    resolved = capture_transport(transport)
    body = build_request(ProcessorCallRequest, processor_id=processor_id, function=function, **call_kwargs)
    url = f'{processor_base_url(resolved)}/call'
    response = (
        resolved.post_model(url, body) if http_timeout is _UNSET else resolved.post_model(
            url, body, timeout=http_timeout))
    return ProcessorCallResponse(**response.json()).result
