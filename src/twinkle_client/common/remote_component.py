# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared transport binding for processor-backed remote component wrappers."""
from __future__ import annotations
from typing import Any

from twinkle.protocol.types.processor import (
    ProcessorCallRequest,
    ProcessorCallResponse,
    ProcessorCreateRequest,
    ProcessorCreateResponse,
)
from twinkle_client._request_builder import build_request
from twinkle_client.http import ClientTransport
from twinkle_client.http.client import DEFAULT_TIMEOUT
from twinkle_client.http.context import capture_transport


class RemoteComponent:
    """Bind one remote component id to one captured transport.

    The class intentionally has no ``__init__``: concrete wrappers retain
    ownership of their domain-specific constructor and MRO.
    """

    _transport: ClientTransport
    processor_id: str

    def _bind_remote(
        self,
        processor_type: str,
        class_type: str,
        *,
        transport: ClientTransport | None = None,
        **kwargs: Any,
    ) -> None:
        self._transport = capture_transport(transport)
        body = build_request(ProcessorCreateRequest, processor_type=processor_type, class_type=class_type, **kwargs)
        response = self._transport.post_model(self._transport.url('processor/twinkle/create'), body)
        self.processor_id = ProcessorCreateResponse(**response.json()).processor_id

    def _call(self, function: str, http_timeout: Any = DEFAULT_TIMEOUT, /, **kwargs: Any) -> Any:
        body = build_request(ProcessorCallRequest, processor_id=self.processor_id, function=function, **kwargs)
        response = self._transport.post_model(self._transport.url('processor/twinkle/call'), body, timeout=http_timeout)
        return ProcessorCallResponse(**response.json()).result
