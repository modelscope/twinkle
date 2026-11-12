# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared transport binding for processor-backed remote component wrappers."""
from __future__ import annotations

from typing import Any

from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from twinkle_client.http import ClientTransport
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
        self.processor_id = create_remote_component(
            processor_type,
            class_type,
            transport=self._transport,
            **kwargs,
        )

    def _call(self, function: str, *args: Any, **kwargs: Any) -> Any:
        return call_remote_component(
            self.processor_id,
            function,
            *args,
            transport=self._transport,
            **kwargs,
        )
