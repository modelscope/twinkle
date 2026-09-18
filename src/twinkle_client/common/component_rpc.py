# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared plumbing for the processor-backed component clients.

Dataset / dataloader / processor wrappers all talk to the same two generic endpoints,
so they share one place that builds those two request bodies. Before this, each method
spread its arguments over the top level of a hand-built dict, which made the envelope
indistinguishable from its payload -- a misspelt ``processor_id`` was just another
argument. Nesting the payload under the declared passthrough region is what lets the
envelope be strict.
"""
from __future__ import annotations

from typing import Any

from twinkle_client._request_builder import build_request
from twinkle_client.http import get_base_url, http_post_model
from twinkle_client.types.processor import (ProcessorCallRequest, ProcessorCallResponse, ProcessorCreateRequest,
                                            ProcessorCreateResponse)

# Sentinel: "caller did not pass an HTTP timeout", so the shared default applies. A
# literal ``None`` means "no timeout at all" in ``http_post_model``, so it cannot double
# as the unset marker.
_UNSET = object()


def processor_base_url() -> str:
    """The single processor route prefix used by every component client."""
    return f'{get_base_url()}/processor/twinkle'


def create_remote_component(processor_type: str, class_type: str, **init_kwargs: Any) -> str:
    """Create a server-side component and return its ``pid:``-prefixed id."""
    body = build_request(ProcessorCreateRequest, processor_type=processor_type, class_type=class_type, **init_kwargs)
    response = http_post_model(f'{processor_base_url()}/create', body)
    return ProcessorCreateResponse(**response.json()).processor_id


def call_remote_component(processor_id: str, function: str, http_timeout: Any = _UNSET, /, **call_kwargs: Any) -> Any:
    """Invoke ``function`` on a server-side component and return its result.

    ``http_timeout`` is positional-only so it can never be mistaken for -- or collide
    with -- one of the remote callable's own arguments, which are all keywords.

    ``StopIteration`` propagates from the HTTP layer on an exhausted iterator (the
    server answers 410), which is what makes a remote ``__next__`` usable in a plain
    ``for`` loop.
    """
    body = build_request(ProcessorCallRequest, processor_id=processor_id, function=function, **call_kwargs)
    url = f'{processor_base_url()}/call'
    response = (
        http_post_model(url, body) if http_timeout is _UNSET else http_post_model(url, body, timeout=http_timeout))
    return ProcessorCallResponse(**response.json()).result
