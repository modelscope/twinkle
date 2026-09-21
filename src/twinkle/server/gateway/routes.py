# Copyright (c) ModelScope Contributors. All rights reserved.
"""Route-path constants shared by the gateway's proxy and its OpenAI bridge.

Single source for the ``route_prefix`` convention these three places must agree on; they
previously agreed by having the same string typed out in three files. Placed on
the consumer side (gateway) rather than in ``launcher`` -- launcher *produces* the
``route_prefix``, gateway consumes it -- to avoid a new ``gateway -> launcher`` edge.
"""
from __future__ import annotations

# Downstream endpoint paths the gateway proxies to (relative to the service's route prefix).
TWINKLE_SAMPLE = 'twinkle/sample'
TWINKLE_SAMPLE_STREAM = 'twinkle/sample_stream'
TWINKLE_SET_TEMPLATE = 'twinkle/set_template'

TINKER_PREFIX = 'tinker'


def tinker_endpoint(endpoint: str) -> str:
    """``tinker/<endpoint>`` -- the shape ``proxy_to_model`` / ``proxy_to_sampler`` build."""
    return f'{TINKER_PREFIX}/{endpoint}'


def target_url(route_prefix: str, service_type: str, base_model: str, endpoint: str) -> str:
    """The one definition of ``{route_prefix}/{service_type}/{base_model}/{endpoint}``."""
    return f'{route_prefix.rstrip("/")}/{service_type}/{base_model}/{endpoint}'
