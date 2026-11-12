# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared header constants and builder for Ray Serve sticky routing.

Ray Serve 2.55+ reads ``x-request-id`` / ``serve_multiplexed_model_id``
(constants from ``ray/serve/_private/constants.py``).

The multiplexed-model-id header is sent in BOTH underscore and hyphen
forms because HTTP proxies (nginx / ALB) convert underscores to hyphens,
and ``ray_serve_patch.py`` normalises hyphens back for Ray Serve's exact
match.  Dropping either form would break sticky routing in one
environment.
"""

# -- request id --
H_REQUEST_ID = 'x-request-id'
H_REQUEST_ID_LEGACY = 'X-Ray-Serve-Request-Id'

# -- multiplexed model id (sticky routing) --
H_MULTIPLEX = 'serve_multiplexed_model_id'
H_MULTIPLEX_LEGACY = 'Serve-Multiplexed-Model-Id'

# -- auth --
H_AUTH = 'Authorization'
H_AUTH_TWINKLE = 'Twinkle-Authorization'

_ROUTING_HEADERS = (H_REQUEST_ID, H_REQUEST_ID_LEGACY, H_MULTIPLEX, H_MULTIPLEX_LEGACY)
_AUTH_HEADERS = (H_AUTH, H_AUTH_TWINKLE)


def build_routing_headers(request_id: str, auth: str = '') -> dict[str, str]:
    """Build headers for Ray Serve sticky routing + Twinkle auth."""
    headers = {name: request_id for name in _ROUTING_HEADERS}
    for name in _AUTH_HEADERS:
        headers[name] = auth
    return headers
