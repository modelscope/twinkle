# Copyright (c) ModelScope Contributors. All rights reserved.
"""The single declaration point for Long_Poll_Window and the retrieve poll interval.

Both retrieve endpoints -- twinkle's ``POST /twinkle/retrieve_future`` and tinker's
``POST /retrieve_future`` -- read their timing from here so there is one source of truth,
and both long-poll at the same fixed interval.

Why fixed and not exponential backoff (measured, and it overturned the original design):
the spec's argument for backoff was "a 5ms zero_grad that missed the Inline_Fast_Path
window should not wait out a full 500ms tick". On real hardware that case does not exist
-- control-plane ops measure 0.00-0.09s and are absorbed by the 50ms inline window, so
they never reach this endpoint at all. What does reach it is ``forward_backward`` at
0.52-0.65s, and there a 0.05->1.0s doubling schedule checks at 0.05/0.10/0.20/0.40/0.80,
landing on 0.80 for the whole cluster, whereas a fixed 0.5s checks at 0.05/0.55/1.05 and
catches most of it at 0.55. Backoff measured ~22% SLOWER per step (0.980s vs 0.800s mean)
because its interval grows fastest exactly across the band where real tasks finish.

A denser ceiling (~0.2s) would beat both, at 2.5x the poll rate against a shared state
backend. That is a tuning knob, not a correctness one, and it is not worth optimising for
small-model step times: at production scale a data-plane call runs for minutes and any of
these granularities is noise.
"""
from __future__ import annotations

import os

from twinkle.utils.logger import get_logger

logger = get_logger()

# Documented assumption for a typical ingress / L7 gateway idle-connection limit.
# It is NOT hard-coded into any decision logic -- it is only the threshold at which
# ``long_poll_window()`` warns that a configured window is likely to be cut off.
# Operators can override the real limit for their deployment.
_ASSUMED_GATEWAY_IDLE_LIMIT = 60.0

# Default Long_Poll_Window. 30 < assumed gateway limit (60) and 30 < client HTTP
# timeout (90), so a retrieve request that waits a full window survives the gateway.
_DEFAULT_LONG_POLL_TIMEOUT = 30.0

# Fixed poll interval for BOTH retrieve endpoints (env ``TWINKLE_POLL_INTERVAL``, default
# 0.5s). Declared here so neither endpoint reads ``os.environ`` on its own. See the module
# docstring for the measurement that rejected exponential backoff.
_DEFAULT_POLL_INTERVAL = 0.5

# The last window value we warned about. ``long_poll_window()`` is on the hot path of both
# retrieve endpoints -- not just startup -- so an unguarded warning would fire on every
# retrieve request (roughly twice a second during training) to say something that only
# needs saying once. Keyed on the value rather than a bare bool so that a *changed*
# misconfiguration warns again instead of being swallowed by the first one.
_warned_window: float | None = None


def long_poll_window() -> float:
    """Return the Long_Poll_Window in seconds (env ``TWINKLE_LONG_POLL_TIMEOUT``).

    Warns **once per configured value** when that value is at least the assumed gateway
    idle limit: the retrieve endpoint is itself served through the gateway, so a window
    past the gateway's limit would recreate the connection-cut problem this spec removes.

    The env var is re-read on every call (callers may change it, and tests do), so the
    warning needs its own de-duplication -- this function runs per retrieve request, not
    only at startup.
    """
    global _warned_window
    value = float(os.environ.get('TWINKLE_LONG_POLL_TIMEOUT', str(_DEFAULT_LONG_POLL_TIMEOUT)))
    if value >= _ASSUMED_GATEWAY_IDLE_LIMIT and value != _warned_window:
        _warned_window = value
        logger.warning(
            '[poll_config] TWINKLE_LONG_POLL_TIMEOUT=%.1fs is >= the assumed gateway idle limit '
            '(%.1fs). The retrieve endpoint is served through the gateway too, so a window this '
            'large may be cut off mid-request. Lower it or confirm your gateway idle limit.', value,
            _ASSUMED_GATEWAY_IDLE_LIMIT)
    return value


def retrieve_poll_interval() -> float:
    """Fixed poll interval shared by both retrieve endpoints (env ``TWINKLE_POLL_INTERVAL``)."""
    return float(os.environ.get('TWINKLE_POLL_INTERVAL', str(_DEFAULT_POLL_INTERVAL)))
