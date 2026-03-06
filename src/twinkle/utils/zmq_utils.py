# Copyright (c) ModelScope Contributors. All rights reserved.
"""Utilities for configuring ZeroMQ sockets consistently."""

from __future__ import annotations

import os
import zmq


def get_timeout_s_from_env(env_name: str, default: int) -> int:
    """Read timeout seconds from env and validate it."""
    raw_value = os.environ.get(env_name, str(default))
    try:
        timeout_s = int(raw_value)
    except ValueError as e:
        raise ValueError(f'Invalid {env_name}={raw_value}, must be an integer > 0') from e
    if timeout_s <= 0:
        raise ValueError(f'Invalid {env_name}={timeout_s}, must be > 0')
    return timeout_s


def configure_zmq_socket(socket: zmq.Socket, timeout_ms: int, linger: int = 0) -> None:
    """Apply timeout/linger options to a ZMQ socket."""
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, linger)
