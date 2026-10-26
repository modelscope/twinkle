# Copyright (c) ModelScope Contributors. All rights reserved.
"""zmq PUB/SUB helpers for the manifest channel, shared by the NCCL and HCCL engines."""

from __future__ import annotations

import zmq

from twinkle.utils.network import is_valid_ipv6_address


def format_tcp_address(ip: str, port: int) -> tuple[str, bool]:
    """Return ``(tcp address, is_ipv6)``; IPv6 literals are bracketed."""
    if is_valid_ipv6_address(ip):
        return f'tcp://[{ip}]:{port}', True
    return f'tcp://{ip}:{port}', False


def start_publisher(ip: str, port: int) -> zmq.Socket:
    """Bind a PUB socket on ``ip:port`` (on its own context, reachable as ``socket.context``)."""
    socket = zmq.Context().socket(zmq.PUB)
    address, ipv6 = format_tcp_address(ip, port)
    if ipv6:
        socket.setsockopt(zmq.IPV6, 1)
    socket.bind(address)
    return socket


def connect_subscriber(ip: str, port: int, topic: str) -> zmq.Socket:
    """Connect a SUB socket to ``ip:port`` and subscribe to ``topic``."""
    socket = zmq.Context().socket(zmq.SUB)
    address, ipv6 = format_tcp_address(ip, port)
    if ipv6:
        socket.setsockopt(zmq.IPV6, 1)
    socket.connect(address)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    return socket
