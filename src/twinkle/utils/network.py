# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import socket
from typing import Optional


def is_valid_ipv6_address(ip: str) -> bool:
    """Check if the given string is a valid IPv6 address."""
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except OSError:
        return False


def find_node_ip() -> str:
    """Return a usable IPv4 address for this node.

    Prefers a non-loopback, non-virtual interface, then falls back to a
    virtual interface. When the host only exposes loopback addresses (for
    example ``docker run --network none``), ``127.0.0.1`` is returned instead
    of ``None``: callers use the result as ``MASTER_ADDR`` and a ``None``
    there fails deep inside Ray's ``env_vars`` validation rather than raising
    a useful error.
    """
    import psutil
    main_ip, virtual_ip = None, None
    for name, addrs in sorted(psutil.net_if_addrs().items()):
        for addr in addrs:
            if addr.family.name == 'AF_INET' and not addr.address.startswith('127.'):
                # Heuristic to prefer non-virtual interfaces
                if any(s in name for s in ['lo', 'docker', 'veth', 'vmnet']):
                    if virtual_ip is None:
                        virtual_ip = addr.address
                else:
                    if main_ip is None:
                        main_ip = addr.address
    return main_ip or virtual_ip or '127.0.0.1'


def find_free_port(address: str = '', start_port: Optional[int] = None, retry: int = 100) -> int:
    family = socket.AF_INET
    if address and is_valid_ipv6_address(address):
        family = socket.AF_INET6
    if start_port is None:
        start_port = 0
    # Two socket options are platform-dependent and both changed what this
    # function returns:
    #   * SO_REUSEPORT does not exist on Windows, so setting it unconditionally
    #     made every call raise AttributeError there;
    #   * on Windows SO_REUSEADDR lets a second socket bind and listen on a
    #     port another process is already serving, so the probe reported that
    #     port as free. POSIX rebinds it only for TIME_WAIT, which is what the
    #     option is for here.
    reuseport = getattr(socket, 'SO_REUSEPORT', None)
    for port in range(start_port, start_port + retry):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            if os.name != 'nt':
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if reuseport is not None:
                sock.setsockopt(socket.SOL_SOCKET, reuseport, 1)
            try:
                sock.bind(('', port))
                return sock.getsockname()[1]
            except OSError:
                pass
    raise RuntimeError(f'No free port found in [{start_port}, {start_port + retry})')
