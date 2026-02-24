# Copyright (c) ModelScope Contributors. All rights reserved.
import socket
from typing import Optional


def is_valid_ipv6_address(ip: str) -> bool:
    """Check if the given string is a valid IPv6 address."""
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except OSError:
        return False


def find_node_ip() -> Optional[str]:
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
    return main_ip or virtual_ip


def find_free_port(address: str = '', start_port: Optional[int] = None, retry: int = 100) -> int:
    family = socket.AF_INET
    if address and is_valid_ipv6_address(address):
        family = socket.AF_INET6
    if start_port is None:
        start_port = 0
    for port in range(start_port, start_port + retry):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            try:
                sock.bind(('', port))
                port = sock.getsockname()[1]
                break
            except OSError:
                pass
    return port
