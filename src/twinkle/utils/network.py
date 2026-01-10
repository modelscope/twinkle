import socket
from typing import Optional


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


def find_free_port(start_port: Optional[int] = None, retry: int = 100) -> int:
    if start_port is None:
        start_port = 0
    for port in range(start_port, start_port + retry):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('', port))
                port = sock.getsockname()[1]
                break
            except OSError:
                pass
    return port