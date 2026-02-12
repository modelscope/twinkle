# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import socket
from datetime import timedelta
from typing import Optional

import torch

# ref: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0144.html
# HCCL base port anchor. HCCL derives internal listen/connect ports from this base.
_HCCL_IF_BASE_PORT_ENV = "HCCL_IF_BASE_PORT"
# Host-side socket port pool used by HCCL in multi-process communication.
_HCCL_HOST_SOCKET_PORT_RANGE_ENV = "HCCL_HOST_SOCKET_PORT_RANGE"
# NPU-side socket port pool used by HCCL for device communication channels.
_HCCL_NPU_SOCKET_PORT_RANGE_ENV = "HCCL_NPU_SOCKET_PORT_RANGE"


def _derive_hccl_socket_env_defaults(master_port: int) -> dict:
    """Derive deterministic default HCCL socket env values from master_port."""
    # Keep values stable per job and spread jobs across non-overlapping ranges.
    host_offset = master_port % 8000
    return {
        _HCCL_IF_BASE_PORT_ENV: str(20000 + ((master_port + 997) % 20000)),
        _HCCL_HOST_SOCKET_PORT_RANGE_ENV: f"{40000 + host_offset}-{40000 + host_offset + 511}",
        _HCCL_NPU_SOCKET_PORT_RANGE_ENV: f"{50000 + host_offset}-{50000 + host_offset + 511}",
    }


def _ensure_hccl_socket_env(master_port: int, environ: Optional[dict] = None) -> None:
    """Set deterministic HCCL socket env defaults to avoid port collisions.

    In multi-job environments, HCCL's default base port (60000) can collide
    across concurrent jobs and lead to:
    `ra_hdc_socket_listen_start ... ret(-98)`.

    We derive a per-job port layout from `master_port` so all ranks use the
    same values while reducing cross-job conflicts. Explicit user settings are
    preserved and never overwritten.
    """
    # fix: We hit `ra_hdc_socket_listen_start ... ret(-98)` due to HCCL port collisions.
    # fix: Derive stable ranges from master_port and preserve explicit user overrides.
    env = os.environ if environ is None else environ
    for key, value in _derive_hccl_socket_env_defaults(master_port).items():
        env.setdefault(key, value)


def is_valid_ipv6_address(ip: str) -> bool:
    """Check if the given string is a valid IPv6 address."""
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except socket.error:
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



def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int | torch.device = None,
    backend: str = "nccl",
    listen_socket: socket.socket = None,
    listen_fd: int = None,
):
    """Create a stateless process group using vLLM's StatelessProcessGroup.
    
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL/HCCL) between external (train processes)
    and vLLM workers.
    
    Args:
        master_address: The IP address of the master (rank 0).
        master_port: The port of the master.
        rank: The rank of this process.
        world_size: Total number of processes.
        device: The CUDA device to use. If None, uses current device.
        backend: The communication backend ("nccl" or "hccl").
        listen_socket: Optional pre-created listening socket for master (rank 0).
            If provided, this socket will be reused instead of creating a new one.
        listen_fd: Optional file descriptor of the listening socket.
        
    Returns:
        PyNcclCommunicator or PyHcclCommunicator instance.
    """
    from torch.distributed import TCPStore
    from vllm.distributed.utils import StatelessProcessGroup
    
    if backend == "hccl":
        # fix: Stateless PG + HCCL path needs the same port policy, otherwise workers can still collide.
        _ensure_hccl_socket_env(master_port)
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as Communicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator as Communicator,
        )
    
    if device is None:
        device = torch.cuda.current_device() if backend == "nccl" else torch.npu.current_device()
    
    # Create the stateless process group
    launch_server = rank == 0
    
    if launch_server and listen_socket is None:
        # For master, create a listening socket if not provided
        if is_valid_ipv6_address(master_address):
            listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((master_address, master_port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()
    elif launch_server and listen_fd is None:
        listen_fd = listen_socket.fileno()
    
    store = TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=300),
        use_libuv=False,  # for compatibility
        master_listen_fd=listen_fd,
    )
    
    pg = StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=3600,
    )
    
    communicator = Communicator(pg, device=device)
    return communicator
