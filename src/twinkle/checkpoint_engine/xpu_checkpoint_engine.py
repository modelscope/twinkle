# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/baidu/vLLM-Kunlun/blob/v0.21.0-dev/vllm_kunlun/distributed/py_kunlun_communicator.py
import pickle
import socket
import struct
import time
from datetime import timedelta
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist

from twinkle import get_logger
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine

logger = get_logger()

_RELAY_SETUP_TIMEOUT = 300.0
_RELAY_IO_TIMEOUT = 600.0
_STORE_WAIT_TIMEOUT = timedelta(seconds=300)


def _store_wait(store, keys: List[str], timeout: timedelta = _STORE_WAIT_TIMEOUT) -> None:
    try:
        store.wait(keys, timeout)
    except TypeError:
        # Older/newer signatures disagree on the timeout type.
        store.wait(keys, timeout.total_seconds())


def _recv_exact(sock, n: int) -> bytearray:
    """Read exactly ``n`` bytes from ``sock`` into a fresh bytearray."""
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(mv[got:])
        if k == 0:
            raise ConnectionError('relay socket closed by peer')
        got += k
    return buf


def _sock_send_tensor(sock, tensor: torch.Tensor) -> None:
    """Stream ``tensor`` as raw bytes: [meta_len][meta][payload_len][payload]."""
    meta = pickle.dumps((tensor.dtype, tuple(tensor.shape)))
    cpu = tensor.detach()
    if cpu.device.type != 'cpu':
        cpu = cpu.cpu()
    if not cpu.is_contiguous():
        cpu = cpu.contiguous()
    cpu = cpu.reshape(-1)
    sock.sendall(struct.pack('!Q', len(meta)))
    sock.sendall(meta)
    nbytes = cpu.numel() * cpu.element_size()
    sock.sendall(struct.pack('!Q', nbytes))
    sock.sendall(cpu.view(torch.uint8).numpy())


def _sock_recv_tensor(sock, device: torch.device) -> torch.Tensor:
    """Receive a tensor sent by ``_sock_send_tensor`` onto ``device``."""
    (mlen,) = struct.unpack('!Q', _recv_exact(sock, 8))
    dtype, shape = pickle.loads(bytes(_recv_exact(sock, mlen)))
    (nbytes,) = struct.unpack('!Q', _recv_exact(sock, 8))
    buf = _recv_exact(sock, nbytes)
    flat = torch.frombuffer(buf, dtype=torch.uint8)
    return flat.view(dtype).reshape(shape).to(device)


class _DoneWork:
    """Already-completed work object (our collectives are synchronous)."""

    def wait(self):
        return None


def _build_xccl_pg(store, name: str, ranks: Sequence[int], my_rank: int, device: torch.device):
    """Assemble a stateless ProcessGroupXCCL over ``ranks`` (global ids).

    Mirrors ms-swift's ``_build_pg``: a plain ``ProcessGroup`` wrapper with a
    ``PrefixStore`` per group and a registered CUSTOM XCCL backend.
    """
    from torch._C._distributed_c10d import ProcessGroup
    from torch.distributed import PrefixStore, ProcessGroupXCCL

    n = len(ranks)
    local_rank = ranks.index(my_rank)
    pstore = PrefixStore(name, store)

    pg = ProcessGroup(pstore, local_rank, n)
    options = ProcessGroupXCCL.Options()
    if hasattr(options, '_timeout'):
        options._timeout = _RELAY_IO_TIMEOUT
    backend = ProcessGroupXCCL(pstore, local_rank, n, options)
    backend._set_sequence_number_for_group()

    backend_type = ProcessGroup.BackendType.CUSTOM
    pg._set_default_backend(backend_type)
    pg._register_backend(device, backend_type, backend)
    return pg


class _FlatPG:
    """Adapter: raw ``broadcast([tensor], opts)`` API over a ProcessGroupXCCL wrapper."""

    def __init__(self, pg):
        self.pg = pg

    def broadcast(self, tensors, opts=None, *args, **kwargs):
        tensor = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
        src = getattr(opts, 'rootRank', 0) if opts is not None else 0
        self.pg.broadcast(tensor, src).wait()
        return _DoneWork()


class _RelayPG:
    """Relay-mode collective facade (single host, duplicate local device indices).

    Colliding ranks are kept out of the XCCL group and served through the
    lowest-rank XCCL member over direct TCP sockets; the members themselves
    keep the device-to-device XCCL path. ``src`` is always rank 0 (trainer
    master) in the checkpoint engine's broadcast topology.
    """

    def __init__(self, rank: int, world_size: int, members: List[int],
                 excluded: List[int], pg, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.members = members
        self.excluded = excluded
        self.leader = members[0] if members else 0
        self.pg = pg  # ProcessGroupXCCL wrapper among members (None when degenerate)
        self.device = device
        self._listen = None
        self._conns = {}  # leader: {excluded rank: socket}
        self._sock = None  # excluded: socket to the leader

    # -- direct-socket rendezvous (store-coordinated, tiny payloads) --
    def setup_sockets(self, store, prefix: str) -> None:
        addr_key = f'{prefix}_relay_addr'
        if not self.excluded:
            return
        if self.rank == self.leader:
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(('', 0))
            srv.listen(len(self.excluded))
            srv.settimeout(15)
            store.set(addr_key, pickle.dumps((socket.gethostname(), srv.getsockname()[1])))
            conns = {}
            deadline = time.time() + _RELAY_SETUP_TIMEOUT
            while len(conns) < len(self.excluded):
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    if time.time() > deadline:
                        raise RuntimeError('relay socket rendezvous timed out')
                    continue
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(_RELAY_IO_TIMEOUT)
                (peer,) = struct.unpack('!I', bytes(_recv_exact(conn, 4)))
                conns[peer] = conn
            self._listen = srv
            self._conns = conns
        elif self.rank in self.excluded:
            host, port = pickle.loads(store.get(addr_key))
            deadline = time.time() + 60
            while True:
                try:
                    s = socket.create_connection((host, port), timeout=10)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise RuntimeError('relay socket connect timed out')
                    time.sleep(0.5)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(_RELAY_IO_TIMEOUT)
            s.sendall(struct.pack('!I', self.rank))
            self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        for c in self._conns.values():
            try:
                c.close()
            except OSError:
                pass
        self._conns = {}
        if self._listen is not None:
            try:
                self._listen.close()
            except OSError:
                pass
            self._listen = None

    # -- collective facade --
    def broadcast(self, tensors, opts=None, *args, **kwargs):
        tensor = tensors[0] if isinstance(tensors, (list, tuple)) else tensors
        src = getattr(opts, 'rootRank', 0) if opts is not None else 0
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if self.rank in self.excluded:
            # Excluded from XCCL: TCP only (src pushes, or leader forwards).
            if self.rank == src:
                _sock_send_tensor(self._sock, tensor)
            # else: the sender (src or leader) pushes to us over the socket.
            else:
                tensor.copy_(_sock_recv_tensor(self._sock, self.device))
            return _DoneWork()

        # This rank is an XCCL member.
        if src in self.excluded:
            if self.rank == self.leader:
                conn = self._conns.get(src)
                tensor.copy_(_sock_recv_tensor(conn, self.device))
            if self.pg is not None:
                self.pg.broadcast(tensor, 0).wait()  # leader == members[0]
            if self.rank == self.leader:
                for x in self.excluded:
                    if x != src:
                        _sock_send_tensor(self._conns[x], tensor)
        else:
            if self.rank == src:
                for x in self.excluded:
                    _sock_send_tensor(self._conns[x], tensor)
            if self.pg is not None:
                self.pg.broadcast(tensor, self.members.index(src)).wait()
        return _DoneWork()


class XCCLCheckpointEngine(NCCLCheckpointEngine):
    """NCCL-checkpoint-engine drop-in for Kunlunxin XPU (BKCL as 'nccl').

    Replaces the direct ``ProcessGroupNCCL`` construction (which deadlocks
    when two ranks share a local device index, see module docstring) with a
    stateless ``ProcessGroupXCCL`` plus a relay fallback. Everything else
    (bucketing, ZMQ metadata, double buffering) is inherited unchanged.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._relay: Optional[_RelayPG] = None

    def _init_xccl_pg(self) -> None:
        store = self._store
        prefix = self.group_name

        # Exchange (hostname, local device index) per rank through the store
        # (a round-trip, not a device collective -- safe on every rank).
        devidx = torch.cuda.current_device()
        store.set(f'{prefix}_dev_{self.rank}', f'{socket.gethostname()}|{devidx}')
        keys = [f'{prefix}_dev_{r}' for r in range(self.world_size)]
        _store_wait(store, keys)
        info = [store.get(k).decode().split('|') for k in keys]
        devs = [(h, int(d)) for h, d in info]

        device = torch.device('cuda', devidx)

        # Single host assumed for the flat case; membership by first
        # occurrence of each (host, local device index) pair.
        first_owner = {}
        excluded = set()
        for r, hd in enumerate(devs):
            if hd in first_owner:
                excluded.add(r)
            else:
                first_owner[hd] = r
        members = [r for r in range(self.world_size) if r not in excluded]

        if not excluded:
            self._pg = _FlatPG(_build_xccl_pg(store, f'{prefix}_ws',
                                              list(range(self.world_size)), self.rank, device))
            logger.info(f'XCCLCheckpointEngine: flat XCCL group, world_size={self.world_size}')
        else:
            pg = None
            if len(members) > 1:
                pg = _build_xccl_pg(store, f'{prefix}_relay', members, self.rank, device)
            self._relay = _RelayPG(self.rank, self.world_size, members, sorted(excluded), pg, device)
            self._relay.setup_sockets(store, prefix)
            self._pg = self._relay
            logger.info(f'XCCLCheckpointEngine: relay mode, members={members}, excluded={sorted(excluded)}')

    def _store_barrier(self) -> None:
        prefix = self.group_name
        store = self._store
        store.set(f'{prefix}_bar_{self.rank}', '1')
        _store_wait(store, [f'{prefix}_bar_{r}' for r in range(self.world_size)])

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the weight-sync process group (XPU flavour).

        Same rendezvous as the NCCL engine (dedicated TCPStore hosted by the
        master), but the XCCL group construction/relay decision happens here
        and the readiness barrier runs through the store instead of a device
        collective.
        """
        # Non-participating trainer ranks: record rank and return.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        # Fast path: group already initialized, skip all setup.
        if self._group_initialized and not self.rebuild_group:
            return

        if self._pg is None:
            self.rank = rank
            self.world_size = world_size

            is_store_master = (rank == 0)
            self._store = dist.TCPStore(
                host_name=master_metadata.nccl_store_host,
                port=master_metadata.nccl_store_port,
                world_size=world_size,
                is_master=is_store_master,
                wait_for_workers=True,
            )
            self._init_xccl_pg()
        else:
            assert self.rank == rank, f'rank {rank} != self.rank {self.rank}'
            assert self.world_size == world_size, (
                f'world_size {world_size} != self.world_size {self.world_size}')

        # Receivers connect to master's ZMQ PUB server.
        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        # Store-based readiness barrier (no device collective during init).
        self._store_barrier()

        self._group_initialized = True
        logger.info(f'init_process_group: rank={self.rank}, world_size={self.world_size}')

    def finalize(self):
        """Tear down relay sockets before the base cleanup."""
        if self._relay is not None:
            self._relay.close()
            self._relay = None
        super().finalize()
