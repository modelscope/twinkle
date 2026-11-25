# Copyright (c) ModelScope Contributors. All rights reserved.
"""XCCL backend: ``ProcessGroupXCCL`` for Kunlunxin XPU (BKCL mounted as torch 'nccl').

A directly built ``ProcessGroupNCCL`` deadlocks when two ranks share a local device
index -- every single-card Ray worker reports local index 0. This backend builds a
stateless ``ProcessGroupXCCL`` outside the default world instead; when BKCL's
non-uniform-visible-devices mode is unavailable and ranks collide on
(host, local index), the colliding ranks are kept out of the XCCL group and served
by the lowest-rank member over direct TCP sockets (relay mode).

Only ``broadcast`` is a device collective here: it is the one the checkpoint engine needs.
The readiness barrier runs through the rendezvous store, so ``barrier`` is valid right after
``create`` even before any device collective has run.
"""
from __future__ import annotations
import os
import pickle
import socket
import struct
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from ..base import Communicator, Rendezvous, Work

_RELAY_SETUP_TIMEOUT = 300.0
_RELAY_IO_TIMEOUT = 600.0


def _store_wait(store: dist.Store, keys: list[str], timeout: timedelta) -> None:
    try:
        store.wait(keys, timeout)
    except TypeError:
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
    """Already-completed work handle (XCCL collectives here are synchronous)."""

    def wait(self) -> bool:
        return True


def _import_xccl():
    """``(ProcessGroup, PrefixStore, ProcessGroupXCCL)``, or a clear error on a non-XPU torch.

    ``ProcessGroupXCCL`` only exists in the Kunlunxin torch plugin, which also mounts BKCL
    under torch's ``cuda`` namespace. Importing it is therefore the honest test for "can this
    process build an XCCL group", stricter than looking at the device type alone.
    """
    try:
        from torch._C._distributed_c10d import ProcessGroup
        from torch.distributed import PrefixStore, ProcessGroupXCCL
    except ImportError as error:
        raise RuntimeError('xccl communicator needs the Kunlunxin torch plugin (torch_xmlir), whose '
                           'torch.distributed exposes ProcessGroupXCCL') from error
    return ProcessGroup, PrefixStore, ProcessGroupXCCL


def _build_xccl_pg(store: dist.Store, name: str, ranks: Sequence[int], my_rank: int, device: torch.device, *,
                   group_id: str, timeout: timedelta):
    """Assemble a stateless ``ProcessGroupXCCL`` over ``ranks`` (global ids).

    A plain ``ProcessGroup`` wrapper with a ``PrefixStore`` per group and a
    registered CUSTOM XCCL backend; not registered with the default world.
    """
    ProcessGroup, PrefixStore, ProcessGroupXCCL = _import_xccl()

    n = len(ranks)
    local_rank = ranks.index(my_rank)
    pstore = PrefixStore(name, store)

    pg = ProcessGroup(pstore, local_rank, n)
    options = ProcessGroupXCCL.Options()
    if hasattr(options, '_timeout'):
        options._timeout = timeout
    if hasattr(options, 'group_id'):
        # BKCL names its communicator after the group id; two live groups sharing the (default
        # empty) id in one process collide during init. The rendezvous address is identical on
        # every rank and unique per live group.
        options.group_id = group_id
    backend = ProcessGroupXCCL(pstore, local_rank, n, options)
    backend._set_sequence_number_for_group()

    backend_type = ProcessGroup.BackendType.CUSTOM
    pg._set_default_backend(backend_type)
    pg._register_backend(device, backend_type, backend)
    return pg


class _FlatPG:
    """Broadcast over a single flat ``ProcessGroupXCCL`` wrapper (no colliding devices)."""

    def __init__(self, pg):
        self.pg = pg

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.pg.broadcast(tensor, src).wait()

    def close(self) -> None:
        pg, self.pg = self.pg, None
        if pg is not None and hasattr(pg, 'shutdown'):
            pg.shutdown()


class _RelayPG:
    """Relay-mode broadcast facade (single host, duplicate local device indices).

    Colliding ranks are kept out of the XCCL group and served through the
    lowest-rank XCCL member over direct TCP sockets; the members keep the
    device-to-device XCCL path.
    """

    def __init__(self, rank: int, world_size: int, members: list[int],
                 excluded: list[int], pg, device: torch.device):
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

    def setup_sockets(self, store: dist.Store, prefix: str) -> None:
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
                except TimeoutError:
                    if time.time() > deadline:
                        raise RuntimeError('relay socket rendezvous timed out') from None
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
                except OSError as error:
                    if time.time() > deadline:
                        # The OSError says whether it was refused, unreachable or unresolved.
                        raise RuntimeError('relay socket connect timed out') from error
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
        pg, self.pg = self.pg, None
        if pg is not None and hasattr(pg, 'shutdown'):
            pg.shutdown()

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if self.rank in self.excluded:
            # Excluded from XCCL: TCP only (src pushes, or leader forwards).
            if self.rank == src:
                _sock_send_tensor(self._sock, tensor)
            else:
                tensor.copy_(_sock_recv_tensor(self._sock, self.device))
            return

        # This rank is an XCCL member.
        if src in self.excluded:
            if self.rank == self.leader:
                tensor.copy_(_sock_recv_tensor(self._conns.get(src), self.device))
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


class XCCLCommunicator(Communicator):
    """Standalone XCCL group for Kunlunxin XPU, with a socket relay fallback.

    Mirrors the ``GlooCommunicator`` shape: a dedicated ``TCPStore`` hosted by rank 0
    plus a stateless ``ProcessGroupXCCL`` (or relay) built on top of it. ``broadcast``
    is the device collective; ``barrier`` is a store round-trip so it is safe before
    the group has run any collective.
    """

    backend_name = 'xccl'

    def __init__(self, rank: int, world_size: int, device: torch.device, store: dist.Store,
                 group_name: str, timeout: timedelta) -> None:
        super().__init__(rank, world_size, device)
        self._store = store
        self._group_name = group_name
        self._timeout = timeout
        # Replaced by ``_create`` with the rendezvous-qualified id; see ``_xccl_pg``.
        self._group_id = group_name
        self._pg = None  # _FlatPG | _RelayPG
        self._relay: _RelayPG | None = None
        self._barrier_gen = 0

    @property
    def native(self) -> Any:
        # Flat groups expose their ProcessGroup for migration shims; relay mode has none.
        return getattr(self._pg, 'pg', None) if isinstance(self._pg, _FlatPG) else None

    @classmethod
    def _create(cls, rendezvous: Rendezvous, rank: int, world_size: int, device: torch.device, *,
                timeout: timedelta) -> XCCLCommunicator:
        # Kunlunxin's torch plugin mounts BKCL under torch's cuda namespace, so an XPU rank
        # reports a 'cuda' device; _import_xccl is what separates it from a real NVIDIA box.
        if device.type != 'cuda':
            raise ValueError(f'xccl communicator needs a cuda-alike (Kunlunxin XPU) device, got {device}')
        _import_xccl()

        store = dist.TCPStore(
            host_name=rendezvous.host,
            port=rendezvous.port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timeout,
            wait_for_workers=True,
        )
        comm = cls(rank, world_size, device, store, rendezvous.group_name, timeout)
        comm._group_id = f'{rendezvous.group_name}@{rendezvous.host}:{rendezvous.port}'
        comm._init_pg()
        return comm

    def _init_pg(self) -> None:
        store, prefix, device = self._store, self._group_name, self.device

        # Exchange (hostname, local device index) through the store -- a round-trip,
        # not a device collective, so it is safe on every rank before the group exists.
        devidx = device.index if device.index is not None else 0
        store.set(f'{prefix}_dev_{self.rank}', f'{socket.gethostname()}|{devidx}')
        keys = [f'{prefix}_dev_{r}' for r in range(self.world_size)]
        _store_wait(store, keys, self._timeout)
        devs = [tuple(store.get(k).decode().split('|')) for k in keys]
        devs = [(h, int(d)) for h, d in devs]

        # BKCL_NONUNIFORM_VISIBLE_DEVICES=1: BKCL keys each rank by PCI address, so
        # colliding local indices are fine and a flat group works (~30 GB/s vs the
        # relay's ~0.4 GB/s). Only fall back to the collision relay when nonuniform
        # mode is unavailable (older libbkcl).
        if os.environ.get('BKCL_NONUNIFORM_VISIBLE_DEVICES') == '1':
            self._pg = _FlatPG(self._xccl_pg(f'{prefix}_ws', list(range(self.world_size))))
            return

        # Single host assumed; membership by first occurrence of (host, local index).
        first_owner: dict = {}
        excluded = set()
        for r, hd in enumerate(devs):
            if hd in first_owner:
                excluded.add(r)
            else:
                first_owner[hd] = r
        members = [r for r in range(self.world_size) if r not in excluded]

        if not excluded:
            self._pg = _FlatPG(self._xccl_pg(f'{prefix}_ws', list(range(self.world_size))))
        else:
            pg = None
            if len(members) > 1:
                pg = self._xccl_pg(f'{prefix}_relay', members)
            self._relay = _RelayPG(self.rank, self.world_size, members, sorted(excluded), pg, device)
            self._relay.setup_sockets(store, prefix)
            self._pg = self._relay

    def _xccl_pg(self, name: str, ranks: Sequence[int]):
        """One XCCL group over ``ranks``, with a group id unique per live group."""
        return _build_xccl_pg(self._store, name, ranks, self.rank, self.device,
                              group_id=f'{self._group_id}/{name}', timeout=self._timeout)

    def _broadcast(self, tensor: torch.Tensor, src: int, async_op: bool) -> Work | None:
        self._pg.broadcast(tensor, src)
        return _DoneWork() if async_op else None

    def _barrier(self) -> None:
        # Store round-trip: a real readiness point that needs no device collective,
        # so it is valid right after create(). A generation counter makes it reusable.
        gen = self._barrier_gen
        self._barrier_gen += 1
        self._store.set(f'{self._group_name}_bar{gen}_{self.rank}', b'1')
        _store_wait(self._store, [f'{self._group_name}_bar{gen}_{r}' for r in range(self.world_size)],
                    self._timeout)

    def _destroy(self) -> None:
        if self._relay is not None:
            self._relay.close()
            self._relay = None
        pg, self._pg = self._pg, None
        if isinstance(pg, _FlatPG):
            pg.close()
        self._store = None
