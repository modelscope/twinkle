# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-neutral interface for a standalone (stateless) communication group.

A :class:`Communicator` is a group of processes, one rank per process, that is NOT registered
with the ``torch.distributed`` default world. Checkpoint engines use it to move weights between
trainer and rollout processes; the backend (NCCL / HCCL / XCCL / Gloo) decides how the bytes
travel.

Contract (enforced here, so every backend behaves the same):

* ``src`` / ``dst`` are ranks inside this group, ``0 .. world_size - 1``.
* Tensors must be contiguous and live on :attr:`Communicator.device`; anything else raises
  ``ValueError`` instead of being silently copied (a copy would leave the caller's buffer stale).
* Zero-element tensors are a no-op on every rank (all ranks pass the same shape).
* With ``async_op=False`` a call returns ``None`` once the tensor is safe to use on the current
  stream. With ``async_op=True`` it returns a :class:`Work`; call ``wait()`` before using it.
* Reading a result on the host, or freeing a buffer right after a collective, additionally
  needs :meth:`Communicator.synchronize`.
* One communicator is used by one thread at a time; it may be a different thread from the one
  that created it. Backends must use :attr:`device` / the tensor's device and never the calling
  thread's "current device".
* :meth:`Communicator.destroy` is idempotent and never raises on backend teardown errors (they
  are logged); afterwards every operation raises ``RuntimeError``.

Extending: public methods validate and then call a private hook (``_create``, ``_broadcast``,
``_destroy`` are mandatory; ``_all_reduce`` / ``_barrier`` are optional). A new operation is a
new public method plus an optional hook with a default, so existing backends keep working;
:meth:`Communicator.supports` reports which hooks a backend implements itself.
"""
from __future__ import annotations
import enum
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar, Protocol

import torch

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = timedelta(seconds=300)


@dataclass(frozen=True)
class Rendezvous:
    """Where the group meets while it is being built; identical on every rank.

    Rank 0 hosts the store on ``host:port``. ``group_name`` namespaces the store keys, so groups
    that share a store (or a backend that builds sub-groups on one store) do not collide.
    """

    host: str
    port: int
    group_name: str = 'twinkle_comm'


class ReduceOp(str, enum.Enum):
    SUM = 'sum'
    PRODUCT = 'product'
    MIN = 'min'
    MAX = 'max'
    AVG = 'avg'

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value == lowered:
                    return member
        return None


class Work(Protocol):
    """Handle of an in-flight ``async_op=True`` operation."""

    def wait(self) -> Any:
        ...


class _CompletedWork:

    def wait(self) -> bool:
        return True


#: Public operation -> backend hook. ``supports`` only answers for these names.
_OP_HOOKS = {
    'broadcast': '_broadcast',
    'all_reduce': '_all_reduce',
    'barrier': '_barrier',
}
_MANDATORY_OPS = frozenset({'broadcast'})


def normalize_device(device: torch.device | str | int) -> torch.device:
    """``torch.device`` with an explicit index for accelerators (``'cuda'`` -> ``cuda:<current>``)."""
    if isinstance(device, int):
        raise TypeError('pass a torch.device or a device string, not a bare index')
    device = torch.device(device)
    if device.type == 'cpu' or device.index is not None:
        return device
    current_device = getattr(getattr(torch, device.type, None), 'current_device', None)
    if callable(current_device):
        return torch.device(device.type, current_device())
    return device


def _same_device(a: torch.device, b: torch.device) -> bool:
    # Devices without an index API (e.g. 'mps') stay unindexed; treat a missing index as a match.
    return a.type == b.type and (a.index is None or b.index is None or a.index == b.index)


def _check_group(rank: int, world_size: int) -> None:
    if world_size < 1:
        raise ValueError(f'world_size must be >= 1, got {world_size}')
    if not 0 <= rank < world_size:
        raise ValueError(f'rank must be in [0, {world_size}), got {rank}')


class Communicator(ABC):
    """A standalone group of processes exchanging tensors over one backend."""

    backend_name: ClassVar[str]
    #: Backend-specific ``create`` keyword options; anything else is rejected.
    supported_options: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, rank: int, world_size: int, device: torch.device | str) -> None:
        _check_group(rank, world_size)
        self.rank = rank
        self.world_size = world_size
        self.device = normalize_device(device)
        self._destroyed = False
        self._destroying = False
        self._barrier_token: torch.Tensor | None = None

    @property
    def device_index(self) -> int | None:
        return self.device.index

    @property
    def native(self) -> Any:
        """The backend's underlying group object, for migration shims only; ``None`` if unexposed."""
        return None

    # ---- construction / teardown -------------------------------------------------

    @classmethod
    def create(
        cls,
        rendezvous: Rendezvous,
        rank: int,
        world_size: int,
        device: torch.device | str,
        *,
        timeout: timedelta = DEFAULT_TIMEOUT,
        **options,
    ) -> Communicator:
        """Build the group. Collective: returns only after all ``world_size`` ranks joined.

        Raises (instead of hanging) when the group is not complete within ``timeout``.
        """
        _check_group(rank, world_size)
        unknown = set(options) - cls.supported_options
        if unknown:
            raise ValueError(f'{cls.backend_name} communicator does not accept options {sorted(unknown)}; '
                             f'supported: {sorted(cls.supported_options)}')
        if timeout <= timedelta(0):
            raise ValueError(f'timeout must be positive, got {timeout}')
        return cls._create(rendezvous, rank, world_size, normalize_device(device), timeout=timeout, **options)

    @classmethod
    @abstractmethod
    def _create(cls, rendezvous: Rendezvous, rank: int, world_size: int, device: torch.device, *,
                timeout: timedelta, **options) -> Communicator:
        """Backend construction; arguments are already validated and the device normalized."""

    @abstractmethod
    def _destroy(self) -> None:
        """Release backend resources. Called at most once; :meth:`synchronize` still works here."""

    def destroy(self) -> None:
        """Release the group. Idempotent; teardown errors are logged, never raised."""
        if self._destroyed or self._destroying:
            return
        self._destroying = True
        try:
            self._destroy()
        except Exception as error:
            logger.warning('%s communicator (rank %d) teardown failed: %s', self.backend_name, self.rank, error)
        finally:
            self._destroyed = True
            self._destroying = False

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    # ---- operations --------------------------------------------------------------

    def broadcast(self, tensor: torch.Tensor, src: int = 0, async_op: bool = False) -> Work | None:
        """Copy ``tensor`` from rank ``src`` into ``tensor`` on every other rank."""
        self._check_alive()
        self._check_rank(src)
        self._check_tensor(tensor)
        if tensor.numel() == 0:
            return _CompletedWork() if async_op else None
        return self._broadcast(tensor, src, async_op)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp | str = ReduceOp.SUM,
                   async_op: bool = False) -> Work | None:
        """Reduce ``tensor`` across all ranks in place with ``op``."""
        op = ReduceOp(op)
        self._check_alive()
        self._check_tensor(tensor)
        if tensor.numel() == 0:
            return _CompletedWork() if async_op else None
        return self._all_reduce(tensor, op, async_op)

    def barrier(self) -> None:
        """Readiness point: returns once the group's ranks have all reached it.

        May be called right after :meth:`create`, when some backends cannot run device
        collectives yet; such backends override ``_barrier`` (e.g. with a store barrier).
        """
        self._check_alive()
        self._barrier()

    def synchronize(self) -> None:
        """Block the host until work queued on this communicator's device has finished."""
        if self._destroyed:
            self._check_alive()
        if self.device.type == 'cpu':
            return
        from twinkle.utils.framework import Torch
        Torch.synchronize(self.device)

    @classmethod
    def supports(cls, op: str) -> bool:
        """Whether this backend implements ``op`` itself (``False``: default fallback or unsupported)."""
        try:
            hook = _OP_HOOKS[op]
        except KeyError:
            raise ValueError(f'unknown communicator op {op!r}; known: {sorted(_OP_HOOKS)}') from None
        if op in _MANDATORY_OPS:
            return True
        return inspect.getattr_static(cls, hook) is not inspect.getattr_static(Communicator, hook)

    # ---- backend hooks -----------------------------------------------------------

    @abstractmethod
    def _broadcast(self, tensor: torch.Tensor, src: int, async_op: bool) -> Work | None:
        ...

    def _all_reduce(self, tensor: torch.Tensor, op: ReduceOp, async_op: bool) -> Work | None:
        raise NotImplementedError(f'{self.backend_name} communicator does not support all_reduce')

    def _barrier(self) -> None:
        # all_reduce makes every rank wait for every other; a broadcast only waits on the root.
        if self._barrier_token is None:
            self._barrier_token = torch.zeros(1, dtype=torch.int32, device=self.device)
        if type(self).supports('all_reduce'):
            self._all_reduce(self._barrier_token, ReduceOp.SUM, False)
        else:
            self._broadcast(self._barrier_token, 0, False)
        self.synchronize()

    # ---- validation --------------------------------------------------------------

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError(f'{self.backend_name} communicator (rank {self.rank}) was destroyed')

    def _check_rank(self, rank: int, name: str = 'src') -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(f'{name} must be in [0, {self.world_size}), got {rank}')

    def _check_tensor(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'expected a torch.Tensor, got {type(tensor).__name__}')
        if not _same_device(tensor.device, self.device):
            raise ValueError(f'tensor is on {tensor.device} but the {self.backend_name} communicator '
                             f'is bound to {self.device}')
        if not tensor.is_contiguous():
            raise ValueError('communicator tensors must be contiguous')

    def __repr__(self) -> str:
        state = ', destroyed' if self._destroyed else ''
        return (f'{type(self).__name__}(rank={self.rank}, world_size={self.world_size}, '
                f'device={self.device}{state})')
