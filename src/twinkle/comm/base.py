# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-neutral interface for a standalone (stateless) communication group.

A :class:`Communicator` is a group of processes, one rank per process, that is NOT registered
with the ``torch.distributed`` default world. Checkpoint engines use it to move weights between
trainer and rollout processes; the backend (NCCL / HCCL / XCCL / Gloo) decides how the bytes
travel.

Semantics follow ``torch.distributed``:

* ``src`` / ``dst`` are ranks inside this group, ``0 .. world_size - 1``.
* With ``async_op=False`` a call returns once the tensor is safe to use on the current stream.
  With ``async_op=True`` it returns a :class:`Work`; call ``wait()`` before using the tensor.
* Reading a result on the host, or freeing a buffer right after a collective, additionally
  needs :meth:`Communicator.synchronize`.

Only ``create``, ``broadcast`` and ``destroy`` are mandatory. Every other operation has a
default (a fallback built from mandatory ones, or ``NotImplementedError``), so adding an
operation to this class never breaks an existing backend; :meth:`Communicator.supports`
reports which ones a backend implements natively.
"""
from __future__ import annotations
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import torch


@dataclass(frozen=True)
class Rendezvous:
    """Where the group meets while it is being built. Rank 0 hosts the store on ``host:port``."""

    host: str
    port: int


class ReduceOp(str, enum.Enum):
    SUM = 'sum'
    PRODUCT = 'product'
    MIN = 'min'
    MAX = 'max'


class Work(Protocol):
    """Handle of an in-flight ``async_op=True`` operation."""

    def wait(self) -> Any:
        ...


#: Operations every backend must provide; never reported as optional by ``supports``.
_MANDATORY_OPS = frozenset({'broadcast'})


class Communicator(ABC):
    """A standalone group of processes exchanging tensors over one backend."""

    backend_name: ClassVar[str]

    def __init__(self, rank: int, world_size: int, device: torch.device | str) -> None:
        if world_size < 1:
            raise ValueError(f'world_size must be >= 1, got {world_size}')
        if not 0 <= rank < world_size:
            raise ValueError(f'rank must be in [0, {world_size}), got {rank}')
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self._destroyed = False

    # ---- construction / teardown -------------------------------------------------

    @classmethod
    @abstractmethod
    def create(cls, rendezvous: Rendezvous, rank: int, world_size: int, device: torch.device | str,
               **options) -> Communicator:
        """Build the group. Collective: returns only after all ``world_size`` ranks joined.

        Backends ignore ``options`` they do not understand.
        """

    @abstractmethod
    def _destroy(self) -> None:
        """Release backend resources. Called at most once, by :meth:`destroy`."""

    def destroy(self) -> None:
        """Release the group. Idempotent; any later operation raises ``RuntimeError``."""
        if self._destroyed:
            return
        self._destroyed = True
        self._destroy()

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    # ---- mandatory operations ----------------------------------------------------

    @abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int = 0, async_op: bool = False) -> Work | None:
        """Copy ``tensor`` from rank ``src`` into ``tensor`` on every other rank."""

    # ---- optional operations -----------------------------------------------------

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp | str = ReduceOp.SUM,
                   async_op: bool = False) -> Work | None:
        """Reduce ``tensor`` across all ranks in place with ``op``."""
        raise NotImplementedError(f'{self.backend_name} communicator does not support all_reduce')

    def barrier(self) -> None:
        """Block until every rank reached this point.

        Default: a one-element broadcast followed by :meth:`synchronize`. Backends whose device
        collectives are unsafe at that moment (e.g. during initialization) override it.
        """
        self._check_alive()
        token = torch.zeros(1, dtype=torch.uint8, device=self.device)
        self.broadcast(token, src=0)
        self.synchronize()

    def synchronize(self) -> None:
        """Block the host until work queued on this communicator's device has finished."""
        self._check_alive()
        if self.device.type == 'cpu':
            return
        from twinkle.utils.framework import Torch
        Torch.synchronize(self.device)

    @classmethod
    def supports(cls, op: str) -> bool:
        """Whether this backend implements ``op`` itself rather than inheriting the default."""
        if op in _MANDATORY_OPS:
            return True
        default = getattr(Communicator, op, None)
        if default is None or not callable(default):
            return False
        return getattr(cls, op) is not default

    # ---- helpers for backends ----------------------------------------------------

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError(f'{self.backend_name} communicator (rank {self.rank}) was destroyed')

    def _check_rank(self, rank: int, name: str = 'src') -> None:
        if not 0 <= rank < self.world_size:
            raise ValueError(f'{name} must be in [0, {self.world_size}), got {rank}')

    def __repr__(self) -> str:
        state = ', destroyed' if self._destroyed else ''
        return (f'{type(self).__name__}(rank={self.rank}, world_size={self.world_size}, '
                f'device={self.device}{state})')
