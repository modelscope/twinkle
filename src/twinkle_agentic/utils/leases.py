# Copyright (c) ModelScope Contributors. All rights reserved.
"""Lending a fixed set of scarce resources out, one per job.

Some resources cannot be shared for the duration of a job: an
:class:`~twinkle_agentic.envs.base.Env` is a working directory with state in it,
and two jobs in it at once are two jobs editing each other's files. The routing
question is always the same -- *any* free one will do, but not one another job is
holding -- so it is answered once, here.
"""
import queue
import threading
from contextlib import contextmanager
from typing import Generic, Iterator, List, Sequence, TypeVar

T = TypeVar('T')


class Leases(Generic[T]):
    """Lend one resource to one job for the whole life of that job.

    How many resources there are is how many jobs may run at once, and a job
    that finishes early hands its resource to the next job in line rather than
    to the next round. Give the pool as many members as there are workers and a
    lease never blocks.

    A resource is cleaned on the way *in* (:meth:`_prepare`), not on the way
    out, so a job that died holding one costs the next job nothing.
    """

    def __init__(self, items: Sequence[T]):
        if not items:
            raise ValueError(f'{type(self).__name__} is empty: there is nothing to lend out')
        # Slots rather than the resources themselves, so :meth:`_prepare` may
        # answer with a replacement and everyone -- ``__getitem__`` included --
        # sees it from then on.
        self._items: List[T] = list(items)
        self._free: 'queue.Queue[int]' = queue.Queue()
        for slot in range(len(self._items)):
            self._free.put(slot)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """How many jobs may hold a resource at once."""
        return len(self._items)

    def __getitem__(self, slot: int) -> T:
        """One of the resources, lent out or not -- for asking what they can do."""
        return self._items[slot]

    @contextmanager
    def lease(self) -> Iterator[T]:
        """Take a clean resource; give it back however the job ends."""
        slot = self._free.get()
        try:
            self._items[slot] = self._prepare(self._items[slot])
            yield self._items[slot]
        finally:
            self._free.put(slot)

    def _prepare(self, item: T) -> T:
        """Hand the next job a resource with nothing of the last job left on it.

        Return the resource to lend, which may be a replacement for one that
        could not be cleaned. Default: resources need no cleaning.
        """
        return item

    def close(self) -> None:
        for item in self._items:
            close = getattr(item, 'close', None)
            if close is not None:
                close()
