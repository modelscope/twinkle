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
import time
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
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._leased = set()
        self._quarantined = set()
        self._closing = False
        self._closed_slots = set()

    def __len__(self) -> int:
        """How many jobs may hold a resource at once."""
        return len(self._items)

    def __getitem__(self, slot: int) -> T:
        """One of the resources, lent out or not -- for asking what they can do."""
        return self._items[slot]

    @contextmanager
    def lease(self, *, cancel_event=None, timeout=None) -> Iterator[T]:
        """等待可取消；清理未确认的资源被隔离，不再发给下一个 episode。"""
        if timeout is not None and timeout < 0:
            raise ValueError('lease timeout 必须非负。')
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError('租约等待已中断。')
            with self._lock:
                if self._closing:
                    raise RuntimeError('租约池已关闭准入。')
            try:
                slot = self._free.get(timeout=0.05 if deadline is None else max(0, min(0.05, deadline - time.monotonic())))
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('等待租约超时。')
                continue
            with self._lock:
                if self._closing:
                    self._free.put(slot)
                    raise RuntimeError('租约池已关闭准入。')
                self._leased.add(slot)
            break
        try:
            try:
                self._items[slot] = self._prepare(self._items[slot])
            except BaseException:
                with self._lock:
                    self._quarantined.add(slot)
                raise
            yield self._items[slot]
        finally:
            try:
                ensure_idle = getattr(self._items[slot], 'ensure_idle', None)
                if ensure_idle is not None:
                    ensure_idle()
            except BaseException:
                with self._lock:
                    self._quarantined.add(slot)
                raise
            finally:
                with self._idle:
                    self._leased.remove(slot)
                    if slot not in self._quarantined:
                        self._free.put(slot)
                    self._idle.notify_all()

    def quarantine(self, item: T) -> None:
        """保留不确定的资源身份；只能由其所属执行层确认恢复。"""
        with self._lock:
            slot = next((slot for slot, candidate in enumerate(self._items) if candidate is item), None)
            if slot is None or slot not in self._leased:
                raise ValueError('只能隔离当前已租出的资源。')
            self._quarantined.add(slot)

    def _prepare(self, item: T) -> T:
        """Hand the next job a resource with nothing of the last job left on it.

        Return the resource to lend, which may be a replacement for one that
        could not be cleaned. Default: resources need no cleaning.
        """
        return item

    def close(self, timeout=None) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError('close timeout 必须非负。')
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._idle:
            self._closing = True
            while self._leased:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError('租约仍被占用；资源未关闭。')
                self._idle.wait(remaining)
            slots = [slot for slot in range(len(self._items))
                     if slot not in self._quarantined and slot not in self._closed_slots]
        errors = []
        for slot in slots:
            try:
                close = getattr(self._items[slot], 'close', None)
                if close is not None:
                    close()
                with self._lock:
                    self._closed_slots.add(slot)
            except BaseException as exc:
                errors.append(exc)
                with self._lock:
                    self._quarantined.add(slot)
        if errors or self._quarantined:
            raise RuntimeError('租约池包含清理未确认的隔离资源。') from (errors[0] if errors else None)
