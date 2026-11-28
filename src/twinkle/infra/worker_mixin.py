# Copyright (c) ModelScope Contributors. All rights reserved.
"""Replica lifecycle of a pool component, mounted onto every class with a continuous-work method.

``add_workers`` / ``remove_workers`` / ``pressure`` run on the caller's handle and edit the pool's
membership in the ``WorkerRegistry``; every other process picks the change up through its cached
view (see ``continuous_work``). A component takes part through two optional private hooks run
on each replica: ``_drain()`` (stop the requests it is running, they come back to their callers as
``WorkerDraining`` and are resubmitted elsewhere) and ``_pressure() -> float`` (its load).
"""
import copy
import uuid
from typing import Callable, Dict, List, Optional

from twinkle.utils import get_logger

from .continuous_work import _pool_view, worker_id_of

logger = get_logger()

#: How long a retiring replica's ``shutdown`` may take before it is logged and skipped.
_SHUTDOWN_TIMEOUT = 60.0


class WorkerMixin:
    """The replica-lifecycle methods; ``self`` is the caller-side handle (or the local instance)."""

    def add_workers(self, n: int, warmup: Optional[Callable] = None, ranks: Optional[List[int]] = None) -> List[str]:
        """Start ``n`` replicas, warm them, then let them take requests.

        Placement defaults to the group's free slots; ``ranks`` places them on explicit physical ranks
        (devices lent by another group). ``warmup`` receives a handle holding only the new replicas
        (e.g. to load the current weights) before they turn ACTIVE.
        """
        if not hasattr(self, '_actors'):
            raise RuntimeError('add_workers needs ray mode: a local component has no replicas to add.')
        import ray
        from ._ray import RayHelper
        if n <= 0:
            return []
        cls, args, kwargs, torch_kwargs, mesh_name = self._worker_factory
        group = self.remote_group
        info = RayHelper.group_info(group)
        width = info.get('gpus_per_worker', 1) or 1
        if ranks is not None:
            placements = RayHelper.rank_placements(ranks, width)
            slots = [None] * len(placements)
            if len(placements) != n:
                raise ValueError(f'{len(ranks)} ranks make {len(placements)} replicas, but n={n}.')
        else:
            slots, placements = _free_placements(group, info, n)
        # The factory's kwargs still carry the mesh the initial replicas were built with: a
        # replica born from an in-place mesh here (dp_size=n) would disagree with them about
        # the data world, and with itself again at every later growth.
        kwargs = dict(kwargs)
        actors = RayHelper.create_workers(
            cls, group, 'all', *args, instance_id=uuid.uuid4().hex, extra_torch_kwargs=torch_kwargs,
            max_concurrency=self._max_concurrency, placements=placements, **kwargs)
        from .dispatch import _owner_identity
        owner = _owner_identity()
        if self.__dict__.get('_pool_actor_owner') != owner:
            self._pool_actor_owner = owner
            self._owned_pool_actors = []
        self._owned_pool_actors.extend(actors)
        worker_ids = [worker_id_of(actor) for actor in actors]
        key = self._pool_key
        try:
            RayHelper.pool_put(key, {
                worker_id: {'actor': actor, 'state': 'WARMING', 'group': group, 'slot': slot,
                            'ranks': tuple(placement.get('ranks', ()))}
                for worker_id, actor, slot, placement in zip(worker_ids, actors, slots, placements)
            })
            if warmup is not None:
                sub = copy.copy(self)
                sub._actors = list(actors)
                sub._pool_key = None
                warmup(sub)
            RayHelper.pool_set_state(key, worker_ids, 'ACTIVE')
            _pool_view(self).refresh(force=True)
        except BaseException:
            try:
                RayHelper.release_workers(actors)
            except BaseException:
                logger.exception('Pool-growth rollback is incomplete.')
            finally:
                self._owned_pool_actors[:] = [actor for actor in self._owned_pool_actors
                                             if actor in RayHelper._created_actors]
            raise
        return worker_ids

    def remove_workers(self, k: int = 0, ranks: Optional[List[int]] = None) -> List[int]:
        """Retire the ``k`` least loaded replicas, or those on ``ranks``; returns the ranks they released.

        The replicas leave the ACTIVE set first, then run ``_drain`` so their in-flight requests come
        back as ``WorkerDraining`` and get resubmitted elsewhere, then shut down.
        """
        if not hasattr(self, '_actors'):
            raise RuntimeError('remove_workers needs ray mode: a local component has no replicas to remove.')
        import ray
        from ._ray import RayHelper
        key = self._pool_key
        view = _pool_view(self)
        members = RayHelper.pool_get(key)
        if ranks is not None:
            ranks = set(ranks)
            targets = [worker_id for worker_id, member in members.items()
                       if member['state'] == 'ACTIVE' and ranks & set(member.get('ranks') or ())]
        else:
            targets = view.least_loaded(k) if k > 0 else []
        if not targets:
            return []
        active = view.actors()
        if len(targets) >= len(active):
            raise ValueError(f'Cannot remove {len(targets)} of {len(active)} active replicas; at least one must stay.')
        RayHelper.pool_set_state(key, targets, 'DRAINING')
        for worker_id in targets:
            view.evict(worker_id)
        actors = [members[worker_id]['actor'] for worker_id in targets]
        for ref in [actor._twinkle_drain.remote() for actor in actors]:
            try:
                ray.get(ref, timeout=_SHUTDOWN_TIMEOUT)
            except Exception:  # noqa
                logger.warning('A replica failed while draining.', exc_info=True)
        if hasattr(type(self), 'shutdown'):
            for ref in [actor.shutdown.remote() for actor in actors]:
                try:
                    ray.get(ref, timeout=_SHUTDOWN_TIMEOUT)
                except Exception:  # noqa
                    logger.warning('A replica failed to shut down cleanly.', exc_info=True)
        released = []
        failures = []
        for worker_id, actor in zip(targets, actors):
            try:
                with RayHelper._actor_lifecycle_lock:
                    ray.kill(actor, no_restart=True)
                    if actor in RayHelper._created_actors:
                        RayHelper._created_actors.remove(actor)
                    RayHelper._released_worker_ids.add(worker_id)
                released.append(worker_id)
            except BaseException as error:
                failures.append(error)
        try:
            RayHelper.release_workers([])
        except BaseException as error:
            failures.append(error)
        if failures:
            raise BaseExceptionGroup('Pool retirement is incomplete', failures)
        return sorted(rank for worker_id in released for rank in members[worker_id].get('ranks') or ())

    def pressure(self) -> float:
        """Mean load of the ACTIVE replicas, as each reports through ``_pressure``."""
        if not hasattr(self, '_actors'):
            return self._twinkle_pressure()
        import ray
        actors = _pool_view(self).actors()
        if not actors:
            return 0.0
        values = ray.get([actor._twinkle_pressure.remote() for actor in actors])
        return float(sum(values) / len(values))

    # ---- run on each replica ----

    def _twinkle_drain(self):
        self._twinkle_draining = True
        drain = getattr(self, '_drain', None)
        if drain is not None:
            drain()

    def _twinkle_pressure(self) -> float:
        hook = getattr(self, '_pressure', None)
        return float(hook()) if hook is not None else 0.0


def _free_placements(group: str, info: dict, n: int):
    """The first ``n`` slots of ``group`` that no replica of any pool occupies."""
    from ._ray import RayHelper
    occupied = RayHelper.pool_occupied()
    used_slots = {slot for member_group, slot, _ in occupied if member_group == group and slot is not None}
    used_ranks = {rank for _, _, ranks in occupied for rank in ranks}
    free = [(slot, placement) for slot, placement in enumerate(info['placements'])
            if slot not in used_slots and not set(placement.get('ranks', ())) & used_ranks]
    if len(free) < n:
        raise RuntimeError(f'Device group {group!r} has {len(free)} free slot(s), {n} requested.')
    free = free[:n]
    return [slot for slot, _ in free], [placement for _, placement in free]


_MIXIN_METHODS = ('add_workers', 'remove_workers', 'pressure', '_twinkle_drain', '_twinkle_pressure')


def mount_worker_mixin(cls) -> None:
    """Attach the replica-lifecycle methods to a pool component class (one with a continuous method)."""
    for name in _MIXIN_METHODS:
        if name in cls.__dict__:
            raise TypeError(f'@remote_class: {cls.__name__} already defines {name!r}, which the replica '
                            'lifecycle needs.')
        setattr(cls, name, getattr(WorkerMixin, name))
