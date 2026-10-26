# Copyright (c) ModelScope Contributors. All rights reserved.
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Tuple

from twinkle.utils import get_logger

from .continuous_work import _cw_drop_locked, _cw_unregister, _select_workers, _sticky_drop_worker, get_worker_id
from .dispatch import remote_function

logger = get_logger()


@dataclass
class WorkerEntry:
    """Placement, admission state and request load of one full inference engine."""

    worker_id: str
    actor: Any
    group: str
    ranks: Tuple[int, ...]
    shared: bool = False
    state: str = 'WARMING'
    inflight: int = 0
    dispatching: int = 0


class WorkerPool:
    """Elasticity-contract methods mounted onto ``@remote_class`` classes by ``mount_worker_pool``;
    ``self`` is the driver-side handler.
    
    Four groups: factory recording (``_save_worker_factory``), wrapper hooks (admission selection,
    the mutex, completion bookkeeping -- probed by dispatch's wrapper via ``getattr``), lifecycle
    (configure/add/activate/pause/drain/remove/close), and queries (worker_ids/ranks/inflight/idle).
    """

    def _save_worker_factory(self, cls, args, kwargs, torch_kwargs):
        self._worker_factory = (cls, tuple(args), dict(kwargs), torch_kwargs)
        self._worker_lock = threading.RLock()
        self._worker_configurations = []

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, '_actors'):
            # The remote payload only needs the native handler's serialisable metadata; admission and
            # in-flight state stay with the original driver.
            for key in ('_worker_lock', '_continous_work_state', '_sampler_workers',
                        '_suspended_workers', '_paused_generation_workers', '_creating_slots', '_worker_factory',
                        '_worker_configurations'):
                state.pop(key, None)
            if hasattr(self, '_sampler_workers'):
                state['_admission_owned_elsewhere'] = True
        return state

    def configure_workers(self, ranks, *, group=None, shared=False, state='WARMING'):
        """Register existing actors on the driver; does not touch the training mesh or the engine's internal parallel layout."""
        if not hasattr(self, '_actors') or getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Worker lifecycle requires its original Ray driver handler.')
        if hasattr(self, '_sampler_workers'):
            raise RuntimeError('Workers have already been configured.')
        ranks = tuple(ranks)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('Physical ranks must be a non-empty sequence of unique non-negative integers.')
        from ._ray import RayHelper
        width = RayHelper.resource_manager.get_config(group or self.remote_group).gpus_per_worker
        if not self._actors or len(ranks) != len(self._actors) * width:
            raise ValueError('Physical ranks must match gpus_per_worker for every complete worker slot.')
        if state not in ('WARMING', 'ACTIVE', 'DRAINING', 'PAUSED'):
            raise ValueError(f'Unknown worker state: {state!r}.')
        if not hasattr(self, '_worker_lock'):
            self._worker_lock = threading.RLock()
        self._sampler_workers = {}
        self._suspended_workers = set()
        self._paused_generation_workers = set()
        self._creating_slots = set()
        self._round_robin = 0
        self._admission_closed = False
        for index, actor in enumerate(self._actors):
            worker_id = get_worker_id(actor)
            self._sampler_workers[worker_id] = WorkerEntry(
                worker_id, actor, group or self.remote_group,
                ranks[index * width:(index + 1) * width], shared=shared, state=state)

    def worker_ids(self, *, shared=None, states=None):
        with self._worker_lock:
            return tuple(worker_id for worker_id, worker in self._sampler_workers.items()
                         if (shared is None or worker.shared == shared)
                         and (states is None or worker.state in states))

    def worker_ranks(self, worker_ids):
        with self._worker_lock:
            return tuple(self._sampler_workers[worker_id].ranks for worker_id in worker_ids)

    def _sampling_worker_ids(self):
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Adaptive sampling must be admitted by its original driver, not a serialized handler.')
        if not hasattr(self, '_sampler_workers'):
            return None
        with self._worker_lock:
            if self._admission_closed:
                raise RuntimeError('Worker admission is closed.')
            return tuple(worker_id for worker_id, worker in self._sampler_workers.items()
                         if worker.state == 'ACTIVE' and worker_id not in self._suspended_workers)

    def _begin_worker_call(self, worker_ids, worker_selector):
        if not hasattr(self, '_sampler_workers'):
            return None
        # Selection, barrier and RPC send share one critical section; ordinary calls collect results outside the lock.
        self._worker_lock.acquire()
        try:
            if worker_ids is None and worker_selector is not None:
                worker_ids = getattr(self, worker_selector)()
            actors = _select_workers(self, worker_ids)
            worker_ids = tuple(get_worker_id(actor) for actor in actors)
            for worker_id in worker_ids:
                self._sampler_workers[worker_id].dispatching += 1
            return worker_ids
        except BaseException:
            self._worker_lock.release()
            raise

    def _end_worker_dispatch(self):
        self._worker_lock.release()

    def _end_worker_call(self, worker_ids):
        with self._worker_lock:
            for worker_id in worker_ids:
                worker = self._sampler_workers.get(worker_id)
                if worker is not None:
                    worker.dispatching -= 1

    # ---- Barrier deadlock exemption for control calls (a long-term mechanism, not transitional) ----
    # pause_generation (sampler/partial_rollout.py) is a suspend-and-resume gate, not a terminator: after
    # it returns the gate stays closed, and samples parked at the gate are still live RPCs, so a barrier
    # waiting on them would deadlock (only resume_generation releases them). Hence control methods (the
    # controls list) skip the barrier wait, and in-flight samples on paused workers get the same
    # exemption. Removing the name list declaratively would need a generic completion contract for
    # pause-like methods ("after this returns, its target worker is suspended"); the three hooks stay.

    def _worker_call_conflicts(self, current_method, pending_method, worker_id):
        if not hasattr(self, '_sampler_workers') or pending_method != 'sample':
            return True
        controls = ('pause_generation', 'resume_generation', 'abort_all_inflight', '_generation_count')
        return (current_method not in controls
                and worker_id not in getattr(self, '_paused_generation_workers', ()))

    def _worker_call_completed(self, method, worker_ids):
        with self._worker_lock:
            if method == 'pause_generation':
                self._paused_generation_workers.update(worker_ids)
            elif method == 'resume_generation':
                self._paused_generation_workers.difference_update(worker_ids)

    def _settle_worker_controls(self, method, submissions):
        """After a dispatch or collect failure, confirm each already-sent pause/resume RPC one by one."""
        if method not in ('pause_generation', 'resume_generation'):
            return
        import ray

        for worker_id, ref in submissions:
            try:
                ray.get(ref)
            except BaseException:
                logger.warning('%s on worker %s did not complete successfully.', method, worker_id, exc_info=True)
            else:
                self._worker_call_completed(method, (worker_id,))

    def _record_worker_configuration(self, method, args, kwargs):
        """Record one configuration call for replay onto newly added workers; the trigger (the replay declaration) is decided by dispatch's wrapper."""
        with self._worker_lock:
            self._worker_configurations.append((method, tuple(args), dict(kwargs)))

    def add_workers(self, group, ranks, *, worker_slots=None, shared=False, engine_args=None):
        """Create actors from reserved full slots; still admits no new requests until configured."""
        from ._ray import RayHelper
        from twinkle.utils import DeviceMesh

        if not hasattr(self, '_sampler_workers'):
            raise RuntimeError('Call configure_workers before adding workers.')
        config = RayHelper.resource_manager.get_config(group)
        placements = RayHelper.resource_manager.get_group(group)
        width = getattr(config, 'gpus_per_worker', 1)
        ranks = tuple(ranks)
        if any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('Physical ranks must be unique non-negative integers.')
        if len(ranks) != len(placements) * width:
            raise ValueError('Physical ranks must match the reserved worker slots.')
        slots = tuple(range(len(placements))) if worker_slots is None else tuple(worker_slots)
        if any(type(slot) is not int or not 0 <= slot < len(placements) for slot in slots) or len(set(slots)) != len(slots):
            raise ValueError(f'Invalid worker_slots: {slots!r}.')
        if not slots:
            return ()
        selected_ranks = tuple(ranks[slot * width:(slot + 1) * width] for slot in slots)
        reservations = {(group, slot_ranks) for slot_ranks in selected_ranks}
        with self._worker_lock:
            occupied = {(worker.group, worker.ranks) for worker in self._sampler_workers.values()}
            if self._admission_closed or reservations & (occupied | self._creating_slots):
                raise RuntimeError('The pool is closing or a selected worker slot is occupied.')
            self._creating_slots.update(reservations)
        actors = []
        worker_ids = ()
        try:
            cls, args, kwargs, torch_kwargs = self._worker_factory
            kwargs = dict(kwargs, remote_group=group, device_mesh=DeviceMesh.from_sizes(dp_size=len(slots)))
            if engine_args is not None:
                kwargs['engine_args'] = dict(kwargs.get('engine_args', {}), **engine_args)
            actors = RayHelper.create_workers(
                cls, group, 'all', *args, instance_id=uuid.uuid4().hex,
                extra_torch_kwargs=torch_kwargs, max_concurrency=self._max_concurrency,
                worker_slots=list(slots), **kwargs)
            worker_ids = tuple(get_worker_id(actor) for actor in actors)
            with self._worker_lock:
                if self._admission_closed:
                    raise RuntimeError('The pool closed while new workers were being constructed.')
                self._actors = tuple(self._actors) + tuple(actors)
                for actor, worker_id, physical_ranks in zip(actors, worker_ids, selected_ranks):
                    self._sampler_workers[worker_id] = WorkerEntry(
                        worker_id, actor, group, physical_ranks, shared=shared)
                self._worker_ready(_worker_ids=worker_ids)
                # Full-configuration calls and new-actor install/replay are mutually exclusive, to avoid missed or duplicated patches.
                for method, args, kwargs in self._worker_configurations:
                    result = getattr(self, method)(*args, _worker_ids=worker_ids, **kwargs)
                    if callable(result):
                        result()
                return worker_ids
        except BaseException:
            try:
                self._discard_workers(actors)
            except Exception:
                logger.warning('Failed to fully shut down workers after a construction error.', exc_info=True)
            raise
        finally:
            with self._worker_lock:
                self._creating_slots.difference_update(reservations)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def _worker_ready(self):
        return True

    def activate_workers(self, worker_ids):
        with self._worker_lock:
            workers = [self._sampler_workers[worker_id] for worker_id in worker_ids]
            if any(worker.state == 'DRAINING' for worker in workers):
                raise RuntimeError('A draining worker cannot accept new requests.')
            for worker in workers:
                worker.state = 'ACTIVE'

    def drain_workers(self, worker_ids):
        with self._worker_lock:
            for worker_id in worker_ids:
                self._sampler_workers[worker_id].state = 'DRAINING'

    def pause_workers(self, worker_ids):
        with self._worker_lock:
            for worker_id in worker_ids:
                worker = self._sampler_workers[worker_id]
                if worker.state in ('ACTIVE', 'WARMING'):
                    worker.state = 'PAUSED'

    def suspend_admission(self, worker_ids):
        with self._worker_lock:
            self._suspended_workers.update(worker_ids)

    def resume_admission(self, worker_ids):
        with self._worker_lock:
            self._suspended_workers.difference_update(worker_ids)

    def close_admission(self):
        with self._worker_lock:
            self._admission_closed = True

    def worker_inflight(self, worker_ids):
        with self._worker_lock:
            return tuple(self._sampler_workers[worker_id].inflight for worker_id in worker_ids)

    @property
    def inflight_count(self):
        return sum(self.worker_inflight(self.worker_ids()))

    def workers_idle(self, worker_ids):
        """Check control state only -- an actor is retired without reading generation output."""
        import ray
        worker_ids = tuple(worker_ids)
        with self._worker_lock:
            if any(self._sampler_workers[worker_id].inflight or self._sampler_workers[worker_id].dispatching
                   for worker_id in worker_ids):
                return False
        state = getattr(self, '_continous_work_state', None)
        if state is not None:
            with state['lock']:
                refs = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
            if refs:
                ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
                if len(ready) != len(refs):
                    return False
                _cw_unregister(self, '', refs)
        generation_count = getattr(self, '_generation_count', None)
        if generation_count is not None:
            counts = generation_count(_worker_ids=worker_ids)
            counts = [counts] if isinstance(counts, int) else counts
            return all(count == 0 for count in counts)
        return True

    def remove_workers(self, worker_ids):
        worker_ids = tuple(worker_ids)
        with self._worker_lock:
            workers = [self._sampler_workers[worker_id] for worker_id in worker_ids]
            if any(worker.state == 'ACTIVE' for worker in workers):
                raise RuntimeError('Drain or pause workers before removing them.')
        # Remote probes and the discard's ray.kill run OUTSIDE the pool lock: a hung worker RPC must
        # never freeze the whole pool (every submit takes this lock to choose workers).
        if not self.workers_idle(worker_ids):
            return False
        self._discard_workers(tuple(worker.actor for worker in workers))
        return True

    def add_replicas(self, group):
        """Add model-component replicas (unlike add_workers' engine pool: no bookkeeping, admission
        or config replay).
        
        The targets are the component's own actor replicas -- scale-up for frozen forward classes
        (RM/reference/teacher): each new actor is cloned from the factory recorded at construction
        (loaded straight from the checkpoint, no weight sync); after ``_actors`` and the cross-replica
        device_mesh grow together, slice_dp dispatch and collect immediately include the new replica.
        Placement follows the group's reserved slots (same source as construction); slot occupancy is
        not re-validated.
        """
        from ._ray import RayHelper
        from twinkle.utils import DeviceMesh

        factory = getattr(self, '_worker_factory', None)
        if factory is None:
            raise RuntimeError('Replica elasticity needs a ray-placed model (the worker factory is '
                               'recorded only under mode=ray).')
        cls, args, kwargs, torch_kwargs = factory
        actors = RayHelper.create_workers(
            cls, group, 'all', *args, extra_torch_kwargs=torch_kwargs, **dict(kwargs))
        with self._worker_lock:
            mesh = getattr(self, 'device_mesh', None)
            tp = getattr(mesh, 'tp_world_size', 1) or 1 if mesh is not None else 1
            replicas = (len(getattr(self, '_actors', ()) or ()) + len(actors)) // max(1, tp)
            self._actors = tuple(getattr(self, '_actors', ()) or ()) + tuple(actors)
            if mesh is not None:
                dp = max(1, replicas)
                self.device_mesh = DeviceMesh.from_sizes(world_size=dp * tp, dp_size=dp, tp_size=tp)
        return tuple(get_worker_id(actor) for actor in actors)

    def remove_replicas(self, worker_ids):
        """Retire model replicas by id: shrink the actor table and mesh in step, terminate via ray.kill (mirror of add_replicas)."""
        import ray

        worker_ids = tuple(worker_ids)
        with self._worker_lock:
            by_id = {get_worker_id(actor): actor for actor in getattr(self, '_actors', ()) or ()}
            unknown = set(worker_ids) - set(by_id)
            if unknown:
                raise ValueError(f'Unknown replica actor ids: {sorted(unknown)}')
            actors = [by_id[worker_id] for worker_id in worker_ids]
            self._actors = tuple(
                actor for actor in getattr(self, '_actors', ()) or ()
                if get_worker_id(actor) not in set(worker_ids))
            mesh = getattr(self, 'device_mesh', None)
            if mesh is not None:
                tp = getattr(mesh, 'tp_world_size', 1) or 1
                dp = max(1, len(self._actors) // tp)
                self.device_mesh = DeviceMesh.from_sizes(world_size=dp * tp, dp_size=dp, tp_size=tp)
            self._drop_worker_bookkeeping(worker_ids)
        error = None
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error
        return True

    def _drop_worker_bookkeeping(self, worker_ids):
        """Retire removed workers' bookkeeping: sticky ownership, pool admission tables, cw load.

        Called from both retirement paths (_discard_workers and remove_replicas) so model replicas
        get the same cleanup as engine workers. Re-entrant: callers already hold _worker_lock (an
        RLock) or it does not exist yet.
        """
        lock = getattr(self, '_worker_lock', None)
        with (lock if lock is not None else nullcontext()):
            for worker_id in worker_ids:
                _sticky_drop_worker(worker_id)
            if hasattr(self, '_sampler_workers'):
                for worker_id in worker_ids:
                    self._sampler_workers.pop(worker_id, None)
                self._suspended_workers.difference_update(worker_ids)
                self._paused_generation_workers.difference_update(worker_ids)
            state = getattr(self, '_continous_work_state', None)
            if state is not None:
                with state['lock']:
                    refs = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
                    _cw_drop_locked(state, refs)
                    for worker_id in worker_ids:
                        state['load'].pop(worker_id, None)

    def _discard_workers(self, actors):
        """Terminate engines and actors; keep cleaning the remaining members even if one close fails."""
        import ray

        actors = tuple(actors)
        worker_ids = {get_worker_id(actor) for actor in actors}
        error = None
        shutdown_refs = []
        for actor in actors:
            try:
                shutdown_refs.append(actor.shutdown.remote())
            except BaseException as exc:
                error = error or exc
        try:
            for ref in shutdown_refs:
                try:
                    ray.get(ref)
                except BaseException as exc:
                    error = error or exc
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except BaseException as exc:
                    error = error or exc
            lock = getattr(self, '_worker_lock', None)
            with (lock if lock is not None else nullcontext()):
                self._actors = tuple(actor for actor in self._actors if get_worker_id(actor) not in worker_ids)
                self._drop_worker_bookkeeping(worker_ids)
        if error is not None:
            raise error

    def close_workers(self):
        """Use only when terminating the whole component: close admission, cancel requests, clean up all engines."""
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Worker lifecycle belongs to its original driver handler.')
        if not hasattr(self, '_actors'):
            return self.shutdown()
        if hasattr(self, '_sampler_workers'):
            self.close_admission()
        try:
            cancel = getattr(self, 'abort_all_inflight', None)
            if self._actors and cancel is not None:
                cancel()
        finally:
            self._discard_workers(tuple(self._actors))


# The elasticity contract, attached to every @remote_class class by the decorator. ``__getstate__``
# only mounts when the class does not define its own, so a component's custom pickling wins.
_POOL_METHODS = (
    '_save_worker_factory', 'configure_workers', 'worker_ids', 'worker_ranks',
    '_sampling_worker_ids', '_begin_worker_call', '_end_worker_dispatch', '_end_worker_call',
    '_worker_call_conflicts', '_worker_call_completed', '_settle_worker_controls',
    '_record_worker_configuration', 'add_workers', '_worker_ready', 'activate_workers', 'drain_workers',
    'pause_workers', 'suspend_admission', 'resume_admission', 'close_admission', 'worker_inflight',
    'inflight_count', 'workers_idle', 'remove_workers', 'add_replicas', 'remove_replicas',
    '_discard_workers', '_drop_worker_bookkeeping', 'close_workers',
)


def mount_worker_pool(cls) -> None:
    """Mount the elasticity-contract methods onto a ``@remote_class`` decorated class.

    The bodies live in infra (this module); the component class itself has none of these methods and
    no RayHelper. Callers write ``component.add_workers(...)`` like any plain method. A same-named
    method already on the class raises, so it cannot silently shadow the component's own semantics.
    """
    for name in _POOL_METHODS:
        if name in cls.__dict__:
            raise TypeError(f'@remote_class: {cls.__name__} already defines {name!r}; the worker-pool '
                            f'contract cannot be mounted over it.')
        setattr(cls, name, getattr(WorkerPool, name))
    if '__getstate__' not in cls.__dict__:
        cls.__getstate__ = WorkerPool.__getstate__
