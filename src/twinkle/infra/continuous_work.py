# Copyright (c) ModelScope Contributors. All rights reserved.
import functools
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

from twinkle.utils import Platform, get_logger

logger = get_logger()


def _get_workers(workers, execute):
    if not workers:
        return []
    if execute == 'first':
        return [workers[0]]
    elif execute == 'all':
        return workers
    elif execute == 'peer':
        return workers[Platform.get_peer_index(len(workers))]
    else:
        raise ValueError(f'Unsupported execute method: {execute}')

# Affinity survives completed calls, but idle sessions must not accumulate forever.
# Expired entries are reclaimed on the next sticky call for the same method.
_CW_SESSION_IDLE_TIMEOUT = 30 * 60
# Limit bindings per driver and method; overflow evicts the least recently used session.
_CW_SESSION_CACHE_SIZE = 10_000


@dataclass
class _StickySessionBinding:
    worker_id: str
    last_used: float


def get_worker_id(worker) -> str:
    """Return the Ray actor's stable identity, independent of its position in the handler."""
    return worker._actor_id.hex()


def _select_workers(self, worker_ids=None):
    workers = tuple(self._actors)
    if worker_ids is None:
        return workers
    worker_ids = tuple(worker_ids)
    if any(not isinstance(worker_id, str) for worker_id in worker_ids):
        raise ValueError('_worker_ids must contain stable actor ID strings.')
    if len(set(worker_ids)) != len(worker_ids):
        raise ValueError('_worker_ids must not contain duplicate actor IDs.')
    by_id = {get_worker_id(worker): worker for worker in workers}
    unknown = set(worker_ids) - by_id.keys()
    if unknown:
        raise ValueError(f'Unknown actor IDs in _worker_ids: {sorted(unknown)}')
    return tuple(by_id[worker_id] for worker_id in worker_ids)


# Guards creating the per-handle state below. Without it two threads arriving at
# once each build their own state, with their own lock, and one overwrites the
# other -- after which the two are no longer excluding each other and the requests
# charged to the discarded one are never given back.
_CW_CREATE_LOCK = threading.Lock()

# Prefix of the awaitable companion generated for a continuous-work method. The
# companion is what the driver actually calls on the worker; see
# ``_make_worker_async_companion``.
_WORKER_ASYNC_PREFIX = '_twinkle_async_'


def _worker_executor(self, ):
    """Threads for running a blocking worker method off the actor's event loop.

    Sized from ``TWINKLE_ACTOR_MAX_CONCURRENCY``, which ``create_workers`` sets to
    the actor's ``max_concurrency``: any fewer threads than that would throttle
    below the concurrency the actor was configured for. A private executor rather
    than the loop's default one, so this never changes behaviour for anything else
    running on that loop.
    """
    executor = getattr(self, '_twinkle_worker_executor', None)
    if executor is not None:
        return executor
    with _CW_CREATE_LOCK:
        executor = getattr(self, '_twinkle_worker_executor', None)
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            n = int(os.environ.get('TWINKLE_ACTOR_MAX_CONCURRENCY') or 0) or 1
            executor = ThreadPoolExecutor(max_workers=n, thread_name_prefix='twinkle-worker')
            self._twinkle_worker_executor = executor
        return executor


def _make_worker_async_companion(func, wrapper):
    """Wrap a blocking worker method so the actor can run several of them at once.

    Ray makes a class with any ``async def`` into an asyncio actor, and there a
    blocking method holds the actor's single event loop for its whole duration --
    so calls queue and run one after another however high ``max_concurrency`` is.
    Measured on this sampler: four concurrent one-prompt calls took 3.98x as long
    as one, while the same four prompts in a single call took 1.02x. Handing the
    blocking body to a thread leaves the loop free to accept the next call, which
    is what puts several requests in the worker's engine together.
    """
    import asyncio

    @functools.wraps(func)
    async def companion(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_worker_executor(self), functools.partial(wrapper, self, *args, **kwargs))

    companion.__name__ = _WORKER_ASYNC_PREFIX + func.__name__
    return companion


def _cw_state(self):
    """Driver-side bookkeeping for ``enable_continous_work``, created on first use.

    ``load`` counts requests handed to each worker and not yet returned, which is
    what picks the next worker. ``inflight`` keeps those counts honest per method
    name, and is also what the barrier reads: a method other than the one with
    requests in flight must wait for them, because the worker now runs methods
    side by side and something like receiving weights or sleeping would otherwise
    land on an engine mid-generation.

    The state's own lock is a plain ``Lock``: nothing here takes it while already
    holding it, so re-entrance is not needed.
    """
    state = getattr(self, '_continous_work_state', None)
    if state is not None:
        return state
    with _CW_CREATE_LOCK:
        state = getattr(self, '_continous_work_state', None)
        if state is None:
            state = {
                'lock': threading.Lock(),
                'load': {},
                'inflight': {},
                'owners': {},
                'session_bindings': {},
            }
            self._continous_work_state = state
        return state


def _cw_drop_locked(state, refs):
    for ref in refs:
        owner = state['owners'].pop(ref, None)
        if owner is not None:
            worker_id, charge = owner
            state['load'][worker_id] = state['load'].get(worker_id, 0) - charge
        for name, pending in tuple(state['inflight'].items()):
            if ref in pending:
                pending.remove(ref)
            if not pending:
                state['inflight'].pop(name, None)


def _cw_barrier(self, current_func: str, workers, *, materialize=True) -> None:
    """Drain every other method's in-flight work before proceeding.

    Best effort by construction: another thread may submit again the moment this
    returns. It removes the case this exists for -- a weight update or a sleep
    issued while generations are still running -- but it is not a global lock on
    the worker.
    """
    state = getattr(self, '_continous_work_state', None)
    if not state:
        return
    worker_ids = {get_worker_id(worker) for worker in workers}
    conflicts = getattr(self, '_worker_call_conflicts', None)
    with state['lock']:
        others = {
            name: [ref for ref in refs if state['owners'][ref][0] in worker_ids
                   and (conflicts is None or conflicts(current_func, name, state['owners'][ref][0]))]
            for name, refs in state['inflight'].items() if name != current_func
        }
        others = {name: refs for name, refs in others.items() if refs}
    if not others:
        return
    import ray
    flat = [ref for refs in others.values() for ref in refs]
    logger.debug(f'continous_work barrier: {current_func} waits for {len(flat)} pending request(s) '
                 f'from {sorted(others)}')
    try:
        if materialize:
            ray.get(flat)
        else:
            ray.wait(flat, num_returns=len(flat))
    finally:
        ready, _ = ray.wait(flat, num_returns=len(flat), timeout=0)
        with state['lock']:
            _cw_drop_locked(state, ready)


def _cw_object_refs(result) -> List[Any]:
    """Every ObjectRef inside a dispatch result, tuples included."""
    import ray
    refs = []
    for item in (result or []):
        for candidate in (item if isinstance(item, tuple) else (item, )):
            if isinstance(candidate, ray.ObjectRef):
                refs.append(candidate)
    return refs


def _cw_register(self, func_name: str, result, workers) -> List[Any]:
    """Record a non-continuous call's refs so a later different method waits for it.

    Needed because a lazily collected method returns before its work finishes:
    ``receive_weights`` hands back a handle while the worker is still swapping
    weights, and with actor concurrency on, a sample issued right after would read
    them half written.
    """
    refs = _cw_object_refs(result)
    if not refs:
        return refs
    state = _cw_state(self)
    with state['lock']:
        for worker, item in zip(workers, result):
            for ref in _cw_object_refs([item]):
                state['owners'][ref] = (get_worker_id(worker), 0)
        state['inflight'].setdefault(func_name, []).extend(refs)
    return refs


def _cw_unregister(self, func_name: str, refs: List[Any]) -> None:
    state = getattr(self, '_continous_work_state', None)
    if not state or not refs:
        return
    import ray
    ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
    with state['lock']:
        _cw_drop_locked(state, ready)


def _cw_plan(n_workers: int, load: List[int], batch_len: int) -> List[List[int]]:
    """Assign each request to the worker holding the fewest, updating ``load``.

    Least-loaded-first, one request at a time, so a call of one request goes to
    one worker instead of being padded up to the worker count, and a call of many
    spreads out. ``load`` is mutated by the caller's lock holder.
    """
    per_worker: List[List[int]] = [[] for _ in range(n_workers)]
    for idx in range(batch_len):
        target = min(range(n_workers), key=lambda w: load[w])
        per_worker[target].append(idx)
        load[target] += 1
    return per_worker


def _validate_cw_session_ids(session_ids, batch_len: Optional[int]) -> tuple:
    """Validate driver-only routing metadata before submitting any RPC."""
    if not isinstance(session_ids, (list, tuple)):
        raise TypeError('_session_ids must be a list or tuple aligned with the input batch.')
    if batch_len is None:
        raise ValueError('_session_ids requires a materialized list batch; LazyCollect inputs are not supported.')
    if len(session_ids) != batch_len:
        raise ValueError(f'_session_ids has {len(session_ids)} entries but the input batch has {batch_len}.')
    if any(session_id is not None and (not isinstance(session_id, str) or not session_id)
           for session_id in session_ids):
        raise ValueError('_session_ids entries must be non-empty strings or None.')
    return tuple(session_ids)


def _sticky_session_cw_plan(worker_ids: tuple, load: List[int], session_ids: tuple,
                            session_bindings: OrderedDict, *, now: Optional[float] = None) -> List[List[int]]:
    """Plan session affinity under the caller's continuous-work lock.

    New sessions use least load; existing sessions keep their stable actor ID even
    when workers are reordered or added. Idle expiry, LRU eviction, or a bound
    actor missing from this call's eligible workers permits rebinding. ``None``
    uses ordinary least-load routing without retaining a binding. Like ``_cw_plan``, update the
    local planning loads; the RPC submission loop owns actual load accounting.
    """
    if not worker_ids:
        raise RuntimeError('No sampler worker is available for sticky session routing.')
    if now is None:
        now = time.monotonic()
    # Entries are ordered by last use, so cleanup scans only expired sessions.
    while session_bindings:
        oldest = next(iter(session_bindings.values()))
        if now - oldest.last_used < _CW_SESSION_IDLE_TIMEOUT:
            break
        session_bindings.popitem(last=False)

    by_id = {worker_id: index for index, worker_id in enumerate(worker_ids)}
    per_worker: List[List[int]] = [[] for _ in worker_ids]
    for index, session_id in enumerate(session_ids):
        binding = session_bindings.get(session_id) if session_id is not None else None
        target = by_id.get(binding.worker_id) if binding is not None else None
        if target is None:
            target = min(range(len(worker_ids)), key=lambda worker_index: load[worker_index])
        if session_id is not None:
            session_bindings[session_id] = _StickySessionBinding(worker_ids[target], now)
            session_bindings.move_to_end(session_id)
            while len(session_bindings) > _CW_SESSION_CACHE_SIZE:
                session_bindings.popitem(last=False)
        per_worker[target].append(index)
        load[target] += 1
    return per_worker


def _cw_batch_len(args, kwargs) -> Optional[int]:
    """Length of the request list, i.e. the first list argument's length.

    Same convention as ``dispatch='slice'``: list arguments are the batch and
    everything else is broadcast. Returns None when there is no list to split,
    which is how the caller knows to fall back to the normal dispatch.
    """
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, list):
            return len(arg)
    return None


def _cw_sub_args(args, kwargs, indices: List[int], batch_len: int):
    """The arguments for one worker: list arguments indexed, the rest as-is."""

    def pick(arg):
        if isinstance(arg, list) and len(arg) == batch_len:
            return [arg[i] for i in indices]
        return arg

    return tuple(pick(a) for a in args), {k: pick(v) for k, v in kwargs.items()}


def _sticky_state(self):
    """Driver-side sticky-ownership bookkeeping (``sticky``/``control`` declarations), created on first use.

    ``owners`` maps a sticky key (e.g. a request id) to the worker id that first served it, so a later
    ``control`` call with the same key lands on the worker actually running that request. Plain dict
    under its own lock: nothing here re-enters while holding it.
    """
    state = getattr(self, '_sticky_owners_state', None)
    if state is not None:
        return state
    with _CW_CREATE_LOCK:
        state = getattr(self, '_sticky_owners_state', None)
        if state is None:
            state = {'lock': threading.Lock(), 'owners': {}}
            self._sticky_owners_state = state
        return state


def _sticky_lookup(self, key):
    """The worker id owning ``key``, or ``None`` when no sticky call has served it yet."""
    state = getattr(self, '_sticky_owners_state', None)
    if state is None:
        return None
    with state['lock']:
        return state['owners'].get(key)


def _sticky_bind(self, key, worker_id):
    """Record that ``worker_id`` now owns ``key``; an existing binding is left alone (first wins)."""
    state = _sticky_state(self)
    with state['lock']:
        state['owners'].setdefault(key, worker_id)


def _sticky_drop_worker(self, worker_id):
    """Remove every ownership pointing at a retired worker (called from worker removal)."""
    state = getattr(self, '_sticky_owners_state', None)
    if state is None:
        return
    with state['lock']:
        state['owners'] = {key: owner for key, owner in state['owners'].items() if owner != worker_id}


def _resolve_sticky_key(sticky_key, func, args, kwargs):
    """Resolve the declared ``sticky_key`` against one bound call to ``func``.

    A string is a ``param.field.path``: the first segment names the parameter (matched against the
    keyword names first, then the positional slots of ``func``'s signature), the rest walk dict keys or
    object attributes (no eval). A callable receives the bound ``args``/``kwargs``. Returns ``None``
    when the path is absent, so the wrapper can treat it as "no key on this call".
    """
    if sticky_key is None:
        return None
    if callable(sticky_key):
        return sticky_key(args, kwargs)
    parts = str(sticky_key).split('.')
    param_name, path = parts[0], parts[1:]
    if param_name in kwargs:
        value = kwargs[param_name]
    else:
        import inspect
        try:
            parameters = list(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            return None
        if parameters and parameters[0] == 'self':
            # func is the unbound class-body function; the wrapper's args exclude self.
            parameters = parameters[1:]
        if param_name not in parameters:
            return None
        position = parameters.index(param_name)
        if position >= len(args):
            return None
        value = args[position]
    for segment in path:
        if isinstance(value, dict):
            if segment not in value:
                return None
            value = value[segment]
        else:
            value = getattr(value, segment, None)
            if value is None:
                return None
    return value


def _run_continous_work(self, func_name: str, execute_method, workers, args, kwargs, batch_len: int,
                        ray_get_timeout: Optional[float], *, on_dispatched=None, session_ids=None):
    """Submit one call per chosen worker and return results in the caller's order.

    Submission happens under the lock so that picking a worker and charging it are
    one step -- several caller threads land here at once, and a split of the two
    would let them all pick the same idle worker. Waiting happens outside it.
    """
    import ray

    state = _cw_state(self)
    worker_ids = tuple(get_worker_id(worker) for worker in workers)
    submitted = []
    # The awaitable form, so the worker can hold several of these at once. Book-
    # keeping still uses the plain name, which is what callers and the barrier see.
    remote_name = _WORKER_ASYNC_PREFIX + func_name
    try:
        with state['lock']:
            pending = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
            if pending:
                ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
                _cw_drop_locked(state, ready)
            load = [state['load'].get(worker_id, 0) for worker_id in worker_ids]
            if session_ids is not None:
                bindings = state['session_bindings'].setdefault(func_name, OrderedDict())
                plan = _sticky_session_cw_plan(worker_ids, load, session_ids, bindings)
            else:
                plan = _cw_plan(len(workers), load, batch_len)
            for worker_index, indices in enumerate(plan):
                if not indices:
                    continue
                sub_args, sub_kwargs = _cw_sub_args(args, kwargs, indices, batch_len)
                ref = execute_method(remote_name, [(workers[worker_index], sub_args, sub_kwargs)])[0]
                worker_id = worker_ids[worker_index]
                state['load'][worker_id] = state['load'].get(worker_id, 0) + len(indices)
                state['owners'][ref] = (worker_id, len(indices))
                state['inflight'].setdefault(func_name, []).append(ref)
                submitted.append((worker_index, indices, ref))
    finally:
        if on_dispatched is not None:
            on_dispatched()

    try:
        ordered: List[Any] = [None] * batch_len
        for _, indices, ref in submitted:
            part = ray.get(ref, timeout=ray_get_timeout) if ray_get_timeout is not None else ray.get(ref)
            if not isinstance(part, (list, tuple)) or len(part) != len(indices):
                raise TypeError(f'{func_name}: enable_continous_work needs one result per request, but a worker given '
                                f'{len(indices)} request(s) returned {type(part).__name__} of length '
                                f'{len(part) if isinstance(part, (list, tuple)) else "n/a"}.')
            for local_index, original_index in enumerate(indices):
                ordered[original_index] = part[local_index]
        return ordered
    finally:
        _cw_unregister(self, func_name, [ref for _, _, ref in submitted])
