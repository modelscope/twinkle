# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-request routing for ``enable_continous_work`` methods, and the replica membership it reads.

A component class with a continuous method is a *pool*: its replicas are listed in the
``WorkerRegistry`` (``{worker_id: {'actor', 'state', ...}}``, state WARMING / ACTIVE / DRAINING).
Every process holding a handle keeps a :class:`_PoolView` -- the ACTIVE members cached for about a
second, the requests it has in flight per member, and its sticky bindings -- so picking a replica
never leaves the process. A replica that is being removed answers with :class:`WorkerDraining`; the
caller drops it from its view and resubmits the request elsewhere, so correctness never depends on
how fresh the cache is.
"""
import functools
import os
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from twinkle.utils import Platform, get_logger

logger = get_logger()


class WorkerDraining(RuntimeError):
    """Raised by a replica that is leaving its pool; the caller resubmits the request to another one."""


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
_CW_SESSION_IDLE_TIMEOUT = 30 * 60
# Limit bindings per pool view; overflow evicts the least recently used session.
_CW_SESSION_CACHE_SIZE = 10_000
# How long a process trusts its cached membership before re-reading the registry.
_POOL_TTL = 1.0
# How many times one request is resubmitted after its replica drained or died.
_CW_MAX_RETRIES = 8

# Prefix of the awaitable companion generated for a continuous-work method. The
# companion is what the caller actually invokes on the worker; see
# ``_make_worker_async_companion``.
_WORKER_ASYNC_PREFIX = '_twinkle_async_'

_CREATE_LOCK = threading.Lock()

# Threads that wait for a generation are cheap: they park on a future and hold no engine
# resource, so their number should track the in-flight request count a rollout driver keeps
# open (a hundred-plus episodes is normal), not the actor's admission concurrency. Threads
# spawn lazily up to this ceiling, so a large default costs nothing when idle.
_WORKER_WAIT_THREADS_FLOOR = 256


def worker_id_of(actor) -> str:
    return actor._actor_id.hex()


def _worker_executor(self, ):
    """Threads for running a blocking worker method off the actor's event loop.

    The floor is decoupled from the actor's ``max_concurrency`` on purpose: a waiting thread
    blocks on a future for the whole generation while holding no engine resource, so sizing it
    like the admission concurrency (24) caps the in-flight generations at 24 -- measured on the
    vLLM sampler as a 3.5x per-request slowdown for a 100-request rollout. ``
    TWINKLE_ACTOR_MAX_CONCURRENCY`` (set by ``create_workers``) still raises the floor when the
    actor is configured wider than it, and ``TWINKLE_ACTOR_WORKER_THREADS`` overrides both.
    A private executor rather than the loop's default one, so this never changes behaviour for
    anything else running on that loop.
    """
    executor = getattr(self, '_twinkle_worker_executor', None)
    if executor is not None:
        return executor
    with _CREATE_LOCK:
        executor = getattr(self, '_twinkle_worker_executor', None)
        if executor is None:
            n = int(os.environ.get('TWINKLE_ACTOR_WORKER_THREADS') or 0) or max(
                int(os.environ.get('TWINKLE_ACTOR_MAX_CONCURRENCY') or 0), _WORKER_WAIT_THREADS_FLOOR)
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

    # functools.wraps pins __wrapped__, and inspect.signature follows it -- so Ray would validate
    # a call against the original method's parameters instead of this companion's (self, *args,
    # **kwargs) and reject the _dispatch_context that _submit adds when the arguments carry refs.
    del companion.__wrapped__
    companion.__name__ = _WORKER_ASYNC_PREFIX + func.__name__
    return companion


# Local mode: one thread pool per component instance, sized by its max_concurrency (or 1), that runs
# the methods declaring both ``enable_continous_work`` and ``lazy_collect=True``. Keyed by id with a
# finalizer, so the component itself carries no unpicklable attribute.
_LOCAL_EXECUTORS: Dict[int, ThreadPoolExecutor] = {}


def _local_executor(self) -> ThreadPoolExecutor:
    key = id(self)
    executor = _LOCAL_EXECUTORS.get(key)
    if executor is not None:
        return executor
    with _CREATE_LOCK:
        executor = _LOCAL_EXECUTORS.get(key)
        if executor is None:
            size = getattr(self, '_max_concurrency', None) or 1
            executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix=f'twinkle-{type(self).__name__}')
            _LOCAL_EXECUTORS[key] = executor
            weakref.finalize(self, _drop_local_executor, key)
        return executor


def _drop_local_executor(key):
    executor = _LOCAL_EXECUTORS.pop(key, None)
    if executor is not None:
        executor.shutdown(wait=False)


def _cw_plan(n_workers: int, load: List[int], batch_len: int) -> List[List[int]]:
    """Assign each request to the worker holding the fewest, updating ``load``.

    Least-loaded-first, one request at a time, so a call of one request goes to
    one worker instead of being padded up to the worker count, and a call of many
    spreads out.
    """
    per_worker: List[List[int]] = [[] for _ in range(n_workers)]
    for idx in range(batch_len):
        target = min(range(n_workers), key=lambda w: load[w])
        per_worker[target].append(idx)
        load[target] += 1
    return per_worker


def _cw_batch_len(args, kwargs) -> Optional[int]:
    """Length of the request list, i.e. the first list argument's length.

    Same convention as ``dispatch='slice'``: list arguments are the batch and
    everything else is broadcast. Returns None when there is no list to split,
    which is a single request.
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


class _PoolView:
    """This process's view of one pool: cached ACTIVE members, local load and sticky bindings.

    ``key`` names the pool in the registry; a view without a key is a fixed actor set (a sub-handle
    holding only new replicas during warmup) and never refreshes.
    """

    def __init__(self, key: Optional[str], actors=()):
        self.key = key
        self.lock = threading.Lock()
        self.members: Dict[str, Any] = {worker_id_of(actor): actor for actor in actors}
        self.load: Dict[str, int] = {}
        self.bindings: 'OrderedDict[Any, list]' = OrderedDict()
        self.refreshed = 0.0 if key is not None else float('inf')

    def refresh(self, force: bool = False):
        if self.key is None:
            return
        now = time.monotonic()
        if not force and now - self.refreshed < _POOL_TTL:
            return
        from ._ray import RayHelper
        members = RayHelper.pool_get(self.key)
        with self.lock:
            self.members = {
                worker_id: member['actor']
                for worker_id, member in members.items() if member['state'] == 'ACTIVE'
            }
            self.refreshed = now

    def actors(self) -> List[Any]:
        self.refresh()
        with self.lock:
            return list(self.members.values())

    def evict(self, worker_id: str):
        """Forget a replica that answered draining or died, so the next pick avoids it."""
        with self.lock:
            self.members.pop(worker_id, None)
            for key in [key for key, binding in self.bindings.items() if binding[0] == worker_id]:
                self.bindings.pop(key)
            if self.key is not None:
                self.refreshed = 0.0

    def assign(self, count: int, sticky_value=None) -> List[Tuple[str, Any, List[int]]]:
        """Split ``count`` requests over members: the sticky owner if bound, else least loaded first."""
        self.refresh()
        now = time.monotonic()
        with self.lock:
            if not self.members:
                raise RuntimeError(f'No active worker in pool {self.key!r}.')
            while self.bindings:
                oldest = next(iter(self.bindings.values()))
                if now - oldest[1] < _CW_SESSION_IDLE_TIMEOUT:
                    break
                self.bindings.popitem(last=False)
            worker_ids = list(self.members)
            if sticky_value is not None:
                binding = self.bindings.get(sticky_value)
                target = binding[0] if binding is not None and binding[0] in self.members else None
                if target is None:
                    target = min(worker_ids, key=lambda w: self.load.get(w, 0))
                self.bindings[sticky_value] = [target, now]
                self.bindings.move_to_end(sticky_value)
                while len(self.bindings) > _CW_SESSION_CACHE_SIZE:
                    self.bindings.popitem(last=False)
                plan = [list(range(count)) if worker_id == target else [] for worker_id in worker_ids]
            else:
                plan = _cw_plan(len(worker_ids), [self.load.get(w, 0) for w in worker_ids], count)
            assigned = []
            for worker_id, indices in zip(worker_ids, plan):
                if indices:
                    self.load[worker_id] = self.load.get(worker_id, 0) + len(indices)
                    assigned.append((worker_id, self.members[worker_id], indices))
            return assigned

    def release(self, worker_id: str, count: int):
        with self.lock:
            left = self.load.get(worker_id, 0) - count
            if left > 0:
                self.load[worker_id] = left
            else:
                self.load.pop(worker_id, None)

    def least_loaded(self, k: int) -> List[str]:
        self.refresh(force=True)
        with self.lock:
            return sorted(self.members, key=lambda w: self.load.get(w, 0))[:k]


_POOL_VIEWS: Dict[str, _PoolView] = {}


def _pool_view(handle) -> _PoolView:
    """The view a handle routes through: its registry pool, or the fixed actor set it holds."""
    key = handle.__dict__.get('_pool_key')
    if key is None:
        key = 'fixed:' + ','.join(worker_id_of(actor) for actor in handle._actors)
        fixed = True
    else:
        fixed = False
    view = _POOL_VIEWS.get(key)
    if view is None:
        with _CREATE_LOCK:
            view = _POOL_VIEWS.get(key)
            if view is None:
                view = _PoolView(None if fixed else key, handle._actors if fixed else ())
                _POOL_VIEWS[key] = view
    return view


# Resubmissions run here, never on Ray's completion-callback thread: refreshing the view reads the
# registry, which blocks.
_RETRY_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _retry_executor() -> ThreadPoolExecutor:
    global _RETRY_EXECUTOR
    if _RETRY_EXECUTOR is None:
        with _CREATE_LOCK:
            if _RETRY_EXECUTOR is None:
                _RETRY_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix='twinkle-cw-retry')
    return _RETRY_EXECUTOR


def _is_retryable(exc: BaseException) -> bool:
    import ray
    if isinstance(exc, (WorkerDraining, ray.exceptions.RayActorError)):
        return True
    return isinstance(getattr(exc, 'cause', None), WorkerDraining)


class _ContinuousCall:
    """One continuous-work call in flight: its parts on the chosen replicas and the caller's Future.

    Each part completes through ``ObjectRef.future()``; a part whose replica drained or died is
    resubmitted elsewhere with the same request indices. The Future resolves to the per-request
    results in the caller's order (a list for a list call, the single result otherwise).
    """

    def __init__(self, view: _PoolView, func_name: str, args, kwargs, batch_len: Optional[int], sticky_value,
                 has_ref: bool):
        self.view = view
        self.remote_name = _WORKER_ASYNC_PREFIX + func_name
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.batch_len = batch_len
        self.sticky_value = sticky_value
        self.has_ref = has_ref
        self.future: Future = Future()
        self.results: List[Any] = [None] * (batch_len or 1)
        self.pending = 0
        self.retries = 0
        self.lock = threading.Lock()

    def start(self):
        count = self.batch_len if self.batch_len is not None else 1
        if count == 0:
            self.future.set_result([])
            return self.future
        self._submit(list(range(count)))
        return self.future

    def _submit(self, indices: List[int], replacing: bool = False):
        """Send ``indices`` out; ``replacing`` marks a resubmission that takes over one pending part."""
        try:
            assigned = self.view.assign(len(indices), self.sticky_value)
        except BaseException as exc:  # noqa
            self._fail(exc)
            return
        with self.lock:
            self.pending += len(assigned) - (1 if replacing else 0)
        for worker_id, actor, local in assigned:
            part = [indices[i] for i in local]
            if self.batch_len is None:
                sub_args, sub_kwargs = self.args, dict(self.kwargs)
            else:
                sub_args, sub_kwargs = _cw_sub_args(self.args, self.kwargs, part, self.batch_len)
            if self.has_ref:
                # The worker materialises nested refs itself; one worker means no re-slicing there.
                sub_kwargs['_dispatch_context'] = (1, 0, True)
            try:
                ref = getattr(actor, self.remote_name).remote(*sub_args, **sub_kwargs)
            except BaseException as exc:  # noqa
                self._on_part(worker_id, part, None, exc)
                continue
            ref.future().add_done_callback(functools.partial(self._on_ref, worker_id, part))

    def _on_ref(self, worker_id, part, ref_future):
        exc = ref_future.exception()
        self._on_part(worker_id, part, None if exc is not None else ref_future.result(), exc)

    def _on_part(self, worker_id, part, value, exc):
        self.view.release(worker_id, len(part))
        if exc is not None and _is_retryable(exc) and self.retries < _CW_MAX_RETRIES:
            self.view.evict(worker_id)
            with self.lock:
                self.retries += 1
            _retry_executor().submit(self._submit, part, True)
            return
        if exc is not None:
            self._fail(exc)
            return
        if self.batch_len is None:
            self.results[0] = value
        else:
            if not isinstance(value, (list, tuple)) or len(value) != len(part):
                self._fail(
                    TypeError(f'{self.func_name}: enable_continous_work needs one result per request, but a worker '
                              f'given {len(part)} request(s) returned {type(value).__name__} of length '
                              f'{len(value) if isinstance(value, (list, tuple)) else "n/a"}.'))
                return
            for local_index, original_index in enumerate(part):
                self.results[original_index] = value[local_index]
        with self.lock:
            self.pending -= 1
            if self.pending == 0 and not self.future.done():
                self.future.set_result(self.results[0] if self.batch_len is None else self.results)

    def _fail(self, exc):
        with self.lock:
            if self.future.done():
                return
            self.future.set_exception(exc)
