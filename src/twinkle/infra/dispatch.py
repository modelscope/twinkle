# Copyright (c) ModelScope Contributors. All rights reserved.
import functools
import inspect
import os
import sys
import uuid
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar, Union

import numpy as np
from twinkle.notifier import notify_exception
from twinkle.utils import DeviceGroup, DeviceMesh, Platform, check_unsafe, get_logger

from .continuous_work import _WORKER_ASYNC_PREFIX, WorkerDraining, _ContinuousCall, _cw_batch_len, _get_workers, \
    _local_executor, _make_worker_async_companion, _pool_view, _resolve_sticky_key, worker_id_of
from .worker_mixin import mount_worker_mixin

logger = get_logger()

T1 = TypeVar('T1', bound=object)
_PROCESS_OWNER = uuid.uuid4().hex


def _owner_identity():
    return os.getpid(), sys.modules[__name__]._PROCESS_OWNER


def _infra():
    """The ``twinkle.infra`` module object, for reading its process-level globals at call time."""
    return sys.modules['twinkle.infra']


def _capture_caller() -> Optional[str]:
    """Return ``file:line`` of the first frame outside this module, or ``None``."""
    f = sys._getframe(1)
    while f and f.f_code.co_filename == __file__:
        f = f.f_back
    return f'{f.f_code.co_filename}:{f.f_lineno}' if f else None


def _tag_exc(exc: BaseException, caller: Optional[str]) -> None:
    """Stamp driver-caller location onto exc for both traceback and str(exc)."""
    if not caller:
        return
    try:
        marker = f'[twinkle] driver caller: {caller}'
        if marker not in (getattr(exc, '__notes__', None) or []):
            exc.add_note(marker)
        if not getattr(exc, '_twinkle_caller_augmented', False):
            prefix = f'[twinkle driver caller: {caller}] '
            exc.args = (prefix + str(exc.args[0]), *exc.args[1:]) if exc.args else (prefix.rstrip(), )
            exc._twinkle_caller_augmented = True
    except Exception:  # noqa
        pass


def _collect_func(method: Union[Literal['none', 'flatten', 'mean', 'first', 'last_pp_first',
                                        'tensor_dict', 'dp_flatten'], Callable],
                  result: List[Any],
                  device_mesh: DeviceMesh = None):
    """Collect results

    Args:
        method:
            none: Return as is.
            flatten: Flat the nested results.
            mean: Average the results.
            first: Only return the first result.
            tensor_dict: Merge per-worker output dicts, stacking tensors (deduplicated by data rank).
            dp_flatten: Deduplicate per-worker lists by data rank, then flatten in input order.
        result: The results returned by workers.
        device_mesh: The device_mesh, needed by `last_pp_first`
    Returns:
        The collected results.
    """
    if callable(method):
        return method(result, device_mesh=device_mesh)
    if not result:
        return result

    if isinstance(result[0], tuple) and method != 'none':
        output = []
        # if each result of a worker is a tuple
        for i in range(len(result[0])):
            # handle each element in a tuple
            _single_result = [r[i] for r in result]
            output.append(_collect_func(method, _single_result, device_mesh=device_mesh))
        return output
    if method == 'none':
        # As-is: a single worker's tuple stays a tuple (never a positional list) -- the return
        # type must not drift between local and ray deployments.
        if isinstance(result, list) and len(result) == 1:
            # unwrap the result
            return result[0]
        else:
            return result
    elif method == 'flatten':
        # flatten
        flatten = [item for sublist in result for item in sublist]
        if isinstance(result[0], np.ndarray):
            return np.array(flatten)
        return type(result[0])(flatten)
    elif method == 'mean':
        if isinstance(result[0], dict):
            output = {}
            for key in result[0]:
                vals = [r[key] for r in result if key in r]
                try:
                    output[key] = np.mean(vals)
                except (TypeError, ValueError):
                    output[key] = vals
            return output
        return np.mean(result)
    elif method == 'first':
        return result[0]
    elif method == 'last_pp_first':
        # Return the first result from the last PP stage workers.
        # Falls back to result[0] when PP = 1 (all workers are the last stage).
        assert device_mesh is not None
        last_pp = [r for i, r in enumerate(result) if i in device_mesh.get_pp_last_ranks()]
        return last_pp[0] if last_pp else result[0]
    elif method == 'tensor_dict':
        from .collectors import tensor_dict
        return tensor_dict(result, device_mesh=device_mesh)
    elif method == 'dp_flatten':
        from .collectors import dp_flatten
        return dp_flatten(result, device_mesh=device_mesh)
    else:
        raise ValueError(f'Unsupported collect method: {method}')


def _dispatch_args(workers, dispatch, execute, device_mesh: Optional[DeviceMesh], args, kwargs):
    if not workers:
        return []
    if execute == 'first':
        return [(workers[0], args, kwargs)]
    if dispatch == 'all':
        return [(worker, args, kwargs) for worker in workers]
    length = len(workers)
    if callable(dispatch):
        result = []
        for index, worker in enumerate(workers):
            sliced_args, sliced_kwargs = dispatch(length, index, args, kwargs, device_mesh=device_mesh)
            result.append((worker, sliced_args, sliced_kwargs))
        return result
    if dispatch not in ('slice', 'slice_dp'):
        raise ValueError(f'Unsupported dispatch method: {dispatch}')

    from ._ray.ray_helper import LazyCollect, RayHelper
    if dispatch == 'slice_dp':
        if device_mesh is None:
            raise ValueError('slice_dp dispatch requires a device_mesh.')
        stride = max(1, device_mesh.world_size // length)
        if stride * length != device_mesh.world_size:
            raise ValueError(f'{length} workers cannot tile mesh world_size={device_mesh.world_size}.')

    def slices(size):
        if dispatch == 'slice_dp':
            return [device_mesh.get_slice(size, device_mesh.get_data_rank_from_global_rank(index * stride))
                    for index in range(length)]
        count, remainder = divmod(size, length)
        return [slice(index * count + min(index, remainder),
                      (index + 1) * count + min(index + 1, remainder)) for index in range(length)]

    def split(value):
        import torch
        if RayHelper._is_handle(value):
            value = (value if isinstance(value, LazyCollect)
                     else LazyCollect([value], 'none', _collect_func, device_mesh))
            kind, structure = value._project('structure').result()
            if kind == 'sequence' or (dispatch == 'slice_dp' and kind == 'tensor'):
                return [value[part] for part in slices(structure)]
            if kind == 'mapping' and dispatch == 'slice_dp':
                value = {key: value[key] for key in structure}
            else:
                return [value] * length
        if isinstance(value, list) or (dispatch == 'slice_dp' and isinstance(value, torch.Tensor) and value.dim() > 0):
            return [value[part] for part in slices(len(value))]
        if dispatch == 'slice_dp' and isinstance(value, dict):
            columns = {key: split(item) for key, item in value.items()}
            return [{key: column[index] for key, column in columns.items()} for index in range(length)]
        return [value] * length

    arg_parts = [split(value) for value in args]
    kwarg_parts = {key: split(value) for key, value in kwargs.items()}
    return [(worker, tuple(parts[index] for parts in arg_parts),
             {key: parts[index] for key, parts in kwarg_parts.items()})
            for index, worker in enumerate(workers)]


def _get_device_mesh_param_name(init_method) -> str:
    """Try to get the device_mesh param name"""
    sig = inspect.signature(init_method)
    for param in sig.parameters.values():
        ann = param.annotation
        if ann != inspect.Parameter.empty:
            if hasattr(ann, '__name__') and ann.__name__ == 'DeviceMesh':
                return param.name
            if 'DeviceMesh' in str(ann):
                return param.name
    return ''


def _get_device_mesh_param(args, kwargs):
    """Try to get the device_mesh param instance"""
    for arg in (list(args) + list(kwargs.values())):
        if isinstance(arg, DeviceMesh):
            return arg
    return None


def _is_pool_class(cls) -> bool:
    """A class with a continuous-work method is a pool: its replicas are tracked in the registry."""
    return any(
        getattr(getattr(cls, name, None), '_enable_continous_work', False) for name in dir(cls)
        if not name.startswith('__') or name == '__call__')


def remote_class(execute: Literal['first', 'peer', 'all'] = 'all', max_concurrency: Optional[int] = None):
    """Patch each class used in remote clusters with this decorator.

    Use this decorator to wrap your class to enable it to execute in a remote cluster.

    Args:
        execute: which workers the class runs on.
        max_concurrency: Ray actor concurrency, i.e. how many of this class's
            methods one worker may run at once. ``None`` leaves Ray's default of
            1, under which concurrent calls to the same worker queue and run one
            after another. Only set it for a class whose methods tolerate running
            side by side: a class holding NCCL collectives does not, because two
            collectives interleaving on one rank deadlock. It is what
            ``enable_continous_work`` needs to reach the worker's engine
            concurrently instead of stopping at the actor boundary. A constructor
            call may override it with a ``max_concurrency=`` keyword, read here and
            never passed to the component (like ``remote_group``); in local mode it
            sizes the thread pool that runs the continuous lazy methods.
            ``call_timeout=`` on construction bounds calls without a method-level timeout.
            ``remote_group`` accepts a named group or a shared DeviceGroup view of initialized devices.
    """

    def decorator(cls):
        pool_class = _is_pool_class(cls)
        if pool_class:
            # Replica lifecycle (health / add_workers / remove_workers / pressure) lives in infra.
            mount_worker_mixin(cls)
        cls._twinkle_pool = pool_class
        # Give every continuous-work method its awaitable form on the class, so Ray
        # has something to await instead of a call that would sit on the actor's
        # event loop and make the others wait behind it.
        for _name in dir(cls):
            _attr = getattr(cls, _name, None)
            _companion = getattr(_attr, '_worker_async_companion', None)
            if _companion is not None:
                setattr(cls, _WORKER_ASYNC_PREFIX + _name, _companion)
        # Get device mesh parameter name
        device_mesh_name = _get_device_mesh_param_name(cls.__init__)
        init_method = cls.__init__
        # Lifecycle methods are DECLARED by the component: the class-level ``_lifecycle``
        # names this class's teardown entries -- the default protocol releases the class's
        # owned workers on ``close``, families with a real second entry (samplers, envs)
        # declare ``('close', 'shutdown')``. Infra never guesses from method names or
        # signatures: a declared ``close`` without a class body keeps the release-only
        # implementation, and re-decorating composes -- an already-wrapped method unwraps
        # to its component body first, so a business teardown is composed with, not
        # replaced by, the ownership release.
        lifecycle_methods = {}
        for name in getattr(cls, '_lifecycle', ('close',)):
            method = getattr(cls, name, None)
            if getattr(method, '_twinkle_owned_close', False):
                method = method._component_close
            lifecycle_methods[name] = method

        def new_init(self, *args, **kwargs):
            _caller = _capture_caller()
            _ctx = f'{cls.__name__}.__init__'
            if _caller:
                _ctx = f'{_ctx} <- {_caller}'
            try:
                _infra()._maybe_load_worker_notifier()
                _new_init_body(self, _caller, *args, **kwargs)
            except BaseException as _e:
                try:
                    _release_workers(self)
                except BaseException:
                    logger.exception('Component-construction rollback is incomplete.')
                _tag_exc(_e, _caller)
                notify_exception(_infra()._notifier, _ctx, _e, _infra()._name)
                raise

        # Only a class with its own __init__ (a plain function) is metadata-
        # copyable: a class that inherits object.__init__ carries the slot
        # wrapper instead, and functools.wraps would pin __wrapped__ onto it.
        # Ray's tracing injector unwraps every method to its original before
        # setting __signature__, which then dies on the wrapper descriptor.
        if inspect.isfunction(init_method):
            new_init = functools.wraps(init_method)(new_init)

        def _new_init_body(self, _caller, *args, **kwargs):
            infra = _infra()
            concurrency = kwargs.pop('max_concurrency', None) or max_concurrency
            self._ray_get_timeout = kwargs.pop('call_timeout', None)
            if infra._mode == 'local':
                # remote_group names the ray device group (the ray branch below
                # consumes then pops it). Local mode never dispatches, so it is
                # dropped here too: a component constructor must not receive it
                # -- plain HF classes reject the unknown keyword inside
                # from_pretrained kwargs.
                kwargs.pop('remote_group', None)
                # instance_id disambiguates ray actor names for handles built from one call site.
                # It is infra bookkeeping, so local mode drops it for the same reason: a component
                # constructor must not receive it, or the same call site would work under --mode ray
                # and raise TypeError under --mode local.
                kwargs.pop('instance_id', None)
                self._max_concurrency = concurrency
                # Get the actual device_mesh
                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name and infra._device_group is not None:
                    if device_mesh is None:
                        # Local mode can safely assign the default device mesh
                        device_mesh = infra._device_mesh
                        kwargs[device_mesh_name] = infra._device_mesh
                    assert len(infra._device_group) == 1  # only one device group is allowed
                    infra._device_group[0]._device_mesh[self.__class__.__name__] = device_mesh
                    init_method(self, *args, **kwargs)
                else:
                    if device_mesh is not None:
                        logger.warning(f'{cls.__name__} was given a device_mesh but it is being DROPPED: twinkle '
                                       'holds no device group, so call twinkle.initialize(...) before '
                                       'constructing it. Training will otherwise run with device_mesh=None '
                                       '(single-rank loss normalisation and no metric aggregation).')
                    args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                    kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                    init_method(self, *args, **kwargs)
            elif infra._mode == 'ray':
                from ._ray import RayHelper

                # In case the same class created twice in the same device group
                # Try to get the caller's line (resolved in ``new_init`` so it points
                # at user code, not at the wrapper itself).
                _cf, _, _cl = (_caller or f'{__file__}:0').rpartition(':')
                caller_file = _cf.replace(os.sep, '_').replace('.', '_')
                caller_line = _cl
                # Pass an instance_id is recommended
                instance_id = kwargs.pop('instance_id', '') + f'{caller_file}_{caller_line}'
                remote_group = kwargs.get('remote_group')
                if os.environ.get('WORKER_NAME') is None and remote_group is None:
                    logger.info(f'⚠️ Using local initialization of class: {cls}, please make sure the class '
                                'does not need remote execution.')
                # If cannot trust_remote_code, no callable and type can be used.
                check_unsafe(*args, **kwargs)

                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name:
                    if execute == 'first':
                        # Manually create a device_mesh because there is only one worker
                        device_mesh = DeviceMesh.from_sizes(dp_size=1)
                        kwargs[device_mesh_name] = device_mesh

                    if remote_group:
                        if device_mesh is None:
                            if infra._device_mesh is not None:
                                device_mesh = infra._device_mesh
                                kwargs[device_mesh_name] = device_mesh
                            else:
                                raise ValueError('Set device_mesh=DeviceMesh(...) to enable ray.')

                    if infra._device_group and remote_group:
                        # usually this happens in driver because worker does not has a valid _device_group
                        # this is used to print the device_group info, so pass the worker is ok
                        group_name = remote_group if isinstance(remote_group, str) else remote_group.name
                        device_group = next((dg for dg in infra._device_group if dg.name == group_name), None)
                        if device_group is not None:
                            device_group._device_mesh[self.__class__.__name__] = device_mesh

                # ``__iter_origin__`` (set once, below) is the user-declared iterator: without it,
                # a second instantiation of the same class would read the stash off our own
                # first-round wrapper, whose ``_stash`` is None.
                iter_origin = getattr(cls, '__iter_origin__', None) or getattr(cls, '__iter__', None)
                iter_stash = getattr(iter_origin, '_stash', None)

                # This will solve the iterator cannot be passed through ray.
                # ``_dispatch_context`` is absorbed here because these rebuilt bodies are the
                # actor-side methods Ray captures at create_workers() time -- BEFORE the
                # remote_function re-decoration below pins the signature. The dispatch path puts
                # ``_dispatch_context`` on the wire (see _call_replicas), so the actor method must
                # declare it or Ray's signature validator refuses the call.
                def __iter__(_self, _dispatch_context=None):
                    if os.environ.get('WORKER_NAME'):
                        # This is a worker, iter keeps in the class, pass nothing to driver
                        _iter = _self.__iter_origin__()
                        assert _iter is not _self
                        _self._iter = _iter
                    else:
                        # This is executed in driver
                        return _self.__iter_origin__()

                def __next__(_self, _dispatch_context=None):
                    # Use _self._iter to get the next data
                    # Only one driver can use this at one time
                    try:
                        # Return a tuple, get the second output in the driver to stop the for loop
                        value = next(_self._iter)
                    except StopIteration:
                        return [], True
                    if iter_stash is not None and iter_stash is not False:
                        # The stash declared on ``__iter__`` applies to each yielded batch;
                        # ``True`` stashes the whole (bare-list) batch element-wise.
                        value = _stash_channels(value, iter_stash)
                    return value, False

                if (not remote_group) or os.environ.get('CLUSTER_NAME') == remote_group:
                    # not remote_group: Ray mode with local component
                    # os.environ.get('CLUSTER_NAME') == remote_group: a normal worker's init
                    # All per-process torch setup (seed / determinism / tf32 / the worker CUDA-env pins)
                    # is isolated in _torch_setup, so none of it leaks into the Ray dispatch logic here.
                    from . import _torch_setup
                    _torch_setup.apply_in_worker(infra._extra_torch_kwargs)
                    if not device_mesh_name:
                        # pop the device_mesh
                        args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                        kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                    kwargs.pop('remote_group', None)  # component does not need this
                    self._max_concurrency = concurrency
                    init_method(self, *args, **kwargs)
                else:
                    if hasattr(cls, '__iter__'):
                        # Read the iterator's dispatch contract off the ORIGINAL user-declared
                        # ``__iter__`` (``iter_origin``), never off ``self.__iter__``: on a SECOND
                        # instantiation of the same class ``self.__iter__`` is already the round-1
                        # re-wrapper below (``collect='none'``, ``lazy_collect=None``), so reading it
                        # would silently drop the user's ``collect``/``lazy_collect`` and make every
                        # later round's ``__next__`` fall back to the global ``infra._lazy_collect``
                        # (True under ray) -- handing the trainer an unmaterialised LazyCollect batch
                        # instead of the plain (rows, facts) it declared (G35 item7 round-2 crash:
                        # 'LazyCollect' object has no attribute 'values'). ``iter_origin`` is the
                        # pre-wrap method on the first instantiation and the preserved original on
                        # every later one, so the contract is stable across rounds.
                        _dispatch = iter_origin._dispatch
                        _execute = iter_origin._execute
                        _collect = iter_origin._collect
                        _lazy = iter_origin._lazy_collect
                        import ray
                        # The user's original iterator is recorded ONCE: a later instantiation of
                        # the same class must not overwrite it with our own wrapper, whose body
                        # calls ``__iter_origin__`` back -- a chain of such wrappers would recurse
                        # forever if Ray ever re-serialised the current class instead of its
                        # cached snapshot.
                        if not hasattr(cls, '__iter_origin__'):
                            cls.__iter_origin__ = cls.__iter__
                        cls.__iter__ = __iter__
                        # Return 2 object refs to enable get the stop flag in driver
                        cls.__next__ = ray.method(num_returns=2)(__next__)

                    # Create remote workers
                    # Remove potential duplicate keys from kwargs before passing
                    kwargs_for_workers = kwargs.copy()
                    kwargs_for_workers.pop('instance_id', None)
                    kwargs_for_workers.pop('remote_group', None)
                    # create_workers takes extra_torch_kwargs explicitly, so strip it from the forwarded
                    # component kwargs to avoid a duplicate-keyword collision.
                    kwargs_for_workers.pop('extra_torch_kwargs', None)

                    _actors = RayHelper.create_workers(
                        cls,
                        remote_group,
                        execute,
                        instance_id=instance_id,
                        extra_torch_kwargs=infra._extra_torch_kwargs,
                        max_concurrency=concurrency,
                        creation_timeout=self._ray_get_timeout,
                        *args,
                        **kwargs_for_workers)
                    self._actors = _actors
                    self._actor_owner = _owner_identity()
                    self._max_concurrency = concurrency
                    if pool_class:
                        _register_pool(self, cls, remote_group, execute, instance_id, args, kwargs_for_workers,
                                       infra._extra_torch_kwargs, device_mesh_name)
                    if hasattr(cls, '__iter__'):
                        # wraps again, because ray uses cls method to call remote
                        cls.__iter__ = remote_function(dispatch=_dispatch, execute=_execute, collect='none')(__iter__)
                        cls.__next__ = remote_function(
                            dispatch=_dispatch, execute=_execute, collect=_collect, lazy_collect=_lazy)(__next__)
                    for arg in (list(args) + list(kwargs.values())):
                        # keeps the device_mesh in the handler
                        if isinstance(arg, DeviceMesh):
                            self.device_mesh = arg
                            break

                self.remote_group = remote_group
                self._instance_id = instance_id
            else:
                raise ValueError(f'Unsupported mode: {infra._mode}')

        def perf_on(self, *, profiler_spec=None):
            from twinkle.perf import Perf
            self._profiler = Perf.enable_process(profiler_spec=profiler_spec)

        def perf_step(self):
            from twinkle.perf import Perf
            Perf.step_process(getattr(self, '_profiler', None))

        def perf_dump(self):
            from twinkle.perf import Perf
            return Perf.dump_process(getattr(self, '_profiler', None))

        def perf_off(self):
            from twinkle.perf import Perf
            Perf.disable_process(getattr(self, '_profiler', None))
            self._profiler = None

        def _owned_workers(self):
            owner = _owner_identity()
            actors = list(self.__dict__.get('_actors', ())) if self.__dict__.get('_actor_owner') == owner else []
            if self.__dict__.get('_pool_actor_owner') == owner:
                actors.extend(actor for actor in self.__dict__.get('_owned_pool_actors', ()) if actor not in actors)
            return actors

        def _release_workers(self):
            from ._ray import RayHelper
            actors = _owned_workers(self)
            try:
                RayHelper.release_workers(actors)
            finally:
                for field, owner_field in (('_actors', '_actor_owner'), ('_owned_pool_actors', '_pool_actor_owner')):
                    if self.__dict__.get(owner_field) == _owner_identity():
                        self.__dict__[field][:] = [actor for actor in self.__dict__[field]
                                                  if actor in RayHelper._created_actors]

        def owned_lifecycle(method):
            def close(self, *args, **kwargs):
                """Run component teardown and release only replicas owned by this process."""
                if '_actors' not in self.__dict__:
                    return method(self, *args, **kwargs) if method is not None else None
                actors = _owned_workers(self)
                if not actors:
                    _release_workers(self)
                    return
                try:
                    if method is not None:
                        import copy
                        handle = copy.copy(self)
                        handle._actors = actors
                        handle._pool_key = None
                        result = method(handle, *args, **kwargs)
                        from ._ray.ray_helper import LazyCollect
                        if isinstance(result, (Future, LazyCollect)):
                            return result.result()
                        return result
                finally:
                    primary_error = sys.exc_info()[0]
                    try:
                        _release_workers(self)
                    except BaseException:
                        if primary_error is None:
                            raise
                        logger.exception('Worker cleanup failed while preserving the component teardown error.')

            close._twinkle_owned_close = True
            close._component_close = method
            return close

        if not hasattr(cls, 'perf_on'):
            cls.perf_on = remote_function(dispatch='all', execute=execute, collect='none',
                                          lazy_collect=False)(perf_on)
        if not hasattr(cls, 'perf_step'):
            cls.perf_step = remote_function(dispatch='all', execute=execute, collect='none')(perf_step)
        if not hasattr(cls, 'perf_dump'):
            cls.perf_dump = remote_function(dispatch='all', execute=execute, collect='none',
                                            lazy_collect=False)(perf_dump)
        if not hasattr(cls, 'perf_off'):
            cls.perf_off = remote_function(dispatch='all', execute=execute, collect='none',
                                           lazy_collect=False)(perf_off)
        for name, method in lifecycle_methods.items():
            if method is None and name != 'close':
                continue
            if method is not None and not hasattr(method, '_dispatch'):
                method = remote_function(dispatch='all', execute=execute, collect='none', lazy_collect=False)(method)
            setattr(cls, name, owned_lifecycle(method))

        def _twinkle_ready(self):
            return None

        cls._twinkle_ready = _twinkle_ready
        cls.__init__ = new_init
        return cls

    return decorator


def _register_pool(handle, cls, group, execute, instance_id, args, kwargs, torch_kwargs, device_mesh_name):
    """List a new pool's replicas ACTIVE in the registry and keep what ``add_workers`` needs to grow it."""
    from ._ray import RayHelper
    handle._worker_factory = (cls, tuple(args), dict(kwargs), torch_kwargs, device_mesh_name)
    handle._pool_key = f'{group}-{cls.__name__}-{instance_id}-{uuid.uuid4().hex[:8]}'
    placements = list(RayHelper.group_info(group)['placements'])
    slots = list(range(len(placements)))
    if execute == 'first':
        slots = slots[:1]
    elif execute == 'peer':
        slots = slots[Platform.get_peer_index(len(slots))]
    RayHelper.pool_put(handle._pool_key, {
        worker_id_of(actor): {'actor': actor, 'state': 'ACTIVE', 'group': group, 'slot': slot,
                              'ranks': tuple(placements[slot].get('ranks', ()))}
        for actor, slot in zip(handle._actors, slots)
    })


def _stash_value(value: Any) -> Any:
    """Lift ``value`` to an ObjectRef inside a remote worker; pass it through in the driver.

    A value that already is a reference (a payload this worker only forwards) stays as it is.
    """
    if not os.environ.get('WORKER_NAME'):
        return value
    import ray

    if isinstance(value, ray.ObjectRef):
        return value
    return ray.put(value)


def _stash_channel(value: Any) -> Any:
    """One declared channel of a stashed return value, lifted at its birthplace in the worker.

    A list channel is stashed element-wise: the caller's collect materialises the list itself
    (cheap) with every element still a reference it can index or forward, which is what keeps
    per-sample hand-off (``process[index]``) working without dragging the payloads along.
    Anything else is stashed whole.
    """
    if isinstance(value, list):
        return [_stash_value(item) for item in value]
    return _stash_value(value)


def _stash_channels(result: Any, channels: Union[bool, Sequence[Union[int, str]]]) -> Any:
    """Lift a remote return value's declared ``stash`` channels to ObjectRefs, in the worker.

    The outgoing half of partial collect. ``lazy_collect`` keeps the WHOLE return value as one
    reference, which is enough when the caller never opens the container; a mixed return (light
    control scalars plus heavy payloads) whose light half the caller must read needs per-channel
    refs instead -- ``ray.get`` resolves only top-level refs, so the caller's collect materialises
    the light half while the stashed channels ride through as references the next remote call
    dereferences at its worker.

    ``True`` lifts the whole return value as a single channel (a bare-list payload element-wise,
    an existing ref untouched, anything else whole); ``False`` never reaches here -- the
    decorators skip it. A tuple channel is addressed by position and must exist (arity is the
    method's contract); a dict channel is a key that MAY be absent (a branch-dependent return
    carries one signal key per mode) and is skipped when it is.
    """
    if channels is True:
        return _stash_channel(result)
    if isinstance(channels, bool):
        raise TypeError(f'stash {channels!r} is not a usable channel declaration')
    if isinstance(result, tuple):
        lifted = list(result)
        for channel in channels:
            if isinstance(channel, bool) or not isinstance(channel, int):
                # bool is an int subclass: True must never address tuple position 1.
                raise TypeError(f'stash channel {channel!r} does not address a tuple return value')
            lifted[channel] = _stash_channel(lifted[channel])
        return tuple(lifted)
    if isinstance(result, dict):
        for channel in channels:
            if channel not in result:
                continue
            result[channel] = _stash_channel(result[channel])
        return result
    raise TypeError(f'stash requires a tuple or dict return value, got {type(result).__name__}')


def _record_return_probe(func_name: str, call_id=None) -> None:
    """Append one record when a worker method finishes its whole return path, stash included.

    A probe inside the method body cannot tell "the body returned" from "the caller's reference
    became ready": the stash lifts channels to refs after the body is done, and the caller waits on
    the reference, not on the body. Recording here separates an actor that never finished returning
    from a finished return whose future the caller never saw resolve.
    """
    import os
    root = os.environ.get('TWINKLE_DISPATCH_PROBE')
    if not root:
        return
    import json
    import time
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'dispatch.{os.getpid()}.jsonl')
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps({'t': time.monotonic(), 'event': 'worker_returned',
                                 'func': func_name, 'call_id': call_id or uuid.uuid4().hex,
                                 'instance': os.environ.get('WORKER_NAME')}) + '\n')


def remote_function(dispatch: Union[Literal['slice', 'all', 'slice_dp'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'mean', 'first', 'last_pp_first',
                                           'tensor_dict', 'dp_flatten'], Callable] = 'none',
                    sync: bool = False,
                    lazy_collect: Optional[bool] = None,
                    timeout: Optional[float] = None,
                    enable_continous_work: bool = False,
                    sticky_key: Optional[Union[str, Callable]] = None,
                    stash: Optional[Union[bool, Sequence[Union[int, str]]]] = None,
                    offload_blocking: bool = False):
    """Patch each method called from remote(which class should be decorated with `remote_class`) with this decorator.

    Args:
        dispatch: How to dispatch the arguments.
            'slice': load balance
            'all': all processes do the same thing
            'slice_dp': Slice the input by data ranks in device_mesh
            Callable: A callable that handles the dispatching
        execute: How to execute
            'first': Only first worker
            'peer': Only peer workers
            'all': All processes
        collect: How to collect the results.
            'none': Return as-is
            'flatten': Return a flattened list
            'mean': Return the mean value of all processes
            'first': Return the first worker's result but executed in each process, usually
                works for scenarios of all-gather (e.g., get length).
            'tensor_dict': Merge per-worker output dicts (model forward outputs).
            'dp_flatten': Deduplicate per-worker lists by data rank, then flatten.
            Callable: A callable that handles the collection
        sync: If True, use synchronous execution (execute_all_sync) instead of async.
            Required for methods with NCCL collective operations (e.g., Megatron forward_backward).
        lazy_collect: Do lazy collect, this boolean value decides whether this function needs lazy collect. If setting to None, it will follow the global setting.
        timeout: Timeout in seconds for ray.get() when collecting results. The decorator's
            explicitly declared value takes priority; the instance attribute ``_ray_get_timeout``
            is the fallback for methods that declare none (``timeout if timeout is not None
            else instance``).
        enable_continous_work: Route each request to one replica instead of slicing the batch over
            all of them: a list argument is split request by request onto the least loaded replicas
            (results come back in the caller's order), a call without a list is one request. The
            replicas are the pool's ACTIVE members; a replica that is draining or died has the request
            resubmitted elsewhere. Without ``lazy_collect=True`` the call returns the results; with it,
            a ``concurrent.futures.Future`` -- in local mode too, where the call runs on the
            instance's thread pool (sized by ``max_concurrency``). Requires the class to be declared
            with ``max_concurrency`` above 1 to reach the worker's engine concurrently. A
            resubmission is at-least-once: a request whose replica drained after finishing it
            (or halfway through it) runs again elsewhere, so the method must tolerate being
            called twice on the same request.
        sticky_key: Pin calls sharing a key to one replica (session affinity): a ``param.field.path``
            string read off the bound arguments, or a callable taking ``(args, kwargs)``. The binding
            lives in the caller's process (LRU, idle expiry); a vanished replica is rebound.
        stash: Return-value channels that stay references through the caller's collect -- the
            standard mechanism for a mixed (light + heavy) multi-part return whose light half the
            caller must read. Tuple channels are addressed by position (must exist), dict channels
            by key (skipped when absent); a list channel is lifted element-wise, anything else
            whole. ``True`` lifts the whole return value as one such channel -- the declaration
            for a bare-list payload (e.g. a trajectory batch) that must ride element-wise. ``False``
            and ``None`` stash nothing. The caller's collect then materialises only the rest of the
            container, and the next remote call dereferences the stashed references at its worker.
            Local mode has no remote path, so the natural value passes through untouched. Declared
            on ``__iter__``, it applies to every batch ``__next__`` yields.
        offload_blocking: Run the worker body in the instance's thread pool instead of the actor's
            event loop, for a sync method that blocks for a long time (a weight receive waiting on
            a transport). Ray makes a class with any ``async def`` an asyncio actor and runs its
            sync methods ON the default concurrency-group loop (``async_compat.sync_to_async``); a
            blocking body there monopolizes that loop, so every other call in the same group --
            notably ``resume_generation`` in a weight sync's ``finally`` -- queues forever behind
            it (defect88). With this flag the caller routes to the method's awaitable companion
            (``_WORKER_ASYNC_PREFIX``), which offloads the body to ``_worker_executor`` and awaits
            it, so the loop stays free to dispatch while the body blocks in a thread. Only the ray
            remote path is rerouted: local mode calls the plain sync method unchanged.
    """ # noqa
    if sticky_key is not None and not enable_continous_work:
        raise ValueError('remote_function(sticky_key=...) requires enable_continous_work=True.')
    if stash is not None and stash is not False and stash is not True:
        if isinstance(stash, (int, str, bool)) or not isinstance(stash, (list, tuple)):
            raise TypeError(f'stash {stash!r} must be True/False/None or a sequence of int/str channels')
        for channel in stash:
            if isinstance(channel, bool) or not isinstance(channel, (int, str)):
                raise TypeError(f'stash channel {channel!r} must be an int position or a str key')
    continuous_lazy = enable_continous_work and lazy_collect is True

    def decorator(func: Callable[..., T1]) -> Callable[..., T1]:

        def _run_here(self, args, kwargs, dispatch_context):
            """The method body in this process: a Ray worker, or a component built without a remote group."""
            from ._ray import RayHelper
            if RayHelper.has_ref(args, kwargs):
                resolve_timeout = timeout if timeout is not None else getattr(self, '_ray_get_timeout', None)
                args, kwargs = RayHelper.do_get_and_collect(args, kwargs, timeout=resolve_timeout)
            if enable_continous_work and getattr(self, '_twinkle_draining', False):
                raise WorkerDraining(f'{func.__name__}: this replica is leaving its pool.')
            try:
                result = func(self, *args, **kwargs)
            except Exception as exc:
                if enable_continous_work and getattr(self, '_twinkle_draining', False):
                    raise WorkerDraining(f'{func.__name__}: this replica left its pool mid-request.') from exc
                raise
            if enable_continous_work and getattr(self, '_twinkle_draining', False):
                raise WorkerDraining(f'{func.__name__}: this replica left its pool mid-request.')
            if isinstance(result, Future):
                # A worker method may hand back a plain Future (an engine-side async submission);
                # waiting here makes the caller's reference come ready only when the work finishes.
                result = result.result()
            if stash is not None and stash is not False and func.__name__ != '__iter__':
                # Partial collect: lift the declared heavy channels to refs here, at the value's
                # birthplace, so the caller's collect materialises only the light half.
                # ``__iter__`` is exempt: it hands back the generator, not a batch, and the stash
                # declared on it is applied per yielded value by the ``__next__`` wrapper.
                result = _stash_channels(result, stash)
            _record_return_probe(func.__name__, dispatch_context)
            return result

        def _call_replicas(self, args, kwargs):
            """Caller side of a ray handle: pick replicas, send, and collect (or hand back the handle)."""
            from ._ray import RayHelper
            infra = _infra()
            _rgt = timeout if timeout is not None else getattr(self, '_ray_get_timeout', None)
            pool = getattr(type(self), '_twinkle_pool', False)
            if enable_continous_work:
                assert not sync, (f'{func.__name__}: enable_continous_work cannot be used with sync=True, '
                                  'which exists for collectives that must run in lock step.')
                call = _ContinuousCall(_pool_view(self), func.__name__, args, kwargs, _cw_batch_len(args, kwargs),
                                       _resolve_sticky_key(sticky_key, func, args, kwargs))
                future = call.start()
                if lazy_collect is True:
                    return future
                try:
                    return future.result(timeout=_rgt)
                except TimeoutError as exc:
                    # ``future.result`` raises ``TimeoutError`` for two different failures, and only one of
                    # them is a timeout of this call. The wait expiring leaves the future unfinished; the
                    # callee raising its own ``TimeoutError`` leaves it finished and carrying that
                    # exception. Rewrapping both destroyed the second one's only diagnosis: an ``Env.reset``
                    # that ran out of session slots says "no free session within 600s (8 in use)", which
                    # reached the caller as "reset: no result within Nones" -- a pool that is exhausted
                    # reading as a call that was slow, with the timeout that was never even set named as
                    # the cause. Re-raise the callee's exception untouched and wrap only a real wait.
                    #
                    # Every other path here waits through ``ray.get``, whose timeout is a
                    # ``ray.exceptions.GetTimeoutError``. A continuous-work call instead waits on a
                    # ``concurrent.futures.Future``, whose native timeout is a plain ``TimeoutError``,
                    # so a caller catching the infra's one timeout type would silently miss these.
                    # ``GetTimeoutError`` subclasses ``TimeoutError``, so narrowing the type here
                    # unifies the two paths without breaking anyone catching the builtin.
                    if future.done():
                        raise
                    import ray
                    raise ray.exceptions.GetTimeoutError(
                        f'{func.__name__}: no result within {_rgt}s') from exc

            execute_method = RayHelper.execute_all_async if not sync else functools.partial(
                RayHelper.execute_all_sync, timeout=_rgt)
            workers = _pool_view(self).actors() if pool else list(self._actors)
            workers = _get_workers(workers, execute)
            if not workers:
                return []
            # Replicas of a pool are independent engines: a data-parallel slice is a plain slice.
            _dispatch = 'slice' if pool and dispatch == 'slice_dp' else dispatch
            _mesh = None if pool else getattr(self, 'device_mesh', None)
            call_id = uuid.uuid4().hex
            _workers_and_args = [
                (worker, shard_args, dict(shard_kwargs, _dispatch_context=call_id))
                for worker, shard_args, shard_kwargs in _dispatch_args(workers, _dispatch, execute, _mesh, args, kwargs)]
            # offload_blocking: call the awaitable companion so the blocking body runs in a worker
            # thread instead of monopolizing the actor's default event loop (defect88).
            _method_name = (_WORKER_ASYNC_PREFIX + func.__name__) if offload_blocking else func.__name__
            result = execute_method(_method_name, _workers_and_args)
            # This is a result future, call it to get the actual result
            result_func = RayHelper.do_get_and_collect_func(_collect_func, collect, result, _mesh, timeout=_rgt)
            if func.__name__ == '__iter__':
                # The replica's __iter__ stores the iterator on itself and __next__ reads it back
                # there, so the two calls are ordered by state rather than by argument. Waiting here
                # is what makes that order hold: without it the first __next__ can reach the replica
                # before the iterator exists.
                import ray
                for _res in result:
                    ray.get(_res, timeout=_rgt)
                return self

            if func.__name__ == '__len__':
                # Get the first result and ignore the `lazy_collect`
                import ray
                return ray.get(result[0], timeout=_rgt)

            if func.__name__ == '__next__':
                import ray
                for _res in result:
                    # raise when any worker raises StopIteration
                    stop = ray.get(_res[1], timeout=_rgt)
                    if stop:
                        raise StopIteration()
                result_func._futures = [_res[0] for _res in result]

            _local_lazy_collect = infra._lazy_collect if lazy_collect is None else lazy_collect
            if _local_lazy_collect:
                return result_func
            return result_func()

        @functools.wraps(func)
        def _call_impl(self, *args, **kwargs) -> T1:
            infra = _infra()
            _ctx = f'{type(self).__name__}.{func.__name__}'
            # Only capture caller on driver side; worker frames are Ray internals
            _caller = _capture_caller() if hasattr(self, '_actors') else None
            if _caller:
                _ctx = f'{_ctx} <- {_caller}'
            try:
                dispatch_context = kwargs.pop('_dispatch_context', None)
                if infra._mode == 'local':
                    # The same entry the ray path runs: Future resolution, the drain check and
                    # stash all apply here too, or a component method handing back a plain engine
                    # Future would return it nested in local mode and materialised in ray.
                    # (stash is a no-op outside a worker -- ``_stash_value`` passes through.)
                    if continuous_lazy:
                        return _local_executor(self).submit(_run_here, self, args, kwargs, None)
                    return _run_here(self, args, kwargs, None)
                elif infra._mode == 'ray':
                    check_unsafe(*args, **kwargs)
                    if hasattr(self, '_actors'):
                        return _call_replicas(self, args, kwargs)
                    if continuous_lazy:
                        # A component built in this process (no remote group) keeps the Future contract.
                        return _local_executor(self).submit(_run_here, self, args, kwargs, dispatch_context)
                    return _run_here(self, args, kwargs, dispatch_context)
                else:
                    raise NotImplementedError(f'Unsupported mode {infra._mode}')
            except StopIteration:
                raise
            except Exception as _e:
                _tag_exc(_e, _caller)
                notify_exception(infra._notifier, _ctx, _e, infra._name)
                raise

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T1:
            if hasattr(self, '_actors'):
                from twinkle.perf import default_recorder
                with default_recorder.span(func.__qualname__):
                    return _call_impl(self, *args, **kwargs)
            return _call_impl(self, *args, **kwargs)

        # Ray validates an actor call against ``inspect.signature(inspect.unwrap(method))``: it
        # unwraps first (``ray/actor.py``, ``method = inspect.unwrap(method)``) and only then reads
        # the signature, so a pin on this wrapper never reaches the validator -- ``functools.wraps``
        # set ``__wrapped__`` to the undecorated body and unwrap descends straight past the pin to
        # it. The pin therefore belongs on the body, which is the object Ray ends up holding.
        # ``__wrapped__`` stays in place: reaching the undecorated body through it is how the test
        # suite drives these methods at all.
        #
        # The declared parameter is metadata only. ``_call_impl`` and ``_worker_entry`` both pop
        # ``_dispatch_context`` before the body runs, so the body never receives it; declaring it is
        # what lets the ref-carrying dispatch path put it on the wire. The body's own parameters are
        # kept in the pin rather than replaced with a var-args form, so ``inspect.signature`` stays
        # honest for anyone reading the component.
        # Ray reads ``inspect.signature(inspect.unwrap(method))`` and unwrap descends the WHOLE
        # ``__wrapped__`` chain to the innermost body. An intermediate decorator (``@timed`` under
        # ``@remote_function``, e.g. CheckpointEngineMixin.send_weights_to_vllm) inserts a
        # ``__wrapped__`` layer: a pin on ``func`` (the timed wrapper) is bypassed because unwrap
        # lands on the undecorated body, whose signature lacks ``_dispatch_context`` and Ray refuses
        # the dispatch kwarg. Pin the innermost body that unwrap actually lands on.
        _body = inspect.unwrap(func)
        _params = list(inspect.signature(_body).parameters.values())
        if not any(_p.name == '_dispatch_context' for _p in _params):
            _context = inspect.Parameter('_dispatch_context', inspect.Parameter.KEYWORD_ONLY,
                                         default=None)
            # A keyword-only parameter may not follow ``**kwargs``, so it goes just before one.
            _at = next((_i for _i, _p in enumerate(_params)
                        if _p.kind is inspect.Parameter.VAR_KEYWORD), len(_params))
            _params.insert(_at, _context)
        _body.__signature__ = inspect.Signature(parameters=_params)

        # ``self`` must be declared here: Ray extracts an actor method's signature with
        # ``extract_signature(func, ignore_first=True)``, which drops the first parameter on the
        # assumption that it is ``self``. Without it the pin loses ``args`` and every positional
        # argument is refused with "too many positional arguments".
        wrapper.__signature__ = inspect.Signature(parameters=[
            inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter('args', inspect.Parameter.VAR_POSITIONAL),
            inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD),
        ])

        def _worker_entry(self, *args, **kwargs):
            # What the awaitable companion runs on the replica: always the body, never a Future.
            dispatch_context = kwargs.pop('_dispatch_context', None)
            try:
                return _run_here(self, args, kwargs, dispatch_context)
            except WorkerDraining:
                raise
            except Exception as _e:
                infra = _infra()
                notify_exception(infra._notifier, f'{type(self).__name__}.{func.__name__}', _e, infra._name)
                raise

        wrapper._execute = execute
        wrapper._collect = collect
        wrapper._dispatch = dispatch
        wrapper._lazy_collect = lazy_collect
        wrapper._sync = sync
        wrapper._enable_continous_work = enable_continous_work
        wrapper._sticky_key = sticky_key
        wrapper._stash = stash
        if enable_continous_work or offload_blocking:
            # Attached to the class by remote_class under _WORKER_ASYNC_PREFIX. continuous_work
            # routes requests to it replica by replica; offload_blocking routes a dispatch='all'
            # call to it so the blocking body runs in a worker thread, not the actor's event loop.
            wrapper._worker_async_companion = _make_worker_async_companion(func, _worker_entry)
        return wrapper

    return decorator
