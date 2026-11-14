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


def _collect_func(method: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp', 'last_pp_first',
                                        'tensor_dict', 'dp_flatten'], Callable],
                  result: List[Any],
                  device_mesh: DeviceMesh = None):
    """Collect results

    Args:
        method:
            none: Return as is.
            flatten: Flat the nested results.
            mean: Average the results.
            sum: Sum the results.
            first: Only return the first result.
            last_pp: Only return the results of the last pp rank.
            tensor_dict: Merge per-worker output dicts, stacking tensors (deduplicated by data rank).
            dp_flatten: Deduplicate per-worker lists by data rank, then flatten in input order.
        result: The results returned by workers.
        device_mesh: The device_mesh, needed by `last_pp`
    Returns:
        The collected results.
    """
    if not result:
        return result

    if isinstance(result[0], tuple):
        output = []
        # if each result of a worker is a tuple
        for i in range(len(result[0])):
            # handle each element in a tuple
            _single_result = [r[i] for r in result]
            output.append(_collect_func(method, _single_result, device_mesh=device_mesh))
        return output
    if method == 'none':
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
    elif method == 'sum':
        return np.sum(result)
    elif method == 'first':
        return result[0]
    elif method == 'last_pp':
        assert device_mesh is not None
        return [r for i, r in enumerate(result) if i in device_mesh.get_pp_last_ranks()]
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
    elif isinstance(method, Callable):
        # Callable
        return method(result, device_mesh=device_mesh)
    else:
        raise ValueError(f'Unsupported collect method: {method}')


def _dispatch_args(workers, dispatch, execute, device_mesh: Optional[DeviceMesh], args, kwargs):
    if execute == 'first':
        return [(workers[0], args, kwargs)]
    elif dispatch == 'all':
        return [(worker, args, kwargs) for worker in workers]
    elif dispatch == 'slice':
        # split arg to workers evenly
        result = []
        length = len(workers)

        def dispatch_func(arg, n):
            if isinstance(arg, list):
                # only list
                _args = []
                k, m = divmod(len(arg), n)
                for i in range(n):
                    _args.append(arg[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
                return _args
            else:
                return [arg] * n

        args = [dispatch_func(arg, length) for arg in args]
        kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}
        for i in range(length):
            sliced_args = tuple(arg[i] for arg in args)
            sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
            result.append((workers[i], sliced_args, sliced_kwargs))

        return result
    elif dispatch == 'slice_dp':
        assert device_mesh is not None
        # split by dp. each worker in one ep will receive the same argument
        result = []
        # if device_mesh is not None:
        # TODO this may occurs error when remote calls remote
        # Comment this because remote_class supports `first``
        # assert device_mesh.world_size == len(workers)
        length = len(workers)
        # Map actor index to global_rank: with gpus_per_worker>1, consecutive
        # global ranks belong to the same actor (TP peers).
        _mesh_world = device_mesh.world_size if device_mesh is not None else length
        _rank_stride = max(1, _mesh_world // length)
        if _rank_stride * length != _mesh_world:
            raise ValueError(
                f'slice_dp dispatch needs the {length} workers to tile the device mesh exactly '
                f'(world_size {_mesh_world}): uneven tiling leaves data ranks unserved, and their '
                'outputs silently drop out of the collect step.')

        def dispatch_func(arg, n):
            import torch
            if isinstance(arg, list) or isinstance(arg, torch.Tensor):
                _args = []
                for i in range(n):
                    _args.append(arg[device_mesh.get_slice(
                        len(arg), device_mesh.get_data_rank_from_global_rank(i * _rank_stride))])
                return _args
            elif isinstance(arg, dict):
                _args = [{} for _ in range(n)]
                for key in arg.keys():
                    value = arg[key]
                    for i, v in enumerate(dispatch_func(value, n)):
                        _args[i][key] = v
                return _args
            else:
                return [arg] * n

        args = [dispatch_func(arg, length) for arg in args]
        kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}

        for i in range(length):
            sliced_args = tuple(arg[i] for arg in args)
            sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
            result.append((workers[i], sliced_args, sliced_kwargs))

        # Raise early if some ranks got data and others didn't (causes hangs).
        def _check_uniform(slices):
            lens = [len(s) if s is not None and isinstance(s, (list, tuple)) else 0 for s in slices]
            return not lens or all(length > 0 for length in lens) or all(length == 0 for length in lens)

        for arg in args:
            if not _check_uniform(arg):
                raise ValueError(f'Batch too small for {length} workers, some ranks have no data. '
                                 f'Please increase batch size to at least {length}.')
        for v in kwargs.values():
            if not _check_uniform(v):
                raise ValueError(f'Batch too small for {length} workers, some ranks have no data. '
                                 f'Please increase batch size to at least {length}.')

        return result
    elif isinstance(dispatch, Callable):
        length = len(workers)
        result = []
        for i in range(length):
            sliced_args, sliced_kwargs = dispatch(length, i, args, kwargs, device_mesh=device_mesh)
            result.append((workers[i], sliced_args, sliced_kwargs))
        return result
    else:
        raise ValueError(f'Unsupported dispatch method: {dispatch}')


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


def _prepare_lazy_collect(args, kwargs):
    # if a worker received an actor handle,
    # lazy collect should be false to prevent any outer function receives an object ref
    if not os.environ.get('WORKER_NAME'):
        # If this is a driver
        return args, kwargs
    else:
        # If this is a worker, collect now
        for arg in list(args) + list(kwargs.values()):
            if hasattr(arg, '_actors'):
                # This arg is an handler, and this is a worker env, so do not do lazy collect
                arg._lazy_collect = False
        return args, kwargs


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

        def new_init(self, *args, **kwargs):
            _caller = _capture_caller()
            _ctx = f'{cls.__name__}.__init__'
            if _caller:
                _ctx = f'{_ctx} <- {_caller}'
            try:
                _infra()._maybe_load_worker_notifier()
                _new_init_body(self, _caller, *args, **kwargs)
            except Exception as _e:
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
            if infra._mode == 'local':
                # remote_group names the ray device group (the ray branch below
                # consumes then pops it). Local mode never dispatches, so it is
                # dropped here too: a component constructor must not receive it
                # -- plain HF classes reject the unknown keyword inside
                # from_pretrained kwargs.
                kwargs.pop('remote_group', None)
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
                    if self.__class__.__name__ == 'DataLoader' and 'min_batch_size' not in kwargs:
                        # TODO An ugly special setting for dataloader to set the min batch size
                        kwargs['min_batch_size'] = device_mesh.data_world_size
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

                    if self.__class__.__name__ == 'DataLoader' and 'min_batch_size' not in kwargs:
                        # TODO An ugly special setting for dataloader to set the min batch size
                        kwargs['min_batch_size'] = kwargs['batch_size']

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
                        device_group = [dg for dg in infra._device_group if dg.name == remote_group][0]
                        device_group._device_mesh[self.__class__.__name__] = device_mesh

                iter_stash = getattr(getattr(cls, '__iter__', None), '_stash', None)

                # This will solve the iterator cannot be passed through ray.
                def __iter__(_self):
                    if os.environ.get('WORKER_NAME'):
                        # This is a worker, iter keeps in the class, pass nothing to driver
                        _iter = _self.__iter_origin__()
                        assert _iter is not _self
                        _self._iter = _iter
                    else:
                        # This is executed in driver
                        return _self.__iter_origin__()

                def __next__(_self):
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
                        # if any handler is passed to other component, lazy collect should be false
                        # for example, dataset pass to the dataloader
                    args, kwargs = _prepare_lazy_collect(args, kwargs)
                    kwargs.pop('remote_group', None)  # component does not need this
                    self._max_concurrency = concurrency
                    init_method(self, *args, **kwargs)
                else:
                    if hasattr(cls, '__iter__'):
                        _dispatch = self.__iter__._dispatch
                        _execute = self.__iter__._execute
                        _collect = self.__iter__._collect
                        _lazy = self.__iter__._lazy_collect
                        import ray
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
                        *args,
                        **kwargs_for_workers)
                    self._actors = _actors
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
            strategy = getattr(self, 'strategy', None)
            model = getattr(self, 'model', None)
            target = strategy.unwrap_model(model) if (strategy is not None and model is not None) else None
            self._profiler = Perf.enable_process(model=target, profiler_spec=profiler_spec)

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


def remote_function(dispatch: Union[Literal['slice', 'all', 'slice_dp'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp', 'last_pp_first',
                                           'tensor_dict', 'dp_flatten'], Callable] = 'none',
                    sync: bool = False,
                    lazy_collect: Optional[bool] = None,
                    timeout: Optional[float] = None,
                    enable_continous_work: bool = False,
                    sticky_key: Optional[Union[str, Callable]] = None,
                    stash: Optional[Union[bool, Sequence[Union[int, str]]]] = None):
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
            'sum': Return the sum value of all processes
            'first': Return the first worker's result but executed in each process, usually
                works for scenarios of all-gather (e.g., get length).
            'last_pp': Return the last pp's result.
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
            with ``max_concurrency`` above 1 to reach the worker's engine concurrently.
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
            device_mesh = getattr(self, 'device_mesh', None)
            if RayHelper.has_ref(args, kwargs):
                args, kwargs = RayHelper.do_get_and_collect(args, kwargs)
                if dispatch_context is None:
                    world_size = Platform.get_world_size()
                    rank = Platform.get_rank()
                    workers = _get_workers([None] * world_size, execute)
                    local_dispatch = dispatch
                    local_mesh = device_mesh
                    # 'first' and 'peer' hand this process a one-entry subset (its own slot);
                    # indexing it by the GLOBAL rank reads past it. 'all' keeps every rank, so
                    # the rank itself is the entry index.
                    entry = rank if execute == 'all' else 0
                else:
                    # The caller sent the refs to every worker unsliced; slice here, now that the
                    # structure is visible.
                    world_size, rank, pool = dispatch_context
                    workers = [None] * world_size
                    local_dispatch = 'slice' if pool and dispatch == 'slice_dp' else dispatch
                    local_mesh = None if pool else device_mesh
                    # The caller numbered the entries of the worker list it dispatched to, and
                    # this rank's index is its position in that same list.
                    entry = rank
                _, args, kwargs = _dispatch_args(workers, local_dispatch, execute, local_mesh, args, kwargs)[entry]
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
                                       _resolve_sticky_key(sticky_key, func, args, kwargs),
                                       RayHelper.has_ref(args, kwargs))
                future = call.start()
                return future if lazy_collect is True else future.result(timeout=_rgt)

            execute_method = RayHelper.execute_all_async if not sync else functools.partial(
                RayHelper.execute_all_sync, timeout=_rgt)
            workers = _pool_view(self).actors() if pool else list(self._actors)
            workers = _get_workers(workers, execute)
            if not workers:
                return []
            # Replicas of a pool are independent engines: a data-parallel slice is a plain slice.
            _dispatch = 'slice' if pool and dispatch == 'slice_dp' else dispatch
            _mesh = None if pool else getattr(self, 'device_mesh', None)
            if RayHelper.has_ref(args, kwargs):
                # If has any object-ref, dispatch in worker, because we don't know the structure in the ref.
                # for example, dataloader returns any data list.
                _workers_and_args = [(worker, args, dict(kwargs, _dispatch_context=(len(workers), index, pool)))
                                     for index, worker in enumerate(workers)]
            else:
                _workers_and_args = _dispatch_args(workers, _dispatch, execute, _mesh, args, kwargs)
            result = execute_method(func.__name__, _workers_and_args)
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

            _local_lazy_collect = infra._lazy_collect
            if lazy_collect is not None:
                # Maybe this function returns a small object
                _local_lazy_collect = lazy_collect
            if hasattr(self, '_lazy_collect'):
                # _lazy_collect in class has the highest priority
                # This is the unique case that an object ref contains another
                # And this is user independent, only decided by the code.
                _local_lazy_collect = self._lazy_collect
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
                    if continuous_lazy:
                        return _local_executor(self).submit(func, self, *args, **kwargs)
                    return func(self, *args, **kwargs)
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

        # Same hazard as the continuous-work companion: __wrapped__ makes inspect.signature (and so
        # Ray's argument validation) report the original method's parameters, so the
        # _dispatch_context that _call_replicas adds for ref-carrying calls is rejected as an
        # unexpected keyword. This wrapper takes *args/**kwargs by design.
        del wrapper.__wrapped__

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
        if enable_continous_work:
            # Attached to the class by remote_class, and called instead of this
            # method when the caller routes requests replica by replica.
            wrapper._worker_async_companion = _make_worker_async_companion(func, _worker_entry)
        return wrapper

    return decorator
