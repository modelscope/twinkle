# Copyright (c) ModelScope Contributors. All rights reserved.
import functools
import inspect
import os
import sys
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar, Union

import numpy as np
from twinkle.notifier import notify_exception
from twinkle.utils import DeviceGroup, DeviceMesh, Platform, check_unsafe, get_logger

from .continuous_work import _WORKER_ASYNC_PREFIX, _cw_barrier, _cw_batch_len, _cw_register, _cw_unregister, \
    _get_workers, _make_worker_async_companion, _resolve_sticky_key, _run_continous_work, _select_workers, \
    _sticky_bind, _sticky_lookup, _validate_cw_session_ids, get_worker_id

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


class _SideEffectLazyCollect:
    """The lazy-collect handle ``remote_function`` returns: LazyCollect's FULL access protocol plus the
    driver-side hooks that must fire at the real collect.

    The lazy path used to wrap the ``LazyCollect`` in a ``functools.wraps`` function and copy only
    ``_futures`` / ``ready`` / ``_is_lazy_collect`` onto it. But ``.get`` / ``.result`` are instance
    methods and ``[]`` / ``in`` / ``iter`` / ``len`` resolve on the TYPE -- a bare function carries none
    of them -- so a consumer that reads a forwarded output directly (``forward_only(...).get('logps')``,
    ``out['logits']``: the pattern the recipes and infer use) broke with ``'function' object has no
    attribute 'get'``. This is a real class exposing every protocol entry point; all of them funnel
    through one :meth:`_get_result` that runs the hooks EXACTLY once, at the real collect, so a cached
    re-read cannot double-fire the deferred configuration-lock release or the continuous-work unregister.

    Composition (not subclassing) keeps this module free of a module-level ``LazyCollect`` import, which
    would cycle through ``infra/__init__`` (it imports dispatch). ``__getattr__`` forwards anything not
    modelled here (``device_mesh``, ``_method``, ...) to the wrapped handle, so the proxy is transparent
    to code that introspects the underlying ``LazyCollect``.
    """

    _is_lazy_collect = True

    def __init__(self, base, *, on_error: Optional[Callable] = None, on_collect: Optional[Callable] = None):
        self._base = base
        #: Snapshot: the ``__next__`` path mutates ``base._futures`` BEFORE this proxy is built, and
        #: resolvers read ``_futures`` off the returned handle (the old wrapper copied it too).
        self._futures = getattr(base, '_futures', None)
        self._on_error = on_error
        self._on_collect = on_collect
        self._succeeded = False
        self._hook_done = False
        self._value = None

    def _get_result(self):
        if self._succeeded:
            return self._value
        try:
            self._value = self._base()
            self._succeeded = True
            return self._value
        except Exception as _e:  # noqa
            if self._on_error is not None:
                self._on_error(_e)
            raise
        finally:
            if self._on_collect is not None and not self._hook_done:
                self._hook_done = True
                self._on_collect()

    def __call__(self):
        return self._get_result()

    def result(self, timeout=None):
        if timeout is not None and not self._succeeded:
            self._base._timeout = timeout
        return self._get_result()

    def exception(self, timeout=None):
        try:
            self.result(timeout)
            return None
        except StopIteration:
            raise
        except Exception as exc:  # noqa
            return exc

    def done(self) -> bool:
        return self.ready()

    def ready(self) -> bool:
        return self._base.ready()

    def __iter__(self):
        return iter(self._get_result())

    def __len__(self):
        return len(self._get_result())

    def __getitem__(self, key):
        return self._get_result()[key]

    def get(self, key, default=None):
        return self._get_result().get(key, default)

    def __contains__(self, key):
        return key in self._get_result()

    def __getattr__(self, name):
        # Only reached for attrs not modelled above; ``_base`` is set first in __init__, so no recursion.
        return getattr(self._base, name)


def _collect_func(method: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable],
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
    elif method in ('avg', 'mean'):
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
            concurrently instead of stopping at the actor boundary.
    """

    def decorator(cls):
        # Mount the worker-pool elasticity contract (add_workers/configure_workers/...): the code
        # lives in infra so the component class stays pure business, yet callers get plain method
        # calls. Imported here rather than at module top: worker_pool imports this module's
        # remote_function, and a load-time cycle would leave it half-initialised.
        from .worker_pool import mount_worker_pool
        mount_worker_pool(cls)
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

        @functools.wraps(init_method)
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

        def _new_init_body(self, _caller, *args, **kwargs):
            infra = _infra()
            if infra._mode == 'local':
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
                        return next(_self._iter), False
                    except StopIteration:
                        return [], True

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
                    init_method(self, *args, **kwargs)
                else:
                    if hasattr(cls, '__iter__'):
                        _dispatch = self.__iter__._dispatch
                        _execute = self.__iter__._execute
                        _collect = self.__iter__._collect

                    if hasattr(cls, '__iter__'):
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
                        max_concurrency=max_concurrency,
                        *args,
                        **kwargs_for_workers)
                    self._actors = _actors
                    # Recorded unconditionally (an actor-native elasticity input: re-running this
                    # factory on a pre-reserved slot grows the replica set, see the worker-pool mount);
                    # classes that need richer bookkeeping layer their own _save_worker_factory on top.
                    self._worker_factory = (cls, tuple(args), dict(kwargs_for_workers), infra._extra_torch_kwargs)
                    factory_recorder = getattr(self, '_save_worker_factory', None)
                    if factory_recorder is not None:
                        factory_recorder(cls, args, kwargs_for_workers, infra._extra_torch_kwargs)
                    # Remembered so remote_function knows this class's workers run
                    # methods side by side, and that it must therefore track what is
                    # in flight. Without concurrency Ray orders calls per actor and
                    # the tracking would be dead weight.
                    self._max_concurrency = max_concurrency
                    if hasattr(cls, '__iter__'):
                        # wraps again, because ray uses cls method to call remote
                        cls.__iter__ = remote_function(dispatch=_dispatch, execute=_execute, collect='none')(__iter__)
                        cls.__next__ = remote_function(dispatch=_dispatch, execute=_execute, collect=_collect)(__next__)
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


def remote_map(func: Callable, args_list: List[tuple]) -> List[Any]:
    """Fan a picklable function out over short-lived CPU tasks, returning results in input order.

    Unlike remote_class/remote_function (which drive the resident training worker pool bound to
    device groups), this serves stateless embarrassingly-parallel jobs: each tuple in ``args_list``
    runs ``func(*tuple)`` on its own Ray CPU task. Starts a local head if no cluster is running.
    """
    if not args_list:
        return []
    import ray

    if not ray.is_initialized():
        from twinkle.server.launcher import build_ray_runtime_env
        ray.init(runtime_env=build_ray_runtime_env(), ignore_reinit_error=True)
    worker = ray.remote(func)
    return ray.get([worker.remote(*args) for args in args_list])


def node_ip() -> str:
    """This process's node IP as Ray sees it, brackets stripped."""
    import ray

    return ray.util.get_node_ip_address().strip('[]')


def alive_node_ids() -> Dict[str, str]:
    """Map of alive node IP -> Ray NodeID, for pinning an actor next to a device."""
    import ray

    return {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes() if node['Alive']}


def spawn_sidecar(cls, *, node_id: str, env_vars: Dict[str, str], args: tuple = ()) -> Any:
    """One zero-resource actor pinned to ``node_id`` -- a sidecar holding a device context.

    Unlike remote_class (the resident, device-group-bound worker pool), a sidecar is a
    helper placed next to one specific device (``env_vars`` e.g. CUDA_VISIBLE_DEVICES) whose
    handle the caller drives with plain ``handle.method.remote()`` calls. Lifecycle is the
    caller's: terminate with :func:`kill_actors`.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    template = ray.remote(num_cpus=0, num_gpus=0)(cls)
    return template.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        runtime_env={'env_vars': dict(env_vars)},
    ).remote(*args)


def resolve_all(handles: List[Any], timeout: Optional[float] = None) -> List[Any]:
    """Block until every actor-method handle resolves (a bare ``get`` over a list)."""
    import ray

    return ray.get(handles, timeout=timeout)


def kill_actors(handles: List[Any]) -> None:
    """Terminate sidecar actors without restart; their lifecycle is the caller's."""
    import ray

    for handle in handles:
        ray.kill(handle, no_restart=True)


def cluster_resources() -> Dict[str, float]:
    """The running cluster's resource labels (empty when no Ray cluster is up)."""
    import ray

    return dict(ray.cluster_resources()) if ray.is_initialized() else {}


def stash(value: Any) -> Any:
    """Lift ``value`` to an ObjectRef inside a remote worker; pass it through in the driver.

    The return-path mirror of an actor argument: a payload a worker produces for another
    remote consumer crosses the boundary cheaper as a ref, while a local worker shares the
    caller's process and the value itself is the cheaper hand-off.
    """
    if not os.environ.get('WORKER_NAME'):
        return value
    import ray

    return ray.put(value)


def _stash_channel(value: Any) -> Any:
    """One declared channel of a stashed return value, lifted at its birthplace in the worker.

    A list channel is stashed element-wise: the caller's collect materialises the list itself
    (cheap) with every element still a reference it can index or forward, which is what keeps
    per-sample hand-off (``process[index]``) working without dragging the payloads along.
    Anything else is stashed whole.
    """
    if isinstance(value, list):
        return [stash(item) for item in value]
    return stash(value)


def _stash_channels(result: Any, channels: Sequence[Union[int, str]]) -> Any:
    """Lift a remote return value's declared ``stash`` channels to ObjectRefs, in the worker.

    The outgoing half of partial collect. ``lazy_collect`` keeps the WHOLE return value as one
    reference, which is enough when the caller never opens the container; a mixed return (light
    control scalars plus heavy payloads) whose light half the caller must read needs per-channel
    refs instead -- ``ray.get`` resolves only top-level refs, so the caller's collect materialises
    the light half while the stashed channels ride through as references the next remote call
    dereferences at its worker.

    A tuple channel is addressed by position and must exist (arity is the method's contract); a
    dict channel is a key that MAY be absent (a branch-dependent return carries one signal key
    per mode) and is skipped when it is.
    """
    if isinstance(result, tuple):
        lifted = list(result)
        for channel in channels:
            if not isinstance(channel, int):
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


def remote_function(dispatch: Union[Literal['slice', 'all', 'slice_dp', 'last_pp_first'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable] = 'none',
                    sync: bool = False,
                    lazy_collect: Optional[bool] = None,
                    timeout: Optional[float] = None,
                    enable_continous_work: bool = False,
                    sticky: bool = False,
                    sticky_key: Optional[Union[str, Callable]] = None,
                    exclusive: bool = False,
                    control: bool = False,
                    replay: bool = False,
                    worker_selector: Optional[str] = None,
                    stash: Optional[Sequence[Union[int, str]]] = None):
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
            'first': Return the first worker's result but executed in each process, usually works for scenarios of all-gather.
            'mean'/'sum': Avg or sum the results.
            'first': Return the first worker's result, for example, get length
            'last_pp': Return the last pp's result.
            Callable: A callable that handles the collection
        sync: If True, use synchronous execution (execute_all_sync) instead of async.
            Required for methods with NCCL collective operations (e.g., Megatron forward_backward).
        lazy_collect: Do lazy collect, this boolean value decides whether this function needs lazy collect. If setting to None, it will follow the global setting.
        timeout: Timeout in seconds for ray.get() when collecting results. The decorator's
            explicitly declared value takes priority; the instance attribute ``_ray_get_timeout``
            is the fallback for methods that declare none (``timeout if timeout is not None
            else instance``).
        enable_continous_work: Route each request to the least busy worker instead
            of slicing the batch over all of them, and return the results in the
            caller's order. This is what lets a batch smaller than the worker
            count through: ``slice_dp`` would hand some ranks nothing and raise,
            which is why callers pad a single request up to the worker count and
            throw the duplicate generations away. Requires the class to be
            declared with ``max_concurrency`` above 1, otherwise the requests
            queue at the actor and run one at a time instead of reaching the
            worker's engine together. Only for methods that take a list of
            independent requests and return one result each, and whose workers
            need no collective between them -- data-parallel sampling, not a
            method with an all-reduce in it. While one such method has requests in
            flight, calling any other method on the same handle waits for them.
        sticky: Pin each call to the worker that first served its key, so later calls
            with the same key land on the same worker (request ownership / session
            affinity). Requires ``sticky_key``.
        sticky_key: The pinning key: a field-path string (read off the bound call's
            arguments, no eval) or a callable taking the bound ``args``/``kwargs``.
        exclusive: The method mutates shared worker state, so it takes the handle's
            configuration lock: in-flight calls drain first, new ordinary calls
            block until it finishes, and its own result is collected eagerly.
        control: The method is a control-plane call (pause/stop/status): it stays
            reachable while exclusive calls drain, and reuses a sticky request's
            pinned worker instead of re-routing.
        replay: Record the call so newly added workers replay it on mount
            (configuration replay: template/adapter/patch setup).
        worker_selector: optional instance-method name returning the stable actor-ID subset for this
            call; the internal ``_worker_ids`` parameter, when passed explicitly, takes precedence.
        stash: Return-value channels that stay references through the caller's collect -- the
            standard mechanism for a mixed (light + heavy) multi-part return whose light half the
            caller must read. Tuple channels are addressed by position (must exist), dict channels
            by key (skipped when absent); a list channel is lifted element-wise, anything else
            whole. The caller's collect then materialises only the rest of the container, and the
            next remote call dereferences the stashed references at its worker. Local mode has no
            remote path, so the natural value passes through untouched.

    Wrapped continuous-work calls also accept ``_session_ids``, a list or tuple of
    non-empty session strings (or ``None`` for ordinary load balancing), aligned
    with the materialized list batch. This metadata stays on the driver. Bindings
    are local to the driver and method, survive completed calls, and expire after
    30 minutes idle. A bound worker outside the current eligible subset causes
    rebinding. LazyCollect inputs and ``sync=True`` do not support this option.
    """ # noqa
    if sticky and sticky_key is None:
        raise ValueError('remote_function(sticky=True) requires sticky_key (a field-path string or a callable).')


    def decorator(func: Callable[..., T1]) -> Callable[..., T1]:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T1:
            infra = _infra()
            _ctx = f'{type(self).__name__}.{func.__name__}'
            # Only capture caller on driver side; worker frames are Ray internals
            _caller = _capture_caller() if hasattr(self, '_actors') else None
            if _caller:
                _ctx = f'{_ctx} <- {_caller}'
            _worker_reservation = None
            _worker_dispatching = False
            _configuration_lock = None
            # Set when an exclusive + explicit-lazy_collect call hands back a future whose collection
            # (not this method's return) releases _configuration_lock -- so the finally below skips it.
            _lock_release_deferred = False
            _submitted_calls = []
            _call_completed = False

            def _finish_dispatch():
                nonlocal _worker_dispatching
                if _worker_dispatching:
                    _worker_dispatching = False
                    self._end_worker_dispatch()

            try:
                device_mesh = getattr(self, 'device_mesh', None)
                worker_ids = kwargs.pop('_worker_ids', None)
                dispatch_context = kwargs.pop('_dispatch_context', None)
                session_ids = kwargs.pop('_session_ids', None)
                if session_ids is not None:
                    if not enable_continous_work or sync:
                        raise ValueError('_session_ids requires enable_continous_work=True and sync=False.')
                    session_ids = _validate_cw_session_ids(session_ids, _cw_batch_len(args, kwargs))
                # Sticky ownership: resolve the declared key once per call, on the driver only (the
                # ownership table lives on the driver-side handler). A ``sticky`` call with an
                # existing owner is pinned to that worker (request ownership); a ``control`` call
                # reuses the same binding so pause/stop reach the worker actually running the
                # request, and fails loudly when no such request is in flight. First-served calls
                # bind their owner at submit time (see ``_track_submission``).
                _sticky_key_value = (_resolve_sticky_key(sticky_key, func, args, kwargs)
                                     if hasattr(self, '_actors') and (sticky or control) and worker_ids is None
                                     else None)
                if _sticky_key_value is not None:
                    _owner = _sticky_lookup(self, _sticky_key_value)
                    if _owner is not None:
                        worker_ids = (_owner, )
                    elif control and hasattr(self, '_actors'):
                        # Ray driver only: the ownership table exists only there (bindings are made
                        # by the driver-side submission tracking). In local mode there is no table
                        # and no routing -- the call runs once on the local object, so a missing
                        # owner is not an error.
                        raise ValueError(f'{func.__name__}: no in-flight request for sticky key '
                                         f'{_sticky_key_value!r}')
                if infra._mode == 'local':
                    return func(self, *args, **kwargs)
                elif infra._mode == 'ray':
                    check_unsafe(*args, **kwargs)
                    if not hasattr(self, '_actors'):
                        # This is the worker
                        from ._ray import RayHelper
                        if RayHelper.has_ref(args, kwargs):
                            # In this case, driver dispatch is all, redispatch here
                            args, kwargs = RayHelper.do_get_and_collect(args, kwargs)
                            if dispatch_context is None:
                                world_size = Platform.get_world_size()
                                rank = Platform.get_rank()
                                workers = _get_workers([None] * world_size, execute)
                                local_dispatch = dispatch
                                local_mesh = device_mesh
                            else:
                                world_size, rank, subset = dispatch_context
                                workers = [None] * world_size
                                local_dispatch = (getattr(self, '_dispatch_sampling_inputs', 'slice')
                                                  if subset and dispatch == 'slice_dp' else dispatch)
                                local_mesh = None if subset else device_mesh
                            _workers_and_args = _dispatch_args(
                                workers, local_dispatch, execute, local_mesh, args, kwargs)
                            _, args, kwargs = _workers_and_args[rank]
                        result = func(self, *args, **kwargs)
                        if isinstance(result, Future):
                            # A worker method may hand back a plain Future (an engine-side async
                            # submission). Waiting here makes the caller's reference come ready only
                            # when that work truly finishes, which is what gives a lazily-collected
                            # call its Future semantics; in local mode the same return goes to the
                            # caller as-is, so both modes speak one contract with no probing.
                            result = result.result()
                        if stash is not None:
                            # Partial collect: lift the declared heavy channels to refs here, at the
                            # value's birthplace, so the caller's collect materialises only the
                            # light half. This branch exists solely on the worker path -- local
                            # mode never reaches it, so no environment probing anywhere.
                            result = _stash_channels(result, stash)
                        return result
                    else:
                        # This is the driver
                        from ._ray import RayHelper

                        if session_ids is not None and RayHelper.has_ref(args, kwargs):
                            raise ValueError(
                                '_session_ids does not support LazyCollect inputs; materialize the batch first.')

                        # Resolve the effective ray.get timeout before choosing execute_method:
                        # the decorator's explicit value wins, the instance attribute is the
                        # fallback. ``is not None`` (not ``or``) so that a decorator ``timeout=0``
                        # is honored instead of falling back to unbounded waiting.
                        _rgt = timeout if timeout is not None else getattr(self, '_ray_get_timeout', None)
                        execute_method = RayHelper.execute_all_async if not sync else functools.partial(
                            RayHelper.execute_all_sync, timeout=_rgt)
                        _default_workers = worker_ids is None and worker_selector is None
                        if exclusive:
                            # Exclusive methods take the handle's configuration lock: in-flight calls
                            # drain first and new ordinary calls wait (declared per-method instead of a
                            # component-side lock-provider hook).
                            lock = getattr(self, '_worker_lock', None)
                            if lock is not None:
                                lock.acquire()
                                _configuration_lock = lock
                        begin_call = getattr(self, '_begin_worker_call', None)
                        if begin_call is not None:
                            _worker_reservation = begin_call(worker_ids, worker_selector)
                            _worker_dispatching = _worker_reservation is not None
                        if _worker_reservation is not None:
                            selected_ids = _worker_reservation
                            _subset = not _default_workers
                        else:
                            if worker_ids is None and worker_selector is not None:
                                worker_ids = getattr(self, worker_selector)()
                            selected_ids = worker_ids
                            _subset = worker_ids is not None
                        _workers = _get_workers(_select_workers(self, selected_ids), execute)
                        if not _workers:
                            if session_ids:
                                raise RuntimeError('No sampler worker is available for sticky session routing.')
                            return []
                        if _subset and worker_selector is not None:
                            args, kwargs = self._normalize_sampling_args(args, kwargs)
                        _dispatch = (getattr(self, '_dispatch_sampling_inputs', 'slice')
                                     if _subset and dispatch == 'slice_dp' else dispatch)
                        # Read the mesh only after the worker set is chosen (inside the pool lock's
                        # dispatch window) so an add/remove replica cannot interleave a stale mesh with a
                        # new actor list.
                        _mesh = None if _subset else getattr(self, 'device_mesh', None)
                        _concurrent_actor = bool(getattr(self, '_max_concurrency', None))
                        _track_calls = (_concurrent_actor or _subset or _worker_reservation is not None
                                        or _sticky_key_value is not None)
                        if _track_calls and not control:
                            # Every method waits here, not just the continuous ones:
                            # the point is to keep a weight update or a sleep from
                            # reaching a worker that still has generations running.
                            # Control calls are exempt: they are the ones that *end* those
                            # generations, so waiting for them would be waiting on themselves.
                            _cw_barrier(self, func.__name__, _workers,
                                        materialize=not (_subset or _worker_reservation is not None))
                        if enable_continous_work and not RayHelper.has_ref(args, kwargs):
                            assert not sync, (f'{func.__name__}: enable_continous_work cannot be used with sync=True, '
                                              'which exists for collectives that must run in lock step.')
                            _batch_len = _cw_batch_len(args, kwargs)
                            if _batch_len or session_ids is not None:
                                return _run_continous_work(self, func.__name__, execute_method, _workers, args, kwargs,
                                                           _batch_len,
                                                           _rgt,
                                                           on_dispatched=_finish_dispatch, session_ids=session_ids)
                        if RayHelper.has_ref(args, kwargs):
                            # If has any object-ref, dispatch in worker, because we don't know the structure in the ref.
                            # for example, dataloader returns any data list.
                            _workers_and_args = [
                                (worker, args, dict(kwargs, _dispatch_context=(len(_workers), index, _subset)))
                                for index, worker in enumerate(_workers)
                            ]
                        else:
                            # dispatch now
                            _workers_and_args = _dispatch_args(
                                _workers, _dispatch, execute, _mesh, args, kwargs)

                        _tracked_refs = []

                        def _track_submission(worker, result):
                            _submitted_calls.append((get_worker_id(worker), result))
                            if sticky and _sticky_key_value is not None:
                                # First worker to serve this key owns it from now on.
                                _sticky_bind(self, _sticky_key_value, get_worker_id(worker))
                            _tracked_refs.extend(_cw_register(self, func.__name__, [result], [worker]))

                        # A configured sampler does not hold the selection lock while waiting for a sync RPC's result.
                        _sync_sampler = sync and _worker_reservation is not None
                        if _sync_sampler:
                            execute_method = RayHelper.execute_all_async
                        result = execute_method(func.__name__, _workers_and_args,
                                                on_submit=_track_submission if _track_calls else None)
                        _finish_dispatch()
                        if _sync_sampler:
                            import ray
                            result = ray.get(result)
                        # This is a result future, call it to get the actual result
                        result_func = RayHelper.do_get_and_collect_func(
                            _collect_func, collect, result, device_mesh, timeout=_rgt)
                        _local_lazy_collect = infra._lazy_collect
                        recorder = getattr(self, '_record_worker_configuration', None)
                        if func.__name__ == '__iter__':
                            # return self
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
                            result = [_res[0] for _res in result]
                            result_func._futures = result

                        if lazy_collect is not None:
                            # Maybe this function returns a small object
                            _local_lazy_collect = lazy_collect
                        if hasattr(self, '_lazy_collect'):
                            # _lazy_collect in class has the highest priority
                            # This is the unique case that an object ref contains another
                            # And this is user independent, only decided by the code.
                            _local_lazy_collect = self._lazy_collect
                        if _configuration_lock is not None and lazy_collect is not True:
                            # An exclusive call normally collects eagerly so the configuration lock is
                            # held through completion. A method that EXPLICITLY declared
                            # lazy_collect=True (the two-phase checkpoint-engine weight sync) must still
                            # hand back a future -- its lock release is deferred to collection below.
                            _local_lazy_collect = False
                        if _local_lazy_collect:
                            if _subset or _worker_reservation is not None:
                                # When a subset call's payload crosses Ray, it must not capture the driver handler's locks or lifecycle state.
                                _lazy_result = result_func
                            else:
                                # Notify exceptions and unregister continuous-work refs at the real collect.
                                # A _SideEffectLazyCollect (NOT a functools.wraps function) so the wrapped
                                # LazyCollect's full access protocol survives -- a bare function drops
                                # ``.get`` / ``.result`` (instance methods) and ``[]`` / ``in`` / ``iter`` /
                                # ``len`` (type-resolved dunders), breaking a driver that reads a forwarded
                                # output directly (``forward_only(...).get('logps')``). ``_is_lazy_collect``
                                # stays a class attr so the getattr-based resolvers (ScoreWorker._resolve,
                                # scoring.response_token_scores, RayHelper.has_ref) still detect the handle.
                                def _notify_collect_error(_e, _caller=_caller, _ctx=_ctx, _infra=infra):
                                    _tag_exc(_e, _caller)
                                    notify_exception(_infra._notifier, _ctx, _e, _infra._name)

                                _lazy_result = _SideEffectLazyCollect(
                                    result_func, on_error=_notify_collect_error,
                                    on_collect=lambda: _cw_unregister(self, func.__name__, _tracked_refs))
                            if _configuration_lock is not None:
                                # exclusive + explicit lazy_collect (the two-phase checkpoint-engine weight
                                # sync): the caller launches the peer half CONCURRENTLY and waits on both
                                # futures, so the configuration lock must stay held until the future is
                                # COLLECTED -- releasing at method return would let an ordinary call mutate
                                # params mid-sync. The same driver thread collects, so releasing the
                                # reentrant lock there is valid. Composition chains the inner hooks (notify /
                                # unregister) ahead of this release, and the once-only guard in
                                # _SideEffectLazyCollect keeps a cached re-read from releasing twice.
                                _lock_release_deferred = True
                                return _SideEffectLazyCollect(_lazy_result, on_collect=_configuration_lock.release)
                            return _lazy_result
                        try:
                            output = result_func()
                            completed = getattr(self, '_worker_call_completed', None)
                            if completed is not None and _worker_reservation is not None:
                                completed(func.__name__, _worker_reservation)
                            _call_completed = True
                            if recorder is not None and replay and _default_workers and _configuration_lock is not None:
                                recorder(func.__name__, args, kwargs)
                            return output
                        finally:
                            _cw_unregister(self, func.__name__, _tracked_refs)
                else:
                    raise NotImplementedError(f'Unsupported mode {infra._mode}')
            except StopIteration:
                raise
            except Exception as _e:
                _tag_exc(_e, _caller)
                notify_exception(infra._notifier, _ctx, _e, infra._name)
                raise
            finally:
                try:
                    _finish_dispatch()
                    if _worker_reservation is not None:
                        try:
                            settle = getattr(self, '_settle_worker_controls', None)
                            if not _call_completed and settle is not None:
                                settle(func.__name__, _submitted_calls)
                        finally:
                            self._end_worker_call(_worker_reservation)
                finally:
                    if _configuration_lock is not None and not _lock_release_deferred:
                        _configuration_lock.release()

        wrapper._execute = execute
        wrapper._collect = collect
        wrapper._dispatch = dispatch
        wrapper._lazy_collect = lazy_collect
        wrapper._sync = sync
        wrapper._enable_continous_work = enable_continous_work
        wrapper._sticky = sticky
        wrapper._sticky_key = sticky_key
        wrapper._exclusive = exclusive
        wrapper._control = control
        wrapper._replay = replay
        wrapper._worker_selector = worker_selector
        if enable_continous_work:
            # Attached to the class by remote_class, and called instead of this
            # method when the driver routes requests worker by worker.
            wrapper._worker_async_companion = _make_worker_async_companion(func, wrapper)
        return wrapper

    return decorator
