import functools
import inspect
import os
from typing import Literal, List, Optional, Union, Callable
from typing import TypeVar

import numpy as np

from twinkle import DeviceGroup, DeviceMesh
from .ray import RayHelper
from twinkle import requires, framework_util
from twinkle import check_unsafe

T1 = TypeVar('T1', bound=object)

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

_nproc_per_node: Optional[int] = 8

_seed = 42

_full_determinism = False

_device_group: Optional[List[DeviceGroup]] = None

_remote_components: dict = {}


def initialize(mode: Literal['local', 'ray'],
               nproc_per_node: Optional[int] = 8,
               seed: int = 42,
               full_determinism: bool = False,
               groups: Optional[List[DeviceGroup]] = None,):
    global _mode, _device_group, _nproc_per_node, _seed, _full_determinism
    assert mode in ('local', 'ray')
    _mode = mode
    _full_determinism = full_determinism
    if seed is not None:
        _seed = seed
        framework_util.seed_everything(seed, full_determinism)
    if _mode == 'ray':
        requires('ray')
        _device_group = groups
        _nproc_per_node = nproc_per_node


def get_workers(workers, execute):
    if execute == 'first':
        return [workers[0]]
    elif execute == 'all':
        return workers
    elif execute == 'peer':
        return workers[framework_util.get_peer_index(len(workers))]
    else:
        raise ValueError(f'Unsupported execute method: {execute}')


def collect_func(method: Union[Literal['none', 'flatten'], Callable], result):
    if not result:
        return result
    if isinstance(result[0], tuple):
        output = []
        for i in range(len(result[0])):
            _single_result = [r[i] for r in result]
            output.append(collect_func(method, _single_result))
        return output
    if method == 'none':
        return result
    elif method == 'flatten':
        flatten = [item for sublist in result for item in sublist]
        if isinstance(result[0], np.ndarray):
            return np.array(flatten)
        return type(result[0])(flatten)
    elif isinstance(method, Callable):
        # Callable
        return method(result)
    else:
        raise ValueError(f'Unsupported collect method: {method}')


def dispatch_args(workers, dispatch, execute, device_mesh: DeviceMesh, args, kwargs):
    if execute == 'first':
        return [(workers[0], args, kwargs)]
    elif dispatch == 'all':
        return [(worker, args, kwargs) for worker in workers]
    elif dispatch == 'slice':
        result = []
        if device_mesh is not None:
            # TODO this may occurs error when remote calls remote
            assert device_mesh.world_size == len(workers)
        length = len(workers) if not device_mesh else device_mesh.dp_world_size
        dp_repeat = len(workers) // length

        def dispatch_func(arg, n):
            if isinstance(arg, list):
                _args = []
                k, m = divmod(len(arg), n)
                for i in range(n):
                    for j in range(dp_repeat):
                        _args.append(arg[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
                return _args
            else:
                return [arg] * n

        args = [dispatch_func(arg, length) for arg in args]
        kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}
        for i in range(length):
            sliced_args = tuple(arg[i] for arg in args)
            sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
            if (sliced_args and sliced_args[0]) or (kwargs and list(kwargs.values())):
                result.append((workers[i], sliced_args, sliced_kwargs))

        return result
    elif isinstance(dispatch, Callable):
        # dispatch is Callable
        length = len(workers)
        result = []
        for i in range(length):
            sliced_args, sliced_kwargs = dispatch(length, i, *args, **kwargs)
            result.append((workers[i], sliced_args, sliced_kwargs))
        return result
    else:
        raise ValueError(f'Unsupported dispatch method: {dispatch}')


def remote_class():

    def decorator(cls):
        if _mode == 'local':
            return cls
        elif _mode == 'ray':
            from .ray import RayHelper

            cls.decorated = True
            init_method = cls.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                frame = inspect.currentframe().f_back
                caller_file = frame.f_code.co_filename
                caller_line = frame.f_lineno
                instance_id = f"{caller_file}:{caller_line}"
                remote_group = kwargs.pop('remote_group', None)
                check_unsafe(*args, **kwargs)
                if (not remote_group) or os.environ.get('CLUSTER_NAME') == remote_group:
                    init_method(self, *args, **kwargs)

                    if remote_group and os.environ.get('CLUSTER_NAME') == remote_group:
                        framework_util.seed_everything(int(os.environ['TWINKLE_SEED']),
                                                       bool(int(os.environ['TWINKLE_FULL_DETERMINISM'])))
                        if hasattr(cls, '__iter__'):
                            assert not hasattr(cls, '__next__')

                            def __iter__(self):
                                _iter = self.__iter_origin__()
                                assert _iter is not self
                                self._iter = _iter

                            def __next__(self):
                                return next(self._iter)

                            cls.__iter_origin__ = cls.__iter__
                            cls.__iter__ = remote_function()(__iter__)
                            cls.__next__ = remote_function()(__next__)
                else:
                    # Create remote workers
                    _actors = RayHelper.create_workers(cls,
                                                       remote_group,
                                                       'peer',
                                                       instance_id=instance_id,
                                                       seed=_seed,
                                                       full_determinism=_full_determinism,
                                                       *args, **kwargs)
                    self._actors = _actors
                    for arg in (list(args) + list(kwargs.values())):
                        if isinstance(arg, DeviceMesh):
                            self.device_mesh = arg
                            break
                    if hasattr(cls, '__iter__'):

                        def __iter__(self):
                            _workers_and_args = dispatch_args(get_workers(self._actors, 'all'), 'all',
                                                              'all', None, (), {})
                            RayHelper.execute_all_sync('__iter__', _workers_and_args)
                            return self

                        cls.__iter__ = __iter__

                self.remote_group = remote_group
                self._instance_id = instance_id

            cls.__init__ = new_init
            return cls
        else:
            raise ValueError(f'Unsupported mode: {_mode}')

    return decorator


def remote_function(dispatch: Union[Literal['slice', 'all'], Callable] = 'slice',
             execute: Literal['first', 'peer', 'all'] = 'all',
             collect: Union[Literal['none', 'flatten'], Callable] = 'none'):
    """Remote execution function.

    Args:
        dispatch: How to dispatch the arguments.
            'slice': load balance
            'all': all processes do the same thing
            Callable: A callable that handles the dispatching
        execute: How to execute
            'first': Only first worker
            'peer': Only peer workers
            'all': All processes
        collect: How to collect the results.
            'none': Return as-is
            'flatten': Return a flattened list
            Callable: A callable that handles the collection
    Returns:
        The execution result.
    """

    def decorator(func: Callable[..., T1]) -> Callable[..., T1]:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T1:
            check_unsafe(*args, **kwargs)
            device_mesh = getattr(self, 'device_mesh', None)
            if _mode == 'local':
                return func(*args, **kwargs)
            elif _mode == 'ray':
                if not hasattr(self, '_actors'):
                    return func(self, *args, **kwargs)
                else:
                    from .ray import RayHelper
                    args, kwargs = RayHelper.do_get_and_collect(args, kwargs)
                    _workers_and_args = dispatch_args(get_workers(self._actors, execute), dispatch,
                                                        execute, device_mesh, args, kwargs)
                    result = RayHelper.execute_all_async(func.__name__, _workers_and_args)
                    return RayHelper.do_get_and_collect_func(collect_func, collect, result)
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        return wrapper

    return decorator
