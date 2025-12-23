import functools
import inspect
import os
from typing import Literal, List, Optional, Union, Callable
from typing import TypeVar

import numpy as np

from .device_group import DeviceGroup
from .ray import RayHelper
from ..utils import requires, framework_util

T1 = TypeVar('T1', bound=object)

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

_nproc_per_node: Optional[int] = 8

_device_group: Optional[List[DeviceGroup]] = None

_remote_components: dict = {}


def initialize(mode: Literal['local', 'ray'],
               nproc_per_node: Optional[int] = 8,
               groups: Optional[List[DeviceGroup]] = None,):
    global _mode, _device_group, _nproc_per_node
    assert mode in ('local', 'ray')
    _mode = mode
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

        def get_peer_index(target_size):
            rank = framework_util.get_rank()
            world_size = framework_util.get_world_size()

            k, m = divmod(target_size, world_size)
            start_idx = rank * k + min(rank, m)
            end_idx = (rank + 1) * k + min(rank + 1, m)
            if target_size < world_size:
                start_idx = rank % target_size
                end_idx = start_idx + 1

            return slice(start_idx, end_idx)
        return workers[get_peer_index(len(workers))]
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


def dispatch_args(workers, dispatch, execute, args, kwargs):

    length = len(workers)
    if execute == 'first':
        return [(workers[0], args, kwargs)]
    elif dispatch == 'all':
        return [(worker, args, kwargs) for worker in workers]
    elif dispatch == 'slice':
        result = []

        def dispatch_func(arg, n):
            if isinstance(arg, list):
                k, m = divmod(len(arg), n)
                return [arg[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
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
            init_params = inspect.signature(init_method).parameters
            has_remote_group = 'remote_group' in init_params

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                if not has_remote_group:
                    remote_group = kwargs.pop('remote_group', None)
                else:
                    remote_group = kwargs.get('remote_group', None)
                if (not remote_group) or os.environ.get('CLUSTER_NAME') == remote_group:
                    init_method(self, *args, **kwargs)
                else:
                    # Create remote workers
                    _actors = RayHelper.create_workers(cls, remote_group, *args, **kwargs)
                    self._actors = _actors

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
            if _mode == 'local':
                return func(*args, **kwargs)
            elif _mode == 'ray':
                from .ray import RayHelper
                _workers_and_args = dispatch_args(get_workers(self._actors, execute), dispatch,
                                                    execute, args, kwargs)
                result = RayHelper.execute_all_sync(func.__name__, _workers_and_args)
                return collect_func(collect, result)
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        return wrapper

    return decorator
