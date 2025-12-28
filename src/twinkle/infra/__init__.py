import functools
import inspect
import os
from typing import Literal, List, Optional, Union, Callable
from typing import TypeVar

import numpy as np

from ..utils import DeviceGroup, DeviceMesh, Platform
from .ray import RayHelper
from ..utils import requires, framework_util, check_unsafe

T1 = TypeVar('T1', bound=object)

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

_nproc_per_node: Optional[int] = 8

_seed = 42

_full_determinism = False

_device_group: Optional[List[DeviceGroup]] = [
    DeviceGroup(
        name='local',
        ranks=list(range(Platform.get_world_size())),
        device_type=Platform.get_platform().device_prefix(),
    )
]

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


def get_device_placement(device_group = None) -> str:
    global _device_group
    if device_group is None:
        device_group = _device_group

    lines = ["=" * 80, "                        DEVICE PLACEMENT TOPOLOGY", "=" * 80]

    for group_idx, group in enumerate(device_group):
        lines.append("")
        lines.append(f"┌{'─' * 76}┐")
        lines.append(f"│ DeviceGroup: {group.name:<61} │")
        lines.append(f"│ Device Type: {group.device_type:<61} │")

        if isinstance(group.ranks, list):
            ranks_str = str(group.ranks) if len(group.ranks) <= 16 else f"{group.ranks[:8]} ... {group.ranks[-4:]}"
        else:
            ranks_str = str(group.ranks)
        lines.append(f"│ Ranks: {ranks_str:<67} │")
        lines.append(f"├{'─' * 76}┤")

        if not group._device_mesh:
            lines.append(f"│ {'(No device meshes configured)':<74} │")
        else:
            for mesh_name, mesh in group._device_mesh.items():
                lines.append(f"│                                                                            │")
                lines.append(f"│  ◆ DeviceMesh: {mesh_name:<59} │")

                if mesh.mesh_dim_names:
                    dim_info = " × ".join(
                        f"{name}={size}"
                        for name, size in zip(mesh.mesh_dim_names, mesh.mesh.shape)
                    )
                    lines.append(f"│    Dimensions: {dim_info:<59} │")

                world_sizes = []
                for dim_name in ['pp', 'dp', 'tp', 'ep', 'sp', 'cp']:
                    ws = mesh._get_world_size_for_dim(dim_name)
                    if ws > 1:
                        world_sizes.append(f"{dim_name.upper()}={ws}")
                if world_sizes:
                    lines.append(f"│    Active Parallelism: {', '.join(world_sizes):<51} │")

                lines.append(f"│                                                                            │")
                lines.append(f"│    Mesh Layout:                                                            │")

                mesh_array = mesh.mesh
                if mesh_array.ndim == 1:
                    mesh_array = mesh_array.reshape(1, -1)

                if mesh_array.ndim == 2:
                    rows, cols = mesh_array.shape

                    if mesh.mesh_dim_names and len(mesh.mesh_dim_names) >= 2:
                        col_label = mesh.mesh_dim_names[-1].upper()
                    else:
                        col_label = "COL"

                    if mesh.mesh_dim_names and len(mesh.mesh_dim_names) >= 1:
                        row_label = mesh.mesh_dim_names[0].upper() if mesh_array.ndim == 2 and len(
                            mesh.mesh_dim_names) >= 2 else mesh.mesh_dim_names[0].upper()
                    else:
                        row_label = "ROW"

                    max_val = mesh_array.max()
                    cell_width = max(3, len(str(max_val)) + 2)

                    header = "│    " + " " * 4
                    for c in range(min(cols, 12)):
                        header += f"{c:^{cell_width}}"
                    if cols > 12:
                        header += " ..."
                    header = header.ljust(75) + "│"
                    lines.append(header)

                    sep_line = "│    " + " " * 4 + "┌" + ("─" * cell_width + "┬") * (
                                min(cols, 12) - 1) + "─" * cell_width + "┐"
                    sep_line = sep_line.ljust(75) + "│"
                    lines.append(sep_line)

                    for r in range(min(rows, 8)):
                        row_str = f"│    {r:>3} │"
                        for c in range(min(cols, 12)):
                            val = mesh_array[r, c]
                            row_str += f"{val:^{cell_width}}│"
                        if cols > 12:
                            row_str += " ..."
                        row_str = row_str.ljust(75) + "│"
                        lines.append(row_str)

                        if r < min(rows, 8) - 1:
                            mid_line = "│    " + " " * 4 + "├" + ("─" * cell_width + "┼") * (
                                        min(cols, 12) - 1) + "─" * cell_width + "┤"
                            mid_line = mid_line.ljust(75) + "│"
                            lines.append(mid_line)

                    if rows > 8:
                        lines.append(f"│    {'...':<71} │")

                    bottom_line = "│    " + " " * 4 + "└" + ("─" * cell_width + "┴") * (
                                min(cols, 12) - 1) + "─" * cell_width + "┘"
                    bottom_line = bottom_line.ljust(75) + "│"
                    lines.append(bottom_line)

                elif mesh_array.ndim > 2:
                    lines.append(f"│    (High-dimensional mesh: shape={mesh_array.shape})                      │")

                lines.append(f"│                                                                            │")

        lines.append(f"└{'─' * 76}┘")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def _get_workers(workers, execute):
    if execute == 'first':
        return [workers[0]]
    elif execute == 'all':
        return workers
    elif execute == 'peer':
        return workers[Platform.get_peer_index(len(workers))]
    else:
        raise ValueError(f'Unsupported execute method: {execute}')


def _collect_func(method: Union[Literal['none', 'flatten'], Callable], result):
    if not result:
        return result
    if isinstance(result[0], tuple):
        output = []
        for i in range(len(result[0])):
            _single_result = [r[i] for r in result]
            output.append(_collect_func(method, _single_result))
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


def _dispatch_args(workers, dispatch, execute, device_mesh: Optional[DeviceMesh], args, kwargs):
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
        length = len(workers)
        result = []
        for i in range(length):
            sliced_args, sliced_kwargs = dispatch(length, i, *args, **kwargs)
            result.append((workers[i], sliced_args, sliced_kwargs))
        return result
    else:
        raise ValueError(f'Unsupported dispatch method: {dispatch}')


def _get_device_mesh_param_name(init_method) -> str:
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
    for arg in (list(args) + list(kwargs.values())):
        if isinstance(arg, DeviceMesh):
            return arg
    return None


def remote_class():

    def decorator(cls):
        # Get device mesh parameter name
        device_mesh_name = _get_device_mesh_param_name(cls.__init__)
        if _mode == 'local':
            init_method = cls.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                # Get the actual device_mesh
                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name:
                    if device_mesh is None:
                        # DeviceMesh not passed, create DDP by default
                        device_mesh = DeviceMesh(
                            device_type=Platform.get_platform().device_prefix(),
                            mesh=np.array([Platform.get_world_size()]),
                            mesh_dim_names=('data',)
                        )
                        kwargs[device_mesh_name] = device_mesh
                    assert len(_device_group) == 1
                    _device_group[0]._device_mesh[self.__class__.__name__] = device_mesh
                    init_method(*args, **kwargs)
                else:
                    init_method(*args, **kwargs)
            cls.__init__ = new_init
            return cls
        elif _mode == 'ray':
            from .ray import RayHelper

            cls.decorated = True
            init_method = cls.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                # In case the same class created twice in the same device group
                frame = inspect.currentframe().f_back
                caller_file = frame.f_code.co_filename
                caller_line = frame.f_lineno
                instance_id = f"{caller_file}:{caller_line}"
                remote_group = kwargs.pop('remote_group', None)
                check_unsafe(*args, **kwargs)

                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh is None and device_mesh_name:
                    # DeviceMesh not passed, create DDP by default
                    device_mesh = DeviceMesh(
                        device_type=Platform.get_platform().device_prefix(),
                        mesh=np.array([Platform.get_world_size()]),
                        mesh_dim_names=('data',)
                    )
                    kwargs[device_mesh_name] = device_mesh

                if remote_group and device_mesh_name:
                    device_group = [dg for dg in _device_group if dg.name == remote_group][0]
                    device_group._device_mesh[self.__class__.__name__] = device_mesh

                if (not remote_group) or os.environ.get('CLUSTER_NAME') == remote_group:
                    init_method(self, *args, **kwargs)

                    if remote_group and os.environ.get('CLUSTER_NAME') == remote_group:
                        # Seed when a remote class is created.
                        framework_util.seed_everything(int(os.environ['TWINKLE_SEED']),
                                                       bool(int(os.environ['TWINKLE_FULL_DETERMINISM'])))
                        if hasattr(cls, '__iter__'):
                            assert not hasattr(cls, '__next__')

                            def __iter__(_self):
                                _iter = _self.__iter_origin__()
                                assert _iter is not _self
                                _self._iter = _iter

                            def __next__(_self):
                                return next(_self._iter)

                            cls.__iter_origin__ = cls.__iter__
                            cls.__iter__ = remote_function(execute=cls.__iter_origin__._execute,
                                                           dispatch=cls.__iter_origin__._dispatch,
                                                           collect=cls.__iter_origin__._collect)(__iter__)
                            cls.__next__ = remote_function(execute=cls.__iter_origin__._execute,
                                                           dispatch=cls.__iter_origin__._dispatch,
                                                           collect=cls.__iter_origin__._collect)(__next__)
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
                            _workers_and_args = _dispatch_args(_get_workers(self._actors, 'all'), 'all',
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
                    _workers_and_args = _dispatch_args(_get_workers(self._actors, execute), dispatch,
                                                        execute, device_mesh, args, kwargs)
                    result = RayHelper.execute_all_async(func.__name__, _workers_and_args)
                    return RayHelper.do_get_and_collect_func(_collect_func, collect, result)
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        wrapper._execute = func._execute = execute
        wrapper._collect = func._collect = collect
        wrapper._dispatch = func._dispatch = dispatch

        return wrapper

    return decorator
