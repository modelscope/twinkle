# Copyright (c) ModelScope Contributors. All rights reserved.
import functools
import inspect
import os
from typing import Literal, List, Optional, Union, Callable, Any
from typing import TypeVar

import numpy as np

from twinkle.utils import DeviceGroup, DeviceMesh, Platform
from twinkle.utils import requires, framework_util, check_unsafe


T1 = TypeVar('T1', bound=object)

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

_seed = 42

_lazy_collect = True

_full_determinism = False

_device_group: Optional[List[DeviceGroup]] = None

_device_mesh = None

_remote_components: dict = {}


def initialize(mode: Literal['local', 'ray'] = 'local',
               nproc_per_node: int = 8,
               ncpu_proc_per_node: int = 8,
               seed: int = 42,
               full_determinism: bool = False,
               groups: Optional[List[DeviceGroup]] = None,
               global_device_mesh: Optional[DeviceMesh] = None,
               lazy_collect: bool = True):
    """Initialize the twinkle infrastructure.

    Args:
        mode: The mode of twinkle works in.
            'local': Run with a single GPU, or torchrun.
            'ray': Run in ray cluster.
        nproc_per_node: The GPU count(number of processes) per node.
        ncpu_proc_per_node: The CPU processes count per node.
        seed: Seed everything with this.
        full_determinism: Freeze the random, use determinism kernels, default `False`.
        groups: The device groups of the training.
        global_device_mesh: The global default device mesh.
        lazy_collect: Lazy collect all outputs in workers, default `True`.
    """
    global _mode, _device_group, _seed, _full_determinism, _lazy_collect, _device_mesh
    assert mode in ('local', 'ray')
    _mode = mode
    _full_determinism = full_determinism
    _lazy_collect = lazy_collect
    if global_device_mesh is not None:
        _device_mesh = global_device_mesh
    else:
        # groups can be given by default
        _device_mesh = DeviceMesh(
            device_type=Platform.device_prefix(),
            mesh=np.arange(Platform.get_world_size()),
            mesh_dim_names=('dp',)
        )
    if seed is not None:
        _seed = seed
        framework_util.seed_everything(seed, full_determinism)
    if _mode == 'local':
        if groups is not None:
            _device_group = groups
        else:
            _device_group = [
                DeviceGroup(
                    name='default',
                    ranks=list(range(Platform.get_world_size())),
                    device_type=Platform.get_platform().device_prefix(),
                )]
            
        assert Platform.get_world_size() == _device_mesh.world_size
    else:
        requires('ray')
        from ._ray import RayHelper
        assert groups is not None
        # groups is needed for ray
        _device_group = groups
        RayHelper.initialize(nproc_per_node=nproc_per_node,
                             ncpu_proc_per_node=ncpu_proc_per_node,
                             device_groups=_device_group)


def get_device_placement(device_group=None) -> str:
    """Get the device placement graph, can be used to show the training topology.

    Args:
        device_group: The device group of the training, default will use the global `device_group`.

    Returns:
        A string containing the training topology.
    """
    global _device_group
    if device_group is None:
        device_group = _device_group

    WIDTH = 80

    def box_line(content="", align="left", prefix="│", suffix="│"):
        inner_width = WIDTH - 4
        if align == "center":
            text = content.center(inner_width)
        else:
            text = content.ljust(inner_width)
        return f"{prefix} {text} {suffix}"

    def header_box(title):
        return [
            "╔" + "═" * (WIDTH - 2) + "╗",
            box_line(title, align="center", prefix="║", suffix="║"),
            "╚" + "═" * (WIDTH - 2) + "╝",
        ]

    def section_top(title=""):
        lines = ["┌" + "─" * (WIDTH - 2) + "┐"]
        if title:
            lines.append(box_line(f"◈ {title}", prefix="│", suffix="│"))
            lines.append("├" + "─" * (WIDTH - 2) + "┤")
        return lines

    def section_bottom():
        return ["└" + "─" * (WIDTH - 2) + "┘"]

    def format_ranks(ranks):
        if isinstance(ranks, list):
            if len(ranks) <= 16:
                return str(ranks)
            return f"{ranks[:6]} ... {ranks[-3:]} ({len(ranks)} total)"
        return str(ranks)

    def render_mesh_grid(mesh_array, dim_names):
        """Render a compact mesh visualization."""
        lines = []

        if mesh_array.ndim == 1:
            mesh_array = mesh_array.reshape(1, -1)

        if mesh_array.ndim > 2:
            lines.append(box_line(f"    ⊞ High-dim mesh: shape={mesh_array.shape}"))
            return lines

        rows, cols = mesh_array.shape
        max_rows, max_cols = 6, 10
        show_rows, show_cols = min(rows, max_rows), min(cols, max_cols)

        cell_w = max(4, len(str(mesh_array.max())) + 2)

        header = "      " + "".join(f"{i:^{cell_w}}" for i in range(show_cols))
        if cols > max_cols:
            header += " ⋯"
        lines.append(box_line(f"    {header}"))

        # Top border
        border = "      ╭" + "─" * (cell_w * show_cols + show_cols - 1) + "╮"
        lines.append(box_line(f"    {border}"))

        # Data rows
        for r in range(show_rows):
            row_data = "│".join(f"{mesh_array[r, c]:^{cell_w}}" for c in range(show_cols))
            row_str = f"   {r:>2} │{row_data}│"
            if cols > max_cols:
                row_str += " ⋯"
            lines.append(box_line(f"    {row_str}"))

        if rows > max_rows:
            lines.append(box_line(f"         {'⋮':^{cell_w * show_cols}}"))

        # Bottom border
        border = "      ╰" + "─" * (cell_w * show_cols + show_cols - 1) + "╯"
        lines.append(box_line(f"    {border}"))

        return lines

    # Build output
    lines = header_box("DEVICE PLACEMENT TOPOLOGY")
    lines.append("")

    for group in device_group:
        lines.extend(section_top(f"DeviceGroup: {group.name}"))
        lines.append(box_line(f"  ├─ Device Type : {group.device_type}"))
        lines.append(box_line(f"  └─ Ranks       : {format_ranks(group.ranks)}"))

        if not group._device_mesh:
            lines.append(box_line(""))
            lines.append(box_line("  (No device meshes configured)", align="center"))
        else:
            for mesh_name, mesh in group._device_mesh.items():
                lines.append(box_line(""))
                lines.append(box_line(f"  ┌─ DeviceMesh: {mesh_name}"))

                # Dimensions
                if mesh.mesh_dim_names:
                    dim_info = " × ".join(
                        f"{name}={size}" for name, size in zip(mesh.mesh_dim_names, mesh.mesh.shape)
                    )
                    lines.append(box_line(f"  │  Dimensions : {dim_info}"))

                # Active parallelism
                parallelism = []
                for dim in ['pp', 'dp', 'tp', 'ep', 'sp', 'cp', 'fsdp']:
                    ws = mesh._get_world_size_for_dim(dim)
                    if ws is not None and ws > 1:
                        parallelism.append(f"{dim.upper()}={ws}")

                if parallelism:
                    lines.append(box_line(f"  │  Parallelism: {', '.join(parallelism)}"))

                # Mesh layout
                lines.append(box_line(f"  │"))
                lines.append(box_line(f"  └─ Mesh Layout:"))
                lines.extend(render_mesh_grid(mesh.mesh, mesh.mesh_dim_names or []))

        lines.append(box_line(""))
        lines.extend(section_bottom())
        lines.append("")

    return "\n" + "\n".join(lines)


def _get_workers(workers, execute):
    if execute == 'first':
        return [workers[0]]
    elif execute == 'all':
        return workers
    elif execute == 'peer':
        return workers[Platform.get_peer_index(len(workers))]
    else:
        raise ValueError(f'Unsupported execute method: {execute}')


def _collect_func(method: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable], result: List[Any], device_mesh: DeviceMesh=None):
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
        return np.mean(result)
    elif method == 'sum':
        return np.sum(result)
    elif method == 'first':
        return result[0]
    elif method == 'last_pp':
        assert device_mesh is not None
        return [r for i, r in enumerate(result) if i in device_mesh.get_pp_last_ranks()]
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
        # split by dp. each worker in one ep will receive the same argument
        result = []
        # if device_mesh is not None:
            # TODO this may occurs error when remote calls remote
            # Comment this because remote_class supports `first``
            # assert device_mesh.world_size == len(workers)
        length = len(workers)

        def dispatch_func(arg, n):
            if isinstance(arg, list):
                _args = []
                for i in range(n):
                    _args.append(arg[device_mesh.get_slice(len(arg), device_mesh.get_data_rank_from_global_rank(i))])
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
    elif isinstance(dispatch, Callable):
        length = len(workers)
        result = []
        for i in range(length):
            sliced_args, sliced_kwargs = dispatch(length, i, args, kwargs)
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
    from ._ray import RayHelper
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

def remote_class(execute: Literal['first', 'peer', 'all'] = 'peer'):
    """Patch each class used in remote clusters with this decorator.

    Use this decorator to wrap your class to enable it to execute in a remote cluster.

    """
    def decorator(cls):
        # Get device mesh parameter name
        device_mesh_name = _get_device_mesh_param_name(cls.__init__)
        init_method = cls.__init__

        @functools.wraps(init_method)
        def new_init(self, *args, **kwargs):
            if _mode == 'local':
                # Get the actual device_mesh
                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name and _device_group is not None:
                    if device_mesh is None:
                        # Local mode can safely assign the default device mesh
                        device_mesh = _device_mesh
                        kwargs[device_mesh_name] = _device_mesh
                    assert len(_device_group) == 1 # only one device group is allowed
                    _device_group[0]._device_mesh[self.__class__.__name__] = device_mesh
                    init_method(self, *args, **kwargs)
                else:
                    # Pop the device_mesh
                    args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                    kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                    init_method(self, *args, **kwargs)
            elif _mode == 'ray':
                from ._ray import RayHelper

                # In case the same class created twice in the same device group
                # Try to get the caller's line
                frame = inspect.currentframe().f_back
                caller_file = frame.f_code.co_filename.replace(os.sep, '_').replace('.', '_')
                caller_line = frame.f_lineno
                # Pass an instance_id is recommended
                instance_id = kwargs.pop('instance_id', '') + f"{caller_file}_{caller_line}"
                remote_group = kwargs.get('remote_group')
                # If cannot trust_remote_code, no callable and type can be used.
                check_unsafe(*args, **kwargs)

                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name:
                    if remote_group:
                        if device_mesh is None:
                            if _device_mesh is not None:
                                device_mesh = _device_mesh
                                kwargs[device_mesh_name] = device_mesh
                            else:
                                raise ValueError('Set device_mesh=DeviceMesh(...) to enable ray.')

                    if execute == 'first':
                        # Manually create a device_mesh because there is only one worker
                        device_mesh = DeviceMesh.from_sizes(dp_size=1)
                        kwargs[device_mesh_name] = device_mesh

                    if _device_group and remote_group:
                        # usually this happens in driver because worker does not has a valid _device_group
                        # this is used to print the device_group info, so pass the worker is ok
                        device_group = [dg for dg in _device_group if dg.name == remote_group][0]
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
                    seed = int(os.environ.get('TWINKLE_SEED', _seed))
                    determinism = int(os.environ.get('TWINKLE_FULL_DETERMINISM', int(_full_determinism)))
                    framework_util.seed_everything(seed, bool(determinism))
                    # Ensure torch.distributed is initialized inside Ray workers.
                    if os.environ.get('WORKER_NAME'):
                        # This will depress the warnings of megatron and reduce overhead
                        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
                        # This will prevent the unlimited threads started by torch
                        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
                        # Use parallelism mode of tokenizers
                        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
                    if not device_mesh_name:
                        # pop the device_mesh
                        args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                        kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                        # if any handler is passed to other component, lazy collect should be false
                        # for example, dataset pass to the dataloader
                    args, kwargs = _prepare_lazy_collect(args, kwargs)
                    kwargs.pop('remote_group', None) # component does not need this
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
                    kwargs_for_workers.pop('seed', None)
                    kwargs_for_workers.pop('full_determinism', None)

                    _actors = RayHelper.create_workers(
                        cls,
                        remote_group,
                        execute,
                        instance_id=instance_id,
                        seed=_seed,
                        full_determinism=_full_determinism,
                        *args,
                        **kwargs_for_workers
                    )
                    self._actors = _actors
                    if hasattr(cls, '__iter__'):
                        # wraps again, because ray uses cls method to call remote
                        cls.__iter__ = remote_function(dispatch=_dispatch,
                                                        execute=_execute,
                                                        collect='none')(__iter__)
                        cls.__next__ = remote_function(dispatch=_dispatch,
                                                        execute=_execute,
                                                        collect=_collect)(__next__)
                    for arg in (list(args) + list(kwargs.values())):
                        # keeps the device_mesh in the handler
                        if isinstance(arg, DeviceMesh):
                            self.device_mesh = arg
                            break

                self.remote_group = remote_group
                self._instance_id = instance_id
            else:
                raise ValueError(f'Unsupported mode: {_mode}')

        cls.__init__ = new_init
        return cls

    return decorator


def remote_function(dispatch: Union[Literal['slice', 'all', 'slice_dp'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable] = 'none',
                    sync: bool = False,
                    lazy_collect: Optional[bool] = None):
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
    """

    def decorator(func: Callable[..., T1]) -> Callable[..., T1]:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T1:
            device_mesh = getattr(self, 'device_mesh', None)
            if _mode == 'local':
                return func(self, *args, **kwargs)
            elif _mode == 'ray':
                check_unsafe(*args, **kwargs)
                if not hasattr(self, '_actors'):
                    # This is the worker
                    from ._ray import RayHelper
                    if RayHelper.has_ref(args, kwargs):
                        # In this case, driver dispatch is all, redispatch here
                        args, kwargs = RayHelper.do_get_and_collect(args, kwargs)
                        world_size = Platform.get_world_size()
                        rank = Platform.get_rank()
                        # Redispatch here
                        _workers_and_args = _dispatch_args(_get_workers([None]*world_size, execute), dispatch,
                                                           execute, device_mesh, args, kwargs)
                        _, args, kwargs = _workers_and_args[rank]
                    return func(self, *args, **kwargs)
                else:
                    # This is the driver
                    from ._ray import RayHelper
                    execute_method = RayHelper.execute_all_async if not sync else RayHelper.execute_all_sync
                    if RayHelper.has_ref(args, kwargs):
                        # If has any object-ref, dispatch in worker, because we don't know the structure in the ref.
                        # for example, dataloader returns any data list.
                        _workers_and_args = _dispatch_args(_get_workers(self._actors, execute), 'all',
                                                           execute, device_mesh, args, kwargs)
                    else:
                        # dispatch now
                        _workers_and_args = _dispatch_args(_get_workers(self._actors, execute), dispatch,
                                                           execute, device_mesh, args, kwargs)

                    result = execute_method(func.__name__, _workers_and_args)
                    # This is a result future, call it to get the actual result
                    result_func = RayHelper.do_get_and_collect_func(_collect_func, collect, result, device_mesh)
                    _local_lazy_collect = _lazy_collect
                    if func.__name__ == '__iter__':
                        # return self
                        return self
                    
                    if func.__name__ == '__len__':
                        # Get the first result and ignore the `lazy_collect`
                        import ray
                        return ray.get(result[0])

                    if func.__name__ == '__next__':
                        import ray
                        for _res in result:
                            # raise when any worker raises StopIteration
                            stop = ray.get(_res[1])
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
                    result = result_func if _local_lazy_collect else result_func()
                    return result
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        wrapper._execute = execute
        wrapper._collect = collect
        wrapper._dispatch = dispatch
        wrapper._lazy_collect = _lazy_collect
        wrapper._sync = sync
        return wrapper

    return decorator
