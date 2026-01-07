import functools
import inspect
import os
from typing import Literal, List, Optional, Union, Callable
from typing import TypeVar
from types import MethodType
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

_lazy_collect = True

_full_determinism = False

_device_group: Optional[List[DeviceGroup]] = [
    DeviceGroup(
        name='default',
        ranks=list(range(Platform.get_world_size())),
        device_type=Platform.get_platform().device_prefix(),
    )
]

_device_mesh = DeviceMesh(
    device_type=Platform.get_platform().device_prefix(),
    mesh=np.arange(Platform.get_world_size()),
    mesh_dim_names=('dp',)
)

_remote_components: dict = {}


def initialize(mode: Literal['local', 'ray'] = 'local',
               nproc_per_node: Optional[int] = 8,
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
        nproc_per_node: The GPU count per node.
        seed: Seed everything with this.
        full_determinism: Freeze the random, use determinism kernels, default `False`.
        groups: The device groups of the training.
        global_device_mesh: The global default device mesh.
        lazy_collect: Lazy collect all outputs in workers, default `True`.
    """
    global _mode, _device_group, _nproc_per_node, _seed, _full_determinism, _lazy_collect, _device_mesh
    assert mode in ('local', 'ray')
    _mode = mode
    _full_determinism = full_determinism
    _lazy_collect = lazy_collect
    if global_device_mesh is not None:
        _device_mesh = global_device_mesh
    if seed is not None:
        _seed = seed
        framework_util.seed_everything(seed, full_determinism)
    if _mode == 'ray':
        requires('ray')
        if groups is not None:
            _device_group = groups
        _nproc_per_node = nproc_per_node
        RayHelper.initialize(nproc_per_node=_nproc_per_node, device_groups=_device_group)


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

        # Column headers
        col_label = dim_names[-1].upper() if dim_names else "COL"
        row_label = dim_names[0].upper() if len(dim_names) >= 2 else "ROW"

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
                    if ws > 1:
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
        if isinstance(result, list) and len(result) == 1:
            return result[0]
        else:
            return result
    elif method == 'flatten':
        flatten = [item for sublist in result for item in sublist]
        if isinstance(result[0], np.ndarray):
            return np.array(flatten)
        return type(result[0])(flatten)
    elif method == 'avg':
        return np.mean(result)
    elif method == 'sum':
        return np.sum(result)
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
        # if device_mesh is not None:
            # TODO this may occurs error when remote calls remote
            # Comment this because remote_class supports `first``
            # assert device_mesh.world_size == len(workers)
        length = len(workers) if not device_mesh else device_mesh.data_parallel_world_size
        length = min(length, len(workers))
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


def _prepare_lazy_collect(args, kwargs):
    # if a worker received an actor handle, lazy collect should be false to prevent any outer function receives
    # an object ref
    if not RayHelper.is_worker():
        return args, kwargs
    for arg in list(args) + list(kwargs.values()):
        if hasattr(arg, '_actors'):
            arg._lazy_collect = False
    return args, kwargs

def remote_class(execute: Literal['first', 'peer', 'all'] = 'peer'):
    """Patch each class used in remote clusters with this decorator."""

    def decorator(cls):
        # Get device mesh parameter name
        device_mesh_name = _get_device_mesh_param_name(cls.__init__)
        init_method = cls.__init__

        @functools.wraps(init_method)
        def new_init(self, *args, **kwargs):
            if _mode == 'local':
                # Get the actual device_mesh
                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name:
                    if device_mesh is None:
                        device_mesh = _device_mesh
                        kwargs[device_mesh_name] = _device_mesh
                    assert len(_device_group) == 1
                    _device_group[0]._device_mesh[self.__class__.__name__] = device_mesh
                    init_method(self, *args, **kwargs)
                else:
                    args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                    kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                    init_method(self, *args, **kwargs)
            elif _mode == 'ray':
                from .ray import RayHelper

                # In case the same class created twice in the same device group
                frame = inspect.currentframe().f_back
                caller_file = frame.f_code.co_filename.replace(os.sep, '_').replace('.', '_')
                caller_line = frame.f_lineno
                instance_id = kwargs.pop('instance_id', '') + f"{caller_file}_{caller_line}"
                remote_group = kwargs.pop('remote_group', None)
                if not remote_group and len(_device_group) == 1 and not RayHelper.is_worker():
                    # To the default group
                    remote_group = _device_group[0].name
                check_unsafe(*args, **kwargs)

                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh is None and device_mesh_name:
                    device_mesh = _device_mesh
                    kwargs[device_mesh_name] = _device_mesh

                if remote_group and device_mesh_name:
                    device_group = [dg for dg in _device_group if dg.name == remote_group][0]
                    device_group._device_mesh[self.__class__.__name__] = device_mesh

                def __iter__(_self):
                    _iter = _self.__iter_origin__()
                    assert _iter is not _self
                    _self._iter = _iter

                def __next__(_self):
                    return next(_self._iter)

                if (not remote_group) or os.environ.get('CLUSTER_NAME') == remote_group:
                    # remote_group is None when it's worker and remote_group not passed through arguments
                    # Seed when a remote class is created.
                    framework_util.seed_everything(int(os.environ['TWINKLE_SEED']),
                                                    bool(int(os.environ['TWINKLE_FULL_DETERMINISM'])))
                    if not device_mesh_name:
                        args = [arg for arg in args if not isinstance(arg, DeviceMesh)]
                        kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, DeviceMesh)}
                    from ray.actor import ActorHandle
                    args, kwargs = _prepare_lazy_collect(args, kwargs)
                    init_method(self, *args, **kwargs)
                else:
                    if hasattr(cls, '__iter__'):
                        _dispatch = self.__iter__._dispatch
                        _execute = self.__iter__._execute
                        _collect = self.__iter__._collect

                    if hasattr(cls, '__iter__'):
                        cls.__iter_origin__ = cls.__iter__
                        cls.__iter__ = __iter__
                        cls.__next__ = __next__

                    # Create remote workers
                    _actors = RayHelper.create_workers(cls,
                                                       remote_group,
                                                       execute,
                                                       instance_id=instance_id,
                                                       seed=_seed,
                                                       full_determinism=_full_determinism,
                                                       *args, **kwargs)
                    self._actors = _actors
                    if hasattr(cls, '__iter__'):
                        cls.__iter__ = remote_function(dispatch=_dispatch,
                                                        execute=_execute,
                                                        collect='none')(__iter__)
                        cls.__next__ = remote_function(dispatch=_dispatch,
                                                        execute=_execute,
                                                        collect=_collect)(__next__)
                    for arg in (list(args) + list(kwargs.values())):
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


def remote_function(dispatch: Union[Literal['slice', 'all'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'avg', 'sum'], Callable] = 'none'):
    """Patch each method called from remote(which class should be decorated with `remote_class`) with this decorator.

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
                    return func(self, *args, **kwargs)
                else:
                    from .ray import RayHelper
                    args, kwargs = RayHelper.do_get_and_collect(args, kwargs)
                    _workers_and_args = _dispatch_args(_get_workers(self._actors, execute), dispatch,
                                                       execute, device_mesh, args, kwargs)
                    result = RayHelper.execute_all_async(func.__name__, _workers_and_args)
                    result_func = RayHelper.do_get_and_collect_func(_collect_func, collect, result)
                    lazy_collect = _lazy_collect
                    if hasattr(self, '_lazy_collect'):
                        lazy_collect = self._lazy_collect
                    result = result_func if lazy_collect else result_func()
                    if func.__name__ == '__iter__':
                        return self
                    else:
                        return result
            else:
                raise NotImplementedError(f'Unsupported mode {_mode}')

        wrapper._execute = execute
        wrapper._collect = collect
        wrapper._dispatch = dispatch
        wrapper._lazy_collect = _lazy_collect
        return wrapper

    return decorator
