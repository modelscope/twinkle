# Copyright (c) ModelScope Contributors. All rights reserved.
import functools
import inspect
import json
import numpy as np
import os
import sys
import threading
from typing import Any, Callable, List, Literal, Optional, TypeVar, Union

from twinkle.notifier import Notifier, notify_exception
from twinkle.utils import DeviceGroup, DeviceMesh, Platform, check_unsafe, framework_util, get_logger, requires
from .collectors import collect_tensor_dict

logger = get_logger()

T1 = TypeVar('T1', bound=object)

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

_seed = 42

_lazy_collect = True

_full_determinism = False

_device_group: Optional[List[DeviceGroup]] = None

_device_mesh = None

_notifier: Optional[Any] = None

_name: Optional[str] = None

_TWINKLE_NOTIFIER_ENV = 'TWINKLE_NOTIFIER'


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


def _maybe_load_worker_notifier() -> None:
    """Lazily reconstruct notifier + name on ray workers from inherited env vars."""
    global _notifier, _name
    if _notifier is not None:
        return
    if _name is None:
        _name = os.environ.get('TWINKLE_NAME') or None
    raw = os.environ.get(_TWINKLE_NOTIFIER_ENV)
    if not raw:
        return

    candidate = Notifier.from_dict(json.loads(raw))
    if candidate is not None:
        _notifier = candidate


def initialize(mode: Literal['local', 'ray'] = 'local',
               nproc_per_node: int = 8,
               ncpu_proc_per_node: int = 8,
               seed: int = 42,
               full_determinism: bool = False,
               groups: Optional[List[DeviceGroup]] = None,
               global_device_mesh: Optional[DeviceMesh] = None,
               lazy_collect: bool = True,
               name: Optional[str] = None,
               notifier: Optional[Any] = None):
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
        name: The name of this run.
        notifier: Optional callable (e.g. ``DingNotifier``) invoked with a
            single ``str`` message whenever any ``remote_function``-decorated
            method raises. The original exception is always re-raised; the
            notifier is best-effort and its own failures are swallowed.
    """
    global _mode, _device_group, _seed, _full_determinism, _lazy_collect, _device_mesh, _name, _notifier
    assert mode in ('local', 'ray')
    _mode = mode
    _name = name
    _full_determinism = full_determinism
    _lazy_collect = lazy_collect
    _notifier = notifier
    if name is not None:
        os.environ['TWINKLE_NAME'] = name
    os.environ.setdefault('TWINKLE_SESSION_ID', str(os.getpid()))
    if notifier is not None and hasattr(notifier, 'to_dict'):
        os.environ[_TWINKLE_NOTIFIER_ENV] = json.dumps(notifier.to_dict())
    if global_device_mesh is not None:
        _device_mesh = global_device_mesh

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
                )
            ]

        if _device_mesh is None:
            _device_mesh = DeviceMesh(
                device_type=Platform.device_prefix(),
                mesh=np.arange(Platform.get_world_size()),
                mesh_dim_names=('dp', ))

        assert Platform.get_world_size() == _device_mesh.world_size
    else:
        requires('ray')
        from ._ray import RayHelper
        assert groups is not None
        # groups is needed for ray
        _device_group = groups
        RayHelper.initialize(
            nproc_per_node=nproc_per_node, ncpu_proc_per_node=ncpu_proc_per_node, device_groups=_device_group)


def get_device_placement(device_group=None) -> str:
    """Get the device placement graph, can be used to show the training topology.

    Args:
        device_group: The device group of the training, default will use the global `device_group`.

    Returns:
        A string containing the training topology.
    """
    if device_group is None:
        device_group = _device_group

    if device_group is None:
        return 'No device group provided.'

    WIDTH = 80

    def box_line(content='', align='left', prefix='│', suffix='│'):
        inner_width = WIDTH - 4
        if align == 'center':
            text = content.center(inner_width)
        else:
            text = content.ljust(inner_width)
        return f'{prefix} {text} {suffix}'

    def header_box(title):
        return [
            '╔' + '═' * (WIDTH - 2) + '╗',
            box_line(title, align='center', prefix='║', suffix='║'),
            '╚' + '═' * (WIDTH - 2) + '╝',
        ]

    def section_top(title=''):
        lines = ['┌' + '─' * (WIDTH - 2) + '┐']
        if title:
            lines.append(box_line(f'◈ {title}', prefix='│', suffix='│'))
            lines.append('├' + '─' * (WIDTH - 2) + '┤')
        return lines

    def section_bottom():
        return ['└' + '─' * (WIDTH - 2) + '┘']

    def format_ranks(ranks):
        if isinstance(ranks, list):
            if len(ranks) <= 16:
                return str(ranks)
            return f'{ranks[:6]} ... {ranks[-3:]} ({len(ranks)} total)'
        return str(ranks)

    def render_mesh_grid(mesh_array, dim_names):
        """Render a compact mesh visualization."""
        lines = []

        if mesh_array.ndim == 1:
            mesh_array = mesh_array.reshape(1, -1)

        if mesh_array.ndim > 2:
            lines.append(box_line(f'    ⊞ High-dim mesh: shape={mesh_array.shape}'))
            return lines

        rows, cols = mesh_array.shape
        max_rows, max_cols = 6, 10
        show_rows, show_cols = min(rows, max_rows), min(cols, max_cols)

        cell_w = max(4, len(str(mesh_array.max())) + 2)

        header = '      ' + ''.join(f'{i:^{cell_w}}' for i in range(show_cols))
        if cols > max_cols:
            header += ' ⋯'
        lines.append(box_line(f'    {header}'))

        # Top border
        border = '      ╭' + '─' * (cell_w * show_cols + show_cols - 1) + '╮'
        lines.append(box_line(f'    {border}'))

        # Data rows
        for r in range(show_rows):
            row_data = '│'.join(f'{mesh_array[r, c]:^{cell_w}}' for c in range(show_cols))
            row_str = f'   {r:>2} │{row_data}│'
            if cols > max_cols:
                row_str += ' ⋯'
            lines.append(box_line(f'    {row_str}'))

        if rows > max_rows:
            lines.append(box_line(f"         {'⋮':^{cell_w * show_cols}}"))

        # Bottom border
        border = '      ╰' + '─' * (cell_w * show_cols + show_cols - 1) + '╯'
        lines.append(box_line(f'    {border}'))

        return lines

    # Build output
    lines = header_box('DEVICE PLACEMENT TOPOLOGY')
    lines.append('')

    for group in device_group:
        lines.extend(section_top(f'DeviceGroup: {group.name}'))
        lines.append(box_line(f'  ├─ Device Type : {group.device_type}'))
        lines.append(box_line(f'  └─ Ranks       : {format_ranks(group.ranks)}'))

        if not group._device_mesh:
            lines.append(box_line(''))
            lines.append(box_line('  (No device meshes configured)', align='center'))
        else:
            for mesh_name, mesh in group._device_mesh.items():
                lines.append(box_line(''))
                lines.append(box_line(f'  ┌─ DeviceMesh: {mesh_name}'))

                # Dimensions
                if mesh.mesh_dim_names:
                    dim_info = ' × '.join(f'{name}={size}' for name, size in zip(mesh.mesh_dim_names, mesh.mesh.shape))
                    lines.append(box_line(f'  │  Dimensions : {dim_info}'))

                # Active parallelism
                parallelism = []
                for dim in ['pp', 'dp', 'tp', 'ep', 'sp', 'cp', 'fsdp']:
                    ws = mesh._get_world_size_for_dim(dim)
                    if ws is not None and ws > 1:
                        parallelism.append(f'{dim.upper()}={ws}')

                if parallelism:
                    lines.append(box_line(f"  │  Parallelism: {', '.join(parallelism)}"))

                # Mesh layout
                lines.append(box_line('  │'))
                lines.append(box_line('  └─ Mesh Layout:'))
                lines.extend(render_mesh_grid(mesh.mesh, mesh.mesh_dim_names or []))

        lines.append(box_line(''))
        lines.extend(section_bottom())
        lines.append('')

    return '\n' + '\n'.join(lines)


def _get_workers(workers, execute):
    if execute == 'first':
        return [workers[0]]
    elif execute == 'all':
        return workers
    elif execute == 'peer':
        return workers[Platform.get_peer_index(len(workers))]
    else:
        raise ValueError(f'Unsupported execute method: {execute}')


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


def _cw_state(self, n_workers: int):
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
    if state is not None and len(state['load']) == n_workers:
        return state
    with _CW_CREATE_LOCK:
        # Re-read: another thread may have created it while this one waited.
        state = getattr(self, '_continous_work_state', None)
        if state is None or len(state['load']) != n_workers:
            state = {
                'lock': threading.Lock(),
                'load': [0] * n_workers,
                'inflight': {},  # method name -> list of pending object refs
            }
            self._continous_work_state = state
        return state


def _cw_barrier(self, current_func: str) -> None:
    """Drain every other method's in-flight work before proceeding.

    Best effort by construction: another thread may submit again the moment this
    returns. It removes the case this exists for -- a weight update or a sleep
    issued while generations are still running -- but it is not a global lock on
    the worker.
    """
    state = getattr(self, '_continous_work_state', None)
    if not state:
        return
    with state['lock']:
        others = {name: list(refs) for name, refs in state['inflight'].items() if name != current_func and refs}
    if not others:
        return
    import ray
    flat = [ref for refs in others.values() for ref in refs]
    logger.debug(f'continous_work barrier: {current_func} waits for {len(flat)} pending request(s) '
                 f'from {sorted(others)}')
    ray.get(flat)
    # They are finished now, so drop them instead of re-getting them on every
    # later call. The owning thread's own cleanup tolerates them being gone.
    with state['lock']:
        for name, refs in others.items():
            pending = state['inflight'].get(name)
            if pending is None:
                continue
            for ref in refs:
                if ref in pending:
                    pending.remove(ref)
            if not pending:
                state['inflight'].pop(name, None)


def _cw_object_refs(result) -> List[Any]:
    """Every ObjectRef inside a dispatch result, tuples included."""
    import ray
    refs = []
    for item in (result or []):
        for candidate in (item if isinstance(item, tuple) else (item, )):
            if isinstance(candidate, ray.ObjectRef):
                refs.append(candidate)
    return refs


def _cw_register(self, func_name: str, result) -> List[Any]:
    """Record a non-continuous call's refs so a later different method waits for it.

    Needed because a lazily collected method returns before its work finishes:
    ``receive_weights`` hands back a handle while the worker is still swapping
    weights, and with actor concurrency on, a sample issued right after would read
    them half written.
    """
    refs = _cw_object_refs(result)
    if not refs:
        return refs
    state = _cw_state(self, len(getattr(self, '_actors', ())) or 1)
    with state['lock']:
        state['inflight'].setdefault(func_name, []).extend(refs)
    return refs


def _cw_unregister(self, func_name: str, refs: List[Any]) -> None:
    state = getattr(self, '_continous_work_state', None)
    if not state or not refs:
        return
    with state['lock']:
        pending = state['inflight'].get(func_name)
        if pending is None:
            return
        for ref in refs:
            if ref in pending:
                pending.remove(ref)
        if not pending:
            state['inflight'].pop(func_name, None)


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


def _run_continous_work(self, func_name: str, execute_method, workers, args, kwargs, batch_len: int,
                        ray_get_timeout: Optional[float]):
    """Submit one call per chosen worker and return results in the caller's order.

    Submission happens under the lock so that picking a worker and charging it are
    one step -- several caller threads land here at once, and a split of the two
    would let them all pick the same idle worker. Waiting happens outside it.
    """
    import ray

    state = _cw_state(self, len(workers))
    submitted = []
    # The awaitable form, so the worker can hold several of these at once. Book-
    # keeping still uses the plain name, which is what callers and the barrier see.
    remote_name = _WORKER_ASYNC_PREFIX + func_name
    with state['lock']:
        plan = _cw_plan(len(workers), state['load'], batch_len)
        for worker_index, indices in enumerate(plan):
            if not indices:
                continue
            sub_args, sub_kwargs = _cw_sub_args(args, kwargs, indices, batch_len)
            ref = execute_method(remote_name, [(workers[worker_index], sub_args, sub_kwargs)])[0]
            submitted.append((worker_index, indices, ref))
        state['inflight'].setdefault(func_name, []).extend(ref for _, _, ref in submitted)

    try:
        ordered: List[Any] = [None] * batch_len
        for _, indices, ref in submitted:
            part = ray.get(ref, timeout=ray_get_timeout) if ray_get_timeout else ray.get(ref)
            if not isinstance(part, (list, tuple)) or len(part) != len(indices):
                raise TypeError(f'{func_name}: enable_continous_work needs one result per request, but a worker given '
                                f'{len(indices)} request(s) returned {type(part).__name__} of length '
                                f'{len(part) if isinstance(part, (list, tuple)) else "n/a"}.')
            for local_index, original_index in enumerate(indices):
                ordered[original_index] = part[local_index]
        return ordered
    finally:
        with state['lock']:
            pending = state['inflight'].get(func_name, [])
            for worker_index, indices, ref in submitted:
                state['load'][worker_index] -= len(indices)
                if ref in pending:
                    pending.remove(ref)
            if not pending:
                state['inflight'].pop(func_name, None)


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


def remote_class(execute: Literal['first', 'peer', 'all'] = 'all',
                 max_concurrency: Optional[int] = None):
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
                _maybe_load_worker_notifier()
                _new_init_body(self, _caller, *args, **kwargs)
            except Exception as _e:  # noqa: BLE001
                _tag_exc(_e, _caller)
                notify_exception(_notifier, _ctx, _e, _name)
                raise

        def _new_init_body(self, _caller, *args, **kwargs):
            if _mode == 'local':
                # Get the actual device_mesh
                device_mesh = _get_device_mesh_param(args, kwargs)
                if device_mesh_name and _device_group is not None:
                    if device_mesh is None:
                        # Local mode can safely assign the default device mesh
                        device_mesh = _device_mesh
                        kwargs[device_mesh_name] = _device_mesh
                    assert len(_device_group) == 1  # only one device group is allowed
                    _device_group[0]._device_mesh[self.__class__.__name__] = device_mesh
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
            elif _mode == 'ray':
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
                            if _device_mesh is not None:
                                device_mesh = _device_mesh
                                kwargs[device_mesh_name] = device_mesh
                            else:
                                raise ValueError('Set device_mesh=DeviceMesh(...) to enable ray.')

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
                        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
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
                    kwargs_for_workers.pop('seed', None)
                    kwargs_for_workers.pop('full_determinism', None)

                    _actors = RayHelper.create_workers(
                        cls,
                        remote_group,
                        execute,
                        instance_id=instance_id,
                        seed=_seed,
                        full_determinism=_full_determinism,
                        max_concurrency=max_concurrency,
                        *args,
                        **kwargs_for_workers)
                    self._actors = _actors
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
                raise ValueError(f'Unsupported mode: {_mode}')

        cls.__init__ = new_init
        return cls

    return decorator


def remote_function(dispatch: Union[Literal['slice', 'all', 'slice_dp', 'last_pp_first'], Callable] = 'slice',
                    execute: Literal['first', 'peer', 'all'] = 'all',
                    collect: Union[Literal['none', 'flatten', 'mean', 'sum', 'first', 'last_pp'], Callable] = 'none',
                    sync: bool = False,
                    lazy_collect: Optional[bool] = None,
                    timeout: Optional[float] = None,
                    enable_continous_work: bool = False):
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
        timeout: Timeout in seconds for ray.get() when collecting results. Instance attribute ``_ray_get_timeout`` overrides this.
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
    """ # noqa

    def decorator(func: Callable[..., T1]) -> Callable[..., T1]:

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T1:
            _ctx = f'{type(self).__name__}.{func.__name__}'
            # Only capture caller on driver side; worker frames are Ray internals
            _caller = _capture_caller() if hasattr(self, '_actors') else None
            if _caller:
                _ctx = f'{_ctx} <- {_caller}'
            try:
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
                            _workers_and_args = _dispatch_args(
                                _get_workers([None] * world_size, execute), dispatch, execute, device_mesh, args,
                                kwargs)
                            _, args, kwargs = _workers_and_args[rank]
                        return func(self, *args, **kwargs)
                    else:
                        # This is the driver
                        from ._ray import RayHelper
                        execute_method = RayHelper.execute_all_async if not sync else RayHelper.execute_all_sync
                        # Only classes whose workers run methods side by side need
                        # this; elsewhere Ray already orders calls per actor.
                        _concurrent_actor = bool(getattr(self, '_max_concurrency', None))
                        if _concurrent_actor:
                            # Every method waits here, not just the continuous ones:
                            # the point is to keep a weight update or a sleep from
                            # reaching a worker that still has generations running.
                            _cw_barrier(self, func.__name__)
                        if enable_continous_work and not RayHelper.has_ref(args, kwargs):
                            assert not sync, (f'{func.__name__}: enable_continous_work cannot be used with sync=True, '
                                             'which exists for collectives that must run in lock step.')
                            _workers = _get_workers(self._actors, execute)
                            _batch_len = _cw_batch_len(args, kwargs)
                            if _batch_len:
                                return _run_continous_work(self, func.__name__, execute_method, _workers, args, kwargs,
                                                           _batch_len,
                                                           getattr(self, '_ray_get_timeout', None) or timeout)
                        if RayHelper.has_ref(args, kwargs):
                            # If has any object-ref, dispatch in worker, because we don't know the structure in the ref.
                            # for example, dataloader returns any data list.
                            _workers_and_args = _dispatch_args(
                                _get_workers(self._actors, execute), 'all', execute, device_mesh, args, kwargs)
                        else:
                            # dispatch now
                            _workers_and_args = _dispatch_args(
                                _get_workers(self._actors, execute), dispatch, execute, device_mesh, args, kwargs)

                        result = execute_method(func.__name__, _workers_and_args)
                        # Tracked from here so that a different method called next
                        # waits for this one. It matters most for the lazily
                        # collected methods, which return while the worker is still
                        # busy.
                        _tracked_refs = _cw_register(self, func.__name__, result) if _concurrent_actor else []
                        # This is a result future, call it to get the actual result
                        _rgt = getattr(self, '_ray_get_timeout', None) or timeout
                        result_func = RayHelper.do_get_and_collect_func(
                            _collect_func, collect, result, device_mesh, timeout=_rgt)
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
                        if _local_lazy_collect:
                            _orig_result_func = result_func

                            @functools.wraps(_orig_result_func)
                            def _notifying_result_func(*rargs, **rkwargs):
                                try:
                                    return _orig_result_func(*rargs, **rkwargs)
                                except Exception as _e:  # noqa
                                    _tag_exc(_e, _caller)
                                    notify_exception(_notifier, _ctx, _e, _name)
                                    raise
                                finally:
                                    _cw_unregister(self, func.__name__, _tracked_refs)

                            for _attr in ('_futures', ):
                                if hasattr(_orig_result_func, _attr):
                                    setattr(_notifying_result_func, _attr, getattr(_orig_result_func, _attr))
                            return _notifying_result_func
                        try:
                            return result_func()
                        finally:
                            _cw_unregister(self, func.__name__, _tracked_refs)
                else:
                    raise NotImplementedError(f'Unsupported mode {_mode}')
            except StopIteration:
                raise
            except Exception as _e:  # noqa: BLE001
                _tag_exc(_e, _caller)
                notify_exception(_notifier, _ctx, _e, _name)
                raise

        wrapper._execute = execute
        wrapper._collect = collect
        wrapper._dispatch = dispatch
        wrapper._lazy_collect = _lazy_collect
        wrapper._sync = sync
        wrapper._enable_continous_work = enable_continous_work
        if enable_continous_work:
            # Attached to the class by remote_class, and called instead of this
            # method when the driver routes requests worker by worker.
            wrapper._worker_async_companion = _make_worker_async_companion(func, wrapper)
        return wrapper

    return decorator
