# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle infrastructure: process-level state, ``initialize``, and the remote-call re-exports.

This package keeps the process-level knobs (mode, lazy-collect policy, device groups/mesh,
notifier) together with :func:`initialize`, the sanctioned setup seam. The dispatch decorators
live in ``.dispatch`` and the continuous-work protocol in ``.continuous_work``; both read the
globals above through this module object at call time, so a later ``initialize`` is honoured.
"""
import json
import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from twinkle.notifier import Notifier
from twinkle.utils import DeviceGroup, DeviceMesh, Platform, get_logger, requires

from . import _torch_setup
from .collectors import collect_dp_and_flatten, collect_tensor_dict
from .continuous_work import _cw_barrier, _cw_drop_locked, _cw_unregister, _get_workers, _select_workers, \
    get_worker_id
from .dispatch import _collect_func, _dispatch_args, _get_device_mesh_param_name, _prepare_lazy_collect, \
    alive_node_ids, cluster_resources, kill_actors, node_ip, remote_class, remote_function, remote_map, \
    resolve_all, spawn_sidecar, stash

__all__ = ['collect_dp_and_flatten', 'collect_tensor_dict', 'get_device_mesh', 'get_device_placement', 'get_mode',
           'get_worker_id', 'initialize', 'remote_class', 'remote_function', 'remote_map',
           'node_ip', 'alive_node_ids', 'spawn_sidecar', 'resolve_all', 'kill_actors',
           'cluster_resources', 'stash']

logger = get_logger()

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

_lazy_collect = True

# Torch process-global init knobs (seed / determinism / tf32), bundled into one dict so a new knob never
# means a new initialize()/create_workers() argument or a new env var. Each only takes effect in the
# process that runs the compute, so the set is published to every Ray worker through the
# EXTRA_TORCH_KWARGS env var and re-applied there; _torch_setup owns the whole mechanism.
_extra_torch_kwargs: Dict[str, Any] = _torch_setup.normalize(None)

_device_group: Optional[List[DeviceGroup]] = None

_device_mesh = None

_notifier: Optional[Any] = None

_name: Optional[str] = None

_TWINKLE_NOTIFIER_ENV = 'TWINKLE_NOTIFIER'


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
               extra_torch_kwargs: Optional[Dict[str, Any]] = None,
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
        extra_torch_kwargs: Per-process PyTorch init knobs, bundled so a new one needs no new argument
            (see ``_torch_setup``): ``seed`` (seed everything with this), ``full_determinism`` (freeze
            the random, use determinism kernels, default `False`) and ``tf32`` (explicit matmul/conv
            precision; None leaves torch's default). Applied here and re-applied on every Ray worker
            through the EXTRA_TORCH_KWARGS env var, since each is a per-process global.
        groups: The device groups of the training.
        global_device_mesh: The global default device mesh.
        lazy_collect: Lazy collect all outputs in workers, default `True`.
        name: The name of this run.
        notifier: Optional callable (e.g. ``DingNotifier``) invoked with a
            single ``str`` message whenever any ``remote_function``-decorated
            method raises. The original exception is always re-raised; the
            notifier is best-effort and its own failures are swallowed.
    """
    global _mode, _device_group, _extra_torch_kwargs, _lazy_collect, _device_mesh, _name, _notifier
    assert mode in ('local', 'ray')
    _mode = mode
    _name = name
    _lazy_collect = lazy_collect
    _notifier = notifier
    if name is not None:
        os.environ['TWINKLE_NAME'] = name
    os.environ.setdefault('TWINKLE_SESSION_ID', str(os.getpid()))
    if notifier is not None and hasattr(notifier, 'to_dict'):
        os.environ[_TWINKLE_NOTIFIER_ENV] = json.dumps(notifier.to_dict())
    if global_device_mesh is not None:
        _device_mesh = global_device_mesh

    # Seed + set determinism/tf32 in THIS process, and publish the knob set for the Ray workers (which
    # inherit env, not this process's globals). In local/torchrun mode this process does the compute; in
    # ray mode the driver still runs its own light compute (e.g. collect), so both want it applied here.
    _extra_torch_kwargs = _torch_setup.apply(extra_torch_kwargs)
    os.environ[_torch_setup.EXTRA_TORCH_KWARGS_ENV] = _torch_setup.to_env_value(_extra_torch_kwargs)
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
                mesh_dim_names=('dp', ),
            )

        assert Platform.get_world_size() == _device_mesh.world_size
    else:
        requires('ray')
        from ._ray import RayHelper
        assert groups is not None
        # groups is needed for ray
        _device_group = groups
        RayHelper.initialize(
            nproc_per_node=nproc_per_node, ncpu_proc_per_node=ncpu_proc_per_node, device_groups=_device_group)


def get_device_mesh() -> Optional[DeviceMesh]:
    """The global default device mesh set by ``initialize`` (None before init).

    A ``@remote_class`` model receives this mesh automatically: its wrapped ``__init__`` injects it
    whenever the caller passes ``device_mesh=None`` (see the dispatch module's ``_new_init_body``).
    A model that builds *outside* that wrapper -- one overriding ``__init__`` rather than inheriting
    the wrapped base -- resolves its mesh through this accessor to uphold the same invariant: a
    constructed model holds a real mesh, so the InputProcessor / OptimizerGroup it builds get proper
    DP placement instead of silently degrading to single-rank loss normalisation and no metric
    aggregation.
    """
    return _device_mesh


def get_mode() -> Optional[Literal['local', 'ray']]:
    """The twinkle execution mode (``'local'`` or ``'ray'``) set by :func:`initialize` (None before init).

    Public accessor over the module-private ``_mode``. Callers outside twinkle (e.g. swift/dev) need to
    know whether cross-process isolation is in effect: under ``'ray'`` a ``forward_only`` serializes its
    inputs to a worker process, so a driver-side defensive copy of those inputs is redundant, while under
    ``'local'`` the compute shares this process and the copy is load-bearing. Reading ``_mode`` directly
    would reach into a private global, so this is the sanctioned seam.
    """
    return _mode


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
                lines.append(box_line(f'  │'))
                lines.append(box_line(f'  └─ Mesh Layout:'))
                lines.extend(render_mesh_grid(mesh.mesh, mesh.mesh_dim_names or []))

        lines.append(box_line(''))
        lines.extend(section_bottom())
        lines.append('')

    return '\n' + '\n'.join(lines)
