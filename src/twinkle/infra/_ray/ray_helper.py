# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
import threading
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from twinkle import Platform, find_free_port, find_node_ip, get_logger, requires
from .. import _torch_setup
from .resource_manager import ResourceManager, _allocation_address


def worker_setup_hook() -> None:
    """Publish this worker's initialization address before loading startup plugins or child processes."""
    os.environ['TWINKLE_NODE_IP'] = _allocation_address()
    from twinkle.utils.loader import run_worker_setup_hook

    run_worker_setup_hook()

T = TypeVar('T')

#: Registry key holding every device group's placements, published by the process that ran
#: ``initialize`` so that any other process (a driver actor growing a sampler pool) can place workers.
_GROUPS_KEY = '__twinkle_groups__'
#: Registry key holding the placement of every allocated physical rank (lending trainer ranks).
_RANKS_KEY = '__twinkle_rank_placements__'


class LazyCollect:
    """A remote call's pending result, addressed through the Future protocol.

    Holds the per-worker refs of one dispatched call and collects them on demand. ``result`` (and
    the callable form) materialise the collected value; ``done``/``ready`` probe completion without
    fetching any payload, so a driver can time control-plane side effects (release a pin, flip a
    window) to the true end of a call while the payload keeps flowing as a reference to a
    downstream actor. A FAILED task also reports ready (``ray.wait`` surfaces completion, not
    success); the exception is raised at the real collect.
    """

    def __init__(self, futures, method, collect_func, device_mesh, timeout=None):
        self._futures = futures
        self._method = method
        self._collect_func = collect_func
        self.device_mesh = device_mesh
        self._result = None
        self._has_result = False
        self._timeout = timeout
        self._completion_ref = None
        self._completion_future = None

    def __getstate__(self):
        # Completion waiters belong to this process, not to a serialized handle.
        state = self.__dict__.copy()
        state['_completion_future'] = None
        return state

    def _get_result(self, fetched: Optional[Dict[Any, Any]] = None):
        """Collect and cache the result; ``fetched`` holds ref values a caller already got in one batch."""
        import ray
        if not self._has_result:
            values = dict(fetched or {})
            pending = list(dict.fromkeys(ref for ref in self._futures
                                         if isinstance(ref, ray.ObjectRef) and ref not in values))
            if pending:
                values.update(zip(pending, ray.get(pending, timeout=self._timeout)))
            result = [values[ref] if isinstance(ref, ray.ObjectRef) else ref for ref in self._futures]
            self._result = self._collect_func(self._method, result, device_mesh=self.device_mesh)
            self._has_result = True
        return self._result

    def __call__(self):
        """Lazily collect results, support repeated calls (with caching)"""
        return self._get_result()

    def result(self, timeout=None):
        """Materialise the collected value; an explicit timeout overrides the declared one."""
        if timeout is not None:
            self._timeout = timeout
        return self._get_result()

    def exception(self, timeout=None):
        """The raised exception if the call failed, else ``None`` (Future protocol)."""
        try:
            self.result(timeout)
            return None
        except StopIteration:
            raise
        except Exception as exc:  # noqa
            return exc

    def done(self) -> bool:
        """Whether every referenced task has finished (Future protocol alias of :meth:`ready`)."""
        return self.ready()

    def ready(self) -> bool:
        """Whether every referenced task has finished, WITHOUT fetching any payload.

        A control-plane probe only: it exists so a driver can time a side effect (e.g. releasing an
        adapter pin) to the true completion of a lazily-collected call, while the payload keeps
        flowing as a reference to a downstream actor. ``ray.wait(timeout=0)`` inspects task status and
        never materialises the result, so ``_result`` stays untouched and no data returns to the
        caller. A FAILED task also reports ready (``ray.wait`` surfaces completion, not success); the
        exception is raised later, at the real :meth:`__call__` collect.
        """
        import ray
        refs = [future for future in self._futures if isinstance(future, ray.ObjectRef)]
        if not refs:
            return True
        refs = list(dict.fromkeys(refs))
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0, fetch_local=False)
        return len(ready) == len(refs)

    def add_done_callback(self, fn: Callable[['LazyCollect'], Any]):
        """Call ``fn(self)`` on completion; only a payload-free status token reaches the caller."""
        import ray
        refs = list(dict.fromkeys(ref for ref in self._futures if isinstance(ref, ray.ObjectRef)))
        if not refs:
            fn(self)
            return
        if self._completion_ref is None:
            remote = RayHelper._get_remote_component(_completion_token)
            self._completion_ref = remote.options(num_cpus=0).remote(refs)
        if self._completion_future is None:
            self._completion_future = self._completion_ref.future()
        self._completion_future.add_done_callback(lambda _: fn(self))

    def _project(self, operation, *arguments):
        """Project the return value remotely; the caller retains only the selected reference."""
        remote = RayHelper._get_remote_component(_project_collected)
        ref = remote.options(num_cpus=0).remote(
            self._futures, self._method, self._collect_func, self.device_mesh, operation, arguments,
            self._timeout)
        return LazyCollect([ref], 'none', self._collect_func, self.device_mesh, timeout=self._timeout)

    def __iter__(self):
        kind, structure = self._project('structure').result()
        if kind == 'mapping':
            return iter(structure)
        return (self[index] for index in range(structure))

    def __len__(self):
        return self._project('len').result()

    def __getitem__(self, key):
        return self._project('getitem', key)

    def get(self, key, default=None):
        return self._project('get', key, default)

    def __contains__(self, key):
        return self._project('contains', key).result()


def _completion_token(refs):
    """Wait for task completion without fetching any source payload."""
    import ray
    if refs:
        ray.wait(refs, num_returns=len(refs), fetch_local=False)


def _project_collected(futures, method, collect_func, device_mesh, operation, arguments, timeout):
    """Project a collected value without assuming any domain-specific fields."""
    import operator
    import ray
    from collections.abc import Mapping

    value = LazyCollect(futures, method, collect_func, device_mesh, timeout=timeout).result()
    # Open only the container; unselected stashed channels must remain references.
    while RayHelper._is_handle(value):
        value = value.result() if isinstance(value, LazyCollect) else ray.get(value, timeout=timeout)
    if operation == 'structure':
        if isinstance(value, Mapping):
            return 'mapping', tuple(value)
        if isinstance(value, (str, bytes)):
            return 'scalar', None
        try:
            kind = 'tuple' if isinstance(value, tuple) else ('tensor' if hasattr(value, 'shape') else 'sequence')
            return kind, len(value)
        except TypeError:
            return 'scalar', None
    operations = {'getitem': operator.getitem, 'get': lambda value, key, default: value.get(key, default),
                  'contains': operator.contains, 'len': len}
    return operations[operation](value, *arguments)


class RayHelper:

    resource_manager: Optional[ResourceManager] = None

    _registry = None

    _remote_components: Dict[str, Any] = {}

    #: Every actor this process created, so ``shutdown`` can terminate them.
    _created_actors: List[Any] = []
    _actor_lifecycle_lock = threading.RLock()
    _released_worker_ids: set[str] = set()

    @staticmethod
    def _get_ray_custom_resources(device_groups) -> Dict[str, float]:
        """Return accelerator resources needed for local Ray startup.

        Ray discovers CUDA GPUs itself.  Other accelerators -- Ascend NPUs, and
        Kunlunxin XPUs whose runtime ships no nvidia-smi -- must be registered
        explicitly so that placement groups can request them.  Each platform
        answers for itself via ``Platform.ray_resources()``; the detected
        platform is authoritative because XPU runs declare their device type as
        the cuda-alike 'GPU'.
        """
        # ResourceManager supports one accelerator type per run.
        device_types = {group.device_type.upper() for group in device_groups} - {'CPU'}
        if not device_types:
            return {}
        return Platform.get_platform().ray_resources()

    @staticmethod
    def init_registry():
        if RayHelper._registry is not None:
            return

        import ray

        @ray.remote
        class WorkerRegistry:
            """A config center for global configs and the membership table of every worker pool.

            A pool entry is ``{worker_id: {'actor', 'state', 'group', 'slot', 'ranks'}}``; ``state`` is one of
            WARMING / ACTIVE / DRAINING. Ray runs this actor's methods one at a time, so each call is atomic.
            """

            def __init__(self):
                self.config = {}
                self.pools = {}

            def add_config(self, key: str, value: Any):
                self.config[key] = value

            def add_or_get(self, key: str, value: Any) -> Tuple[bool, Any]:
                """Add or get config, because ray is single threaded."""
                if key in self.config:
                    return self.config[key]
                self.config[key] = value
                return value

            def get_config(self, key: str):
                return self.config.get(key)

            def pool_get(self, key: str) -> Dict[str, Dict[str, Any]]:
                return dict(self.pools.get(key, {}))

            def pool_put(self, key: str, members: Dict[str, Dict[str, Any]]):
                self.pools.setdefault(key, {}).update(members)

            def pool_set_state(self, key: str, worker_ids: List[str], state: str):
                pool = self.pools.get(key, {})
                for worker_id in worker_ids:
                    if worker_id in pool:
                        pool[worker_id]['state'] = state

            def pool_remove(self, key: str, worker_ids: List[str]):
                pool = self.pools.get(key, {})
                for worker_id in worker_ids:
                    pool.pop(worker_id, None)

            def remove_actors(self, worker_ids: List[str]):
                """Drop these replicas from every pool (a shutdown only ever removes its own)."""
                for pool in self.pools.values():
                    for worker_id in worker_ids:
                        pool.pop(worker_id, None)

            def occupied(self) -> List[Tuple[str, Any, Tuple[int, ...]]]:
                """``(group, slot, ranks)`` of every replica of every pool."""
                return [(member['group'], member.get('slot'), tuple(member.get('ranks') or ()))
                        for pool in self.pools.values() for member in pool.values()]
        
        registry_name = os.environ.get('TWINKLE_RAY_REGISTRY_NAME', 'config_registry')
        registry_namespace = os.environ.get('TWINKLE_RAY_NAMESPACE')
        try:
            RayHelper._registry = ray.get_actor(registry_name, namespace=registry_namespace)
        except ValueError:
            try:
                RayHelper._registry = WorkerRegistry.options(
                    name=registry_name,
                    namespace=registry_namespace,
                    lifetime='detached',
                    # Preserve Ray's default CPU policy outside managed instances.
                    num_cpus=0.01 if registry_namespace else None).remote()
            except ValueError:
                RayHelper._registry = ray.get_actor(registry_name, namespace=registry_namespace)
        assert RayHelper._registry is not None

    @staticmethod
    def initialize(nproc_per_node: int, ncpu_proc_per_node: int, device_groups):
        """Initialize RayHelper.

        Args:
            nproc_per_node: How many processes in one node.
            ncpu_proc_per_node: How many cpu processes in one node.
            device_groups: The device groups to initialize.
        """
        requires('ray')
        import ray
        RayHelper.device_groups = device_groups
        if not RayHelper.ray_inited():
            # Connecting to an existing cluster (e.g. RAY_ADDRESS set for multi-node): custom
            # resources are declared at ``ray start`` time, and Ray rejects ``resources`` on connect.
            connecting = bool(os.environ.get('RAY_ADDRESS'))
            resources = None if connecting else RayHelper._get_ray_custom_resources(device_groups)
            # XPU registers the cuda-alike 'GPU' count, but 'GPU' is one of Ray's
            # default resources and ray.init() rejects it via `resources=`; route
            # it through `num_gpus` instead.
            num_gpus = None
            if resources:
                num_gpus = int(resources.pop('GPU')) if 'GPU' in resources else None
            # The worker setup hook has to be installed here, at the job level, and not on each
            # actor's runtime_env: Ray 2.56 accepts ``worker_process_setup_hook`` in a per-actor
            # ``RuntimeEnv`` and then silently never calls it. Measured with a three-arm probe on this
            # Ray -- per-actor only: the hook never ran; job level only: it ran; job level plus a
            # per-actor ``RuntimeEnv`` carrying ``env_vars`` (the shape ``create_workers`` builds): it
            # ran, in the same worker process that hosted the actor, and the per-actor ``env_vars``
            # still applied. Nothing warned in the first arm, so a plugin the driver imported reached
            # every actor as a module name no worker could import.
            ray.init(ignore_reinit_error=True,
                     resources=resources or None,
                     num_gpus=num_gpus,
                     runtime_env={'worker_process_setup_hook': 'twinkle.infra._ray.ray_helper.worker_setup_hook'})

        if RayHelper.resource_manager is None:
            # Resource manager initializes only once in the pipeline process.
            RayHelper.resource_manager = ResourceManager(nproc_per_node, ncpu_proc_per_node, device_groups)
        RayHelper.init_registry()
        manager = RayHelper.resource_manager
        RayHelper.add_config(_GROUPS_KEY, {
            config.name: {
                'device_type': config.device_type,
                'gpus_per_worker': getattr(config, 'gpus_per_worker', 1),
                'placements': manager.get_group(config.name),
            }
            for config in manager.group_configs
        })
        RayHelper.add_config(_RANKS_KEY, getattr(manager, 'rank_placements', {}))

    @staticmethod
    def release_workers(actors):
        """Release process-owned replicas; retain failed handles for a later cleanup attempt."""
        import ray
        from ..continuous_work import worker_id_of

        failures = []
        with RayHelper._actor_lifecycle_lock:
            for actor in tuple(actors):
                if actor not in RayHelper._created_actors:
                    if actor in actors:
                        actors.remove(actor)
                    continue
                try:
                    ray.kill(actor, no_restart=True)
                except BaseException as error:
                    failures.append(error)
                    get_logger().warning('Failed to release owned worker %s.', worker_id_of(actor), exc_info=True)
                else:
                    RayHelper._released_worker_ids.add(worker_id_of(actor))
                    if actor in actors:
                        actors.remove(actor)
                    RayHelper._created_actors.remove(actor)
            pending = list(RayHelper._released_worker_ids)
            if pending and RayHelper._registry is not None:
                try:
                    ray.get(RayHelper._registry.remove_actors.remote(pending))
                except BaseException as error:
                    failures.append(error)
                    get_logger().warning('Failed to remove released workers from the registry.', exc_info=True)
                else:
                    RayHelper._released_worker_ids.difference_update(pending)
        if failures:
            raise BaseExceptionGroup('Owned worker cleanup is incomplete', failures)

    @staticmethod
    def shutdown():
        """Release owned actors and allocations; retain the shared registry for other clients.

        Every step runs even when an earlier one fails; failures aggregate in one group
        so a single bad cleanup cannot skip the placement-group teardown or ray shutdown.
        """
        import ray
        if not RayHelper.ray_inited():
            return
        failures = []
        try:
            RayHelper.release_workers(list(RayHelper._created_actors))
        except BaseException as error:
            failures.append(error)
            get_logger().warning('Owned worker release failed during shutdown.', exc_info=True)
        try:
            if RayHelper.resource_manager is not None:
                RayHelper.resource_manager.destroy_placement_group()
                RayHelper.resource_manager = None
        except BaseException as error:
            failures.append(error)
            get_logger().warning('Placement group teardown failed during shutdown.', exc_info=True)
        RayHelper._registry = None
        RayHelper._remote_components.clear()
        try:
            ray.shutdown()
        except BaseException as error:
            failures.append(error)
            get_logger().warning('Ray shutdown failed.', exc_info=True)
        if failures:
            raise BaseExceptionGroup('RayHelper shutdown is incomplete', failures)

    @staticmethod
    def ray_inited():
        """Check if Ray is initialized."""
        try:
            import ray
        except ImportError:
            # not installed, not inited
            return False
        return ray.is_initialized()

    @staticmethod
    def is_worker():
        """Check if this process is the worker"""
        import ray
        return RayHelper.ray_inited() and ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE

    @staticmethod
    def execute_all_sync(method_name: str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]], *,
                         timeout=None):
        """Execute method and return results; ``timeout`` bounds the whole ``ray.get``."""
        import ray
        return ray.get(RayHelper.execute_all_async(method_name, workers_and_args), timeout=timeout)

    @staticmethod
    def execute_all_async(method_name: str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]]):
        """Execute method and return refs."""
        output = []
        for worker, args, kwargs in workers_and_args:
            output.append(getattr(worker, method_name).remote(*args, **kwargs))
        return output

    @staticmethod
    def _registry_call(name: str, *args):
        import ray
        RayHelper.init_registry()
        return ray.get(getattr(RayHelper._registry, name).remote(*args))

    @staticmethod
    def add_or_get_config(key: str, value: Any):
        return RayHelper._registry_call('add_or_get', key, value)

    @staticmethod
    def add_config(key: str, value: Any):
        RayHelper._registry_call('add_config', key, value)

    @staticmethod
    def get_config(key: str):
        return RayHelper._registry_call('get_config', key)

    @staticmethod
    def pool_get(key: str) -> Dict[str, Dict[str, Any]]:
        return RayHelper._registry_call('pool_get', key)

    @staticmethod
    def pool_put(key: str, members: Dict[str, Dict[str, Any]]):
        RayHelper._registry_call('pool_put', key, members)

    @staticmethod
    def pool_set_state(key: str, worker_ids: List[str], state: str):
        RayHelper._registry_call('pool_set_state', key, list(worker_ids), state)

    @staticmethod
    def pool_remove(key: str, worker_ids: List[str]):
        RayHelper._registry_call('pool_remove', key, list(worker_ids))

    @staticmethod
    def pool_occupied() -> List[Tuple[str, Any, Tuple[int, ...]]]:
        return RayHelper._registry_call('occupied')

    @staticmethod
    def group_info(group) -> Dict[str, Any]:
        """Resolve a named group or a device-address view of allocations already registered at initialization."""
        from twinkle.utils import DeviceGroup

        if isinstance(group, DeviceGroup):
            if not group.shared or not group.node_addresses:
                raise ValueError('A constructor DeviceGroup must be a shared view with explicit node addresses.')
            devices = group.visible_devices
            devices = list(devices) if isinstance(devices, (list, tuple)) else [devices]
            if len(devices) != len(group.node_addresses) * group.gpus_per_worker or any(
                    device is None for device in devices):
                raise ValueError('Each shared worker needs a node address and its visible device selectors.')
            manager = RayHelper.resource_manager
            groups = ({config.name: RayHelper.group_info(config.name) for config in manager.group_configs}
                      if manager is not None else RayHelper.get_config(_GROUPS_KEY) or {})
            allocations = [placement for info in groups.values()
                           if info['device_type'].upper() == group.device_type.upper()
                           for placement in info['placements']]
            placements = []
            for index, address in enumerate(group.node_addresses):
                selectors = devices[index * group.gpus_per_worker:(index + 1) * group.gpus_per_worker]
                matches = [placement for placement in allocations
                           if placement.get('node_ip') == address
                           and set(map(str, selectors)) <= set(map(str, placement.get('gpu_rank', ())))]
                if not matches:
                    raise ValueError(f'Shared devices {selectors!r} on {address!r} are not in an initialized allocation.')
                placements.append(dict(matches[0], gpu_rank=selectors))
            return {'device_type': group.device_type, 'gpus_per_worker': group.gpus_per_worker,
                    'placements': placements}
        manager = RayHelper.resource_manager
        if manager is not None:
            config = manager.get_config(group)
            return {
                'device_type': config.device_type,
                'gpus_per_worker': getattr(config, 'gpus_per_worker', 1),
                'placements': manager.get_group(group),
            }
        groups = RayHelper.get_config(_GROUPS_KEY) or {}
        if group not in groups:
            raise ValueError(f'No device group {group!r}; known groups: {sorted(groups)}')
        return groups[group]

    @staticmethod
    def rank_placements(ranks: List[int], width: int) -> List[Dict[str, Any]]:
        """Placements for explicit physical ranks, ``width`` consecutive ranks per worker."""
        manager = RayHelper.resource_manager
        table = getattr(manager, 'rank_placements', None) if manager is not None else None
        if table is None:
            table = RayHelper.get_config(_RANKS_KEY) or {}
        ranks = list(ranks)
        if len(ranks) % width:
            raise ValueError(f'{len(ranks)} ranks cannot be split into workers of {width} device(s).')
        placements = []
        for start in range(0, len(ranks), width):
            chunk = ranks[start:start + width]
            missing = [rank for rank in chunk if rank not in table]
            if missing:
                raise ValueError(f'Physical ranks {missing} are not allocated by any device group.')
            groups = {id(table[rank]['placement_group']) for rank in chunk}
            if len(groups) > 1:
                raise ValueError(f'Ranks {chunk} span multiple nodes; one worker must stay on one node.')
            placements.append({
                'gpu_rank': [table[rank]['gpu_rank'] for rank in chunk],
                'placement_group': table[chunk[0]]['placement_group'],
                'node_ip': table[chunk[0]]['node_ip'],
                'ranks': chunk,
            })
        return placements

    @staticmethod
    def _get_remote_component(component):
        """Avoid create remote component twice."""
        if component not in RayHelper._remote_components:
            import ray
            RayHelper._remote_components[component] = ray.remote(component)
        return RayHelper._remote_components[component]

    @staticmethod
    def get_master_id_port(placement_group):
        import ray

        @ray.remote
        def get_node_address():
            return find_node_ip(), find_free_port()

        ip, port = ray.get(get_node_address.options(placement_group=placement_group, num_cpus=0.01).remote())
        return ip, port

    @staticmethod
    def do_get_and_collect_func(collect_func: Callable,
                                method: Union[str, Callable],
                                futures,
                                device_mesh,
                                timeout=None):
        """Return the Future-protocol handle over one dispatched call's refs."""
        return LazyCollect(futures, method, collect_func, device_mesh, timeout=timeout)

    @staticmethod
    def _is_handle(value) -> bool:
        ray = sys.modules.get('ray')
        return isinstance(value, LazyCollect) or (ray is not None and isinstance(value, ray.ObjectRef))

    @staticmethod
    def _resolve_arg_refs(value: Any, resolved: Dict[Any, Any]) -> Any:
        """Materialise remote handles (ObjectRef / LazyCollect) anywhere inside an argument value.

        Ray dereferences only refs passed at the TOP level of an actor call; a handle nested in a
        list element, a dict value or a dataclass field rides through to the worker as a handle.
        This is where every one of them turns back into a value, so a component method body reads
        its arguments as plain single-process data. ``resolved`` maps each ObjectRef to its value,
        fetched beforehand in one batch (:meth:`do_get_and_collect`).
        """
        import ray
        from copy import copy
        from dataclasses import fields, is_dataclass
        if isinstance(value, ray.ObjectRef):
            return resolved.get(value, value)
        if isinstance(value, LazyCollect):
            return value._get_result(resolved)
        if isinstance(value, list):
            return [RayHelper._resolve_arg_refs(item, resolved) for item in value]
        if isinstance(value, tuple):
            items = tuple(RayHelper._resolve_arg_refs(item, resolved) for item in value)
            return type(value)(*items) if hasattr(type(value), '_fields') else items
        if isinstance(value, dict):
            return {key: RayHelper._resolve_arg_refs(item, resolved) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            result = copy(value)
            for field in fields(value):
                object.__setattr__(result, field.name,
                                   RayHelper._resolve_arg_refs(getattr(value, field.name), resolved))
            return result
        return value

    @staticmethod
    def _collect_arg_refs(value: Any, refs: List[Any]) -> None:
        """Append every ObjectRef nested in ``value``; a LazyCollect contributes its own refs."""
        import ray
        from dataclasses import fields, is_dataclass
        if isinstance(value, ray.ObjectRef):
            refs.append(value)
        elif isinstance(value, LazyCollect):
            # Fetched with the rest; its collect then reads them from ``resolved``.
            refs.extend(ref for ref in value._futures if isinstance(ref, ray.ObjectRef))
        elif isinstance(value, (list, tuple)):
            for item in value:
                RayHelper._collect_arg_refs(item, refs)
        elif isinstance(value, dict):
            for item in value.values():
                RayHelper._collect_arg_refs(item, refs)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                RayHelper._collect_arg_refs(getattr(value, field.name), refs)

    @staticmethod
    def anchor_refs(value: Any):
        """Anchor nested payload ObjectRefs so they outlive this process's local references.

        A stashed payload rides back as a reference nested in a return value. The process that
        receives it (e.g. a streaming driver reading ``rollout.step``) is only a *borrower*: once
        it drops its local Python reference -- a group cleared after delivery, a close drain, or
        the declarative teardown killing the actor -- Ray's reference-counting protocol can free
        the payload on its owner BEFORE a downstream consumer (the trainer's ``forward_backward``)
        retrieves it, surfacing as ``ReferenceCountingAssertionError`` (defect129). The race is
        intermittent because it depends on when the borrower's release reaches the owner relative
        to the consumer's borrow.

        Submitting a payload-free completion token over the refs registers an independent borrower
        session (the token task's argument references) that keeps the payload alive until consumed,
        without fetching any payload into this process. Returns the anchoring :class:`LazyCollect`
        -- the caller keeps it alive -- or ``None`` when ``value`` carries no reference.
        """
        import ray
        refs: List[Any] = []
        RayHelper._collect_arg_refs(value, refs)
        refs = [ref for ref in dict.fromkeys(refs) if isinstance(ref, ray.ObjectRef)]
        if not refs:
            return None
        anchor = LazyCollect(refs, 'none', lambda method, result, device_mesh=None: result, None)
        anchor.add_done_callback(lambda finished: None)
        return anchor

    @staticmethod
    def do_get_and_collect(args, kwargs, timeout=None):
        """Resolve nested handles in batches, sharing one timeout and fetching each reference once."""
        import ray
        from time import monotonic

        deadline = None if timeout is None else monotonic() + timeout
        resolved: Dict[Any, Any] = {}
        while RayHelper.has_ref(args, kwargs):
            refs: List[Any] = []
            for value in list(args) + list(kwargs.values()):
                RayHelper._collect_arg_refs(value, refs)
            pending = [ref for ref in dict.fromkeys(refs) if ref not in resolved]
            if pending:
                remaining = None if deadline is None else max(0.0, deadline - monotonic())
                resolved.update(zip(pending, ray.get(pending, timeout=remaining)))
            args = [RayHelper._resolve_arg_refs(arg, resolved) for arg in args]
            kwargs = {key: RayHelper._resolve_arg_refs(value, resolved) for key, value in kwargs.items()}
        return args, kwargs

    @staticmethod
    def has_ref(args, kwargs) -> bool:
        """Whether any argument carries a remote handle, at any nesting depth."""
        from dataclasses import fields, is_dataclass

        def _carries_ref(value) -> bool:
            if RayHelper._is_handle(value):
                return True
            if isinstance(value, (list, tuple)):
                return any(_carries_ref(item) for item in value)
            if isinstance(value, dict):
                return any(_carries_ref(item) for item in value.values())
            if is_dataclass(value) and not isinstance(value, type):
                return any(_carries_ref(getattr(value, field.name)) for field in fields(value))
            return False

        return any(_carries_ref(arg) for arg in args) or any(_carries_ref(value) for value in kwargs.values())

    @staticmethod
    def create_workers(*args, creation_timeout=None, **kwargs):
        """Create all replicas before awaiting their constructors, rolling back on any failure.

        The rollback releases only the handles THIS call produced: ``_create_workers``
        cleans up its own partial submissions, and a failed readiness wait releases the
        returned list -- never a slice of the global created-actors log, which could sweep
        in actors appended by other concurrent creators.
        """
        import ray
        with RayHelper._actor_lifecycle_lock:
            workers = None
            try:
                workers = RayHelper._create_workers(*args, **kwargs)
                ray.get([worker._twinkle_ready.remote() for worker in workers], timeout=creation_timeout)
                return workers
            except BaseException:
                if workers:
                    try:
                        RayHelper.release_workers(list(workers))
                    except BaseException:
                        get_logger().exception('Worker-construction rollback is incomplete.')
                raise

    @staticmethod
    def _create_workers(worker_cls: Type[T],
                       group: str,
                       execute: Literal['all', 'peer', 'first'],
                       *args,
                       instance_id,
                       extra_torch_kwargs: Optional[Dict[str, Any]] = None,
                       max_concurrency: Optional[int] = None,
                       placements: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> List[T]:
        """Create one actor per placement of ``group`` (or per explicit ``placements``)."""
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        info = RayHelper.group_info(group)
        # Workers added to a live pool get a unique name suffix: the initial ``key-<index>`` names are taken.
        suffix = None if placements is None else uuid.uuid4().hex[:8]
        if placements is None:
            placements = list(info['placements'])
        if not placements:
            return []
        # The class name must be read before ``_get_remote_component`` wraps it: afterwards it is
        # Ray's ActorClass, every actor would be named ``group-ActorClass-...`` and two handles
        # built from the same line would collide on the same name.
        class_name = worker_cls.__name__
        worker_cls = RayHelper._get_remote_component(worker_cls)
        group_name = group if isinstance(group, str) else group.name
        key = f'{group_name}-{class_name}-{instance_id}'
        if execute == 'peer':
            placements = placements[Platform.get_peer_index(len(placements))]
            ip, port = RayHelper.get_master_id_port(placements[0]['placement_group'])
            ip = RayHelper.add_or_get_config(key + '-ip', ip)
            port = RayHelper.add_or_get_config(key + '-port', port)
        else:
            if execute == 'first':
                placements = placements[:1]
            ip, port = RayHelper.get_master_id_port(placements[0]['placement_group'])

        device_type_upper = (info['device_type'] or '').upper()
        world_size = len(placements)
        workers = []
        for index, placement in enumerate(placements):
            worker_name = f'{key}-{index}' if suffix is None else f'{key}-{suffix}-{index}'
            env_vars = os.environ.copy()
            # Ray sets the local raylet PID on the destination node.
            env_vars.pop('RAY_RAYLET_PID', None)
            env_vars.update({
                'WORLD_SIZE': str(world_size),
                'RANK': str(index),
                'LOCAL_RANK': str(0),
                'CLUSTER_NAME': group_name,
                'TWINKLE_NODE_IP': placement['node_ip'],
                'WORKER_NAME': worker_name,
                'TWINKLE_MODE': 'ray',
                _torch_setup.EXTRA_TORCH_KWARGS_ENV: _torch_setup.to_env_value(extra_torch_kwargs),
            })
            if device_type_upper != 'CPU':
                env_vars[Platform.get_platform(device_type_upper).visible_device_env()] = ','.join(
                    str(r) for r in placement['gpu_rank'])
                env_vars['MASTER_ADDR'] = ip
                env_vars['MASTER_PORT'] = str(port)
                # Prevent Ray from overriding CUDA_VISIBLE_DEVICES set in runtime_env
                # This is critical for multi-GPU workers (gpus_per_worker > 1)
                env_vars.update(ResourceManager.noset_env())
            if max_concurrency is not None:
                # Read back in the worker to size the thread pool that runs
                # blocking continuous-work methods off the actor's event loop.
                env_vars['TWINKLE_ACTOR_MAX_CONCURRENCY'] = str(max_concurrency)

            # No ``worker_process_setup_hook`` here: Ray ignores that field in a per-actor runtime_env
            # and honours it only at the job level, where ``initialize`` installs it. This env_vars is
            # merged with the job-level runtime_env rather than replacing it, so the hook still runs in
            # these workers while each gets its own RANK / WORLD_SIZE / visible-device values.
            runtime_env = RuntimeEnv(env_vars=env_vars)
            # The placement group already reserves a worker's full accelerator allocation and the
            # visible-device env pins the actor to its ranks, so no fractional accelerator is requested:
            # Ray would translate the bundle's device id through the already narrowed visible list.
            worker_options = {
                'scheduling_strategy': PlacementGroupSchedulingStrategy(placement_group=placement['placement_group']),
                'name': worker_name,
                'namespace': os.environ.get('TWINKLE_RAY_NAMESPACE', 'default'),
                'runtime_env': runtime_env,
                'num_cpus': 0.01,
            }
            if max_concurrency is not None:
                worker_options['max_concurrency'] = max_concurrency
            if os.environ.get('TWINKLE_RAY_NAMESPACE'):
                worker_options.update(max_restarts=0, max_task_retries=0)
            try:
                worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
            except BaseException:
                # A failed submission rolls back only THIS call's already-created replicas;
                # the global created-actors log is left untouched for concurrent creators.
                if workers:
                    try:
                        RayHelper.release_workers(list(workers))
                    except BaseException:
                        get_logger().exception('Worker-construction rollback is incomplete.')
                raise
            workers.append(worker)
            RayHelper._created_actors.append(worker)
        return workers
