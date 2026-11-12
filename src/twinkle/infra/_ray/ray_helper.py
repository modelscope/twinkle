# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import threading
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from twinkle import Platform, find_free_port, find_node_ip, requires
from .. import _torch_setup
from .resource_manager import ResourceManager

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
        self._result = None  # Cache collected results
        self._timeout = timeout

    def _get_result(self):
        """Internal method to lazily collect and cache results"""
        import ray
        if self._result is None:
            result = []
            for future in self._futures:
                if isinstance(future, ray.ObjectRef):
                    result.append(ray.get(future, timeout=self._timeout))
                else:
                    result.append(future)
            self._result = self._collect_func(self._method, result, device_mesh=self.device_mesh)
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
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        return len(ready) == len(refs)

    def add_done_callback(self, fn: Callable[['LazyCollect'], Any]):
        """Call ``fn(self)`` once every referenced task has finished, without polling.

        Each ``ObjectRef.future()`` reports its own completion; the last one fires ``fn`` on Ray's
        callback thread, so ``fn`` must not block (hand the handle to a queue and read it there).
        A failed task also completes; the exception surfaces at :meth:`result`.
        """
        import ray
        refs = [future for future in self._futures if isinstance(future, ray.ObjectRef)]
        if not refs:
            fn(self)
            return
        remaining = [len(refs)]
        lock = threading.Lock()

        def _on_ref_done(_):
            with lock:
                remaining[0] -= 1
                last = remaining[0] == 0
            if last:
                fn(self)

        for ref in refs:
            ref.future().add_done_callback(_on_ref_done)

    def __iter__(self):
        """Support iteration: automatically collect results then iterate"""
        return iter(self._get_result())

    def __len__(self):
        """Support len() function"""
        return len(self._get_result())

    def __getitem__(self, key):
        """Index into the collected value (a dict key or a list index/slice); materialises it.

        Together with :meth:`get` / :meth:`__contains__` this lets the SAME subscript syntax read a
        lazy remote return and a local plain dict/list -- a caller writes ``out['logps']`` and works
        in both deployment forms without a shape branch.
        """
        return self._get_result()[key]

    def get(self, key, default=None):
        """Dict-style ``get`` over the collected value; materialises it (missing key -> ``default``)."""
        return self._get_result().get(key, default)

    def __contains__(self, key):
        """Membership over the collected value; materialises it."""
        return key in self._get_result()


class RayHelper:

    resource_manager: Optional[ResourceManager] = None

    _registry = None

    _remote_components: Dict[str, Any] = {}

    #: Every actor this process created, so ``shutdown`` can terminate them.
    _created_actors: List[Any] = []

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

            def occupied(self) -> List[Tuple[str, Any, Tuple[int, ...]]]:
                """``(group, slot, ranks)`` of every replica of every pool."""
                return [(member['group'], member.get('slot'), tuple(member.get('ranks') or ()))
                        for pool in self.pools.values() for member in pool.values()]

            def all_actors(self) -> List[Any]:
                return [member['actor'] for pool in self.pools.values() for member in pool.values()]

            def clear(self):
                self.config.clear()
                self.pools.clear()

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
            ray.init(ignore_reinit_error=True, resources=resources or None, num_gpus=num_gpus)

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
    def shutdown():
        """Terminate every actor this run created, release the placement groups and stop Ray."""
        import ray
        if not RayHelper.ray_inited():
            return
        actors = list(RayHelper._created_actors)
        if RayHelper._registry is not None:
            try:
                actors.extend(ray.get(RayHelper._registry.all_actors.remote()))
            except Exception:  # noqa
                pass
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:  # noqa
                pass
        RayHelper._created_actors.clear()
        if RayHelper.resource_manager is not None:
            RayHelper.resource_manager.destroy_placement_group()
            RayHelper.resource_manager = None
        if RayHelper._registry is not None:
            try:
                ray.get(RayHelper._registry.clear.remote())
                ray.kill(RayHelper._registry)
            except Exception:  # noqa
                pass
            RayHelper._registry = None
        ray.shutdown()

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
    def group_info(group: str) -> Dict[str, Any]:
        """Device type, worker width and placements of one device group, from any process."""
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
        import ray
        return isinstance(value, ray.ObjectRef) or isinstance(value, LazyCollect)

    @staticmethod
    def _resolve_arg_refs(value: Any) -> Any:
        """Materialise remote handles (ObjectRef / LazyCollect) anywhere inside an argument value.

        Ray dereferences only refs passed at the TOP level of an actor call; a handle nested in a
        list element, a dict value or a dataclass field rides through to the worker as a handle.
        This is where every one of them turns back into a value, so a component method body reads
        its arguments as plain single-process data.
        """
        import ray
        from dataclasses import fields, is_dataclass, replace
        if isinstance(value, ray.ObjectRef):
            return ray.get(value)
        if isinstance(value, LazyCollect):
            return value()
        if isinstance(value, list):
            return [RayHelper._resolve_arg_refs(item) for item in value]
        if isinstance(value, tuple):
            resolved = tuple(RayHelper._resolve_arg_refs(item) for item in value)
            return type(value)(*resolved) if hasattr(type(value), '_fields') else resolved
        if isinstance(value, dict):
            return {key: RayHelper._resolve_arg_refs(item) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{field.name: RayHelper._resolve_arg_refs(getattr(value, field.name))
                                     for field in fields(value) if field.init})
        return value

    @staticmethod
    def do_get_and_collect(args, kwargs):
        """Materialise every remote handle the arguments carry, wherever it is nested."""
        new_args = [RayHelper._resolve_arg_refs(arg) for arg in args]
        new_kwargs = {key: RayHelper._resolve_arg_refs(value) for key, value in kwargs.items()}
        return new_args, new_kwargs

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
    def create_workers(worker_cls: Type[T],
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
        worker_cls = RayHelper._get_remote_component(worker_cls)
        key = f'{group}-{worker_cls.__class__.__name__}-{instance_id}'
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
            env_vars.update({
                'WORLD_SIZE': str(world_size),
                'RANK': str(index),
                'LOCAL_RANK': str(0),
                'CLUSTER_NAME': group,
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

            # Ray serialises runtime_env to JSON; a callable setup hook is rejected by Ray 2.58
            # ("Object of type function is not JSON serializable"). Pass the import path instead.
            runtime_env = RuntimeEnv(
                env_vars=env_vars, worker_process_setup_hook='twinkle.utils.loader.run_worker_setup_hook')
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
                for created in workers:
                    ray.kill(created, no_restart=True)
                raise
            workers.append(worker)
        RayHelper._created_actors.extend(workers)
        return workers
