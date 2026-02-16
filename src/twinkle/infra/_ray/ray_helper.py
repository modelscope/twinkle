# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from twinkle import DeviceGroup, Platform, find_free_port, find_node_ip, requires
from .resource_manager import ResourceManager

T = TypeVar('T')


class RayHelper:

    resource_manager: Optional[ResourceManager] = None

    _registry = None

    _remote_components: Dict[str, Any] = {}

    @staticmethod
    def init_registry():
        if RayHelper._registry is not None:
            return

        import ray

        @ray.remote
        class WorkerRegistry:
            """A config center to store global configs"""

            def __init__(self):
                self.config = {}

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

            def clear(self):
                self.config.clear()

        try:
            RayHelper._registry = ray.get_actor('config_registry')
        except ValueError:
            try:
                RayHelper._registry = WorkerRegistry.options(name='config_registry', lifetime='detached').remote()
            except ValueError:
                RayHelper._registry = ray.get_actor('config_registry')
        assert RayHelper._registry is not None

    @staticmethod
    def initialize(nproc_per_node: int, ncpu_proc_per_node: int, device_groups: List[DeviceGroup]):
        """Initialize RayHelper.

        Args:
            nproc_per_node: How many processes in one node.
            ncpu_proc_per_node: How many cpu processes in one node.
            device_groups: The device groups to initialize.

        Returns:
            None
        """
        requires('ray')
        import ray
        RayHelper.device_groups = device_groups
        if not RayHelper.ray_inited():
            ray.init(ignore_reinit_error=True)

        if RayHelper.resource_manager is None:
            # Resource manager initializes only once in the pipeline process.
            RayHelper.resource_manager = ResourceManager(nproc_per_node, ncpu_proc_per_node, device_groups)
        RayHelper.init_registry()

    @staticmethod
    def teardown():
        """Teardown RayHelper."""
        if RayHelper.resource_manager is not None:
            RayHelper.resource_manager.destroy_placement_group()
            RayHelper.resource_manager = None

        if RayHelper._registry is not None:
            import ray
            try:
                ray.get(RayHelper._registry.clear.remote())
                ray.kill(RayHelper._registry)
            except:  # noqa
                pass
            RayHelper._registry = None

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
    def execute_all_sync(method_name: str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]]):
        """Execute method and return results."""
        import ray
        return ray.get(RayHelper.execute_all_async(method_name, workers_and_args))

    @staticmethod
    def execute_all_async(method_name: str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]]):
        """Execute method and return futures."""
        output = []
        for worker_and_args in workers_and_args:
            worker, args, kwargs = worker_and_args
            remote_call = getattr(worker, method_name)
            output.append(remote_call.remote(*args, **kwargs))
        return output

    @staticmethod
    def add_or_get_config(key: str, value: Any):
        import ray
        return ray.get(RayHelper._registry.add_or_get.remote(key, value))

    @staticmethod
    def add_config(key: str, value: Any):
        import ray
        ray.get(RayHelper._registry.add_config.remote(key, value))

    @staticmethod
    def get_config(key: str):
        import ray
        return ray.get(RayHelper._registry.get_config.remote(key))

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

        ip, port = ray.get(get_node_address.options(placement_group=placement_group).remote())
        return ip, port

    @staticmethod
    def do_get_and_collect_func(collect_func: Callable, method: Union[str, Callable], futures, device_mesh):
        """Return a callable to collect results in the workers."""

        class LazyCollect:

            def __init__(self, futures, method, collect_func, device_mesh):
                self._futures = futures
                self._method = method
                self._collect_func = collect_func
                self._is_lazy_collect = True
                self.device_mesh = device_mesh
                self._result = None  # Cache collected results

            def _get_result(self):
                """Internal method to lazily collect and cache results"""
                import ray
                if self._result is None:
                    result = []
                    for future in self._futures:
                        if isinstance(future, ray.ObjectRef):
                            result.append(ray.get(future))
                        else:
                            result.append(future)
                    self._result = self._collect_func(self._method, result, device_mesh=self.device_mesh)
                return self._result

            def __call__(self):
                """Lazily collect results, support repeated calls (with caching)"""
                return self._get_result()

            def __iter__(self):
                """Support iteration: automatically collect results then iterate"""
                return iter(self._get_result())

            def __len__(self):
                """Support len() function"""
                return len(self._get_result())

        return LazyCollect(futures, method, collect_func, device_mesh)

    @staticmethod
    def do_get_and_collect(args, kwargs):
        """Collect `LazyCollect` in each arg."""
        new_args = []
        for arg in args:
            if isinstance(arg, Callable) and getattr(arg, '_is_lazy_collect', False):
                arg = arg()
            new_args.append(arg)

        new_kwargs = {}
        for key in list(kwargs.keys()):
            value = kwargs[key]
            if isinstance(value, Callable) and getattr(value, '_is_lazy_collect', False):
                value = value()
            new_kwargs[key] = value
        return new_args, new_kwargs

    @staticmethod
    def has_ref(args, kwargs) -> bool:
        for arg in args:
            if isinstance(arg, Callable) and getattr(arg, '_is_lazy_collect', False):
                return True
        for key in list(kwargs.keys()):
            value = kwargs[key]
            if isinstance(value, Callable) and getattr(value, '_is_lazy_collect', False):
                return True
        return False

    @staticmethod
    def create_workers(worker_cls: Type[T],
                       group: str,
                       execute: Literal['all', 'peer', 'first'],
                       *args,
                       instance_id,
                       seed=42,
                       full_determinism=False,
                       **kwargs) -> List[T]:
        # TODO when will remote create remote?
        # Should it peer create peer? or peer create all?
        # Whether the input data of each remote is independent, or they are a part of the whole device mesh?
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        workers = []
        device_config = RayHelper.resource_manager.get_config(group)
        placement_groups = RayHelper.resource_manager.get_group(group)
        worker_cls = RayHelper._get_remote_component(worker_cls)
        ranks = device_config.ranks
        if isinstance(ranks, int):
            ranks = list(range(ranks))
        assert len(placement_groups) == len(ranks)
        key = f'{group}-{worker_cls.__class__.__name__}-{instance_id}'
        if execute == 'peer':
            # Create the peer worker
            # 0 1 2 3
            # | | | |
            # 0 1 2 3
            _slice = Platform.get_peer_index(len(ranks))
            placement_groups = placement_groups[_slice]
            ranks = ranks[_slice]
            ip, port = RayHelper.get_master_id_port(placement_groups[0]['placement_group'])
            ip = RayHelper.add_or_get_config(key + '-ip', ip)
            port = RayHelper.add_or_get_config(key + '-port', port)
        elif execute == 'first':
            placement_groups = placement_groups[:1]
            ranks = ranks[:1]
            ip, port = RayHelper.get_master_id_port(placement_groups[0]['placement_group'])
        else:
            ip, port = RayHelper.get_master_id_port(placement_groups[0]['placement_group'])

        device_type_upper = (device_config.device_type or '').upper()
        if device_type_upper != 'CPU':
            world_size = len(ranks)
            device_type = Platform.get_platform(device_type_upper).__name__
            for pg_idx, (deploy_pg, gpu) in enumerate(zip(placement_groups, ranks)):
                deploy_pg: Dict
                cluster_name = group
                worker_name = key + '-' + str(pg_idx)
                env_vars = os.environ.copy()
                env_vars.update({
                    'WORLD_SIZE':
                    str(world_size),
                    'RANK':
                    str(pg_idx),
                    'LOCAL_RANK':
                    str(0),
                    'CLUSTER_NAME':
                    cluster_name,
                    'WORKER_NAME':
                    worker_name,
                    Platform.get_platform(device_type_upper).visible_device_env():
                    ','.join([str(r) for r in deploy_pg['gpu_rank']]),
                    'TWINKLE_MODE':
                    'ray',
                    'TWINKLE_SEED':
                    str(seed),
                    'TWINKLE_FULL_DETERMINISM':
                    str(int(full_determinism)),
                })

                env_vars['MASTER_ADDR'] = ip
                env_vars['MASTER_PORT'] = str(port)

                # Prevent Ray from overriding CUDA_VISIBLE_DEVICES set in runtime_env
                # This is critical for multi-GPU workers (gpus_per_worker > 1)
                env_vars.update(ResourceManager.noset_env())

                runtime_env = RuntimeEnv(env_vars=env_vars)

                worker_options = {
                    'scheduling_strategy':
                    PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                    'name': worker_name,
                    'namespace': 'default',
                    'runtime_env': runtime_env,
                    'num_cpus': 0.01,
                }

                if device_type == 'GPU':
                    worker_options['num_gpus'] = 0.01
                else:
                    # Use custom resource key for non-GPU accelerators (e.g., NPU).
                    worker_options['resources'] = {device_type: 0.01}

                worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                workers.append(worker)
        else:
            world_size = len(ranks)
            workers = []
            # For CPU case, don't set visible device environment variables
            _visible_device_env = {}
            for rank, (deploy_pg, index) in enumerate(zip(placement_groups, list(range(world_size)))):
                deploy_pg: Dict
                cluster_name = group
                worker_name = key + '-' + str(rank)
                env_vars = os.environ.copy()
                env_vars.update({
                    'CLUSTER_NAME': cluster_name,
                    'WORKER_NAME': worker_name,
                    'TWINKLE_MODE': 'ray',
                    'TWINKLE_SEED': str(seed),
                    'TWINKLE_FULL_DETERMINISM': str(int(full_determinism)),
                    **_visible_device_env
                })
                runtime_env = RuntimeEnv(env_vars=env_vars)

                worker_options = {
                    'scheduling_strategy':
                    PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                    'name': worker_name,
                    'namespace': 'default',
                    'runtime_env': runtime_env,
                    'num_cpus': 0.01,
                }

                worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                workers.append(worker)
        return workers
