# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import Dict, List, Optional, TypeVar, Type, Tuple, Any, Literal, Callable, Union

import ray

from .resource_manager import ResourceManager
from twinkle import DeviceGroup, Platform, find_node_ip, find_free_port, requires

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
                RayHelper._registry = WorkerRegistry.options(
                    name='config_registry',
                    lifetime='detached',
                ).remote()
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
            ray.init()

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
        import ray
        return RayHelper.ray_inited() and ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE

    @staticmethod
    def execute_all_sync(method_name:str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]]):
        """Execute method and return results."""
        import ray
        return ray.get(RayHelper.execute_all_async(method_name, workers_and_args))

    @staticmethod
    def execute_all_async(method_name:str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]]):
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

        ip, port = ray.get(
            get_node_address.options(placement_group=placement_group).remote())
        return ip, port

    @staticmethod
    def do_get_and_collect_func(collect_func: Callable, method: Union[Literal['none', 'flatten'], Callable], futures):
        """Return a callable to collect results in the workers."""

        class LazyCollect:
            def __init__(self, futures, method, collect_func):
                self._futures = futures
                self._method = method
                self._collect_func = collect_func
                self._is_lazy_collect = True

            def __call__(self):
                result = []
                for future in self._futures:
                    if isinstance(future, ray.ObjectRef):
                        result.append(ray.get(future))
                    else:
                        result.append(future)
                return self._collect_func(self._method, result)

            def __getattr__(self, name):
                raise RuntimeError(f'This is a lazy function, use value().attr instead.')

            def __getitem__(self, key):
                raise RuntimeError(f'This is a lazy function, use value()[index] instead.')

        return LazyCollect(futures, method, collect_func)

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
    def create_workers(worker_cls: Type[T], group: str, execute: Literal['all', 'peer', 'first'], instance_id, seed=42, full_determinism=False, *args, **kwargs) -> List[T]:
        # TODO when will remote create remote?
        # Should it peer create peer? or peer create all?
        # Whether the input data of each remote is independent, or they are a part of the whole device mesh?
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        workers = []
        device_config = RayHelper.resource_manager.get_config(group)
        placement_groups = RayHelper.resource_manager.get_group(group)
        worker_cls = RayHelper._get_remote_component(worker_cls)
        ranks = device_config.ranks
        if isinstance(ranks, int):
            ranks = list(range(ranks))
        world_size = len(ranks)
        assert len(placement_groups) == len(ranks)
        key = f'{group}-{worker_cls.__class__.__name__}-{instance_id}'
        if execute == 'peer':
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

        if device_config.device_type.upper() != 'CPU':
            for rank, (deploy_pg, gpu) in enumerate(zip(placement_groups, ranks)):
                deploy_pg: Dict
                cluster_name = group
                worker_name = key + '-' + str(rank)
                env_vars = os.environ.copy()
                env_vars.update({
                    'WORLD_SIZE':
                    str(world_size),
                    'RANK':
                    str(rank),
                    'LOCAL_RANK':
                    str(0),
                    'CLUSTER_NAME':
                    cluster_name,
                    'WORKER_NAME':
                    worker_name,
                    Platform.get_platform(device_config.device_type.upper()).visible_device_env():
                    ','.join([str(r) for r in deploy_pg['gpu_rank']]),
                    'TWINKLE_MODE': 'ray',
                    'TWINKLE_SEED': str(seed),
                    'TWINKLE_FULL_DETERMINISM': str(int(full_determinism)),
                })

                env_vars['MASTER_ADDR'] = ip
                env_vars['MASTER_PORT'] = str(port)

                runtime_env = RuntimeEnv(env_vars=env_vars)

                worker_options = {
                    'scheduling_strategy':
                    PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                    'name':
                    worker_name,
                    'namespace':
                    'default',
                    'runtime_env':
                    runtime_env,
                    'num_cpus':
                    0.01,
                    'num_gpus':
                    0.01,
                }

                worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                workers.append(worker)
        else:
            world_size = len(ranks)
            workers = []
            if device_config.device_type != 'CPU':
                _visible_device_env = {Platform.get_platform(device_config.device_type.upper()).visible_device_env(): ''}
            else:
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
                    'name':
                    worker_name,
                    'namespace':
                    'default',
                    'runtime_env':
                    runtime_env,
                    'num_cpus':
                    0.01,
                }

                worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                workers.append(worker)
        return workers
