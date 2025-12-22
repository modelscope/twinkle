# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict, List, Optional, TypeVar, Type, Tuple, Any

from .resource_manager import ResourceManager
from .. import DeviceGroup
from ...utils import Platform, find_node_ip, find_free_port
from ...utils import requires

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
                self.workers = {}

            def register_workers(self, group: str, worker_handles: List):
                self.workers[group] = worker_handles

            def get_workers(self, group: str):
                return self.workers.get(group, [])

            def clear(self):
                self.workers.clear()

        try:
            RayHelper._registry = ray.get_actor('worker_registry')
        except ValueError:
            try:
                RayHelper._registry = WorkerRegistry.options(
                    name='worker_registry',
                    lifetime='detached',
                ).remote()
            except ValueError:
                RayHelper._registry = ray.get_actor('worker_registry')
        assert RayHelper._registry is not None

    @staticmethod
    def initialize(nproc_per_node: int, device_groups: List[DeviceGroup]):
        """Initialize RayHelper.

        Args:
            nproc_per_node: How many devices in one node.
            device_groups: The device groups to initialize.

        Returns:
            None
        """
        if RayHelper.ray_inited():
            return

        requires('ray')
        import ray
        RayHelper.device_groups = device_groups
        ray.init()
        if RayHelper.resource_manager is None:
            # Resource manager initializes only once in the pipeline process.
            RayHelper.resource_manager = ResourceManager(nproc_per_node, device_groups)
        RayHelper.init_registry()

    @staticmethod
    def teardown():
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
        import ray
        return ray.get(RayHelper.execute_all_async(method_name, workers_and_args))

    @staticmethod
    def execute_all_async(method_name:str, workers_and_args: List[Tuple[Any, List[Any], Dict[str, Any]]]):
        output = []
        for worker_and_args in workers_and_args:
            worker, args, kwargs = worker_and_args
            remote_call = getattr(worker, method_name)
            output.append(remote_call.remote(*args, **kwargs))
        return output

    @staticmethod
    def _get_remote_component(component):
        if component not in RayHelper._remote_components:
            import ray
            RayHelper._remote_components[component] = ray.remote(component)
        return RayHelper._remote_components[component]

    @staticmethod
    def create_workers(worker_cls: Type[T], group: str, *args, **kwargs) -> List[T]:
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        role = group + '-' + worker_cls.__name__
        workers = ray.get(RayHelper._registry.get_workers.remote(role))
        if not workers:
            # TODO need to be sequential
            device_config = RayHelper.resource_manager.get_config(group)
            placement_groups = RayHelper.resource_manager.get_group(group)
            worker_cls = RayHelper._get_remote_component(worker_cls)
            if device_config.device_type.upper() != 'CPU':
                world_size = len(device_config.ranks)
                ip, port = None, None
                for rank, (deploy_pg, gpu) in enumerate(zip(placement_groups, device_config.ranks)):
                    deploy_pg: Dict
                    cluster_name = group
                    worker_name = cluster_name + '-' + str(rank)
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
                        Platform.get_platform(device_config.device_type.upper()):
                        ','.join([str(r) for r in deploy_pg['gpu_rank']]),
                        'TWINKLE_MODE': 'ray',
                    })

                    @ray.remote
                    def get_node_address():
                        return find_node_ip(), find_free_port()

                    if rank == 0:
                        ip, port = ray.get(
                            get_node_address.options(placement_group=deploy_pg['placement_group']).remote())

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
                world_size = len(device_config.ranks)
                workers = []
                for deploy_pg, index in zip(placement_groups, list(range(world_size))):
                    deploy_pg: Dict
                    cluster_name = group
                    worker_name = cluster_name + '-' + str(index)
                    env_vars = os.environ.copy()
                    env_vars.update({
                        'CLUSTER_NAME': cluster_name,
                        'WORKER_NAME': worker_name,
                        Platform.get_platform(device_config.device_type.upper()): '',
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

            ray.get(RayHelper._registry.register_workers.remote(role, workers))
        return workers
