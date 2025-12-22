# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Callable, Dict, List, Literal, Optional, TypeVar, Union, Type

import numpy as np

from .resource_manager import ResourceManager
from .. import DeviceGroup
from ...utils import Platform, find_node_ip, find_free_port
from ...utils import requires

T = TypeVar('T')


class RayHelper:

    resource_manager: Optional[ResourceManager] = None

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

    @staticmethod
    def teardown():
        if RayHelper.resource_manager is not None:
            RayHelper.resource_manager.destroy_placement_group()
            RayHelper.resource_manager = None

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
    def collect_func(method: Union[Literal['none', 'flatten'], Callable], result):
        if isinstance(result[0], tuple):
            output = []
            for i in range(len(result[0])):
                _single_result = [r[i] for r in result]
                output.append(RayHelper.collect_func(method, _single_result))
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

    @staticmethod
    def execute_all_sync(workers, dispatch, execute, method_name: str, *args, **kwargs):
        import ray
        return ray.get(RayHelper.execute_all_async(workers, dispatch, execute, method_name, *args, **kwargs))

    @staticmethod
    def execute_all_async( workers, dispatch, execute, method_name: str, *args, **kwargs):
        length = len(workers)
        if execute == 'first':
            return getattr(workers[0], method_name).remote(*args, **kwargs)
        elif dispatch == 'all':
            return [getattr(worker, method_name).remote(*args, **kwargs) for worker in workers]
        elif dispatch == 'slice':
            result = []

            def dispatch_func(arg, n):
                if isinstance(arg, list):
                    k, m = divmod(len(arg), n)
                    return [arg[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
                else:
                    return [arg] * n

            args = [dispatch_func(arg, length) for arg in args]
            kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}
            for i in range(length):
                sliced_args = tuple(arg[i] for arg in args)
                sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                if (sliced_args and sliced_args[0]) or (kwargs and list(kwargs.values())):
                    # skip empty input
                    remote_call = getattr(workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
            return result
        elif isinstance(dispatch, Callable):
            # dispatch is Callable
            result = []
            for i in range(length):
                sliced_args, sliced_kwargs = dispatch(length, i, *args, **kwargs)
                remote_call = getattr(workers[i], method_name)
                result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
            return result
        else:
            raise ValueError(f'Invalid dispatch method: {dispatch}')

    @staticmethod
    def create_workers(worker_cls: Type[T], group: str, *args, **kwargs) -> List[T]:
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        device_config = RayHelper.resource_manager.get_config(group)
        placement_groups = RayHelper.resource_manager.get_group(group)
        if device_config.device_type.upper() != 'CPU':
            world_size = len(device_config.ranks)
            workers = []
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

        return workers
