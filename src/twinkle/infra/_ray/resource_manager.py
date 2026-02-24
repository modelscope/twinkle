# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from typing import Dict, List

from twinkle import DeviceGroup, Platform
from twinkle.utils import get_logger

logger = get_logger()


class ResourceManager:

    def __init__(self, nproc_per_node: int, ncpu_proc_per_node: int, groups: List[DeviceGroup]):
        # CPU placement group default strategy:
        # - Old approach: use node_cpu//4 as CPU bundle per node, even if only 1~2 CPU processes are needed,
        #   this creates a huge PG (e.g., 640 CPU node requests 160 CPU PG).
        # - Ray PG uses "PACK" scheduling: once cluster has scattered CPU usage by other actors,
        #   such large CPU PGs may stay pending forever, causing Serve replica's __init__ to hang.
        # - New strategy: request based on "actual CPU processes needed on node * CPUs per process",
        #   with node_cpu//4 as the upper bound.
        cpu_pg_cpus_per_proc = int(os.environ.get('TWINKLE_CPU_PG_CPUS_PER_PROC', 1))
        cpu_pg_cpus_per_proc = max(cpu_pg_cpus_per_proc, 1)

        import ray
        from ray.util.placement_group import PlacementGroup
        all_ranks = []
        last_rank = -1
        cpu_proc_count = 0
        device_types = {group.device_type.upper() for group in groups} - {'CPU'}
        assert len(device_types) <= 1

        if not device_types:
            # Pure cpu task
            device_type = 'CPU'
        else:
            device_type = next(iter(device_types))
            device_type = Platform.get_platform(device_type).__name__

        for group in groups:
            ranks = group.ranks
            device = device_type
            if device == 'CPU':
                # Only support totally how many processes needed
                assert isinstance(ranks, int), 'CPU group only supports integer ranks'
                cpu_proc_count += ranks
                continue

            if isinstance(ranks, int):
                # turn to a list of int
                ranks = list(range(last_rank + 1, last_rank + 1 + ranks))
            all_ranks.extend(ranks)
            group.ranks = ranks
            last_rank = ranks[-1]

        assert len(set(all_ranks)) == len(all_ranks)  # no duplication
        if device_type != 'CPU':
            # Calculate required nodes based on actual node indices spanned by all_ranks
            if all_ranks:
                node_indices = [rank // nproc_per_node for rank in all_ranks]
                self.min_node_idx = min(node_indices)
                self.nnodes = max(node_indices) - self.min_node_idx + 1
            else:
                self.min_node_idx = 0
                self.nnodes = 0
        else:
            self.min_node_idx = 0
            self.nnodes = math.ceil(cpu_proc_count / ncpu_proc_per_node)

        self.nodes = []
        for node in ray.nodes():
            # get available nodes
            resource = node['Resources']
            node_device_num = int(resource.get(device_type, 0))
            if device_type != 'CPU' and node_device_num >= nproc_per_node:
                self.nodes.append(node)
            if device_type == 'CPU' and int(node['Resources']['CPU']) // 4 >= ncpu_proc_per_node:
                self.nodes.append(node)

        assert self.nnodes <= len(
            self.nodes), f'Not enough resources, required nodes: {self.nnodes}, available: {len(self.nodes)}'

        bundles = []
        cpu_bundles = []

        for i in range(self.nnodes):
            # TODO not accurate, because placement_group cannot distribute to node same ordered with self.nodes
            node_idx = self.min_node_idx + i if device_type != 'CPU' else i
            try:
                node = self.nodes[node_idx]
            except IndexError:
                # node_idx may not be continuous
                node = self.nodes[0]
            node_cpu = int(node['Resources']['CPU'])
            if device_type != 'CPU':
                bundles.append({device_type: nproc_per_node, 'CPU': max(node_cpu // 2, 1)})  # create bundles

        # CPU placement groups: only create when there are actual CPU processes to allocate.
        if cpu_proc_count > 0:
            cpu_nnodes = math.ceil(cpu_proc_count / ncpu_proc_per_node)
            assert cpu_nnodes <= len(self.nodes), (f'Not enough nodes for CPU processes, required nodes: {cpu_nnodes}, '
                                                   f'available: {len(self.nodes)}')
            for i in range(cpu_nnodes):
                node = self.nodes[i]
                node_cpu = int(node['Resources']['CPU'])
                # How many CPU processes will actually be placed on this node
                # (last node may have fewer than ncpu_proc_per_node)
                procs_on_node = min(
                    ncpu_proc_per_node,
                    max(cpu_proc_count - i * ncpu_proc_per_node, 0),
                )
                # Use node_cpu//4 as the upper bound of "at most 1/4 CPU usage",
                # but don't request 160 CPU for just 1~2 processes.
                node_cap = max(node_cpu // 4, 1)
                need = max(procs_on_node * cpu_pg_cpus_per_proc, 1)
                cpu_bundles.append({'CPU': min(node_cap, need)})

        self.cpu_node_map = {}
        for i in range(cpu_proc_count):
            node_idx = i // ncpu_proc_per_node
            # We don't strictly assert CPU per proc >= 1 here because for tail nodes with fewer processes,
            # the allocated CPU might be small (e.g. 1 process needs 1 CPU, but ncpu_proc_per_node=8).
            self.cpu_node_map[i] = (node_idx, 1)

        self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
        self.cpu_placement_groups = [ray.util.placement_group([bundle]) for bundle in cpu_bundles]
        if self.placement_groups:
            ray.get([pg.ready() for pg in self.placement_groups])
        if self.cpu_placement_groups:
            ray.get([pg.ready() for pg in self.cpu_placement_groups])

        self.node_ranks = []
        if self.placement_groups:
            self.node_ranks = ray.get([
                ray.remote(Platform.get_node_rank).options(placement_group=pg).remote() for pg in self.placement_groups
            ])
        if self.node_ranks.count(0) > 1:
            self.node_ranks = list(range(len(self.placement_groups)))

        self.visible_devices = []

        @ray.remote
        def get_visible_devices():
            return os.environ.get(Platform.get_platform(group.device_type).visible_device_env())

        if self.placement_groups:
            self.visible_devices = ray.get([
                get_visible_devices.options(placement_group=pg, runtime_env={
                    'env_vars': self.noset_env()
                }).remote() for pg in self.placement_groups
            ])

        visible_devices = []
        for visible_device in self.visible_devices:
            if visible_device:
                visible_device = [int(device) for device in visible_device.split(',')]
            else:
                visible_device = list(range(nproc_per_node))
            visible_devices.append(visible_device)
        self.visible_devices = visible_devices

        self.node2pg: Dict[int, PlacementGroup] = {}
        # Map actual node indices to placement groups
        # For GPU/NPU groups, node indices start from self.min_node_idx
        if device_type != 'CPU':
            for i, placement_group in enumerate(self.placement_groups):
                actual_node_idx = self.min_node_idx + i
                self.node2pg[actual_node_idx] = placement_group
        else:
            # For CPU-only or when using default node_ranks
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group

        self.device_groups = {}
        ray_address = str(ray.get_runtime_context().gcs_address)
        for group in groups:
            if group.device_type != 'CPU':
                ranks = group.ranks
                gpus_per_worker = getattr(group, 'gpus_per_worker', 1)
                local_device_groups = []
                # Use original ranks for GPU mapping so each DeviceGroup maps to
                # the correct physical devices.  E.g. ranks=[2,3] with
                # nproc_per_node=4 should map to gpu_rank [2,3], not [0,1].
                normalized_ranks = list(ranks)

                if gpus_per_worker > 1:
                    if len(normalized_ranks) % gpus_per_worker != 0:
                        raise ValueError(f"DeviceGroup '{group.name}': number of ranks ({len(normalized_ranks)}) "
                                         f'must be divisible by gpus_per_worker ({gpus_per_worker})')

                    num_workers = len(normalized_ranks) // gpus_per_worker
                    for worker_idx in range(num_workers):
                        start_idx = worker_idx * gpus_per_worker
                        worker_ranks = normalized_ranks[start_idx:start_idx + gpus_per_worker]

                        # All GPUs for a worker should be on the same node
                        gpu_ranks_local = []
                        for r in worker_ranks:
                            node_rank = r // nproc_per_node
                            node_ranks.append(node_rank)
                            gpu_ranks = self.visible_devices[node_rank][r % nproc_per_node]
                            gpu_ranks_local.append(gpu_ranks)

                        if len(set(node_ranks)) > 1:
                            raise ValueError(f"DeviceGroup '{group.name}': GPUs {worker_ranks} span multiple nodes. "
                                             f"Each worker's GPUs must be on the same node.")

                        node_rank = node_ranks[0]
                        local_device_groups.append(
                            dict(
                                gpu_rank=gpu_ranks_local,
                                placement_group=self.node2pg[node_rank],
                                ray_address=ray_address))
                else:
                    for alloc_rank in normalized_ranks:
                        node_rank = alloc_rank // nproc_per_node
                        gpu_rank = self.visible_devices[node_rank][alloc_rank % nproc_per_node]
                        local_device_groups.append(
                            dict(gpu_rank=[gpu_rank], placement_group=self.node2pg[node_rank], ray_address=ray_address))

                self.device_groups[group.name] = local_device_groups

                # Update the group's ranks to reflect actual worker count
                if gpus_per_worker > 1:
                    # Create virtual ranks for workers (not GPUs)
                    group.ranks = list(range(len(local_device_groups)))
            else:
                assert getattr(group, 'gpus_per_worker', 1) == 1
                ranks = group.ranks
                local_device_groups = []
                global_cpu_proc_idx = 0
                for _ in range(ranks):
                    local_device_groups.append(
                        dict(
                            placement_group=self.cpu_placement_groups[self.cpu_node_map[global_cpu_proc_idx][0]],
                            ray_address=ray_address))
                    global_cpu_proc_idx += 1
                self.device_groups[group.name] = local_device_groups

        self.group_configs = groups
        logger.info(f"nodes: {[n['NodeID'][:8] for n in self.nodes]}")
        logger.info(f'node_ranks: {self.node_ranks}')
        logger.info(f'node2pg keys: {list(self.node2pg.keys())}')

    @staticmethod
    def noset_env():
        return {
            'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
            'RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES': '1',
            'RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES': '1',
            'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES': '1',
            'RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES': '1',
            'RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES': '1',
            'RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS': '1',
            'RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR': '1',
        }

    def get_config(self, group: str):
        for config in self.group_configs:
            if config.name == group:
                return config
        assert False, f'No group {group} found in group list: {[group.name for group in self.group_configs]}'

    def get_group(self, group: str):
        assert group in self.device_groups, (f'No group {group} found in group '
                                             f'list: {[group.name for group in self.group_configs]}')
        return self.device_groups[group]

    def destroy_placement_group(self):
        import ray
        for pg in self.placement_groups:
            ray.util.remove_placement_group(pg)
        for pg in self.cpu_placement_groups:
            ray.util.remove_placement_group(pg)
