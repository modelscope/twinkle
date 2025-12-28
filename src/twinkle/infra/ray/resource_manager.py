# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from typing import Dict, List

import ray
from ray.util.placement_group import PlacementGroup

from .. import DeviceGroup
from ... import Platform


class ResourceManager:

    def __init__(self,
                 nproc_per_node: int,
                 groups: List[DeviceGroup]):
        all_ranks = []
        last_rank = -1
        cpu_proc_count = 0
        device_types = set([group.device_type.upper() for group in groups]) - {'CPU'}
        assert len(device_types) == 1
        device_type = next(iter(device_types))

        for group in groups:
            ranks = group.ranks
            device = group.device_type.upper()
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

        assert len(set(all_ranks)) == len(all_ranks) # no duplication
        self.nnodes = math.ceil(len(all_ranks) / nproc_per_node)

        self.nodes = []
        for node in ray.nodes():
            # get available nodes
            resource = node['Resources']
            node_gpu_num = int(resource.get(device_type, 0))
            if node_gpu_num >= nproc_per_node:
                self.nodes.append(node)

        bundles = []
        cpu_bundles = []
        for i in range(self.nnodes):
            node = self.nodes[i]
            node_cpu = int(node['Resources']['CPU'])
            bundles.append({device_type: nproc_per_node, 'CPU': node_cpu // 2 + 1}) # create bundles
            cpu_bundles.append({'CPU': node_cpu // 4 + 1})  # TODO a fixed cpu count

        nproc_cpu_per_node = cpu_proc_count // len(cpu_bundles) + 1 # how many processes in one node

        self.cpu_node_map = {}
        for i in range(cpu_proc_count):
            node_idx = i // nproc_cpu_per_node
            cpu_cnt = cpu_bundles[node_idx]['CPU']
            cpu_per_proc = cpu_cnt // nproc_cpu_per_node # cpus per process
            assert cpu_per_proc >= 1
            self.cpu_node_map[i] = (node_idx, cpu_per_proc)

        self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
        self.cpu_placement_groups = [ray.util.placement_group([bundle]) for bundle in cpu_bundles]
        cpu_bundles.sort(key=lambda bundle: bundle['CPU'], reverse=True)
        ray.get([pg.ready() for pg in self.placement_groups])
        ray.get([pg.ready() for pg in self.cpu_placement_groups])

        self.node_ranks = ray.get(
            [ray.remote(Platform.get_node_rank).options(placement_group=pg).remote() for pg in self.placement_groups])
        if self.node_ranks.count(0) > 1:
            self.node_ranks = list(range(len(self.placement_groups)))

        self.node2pg: Dict[int, PlacementGroup] = {}
        for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
            self.node2pg[node_rank] = placement_group

        self.device_groups = {}
        ray_address = str(ray.get_runtime_context().gcs_address)
        for group in groups:
            if group.device_type.upper() != 'CPU':
                ranks = group.ranks
                local_device_groups = []
                for rank in ranks:
                    node_rank = rank // nproc_per_node
                    gpu_rank = rank % nproc_per_node
                    local_device_groups.append(
                        dict(
                            node_rank=node_rank,
                            gpu_rank=[gpu_rank],
                            placement_group=self.node2pg[node_rank],
                            ray_address=ray_address))
                self.device_groups[group.name] = local_device_groups
            else:
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

    def get_config(self, group: str):
        for config in self.group_configs:
            if config.name == group:
                return config
        assert False, f'No group {group} found in group list: {[group.name for group in self.group_configs]}'

    def get_group(self, group: str):
        assert group in self.device_groups, f'No group {group} found in group list: {[group.name for group in self.group_configs]}'
        return self.device_groups[group]

    def destroy_placement_group(self):
        import ray
        for pg in self.placement_groups:
            ray.util.remove_placement_group(pg)
        for pg in self.cpu_placement_groups:
            ray.util.remove_placement_group(pg)
