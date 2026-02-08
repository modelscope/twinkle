# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import BatchSampler

from twinkle import DeviceMesh


class DeviceMeshSampler(BatchSampler):
    """A sampler returns the slice of the current dp rank.

    Args:
        original_sampler: The original BatchSampler.
        device_mesh: The device mesh.
    """
    def __init__(self, original_sampler: BatchSampler, device_mesh: DeviceMesh, min_batch_size: int = None):
        self.original_sampler = original_sampler
        self.device_mesh = device_mesh
        self.min_batch_size = min_batch_size
        if self.min_batch_size is None and self.device_mesh is not None:
            self.min_batch_size = self.device_mesh.data_world_size

    def __iter__(self):
        for batch in self.original_sampler:
            if not self.device_mesh:
                yield batch
            else:
                if len(batch) < self.min_batch_size:
                    return
                else:
                    yield batch[self.device_mesh.get_slice(len(batch))]

    def __len__(self):
        return len(self.original_sampler)