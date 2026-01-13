# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import BatchSampler

from twinkle import DeviceMesh


class DeviceMeshSampler(BatchSampler):
    """A sampler returns the slice of the current dp rank.

    Args:
        original_sampler: The original BatchSampler.
        device_mesh: The device mesh.
    """
    def __init__(self, original_sampler: BatchSampler, device_mesh: DeviceMesh):
        self.original_sampler = original_sampler
        self.device_mesh = device_mesh

    def __iter__(self):
        for batch in self.original_sampler:
            if not self.device_mesh:
                yield batch
            else:
                data = batch[self.device_mesh.get_slice(len(batch))]
                if not data:
                    # Use rank0 if data is not enough
                    data = batch[self.device_mesh.get_slice(len(batch), 0)]
                yield data

    def __len__(self):
        return len(self.original_sampler)