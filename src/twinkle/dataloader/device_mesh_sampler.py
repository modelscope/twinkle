from torch.utils.data import BatchSampler

from twinkle import DeviceMesh


class DeviceMeshSampler(BatchSampler):

    def __init__(self, original_sampler, device_mesh: DeviceMesh):
        self.original_sampler = original_sampler
        self.device_mesh = device_mesh

    def __iter__(self):
        for batch in self.original_sampler:
            if not self.device_mesh:
                yield batch
            else:
                yield batch[self.device_mesh.get_slice(len(batch))]

    def __len__(self):
        return len(self.original_sampler)