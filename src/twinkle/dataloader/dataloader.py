from functools import partial
from types import MethodType
from typing import Callable, Union, Optional
from torch.utils.data import DataLoader as TorchDataLoader

from twinkle import DeviceMesh, remote_function, framework_util
from twinkle import remote_class
from twinkle.dataset import Dataset
from .retry_sampler import RetrySampler
from .device_mesh_sampler import DeviceMeshSampler


@remote_class()
class DataLoader(TorchDataLoader):
    """A DataLoader wrapper, will retry failed samples and return the data belongs to the current dp rank.

    Args:
        dataset: A dataset instance, or a callable to create a dataset.
        device_mesh: The device_mesh of this dataloader.
        kwargs: The dataloader creation parameters.
    """

    def __init__(self, dataset: Union[Dataset, Callable], device_mesh: Optional[DeviceMesh]=None,
                 **kwargs):
        breakpoint()
        if isinstance(dataset, Callable):
            self.dataset: Dataset = dataset()
        else:
            self.dataset: Dataset = dataset
        self.dataloader = None
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = device_mesh.data_parallel_world_size
        assert kwargs['batch_size'] >= device_mesh.data_parallel_world_size and kwargs['batch_size'] % device_mesh.data_parallel_world_size == 0
        self.dataloader_params = kwargs
        self.device_mesh = device_mesh
        self._set_work_init_fn()

    def _set_work_init_fn(self):
        num_workers = self.dataloader_params.get('num_workers', 0)
        self.dataloader_params['worker_init_fn'] = partial(
            DataLoader._seed_worker, num_workers=num_workers, rank=self.device_mesh.dp_rank)

    @staticmethod
    def _seed_worker(worker_id: int, num_workers: int, rank: int):
        import torch
        init_seed = torch.initial_seed() % 2 ** 32
        worker_seed = num_workers * rank + init_seed + worker_id
        framework_util.seed_everything(worker_seed)

    @remote_function(collect='flatten')
    def __iter__(self):
        if self.dataloader is None:
            if 'collate_fn' not in self.dataloader_params:
                self.dataloader_params['collate_fn'] = lambda x: x
            self.dataloader = TorchDataLoader(self.dataset, **self.dataloader_params)

            self.dataloader.__initialized = False
            self._repeat_sample_and_shard()
            self.dataloader.__initialized = True

        return self.dataloader.__iter__()

    def _repeat_sample_and_shard(self):
        if self.dataloader.batch_sampler is not None and hasattr(self.dataloader.batch_sampler, 'sampler'):
            self.dataloader.batch_sampler.sampler = RetrySampler(self.dataloader.batch_sampler.sampler, self.dataset)
            self.dataloader.batch_sampler = DeviceMeshSampler(self.dataloader.batch_sampler, self.device_mesh)
        elif self.dataloader.sampler is not None:
            self.dataloader.sampler = RetrySampler(self.dataloader.sampler, self.dataset)
