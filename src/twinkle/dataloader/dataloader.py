from functools import partial
from typing import Callable, Union, Optional
from torch.utils.data import DataLoader as TorchDataLoader

from twinkle import DeviceMesh, remote_function, framework_util
from twinkle import remote_class
from twinkle.dataset import Dataset
from .retry_sampler import RetrySampler
from .device_mesh_sampler import DeviceMeshSampler


@remote_class()
class DataLoader(TorchDataLoader):

    def __init__(self, dataset: Union[Dataset, Callable], device_mesh: Optional[DeviceMesh]=None,
                 **dataloader_params):
        if isinstance(dataset, Callable):
            self.dataset: Dataset = dataset()
        else:
            self.dataset: Dataset = dataset
        self.model_id = None
        self.template = None
        self.dataloader = None
        self.dataloader_params = dataloader_params
        self.device_mesh = device_mesh

    def set_work_init_fn(self):
        num_workers = self.dataloader_params.get('num_workers', 0)
        self.dataloader_params['worker_init_fn'] = partial(
            DataLoader.seed_worker, num_workers=num_workers, rank=self.device_mesh.dp_rank)

    @staticmethod
    def seed_worker(worker_id: int, num_workers: int, rank: int):
        import torch
        init_seed = torch.initial_seed() % 2 ** 32
        worker_seed = num_workers * rank + init_seed + worker_id
        framework_util.seed_everything(worker_seed)

    @remote_function()
    def __iter__(self):
        if self.dataloader is None:
            self.dataloader = TorchDataLoader(self.dataset, **self.dataloader_params,
                                              collate_fn=lambda x: x)
            self._repeat_sample()
        return self.dataloader.__iter__()

    def _repeat_sample(self):
        if self.dataloader.batch_sampler is not None and hasattr(self.dataloader.batch_sampler, 'sampler'):
            self.dataloader.batch_sampler.sampler = RetrySampler(self.dataloader.batch_sampler.sampler, self.dataset)
            self.dataloader.batch_sampler = DeviceMeshSampler(self.dataloader.batch_sampler, self.device_mesh)
        elif self.dataloader.sampler is not None:
            self.dataloader.sampler = RetrySampler(self.dataloader.sampler, self.dataset)
