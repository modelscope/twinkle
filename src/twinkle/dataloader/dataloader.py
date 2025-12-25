from functools import partial
from typing import Callable, Union, Optional
from torch.utils.data import DataLoader as TorchDataLoader

from twinkle import DeviceMesh, remote_function, framework_util
from twinkle import remote_class
from twinkle.dataset import Dataset
from twinkle.processor import DataProcessorMixin


@remote_class()
class DataLoader(TorchDataLoader, DataProcessorMixin):

    def __init__(self, dataset: Union[Dataset, Callable], device_mesh: Optional[DeviceMesh]=None,
                 **dataloader_params):
        if isinstance(dataset, Callable):
            self.dataset: Dataset = dataset()
        else:
            self.dataset: Dataset = dataset
        self.model_id = None
        self.template = None
        self.dataloader = None
        self.encoded = False
        self.dataloader_params = dataloader_params
        super(TorchDataLoader, self).__init__(device_mesh)

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
    def encode(self, **kwargs):
        if not self.encoded:
            self.encoded = True
            self.dataset.map(self.template.encode, **kwargs)

    @remote_function()
    def __iter__(self):
        if self.dataloader is None:
            if self.processor is not None and self.encoded:
                collate_fn = self.processor.collate_fn
            else:
                collate_fn = lambda x: x
            self.dataloader = TorchDataLoader(self.dataset, **self.dataloader_params,
                                              collate_fn=collate_fn)

        return self.dataloader.__iter__()
