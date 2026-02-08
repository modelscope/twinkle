# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import partial
from typing import Callable, Union, Optional, Type

import twinkle.processor
from twinkle import DeviceMesh, remote_function, framework_util
from twinkle import remote_class
from twinkle.dataset import Dataset
from twinkle.processor import InputProcessor
from twinkle.utils import construct_class
from .device_mesh_fetcher import DeviceMeshIterableFetcher
from .device_mesh_sampler import DeviceMeshSampler
from .retry_sampler import RetrySampler


@remote_class(execute='first')
class DataLoader:
    """A DataLoader wrapper, will retry failed samples and return the data belongs to the current dp rank.

    Notes:
        If it is necessary to sample different in each epoch, re-create this dataloader is a better way,
            because the inner sampler does not implement a different seed in different epoches.

    Args:
        dataset: A dataset instance, or a callable to create a dataset.
            If runs in ray mode, it's recommended to use callable to make dataset and dataloader in one worker
        device_mesh: The device_mesh of this dataloader.
        batch_size: How many samples per batch.
        min_batch_size: At least how many samples should be returned.
        max_retries: Number of times to retry at one time if data fetch fails.
        kwargs: The dataloader creation parameters.
    """

    def __init__(self, dataset: Union[Dataset, Callable],
                 *,
                 batch_size: int,
                 min_batch_size: Optional[int] = None,
                 device_mesh: Optional[DeviceMesh]=None,
                 **kwargs):
        if isinstance(dataset, Callable):
            self.dataset: Dataset = dataset()
        else:
            self.dataset: Dataset = dataset
        self.dataloader = None
        self.max_retries = kwargs.pop('max_retries', 20)
        self.min_batch_size = min_batch_size
        if device_mesh is not None:
            assert batch_size >= device_mesh.data_world_size and batch_size % device_mesh.data_world_size == 0
        self.batch_size = batch_size
        self.dataloader_params = kwargs
        self.dataloader_params['batch_size'] = batch_size
        self.device_mesh = device_mesh
        self.processor: Optional[InputProcessor] = None
        self._set_work_init_fn()

    def _set_work_init_fn(self):
        num_workers = self.dataloader_params.get('num_workers', 2)
        self.dataloader_params['worker_init_fn'] = partial(
            DataLoader._seed_worker, num_workers=num_workers, rank=self.device_mesh.data_rank if self.device_mesh else 0)
    
    @remote_function()
    def __len__(self):
        self._lazy_init_dataloader()
        return len(self.dataloader)

    @staticmethod
    def _seed_worker(worker_id: int, num_workers: int, rank: int):
        import torch
        init_seed = torch.initial_seed() % 2 ** 32
        worker_seed = num_workers * rank + init_seed + worker_id
        framework_util.seed_everything(worker_seed)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, InputProcessor, Callable], **kwargs):
        """Set task processor to collate data.

        By default, this function will be used, the model will cover the data collate work.
        Args:
            processor_cls: A processor_cls class name, a processor_cls plugin id, or a processor_cls class type/instance, or a callable.
            **kwargs: Any parameters needed to construct the processor_cls instance.
        """
        self.processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)

    def _lazy_init_dataloader(self):
        if self.dataloader is None:
            from torch.utils.data import DataLoader as TorchDataLoader, IterableDataset
            if 'collate_fn' not in self.dataloader_params:
                if self.processor is not None:
                    self.dataloader_params['collate_fn'] = self.processor
                else:
                    self.dataloader_params['collate_fn'] = lambda x: x
            self.dataloader = TorchDataLoader(self.dataset, **self.dataloader_params)

            if not isinstance(self.dataset, IterableDataset):
                self.dataloader.__initialized = False
                self._repeat_sample_and_shard()
                self.dataloader.__initialized = True

    @remote_function()
    def __iter__(self):
        from torch.utils.data import IterableDataset
        self._lazy_init_dataloader()
        _iter = self.dataloader.__iter__()
        if isinstance(self.dataset, IterableDataset):
            _iter._dataset_fetcher = DeviceMeshIterableFetcher(_iter._dataset_fetcher.dataset,
                                                               _iter._dataset_fetcher.auto_collation,
                                                               _iter._dataset_fetcher.collate_fn,
                                                               _iter._dataset_fetcher.drop_last,
                                                               self.batch_size, self.device_mesh,
                                                               max_retries=self.max_retries)
        return _iter

    def _repeat_sample_and_shard(self):
        if self.dataloader.batch_sampler is not None and hasattr(self.dataloader.batch_sampler, 'sampler'):
            self.dataloader.batch_sampler.sampler = RetrySampler(self.dataloader.batch_sampler.sampler, self.dataset, max_retries=self.max_retries)
            self.dataloader.batch_sampler = DeviceMeshSampler(self.dataloader.batch_sampler, self.device_mesh, self.min_batch_size)
        elif self.dataloader.sampler is not None:
            self.dataloader.sampler = RetrySampler(self.dataloader.sampler, self.dataset, max_retries=self.max_retries)
