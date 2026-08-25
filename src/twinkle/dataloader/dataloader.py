# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import warnings
from functools import partial
from typing import Callable, Optional, Sequence, Type, Union

import twinkle.processor
from twinkle import DeviceMesh, framework_util, remote_class, remote_function
from twinkle.dataset import Dataset
from twinkle.processor import InputProcessor
from twinkle.utils import construct_class
from .device_mesh_dataset import DeviceMeshDataset
from .device_mesh_sampler import DeviceMeshSampler
from .epoch_sampler import EpochSampler
from .retry_dataset import RetryDataset
from .skip_sampler import SkipSampler


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
        data_seed: Seeds the read order, so a run is reproducible and a resumed epoch replays the order
            it is continuing. See :class:`EpochSampler`.
        group_by_length: Batch samples of similar length together to cut padding. Needs ``lengths``.
        lengths: One length per sample, for ``group_by_length``.
        max_retries: Number of times to retry at one time if data fetch fails.
        kwargs: The dataloader creation parameters.
    """

    def __init__(self,
                 dataset: Union[Dataset, Callable],
                 *,
                 batch_size: int,
                 min_batch_size: Optional[int] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 data_seed: Optional[int] = None,
                 group_by_length: bool = False,
                 lengths: Optional[Sequence] = None,
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
        self.data_seed = data_seed
        self.group_by_length = group_by_length
        self.lengths = lengths
        self._epoch_sampler: Optional[EpochSampler] = None
        # Checked here rather than where the sampler is built, which does not happen until the first
        # iteration: a misconfigured dataloader should say so when it is configured, not once a run has
        # already loaded a model and started.
        if group_by_length and lengths is None:
            raise ValueError('group_by_length needs `lengths`, one per sample.')
        self.processor: Optional[InputProcessor] = None
        self._skip_samples = 0
        # Where an interrupted run left off, split into which epoch and how far into it. Pending until
        # the next pass reads it, because nothing here counts epochs -- see `skip_consumed_samples`.
        self._resume_epoch = 0
        self._resume_offset = 0
        self._resume_pending = False
        self._consumed_train_samples = 0
        self._base_batch_sampler = None
        self._base_sampler = None
        self._retry_seed = self._resolve_retry_seed()
        self._set_work_init_fn()

    def _set_work_init_fn(self):
        num_workers = self.dataloader_params.get('num_workers', 2)
        self.dataloader_params['worker_init_fn'] = partial(
            DataLoader._seed_worker,
            num_workers=num_workers,
            rank=self.device_mesh.data_rank if self.device_mesh else 0)

    @staticmethod
    def _resolve_retry_seed() -> int:
        """The seed :class:`RetryDataset` draws replacements with, so they match across ranks."""
        env_seed = os.environ.get('TWINKLE_SEED')
        if env_seed is not None:
            return int(env_seed)
        try:
            from twinkle.infra import _seed
            return int(_seed)
        except Exception:
            return 42

    @remote_function()
    def __len__(self):
        self._lazy_init_dataloader()
        return len(self.dataloader)

    @staticmethod
    def _seed_worker(worker_id: int, num_workers: int, rank: int):
        import torch
        init_seed = torch.initial_seed() % 2**32
        worker_seed = num_workers * rank + init_seed + worker_id
        framework_util.seed_everything(worker_seed)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, InputProcessor, Callable], **kwargs):
        """Set task processor to collate data.

        By default, this function will be used, the model will cover the data collate work.
        Args:
            processor_cls: A processor_cls class name, a processor_cls plugin id, or a processor_cls
                class type/instance, or a callable.
            **kwargs: Any parameters needed to construct the processor_cls instance.
        """
        self.processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)

    def _lazy_init_dataloader(self):
        if self.dataloader is None:
            from torch.utils.data import DataLoader as TorchDataLoader
            from torch.utils.data import IterableDataset
            if 'collate_fn' not in self.dataloader_params:
                if self.processor is not None:
                    self.dataloader_params['collate_fn'] = self.processor
                else:
                    self.dataloader_params['collate_fn'] = lambda x: x
            self.dataloader = TorchDataLoader(self._read_through(self.dataset), **self._resolved_params())

            if not isinstance(self.dataset, IterableDataset):
                self.dataloader.__initialized = False
                self._base_batch_sampler = self.dataloader.batch_sampler
                self._base_sampler = self.dataloader.sampler
                self._rebuild_sampler_stack()
                self.dataloader.__initialized = True

    def _resolved_params(self) -> dict:
        """The torch DataLoader parameters, with the read order taken over by :class:`EpochSampler`.

        ``shuffle`` is translated rather than passed on: torch answers it with a ``RandomSampler`` seeded
        from the global RNG, which is neither reproducible nor replayable on resume.

        A caller who brings their own sampler keeps it -- they have said what order they want. For a
        ``batch_sampler`` that also means dropping ``batch_size`` and ``drop_last``, which torch refuses
        alongside one (the sampler decides the batches, so a second opinion on their size is a
        contradiction). ``self.batch_size`` stays as the declared width, since the mesh slicing and the
        consumed-sample count are stated in terms of it.

        An iterable dataset has no order to decide, arriving in one already.
        """
        params = dict(self.dataloader_params)
        shuffle = params.pop('shuffle', False)
        from torch.utils.data import IterableDataset
        if isinstance(self.dataset, IterableDataset):
            # `DeviceMeshDataset` has already taken this rank's slice out of each global batch, so what
            # reaches the DataLoader is this rank's share alone and is batched at that width. Batching
            # it at the global width again would make every batch `data_world_size` times too wide.
            if self.device_mesh is not None:
                params['batch_size'] = self.batch_size // self.device_mesh.data_world_size
            return params
        if params.get('batch_sampler') is not None:
            params.pop('batch_size', None)
            params.pop('drop_last', None)
            return params
        if params.get('sampler') is not None:
            return params
        self._epoch_sampler = EpochSampler(
            len(self.dataset),
            shuffle=shuffle,
            data_seed=self.data_seed,
            group_by_length=self.group_by_length,
            lengths=self.lengths,
            batch_size=self.batch_size)
        params['sampler'] = self._epoch_sampler
        return params

    @remote_function()
    def set_epoch(self, epoch: int) -> None:
        """Read the next pass in epoch ``epoch``'s order. Call once per epoch, before iterating.

        Without this every epoch repeats the same order, since the order is derived from the seed. The
        dataloader has to be told which epoch it is because it does not count them itself: an epoch can
        end early, or be resumed part-way, and neither is visible from in here.

        Announcing an epoch other than the interrupted one also cancels a pending resume skip: the
        caller has moved on to an epoch that was never partly read, so there is nothing to skip into.
        """
        self._lazy_init_dataloader()
        if self._epoch_sampler is not None:
            self._epoch_sampler.set_epoch(epoch)
        if self._resume_pending and epoch != self._resume_epoch:
            self._resume_pending = False
            self._resume_offset = 0

    def _read_through(self, dataset):
        """The dataset as the torch DataLoader should see it, with the per-rank work already applied.

        Both wrappers exist for the same reason: they do their job inside the worker, at the point the
        data is actually read, rather than in a main-process hook that either reads everything twice or
        (for the stream) is not consulted at all once workers are involved.
        """
        from torch.utils.data import IterableDataset
        if isinstance(dataset, IterableDataset):
            return DeviceMeshDataset(
                dataset,
                self.batch_size,
                self.device_mesh,
                min_batch_size=self.min_batch_size,
                max_retries=self.max_retries)
        return RetryDataset(dataset, max_retries=self.max_retries, seed=self._retry_seed)

    @remote_function()
    def __iter__(self):
        self._lazy_init_dataloader()
        self._begin_pass()
        return self._tracking_iter(self.dataloader.__iter__())

    def _tracking_iter(self, inner):
        """Yield batches, counting the samples consumed as they go past.

        Counted by real width, not by ``batch_size``: a trailing batch is short, and `skip_samples` can
        trim one, so assuming the nominal width overcounts -- and this number is what a checkpoint stores
        and ``skip_consumed_samples`` later trusts, so overcounting means a resumed run skips samples it
        never read. The widths come from :class:`DeviceMeshSampler`, popped in the order the batches
        arrive; without one in the stack (an iterable dataset) the nominal width is all there is.
        """
        widths = getattr(self.dataloader.batch_sampler, 'emitted_batch_sizes', None)
        for batch in inner:
            self._consumed_train_samples += widths.popleft() if widths else self.batch_size
            yield batch

    @remote_function()
    def skip_consumed_samples(self, consumed_train_samples: int) -> None:
        """Resume: treat that many samples as already read, and continue where the run left off.

        The count is a total over the whole run, so it says two things, and ``divmod`` by the dataset
        length separates them: which epoch was interrupted, and how far into it. Skipping that many
        indices linearly instead -- which is what this did -- reads *nothing* once a run has passed its
        first epoch, because the skip then covers every index the epoch has. A resumed multi-epoch run
        therefore trained on an empty dataset and said nothing about it.

        The epoch matters beyond arithmetic: :class:`EpochSampler` derives each epoch's order from its
        number, so resuming into epoch 2 has to read epoch 2's order. Skipping into epoch 0's order
        would revisit samples already seen and never reach others.

        The skip applies to the next pass only. Nothing here counts epochs -- twinkle leaves the epoch
        loop to the caller -- so a skip left in place would re-apply to every later epoch, dropping the
        same head of the dataset over and over.
        """
        from torch.utils.data import IterableDataset

        self._resume_pending = False
        self._resume_epoch, self._resume_offset = 0, 0
        if isinstance(self.dataset, IterableDataset):
            warnings.warn('IterableDataset does not support consumed-data skipping; continuing without skipping.')
            return
        if not consumed_train_samples or consumed_train_samples <= 0:
            return

        consumed = int(consumed_train_samples)
        samples_per_epoch = max(len(self.dataset), 1)
        self._resume_epoch, self._resume_offset = divmod(consumed, samples_per_epoch)
        self._resume_pending = True
        self._consumed_train_samples = consumed

    @remote_function()
    def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
        self.skip_consumed_samples(consumed_train_samples)

    @remote_function()
    def get_state(self) -> dict:
        """The dataloader state for saving.

        ``resume_epoch`` is where a caller's own epoch loop should start after restoring this state --
        ``range(state['resume_epoch'], num_epochs)`` -- since replaying the epochs already finished
        would train on them twice. It is 0 for an iterable dataset, which has no epochs to count.
        """
        from torch.utils.data import IterableDataset
        resume_epoch = 0
        if not isinstance(self.dataset, IterableDataset):
            resume_epoch = self._consumed_train_samples // max(len(self.dataset), 1)
        return {'consumed_train_samples': self._consumed_train_samples, 'resume_epoch': resume_epoch}

    def _begin_pass(self) -> None:
        """Set up the pass about to start, and spend the resume position so the next one starts clean."""
        skip = 0
        if self._resume_pending:
            if self._epoch_sampler is not None:
                self._epoch_sampler.set_epoch(self._resume_epoch)
            skip = self._resume_offset
            self._resume_pending = False
            self._resume_offset = 0
        if skip != self._skip_samples:
            self._skip_samples = skip
            self.dataloader.__initialized = False
            self._rebuild_sampler_stack()
            self.dataloader.__initialized = True

    def _rebuild_sampler_stack(self):
        """Put the resume offset and the device-mesh slice back on top of the untouched base samplers.

        Both only decide which indices this rank takes; reading them is :class:`RetryDataset`'s job, so
        nothing here touches the dataset. Called again after :meth:`skip_consumed_samples`, which is why
        the bases are kept aside rather than rewrapped in place.
        """
        if self._base_batch_sampler is not None:
            self.dataloader.batch_sampler = DeviceMeshSampler(
                self._base_batch_sampler,
                self.device_mesh,
                self.min_batch_size,
                skip_samples=self._skip_samples,
            )
        elif self._base_sampler is not None:
            self.dataloader.sampler = SkipSampler(self._base_sampler, skip_samples=self._skip_samples)
