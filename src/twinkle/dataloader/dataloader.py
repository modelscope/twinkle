# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence, Type, Union

import twinkle.processor
from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.dataset import Dataset
from twinkle.processor import InputProcessor
from twinkle.utils import construct_class, get_logger
from .stream_shard_dataset import StreamShardDataset
from .sharded_batch_sampler import ShardedBatchSampler
from .epoch_sampler import EpochSampler
from .random_state import derive_seed, seed_process_rng
from .map_retry_dataset import MapRetryDataset
from .skip_sampler import SkipSampler

logger = get_logger()


class _IterableBatch(NamedTuple):
    value: Any
    source_size: int


class _IterableCollator:

    def __init__(self, collate_fn: Callable):
        self.collate_fn = collate_fn

    def __call__(self, samples: list) -> _IterableBatch:
        source_sizes = {sample.source_size for sample in samples}
        if len(source_sizes) != 1:
            raise RuntimeError('An iterable batch crossed source-group boundaries; its resume offset is ambiguous.')
        batch = self.collate_fn([sample.value for sample in samples])
        return _IterableBatch(batch, source_sizes.pop())


@remote_class(execute='first')
class DataLoader:
    """A DataLoader wrapper, will retry failed samples and return the data belongs to the current dp rank.

    Notes:
        Call :meth:`set_epoch` before each pass that should use a different deterministic order.

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
        max_retries: How many replacements to try for a sample that cannot be read. 0 substitutes
            nothing and lets the original failure through -- the strict mode, for a caller that would
            rather stop on a bad row than train on a different one. See :class:`MapRetryDataset`.
        retry_seed: Seeds the replacement draw, so a given failing index resolves to the same
            replacement on every rank and on every run. Defaults to the process-wide torch seed
            (see :meth:`_resolve_retry_seed`); pass the caller's own data seed to tie the two together.
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
                 data_sharding: bool = False,
                 retry_seed: Optional[int] = None,
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
        self.data_seed = int(data_seed or 0)
        self.group_by_length = group_by_length
        self.lengths = lengths
        self.data_sharding = data_sharding
        self._epoch = 0
        self._epoch_sampler: Optional[EpochSampler] = None
        # Checked here rather than where the sampler is built, which does not happen until the first
        # iteration: a misconfigured dataloader should say so when it is configured, not once a run has
        # already loaded a model and started.
        if group_by_length and self.lengths is None:
            raise ValueError('group_by_length needs `lengths`, one per sample.')
        self.processor: Optional[InputProcessor] = None
        self._skip_samples = 0
        self._stream_fast_forward = 0
        # Where an interrupted run left off, split into which epoch and how far into it. Pending until
        # the next pass reads it, because nothing here counts epochs -- see `skip_consumed_samples`.
        self._resume_epoch = 0
        self._resume_offset = 0
        self._resume_pending = False
        self._consumed_in_pass = 0
        self._consumed_train_samples = 0
        self._base_batch_sampler = None
        self._base_sampler = None
        self._data_generator = None
        # The iterable wrapper, kept because a stream carries its resume offset there rather than in a
        # sampler (see `_begin_pass`). None for a map-style dataset.
        self._stream_dataset: Optional[StreamShardDataset] = None
        self._map_retry_dataset: Optional[MapRetryDataset] = None
        self._retry_seed = self._resolve_retry_seed() if retry_seed is None else int(retry_seed)
        self._set_worker_init_fn()

    def _set_worker_init_fn(self):
        self.dataloader_params['worker_init_fn'] = partial(
            DataLoader._seed_worker,
            base_seed=self.data_seed,
            rank=(self.device_mesh.data_rank or 0) if self.device_mesh else 0)

    @staticmethod
    def _resolve_retry_seed() -> int:
        """The seed :class:`MapRetryDataset` draws replacements with, so they match across ranks.

        Reads the process-wide torch knob set -- the ``EXTRA_TORCH_KWARGS`` blob that ``initialize``
        writes and a Ray worker inherits through its runtime_env -- so the draw agrees everywhere
        without a caller having to thread a seed through. An explicit ``retry_seed`` wins over this.
        """
        from twinkle.infra import _torch_setup
        return _torch_setup.seed_from_env()

    @remote_function()
    def __len__(self):
        self._lazy_init_dataloader()
        return len(self.dataloader)

    @staticmethod
    def _seed_worker(worker_id: int, base_seed: int, rank: int):
        worker_seed = derive_seed(base_seed, rank, worker_id)
        seed_process_rng(worker_seed)

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
        if params.get('generator') is None:
            import torch
            self._data_generator = torch.Generator()
            self._data_generator.manual_seed(self.data_seed)
            params['generator'] = self._data_generator
        from torch.utils.data import IterableDataset
        if isinstance(self.dataset, IterableDataset):
            # `StreamShardDataset` has already taken this rank's slice out of each global batch, so what
            # reaches the DataLoader is this rank's share alone and is batched at that width. Batching
            # it at the global width again would make every batch `data_world_size` times too wide.
            if self.device_mesh is not None:
                params['batch_size'] = self.batch_size // self.device_mesh.data_world_size
            params['collate_fn'] = _IterableCollator(params['collate_fn'])
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
        if epoch != self._epoch:
            self._consumed_in_pass = 0
        self._epoch = epoch
        # data_sharding derives its per-bucket permutation from the epoch on the batch sampler itself
        # (there is no global EpochSampler order in that mode); a no-op on the default sampler.
        batch_sampler = getattr(self.dataloader, 'batch_sampler', None)
        if batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)
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
            self._stream_dataset = StreamShardDataset(
                dataset,
                self.batch_size,
                self.device_mesh,
                min_batch_size=self.min_batch_size,
                max_retries=self.max_retries,
                track_source_size=True,
                data_seed=self.data_seed)
            return self._stream_dataset
        self._map_retry_dataset = MapRetryDataset(dataset, max_retries=self.max_retries, seed=self._retry_seed)
        return self._map_retry_dataset

    @remote_function()
    def __iter__(self):
        self._lazy_init_dataloader()
        self._begin_pass()
        return self._tracking_iter(self.dataloader.__iter__())

    def _tracking_iter(self, inner):
        """Yield batches while recording only samples handed to the training loop."""
        widths = getattr(self.dataloader.batch_sampler, 'emitted_batch_sizes', None)
        remaining_fast_forward = self._stream_fast_forward
        for raw_batch in inner:
            if isinstance(raw_batch, _IterableBatch):
                batch = raw_batch.value
                width = raw_batch.source_size
            else:
                batch = raw_batch
                width = widths.popleft() if widths else self.batch_size
            if remaining_fast_forward:
                if width > remaining_fast_forward:
                    raise RuntimeError('The iterable resume offset does not align with a previously yielded batch.')
                remaining_fast_forward -= width
                continue
            self._consumed_in_pass += width
            self._consumed_train_samples += width
            yield batch
        if remaining_fast_forward:
            raise RuntimeError(f'The iterable dataset ended {remaining_fast_forward} samples before its resume offset.')
        self._stream_fast_forward = 0
        self._epoch += 1
        self._consumed_in_pass = 0

    @remote_function()
    def skip_consumed_samples(self, consumed_train_samples: int) -> None:
        """Restore an old checkpoint that only stored the cumulative consumed-sample count."""
        from torch.utils.data import IterableDataset

        consumed = max(0, int(consumed_train_samples or 0))
        if isinstance(self.dataset, IterableDataset):
            epoch, offset = 0, consumed
        else:
            epoch, offset = divmod(consumed, max(len(self.dataset), 1))
        self.load_state({
            'version': 1,
            'epoch': epoch,
            'offset': offset,
            'consumed_train_samples': consumed,
        })

    @remote_function()
    def load_state(self, state: dict) -> None:
        """Restore the pass and offset that the next iteration should continue from."""
        if not isinstance(state, dict):
            raise TypeError(f'DataLoader state must be a dict, got {type(state).__name__}.')
        version = int(state.get('version', 1))
        if version not in (1, 2):
            raise ValueError(f'Unsupported DataLoader state version: {version}.')
        if version >= 2:
            self.data_seed = int(state.get('data_seed', self.data_seed))
            self._retry_seed = int(state.get('retry_seed', self._retry_seed))
            set_data_seed = getattr(self.dataset, 'set_data_seed', None)
            if set_data_seed is not None:
                set_data_seed(self.data_seed)
            self._set_worker_init_fn()
            if self._data_generator is not None:
                self._data_generator.manual_seed(self.data_seed)
            if self.dataloader is not None:
                self.dataloader.worker_init_fn = self.dataloader_params['worker_init_fn']
            if self._epoch_sampler is not None:
                self._epoch_sampler.base_seed = self.data_seed
            if self._map_retry_dataset is not None:
                self._map_retry_dataset.seed = self._retry_seed
            if self._stream_dataset is not None:
                self._stream_dataset.set_data_seed(self.data_seed)
            batch_sampler = getattr(self.dataloader, 'batch_sampler', None)
            if batch_sampler is not None and hasattr(batch_sampler, 'data_seed'):
                batch_sampler.data_seed = self.data_seed
        epoch = int(state.get('epoch', 0))
        offset = int(state.get('offset', 0))
        consumed = int(state.get('consumed_train_samples', offset))
        if epoch < 0 or offset < 0 or consumed < offset:
            raise ValueError('DataLoader state requires non-negative epoch/offset and consumed >= offset.')
        # A map-style pass interrupted exactly at its end records (epoch, len(dataset)) -- the same
        # position a finished pass reports as (epoch + 1, 0). Normalize it here (the producer cannot:
        # get_state runs mid-pass) so the resume starts the next pass instead of skipping a whole
        # epoch and yielding nothing; already-written checkpoints restore correctly too.
        from torch.utils.data import IterableDataset
        if offset and not isinstance(self.dataset, IterableDataset):
            length = max(len(self.dataset), 1)
            if offset >= length:
                epoch, offset = epoch + offset // length, offset % length

        self._epoch = epoch
        if self._epoch_sampler is not None:
            self._epoch_sampler.set_epoch(epoch)
        batch_sampler = getattr(self.dataloader, 'batch_sampler', None)
        if batch_sampler is not None and hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)
        self._consumed_in_pass = offset
        self._consumed_train_samples = consumed
        self._resume_epoch = epoch
        self._resume_offset = offset
        self._resume_pending = True
        from torch.utils.data import IterableDataset
        if offset and isinstance(self.dataset, IterableDataset):
            logger.info(f'Resuming iterable dataset pass {epoch}: fast-forwarding past {offset} output rows; '
                        'they are read and encoded again before training continues.')

    @remote_function()
    def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
        self.skip_consumed_samples(consumed_train_samples)

    @remote_function(lazy_collect=False)
    def get_state(self) -> dict:
        """Return the next position and cumulative sample count as JSON-compatible state."""
        return {
            'version': 2,
            'epoch': self._epoch,
            'offset': self._consumed_in_pass,
            'consumed_train_samples': self._consumed_train_samples,
            'data_seed': self.data_seed,
            'retry_seed': self._retry_seed,
        }

    def _begin_pass(self) -> None:
        """Set up the pass about to start, and spend the resume position so the next one starts clean."""
        skip = 0
        if self._resume_pending:
            if self._epoch_sampler is not None:
                self._epoch_sampler.set_epoch(self._resume_epoch)
            self._epoch = self._resume_epoch
            skip = self._resume_offset
            self._resume_pending = False
            self._resume_offset = 0
        if self._data_generator is not None:
            self._data_generator.manual_seed(derive_seed(self.data_seed, self._epoch))
        if self._stream_dataset is not None:
            self._stream_dataset.set_epoch(self._epoch)
            self._skip_samples = skip
            num_workers = int(self.dataloader_params.get('num_workers', 0) or 0)
            if num_workers == 0:
                self._stream_dataset.skip_samples = skip
                self._stream_fast_forward = 0
            else:
                # Worker dataset replicas cannot observe a main-process attribute update, especially with
                # persistent workers. Replaying and discarding collated source groups preserves their order.
                self._stream_dataset.skip_samples = 0
                self._stream_fast_forward = skip
            return
        if skip == self._skip_samples:
            return
        self._skip_samples = skip
        self.dataloader.__initialized = False
        self._rebuild_sampler_stack()
        self.dataloader.__initialized = True

    def _rebuild_sampler_stack(self):
        """Put the resume offset and the device-mesh slice back on top of the untouched base samplers.

        Both only decide which indices this rank takes; reading them is :class:`MapRetryDataset`'s job, so
        nothing here touches the dataset. Called again after :meth:`skip_consumed_samples`, which is why
        the bases are kept aside rather than rewrapped in place.
        """
        if self._base_batch_sampler is not None:
            batch_sampler = ShardedBatchSampler(
                self._base_batch_sampler,
                self.device_mesh,
                self.min_batch_size,
                skip_samples=self._skip_samples,
                data_sharding=self.data_sharding,
                dataset_length=len(self.dataset),
                batch_size=self.batch_size,
                data_seed=self.data_seed,
            )
            batch_sampler.set_epoch(self._epoch)
            self.dataloader.batch_sampler = batch_sampler
        elif self._base_sampler is not None:
            self.dataloader.sampler = SkipSampler(self._base_sampler, skip_samples=self._skip_samples)
