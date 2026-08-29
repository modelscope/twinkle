# Copyright (c) ModelScope Contributors. All rights reserved.
from collections import deque

from torch.utils.data import BatchSampler

from twinkle import DeviceMesh


class DeviceMeshSampler(BatchSampler):
    """A sampler that returns the slice of the current dp rank.

    Two modes:

    - default (``data_sharding=False``): slice each incoming global batch across dp ranks. The base
      sampler decides the (global) order and this only takes this rank's share of every batch, so the
      permutation is global and every rank derives it from the same seed.
    - ``data_sharding=True``: the Megatron per-dp-rank scheme (mirrors
      ``MegatronPretrainingRandomSampler``'s data-sharding branch). Each rank owns a contiguous bucket
      ``[rank*bucket, (rank+1)*bucket)`` of the dataset and permutes only WITHIN it, so no global order
      is materialized. Opt-in and Megatron-only; it needs ``dataset_length`` and ``batch_size`` (the
      global batch, split into per-rank micro batches) and ignores ``original_sampler``'s order.

    Args:
        original_sampler: The original BatchSampler (the global-order batches; unused when
            ``data_sharding=True``).
        device_mesh: The device mesh.
        min_batch_size: Drop a trailing global batch narrower than this (defaults to the dp world size).
        skip_samples: Resume offset, in GLOBAL samples, to skip from the start of the pass.
        data_sharding: Enable the per-dp-rank contiguous-bucket scheme above.
        dataset_length: Total number of samples (required by ``data_sharding``).
        batch_size: The global batch size (required by ``data_sharding``); the per-rank micro batch is
            ``batch_size // data_world_size``.
        data_seed: Base seed for the per-bucket permutation (combined with the epoch).
    """

    def __init__(self,
                 original_sampler: BatchSampler,
                 device_mesh: DeviceMesh,
                 min_batch_size: int = None,
                 skip_samples: int = 0,
                 *,
                 data_sharding: bool = False,
                 dataset_length: int = None,
                 batch_size: int = None,
                 data_seed: int = None):
        self.original_sampler = original_sampler
        self.device_mesh = device_mesh
        self.min_batch_size = min_batch_size
        self.skip_samples = skip_samples
        self.emitted_batch_sizes = deque()
        if self.min_batch_size is None and self.device_mesh is not None:
            self.min_batch_size = self.device_mesh.data_world_size
        # data_sharding state (opt-in; see class docstring).
        self.data_sharding = data_sharding
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.data_seed = data_seed or 0
        self.epoch = 0
        if data_sharding and (device_mesh is None or dataset_length is None or batch_size is None):
            raise ValueError('data_sharding needs device_mesh, dataset_length and batch_size.')

    def set_epoch(self, epoch: int) -> None:
        """Choose which epoch's per-bucket permutation ``data_sharding`` uses. No-op otherwise."""
        self.epoch = epoch

    def __iter__(self):
        self.emitted_batch_sizes.clear()
        if self.data_sharding:
            yield from self._iter_data_sharding()
            return
        skipped = 0
        for batch in self.original_sampler:
            if skipped < self.skip_samples:
                if skipped + len(batch) <= self.skip_samples:
                    skipped += len(batch)
                    continue
                batch = batch[self.skip_samples - skipped:]
                skipped = self.skip_samples

            if self.min_batch_size is not None and len(batch) < self.min_batch_size:
                return
            self.emitted_batch_sizes.append(len(batch))
            if not self.device_mesh:
                yield batch
            else:
                yield batch[self.device_mesh.get_slice(len(batch))]

    def _iter_data_sharding(self):
        """Yield this rank's own bucket, permuted within it (Megatron data-sharding).

        The trailing partial micro batch is dropped, matching Megatron's global-batch invariant (the
        legacy ``MegatronPretrainingRandomSampler`` drops it too). ``emitted_batch_sizes`` records the
        GLOBAL width (``per_rank_bs * dp_world``) so the DataLoader's consumed-sample count matches the
        non-data-sharding path, which appends the pre-slice global width.
        """
        import torch
        dp_world = self.device_mesh.data_world_size
        dp_rank = self.device_mesh.data_rank or 0
        bucket_size = self.dataset_length // dp_world
        per_rank_bs = max(1, self.batch_size // dp_world)
        start_idx = dp_rank * bucket_size
        # skip_samples counts GLOBAL samples already read this epoch; each rank has read
        # skip_samples // dp_world of its own bucket (legacy: bucket_offset = current // dp).
        bucket_offset = self.skip_samples // dp_world
        generator = torch.Generator()
        generator.manual_seed(self.data_seed + self.epoch)
        random_idx = torch.randperm(bucket_size, generator=generator).tolist()

        batch = []
        for i in random_idx[bucket_offset:]:
            batch.append(start_idx + i)
            if len(batch) == per_rank_bs:
                self.emitted_batch_sizes.append(per_rank_bs * dp_world)
                yield batch
                batch = []

    def __len__(self):
        if self.data_sharding:
            dp_world = self.device_mesh.data_world_size
            bucket_size = self.dataset_length // dp_world
            per_rank_bs = max(1, self.batch_size // dp_world)
            return bucket_size // per_rank_bs
        return len(self.original_sampler)
