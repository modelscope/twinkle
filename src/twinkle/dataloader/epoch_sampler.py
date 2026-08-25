# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Sampler


class EpochSampler(Sampler):
    """The order one epoch reads a map-style dataset in, as a function of ``(data_seed, epoch)``.

    Being a function of those two is the point. A DataLoader given ``shuffle=True`` builds torch's
    ``RandomSampler``, which draws from the global RNG: two runs of the same configuration read in
    different orders, and neither order can be recreated afterwards. That last part is what breaks
    resuming -- a run continuing from a checkpoint skips the right *number* of samples but then reads
    an order that has nothing to do with the one it is continuing, so some samples are seen twice in
    the epoch and others not at all. Here epoch ``n`` is always the same permutation, so a resumed
    epoch replays exactly, while consecutive epochs still differ from each other.

    ``group_by_length`` keeps samples of similar length in the same batch, so a batch is padded to
    something near its own longest member rather than to the longest in the dataset. It needs
    ``lengths`` and ``shuffle``: the grouping runs on a shuffled order, which is what keeps the batches
    from being identical every epoch despite being length-correlated.

    Data-parallel sharding is deliberately not here. :class:`DeviceMeshSampler` slices each batch
    across ranks, so this yields the whole global order and every rank derives the same one.

    Args:
        dataset_length: How many samples the dataset holds.
        shuffle: Whether to permute. Off gives the dataset's own order, and no seed is needed.
        data_seed: Base seed. ``None`` means 0, which is still reproducible -- unlike leaving it to the
            global RNG.
        group_by_length: Whether to batch samples of similar length together.
        lengths: One length per sample, required by ``group_by_length``. A list per sample (a
            multimodal sample measured per modality) is reduced to its largest, which is what decides
            the padding.
        batch_size: The global batch size, which ``group_by_length`` groups into.
    """

    def __init__(self,
                 dataset_length: int,
                 *,
                 shuffle: bool = True,
                 data_seed: Optional[int] = None,
                 group_by_length: bool = False,
                 lengths: Optional[Sequence] = None,
                 batch_size: int = 1):
        self.dataset_length = dataset_length
        self.shuffle = shuffle
        self.base_seed = data_seed or 0
        self.epoch = 0
        self.group_by_length = group_by_length
        self.batch_size = batch_size
        if group_by_length:
            if not shuffle:
                raise ValueError('group_by_length needs shuffle=True: it groups a shuffled order, and '
                                 'grouping the dataset order instead would fix the batches for the whole run.')
            if lengths is None:
                raise ValueError('group_by_length needs `lengths`, one per sample.')
        self.lengths: Optional[List[int]] = None
        if lengths is not None:
            self.lengths = [max(length) if isinstance(length, (list, tuple)) else length for length in lengths]

    def set_epoch(self, epoch: int) -> None:
        """Choose which epoch's order the next iteration is. Called once per epoch by the training loop."""
        self.epoch = epoch

    def __iter__(self):
        if not self.shuffle:
            return iter(range(self.dataset_length))
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self.epoch)
        if self.group_by_length:
            from transformers.trainer_pt_utils import get_length_grouped_indices
            return iter(get_length_grouped_indices(self.lengths, self.batch_size, generator=generator))
        return iter(torch.randperm(self.dataset_length, generator=generator).tolist())

    def __len__(self) -> int:
        return self.dataset_length
