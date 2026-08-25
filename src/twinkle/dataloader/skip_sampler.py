# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import Sampler


class SkipSampler(Sampler):
    """A sampler's indices with the first ``skip_samples`` of them left out, to resume mid-epoch.

    It counts indices, not samples successfully read. Those used to differ, back when the sampler read
    each sample to check it and skipped the unreadable ones -- which is why skipping had to read too.
    :class:`RetryDataset` now answers every index with a sample, so the counts are the same and
    resuming costs nothing.

    Args:
        original_sampler: The sampler whose order is resumed.
        skip_samples: How many of its indices were already consumed.
    """

    def __init__(self, original_sampler: Sampler, skip_samples: int = 0):
        self.original_sampler = original_sampler
        self.skip_samples = max(int(skip_samples), 0)

    def __iter__(self):
        for position, index in enumerate(self.original_sampler):
            if position >= self.skip_samples:
                yield index

    def __len__(self):
        return max(len(self.original_sampler) - self.skip_samples, 0)
