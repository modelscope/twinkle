# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from torch.utils.data import IterableDataset, Sampler

from twinkle.dataset import Dataset


class RetrySampler(Sampler):
    """A sampler to retry the failed items.

    Args:
        original_sampler: The original sampler.
        dataset: The original dataset.
        max_retries: The maximum number of retries.
    """

    def __init__(self,
                 original_sampler: Sampler,
                 dataset: Dataset,
                 max_retries=20,
                 skip_samples: int = 0,
                 seed: int = 42):
        self.original_sampler = original_sampler
        self.dataset = dataset
        self.max_retries = max_retries
        self.skip_samples = skip_samples
        self.seed = int(seed)

    def __iter__(self):
        emitted = 0
        seen_valid = 0
        target_total = max(len(self.dataset) - self.skip_samples, 0)
        for idx in self.original_sampler:
            for _ in range(self.max_retries):
                try:
                    assert not isinstance(self.dataset, IterableDataset)
                    # Skip None values and raises
                    data = self.dataset[idx]
                    if not data:
                        continue
                    seen_valid += 1
                    if seen_valid <= self.skip_samples:
                        break
                    yield idx
                    emitted += 1
                    break
                except Exception:  # noqa
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                raise RuntimeError(f'Max retries exceeded: {self.max_retries}, no valid data found.')

        if emitted >= target_total:
            return

        for idx in np.random.RandomState(self.seed).permutation(len(self.dataset)).tolist():
            if emitted >= target_total:
                return
            for _ in range(self.max_retries):
                try:
                    # Skip None values and raises
                    data = self.dataset[idx]
                    if not data:
                        continue
                    yield idx
                    emitted += 1
                    break
                except Exception:  # noqa
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                raise ValueError(f'Max retries exceeded: {self.max_retries}, no valid data found.')

    def __len__(self):
        return len(self.dataset)
