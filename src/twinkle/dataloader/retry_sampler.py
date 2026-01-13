# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from torch.utils.data import Sampler

from twinkle.dataset import Dataset


class RetrySampler(Sampler):
    """A sampler to retry the failed items.

    Args:
        original_sampler: The original sampler.
        dataset: The original dataset.
        max_retries: The maximum number of retries.
    """

    def __init__(self, original_sampler: Sampler, dataset: Dataset, max_retries=20):
        self.original_sampler = original_sampler
        self.dataset = dataset
        self.max_retries = max_retries

    def __iter__(self):
        total = 0
        for idx in self.original_sampler:
            for _ in range(self.max_retries):
                try:
                    # Skip None values and raises
                    data = self.dataset[idx]
                    if not data:
                        continue
                    yield idx
                    total += 1
                    break
                except Exception: # noqa
                    continue
            else:
                raise ValueError(f'Max retries exceeded: {self.max_retries}, no valid data found.')

        if hasattr(self.dataset, '__len__'):
            origin_dataset_len = len(self.dataset)
            if total >= origin_dataset_len:
                return

            for idx in np.random.RandomState().permutation(len(self.dataset)).tolist():
                if total >= origin_dataset_len:
                    break
                for _ in range(self.max_retries):
                    try:
                        # Skip None values and raises
                        data = self.dataset[idx]
                        if not data:
                            continue
                        yield idx
                        total += 1
                    except Exception: # noqa
                        continue
                else:
                    raise ValueError(f'Max retries exceeded: {self.max_retries}, no valid data found.')

    def __len__(self):
        return len(self.dataset)