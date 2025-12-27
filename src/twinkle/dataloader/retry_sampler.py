import numpy as np


class RetrySampler:
    def __init__(self, original_sampler, dataset, max_retries=50):
        self.original_sampler = original_sampler
        self.dataset = dataset
        self.max_retries = max_retries

    def __iter__(self):
        total = 0
        for idx in self.original_sampler:
            for _ in range(self.max_retries):
                try:
                    data = self.dataset[idx]
                    if not data:
                        continue
                    yield idx
                    total += 1
                except Exception: # noqa
                    continue
            else:
                raise ValueError(f'Max retries exceeded: {self.max_retries}, no valid data found.')

        if total >= len(self.original_sampler):
            return

        for idx in np.random.RandomState().permutation(len(self.dataset)).tolist():
            if total >= len(self.original_sampler):
                break
            for _ in range(self.max_retries):
                try:
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
        return len(self.original_sampler)