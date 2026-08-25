# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any

import numpy as np

from twinkle.utils import get_logger

logger = get_logger()


class RetryDataset:
    """A map-style dataset that stands in a replacement sample when one fails to load.

    This wraps the dataset, not the sampler, and that is the whole point. A sampler cannot tell whether
    a sample loads without reading it, and a sampler only yields indices -- so the DataLoader reads
    every sample a second time to actually get it. Worse, samplers run in the main process while
    reading is what the workers are for, so the validating read is serial: against a lazy dataset,
    where ``__getitem__`` tokenizes text and decodes images, it doubles the work and puts half of it in
    front of the training loop, which is exactly the work ``num_workers`` was meant to move away. Here
    the retry sits at the one place the sample is read anyway -- in the worker, once.

    A failing index is replaced rather than dropped, so a batch keeps the width the caller asked for
    and an epoch stays ``len(dataset)`` samples long. Replacements come from a generator seeded by
    ``(seed, index)``: reproducible, and needing no state shared between workers, which a running
    counter would.

    Note this substitutes on a transient failure too, rather than insisting on the requested index. For
    training that is the better trade -- no single sample matters, and a flaky mount stalls nothing --
    but it does mean a run over a half-broken filesystem trains on a skewed sample instead of failing.
    The warnings say which indices were replaced.

    Args:
        dataset: The dataset to read through.
        max_retries: How many replacements to try before giving up on a batch position.
        seed: Seeds the replacement draw, so a given index resolves the same way across runs.
    """

    def __init__(self, dataset, max_retries: int = 20, seed: int = 42):
        self.dataset = dataset
        self.max_retries = max_retries
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        data = self._load(index)
        if data is not None:
            return data

        length = len(self.dataset)
        generator = np.random.RandomState((self.seed + int(index)) % 2**32)
        for _ in range(self.max_retries):
            replacement = int(generator.randint(length))
            data = self._load(replacement)
            if data is not None:
                logger.warning(f'Sample {index} could not be read; using sample {replacement} in its place.')
                return data
        raise RuntimeError(f'Sample {index} could not be read, and neither could any of {self.max_retries} '
                           f'replacements drawn for it. The dataset is likely unreadable rather than '
                           f'holding a few bad rows.')

    def _load(self, index: int) -> Any:
        """The sample at ``index``, or None if it is unreadable or empty.

        Empty counts as unreadable: a preprocessor that could not make sense of a row returns an empty
        result rather than raising, and a batch position holding nothing is no more usable than one
        that threw.
        """
        try:
            data = self.dataset[index]
        except Exception:  # noqa: BLE001
            logger.warning(f'Reading sample {index} failed.', exc_info=True)
            return None
        return data if data else None

    def __getattr__(self, name: str) -> Any:
        # Anything else the caller reaches for belongs to the dataset being wrapped; `dataset` itself is
        # a real attribute, so this cannot recurse through it.
        return getattr(self.dataset, name)
