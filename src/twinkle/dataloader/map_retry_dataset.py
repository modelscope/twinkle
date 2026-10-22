# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any

import numpy as np

from twinkle.utils import get_logger
from .random_state import derive_seed, isolated_random_state, split_sample_index

logger = get_logger()


class MapRetryDataset:
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
    counter would. Being a pure function of those two is also what makes it resumable -- a run
    continuing from a checkpoint resolves a given failing index to the same replacement it would have,
    because nothing about the draw depends on how many failures came before it.

    ``max_retries=0`` turns substitution off: the read is unguarded, so the first failure propagates as
    itself. That is the mode a caller wants when a bad row is a defect to stop for rather than one to
    train around, and the original exception carries the reason (a too-long row says so) which the
    ``RuntimeError`` at the end of the retry path does not.

    Note this substitutes on a transient failure too, rather than insisting on the requested index. For
    training that is the better trade -- no single sample matters, and a flaky mount stalls nothing --
    but it does mean a run over a half-broken filesystem trains on a skewed sample instead of failing.
    The warnings say which indices were replaced.

    Args:
        dataset: The dataset to read through.
        max_retries: How many replacements to try before giving up on a batch position. 0 substitutes
            nothing and lets the original failure through.
        seed: Seeds the replacement draw, so a given index resolves the same way across runs.
    """

    def __init__(self, dataset, max_retries: int = 20, seed: int = 42):
        self.dataset = dataset
        self.max_retries = int(max_retries)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        index, epoch = split_sample_index(index)
        if self.max_retries <= 0:
            # Strict: no substitute and no guard, so the caller sees the exception the dataset itself
            # raised rather than a wrapper reporting only that something could not be read.
            with isolated_random_state(self.seed, epoch, int(index), 0):
                data = self.dataset[index]
            if not data:
                raise ValueError(f'Sample {index} read back empty, and substitution is off '
                                 '(max_retries=0), so there is nothing to serve at that position.')
            return data
        data = self._load(index, epoch, index, 0)
        if data is not None:
            return data

        length = len(self.dataset)
        generator = np.random.RandomState(derive_seed(self.seed, epoch, int(index)) % 2**32)
        for attempt in range(1, self.max_retries + 1):
            replacement = int(generator.randint(length))
            data = self._load(replacement, epoch, index, attempt)
            if data is not None:
                logger.warning(f'Sample {index} could not be read; using sample {replacement} in its place.')
                return data
        raise RuntimeError(f'Sample {index} could not be read, and neither could any of {self.max_retries} '
                           f'replacements drawn for it. The dataset is likely unreadable rather than '
                           f'holding a few bad rows.')

    def _load(self, index: int, epoch: int, logical_index: int, attempt: int) -> Any:
        """The sample at ``index``, or None if it is unreadable or empty.

        Empty counts as unreadable: a preprocessor that could not make sense of a row returns an empty
        result rather than raising, and a batch position holding nothing is no more usable than one
        that threw.
        """
        try:
            with isolated_random_state(self.seed, epoch, int(logical_index), attempt):
                data = self.dataset[index]
        except Exception:  # noqa: BLE001
            logger.warning(f'Reading sample {index} failed.', exc_info=True)
            return None
        return data if data else None

    def __getitems__(self, indexes: Any) -> list:
        # torch's batched fetch calls this INSTEAD of per-index __getitem__; letting __getattr__
        # forward it to the wrapped dataset would bypass both the SampleIndex unwrap and the retry
        # substitution (an HF Dataset answers it with a raw batched read), so serve it per index.
        return [self[index] for index in indexes]

    def __getattr__(self, name: str) -> Any:
        # Anything else the caller reaches for belongs to the dataset being wrapped; `dataset` itself is
        # a real attribute, so this cannot recurse through it.
        return getattr(self.dataset, name)
