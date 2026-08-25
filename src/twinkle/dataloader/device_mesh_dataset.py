# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Iterator, Optional

from torch.utils.data import IterableDataset

from twinkle.utils import get_logger

logger = get_logger()


class DeviceMeshDataset(IterableDataset):
    """One data-parallel rank's share of an iterable dataset, taken where the data is actually read.

    The peer of :class:`RetryDataset`: both wrap the dataset rather than hooking the DataLoader, because
    the wrapped dataset is the one thing a worker process is guaranteed to go through.

    This used to be a torch dataset *fetcher*, installed by assigning over the DataLoader iterator's
    ``_dataset_fetcher``. That attribute exists only on the single-process iterator; with
    ``num_workers > 0`` each worker builds its own fetcher inside its own process, so the assignment just
    added a field nobody reads -- silently, since Python allows it. The slicing then never happened and
    every rank trained on the whole stream: the same samples on every GPU, at ``data_world_size`` times
    the intended batch width.

    The division itself is unchanged: read a whole global batch, keep this rank's slice of it.
    Deliberately not "give each rank its own shard of the stream", which reads less but lets ranks end up
    with different numbers of batches -- and a rank that runs out while its peers have more hangs the job
    at the next collective. Grouping first makes every rank's batch count identical by construction,
    since they all group the same data the same way.

    The cost is that every rank reads everything and drops what is not its slice. That is inherent to
    dividing a stream whose layout is unknown, and was equally true of the fetcher.

    Workers are left to the wrapped dataset, per torch's contract. A HuggingFace ``IterableDataset``
    divides its shards among them, and every rank divides them identically, so ranks stay aligned. A
    hand-written iterable dataset that ignores ``get_worker_info`` yields its full contents in every
    worker -- ``num_workers`` copies of the data -- but that is true of it with or without this wrapper.

    Args:
        dataset: The iterable dataset to divide.
        batch_size: The *global* batch size, the unit each slice is taken from.
        device_mesh: Supplies this rank's data-parallel coordinate. ``None`` means no division.
        min_batch_size: Drop a trailing group narrower than this. Defaults to the number of ranks, below
            which a group cannot give every rank a sample.
        max_retries: How many consecutive unreadable samples to tolerate before giving up.
    """

    def __init__(self,
                 dataset: IterableDataset,
                 batch_size: int,
                 device_mesh: Optional[Any] = None,
                 min_batch_size: Optional[int] = None,
                 max_retries: int = 20):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device_mesh = device_mesh
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size
        if self.min_batch_size is None and device_mesh is not None:
            self.min_batch_size = device_mesh.data_world_size

    def __iter__(self) -> Iterator:
        group = []
        for sample in self._read():
            group.append(sample)
            if len(group) < self.batch_size:
                continue
            yield from self._slice(group)
            group = []
        if group and not (self.min_batch_size and len(group) < self.min_batch_size):
            yield from self._slice(group)

    def _slice(self, group: list) -> list:
        """This rank's part of one global batch."""
        if self.device_mesh is None:
            return group
        return group[self.device_mesh.get_slice(len(group))]

    def _read(self) -> Iterator:
        """The data, with unreadable samples skipped.

        Skipped rather than replaced, unlike :class:`RetryDataset`: there is no index here to draw a
        replacement by. A sample that all ranks fail on keeps them aligned; one that fails on a single
        rank -- a flaky read -- shifts that rank's grouping by one, which is why only a run of failures is
        treated as fatal rather than any single one.
        """
        iterator = iter(self.dataset)
        failures = 0
        while True:
            try:
                sample = next(iterator)
            except StopIteration:
                return
            except Exception:  # noqa: BLE001
                failures += 1
                logger.warning(f'Reading a sample failed ({failures} in a row).', exc_info=True)
                if failures > self.max_retries:
                    raise RuntimeError(f'{failures} consecutive samples could not be read; treating the '
                                       'dataset as unreadable rather than skipping indefinitely.')
                continue
            failures = 0
            if sample:
                yield sample

    def __getattr__(self, name: str) -> Any:
        # Everything else belongs to the dataset being wrapped; `dataset` is a real attribute, so this
        # cannot recurse through it.
        return getattr(self.dataset, name)
