# Copyright (c) ModelScope Contributors. All rights reserved.
import multiprocessing as mp
from typing import Any, Iterator, NamedTuple, Optional

from torch.utils.data import IterableDataset, get_worker_info

from twinkle.utils import get_logger
from .random_state import isolated_random_state

logger = get_logger()


class _IterableSample(NamedTuple):
    value: Any
    source_size: int


class StreamShardDataset(IterableDataset):
    """One data-parallel rank's share of an iterable dataset, taken where the data is actually read.

    The peer of :class:`MapRetryDataset`: both wrap the dataset rather than hooking the DataLoader, because
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
        track_source_size: Attach the source-group width to each yielded sample so the owning DataLoader
            can count and replay the exact global stream position after collation.
        skip_samples: Rows of the underlying stream to discard from the head of the NEXT pass, for a
            resume. Counted in GLOBAL rows -- the unit the consumed-sample count is stated in -- because
            every rank reads the whole stream and takes its slice of each group, so all ranks skip the
            same head and stay aligned. Mutable: the dataloader sets it per pass and clears it after,
            since a stream has no sampler stack to rebuild (see ``DataLoader._begin_pass``).
    """

    def __init__(self,
                 dataset: IterableDataset,
                 batch_size: int,
                 device_mesh: Optional[Any] = None,
                 min_batch_size: Optional[int] = None,
                 max_retries: int = 20,
                 track_source_size: bool = False,
                 skip_samples: int = 0,
                 data_seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device_mesh = device_mesh
        self.max_retries = max_retries
        self.track_source_size = track_source_size
        self.skip_samples = skip_samples
        self.data_seed = int(data_seed or 0)
        self._epoch = mp.Value('q', 0)
        self.min_batch_size = min_batch_size
        if self.min_batch_size is None and device_mesh is not None:
            self.min_batch_size = device_mesh.data_world_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)
        set_epoch = getattr(self.dataset, 'set_epoch', None)
        if set_epoch is not None:
            set_epoch(epoch)

    def set_data_seed(self, data_seed: int) -> None:
        self.data_seed = int(data_seed or 0)
        set_data_seed = getattr(self.dataset, 'set_data_seed', None)
        if set_data_seed is not None:
            set_data_seed(self.data_seed)

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
        source_size = len(group)
        if self.device_mesh is None:
            result = group
        else:
            result = group[self.device_mesh.get_slice(source_size)]
        if not self.track_source_size:
            return result
        return [_IterableSample(sample, source_size) for sample in result]

    def _read(self) -> Iterator:
        """The data, with a resume offset applied and unreadable samples skipped.

        Skipped rather than replaced, unlike :class:`MapRetryDataset`: there is no index here to draw a
        replacement by. A sample that all ranks fail on keeps them aligned; one that fails on a single
        rank -- a flaky read -- shifts that rank's grouping by one, which is why only a run of failures is
        treated as fatal rather than any single one.

        The resume offset is a fast-forward, not a restore: the skipped rows are read and decoded again
        and then thrown away, because a stream keeps no random-access position to seek to. Skipped rows
        are not counted as failures.
        """
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        epoch = self._epoch.value
        with isolated_random_state(self.data_seed, epoch, worker_id, -1):
            iterator = iter(self.dataset)
        source_position = 0
        failures = 0
        skipped = 0
        while True:
            try:
                with isolated_random_state(self.data_seed, epoch, worker_id, source_position):
                    sample = next(iterator)
            except StopIteration:
                return
            except Exception:  # noqa: BLE001
                failures += 1
                logger.warning(f'Reading a sample failed ({failures} in a row).', exc_info=True)
                if failures > self.max_retries:
                    raise RuntimeError(f'{failures} consecutive samples could not be read; treating the '
                                       'dataset as unreadable rather than skipping indefinitely.')
                source_position += 1
                continue
            source_position += 1
            failures = 0
            if not sample:
                continue
            if skipped < self.skip_samples:
                skipped += 1
                continue
            yield sample

    def __getattr__(self, name: str) -> Any:
        # Everything else belongs to the dataset being wrapped; `dataset` is a real attribute, so this
        # cannot recurse through it.
        return getattr(self.dataset, name)
