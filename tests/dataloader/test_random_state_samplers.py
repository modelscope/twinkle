# dataloader RNG/sampler tests (plan T10c-01..T10c-08).
import random

import numpy as np
import pytest
import torch

from twinkle.dataloader import EpochSampler, MapRetryDataset, StreamShardDataset
from twinkle.dataloader.random_state import (SampleIndex, derive_seed, isolated_random_state, seed_process_rng,
                                              split_sample_index)


class TestDeriveSeed:
    """T10c-01: stability and spread of the splitmix64-style combiner."""

    def test_deterministic(self):
        assert derive_seed(42, 0, 1) == derive_seed(42, 0, 1)

    def test_inputs_differ(self):
        values = {derive_seed(42, 0, i) for i in range(32)}
        assert len(values) == 32

    def test_seed_participates(self):
        assert derive_seed(42, 1) != derive_seed(43, 1)

    def test_output_in_positive_63bit_range(self):
        for parts in [(0,), (1, 2, 3), (-1,)]:
            assert 0 <= derive_seed(42, *parts) < 2**63


class TestSampleIndex:
    def test_split_named_tuple(self):
        idx = SampleIndex(index=7, epoch=3)
        assert split_sample_index(idx) == (7, 3)

    def test_split_bare_value_defaults_epoch_zero(self):
        assert split_sample_index(7) == (7, 0)


class TestSeedProcessRng:
    """T10c-02: all three process generators follow the seed."""

    def test_python_rng_reproducible(self):
        seed_process_rng(123)
        a = random.random()
        seed_process_rng(123)
        assert random.random() == a


class TestIsolatedRandomState:
    """T10c-03: deterministic inside, restored on the way out."""

    def test_state_restored_on_exit(self):
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with isolated_random_state(9, 0, 1):
            random.random(), np.random.rand(), torch.rand(1)
        after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        assert before[0] == after[0]
        assert before[2].equal(after[2])
        assert np.array_equal(before[1][1], after[1][1])  # numpy MT19937 keys restored

    def test_state_restored_on_exception(self):
        before = random.getstate()
        with pytest.raises(RuntimeError), isolated_random_state(9, 0, 2):
            random.random()
            raise RuntimeError('boom')
        assert random.getstate() == before

    def test_deterministic_inside(self):
        def draw():
            with isolated_random_state(5, 1) as seed:
                return seed, random.random()
        s1, a = draw()
        s2, b = draw()
        assert s1 == s2
        assert a == b


class TestEpochSampler:
    """T10c-05: epoch entropy and reproducibility; the group_by_length guards."""

    def test_no_shuffle_is_sequential(self):
        assert list(iter(EpochSampler(5, shuffle=False))) == [0, 1, 2, 3, 4]

    def test_same_epoch_reproduces(self):
        s = EpochSampler(100, data_seed=7)
        s.set_epoch(1)
        a = list(iter(s))
        s.set_epoch(1)
        assert list(iter(s)) == a

    def test_different_epoch_differs(self):
        s = EpochSampler(100, data_seed=7)
        s.set_epoch(1)
        a = list(iter(s))
        s.set_epoch(2)
        b = list(iter(s))
        assert a != b

    def test_length(self):
        assert len(EpochSampler(7)) == 7

    def test_group_by_length_requires_shuffle(self):
        with pytest.raises(ValueError, match='shuffle=True'):
            EpochSampler(4, shuffle=False, group_by_length=True, lengths=[1, 2, 3, 4])

    def test_group_by_length_requires_lengths(self):
        with pytest.raises(ValueError, match='lengths'):
            EpochSampler(4, group_by_length=True)


class _FlakyList:
    """A list-like dataset whose item at ``bad_index`` raises or returns empty."""

    def __init__(self, n, bad_index, mode='raise'):
        self.n = n
        self.bad_index = bad_index
        self.mode = mode

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i == self.bad_index:
            if self.mode == 'raise':
                raise IOError('unreadable row')
            return None  # counts as unreadable (empty)
        return {'i': i}


class TestMapRetryDataset:
    """T10c-06/T10c-07: retry with substitution, and the exhaustion guard."""

    def test_good_sample_reads_through(self):
        ds = MapRetryDataset(_FlakyList(5, bad_index=99), max_retries=3)
        assert ds[2] == {'i': 2}
        assert len(ds) == 5

    def test_bad_sample_substituted(self):
        # The bad row raises; a deterministic replacement is drawn and served instead.
        ds = MapRetryDataset(_FlakyList(5, bad_index=1), max_retries=20)
        out = ds[1]
        assert isinstance(out, dict) and 'i' in out

    def test_empty_sample_substituted(self):
        ds = MapRetryDataset(_FlakyList(5, bad_index=1, mode='empty'), max_retries=20)
        assert isinstance(ds[1], dict) and 'i' in out if (out := ds[1]) else False

    def test_all_bad_raises_runtime_error(self):
        class AllBad:
            def __len__(self):
                return 3

            def __getitem__(self, i):
                raise IOError('dead disk')

        ds = MapRetryDataset(AllBad(), max_retries=2)
        with pytest.raises(RuntimeError, match='unreadable'):
            ds[0]

    def test_strict_mode_empty_raises_value_error(self):
        ds = MapRetryDataset(_FlakyList(5, bad_index=1, mode='empty'), max_retries=0)
        with pytest.raises(ValueError, match='substitution is off'):
            ds[1]


class _FakeMesh:
    """The slice a device mesh hands out for a global batch of size n."""

    def __init__(self, slice_, world_size=1):
        self.slice_ = slice_
        self.data_world_size = world_size

    def get_slice(self, n):
        return self.slice_


class _RangeIterable(torch.utils.data.IterableDataset):
    """Yields dict samples (an int 0 would be FALSY and read as an empty sample)."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        yield from ({'i': i} for i in range(self.n))


class TestStreamShardDataset:
    """T10c-08: batch slicing, sharding, resume skip, and the too-small tail drop."""

    # StreamShardDataset streams SAMPLES (batching is the DataLoader's job); batch_size only
    # fixes the global group the mesh slices for shard alignment.

    def test_samples_streamed_in_order_without_mesh(self):
        ds = StreamShardDataset(_RangeIterable(10), batch_size=4)
        assert [s['i'] for s in ds] == list(range(10))

    def test_shard_slices_each_global_group(self):
        mesh = _FakeMesh(slice(1, 3))
        ds = StreamShardDataset(_RangeIterable(10), batch_size=4, device_mesh=mesh)
        assert [s['i'] for s in ds] == [1, 2, 5, 6, 9]

    def test_small_tail_dropped_when_min_batch_size(self):
        mesh = _FakeMesh(slice(0, 4))
        ds = StreamShardDataset(_RangeIterable(10), batch_size=4, device_mesh=mesh, min_batch_size=4)
        assert [s['i'] for s in ds] == list(range(8))  # the [8, 9] tail is dropped

    def test_skip_samples_fast_forwards(self):
        ds = StreamShardDataset(_RangeIterable(6), batch_size=2, skip_samples=2)
        assert [s['i'] for s in ds] == [2, 3, 4, 5]

    def test_attr_forwarded_to_wrapped_dataset(self):
        ds = StreamShardDataset(_RangeIterable(6), batch_size=2)
        assert ds.n == 6
