# perf package tests (plan T4-01..T4-13): CounterRegistry, stats helpers, TimeRecorder gates.
import dataclasses
import threading

import pytest
import torch

from twinkle.perf.counters import CounterRegistry, default_counters
from twinkle.perf.recorder import TimeRecorder
from twinkle.perf.stats import describe_shape, iter_tensors, primary_tensor, tensor_moments, tensor_numel


class TestCounterRegistry:
    """T4-01..T4-05: accumulate, defaults, snapshot isolation, exception timing, concurrency."""

    def test_add_accumulates(self):
        counters = CounterRegistry()
        counters.add('rows', 1)
        counters.add('rows', 2)
        assert counters.get('rows') == 3.0

    def test_unknown_name_reads_zero(self):
        assert CounterRegistry().get('never-fed') == 0.0

    def test_snapshot_is_a_copy(self):
        counters = CounterRegistry()
        counters.add('k', 1)
        snap = counters.snapshot()
        snap['k'] = 99
        snap['new'] = 5
        assert counters.get('k') == 1.0
        assert counters.get('new') == 0.0

    def test_time_span_counts_exception_paths(self):
        counters = CounterRegistry()
        with pytest.raises(RuntimeError):
            with counters.time_span('boom'):
                raise RuntimeError('x')
        assert counters.get('boom') > 0.0

    def test_concurrent_adds_are_lossless(self):
        counters = CounterRegistry()

        def worker():
            for _ in range(1000):
                counters.add('hits', 1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counters.get('hits') == 8000.0

    def test_default_counters_is_a_registry(self):
        assert isinstance(default_counters, CounterRegistry)


class TestStatsHelpers:
    """T4-06..T4-12: tensor walking, empty-tensor guards, dtype stability, shape strings."""

    def test_iter_tensors_walks_nested_containers(self):
        @dataclasses.dataclass
        class Box:
            a: torch.Tensor
            b: torch.Tensor

        obj = {'x': torch.zeros(1), 'list': [torch.zeros(2), (torch.zeros(3), 'skip-me')],
               'box': Box(torch.zeros(4), torch.zeros(5))}
        shapes = [t.numel() for t in iter_tensors(obj)]
        assert shapes == [1, 2, 3, 4, 5]

    def test_primary_tensor_none_for_non_tensor(self):
        assert primary_tensor('just a string') is None

    def test_moments_empty_tensor_all_zero(self):
        out = tensor_moments(torch.zeros(0, 3))
        assert out == {'mean': 0.0, 'absmax': 0.0, 'l2': 0.0, 'std': 0.0}

    def test_moments_hand_computed(self):
        out = tensor_moments(torch.tensor([1.0, 2.0, 3.0]))
        assert out['mean'] == pytest.approx(2.0)
        assert out['absmax'] == pytest.approx(3.0)
        assert out['l2'] == pytest.approx(3.7416573867739413)
        assert out['std'] == pytest.approx(1.0)

    def test_moments_bfloat16_reduces_in_float32(self):
        # The reduction must promote to fp32 FIRST: the stats equal a manual .float() re-computation.
        big = torch.tensor([1e8, 1.0, -1e8], dtype=torch.bfloat16)
        out = tensor_moments(big)
        expected = big.float()
        assert out['mean'] == pytest.approx(expected.mean().item(), rel=1e-6)
        assert out['l2'] == pytest.approx(expected.norm().item(), rel=1e-6)

    def test_moments_single_element_std_zero(self):
        assert tensor_moments(torch.tensor([5.0]))['std'] == 0.0

    def test_tensor_numel_and_describe_shape(self):
        assert tensor_numel(torch.zeros(2, 8)) == 16.0
        assert tensor_numel('no tensor') == 0.0
        assert describe_shape(torch.zeros(2, 8, 1024)) == '2x8x1024'
        assert describe_shape([torch.zeros(2, 8), torch.zeros(4, 16)]) == '(2x8, 4x16)'
        assert describe_shape('nothing') == '-'


class TestTimeRecorder:
    """T4-13: span aggregation and the enable/disable gate."""

    def test_span_records_and_summary_lists_it(self):
        recorder = TimeRecorder(enabled=True)
        with recorder.span('train'):
            pass
        assert 'train' in recorder.summary()['spans']

    def test_disabled_recorder_no_op(self):
        recorder = TimeRecorder(enabled=False)
        with recorder.span('train'):
            pass
        assert recorder.summary()['spans'] == {}
        # Only the wall-clock key survives on an empty recorder.
        assert recorder.flat_summary() == {'perf/wall_s': 0.0}

    def test_enable_toggle_after_construction(self):
        recorder = TimeRecorder(enabled=False)
        recorder.enable()
        with recorder.span('a'):
            pass
        assert 'a' in recorder.summary()['spans']
        recorder.disable()
        with recorder.span('b'):
            pass
        assert 'b' not in recorder.summary()['spans']

    def test_timed_decorator_records(self):
        recorder = TimeRecorder(enabled=True)

        @recorder.timed('fn')
        def work():
            return 7

        assert work() == 7
        assert 'fn' in recorder.summary()['spans']

    def test_reset_clears_window(self):
        recorder = TimeRecorder(enabled=True)
        with recorder.span('a'):
            pass
        recorder.reset()
        assert recorder.summary()['spans'] == {}

    def test_trace_events_stream_one_per_closed_span(self):
        recorder = TimeRecorder(enabled=True)
        with recorder.span('one'):
            pass
        with recorder.span('two'):
            pass
        events = recorder.trace_events()
        assert len(events) == 2
        assert [e[2] for e in events] == ['one', 'two']
