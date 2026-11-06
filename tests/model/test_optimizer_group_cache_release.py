# Copyright (c) ModelScope Contributors. All rights reserved.
"""The DP-group metric gather has to release the active accelerator's cache.

``calculate_metrics`` releases the caching allocator before the metrics gather for
the reason documented in the method: the collective that backs the gather takes its
own device memory outside torch's caching allocator, so whichever rank holds the
least free memory raises inside it while the other ranks wait there forever.  The
release used to be written as ``torch.cuda.empty_cache()`` behind a CUDA-only guard,
which never runs on an Ascend NPU; these tests pin the release to the framework
helper that covers every accelerator.
"""
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.utils.framework import Torch


class _FakeMetric:

    def accumulate(self, inputs, outputs, **kwargs):
        pass

    def calculate(self):
        return {'fake': 1}


def _make_group(metrics):
    kwargs = dict(metrics=metrics, inputs=[1], outputs=object())
    return BaseOptimizerGroup(train_status=TrainStatus(**kwargs), eval_status=TrainStatus(**kwargs))


def test_metric_gather_releases_the_device_cache(monkeypatch):
    released = []
    monkeypatch.setattr(Torch, 'empty_cache', lambda: released.append(True))

    result = _make_group([_FakeMetric()]).calculate_metrics(is_training=True)

    assert result == {'fake': 1}
    assert released, 'calculate_metrics must release the device cache before the metrics gather'


def test_no_device_cache_release_without_metrics(monkeypatch):
    released = []
    monkeypatch.setattr(Torch, 'empty_cache', lambda: released.append(True))

    result = _make_group([]).calculate_metrics(is_training=True)

    assert result == {}
    assert not released
