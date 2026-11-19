# Copyright (c) ModelScope Contributors. All rights reserved.
"""The step-delta probe's gradient-partition digest, averaged_gradients lane.

ZeRO-2 hands the CPU optimizer the reduced gradients in ``averaged_gradients``. This lane used to
record only a bare ``[sum, absmax, norm]`` aggregate, which can show that two runs differ but never
that they agree -- so a fresh/resume pair whose gradients matched only on the aggregate had no
identity criterion. The cases attack that gap and the ways the upgraded entry can still under-report.

PLAN
| # | case | input | expected | failure it kills |
|---|------|-------|----------|------------------|
| 1 | entry carries identity fields | averaged_gradients={0:[fp32 arange]} | dict agg[3]/bits[2]/numel + buffer identity; bits==bits_fingerprint(flat) | a bare aggregate with no identity criterion (the QUEUE2409/3213 gap) |
| 2 | content change detected | two grads differing in one non-max element | bits differ | reporting "gradients matched" over a real change |
| 3 | order independence | grad vs the same values permuted | bits equal | a fingerprint that moves with reduction order, unusable for comparison |
| 4 | sources registers names | averaged_gradients={0:[t],1:[t]} | sources lists both averaged names | a readable gradient dropped from sources, silently excluded from comparison |
| 5 | aliased buffer identity | two views into one storage | storage_ptr equal, ptr differ | the aliased-buffer shape of a divergence invisible to a content fingerprint |
| 6 | unreadable named | engine with neither partitions nor averaged_gradients | unreadable present, sources empty | two empty digests comparing equal, signing off as "gradients matched" |
"""
from types import SimpleNamespace

import torch

from twinkle.model.transformers.strategy import deepspeed as ds
from twinkle.model.transformers.strategy.deepspeed import DeepSpeedStrategy
from twinkle.utils.torch_utils import bits_fingerprint


def make_strategy(monkeypatch, averaged):
    """A DeepSpeedStrategy with no engine state, reading gradients from a fake engine.

    engine_grad_partition_digest touches only methods and class attributes, never ``__init__``
    state, so ``__new__`` gives a usable instance without a DeepSpeed config. ``_resolve_engine``
    normally insists on a real DeepSpeedEngine; pointing it at a fake optimizer exercises the
    averaged_gradients lane on real CPU tensors (where an offload tier's gradients actually live).
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    engine = SimpleNamespace(optimizer=SimpleNamespace(averaged_gradients=averaged))
    monkeypatch.setattr(ds, '_resolve_engine', lambda model: engine)
    return DeepSpeedStrategy.__new__(DeepSpeedStrategy)


def test_averaged_entry_carries_identity_fields(monkeypatch):
    grad = torch.arange(12, dtype=torch.float32)
    strategy = make_strategy(monkeypatch, {0: [grad]})
    out = strategy.engine_grad_partition_digest(object())
    entry = out['averaged_gradients[0][0]']
    assert isinstance(entry, dict), 'a bare aggregate has no identity criterion'
    assert len(entry['agg']) == 3
    assert len(entry['bits']) == 2
    assert entry['numel'] == 12
    assert entry['bits'] == bits_fingerprint(grad.flatten())
    for key in ('ptr', 'storage_ptr', 'device', 'dtype'):
        assert key in entry, f'buffer identity missing {key}'
    assert entry['dtype'] == 'torch.float32'


def test_content_change_is_detected_by_bits(monkeypatch):
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 2.0, 3.5, 4.0])  # one non-max element moved
    strategy = make_strategy(monkeypatch, {0: [a, b]})
    out = strategy.engine_grad_partition_digest(object())
    assert out['averaged_gradients[0][0]']['bits'] != out['averaged_gradients[0][1]']['bits']


def test_fingerprint_is_order_independent(monkeypatch):
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    permuted = grad[torch.tensor([3, 1, 0, 2])]
    strategy = make_strategy(monkeypatch, {0: [grad, permuted]})
    out = strategy.engine_grad_partition_digest(object())
    assert out['averaged_gradients[0][0]']['bits'] == out['averaged_gradients[0][1]']['bits']


def test_sources_registers_every_averaged_name(monkeypatch):
    grad = torch.ones(4)
    strategy = make_strategy(monkeypatch, {0: [grad], 1: [grad.clone()]})
    out = strategy.engine_grad_partition_digest(object())
    assert 'averaged_gradients[0][0]' in out['sources']
    assert 'averaged_gradients[1][0]' in out['sources']


def test_aliased_views_share_storage_but_not_ptr(monkeypatch):
    base = torch.arange(8, dtype=torch.float32)
    view_a = base[0:4]
    view_b = base[4:8]
    strategy = make_strategy(monkeypatch, {0: [view_a, view_b]})
    out = strategy.engine_grad_partition_digest(object())
    ea = out['averaged_gradients[0][0]']
    eb = out['averaged_gradients[0][1]']
    assert ea['storage_ptr'] == eb['storage_ptr']
    assert ea['ptr'] != eb['ptr']


def test_unreadable_gradient_is_named_not_empty(monkeypatch):
    strategy = make_strategy(monkeypatch, {})
    out = strategy.engine_grad_partition_digest(object())
    assert out['sources'] == []
    assert 'unreadable' in out
    assert 'averaged_gradients' in out['unreadable']