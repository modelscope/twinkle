# Copyright (c) ModelScope Contributors. All rights reserved.
"""The clip-coefficient probe channel must record the factor the engine actually applies.

``combined_scale`` is local to DeepSpeed's ``unscale_and_clip_grads``; the probe mirrors its
expression from the same live attributes and the actual ``total_norm`` argument. The oracle
here is the REAL DeepSpeed method bound to a carrier: after a wrapped call the gradient must
equal ``grad_before / recorded_combined_scale`` bit-for-bit, which kills any drift between the
recorded coefficient and the multiplicative factor the engine applies. ``engine_grad_norm``
(the bf16-rounded print aggregate) is explicitly NOT the identity channel (QUEUE3486).
"""
from types import SimpleNamespace

import pytest
import torch

from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

from twinkle.model.transformers.strategy.deepspeed import _clip_coefficient_probe


def _carrier(loss_scale=1.0, clip_grad=1.0):
    self = SimpleNamespace(loss_scale=loss_scale, clip_grad=clip_grad)
    self.unscale_and_clip_grads = DeepSpeedZeroOptimizer.unscale_and_clip_grads.__get__(self)
    return self


def _run(carrier, grad, total_norm):
    before = grad.detach().clone()
    with _clip_coefficient_probe(carrier) as records:
        carrier.unscale_and_clip_grads([grad], total_norm)
    return before, records


def test_below_threshold_coefficient_is_loss_scale():
    carrier = _carrier(loss_scale=1.0, clip_grad=10.0)
    grad = torch.ones(4, dtype=torch.float32)
    _, records = _run(carrier, grad, torch.tensor(1.0))
    assert records[0]['combined_scale'] == 1.0
    assert records[0]['scaled_global_grad_norm'] == 1.0


def test_above_threshold_applies_real_clip():
    carrier = _carrier(loss_scale=1.0, clip_grad=1.0)
    grad = torch.full((4,), 8.0, dtype=torch.float32)
    before, records = _run(carrier, grad, torch.tensor(16.0))
    # norm 16 > clip 1 -> clip = (16+1e-6)/1 -> combined_scale ~ 16
    assert records[0]['combined_scale'] > 1.0
    # rtol 1e-6 kills bf16 rounding (~1e-2) and formula drift, tolerates fp32->fp64 promotion.
    assert torch.allclose(grad, before / records[0]['combined_scale'], rtol=1e-6, atol=0)


def test_clip_grad_zero_disables_clipping():
    carrier = _carrier(loss_scale=1.0, clip_grad=0.0)
    grad = torch.full((4,), 8.0, dtype=torch.float32)
    before, records = _run(carrier, grad, torch.tensor(16.0))
    assert records[0]['combined_scale'] == 1.0
    assert torch.equal(grad, before)


def test_fp16_loss_scale_is_carried():
    carrier = _carrier(loss_scale=8.0, clip_grad=0.0)
    grad = torch.ones(4, dtype=torch.float32)
    before, records = _run(carrier, grad, torch.tensor(32.0))
    assert records[0]['scaled_global_grad_norm'] == 32.0
    assert records[0]['combined_scale'] == 8.0
    assert torch.equal(grad, before / 8.0)


def test_tensor_total_norm_recorded_as_fp64_float():
    carrier = _carrier(loss_scale=1.0, clip_grad=1.0)
    grad = torch.full((4,), 8.0, dtype=torch.float32)
    _, records = _run(carrier, grad, torch.tensor(16.0))
    assert isinstance(records[0]['scaled_global_grad_norm'], float)
    assert isinstance(records[0]['combined_scale'], float)
    assert records[0]['scaled_global_grad_norm'] == 16.0


def test_one_record_per_group_call():
    carrier = _carrier(loss_scale=1.0, clip_grad=1.0)
    g0 = torch.full((4,), 8.0, dtype=torch.float32)
    g1 = torch.full((4,), 2.0, dtype=torch.float32)
    with _clip_coefficient_probe(carrier) as records:
        carrier.unscale_and_clip_grads([g0], torch.tensor(16.0))
        carrier.unscale_and_clip_grads([g1], torch.tensor(4.0))
    assert len(records) == 2
    assert records[0]['scaled_global_grad_norm'] == 16.0
    assert records[1]['scaled_global_grad_norm'] == 4.0


def test_optimizer_without_clip_method_yields_empty():
    carrier = SimpleNamespace()
    with _clip_coefficient_probe(carrier) as records:
        pass
    assert records == []


def test_original_restored_after_exception():
    carrier = _carrier(loss_scale=1.0, clip_grad=1.0)
    original = carrier.unscale_and_clip_grads
    with pytest.raises(RuntimeError):
        with _clip_coefficient_probe(carrier):
            raise RuntimeError('boom')
    assert carrier.unscale_and_clip_grads is original


def test_recorded_coefficient_matches_applied_factor():
    # Independent oracle: the real DeepSpeed method's multiplicative effect on the gradient.
    carrier = _carrier(loss_scale=4.0, clip_grad=2.0)
    grad = torch.randn(64, dtype=torch.float32)
    before, records = _run(carrier, grad, torch.tensor(40.0))
    assert torch.allclose(grad, before / records[0]['combined_scale'], rtol=1e-6, atol=0)