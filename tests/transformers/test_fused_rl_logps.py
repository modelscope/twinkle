# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from twinkle.model.transformers.transformers import (_fused_rl_logps, _resolve_task_context, _validate_fused_rl_task)
from twinkle.utils import selective_log_softmax


@pytest.mark.parametrize('return_entropy', [False, True])
def test_fused_rl_logps_match_native(return_entropy):
    torch.manual_seed(2026)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    head = nn.Linear(8, 31, bias=False)
    labels = torch.randint(0, 31, (2, 5))
    outputs = {'logits': hidden, 'lm_head': head}
    _fused_rl_logps(outputs, labels, 1.7, return_entropy)
    assert 'lm_head' not in outputs

    logits = F.linear(hidden, head.weight).div(1.7)
    expected = selective_log_softmax(logits, labels, return_entropy=return_entropy)
    actual = (outputs['logps'], outputs['entropies']) if return_entropy else outputs['logps']
    actual_outputs = actual if return_entropy else (actual, )
    expected_outputs = expected if return_entropy else (expected, )
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-5)
    output_grads = [torch.randn_like(output) for output in actual_outputs]
    actual_grad, = torch.autograd.grad(actual, hidden, output_grads)
    expected_grad, = torch.autograd.grad(expected, hidden, output_grads)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)


def test_fused_rl_task_context_defers_and_restores_lm_head():

    class _Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(8, 31, bias=False)

        def forward(self):
            return {'logits': self.lm_head(torch.randn(2, 5, 8))}

    model = _Model()
    with _resolve_task_context(model, 'fused_lm_rl'):
        outputs = model()
        assert outputs['logits'].shape == (2, 5, 8)
        assert outputs['lm_head'] is model.lm_head
    assert model()['logits'].shape == (2, 5, 31)
    assert 'lm_head' not in model()


@pytest.mark.parametrize('kwargs', [
    {
        'labels': None
    },
    {
        'loss_instance': SimpleNamespace(require_logps=False)
    },
    {
        'loss_instance': SimpleNamespace(require_logits=True)
    },
    {
        'return_logits': True
    },
    {
        'enable_sampling_replay': True
    },
    {
        'parallel_enabled': True
    },
])
def test_fused_rl_task_rejects_unsupported_contracts(kwargs):
    options = dict(
        labels=torch.zeros(2, 5),
        loss_instance=SimpleNamespace(),
        return_logits=False,
        enable_sampling_replay=False,
        parallel_enabled=False,
    )
    options.update(kwargs)
    with pytest.raises(ValueError, match='fused_lm_rl'):
        _validate_fused_rl_task('fused_lm_rl', **options)


def test_fused_rl_rejects_lm_head_bias():
    outputs = {'logits': torch.randn(2, 5, 8), 'lm_head': nn.Linear(8, 31, bias=True)}
    with pytest.raises(NotImplementedError, match='bias-free'):
        _fused_rl_logps(outputs, torch.randint(0, 31, (2, 5)), 1.0, False)
