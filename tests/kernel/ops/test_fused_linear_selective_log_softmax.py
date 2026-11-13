# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from twinkle.kernel.ops.fused_linear_selective_log_softmax import fused_linear_selective_log_softmax

try:
    import torch_npu  # noqa: F401
    _NPU_OK = True
except ImportError:
    _NPU_OK = False


def _tolerances(dtype):
    return (1.6e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)


def _native(x, weight, targets, temperature, return_entropy):
    logits = F.linear(x, weight).div(temperature).float()
    logsumexp = torch.logsumexp(logits, dim=-1)
    logps = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1) - logsumexp
    if not return_entropy:
        return logps
    entropies = logsumexp - (logits.softmax(-1) * logits).sum(-1)
    return logps, entropies


def _check_forward_backward(device, dtype, return_entropy, shape=(17, 12, 5000)):
    rows, hidden, vocab = shape
    torch.manual_seed(2026)
    x = torch.randn(rows, hidden, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(vocab, hidden, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, vocab, (rows, ), device=device)
    actual = fused_linear_selective_log_softmax(x, weight, targets, 1.7, return_entropy)
    expected = _native(x, weight, targets, 1.7, return_entropy)
    actual_outputs = actual if return_entropy else (actual, )
    expected_outputs = expected if return_entropy else (expected, )
    rtol, atol = _tolerances(dtype)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        assert actual_output.dtype == torch.float32
        torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)
    output_grads = [torch.randn_like(output) for output in actual_outputs]
    actual_grads = torch.autograd.grad(actual, (x, weight), output_grads)
    expected_grads = torch.autograd.grad(expected, (x, weight), output_grads)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('return_entropy', [False, True])
def test_reference_matches_native(dtype, return_entropy):
    _check_forward_backward('cpu', dtype, return_entropy)


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('return_entropy', [False, True])
@pytest.mark.parametrize('shape', [(1, 8, 31), (37, 24, 257), (17, 12, 5000)])
def test_npu_matches_native(shape, dtype, return_entropy):
    _check_forward_backward('npu', dtype, return_entropy, shape)


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
def test_npu_entropy_only_backward():
    x = torch.randn(8, 16, device='npu', requires_grad=True)
    weight = torch.randn(97, 16, device='npu', requires_grad=True)
    targets = torch.randint(0, 97, (8, ), device='npu')
    _, entropies = fused_linear_selective_log_softmax(x, weight, targets, return_entropy=True)
    _, expected = _native(x, weight, targets, 1.0, True)
    grad = torch.randn_like(entropies)
    actual = torch.autograd.grad(entropies, (x, weight), grad)
    wanted = torch.autograd.grad(expected, (x, weight), grad)
    for actual_grad, expected_grad in zip(actual, wanted):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
def test_npu_mixed_precision_matches_native():
    x = torch.randn(8, 16, device='npu', dtype=torch.float32, requires_grad=True)
    weight = torch.randn(97, 16, device='npu', dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, 97, (8, ), device='npu')
    actual = fused_linear_selective_log_softmax(x, weight, targets, return_entropy=True)
    expected = _native(x.to(torch.bfloat16), weight, targets, 1.0, True)
    output_grads = [torch.randn_like(output) for output in actual]
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output, rtol=1.6e-2, atol=1e-2)
    actual_grads = torch.autograd.grad(actual, (x, weight), output_grads)
    expected_grads = torch.autograd.grad(expected, (x, weight), output_grads)
    assert actual_grads[0].dtype == torch.float32
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1.6e-2, atol=1e-2)


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
def test_npu_frozen_weight_skips_weight_gradient():
    x = torch.randn(8, 16, device='npu', requires_grad=True)
    weight = torch.randn(97, 16, device='npu')
    targets = torch.randint(0, 97, (8, ), device='npu')
    logps = fused_linear_selective_log_softmax(x, weight, targets)
    grad_x, = torch.autograd.grad(logps.sum(), x)
    assert grad_x.shape == x.shape


@pytest.mark.skipif(not _NPU_OK, reason='torch_npu unavailable')
def test_npu_repeated_backward_is_rejected():
    x = torch.randn(4, 8, device='npu', requires_grad=True)
    weight = torch.randn(31, 8, device='npu')
    targets = torch.randint(0, 31, (4, ), device='npu')
    logps = fused_linear_selective_log_softmax(x, weight, targets)
    torch.autograd.grad(logps.sum(), x, retain_graph=True)
    with pytest.raises(RuntimeError, match='repeated backward'):
        torch.autograd.grad(logps.sum(), x)


@pytest.mark.parametrize('kwargs', [
    {
        'temperature': 0
    },
    {
        'temperature': float('inf')
    },
    {
        'dtype': torch.float16
    },
])
def test_invalid_inputs_are_rejected(kwargs):
    dtype = kwargs.pop('dtype', torch.float32)
    x = torch.randn(4, 8, dtype=dtype)
    weight = torch.randn(31, 8, dtype=dtype)
    targets = torch.randint(0, 31, (4, ))
    with pytest.raises(ValueError):
        fused_linear_selective_log_softmax(x, weight, targets, **kwargs)


def test_invalid_shapes_are_rejected():
    x = torch.randn(4, 8)
    weight = torch.randn(31, 8)
    with pytest.raises(ValueError, match='targets'):
        fused_linear_selective_log_softmax(x, weight, torch.randint(0, 31, (3, )))
    with pytest.raises(ValueError, match='hidden'):
        fused_linear_selective_log_softmax(x, torch.randn(31, 7), torch.randint(0, 31, (4, )))
