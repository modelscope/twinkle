# Copyright (c) ModelScope Contributors. All rights reserved.
"""分段（packed）因果注意力的正确性：CPU 语义检查 + NPU 前反向对齐原生实现。

NPU 判据：与「逐文档 SDPA」的精确参考实现比较，FP32 用 ``rtol=1e-4 / atol=1e-5``，
BF16 用 ``rtol=1.6e-2 / atol=1e-2``；BF16 梯度另有显式越界元素预算（见 ``BF16_OUTLIER_BUDGET``），
阈值本身不放宽。
"""
import importlib.util
import pytest
import torch

from twinkle.kernel.ops.fused_varlen_attention import fused_varlen_attention, fused_varlen_attention_reference

NPU_DEVICE = 'npu:0'


def _npu_available():
    if importlib.util.find_spec('torch_npu') is None:
        return False
    import torch_npu  # noqa: F401
    return torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason='需要昇腾 NPU 与 torch_npu')

#: BF16 反向在长序列上会有极少数元素站在 ``atol`` 边界：``assert_close`` 是逐元素判定，而
#: ``dv = P^T @ do`` 的上千项求和在两套分块顺序下会产生 1~6 个越界元素（实测最多 6/1,048,576）。
#: 原生 BF16 实现在同形状下对精确值也有 1/1,048,576 同类越界，说明这是 BF16 归约顺序的固有限度。
#: 这里**不放宽 rtol/atol**，而是给出显式的越界元素预算与最大绝对差上限，任一被突破立即失败。
BF16_OUTLIER_BUDGET = 8
BF16_MAX_ABS = 4e-2


def tolerance(dtype):
    return {'rtol': 1e-4, 'atol': 1e-5} if dtype == torch.float32 else {'rtol': 1.6e-2, 'atol': 1e-2}


def assert_matches(got, want, dtype, label=''):
    """FP32 严格按 ``assert_close``；BF16 按上面的预算判定，阈值本身不动。"""
    tolerance_kwargs = tolerance(dtype)
    if dtype != torch.bfloat16:
        torch.testing.assert_close(got.float(), want.float(), check_dtype=False, **tolerance_kwargs)
        return
    difference = (got.float() - want.float()).abs()
    allowed = tolerance_kwargs['atol'] + tolerance_kwargs['rtol'] * want.float().abs()
    outliers = int((difference > allowed).sum())
    assert outliers <= BF16_OUTLIER_BUDGET, f'{label}: BF16 越界 {outliers} 个，超过预算 {BF16_OUTLIER_BUDGET}'
    assert difference.max() <= BF16_MAX_ABS, f'{label}: BF16 最大绝对差 {difference.max():.3e} 超过 {BF16_MAX_ABS}'


def make_inputs(lengths, heads=4, kv_heads=None, dim=8, dtype=torch.float32, seed=0, device='cpu'):
    torch.manual_seed(seed)
    kv_heads = kv_heads or heads
    total = sum(lengths)
    query = torch.randn(total, heads, dim, dtype=dtype, device=device)
    key = torch.randn(total, kv_heads, dim, dtype=dtype, device=device)
    value = torch.randn(total, kv_heads, dim, dtype=dtype, device=device)
    boundaries = [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist())
    return query, key, value, boundaries


def manual_attention(query, key, value, lengths, causal=True):
    """逐文档、逐头显式 softmax，与参考实现走不同的计算路径。"""
    outputs, offset = [], 0
    groups = query.shape[1] // key.shape[1]
    for length in lengths:
        q = query[offset:offset + length].transpose(0, 1)
        k = key[offset:offset + length].transpose(0, 1).repeat_interleave(groups, 0)
        v = value[offset:offset + length].transpose(0, 1).repeat_interleave(groups, 0)
        scores = (q @ k.transpose(1, 2)) * query.shape[-1]**-0.5
        if causal:
            scores = scores.masked_fill(torch.triu(torch.ones_like(scores, dtype=torch.bool), 1), float('-inf'))
        outputs.append((torch.softmax(scores, -1) @ v).transpose(0, 1))
        offset += length
    return torch.cat(outputs, 0)


@pytest.mark.parametrize('lengths', [(16, 8, 24), (1, 7, 33), (64, )])
def test_reference_matches_manual_attention(lengths):
    query, key, value, boundaries = make_inputs(lengths, heads=3, dim=8)
    expected = manual_attention(query, key, value, lengths)
    got = fused_varlen_attention_reference(query, key, value, boundaries)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_reference_supports_gqa():
    lengths = (12, 19)
    query, key, value, boundaries = make_inputs(lengths, heads=4, kv_heads=2)
    expected = manual_attention(query, key, value, lengths)
    assert torch.allclose(fused_varlen_attention_reference(query, key, value, boundaries), expected, atol=1e-6)


def test_reference_supports_non_causal():
    lengths = (10, 6)
    query, key, value, boundaries = make_inputs(lengths, heads=2, dim=4)
    got = fused_varlen_attention_reference(query, key, value, boundaries, causal=False)
    expected = manual_attention(query, key, value, lengths, causal=False)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


@requires_npu
def test_torch_backend_supports_bfloat16():
    """torch 后端在设备上的 BF16 路径（CPU 上 bf16 SDPA 在加载 torch_npu 后不可靠，故不在此验证）。"""
    lengths = (12, 19)
    bf16 = make_inputs(lengths, heads=4, kv_heads=2, dim=16, dtype=torch.bfloat16, device=NPU_DEVICE)
    exact = make_inputs(lengths, heads=4, kv_heads=2, dim=16, dtype=torch.float32, device=NPU_DEVICE)
    got = fused_varlen_attention(*bf16[:3], bf16[3], backend='torch')
    want = fused_varlen_attention_reference(*exact[:3], exact[3])
    assert got.dtype == torch.bfloat16
    assert_matches(got, want, torch.bfloat16, 'torch-backend')


def test_documents_do_not_leak_into_each_other():
    lengths = (8, 9, 11)
    query, key, value, boundaries = make_inputs(lengths, heads=2, dim=6)
    base = fused_varlen_attention_reference(query, key, value, boundaries)
    mutated = query.clone()
    mutated[lengths[0]:lengths[0] + lengths[1]] += 5.0
    got = fused_varlen_attention_reference(mutated, key, value, boundaries)
    assert torch.equal(got[:lengths[0]], base[:lengths[0]])
    middle = slice(lengths[0], lengths[0] + lengths[1])
    assert not torch.allclose(got[middle], base[middle])


def test_backend_torch_matches_reference():
    query, key, value, boundaries = make_inputs((14, 5), heads=2, dim=4)
    got = fused_varlen_attention(query, key, value, boundaries, backend='torch')
    assert torch.allclose(got, fused_varlen_attention_reference(query, key, value, boundaries), atol=1e-7)


def test_accepts_tensor_boundaries():
    query, key, value, boundaries = make_inputs((6, 7), heads=2, dim=4)
    got = fused_varlen_attention(query, key, value, torch.tensor(boundaries), backend='torch')
    assert torch.allclose(got, fused_varlen_attention_reference(query, key, value, boundaries), atol=1e-7)


def test_boundary_conversion_is_cached_per_tensor():
    from twinkle.kernel.ops.fused_varlen_attention.npu import _as_list

    boundaries = torch.tensor([0, 4, 8])
    first = _as_list(boundaries)
    assert first == [0, 4, 8]
    assert _as_list(boundaries) is first
    boundaries.add_(1)
    assert _as_list(boundaries) == [1, 5, 9]
    assert _as_list([0, 2]) == [0, 2]


@pytest.mark.parametrize('boundaries', [[0, 4, 4, 8], [1, 4, 8], [0, 4, 9], [0]])
def test_rejects_invalid_boundaries(boundaries):
    query, key, value, _ = make_inputs((8, ), heads=2, dim=4)
    with pytest.raises(ValueError):
        fused_varlen_attention(query, key, value, boundaries, backend='torch')


def test_rejects_head_mismatch():
    query, key, value, boundaries = make_inputs((8, ), heads=4, kv_heads=3)
    with pytest.raises(ValueError):
        fused_varlen_attention(query, key, value, boundaries, backend='torch')


def test_rejects_unsupported_dtype():
    query, key, value, boundaries = make_inputs((8, ), heads=2, dim=4, dtype=torch.float16)
    with pytest.raises(ValueError):
        fused_varlen_attention(query, key, value, boundaries, backend='torch')


def test_rejects_unknown_backend():
    query, key, value, boundaries = make_inputs((8, ), heads=2, dim=4)
    with pytest.raises(ValueError):
        fused_varlen_attention(query, key, value, boundaries, backend='liger')


def test_op_registered_with_both_backends():
    from twinkle.kernel.registry import get_op
    op = get_op('fused_varlen_attention')
    assert set(op.implementations) == {'npu', 'torch'}
    assert op.installer is not None
    resolved = op.implementations['torch'].load('fused_varlen_attention')
    assert resolved is fused_varlen_attention_reference


def test_attention_interface_signature():
    import inspect

    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    parameters = list(inspect.signature(npu_varlen_attention_forward).parameters)
    assert parameters[:5] == ['module', 'query', 'key', 'value', 'attention_mask']
    assert 'cu_seq_lens_q' in parameters


def test_installer_registers_attention_interface():
    """安装器只替换分发名，调用方可原样恢复；不触碰 transformers 的可用性判断。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    from twinkle.kernel.ops.fused_varlen_attention import ATTENTION_NAME, install_varlen_attention
    sentinel = lambda *args, **kwargs: (None, None)  # noqa: E731
    registry = ALL_ATTENTION_FUNCTIONS
    mapping = getattr(AttentionInterface, '_global_mapping', {})
    original = registry.get(ATTENTION_NAME)
    original_mapping = mapping.get(ATTENTION_NAME)
    try:
        install_varlen_attention(None, ATTENTION_NAME, sentinel)
        assert registry[ATTENTION_NAME] is sentinel
        if mapping:
            assert mapping[ATTENTION_NAME] is sentinel
    finally:
        if original is None:
            registry.pop(ATTENTION_NAME, None)
        else:
            registry[ATTENTION_NAME] = original
        if mapping:
            if original_mapping is None:
                mapping.pop(ATTENTION_NAME, None)
            else:
                mapping[ATTENTION_NAME] = original_mapping


def test_kernelize_logical_target_installs_attention():
    """仓库标准接入形态：``kernelize`` 的逻辑 target + ``KernelChoice``（不进默认配置）。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    from twinkle.kernel import kernelize
    from twinkle.kernel.ops.fused_varlen_attention import ATTENTION_NAME
    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    from twinkle.kernel.registry import KernelChoice
    registry = ALL_ATTENTION_FUNCTIONS
    mapping = getattr(AttentionInterface, '_global_mapping', {})
    original = registry.get(ATTENTION_NAME)
    original_mapping = mapping.get(ATTENTION_NAME)
    try:
        kernelize(
            torch.nn.Linear(2, 2), {'npu_varlen': KernelChoice(op='fused_varlen_attention', backends=('npu', 'torch'))})
        installed = registry[ATTENTION_NAME]
        assert installed in (npu_varlen_attention_forward, fused_varlen_attention_reference)
    finally:
        if original is None:
            registry.pop(ATTENTION_NAME, None)
        else:
            registry[ATTENTION_NAME] = original
        if mapping:
            if original_mapping is None:
                mapping.pop(ATTENTION_NAME, None)
            else:
                mapping[ATTENTION_NAME] = original_mapping


def run_npu_case(lengths, heads, kv_heads, dim, dtype):
    """BF16 与 FP32 精确参考比较（同一批 bf16 数值上采样），阈值按精度固定。"""
    query, key, value, boundaries = make_inputs(lengths, heads, kv_heads, dim, dtype, device=NPU_DEVICE)
    grad = torch.randn_like(query)
    fused_q, fused_k, fused_v = (t.clone().requires_grad_() for t in (query, key, value))
    output = fused_varlen_attention(fused_q, fused_k, fused_v, boundaries)
    grads = torch.autograd.grad(output, (fused_q, fused_k, fused_v), grad)

    ref_q, ref_k, ref_v = (t.float().requires_grad_() for t in (query, key, value))
    expected = fused_varlen_attention_reference(ref_q, ref_k, ref_v, boundaries)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_k, ref_v), grad.float())

    assert_matches(output, expected, dtype, 'out')
    for name, got, want in zip(('dq', 'dk', 'dv'), grads, expected_grads):
        assert_matches(got, want, dtype, name)


@requires_npu
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('lengths,heads,kv_heads,dim', [
    ((512, 300, 1024, 77), 8, 8, 128),
    ((1024, 1024), 8, 8, 128),
    ((33, 1, 256), 4, 4, 64),
    ((64, ) * 16, 4, 4, 32),
    ((2048, 512), 4, 4, 128),
    ((2049, ), 2, 2, 64),
    ((4096, ), 2, 2, 64),
])
def test_npu_forward_and_backward_match_native(lengths, heads, kv_heads, dim, dtype):
    run_npu_case(lengths, heads, kv_heads, dim, dtype)


@requires_npu
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_npu_gqa_matches_native(dtype):
    run_npu_case((1024, 1024), 16, 4, 128, dtype)


@requires_npu
@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('lengths,heads,kv_heads,dim', [
    ((64, ) * 16, 4, 4, 32),
    ((1024, 1024), 16, 4, 128),
    ((1024, 512, 256), 16, 8, 128),
])
def test_npu_bf16_matches_twinkle_sdpa_baseline(lengths, heads, kv_heads, dim, seed):
    from twinkle.kernel.ops.fused_varlen_attention.npu import _additive_mask
    from twinkle.kernel.ops.sdpa_attention.npu import npu_sdpa_attention_forward

    query, key, value, boundaries = make_inputs(
        lengths, heads, kv_heads, dim, torch.bfloat16, seed=seed, device=NPU_DEVICE)
    grad = torch.randn_like(query)
    fused_inputs = [tensor.clone().requires_grad_() for tensor in (query, key, value)]
    native_inputs = [tensor.clone().requires_grad_() for tensor in (query, key, value)]
    output = fused_varlen_attention(*fused_inputs, boundaries)
    module = type('Module', (), {'num_key_value_groups': heads // kv_heads})()
    native_outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q = native_inputs[0][start:end].transpose(0, 1).unsqueeze(0)
        k = native_inputs[1][start:end].transpose(0, 1).unsqueeze(0)
        v = native_inputs[2][start:end].transpose(0, 1).unsqueeze(0)
        native_outputs.append(npu_sdpa_attention_forward(
            module, q, k, v, _additive_mask(None, q, k), is_causal=False)[0][0])
    expected = torch.cat(native_outputs, 0)
    grads = torch.autograd.grad(output, fused_inputs, grad)
    expected_grads = torch.autograd.grad(expected, native_inputs, grad)
    torch.testing.assert_close(output, expected, **tolerance(torch.bfloat16))
    for got, want in zip(grads, expected_grads):
        torch.testing.assert_close(got, want, **tolerance(torch.bfloat16))


@requires_npu
def test_npu_documents_are_isolated():
    lengths = (128, 96, 160)
    query, key, value, boundaries = make_inputs(lengths, 4, 4, 64, torch.bfloat16, device=NPU_DEVICE)
    base = fused_varlen_attention(query, key, value, boundaries)
    mutated = query.clone()
    mutated[lengths[0]:lengths[0] + lengths[1]] += 3.0
    got = fused_varlen_attention(mutated, key, value, boundaries)
    assert torch.equal(got[:lengths[0]], base[:lengths[0]])
    middle = slice(lengths[0], lengths[0] + lengths[1])
    assert not torch.allclose(got[middle], base[middle])


@requires_npu
def test_npu_single_token_documents():
    query, key, value, boundaries = make_inputs((1, 1, 1, 8), 2, 2, 16, device=NPU_DEVICE)
    torch.testing.assert_close(
        fused_varlen_attention(query, key, value, boundaries),
        fused_varlen_attention_reference(query, key, value, boundaries),
        rtol=1e-4,
        atol=1e-5)


@requires_npu
def test_npu_self_check_runs_once():
    from twinkle.kernel.ops.fused_varlen_attention import npu as npu_module
    query, key, value, boundaries = make_inputs((8, ), 2, 2, 8, device=NPU_DEVICE)
    device_key = (torch.device(NPU_DEVICE).type, 0)
    npu_module._VERIFIED.discard(device_key)
    torch.manual_seed(123)
    torch.npu.manual_seed(123)
    cpu_rng_state = torch.random.get_rng_state()
    npu_rng_state = torch.npu.get_rng_state()
    fused_varlen_attention(query, key, value, boundaries)
    assert device_key in npu_module._VERIFIED
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    assert torch.equal(torch.npu.get_rng_state(), npu_rng_state)


@requires_npu
@pytest.mark.parametrize('lengths', [(64, 12), (211, 3)])
def test_npu_attention_interface_forward(lengths):
    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    query, key, value, boundaries = make_inputs(lengths, 4, 2, 64, device=NPU_DEVICE)
    bshd = lambda tensor: tensor.transpose(0, 1).unsqueeze(0)  # noqa: E731
    output, weights = npu_varlen_attention_forward(
        None,
        bshd(query),
        bshd(key),
        bshd(value),
        cu_seq_lens_q=torch.tensor(boundaries, dtype=torch.int32, device=NPU_DEVICE))
    assert weights is None
    assert output.shape == (1, sum(lengths), 4, 64)
    torch.testing.assert_close(
        output[0], fused_varlen_attention_reference(query, key, value, boundaries), rtol=1e-4, atol=1e-5)


@requires_npu
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_npu_attention_interface_single_document_uses_causal_sdpa(dtype, monkeypatch):
    import torch.nn.functional as F
    from transformers.integrations.sdpa_attention import repeat_kv
    from twinkle.kernel.ops.fused_varlen_attention import npu as npu_module

    query, key, value, boundaries = make_inputs((64, ), 4, 2, 16, dtype, device=NPU_DEVICE)
    bshd = lambda tensor: tensor.transpose(0, 1).unsqueeze(0).contiguous()  # noqa: E731
    inputs = [bshd(tensor).requires_grad_() for tensor in (query, key, value)]
    native_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    module = type('Module', (), {'num_key_value_groups': 2})()

    def unexpected_fused_call(*args, **kwargs):
        raise AssertionError('单文档不应进入 TND varlen 路径')

    monkeypatch.setattr(npu_module, 'npu_fused_varlen_attention', unexpected_fused_call)
    output, weights = npu_module.npu_varlen_attention_forward(
        module, *inputs, cu_seq_lens_q=torch.tensor(boundaries, dtype=torch.int32, device=NPU_DEVICE))
    native_output = F.scaled_dot_product_attention(
        native_inputs[0], repeat_kv(native_inputs[1], 2), repeat_kv(native_inputs[2], 2),
        attn_mask=None, dropout_p=0.0, scale=query.shape[-1]**-0.5, is_causal=True)
    native_output = native_output.transpose(1, 2).contiguous()
    grad = torch.randn_like(output)
    grads = torch.autograd.grad(output, inputs, grad)
    native_grads = torch.autograd.grad(native_output, native_inputs, grad)
    assert weights is None
    torch.testing.assert_close(output, native_output, **tolerance(dtype))
    for got, want in zip(grads, native_grads):
        torch.testing.assert_close(got, want, **tolerance(dtype))


@requires_npu
def test_npu_attention_interface_rejects_multi_batch():
    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    query = torch.randn(2, 4, 8, 16, device=NPU_DEVICE)
    boundaries = torch.tensor([0, 8], dtype=torch.int32, device=NPU_DEVICE)
    with pytest.raises(ValueError):
        npu_varlen_attention_forward(None, query, query, query, cu_seq_lens_q=boundaries)


@requires_npu
def test_npu_attention_interface_falls_back_to_sdpa():
    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    torch.manual_seed(7)
    batch, heads, seq, dim = 2, 4, 6, 8
    query = torch.randn(batch, heads, seq, dim, device=NPU_DEVICE)
    key = torch.randn(batch, heads, seq, dim, device=NPU_DEVICE)
    value = torch.randn(batch, heads, seq, dim, device=NPU_DEVICE)
    output, weights = npu_varlen_attention_forward(None, query, key, value)
    assert weights is None and output.shape == (batch, seq, heads, dim)
    scores = (query @ key.transpose(-1, -2)) * dim**-0.5
    scores = scores.masked_fill(torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=NPU_DEVICE), 1), float('-inf'))
    expected = (torch.softmax(scores, -1) @ value).transpose(1, 2)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)


@requires_npu
def test_npu_attention_interface_rejects_dropout_and_sliding_window():
    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    query, key, value, boundaries = make_inputs((16, ), 2, 2, 8, device=NPU_DEVICE)
    bshd = lambda tensor: tensor.transpose(0, 1).unsqueeze(0)  # noqa: E731
    boundaries = torch.tensor(boundaries, dtype=torch.int32, device=NPU_DEVICE)
    with pytest.raises(ValueError):
        npu_varlen_attention_forward(None, bshd(query), bshd(key), bshd(value), cu_seq_lens_q=boundaries, dropout=0.1)
    with pytest.raises(NotImplementedError):
        npu_varlen_attention_forward(
            None, bshd(query), bshd(key), bshd(value), cu_seq_lens_q=boundaries, sliding_window=64)


@requires_npu
def test_npu_rejects_non_causal():
    query, key, value, boundaries = make_inputs((16, ), 2, 2, 8, device=NPU_DEVICE)
    with pytest.raises(NotImplementedError):
        fused_varlen_attention(query, key, value, boundaries, causal=False)
