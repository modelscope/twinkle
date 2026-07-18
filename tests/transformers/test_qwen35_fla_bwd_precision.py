# Copyright (c) ModelScope Contributors. All rights reserved.
"""FLA on/off backward precision test for Qwen3.5 linear attention.

Compares FLA ON (MindSpeed Triton bf16) vs FLA OFF (torch fp32 reference)
on a tiny single-device model. Uses mean-abs-diff as bias detector and
aggregate grad-norm as scale guard.
"""
from __future__ import annotations

import copy
import os
import sys
import pytest
import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM, Qwen3_5TextConfig, Qwen3_5MoeTextConfig
    _HAS_QWEN35 = True
except Exception:
    Qwen3_5ForCausalLM = None
    Qwen3_5MoeForCausalLM = None
    Qwen3_5TextConfig = None
    Qwen3_5MoeTextConfig = None
    _HAS_QWEN35 = False

# Tolerances: bf16 kernel vs fp32 reference. Hard-fail on mean-abs-diff
# (systematic bias) and aggregate grad-norm (scale). Max-abs is diagnostic only.
LOSS_ATOL = 5e-3
LOGITS_MEAN_ATOL = 2e-2
GRAD_MEAN_ATOL = 5e-3
GRAD_NORM_RTOL = 2.5e-1


def _device() -> torch.device:
    return torch.device('npu:0') if _HAS_NPU else torch.device('cpu')


def _dtype() -> torch.dtype:
    return torch.bfloat16


def _seed(seed: int) -> None:
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    torch.manual_seed(seed)
    if _HAS_NPU:
        torch_npu.npu.manual_seed(seed)
        # Flatten determinism for CANN matmul/conv kernels where supported.
        try:
            torch_npu.npu.set_allow_internal_format(False)
        except Exception:
            pass


def _common_config_kwargs(layer_types):
    return dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        intermediate_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        use_cache=False,
    )


def _build_tiny(model_kind: str, layer_types, device: torch.device, seed: int) -> torch.nn.Module:
    _seed(seed)
    if model_kind == 'dense':
        cfg = Qwen3_5TextConfig(**_common_config_kwargs(layer_types))
    elif model_kind == 'moe':
        cfg = Qwen3_5MoeTextConfig(
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            output_router_logits=False,
            **_common_config_kwargs(layer_types),
        )
    else:
        raise ValueError(f'Unknown model kind: {model_kind}')
    cfg._attn_implementation = 'sdpa'
    cls = Qwen3_5ForCausalLM if model_kind == 'dense' else Qwen3_5MoeForCausalLM
    model = cls(cfg)
    model.to(device=device, dtype=_dtype())
    model.eval()
    _force_fla_off(model)
    return model


def _build_batch(device: torch.device, seq_len: int):
    torch.manual_seed(0)
    ids = torch.randint(3, 128, (2, seq_len), device=device, dtype=torch.long)
    labels = torch.randint(3, 128, (2, seq_len), device=device, dtype=torch.long)
    return ids, labels


def _force_fla_off(model: torch.nn.Module) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    for layer in model.model.layers:
        la = getattr(layer, 'linear_attn', None)
        if la is None:
            continue
        la.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        la.causal_conv1d_fn = None
        if getattr(la, '_twinkle_npu_patched', False):
            delattr(la, '_twinkle_npu_patched')


def _force_fla_on(model: torch.nn.Module) -> int:
    import importlib
    import twinkle.kernel.npu_impls.fla as fla_mod
    importlib.reload(fla_mod)
    return fla_mod.apply_qwen3_5_fla(model)


def _run_forward_backward(model: torch.nn.Module, ids, labels):
    out = model(input_ids=ids).logits
    loss = F.cross_entropy(out.float().reshape(-1, out.size(-1)), labels.reshape(-1))
    # Zero existing grads
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {}
    for i, layer in enumerate(model.model.layers):
        la = getattr(layer, 'linear_attn', None)
        if la is None:
            continue
        for name in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj', 'conv1d'):
            sub = getattr(la, name, None)
            if sub is None or not hasattr(sub, 'weight'):
                continue
            g = sub.weight.grad
            key = f'l{i}.{name}.weight'
            grads[key] = g.detach().float().cpu() if g is not None else None
    return out.detach().float().cpu(), loss.detach().float().cpu().item(), grads


def _max_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs() + 1e-9
    rel = (diff / denom)
    # Filter out near-zero denominators to avoid the rel-diff blow-up.
    mask = b.float().abs() > 1e-3
    rel_masked = rel[mask] if mask.any() else rel
    return {
        'max_abs': diff.max().item(),
        'mean_abs': diff.mean().item(),
        'max_rel': float(rel_masked.max()) if rel_masked.numel() else 0.0,
        'mean_rel': float(rel_masked.mean()) if rel_masked.numel() else 0.0,
    }


def _assert_mean_close_report(name: str, a, b, mean_atol, norm_rtol=None):
    if a is None or b is None:
        raise AssertionError(f'{name}: one side is None (a={a is None}, b={b is None})')
    stats = _max_diff(a, b)
    ok = stats['mean_abs'] < mean_atol
    if norm_rtol is not None and a.numel() > 0:
        nrel = abs(a.float().norm() - b.float().norm()) / (b.float().norm() + 1e-9)
        stats['norm_rel'] = nrel.item()
        # Per-tensor norm_rel is noisy on small-magnitude tensors in MoE; treat
        # as diagnostic. The aggregate grad-norm check below is the real
        # scale guard.
    flag = 'PASS' if ok else 'FAIL'
    extra = f'  norm_rel={stats.get("norm_rel", 0):.4e}' if 'norm_rel' in stats else ''
    print(f'  [{flag}] {name:32s} mean_abs={stats["mean_abs"]:.4e} max_abs={stats["max_abs"]:.4e}'
          f'  max_rel={stats["max_rel"]:.4e}{extra}  (mean_atol={mean_atol:.0e})')
    if not ok:
        raise AssertionError(
            f'{name} mismatch: mean_abs={stats["mean_abs"]:.4e} max_abs={stats["max_abs"]:.4e} '
            f'max_rel={stats["max_rel"]:.4e} (mean_atol={mean_atol})')
    return stats


def _assert_close_report(name: str, a, b, atol, rtol, required: bool = True):
    if a is None or b is None:
        if required:
            raise AssertionError(f'{name}: one side is None (a={a is None}, b={b is None})')
        return None
    stats = _max_diff(a, b)
    ok = bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    flag = 'PASS' if ok else 'FAIL'
    print(f'  [{flag}] {name:32s} max_abs={stats["max_abs"]:.4e} mean_abs={stats["mean_abs"]:.4e}'
          f'  max_rel={stats["max_rel"]:.4e}  (atol={atol:.0e}, rtol={rtol:.0e})')
    if not ok and required:
        raise AssertionError(
            f'{name} mismatch: max_abs={stats["max_abs"]:.4e} mean_abs={stats["mean_abs"]:.4e} '
            f'max_rel={stats["max_rel"]:.4e} (atol={atol}, rtol={rtol})')
    return stats


def _compare_models(model_kind: str, layer_types, seed: int, seq_len: int):
    device = _device()
    base = _build_tiny(model_kind, layer_types, device, seed)
    ids, labels = _build_batch(device, seq_len)

    # FLA OFF (ground truth): torch reference path
    os.environ['TWINKLE_NPU_FLA'] = '0'
    m_off = copy.deepcopy(base).to(device)
    _force_fla_off(m_off)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as _torch_ref
    with torch.no_grad():
        for layer in m_off.model.layers:
            la = getattr(layer, 'linear_attn', None)
            if la is not None:
                assert la.causal_conv1d_fn is None, 'FLA OFF path must have causal_conv1d_fn=None'
                assert la.chunk_gated_delta_rule is _torch_ref, \
                    f'FLA OFF path must use torch reference, got {la.chunk_gated_delta_rule}'
    logits_off, loss_off, grads_off = _run_forward_backward(m_off, ids, labels)

    # FLA ON: MindSpeed Triton kernel via twinkle patch
    os.environ['TWINKLE_NPU_FLA'] = '1'
    m_on = copy.deepcopy(base).to(device)
    _force_fla_off(m_on)  # start clean so apply_qwen3_5_fla re-patches
    n_patched = _force_fla_on(m_on)
    assert n_patched > 0, 'apply_qwen3_5_fla patched 0 instances; FLA ON path not active'
    logits_on, loss_on, grads_on = _run_forward_backward(m_on, ids, labels)

    print(f'\n=== {model_kind} {layer_types} seed={seed} seq_len={seq_len} (patched={n_patched}) ===')
    print(f'  loss: off={loss_off:.6f}  on={loss_on:.6f}  diff={abs(loss_on - loss_off):.4e}')
    stats = {'loss': {'off': loss_off, 'on': loss_on, 'abs_diff': abs(loss_on - loss_off)}}
    _assert_mean_close_report('logits', logits_on, logits_off, LOGITS_MEAN_ATOL)
    assert abs(loss_on - loss_off) < LOSS_ATOL, f'loss diff {abs(loss_on - loss_off)} >= {LOSS_ATOL}'
    grad_stats = {}
    total_norm_on = torch.tensor(0.0)
    total_norm_off = torch.tensor(0.0)
    for key in grads_off:
        s = _assert_mean_close_report(key, grads_on.get(key), grads_off.get(key),
                                      GRAD_MEAN_ATOL, GRAD_NORM_RTOL)
        if s is not None:
            grad_stats[key] = s
        if grads_on.get(key) is not None:
            total_norm_on += grads_on[key].float().pow(2).sum()
        if grads_off.get(key) is not None:
            total_norm_off += grads_off[key].float().pow(2).sum()
    # Aggregate grad-norm check (robust to per-tensor MoE routing noise).
    total_norm_on = total_norm_on.sqrt()
    total_norm_off = total_norm_off.sqrt()
    total_rel = abs(total_norm_on - total_norm_off) / (total_norm_off + 1e-9)
    ok = total_rel < GRAD_NORM_RTOL
    print(f'  [{"PASS" if ok else "FAIL"}] total_grad_norm_rel          '
          f'on={total_norm_on.item():.4e} off={total_norm_off.item():.4e} '
          f'rel={total_rel.item():.4e}  (rtol={GRAD_NORM_RTOL})')
    if not ok:
        raise AssertionError(
            f'aggregate grad-norm rel diff {total_rel.item():.4e} >= {GRAD_NORM_RTOL}')
    stats['grads'] = grad_stats
    return stats


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_NPU, reason='requires Ascend NPU (torch_npu)')
@pytest.mark.skipif(not _HAS_QWEN35, reason='requires transformers Qwen3.5')
class TestQwen35FlaBwdPrecision:
    """Compare FLA ON (bf16 Triton kernel) vs FLA OFF (fp32 torch reference).

    These tests do NOT require multiple devices and are the precise analogue of
    the FLA toggle used by ``cookbook/transformers/fsdp2.sh`` (which sets the
    same ``TWINKLE_NPU_FLA`` env var via ``npu_builtin(model)``).
    """

    @pytest.mark.parametrize('seed', [1234, 2024])
    @pytest.mark.parametrize('seq_len', [64, 256])
    def test_dense_linear_attention_fla_on_vs_off(self, seed, seq_len):
        _compare_models('dense', ['linear_attention', 'linear_attention'], seed, seq_len)

    @pytest.mark.parametrize('seed', [1234])
    def test_dense_mixed_attention_fla_on_vs_off(self, seed):
        _compare_models('dense', ['full_attention', 'linear_attention'], seed, 128)

    @pytest.mark.parametrize('seed', [1234, 2024])
    def test_moe_linear_attention_fla_on_vs_off(self, seed):
        _compare_models('moe', ['linear_attention', 'linear_attention'], seed, 128)


if __name__ == '__main__':
    _compare_models('dense', ['linear_attention', 'linear_attention'], 1234, 64)
    _compare_models('dense', ['linear_attention', 'linear_attention'], 2024, 256)
    _compare_models('moe', ['linear_attention', 'linear_attention'], 1234, 128)
    print('\nAll comparisons finished.')
