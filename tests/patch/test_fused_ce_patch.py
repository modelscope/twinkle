# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for TransformersFusedCEPatch (device-free: runs on CPU).

These cover the patch mechanics — identity lm_head forward, lm_head module
stashed into outputs, full HF output preserved (aux_loss etc.), and reversibility.
The fused-loss numerical correctness (which needs the Liger kernel on a GPU/NPU)
lives in tests/loss/test_liger_fused_linear_ce.py.
"""
import torch
import torch.nn as nn
import pytest

from twinkle.patch import apply_context
from twinkle.patch.transformers_fused_ce import TransformersFusedCEPatch, _identity_forward
from twinkle.patch.transformers_emb import get_lm_head_model


class _FakeLMHeadModel(nn.Module):
    """Minimal HF-ForCausalLM-like module: a backbone returning hidden states
    via .logits, plus an lm_head. Mirrors the structure the patch expects."""

    def __init__(self, hidden_dim=8, vocab=16):
        super().__init__()
        self.embed = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab, bias=False)

    def forward(self, input_ids=None, **kwargs):
        bs, seq = input_ids.shape
        h = self.embed(torch.randn(bs, seq, self.embed.in_features, dtype=torch.float32))
        # identity under the patch -> .logits is h; otherwise full logits
        logits = self.lm_head(h)
        return {'logits': logits, 'aux_loss': torch.tensor(0.7), 'past_key_values': None}


def _make_model():
    torch.manual_seed(0)
    return _FakeLMHeadModel()


def test_identity_forward_returns_input_unchanged():
    head = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    y = _identity_forward(head, x)
    assert y is x  # exact same tensor object; no copy, no vocab GEMM


def test_patch_replaces_lm_head_with_identity_and_stashes_head():
    model = _make_model()
    input_ids = torch.randint(0, 10, (1, 3))
    with apply_context(model, TransformersFusedCEPatch()):
        # Under the patch, lm_head.forward is identity.
        assert model.lm_head.forward.__func__ is _identity_forward
        out = model(input_ids=input_ids)
        # The hook stashes the lm_head module and preserves aux_loss.
        assert out['lm_head'] is model.lm_head
        assert torch.equal(out['aux_loss'], torch.tensor(0.7))
        # logits == hidden states (identity, no vocab GEMM): vocab dim dropped.
        assert out['logits'].shape[-1] == 8  # hidden, not vocab
        assert 'past_key_values' in out  # full HF output preserved


def test_patch_unpatch_restores_original_forward():
    """Functional check (bound-method `is` is fragile): under the patch the
    vocab GEMM is skipped; after unpatch it runs again and the lm_head stashed
    reference is gone."""
    model = _make_model()
    input_ids = torch.randint(0, 10, (1, 3))

    # Before patch: full vocab logits.
    out_before = model(input_ids=input_ids)
    assert out_before['logits'].shape[-1] == 16  # vocab
    assert 'lm_head' not in out_before

    with apply_context(model, TransformersFusedCEPatch()):
        out_patched = model(input_ids=input_ids)
        assert out_patched['logits'].shape[-1] == 8  # hidden (identity)
        assert 'lm_head' in out_patched

    # After unpatch: back to full vocab logits, no stashed lm_head.
    out_after = model(input_ids=input_ids)
    assert out_after['logits'].shape[-1] == 16  # vocab restored
    assert 'lm_head' not in out_after


def test_patch_unpatch_on_exception():
    model = _make_model()
    input_ids = torch.randint(0, 10, (1, 3))
    with pytest.raises(RuntimeError):
        with apply_context(model, TransformersFusedCEPatch()):
            raise RuntimeError('boom')
    # unpatch must run even on exception: vocab GEMM restored.
    out = model(input_ids=input_ids)
    assert out['logits'].shape[-1] == 16
    assert 'lm_head' not in out


def test_patch_is_device_free():
    """The patch source must contain no device probes in actual code
    (hardware-unaware contract). Parsed via ast so docstrings/comments don't
    trip the check."""
    import ast
    import inspect
    from twinkle.patch import transformers_fused_ce as mod
    tree = ast.parse(inspect.getsource(mod))
    probes = {'torch', 'cuda', 'npu', 'is_npu_available', 'infer_device', 'select_impl'}
    for node in ast.walk(tree):
        # Attribute access like torch.cuda.is_available / Platform.is_npu
        if isinstance(node, ast.Attribute):
            chain = []
            cur = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            # Names that must not appear as the root of an attribute chain.
            assert cur.id not in ('torch', 'Platform'), \
                f'patch references device probe: {".".join(reversed(chain))}'
