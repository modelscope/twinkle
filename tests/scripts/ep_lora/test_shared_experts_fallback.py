"""Lightweight unit tests for dsv4 shared_experts fallback."""
import torch
import torch.nn as nn

from twinkle.model.transformers.strategy.native_fsdp import _collect_expert_params


def _make_block(use_plural_shared: bool, ignore_shared: bool) -> nn.Module:
    block = nn.Module()
    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.randn(2, 4, 8))
    experts.down_proj = nn.Parameter(torch.randn(2, 4, 4))
    block.experts = experts
    shared = nn.Linear(4, 4)
    if use_plural_shared:
        block.shared_experts = shared
    else:
        block.shared_expert = shared
    block._ep_patched = True
    block._ep_ignore_shared_experts = ignore_shared
    parent = nn.Module()
    parent.block = block
    return parent


def test_singular_shared_expert_collected():
    parent = _make_block(use_plural_shared=False, ignore_shared=True)
    ignored = _collect_expert_params(parent)
    expected_count = 2 + 2
    assert ignored is not None and len(ignored) == expected_count


def test_plural_shared_experts_collected():
    parent = _make_block(use_plural_shared=True, ignore_shared=True)
    ignored = _collect_expert_params(parent)
    expected_count = 2 + 2
    assert ignored is not None and len(ignored) == expected_count, (
        '_collect_expert_params should fall back to shared_experts (plural) for dsv4')


def test_no_ignore_shared_only_collects_experts():
    parent = _make_block(use_plural_shared=True, ignore_shared=False)
    ignored = _collect_expert_params(parent)
    assert ignored is not None and len(ignored) == 2
