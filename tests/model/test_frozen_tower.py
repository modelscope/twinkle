# Copyright (c) ModelScope Contributors. All rights reserved.
"""A fully frozen tower must not keep building an autograd graph (legacy swift #10214)."""
import torch
from torch import nn

from twinkle.model.transformers.frozen_tower import patch_frozen_module


class _Tower(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class _Owner(nn.Module):

    def __init__(self, trainable_head: bool = True):
        super().__init__()
        self.tower = _Tower()
        self.head = nn.Linear(12, 2)
        if not trainable_head:
            for parameter in self.head.parameters():
                parameter.requires_grad_(False)


def test_frozen_tower_drops_the_graph():
    owner = _Owner()
    for parameter in owner.tower.parameters():
        parameter.requires_grad_(False)
    patch_frozen_module(owner.tower)

    out = owner.tower(torch.randn(2, 4))

    assert not out.requires_grad and out.grad_fn is None


def test_trainable_tower_keeps_autograd():
    """The patch decides per call: a tower that is (still) trainable keeps its graph."""
    owner = _Owner()
    patch_frozen_module(owner.tower)

    out = owner.tower(torch.randn(2, 4))

    assert out.requires_grad and out.grad_fn is not None
    out.sum().backward()
    assert owner.tower.linear.weight.grad is not None


def test_input_requiring_grad_keeps_autograd():
    """A differentiable input (e.g. learned embeddings feeding the tower) must not be detached."""
    owner = _Owner()
    for parameter in owner.tower.parameters():
        parameter.requires_grad_(False)
    patch_frozen_module(owner.tower)

    out = owner.tower(torch.randn(2, 4, requires_grad=True))

    assert out.requires_grad


def test_patch_is_idempotent():
    owner = _Owner()
    for parameter in owner.tower.parameters():
        parameter.requires_grad_(False)
    patch_frozen_module(owner.tower)
    wrapped = owner.tower.forward
    patch_frozen_module(owner.tower)

    assert owner.tower.forward == wrapped
