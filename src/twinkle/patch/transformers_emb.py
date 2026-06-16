# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a HF transformers causal LM into a sentence-embedding model.

Two mutations applied to the model:

1. ``lm_head.forward`` is replaced with identity, so the wrapped model returns
   the final hidden states under ``output.logits``.
2. A forward hook on the lm-head-bearing submodule L2-normalizes per-token
   hidden states and stores them under ``outputs['features']`` (shape
   ``[B, T, H]`` or ``[B, T_local, H]`` under SP).

Last-token pooling (incl. padding-free, SP gather) is **deferred** to
``InputProcessor.postprocess_tensor_sp(task='embedding', ...)`` so this patch
stays SP/CP/packed-agnostic and the dispatch sits in one place.

Both mutations are reverted by ``unpatch``.
"""
from types import MethodType
from typing import TYPE_CHECKING, Optional

from twinkle.patch import Patch

if TYPE_CHECKING:
    import torch

_LM_HEADS = ['lm_head', 'embed_out', 'output_layer', 'output']


def get_lm_head_model(module, lm_heads=None):
    from peft import PeftModel
    from torch.nn import Module
    if isinstance(module, PeftModel):
        module = module.model
    if lm_heads is None:
        lm_heads = _LM_HEADS
    for sub in module.modules():
        for name in lm_heads:
            child = getattr(sub, name, None)
            if isinstance(child, Module):
                return sub
    return module


def _output_features_hook(module, args, kwargs, output):
    # Replaces the model's structured output entirely; downstream HF utilities
    # (generate, loss, past_key_values, ...) won't work until ``unpatch``.
    # Pooling + L2-normalize is deferred to ``InputProcessor._postprocess_embedding``
    # so the hook stays pooling-agnostic (last-token / mean / CLS all work uniformly).
    hidden_states = output.logits
    return {'features': hidden_states.contiguous()}


def _identity_forward(self, hidden_states):
    return hidden_states


class TransformersEmbeddingPatch(Patch):
    """Convert a causal LM into a sentence-embedding feature extractor. Reversible via ``unpatch``."""

    def __call__(self, module, *args, **kwargs):
        from torch.nn import Module
        lm_head_model = get_lm_head_model(module, lm_heads=_LM_HEADS)

        head: Optional[Module] = None
        for name in _LM_HEADS:
            if hasattr(lm_head_model, name):
                head = getattr(lm_head_model, name)
                break
        assert head is not None, 'Cannot find the proper lm_head name'

        # Save originals BEFORE mutation so unpatch can restore them verbatim.
        self._head = head
        self._origin_forward = head.forward
        head.forward = MethodType(_identity_forward, head)
        self._hook_handle = lm_head_model.register_forward_hook(_output_features_hook, with_kwargs=True)
        return module

    def unpatch(self, module, *args, **kwargs):
        handle = getattr(self, '_hook_handle', None)
        if handle is not None:
            handle.remove()
            self._hook_handle = None

        head = getattr(self, '_head', None)
        origin = getattr(self, '_origin_forward', None)
        if head is not None and origin is not None:
            head.forward = origin
            self._origin_forward = None
            self._head = None
        return module
