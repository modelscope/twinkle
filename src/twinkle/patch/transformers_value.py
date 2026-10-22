# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a HF sequence-classification model to expose PER-TOKEN scores (skip internal pooling).

HF's ``*ForSequenceClassification`` pools hidden states to the last valid token and returns
``[B, num_labels]``.  This patch captures the score head's per-token output BEFORE pooling and
surfaces it as ``outputs['logits']`` with shape ``[B, T, num_labels]``, letting the downstream
processor (``postprocess_tensor_sp``) do packed-aware pooling via ``_pool_last_valid_token``.

Two consumers:
- ``task='value'`` (PPO critic): needs per-token V(s_t), ``squeeze_single=True`` drops the trailing
  dim so logits is ``[B, T]``.
- ``task='seq_cls'``/``'reranker'`` under padding_free: needs ``[B, T, num_labels]`` (3D) so the
  processor's einsum-based pooling works; ``squeeze_single=False``.

Symmetric with the Megatron backend, where the seq_cls head already emits per-token ``[b, s, 1]``
and ``task='value'`` simply skips the last-token pick in ``forward_step``.
"""
from types import MethodType
from typing import TYPE_CHECKING

from twinkle.patch import Patch

if TYPE_CHECKING:
    import torch

#: Submodule names HF uses for the sequence-classification head across architectures.
_SCORE_NAMES = ('score', 'classifier')


class TransformersValuePatch(Patch):
    """Return the seq_cls head's per-token scores as logits (skip HF's internal pooling). Reversible.

    Args:
        squeeze_single: If True (default, PPO value), squeeze ``[B, T, 1]`` to ``[B, T]``.
            If False (seq_cls/reranker), keep 3D so ``_pool_last_valid_token`` einsum works.
    """

    def __init__(self, squeeze_single: bool = True):
        self._squeeze_single = squeeze_single

    def __call__(self, module, *args, **kwargs):
        score = self._find_score(module)
        # Save originals BEFORE mutation so unpatch restores them verbatim.
        self._score = score
        self._origin_forward = score.forward
        captured: dict = {}
        origin = self._origin_forward

        def _score_capture(self, hidden_states: 'torch.Tensor') -> 'torch.Tensor':
            out = origin(hidden_states)  # [B, T, num_labels], the per-token head output
            captured['per_token'] = out
            return out

        score.forward = MethodType(_score_capture, score)
        self._captured = captured
        # Top-level hook replaces the pooled SequenceClassifierOutput with the captured per-token scores.
        self._hook = module.register_forward_hook(self._surface, with_kwargs=True)
        return module

    def _surface(self, module, args, kwargs, output):
        per_token = self._captured.get('per_token')
        if per_token is None:
            return output
        if self._squeeze_single and per_token.dim() == 3 and per_token.shape[-1] == 1:
            per_token = per_token[..., 0]
        return {'logits': per_token}

    @staticmethod
    def _find_score(module):
        from peft import PeftModel
        base = module.model if isinstance(module, PeftModel) else module
        for name in _SCORE_NAMES:
            head = getattr(base, name, None)
            if head is not None:
                return head
        raise AssertionError('seq_cls score/classifier head not found on the model; '
                             "build with task_type='seq_cls' or task_type='reranker'.")

    def unpatch(self, module, *args, **kwargs):
        hook = getattr(self, '_hook', None)
        if hook is not None:
            hook.remove()
            self._hook = None
        score = getattr(self, '_score', None)
        origin = getattr(self, '_origin_forward', None)
        if score is not None and origin is not None:
            score.forward = origin
            self._origin_forward = None
            self._score = None
        return module
