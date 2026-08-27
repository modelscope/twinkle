# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a HF sequence-classification model to expose PER-TOKEN scores (PPO's value critic).

PPO's value function needs an estimate ``V(s_t)`` at every response token. It rides the SAME
``num_labels=1`` sequence-classification head as ``task='seq_cls'`` (a trainable ``score`` linear over
the last hidden state), but ``*ForSequenceClassification`` pools that head to the last valid token and
returns one scalar ``[B, 1]``. This patch captures the head's per-token output ``score(hidden)`` BEFORE
the pooling and returns it in place of the pooled logits, so ``outputs['logits']`` is ``[B, T]`` (one
value per token). No new parameters are introduced -- the value head IS the seq_cls ``score``.

Symmetric with the Megatron backend, where the seq_cls head already emits a per-token ``[b, s, 1]`` and
``task='value'`` simply skips the last-token pick in ``forward_step``. Like the embedding / generative
reranker patches, both mutations are reverted by ``unpatch``.

Scope: the DDP / AccelerateStrategy path; it does not cover sequence-parallel / packed layouts, so the
PPO recipe runs the critic without those.
"""
from types import MethodType
from typing import TYPE_CHECKING

from twinkle.patch import Patch

if TYPE_CHECKING:
    import torch

#: Submodule names HF uses for the sequence-classification head across architectures.
_SCORE_NAMES = ('score', 'classifier')


class TransformersValuePatch(Patch):
    """Return the seq_cls head's per-token scores ``[B, T]`` as logits (skip pooling). Reversible."""

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
        # num_labels=1 (the PPO critic) -> [B, T]; keep the head width for any other num_labels.
        values = per_token[..., 0] if (per_token.dim() == 3 and per_token.shape[-1] == 1) else per_token
        return {'logits': values}

    @staticmethod
    def _find_score(module):
        from peft import PeftModel
        base = module.model if isinstance(module, PeftModel) else module
        for name in _SCORE_NAMES:
            head = getattr(base, name, None)
            if head is not None:
                return head
        raise AssertionError("task='value' requires a seq_cls score/classifier head (build the critic with "
                             "task_type='seq_cls', num_labels=1); none was found on the model.")

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
