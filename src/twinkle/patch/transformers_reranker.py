# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a HF causal LM into a *generative* reranker (cross-encoder scored via the LM head).

A generative reranker reuses the frozen ``lm_head`` instead of a fresh classification head: the
relevance score of a (query, document) pair is ``logit(positive_token) - logit(negative_token)``
(default tokens ``yes`` / ``no``), read at each position. This mirrors legacy swift's
``get_generative_reranker_logits`` (swift/utils/torch_utils.py) and its ``_patch_generative_reranker``
(swift/model/register.py), which patch the same two rows of the head weight.

Two things differ from :class:`~twinkle.patch.transformers_emb.TransformersEmbeddingPatch`:

1. The head is NOT replaced by identity -- it is wrapped so it emits a per-token score
   ``[B, T, 1]`` (the yes-minus-no difference) instead of full vocab logits. No new parameters
   are introduced (unlike the ``num_labels=1`` sequence-classification head used by the plain
   ``reranker`` task, which trains a fresh ``score`` linear).
2. Last-valid-token reduction ``[B, T, 1] -> [B, 1]`` is DEFERRED to
   ``InputProcessor.postprocess_tensor_sp(task='generative_reranker', ...)`` -- symmetric with how
   embedding defers pooling -- so this patch stays SP/CP/packed-agnostic and the dispatch lives in
   one place.

Both mutations are reverted by ``unpatch``.
"""
from types import MethodType
from typing import TYPE_CHECKING, Optional

from twinkle.patch import Patch
from twinkle.patch.transformers_emb import _LM_HEADS, get_lm_head_model

if TYPE_CHECKING:
    import torch


class TransformersGenerativeRerankerPatch(Patch):
    """Wrap ``lm_head`` so it emits a per-token relevance score ``[B, T, 1]``. Reversible via ``unpatch``.

    ``positive_token_id`` / ``negative_token_id`` are resolved by the caller (from the template
    tokenizer) and passed in, so the patch itself needs no tokenizer and stays device/model agnostic.
    """

    def __init__(self, positive_token_id: int, negative_token_id: int):
        self.positive_token_id = positive_token_id
        self.negative_token_id = negative_token_id

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

        pos_id, neg_id = self.positive_token_id, self.negative_token_id

        def _reranker_head_forward(self, hidden_states: 'torch.Tensor') -> 'torch.Tensor':
            import torch.nn.functional as F
            # Only the two rows we need: score = logit(pos) - logit(neg), computed per token.
            weight = self.weight[[pos_id, neg_id]]
            logits = F.linear(hidden_states, weight)
            return logits[..., 0:1] - logits[..., 1:2]

        head.forward = MethodType(_reranker_head_forward, head)
        return module

    def unpatch(self, module, *args, **kwargs):
        head = getattr(self, '_head', None)
        origin = getattr(self, '_origin_forward', None)
        if head is not None and origin is not None:
            head.forward = origin
            self._origin_forward = None
            self._head = None
        return module
