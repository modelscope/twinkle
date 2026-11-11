# Copyright (c) ModelScope Contributors.
"""The standard per-tenant task heads built by the task specs themselves.

The decision families' heads stay in the dev family loaders (ms-swift
``decision_head.ScoringHead``): they are model-family artifacts -- weights and calibration
state shipped in the model directory, rebuilt worker-side from a shipped loader NAME. The
heads here are the ones any causal base can carry: a plain Linear over the last hidden
states. They implement the same readout interface (``needs_lm_logits`` /
``encoder_backbone`` / ``keep_fp32`` flags and the ``forward(hidden_states, lm_logits,
meta)`` signature) so the one-shot forward hook and the forward_step readout branch feed
them exactly as they feed a decision head.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, Optional


class SeqClsHead(nn.Module):
    """The sequence-classification head: ``Linear(hidden, num_labels)`` over the base
    model's last hidden states.

    ``squeeze_single`` mirrors ``TransformersValuePatch``: True (the value task) surfaces
    per-token ``[B, T]``; False (seq_cls / reranker) keeps ``[B, T, num_labels]`` and leaves
    the last-valid-token pooling to the processor / the forward_step readout branch.
    """

    #: readout-interface flags, same names the decision ScoringHead families declare.
    needs_lm_logits = False
    needs_output_embeddings = False
    keep_fp32 = False
    encoder_backbone = False

    def __init__(self, hidden_size: int, num_labels: int, squeeze_single: bool = False):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.squeeze_single = squeeze_single

    def forward(self, hidden_states: torch.Tensor,
                lm_logits: Optional[torch.Tensor] = None,
                meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # The base model can surface its last hidden states in fp32 (a final-logit upcast) while
        # align_head placed this non-keep_fp32 head in the backbone dtype (bf16); cast the input to the
        # classifier's dtype so the readout matmul matches -- the same cast HF's *ForSequenceClassification
        # does (self.score(pooled.to(self.dtype))). The keep_fp32 heads upcast their inputs themselves.
        out = self.classifier(hidden_states.to(self.classifier.weight.dtype))
        if self.squeeze_single and out.dim() == 3 and out.shape[-1] == 1:
            out = out[..., 0]
        return out
