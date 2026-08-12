# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sequence-classification loss for Twinkle.

Inputs convention:
    inputs['labels']: per-SAMPLE targets, shape ``[B]`` (regression / single-label) or
        ``[B, num_labels]`` (multi-label one-hot).
    outputs['logits']: per-SAMPLE class scores, shape ``[B, num_labels]`` -- already reduced to one
        row per sequence by the caller (transformers: the SequenceClassification head; Megatron: the
        last-valid-token pool in the processor).

Distinct from the embedding/reranker losses: seq_cls scores each sequence independently against a
fixed label set, dispatching to one of three objectives by ``problem_type`` exactly as HF's
``*ForSequenceClassification`` and legacy swift do (regression -> MSE, single-label -> CE,
multi-label -> BCE-with-logits). ``problem_type`` is REQUIRED (no inference here); the recipe layer
decides it.

Returns :class:`LossOutput` with ``num_tokens=0`` (already averaged over samples, not tokens).
"""
from torch import nn

from twinkle.data_format import LossOutput
from .base import Loss

PROBLEM_TYPES = ('regression', 'single_label_classification', 'multi_label_classification')


class SeqClsLoss(Loss):
    """Sequence classification loss dispatching MSE/CE/BCE by ``problem_type``.

    Mirrors HF ``*ForSequenceClassification`` and legacy swift's ``seq_cls_loss_func`` /
    ``transformers_seq_cls_forward`` numerics so a checkpoint trained here scores identically under
    legacy inference.
    """

    require_logits = True
    require_entropy = False
    require_logps = False

    def __init__(self, problem_type: str, num_labels: int, **kwargs):
        super().__init__()
        if problem_type not in PROBLEM_TYPES:
            raise ValueError(f'problem_type must be one of {PROBLEM_TYPES}, got {problem_type!r}. '
                             'It is required (not inferred) so the objective is explicit.')
        self.problem_type = problem_type
        self.num_labels = num_labels

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        logits = outputs['logits']
        labels = inputs['labels'].to(logits.device)

        if self.problem_type == 'regression':
            loss_fct = nn.MSELoss()
            # num_labels==1: squeeze the trailing class axis so [B,1] vs [B] broadcast cleanly, matching
            # legacy (transformers_seq_cls_forward / seq_cls_loss_func).
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.problem_type == 'single_label_classification':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:  # multi_label_classification
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.to(logits.dtype))

        return LossOutput(loss=loss, num_tokens=0)
