# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reranker (cross-encoder) losses for Twinkle.

Inputs convention:
    inputs['labels']: binary relevance labels, ``1`` = relevant document.
    outputs['logits']: a single relevance SCORE per (query, document) pair, shape
        ``[B]`` or ``[B, 1]``.

Distinct from the embedding losses in ``infonce.py``: a reranker is a cross-encoder that
scores the query and document jointly through a classification head, so the signal lives in
``logits`` rather than in a pooled ``embeddings`` vector, and there is no pair interleaving.

All classes return :class:`LossOutput` with ``num_tokens=0`` (the loss is already averaged
over documents/groups, not over tokens).
"""
import torch
from torch import nn

from twinkle.data_format import LossOutput
from .base import Loss


def _extract_scores(outputs) -> torch.Tensor:
    """Return a flat [B] score vector from the reranker head output."""
    logits = outputs['logits']
    # The head emits [B, 1]; squeeze only that trailing axis so a genuine [B] stays untouched.
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    return logits


class PointwiseRerankerLoss(Loss):
    """Score each (query, document) pair independently as binary relevance.

    Every document is its own example, so nothing links the candidates of one query -- the model
    learns a calibrated absolute score rather than a ranking. Use
    :class:`ListwiseRerankerLoss` when relative order within a query matters.
    """

    require_logits = True
    require_entropy = False
    require_logps = False

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        logits = _extract_scores(outputs)
        labels = inputs['labels'].to(logits.dtype).view(-1)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        return LossOutput(loss=loss, num_tokens=0)


class ListwiseRerankerLoss(Loss):
    """Cross-entropy over each query's candidate list: pick the positive among its negatives.

    The batch is a concatenation of per-query groups, each laid out as one positive followed by
    its negatives, so ``inputs['labels']`` looks like ``[1, 0, 0, 0, 1, 0, 0, ...]``; every ``1``
    opens a new group that runs until the next ``1``. Within a group the positive is therefore
    always at index 0, which becomes the classification target.

    Unlike the pointwise variant this compares candidates against each other, so the absolute
    scores are free to drift -- only their order within a query is trained.

    Args:
        temperature: Divides the logits before the softmax. Values below 1 sharpen the
            distribution and push harder on the top-ranked negative.
        min_group_size: Groups with fewer candidates than this are skipped. The default of 2
            drops positive-only groups, which carry no ranking signal (a softmax over a single
            logit is always 1, so the loss and its gradient are exactly zero).
    """

    require_logits = True
    require_entropy = False
    require_logps = False

    def __init__(self, temperature: float = 1.0, min_group_size: int = 2, **kwargs):
        if temperature <= 0:
            raise ValueError(f'temperature must be > 0, got {temperature}: it divides the logits, so a '
                             'non-positive value would flip or destroy the ranking signal.')
        self.temperature = temperature
        self.min_group_size = min_group_size

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        logits = _extract_scores(outputs)
        labels = inputs['labels'].view(-1)

        positive_indices = torch.nonzero(labels == 1, as_tuple=False).view(-1)
        # `* 0.0` rather than a fresh zero tensor: this keeps the loss attached to the graph so
        # DDP/FSDP still see a gradient for every parameter and do not hang on the reduction.
        if positive_indices.numel() == 0:
            return LossOutput(loss=logits.sum() * 0.0, num_tokens=0)

        # Each group spans from its positive up to the next positive (or the end of the batch).
        bounds = positive_indices.tolist() + [labels.shape[0]]
        loss_fct = nn.CrossEntropyLoss()
        total_loss = None
        num_groups = 0
        for start, end in zip(bounds[:-1], bounds[1:]):
            group_logits = logits[start:end]
            if group_logits.shape[0] < self.min_group_size:
                continue
            # The positive is expected first; a group that starts with a negative is malformed
            # and would train the model toward the wrong document, so skip it.
            if labels[start] != 1:
                continue
            target = torch.zeros(1, dtype=torch.long, device=logits.device)
            group_loss = loss_fct((group_logits / self.temperature).unsqueeze(0), target)
            total_loss = group_loss if total_loss is None else total_loss + group_loss
            num_groups += 1

        if num_groups == 0:
            return LossOutput(loss=logits.sum() * 0.0, num_tokens=0)
        return LossOutput(loss=total_loss / num_groups, num_tokens=0)
