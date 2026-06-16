# Copyright (c) ModelScope Contributors. All rights reserved.
"""Embedding / contrastive losses for Twinkle.

Inputs convention:
    inputs['labels']: pair / multi-negative grouping labels (see each class docstring).
    outputs['embeddings']: sentence embeddings produced by the model
        (shape ``[B, D]``). Falls back to ``outputs['logits']`` for
        backward-compatibility with the legacy hook-side pooling layout.

All classes return :class:`LossOutput` with ``num_tokens=0`` (no per-token
normalization, matching the convention used by ``DPOLoss``/``GRPOLoss``).
"""
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from enum import Enum
from torch import nn
from typing import Optional

from twinkle.data_format import LossOutput
from .base import Loss


# Borrowed from sentence_transformers.
class SiameseDistanceMetric(Enum):
    """Distance metrics available to the pairwise contrastive losses."""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa


def _extract_sentences(outputs) -> torch.Tensor:
    """Return [B, D] sentence embeddings from postprocess_tensor_sp output.

    Prefers the canonical ``embeddings`` key (post-pooling); falls back to
    ``logits`` (legacy hook-side pooling) and applies CLS pooling for 3-D.
    """
    sentences = outputs.get('embeddings')
    if sentences is None:
        sentences = outputs['logits']
    if sentences.dim() == 3:
        sentences = sentences[:, 0]
    return sentences


def _parse_pair_sentence(outputs):
    """Split an interleaved [s1_0, s2_0, s1_1, s2_1, ...] tensor into (s1, s2)."""
    sentences = _extract_sentences(outputs)
    return sentences[0::2], sentences[1::2]


def _parse_multi_negative_sentences(sentences: torch.Tensor,
                                    labels: torch.Tensor,
                                    hard_negatives: Optional[int] = None):
    """Split a flat embedding tensor into per-sample groups.

    ``labels`` is a 1-D mask where ``1`` marks the start of a new
    ``anchor(1)+positive(1)+negatives(n)`` group; the inserted offsets account for
    the anchor sitting immediately before each positive in the flat layout.
    """
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    split_tensors = []
    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]
            elif negatives < hard_negatives:
                # upsample negatives with replacement; skip index 0 (positive)
                selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True) + 1
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors


class InfonceLoss(Loss):
    """InfoNCE contrastive loss with optional cross-DP gathering.

    Each sample is laid out as ``anchor(1) + positive(1) + negatives(n)``;
    ``inputs['labels']`` is a 1-D mask where ``1`` marks the start of every
    such group. Setting ``use_batch=True`` enables in-batch negatives and,
    when distributed is initialized, gathers embeddings from all DP ranks
    (only the local shard keeps gradients).

    Args:
        temperature: Logit scaling factor.
        use_batch: Include cross-sample (and cross-rank) in-batch negatives.
        hard_negatives: Fix the per-sample negative count via truncation/upsampling.
            ``None`` keeps the original variable counts.
        mask_fake_negative: Mask any logit greater than ``positive + fake_neg_margin``.
        fake_neg_margin: Threshold offset above the positive logit when masking.
        include_qq: Append the query-query similarity block (self diagonal masked).
        include_dd: Append the positive-doc to all-docs block (self positive masked).
        process_group: Distributed process group used for the all-gather.
            When ``None``, the default group (``dist.group.WORLD``) is used.
    """

    require_logits = True
    require_entropy = False
    require_logps = False

    def __init__(
        self,
        temperature: float = 0.1,
        use_batch: bool = True,
        hard_negatives: Optional[int] = None,
        mask_fake_negative: bool = False,
        fake_neg_margin: float = 0.1,
        include_qq: bool = False,
        include_dd: bool = False,
        process_group=None,
        **kwargs,
    ):
        self.temperature = temperature
        self.use_batch = use_batch
        self.hard_negatives = hard_negatives
        self.mask_fake_negative = mask_fake_negative
        self.fake_neg_margin = fake_neg_margin
        self.include_qq = include_qq
        self.include_dd = include_dd
        self.process_group = process_group

    def _gather_across_dp(self, sentences: torch.Tensor, labels: torch.Tensor):
        """All-gather embeddings & labels across DP ranks; only local shard keeps grad."""
        if not (dist.is_available() and dist.is_initialized()):
            return sentences, labels
        world_size = dist.get_world_size(group=self.process_group)
        if world_size <= 1:
            return sentences, labels
        rank = dist.get_rank(group=self.process_group)

        # variable per-rank shapes require communicating shape first
        local_shape = sentences.new_tensor(sentences.shape, dtype=torch.long)
        shapes = [torch.empty_like(local_shape) for _ in range(world_size)]
        dist.all_gather(shapes, local_shape, group=self.process_group)
        all_sentences = [sentences.new_empty(shape.tolist()) for shape in shapes]
        dist.all_gather(all_sentences, sentences.contiguous(), group=self.process_group)

        local_label_shape = labels.new_tensor(labels.shape, dtype=torch.long)
        label_shapes = [torch.empty_like(local_label_shape) for _ in range(world_size)]
        dist.all_gather(label_shapes, local_label_shape, group=self.process_group)
        all_labels = [labels.new_empty(shape.tolist()) for shape in label_shapes]
        dist.all_gather(all_labels, labels.contiguous(), group=self.process_group)

        # keep the local shard differentiable; detach others
        all_sentences[rank] = sentences
        for idx in range(world_size):
            if idx != rank:
                all_sentences[idx] = all_sentences[idx].detach()
        return torch.cat(all_sentences, dim=0), torch.cat(all_labels, dim=0)

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        labels = inputs['labels'].view(-1)
        sentences = _extract_sentences(outputs)

        if self.use_batch:
            sentences, labels = self._gather_across_dp(sentences, labels)

        split_tensors = _parse_multi_negative_sentences(sentences, labels, self.hard_negatives)
        if not split_tensors:
            # No anchor pairs in this micro-batch; return a zero loss that
            # still participates in autograd so DDP/FSDP do not hang.
            loss = sentences.sum() * 0.0
            return LossOutput(loss=loss, num_tokens=0)
        can_batched = self.hard_negatives is not None or len({s.shape[0] for s in split_tensors}) == 1

        if not self.use_batch:
            loss = self._intra_sample_loss(split_tensors, can_batched)
        else:
            loss = self._in_batch_loss(split_tensors, can_batched)
        return LossOutput(loss=loss, num_tokens=0)

    def _intra_sample_loss(self, split_tensors, can_batched) -> torch.Tensor:
        """InfoNCE with only the per-sample negatives (no cross-sample sharing)."""
        if can_batched:
            sentences = torch.stack(split_tensors, dim=0)  # [B, neg+2, D]
            similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / self.temperature
            labels = torch.zeros(len(split_tensors), dtype=torch.int64, device=sentences.device)
            return nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)

        loss = 0
        for tensor in split_tensors:
            similarity_matrix = torch.matmul(tensor[0], tensor[1:].T) / self.temperature
            labels = torch.tensor(0, device=tensor.device)
            loss = loss + nn.CrossEntropyLoss()(similarity_matrix, labels)
        return loss / len(split_tensors)

    def _in_batch_loss(self, split_tensors, can_batched) -> torch.Tensor:
        """InfoNCE with cross-sample (and optionally cross-rank) negatives."""
        if can_batched:
            return self._in_batch_loss_batched(split_tensors)
        return self._in_batch_loss_unbatched(split_tensors)

    def _in_batch_loss_batched(self, split_tensors) -> torch.Tensor:
        sentences = torch.stack(split_tensors, dim=0)  # [B, neg+2, D]
        queries = sentences[:, 0]  # [B, D]
        docs_all = sentences[:, 1:].reshape(-1, sentences.size(2))  # [B*(neg+1), D]
        qd_matrix = torch.matmul(queries, docs_all.T)  # [B, B*(neg+1)]
        # each row's positive sits at column row_idx * (neg+1)
        block = sentences.size(1) - 1
        labels = torch.arange(0, sentences.size(0) * block, block, device=sentences.device)

        logits_list = [qd_matrix]

        if self.include_qq:
            qq_matrix = torch.matmul(queries, queries.T).clone()
            qq_matrix.fill_diagonal_(float('-inf'))
            logits_list.append(qq_matrix)

        if self.include_dd:
            pos_docs = sentences[:, 1]  # [B, D]
            dd_matrix = torch.matmul(pos_docs, docs_all.T)  # [B, B*(neg+1)]
            if block > 0:
                row_idx = torch.arange(dd_matrix.size(0), device=dd_matrix.device)
                dd_matrix[row_idx, row_idx * block] = float('-inf')
            logits_list.append(dd_matrix)

        if self.mask_fake_negative:
            row_idx = torch.arange(qd_matrix.size(0), device=qd_matrix.device)
            thresholds = (qd_matrix[row_idx, labels].view(-1, 1).detach() + self.fake_neg_margin)

            qd_block = qd_matrix.clone()
            qd_block[qd_block > thresholds] = float('-inf')
            components = [qd_block]
            if self.include_qq:
                qq_block = logits_list[1].clone()
                qq_block[qq_block > thresholds] = float('-inf')
                components.append(qq_block)
            if self.include_dd:
                # align with Qwen3-Embedding: no threshold masking on d-d block
                components.append(logits_list[-1])
            similarity_matrix = torch.cat(components, dim=1)
        else:
            similarity_matrix = torch.cat(logits_list, dim=1)

        return nn.CrossEntropyLoss()(similarity_matrix / self.temperature, labels)

    def _in_batch_loss_unbatched(self, split_tensors) -> torch.Tensor:
        # docs from every sample concatenated as a shared negative bank
        docs_bank = torch.cat([t[1:] for t in split_tensors], dim=0)
        queries_all = torch.stack([t[0] for t in split_tensors], dim=0) if self.include_qq else None

        loss = 0
        length = 0
        for idx, tensor in enumerate(split_tensors):
            qd_vec = torch.matmul(tensor[0], docs_bank.T)
            target = torch.tensor(length, device=tensor.device)
            threshold = qd_vec[target].detach() + self.fake_neg_margin

            qd_masked = torch.where(qd_vec > threshold, qd_vec.new_full(
                (), float('-inf')), qd_vec) if self.mask_fake_negative else qd_vec
            logits_parts = [qd_masked]

            if self.include_qq:
                qq_vec = torch.matmul(tensor[0], queries_all.T).clone()
                qq_vec[idx] = float('-inf')
                if self.mask_fake_negative:
                    qq_vec = torch.where(qq_vec > threshold, qq_vec.new_full((), float('-inf')), qq_vec)
                logits_parts.append(qq_vec)

            if self.include_dd:
                dd_vec = torch.matmul(tensor[1], docs_bank.T)
                dd_vec[length] = float('-inf')
                logits_parts.append(dd_vec)

            logits_row = torch.cat(logits_parts, dim=-1) / self.temperature
            loss = loss + nn.CrossEntropyLoss()(logits_row.unsqueeze(0), target.unsqueeze(0))
            length += tensor.size(0) - 1
        return loss / len(split_tensors)
