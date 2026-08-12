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
from torch import nn
from typing import Callable, Dict, Optional

from twinkle.data_format import LossOutput
from .base import Loss


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, computed pairwise (row i of x against row i of y)."""
    return 1 - F.cosine_similarity(x, y)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.pairwise_distance(x, y, p=2)


def manhattan_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.pairwise_distance(x, y, p=1)


# Named so a config string can select the metric; legacy exposed these as a
# SiameseDistanceMetric Enum of lambdas, which cannot be referenced by name from a config.
DISTANCE_METRICS = {
    'cosine': cosine_distance,
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
}


def _resolve_distance_metric(metric) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if callable(metric):
        return metric
    if metric not in DISTANCE_METRICS:
        raise ValueError(f'Unknown distance metric {metric!r}. Supported: {", ".join(sorted(DISTANCE_METRICS))}.')
    return DISTANCE_METRICS[metric]


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


class EmbeddingLoss(Loss):
    """Base for sentence-embedding losses, adding optional Matryoshka (MRL) aggregation.

    Matryoshka Representation Learning trains several nested prefixes of the embedding at
    once, so a single checkpoint can be truncated to a smaller dimension at serving time.
    The loss is evaluated per prefix and combined as a weighted sum.

    Subclasses keep their own ``__call__`` and opt in by routing their computation through
    :meth:`_mrl_reduce`; nothing happens implicitly. Set ``supports_mrl = False`` on a
    subclass whose objective has no meaningful per-prefix interpretation.

    Args:
        mrl_dims: ``{dim: weight}``. Falsy disables MRL, leaving the plain single-pass path.
    """

    supports_mrl = True

    def __init__(self, mrl_dims: Optional[Dict[int, float]] = None, **kwargs):
        if mrl_dims and not self.supports_mrl:
            raise ValueError(f'{type(self).__name__} does not support mrl_dims: its objective has no '
                             'per-prefix interpretation, so a weighted sum over truncated dimensions '
                             'would not be meaningful. Drop mrl_dims, or pick an MRL-capable loss '
                             '(e.g. InfonceLoss).')
        self.mrl_dims = mrl_dims

    def _mrl_reduce(self, sentences: torch.Tensor, compute: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Weighted sum of ``compute(sentences)`` over each Matryoshka prefix.

        Each prefix is re-normalized after truncation: a slice of a unit vector is not itself
        unit-norm, and re-normalizing recovers exactly the ``normalize(h[..., :dim])`` that
        pooling would have produced at that width. Without MRL configured, ``compute`` is
        called once on ``sentences`` untouched, so the plain path is bit-for-bit unchanged.

        Call this **after** any collective (e.g. the cross-DP gather in :class:`InfonceLoss`):
        ``compute`` runs once per dim, so a collective folded inside would be issued
        ``len(mrl_dims)`` times, and ranks skipping a too-large dim would issue a different
        count and desync.

        Args:
            sentences: ``[B, D]`` embeddings, already gathered if the loss gathers.
            compute: Maps one prefix to its scalar loss.
        """
        if not self.mrl_dims:
            return compute(sentences)
        hidden_size = sentences.shape[-1]
        loss = None
        for dim, weight in self.mrl_dims.items():
            if dim > hidden_size:
                continue
            cur_loss = weight * compute(F.normalize(sentences[..., :dim], p=2, dim=-1))
            loss = cur_loss if loss is None else loss + cur_loss
        if loss is None:
            raise ValueError(f'Every mrl_dims entry {sorted(self.mrl_dims)} exceeds the embedding size '
                             f'{hidden_size}, leaving nothing to optimize. Use dimensions <= {hidden_size}.')
        return loss


class InfonceLoss(EmbeddingLoss):
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
        mrl_dims: See :class:`EmbeddingLoss`.
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
        mrl_dims: Optional[Dict[int, float]] = None,
        **kwargs,
    ):
        super().__init__(mrl_dims=mrl_dims)
        if mask_fake_negative and fake_neg_margin <= 0:
            raise ValueError(f'fake_neg_margin must be > 0 when mask_fake_negative=True, got {fake_neg_margin}. '
                             'A non-positive margin would mask out the positive itself or every above-positive '
                             'logit indiscriminately, collapsing the contrastive signal.')
        self.temperature = temperature
        self.use_batch = use_batch
        self.hard_negatives = hard_negatives
        self.mask_fake_negative = mask_fake_negative
        self.fake_neg_margin = fake_neg_margin
        self.include_qq = include_qq
        self.include_dd = include_dd
        self.process_group = process_group

    def _gather_across_dp(self, sentences: torch.Tensor, labels: torch.Tensor):
        """All-gather embeddings & labels across DP ranks; only local shard keeps grad.

        NCCL ``all_gather`` requires every rank to send the *same* tensor size. Under
        ``slice_dp`` dispatch the per-rank batch is uneven (``divmod`` splits), so we
        pad each rank to the global max along dim-0, do an equal-sized all_gather,
        then strip padding back. Only the local shard retains gradients.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return sentences, labels
        world_size = dist.get_world_size(group=self.process_group)
        if world_size <= 1:
            return sentences, labels
        rank = dist.get_rank(group=self.process_group)

        # ``labels`` is a 1-D mask aligned to ``sentences`` along dim-0, so they
        # share the same per-rank size. Gather sizes once and reuse for both.
        assert sentences.shape[0] == labels.shape[0], (
            f'sentences/labels dim-0 mismatch: {sentences.shape[0]} vs {labels.shape[0]}')
        local_n = torch.tensor([sentences.shape[0]], device=sentences.device, dtype=torch.long)
        sizes = [torch.empty_like(local_n) for _ in range(world_size)]
        dist.all_gather(sizes, local_n, group=self.process_group)
        sizes_int = [int(s.item()) for s in sizes]
        max_n = max(sizes_int)

        def _pad_gather(tensor: torch.Tensor):
            if tensor.shape[0] < max_n:
                pad_shape = (max_n - tensor.shape[0], ) + tuple(tensor.shape[1:])
                padded = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
            else:
                padded = tensor
            buffers = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(buffers, padded.contiguous(), group=self.process_group)
            return buffers

        sent_buffers = _pad_gather(sentences)
        label_buffers = _pad_gather(labels)

        # Strip padding; keep local shard differentiable, detach others.
        all_sentences = []
        all_labels = []
        for idx in range(world_size):
            n = sizes_int[idx]
            if idx == rank:
                all_sentences.append(sentences)
                all_labels.append(labels)
            else:
                all_sentences.append(sent_buffers[idx][:n].detach())
                all_labels.append(label_buffers[idx][:n])
        return torch.cat(all_sentences, dim=0), torch.cat(all_labels, dim=0)

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        labels = inputs['labels'].view(-1)
        sentences = _extract_sentences(outputs)

        # Gather before _mrl_reduce: the collective must be issued exactly once per
        # micro-batch, whereas the MRL body runs once per dim.
        if self.use_batch:
            sentences, labels = self._gather_across_dp(sentences, labels)

        def compute(sentences: torch.Tensor) -> torch.Tensor:
            split_tensors = _parse_multi_negative_sentences(sentences, labels, self.hard_negatives)
            if not split_tensors:
                # No anchor pairs in this micro-batch; return a zero loss that
                # still participates in autograd so DDP/FSDP do not hang.
                return sentences.sum() * 0.0
            can_batched = self.hard_negatives is not None or len({s.shape[0] for s in split_tensors}) == 1
            if not self.use_batch:
                return self._intra_sample_loss(split_tensors, can_batched)
            return self._in_batch_loss(split_tensors, can_batched)

        return LossOutput(loss=self._mrl_reduce(sentences, compute), num_tokens=0)

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


class CosineSimilarityLoss(EmbeddingLoss):
    """Regress the cosine similarity of a sentence pair onto a continuous label.

    Embeddings arrive interleaved as ``[s1_0, s2_0, s1_1, s2_1, ...]``, so
    ``inputs['labels']`` holds ONE score per pair (i.e. ``B/2`` entries, not ``B``).

    MRL is rejected: the target is an absolute similarity value, and truncated prefixes do
    not preserve it, so summing the per-prefix regression errors optimizes toward a target
    that no single prefix actually has.
    """

    require_logits = True
    require_entropy = False
    require_logps = False
    supports_mrl = False

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        output = torch.cosine_similarity(sentence1, sentence2)
        labels = inputs['labels'].to(output.dtype).view(-1)
        loss = nn.MSELoss()(output, labels)
        return LossOutput(loss=loss, num_tokens=0)


class ContrastiveLoss(EmbeddingLoss):
    """Pull labelled-similar pairs together, push dissimilar pairs beyond ``margin``.

    ``inputs['labels']`` is one binary flag per PAIR (1 = similar). Dissimilar pairs only
    contribute while their distance is still below ``margin`` (the hinge), which is what stops
    the model from pushing negatives apart without bound.

    Args:
        margin: Distance the dissimilar pairs must exceed before their loss becomes zero.
        distance_metric: Name in :data:`DISTANCE_METRICS` or a callable ``(x, y) -> distances``.
        mrl_dims: See :class:`EmbeddingLoss`.
    """

    require_logits = True
    require_entropy = False
    require_logps = False

    def __init__(self, margin: float = 0.5, distance_metric='cosine', mrl_dims=None, **kwargs):
        super().__init__(mrl_dims=mrl_dims)
        self.margin = margin
        self.distance_metric = _resolve_distance_metric(distance_metric)

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        sentences = _extract_sentences(outputs)

        def compute(sentences: torch.Tensor) -> torch.Tensor:
            sentence1, sentence2 = sentences[0::2], sentences[1::2]
            distances = self.distance_metric(sentence1, sentence2)
            labels = inputs['labels'].to(sentence1.dtype).view(-1)
            losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(self.margin - distances).pow(2))
            return losses.mean()

        return LossOutput(loss=self._mrl_reduce(sentences, compute), num_tokens=0)


class OnlineContrastiveLoss(EmbeddingLoss):
    """Contrastive loss restricted to the HARD pairs within the batch.

    Only positives that are farther apart than the closest negative -- and negatives that are
    closer than the farthest positive -- contribute. Easy pairs are already ranked correctly, so
    dropping them concentrates the gradient on the decision boundary. The single-element
    fallbacks (``negs.mean()`` / ``poss.mean()``) mirror legacy: with fewer than two pairs on a
    side there is no counterpart to compare against.

    Args:
        margin: Hinge for the selected negative pairs.
        distance_metric: Name in :data:`DISTANCE_METRICS` or a callable ``(x, y) -> distances``.
        mrl_dims: See :class:`EmbeddingLoss`.
    """

    require_logits = True
    require_entropy = False
    require_logps = False

    def __init__(self, margin: float = 0.5, distance_metric='cosine', mrl_dims=None, **kwargs):
        super().__init__(mrl_dims=mrl_dims)
        self.margin = margin
        self.distance_metric = _resolve_distance_metric(distance_metric)

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        sentences = _extract_sentences(outputs)

        def compute(sentences: torch.Tensor) -> torch.Tensor:
            sentence1, sentence2 = sentences[0::2], sentences[1::2]
            distances = self.distance_metric(sentence1, sentence2)
            labels = inputs['labels'].view(-1)
            negs = distances[labels == 0]
            poss = distances[labels == 1]
            if len(negs) == 0 or len(poss) == 0:
                # One side of the contrast is absent, so no pair can be ranked; keep the graph alive
                # so DDP/FSDP does not hang waiting for gradients.
                return distances.sum() * 0.0

            negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
            return positive_loss + negative_loss

        return LossOutput(loss=self._mrl_reduce(sentences, compute), num_tokens=0)
