# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the embedding / contrastive loss family (``infonce.py``).

Every case drives the real loss ``__call__`` on real embedding tensors and, where
the objective is differentiable, runs ``backward()`` to assert the gradient actually
reaches the embeddings. No stubs, no GPU, no distributed: ``InfonceLoss._gather_across_dp``
early-returns when the process group is uninitialised, so the whole family runs on CPU.

Covered classes: ``EmbeddingLoss`` (MRL aggregation base), ``InfonceLoss``,
``CosineSimilarityLoss``, ``ContrastiveLoss``, ``OnlineContrastiveLoss``.
"""
import numpy as np
import pytest
import torch

from twinkle.loss import (ContrastiveLoss, CosineSimilarityLoss, EmbeddingLoss, InfonceLoss, OnlineContrastiveLoss)


# ── Batch builders ────────────────────────────────────────────────────────────
# InfoNCE layout: each sample is ``anchor(1) + positive(1) + negatives(n)`` laid out
# flat; ``labels`` is a 1-D mask whose ``1`` entries mark the start of every group.


def _make_infonce_batch(n_groups=3, n_neg=2, dim=8, seed=0, requires_grad=True):
    """Uniform groups: every sample has the same ``anchor+positive+n_neg`` shape."""
    torch.manual_seed(seed)
    per_group = 2 + n_neg
    total = n_groups * per_group
    embeddings = torch.randn(total, dim, requires_grad=requires_grad)
    labels = torch.zeros(total, dtype=torch.long)
    for g in range(n_groups):
        labels[g * per_group] = 1
    return {'labels': labels}, {'embeddings': embeddings}


def _make_ragged_infonce_batch(neg_counts, dim=8, seed=0, requires_grad=True):
    """Ragged groups: per-sample negative counts differ, exercising the unbatched path."""
    torch.manual_seed(seed)
    embs, labels = [], []
    for n in neg_counts:
        embs.append(torch.randn(2 + n, dim))
        grp = torch.zeros(2 + n, dtype=torch.long)
        grp[0] = 1
        labels.append(grp)
    embeddings = torch.cat(embs)
    if requires_grad:
        embeddings = embeddings.clone().requires_grad_(True)
    return {'labels': torch.cat(labels)}, {'embeddings': embeddings}


def _make_pair_batch(vectors, labels, requires_grad=True):
    """Interleaved-pair layout ``[s1_0, s2_0, s1_1, s2_1, ...]`` for the pair losses."""
    embeddings = torch.stack(vectors)
    if requires_grad:
        embeddings = embeddings.clone().requires_grad_(True)
    return {'labels': torch.as_tensor(labels)}, {'embeddings': embeddings}


def _assert_grad_reached(embeddings):
    assert embeddings.grad is not None, 'backward() produced no gradient on the embeddings'
    assert torch.isfinite(embeddings.grad).all(), 'embedding gradient contains NaN/Inf'
    assert embeddings.grad.abs().sum() > 0, 'embedding gradient is exactly zero everywhere'


# ── EmbeddingLoss: Matryoshka (MRL) aggregation base ──────────────────────────


class TestEmbeddingLossMrl:

    def test_mrl_disabled_calls_compute_once_untouched(self):
        """Without mrl_dims the plain path runs compute() on the raw embeddings exactly once."""
        loss = EmbeddingLoss(mrl_dims=None)
        sentences = torch.randn(4, 8)
        calls = []

        def compute(x):
            calls.append(x)
            return x.sum()

        out = loss._mrl_reduce(sentences, compute)
        assert len(calls) == 1
        # bit-for-bit unchanged: compute received the very same tensor object
        assert calls[0] is sentences
        assert torch.equal(out, sentences.sum())

    def test_mrl_weighted_sum_over_prefixes(self):
        """Each prefix is re-normalized then scaled by its weight; result is the weighted sum."""
        loss = EmbeddingLoss(mrl_dims={4: 1.0, 8: 0.5})
        sentences = torch.randn(3, 8)

        def compute(x):
            # scalar per prefix: mean of the (already re-normalized) embeddings
            return x.mean()

        out = loss._mrl_reduce(sentences, compute)
        import torch.nn.functional as F
        expected = 1.0 * compute(F.normalize(sentences[..., :4], p=2, dim=-1))
        expected = expected + 0.5 * compute(F.normalize(sentences[..., :8], p=2, dim=-1))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_mrl_skips_prefixes_larger_than_hidden(self):
        """A prefix wider than the embedding is skipped, not padded; the narrower one still runs."""
        loss = EmbeddingLoss(mrl_dims={4: 1.0, 16: 2.0})
        sentences = torch.randn(3, 8)
        out = loss._mrl_reduce(sentences, lambda x: x.sum())
        assert torch.isfinite(out)

    def test_mrl_all_prefixes_too_large_raises(self):
        """If every configured dim exceeds the embedding size there is nothing to optimize."""
        loss = EmbeddingLoss(mrl_dims={16: 1.0, 32: 0.5})
        sentences = torch.randn(3, 8)
        with pytest.raises(ValueError, match='exceeds the embedding size'):
            loss._mrl_reduce(sentences, lambda x: x.sum())

    def test_subclass_without_mrl_support_rejects_mrl_dims(self):
        """supports_mrl=False subclasses refuse mrl_dims at construction time."""
        assert CosineSimilarityLoss.supports_mrl is False
        with pytest.raises(ValueError, match='does not support mrl_dims'):
            CosineSimilarityLoss(mrl_dims={4: 1.0})


# ── InfonceLoss ───────────────────────────────────────────────────────────────


class TestInfonceLoss:

    def test_intra_sample_loss_runs_and_backprops(self):
        """use_batch=False: contrast only against each sample's own negatives."""
        inputs, outputs = _make_infonce_batch(n_groups=3, n_neg=2)
        loss_fn = InfonceLoss(temperature=0.1, use_batch=False)
        result = loss_fn(inputs, outputs)
        assert result['loss'].dim() == 0
        assert torch.isfinite(result['loss'])
        assert result['num_tokens'] == 0
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_in_batch_loss_runs_and_backprops(self):
        """use_batch=True: cross-sample in-batch negatives (single rank -> no gather)."""
        inputs, outputs = _make_infonce_batch(n_groups=4, n_neg=2)
        loss_fn = InfonceLoss(temperature=0.1, use_batch=True)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_intra_and_in_batch_differ(self):
        """The two negative-sharing regimes produce genuinely different objectives."""
        inputs_a, outputs_a = _make_infonce_batch(n_groups=4, n_neg=2, seed=7)
        inputs_b = {'labels': inputs_a['labels'].clone()}
        outputs_b = {'embeddings': outputs_a['embeddings'].detach().clone()}
        intra = InfonceLoss(use_batch=False)(inputs_a, outputs_a)['loss']
        in_batch = InfonceLoss(use_batch=True)(inputs_b, outputs_b)['loss']
        assert not torch.allclose(intra, in_batch, atol=1e-5)

    def test_ragged_groups_use_unbatched_path(self):
        """Variable per-sample negative counts fall back to the loop-based unbatched loss."""
        inputs, outputs = _make_ragged_infonce_batch(neg_counts=[1, 2, 3])
        loss_fn = InfonceLoss(use_batch=True)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_hard_negatives_truncation(self):
        """hard_negatives smaller than the available negatives truncates each group."""
        inputs, outputs = _make_infonce_batch(n_groups=3, n_neg=3)
        loss_fn = InfonceLoss(use_batch=True, hard_negatives=1)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_hard_negatives_upsampling_on_ragged(self):
        """hard_negatives larger than available upsamples negatives to a uniform count."""
        np.random.seed(0)
        inputs, outputs = _make_ragged_infonce_batch(neg_counts=[1, 1, 1])
        loss_fn = InfonceLoss(use_batch=True, hard_negatives=3)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_include_qq_and_dd(self):
        """Query-query and doc-doc similarity blocks are appended to the logit matrix."""
        inputs, outputs = _make_infonce_batch(n_groups=4, n_neg=2)
        loss_fn = InfonceLoss(use_batch=True, include_qq=True, include_dd=True)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_mask_fake_negative(self):
        """Logits above positive+margin are masked as suspected false negatives."""
        inputs, outputs = _make_infonce_batch(n_groups=4, n_neg=2)
        loss_fn = InfonceLoss(use_batch=True, mask_fake_negative=True, fake_neg_margin=0.1)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_mask_fake_negative_requires_positive_margin(self):
        """A non-positive margin would mask the positive itself -> rejected at construction."""
        with pytest.raises(ValueError, match='fake_neg_margin must be > 0'):
            InfonceLoss(mask_fake_negative=True, fake_neg_margin=0.0)
        with pytest.raises(ValueError, match='fake_neg_margin must be > 0'):
            InfonceLoss(mask_fake_negative=True, fake_neg_margin=-0.5)

    def test_mrl_dims_weighted_prefixes(self):
        """InfonceLoss is MRL-capable: the objective is summed over truncated prefixes."""
        inputs, outputs = _make_infonce_batch(n_groups=3, n_neg=2, dim=8)
        loss_fn = InfonceLoss(use_batch=True, mrl_dims={4: 1.0, 8: 0.5})
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_no_groups_returns_zero_but_keeps_graph(self):
        """labels with no group start yields an empty split -> zero loss that still backprops."""
        torch.manual_seed(0)
        embeddings = torch.randn(6, 8, requires_grad=True)
        inputs = {'labels': torch.zeros(6, dtype=torch.long)}
        result = InfonceLoss(use_batch=True)(inputs, {'embeddings': embeddings})
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-8)
        result['loss'].backward()
        assert embeddings.grad is not None, 'zero loss must stay attached to the autograd graph'

    def test_reads_logits_fallback_and_cls_pooling(self):
        """Without 'embeddings', falls back to 'logits'; a 3-D tensor takes the CLS token [:, 0]."""
        torch.manual_seed(0)
        logits3d = torch.randn(6, 5, 8, requires_grad=True)  # [B, T, D] -> pooled to [:, 0]
        labels = torch.tensor([1, 0, 0, 1, 0, 0])
        result = InfonceLoss(use_batch=True)({'labels': labels}, {'logits': logits3d})
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        assert logits3d.grad is not None


# ── CosineSimilarityLoss ──────────────────────────────────────────────────────


class TestCosineSimilarityLoss:

    def test_identical_pairs_with_unit_label_have_zero_loss(self):
        """cos(v, v)=1; regressing onto label 1.0 gives MSE 0."""
        v0 = torch.randn(8)
        v1 = torch.randn(8)
        inputs, outputs = _make_pair_batch([v0, v0, v1, v1], labels=[1.0, 1.0])
        result = CosineSimilarityLoss()(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)
        assert result['num_tokens'] == 0

    def test_orthogonal_pairs_against_unit_label(self):
        """cos of orthogonal vectors is 0; MSE against label 1.0 is exactly 1.0."""
        e0 = torch.zeros(8)
        e0[0] = 1.0
        e1 = torch.zeros(8)
        e1[1] = 1.0
        inputs, outputs = _make_pair_batch([e0, e1], labels=[1.0])
        result = CosineSimilarityLoss()(inputs, outputs)
        assert result['loss'].item() == pytest.approx(1.0, abs=1e-6)

    def test_gradient_flows_to_both_sentences(self):
        inputs, outputs = _make_pair_batch([torch.randn(8), torch.randn(8)], labels=[0.3])
        result = CosineSimilarityLoss()(inputs, outputs)
        assert torch.isfinite(result['loss'])
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_labels_are_per_pair_not_per_row(self):
        """4 interleaved rows -> 2 pairs -> exactly 2 label scores."""
        vecs = [torch.randn(8) for _ in range(4)]
        inputs, outputs = _make_pair_batch(vecs, labels=[0.9, 0.1])
        result = CosineSimilarityLoss()(inputs, outputs)
        assert torch.isfinite(result['loss'])


# ── ContrastiveLoss ───────────────────────────────────────────────────────────


class TestContrastiveLoss:

    def test_similar_identical_pairs_zero_loss(self):
        """label=1 with distance 0 contributes 0.5*0^2 = 0."""
        v = torch.randn(8)
        inputs, outputs = _make_pair_batch([v, v], labels=[1])
        result = ContrastiveLoss(margin=0.5, distance_metric='cosine')(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)

    def test_dissimilar_beyond_margin_zero_loss(self):
        """label=0 with distance > margin: relu(margin - dist)=0, so the hinge is inactive."""
        a = torch.zeros(8)
        a[0] = 1.0
        b = torch.zeros(8)
        b[1] = 1.0  # cosine distance = 1 - 0 = 1.0 > margin 0.5
        inputs, outputs = _make_pair_batch([a, b], labels=[0])
        result = ContrastiveLoss(margin=0.5, distance_metric='cosine')(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)

    def test_dissimilar_within_margin_is_penalised(self):
        """label=0 but distance below margin -> positive hinge loss."""
        a = torch.zeros(8)
        a[0] = 1.0
        b = a.clone()
        b[1] = 0.05  # near-identical, cosine distance << margin
        inputs, outputs = _make_pair_batch([a, b], labels=[0])
        result = ContrastiveLoss(margin=0.5, distance_metric='cosine')(inputs, outputs)
        assert result['loss'].item() > 0.0

    def test_gradient_flows(self):
        inputs, outputs = _make_pair_batch([torch.randn(8), torch.randn(8)], labels=[1])
        result = ContrastiveLoss()(inputs, outputs)
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    @pytest.mark.parametrize('metric', ['cosine', 'euclidean', 'manhattan'])
    def test_named_distance_metrics_run(self, metric):
        inputs, outputs = _make_pair_batch([torch.randn(8), torch.randn(8)], labels=[1])
        result = ContrastiveLoss(distance_metric=metric)(inputs, outputs)
        assert torch.isfinite(result['loss'])

    def test_callable_distance_metric_accepted(self):
        """A custom callable (x, y) -> distances is used directly."""
        inputs, outputs = _make_pair_batch([torch.randn(8), torch.randn(8)], labels=[1])
        result = ContrastiveLoss(distance_metric=lambda x, y: (x - y).abs().sum(-1))(inputs, outputs)
        assert torch.isfinite(result['loss'])

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match='Unknown distance metric'):
            ContrastiveLoss(distance_metric='chebyshev')


# ── OnlineContrastiveLoss ─────────────────────────────────────────────────────


class TestOnlineContrastiveLoss:

    def test_mixed_hard_pairs_run_and_backprop(self):
        """Overlapping positive/negative distances keep hard pairs active -> nonzero gradient.

        Only *hard* pairs contribute: positives farther than the closest negative, and
        negatives closer than the farthest positive. We therefore build a far positive
        (orthogonal, cosine distance 1.0) and a near negative (near-identical, distance
        ~0) so both selections are non-empty and the objective is genuinely exercised.
        """

        def basis(i):
            e = torch.zeros(8)
            e[i] = 1.0
            return e

        vecs, labels = [], []
        # Hard positives: orthogonal pairs -> distance ~1.0 (far for a positive).
        vecs += [basis(0), basis(1)]
        labels += [1]
        vecs += [basis(2), basis(3)]
        labels += [1]
        # Hard negatives: near-identical pairs -> distance ~0 (close for a negative).
        vecs += [basis(4), basis(4) + 0.1 * basis(5)]
        labels += [0]
        vecs += [basis(6), basis(6) + 0.1 * basis(7)]
        labels += [0]
        inputs, outputs = _make_pair_batch(vecs, labels=labels)
        result = OnlineContrastiveLoss(margin=0.5)(inputs, outputs)
        assert torch.isfinite(result['loss'])
        assert result['loss'].item() > 0.0
        result['loss'].backward()
        _assert_grad_reached(outputs['embeddings'])

    def test_easy_pairs_contribute_nothing(self):
        """Correctly-ranked easy pairs (near positives, far negatives) drop out -> zero gradient."""
        torch.manual_seed(0)
        vecs, labels = [], []
        for _ in range(3):  # near-identical positives -> distance ~0
            v = torch.randn(8)
            vecs += [v, v + 0.01 * torch.randn(8)]
            labels += [1]
        for _ in range(3):  # far negatives -> distance ~1
            vecs += [torch.randn(8), torch.randn(8)]
            labels += [0]
        inputs, outputs = _make_pair_batch(vecs, labels=labels)
        result = OnlineContrastiveLoss(margin=0.5)(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-8)
        result['loss'].backward()
        assert torch.isfinite(outputs['embeddings'].grad).all()

    def test_only_positives_returns_zero_but_keeps_graph(self):
        """No negatives -> one side of the contrast absent -> zero loss, graph alive."""
        v = torch.randn(8)
        inputs, outputs = _make_pair_batch([v, v, v, v], labels=[1, 1])
        result = OnlineContrastiveLoss()(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-8)
        result['loss'].backward()
        assert outputs['embeddings'].grad is not None

    def test_only_negatives_returns_zero_but_keeps_graph(self):
        """No positives -> symmetric degenerate case."""
        torch.manual_seed(0)
        vecs = [torch.randn(8) for _ in range(4)]
        inputs, outputs = _make_pair_batch(vecs, labels=[0, 0])
        result = OnlineContrastiveLoss()(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-8)
        result['loss'].backward()
        assert outputs['embeddings'].grad is not None

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match='Unknown distance metric'):
            OnlineContrastiveLoss(distance_metric='nonsense')
