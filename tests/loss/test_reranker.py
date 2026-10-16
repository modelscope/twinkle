# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the reranker (cross-encoder) losses (``reranker.py``).

A reranker scores each (query, document) pair jointly through a classification head, so the
signal lives in ``outputs['logits']`` (shape ``[B]`` or ``[B, 1]``) rather than in a pooled
embedding, and there is no pair interleaving. ``PointwiseRerankerLoss`` treats every document
as an independent binary-relevance example (BCE); ``ListwiseRerankerLoss`` runs a softmax over
each query's candidate list with the positive at index 0 (CE).

Every case drives the real ``__call__`` on real score tensors with ``backward()`` and numeric
assertions against closed-form references. CPU only, no stubs.
"""
import math

import pytest
import torch
from torch import nn

from twinkle.loss import ListwiseRerankerLoss, PointwiseRerankerLoss, torch_loss_mapping


# ── registration ──────────────────────────────────────────────────────────────


def test_rerankers_registered_in_mapping():
    assert torch_loss_mapping['pointwise_reranker'] is PointwiseRerankerLoss
    assert torch_loss_mapping['listwise_reranker'] is ListwiseRerankerLoss


# ── PointwiseRerankerLoss ─────────────────────────────────────────────────────


class TestPointwiseRerankerLoss:

    def test_requires_logits_not_logps(self):
        assert PointwiseRerankerLoss.require_logits is True
        assert PointwiseRerankerLoss.require_logps is False

    def test_zero_scores_unit_labels_known_value(self):
        """sigmoid(0)=0.5, so BCE against label 1 is exactly -log(0.5)=ln 2."""
        logits = torch.zeros(4)
        labels = torch.ones(4)
        result = PointwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        assert result['loss'].item() == pytest.approx(math.log(2.0), abs=1e-6)
        assert result['num_tokens'] == 0

    def test_matches_manual_bce_with_logits(self):
        torch.manual_seed(0)
        logits = torch.randn(8)
        labels = torch.randint(0, 2, (8,)).float()
        result = PointwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        expected = nn.BCEWithLogitsLoss()(logits, labels)
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    def test_squeezes_trailing_axis_from_2d_scores(self):
        """A [B, 1] head output is squeezed to [B]; result matches the flat form."""
        torch.manual_seed(0)
        flat = torch.randn(6)
        labels = torch.randint(0, 2, (6,)).float()
        r_flat = PointwiseRerankerLoss()({'labels': labels}, {'logits': flat})
        r_2d = PointwiseRerankerLoss()({'labels': labels}, {'logits': flat.unsqueeze(-1)})
        assert torch.allclose(r_flat['loss'], r_2d['loss'], atol=1e-6)

    def test_confident_correct_prediction_near_zero_loss(self):
        """Large positive score for label 1 -> BCE approaches 0."""
        logits = torch.tensor([20.0, 20.0])
        labels = torch.ones(2)
        result = PointwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        assert result['loss'].item() < 1e-6

    def test_gradient_flows_to_scores(self):
        logits = torch.randn(6, requires_grad=True)
        labels = torch.randint(0, 2, (6,)).float()
        result = PointwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        result['loss'].backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


# ── ListwiseRerankerLoss ──────────────────────────────────────────────────────


class TestListwiseRerankerLoss:

    def test_requires_logits_not_logps(self):
        assert ListwiseRerankerLoss.require_logits is True
        assert ListwiseRerankerLoss.require_logps is False

    def test_nonpositive_temperature_raises(self):
        with pytest.raises(ValueError, match='temperature must be > 0'):
            ListwiseRerankerLoss(temperature=0.0)
        with pytest.raises(ValueError, match='temperature must be > 0'):
            ListwiseRerankerLoss(temperature=-1.0)

    def test_perfect_ranking_near_zero_loss(self):
        """Positive (index 0) far above negatives -> softmax -> CE ~ 0."""
        logits = torch.tensor([10.0, -10.0, -10.0])
        labels = torch.tensor([1, 0, 0])
        result = ListwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        assert result['loss'].item() < 1e-3
        assert result['num_tokens'] == 0

    def test_single_group_matches_manual_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(4)
        labels = torch.tensor([1, 0, 0, 0])
        result = ListwiseRerankerLoss(temperature=1.0)({'labels': labels}, {'logits': logits})
        expected = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long))
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    def test_two_equal_groups_average_to_one(self):
        """Loss is the mean over groups; two identical groups give the single-group value."""
        group = torch.tensor([2.0, 0.5, -1.0])
        logits = torch.cat([group, group])
        labels = torch.tensor([1, 0, 0, 1, 0, 0])
        result = ListwiseRerankerLoss(temperature=1.0)({'labels': labels}, {'logits': logits})
        single = nn.CrossEntropyLoss()(group.unsqueeze(0), torch.zeros(1, dtype=torch.long))
        assert torch.allclose(result['loss'], single, atol=1e-6)

    def test_temperature_sharpens_the_distribution(self):
        """A smaller temperature sharpens softmax, changing the loss for imperfect rankings."""
        logits = torch.tensor([1.0, 0.8, -0.5])
        labels = torch.tensor([1, 0, 0])
        hot = ListwiseRerankerLoss(temperature=2.0)({'labels': labels}, {'logits': logits})['loss']
        cold = ListwiseRerankerLoss(temperature=0.5)({'labels': labels}, {'logits': logits})['loss']
        assert not torch.allclose(hot, cold, atol=1e-5)

    def test_min_group_size_skips_positive_only_group(self):
        """A group of size 1 (positive immediately followed by another positive) is dropped."""
        # positive_indices = [0, 1]; group0 = logits[0:1] (size 1 < 2 -> skipped),
        # group1 = logits[1:4] (size 3 -> counted). Only the second group contributes.
        logits = torch.tensor([5.0, 2.0, 0.5, -1.0])
        labels = torch.tensor([1, 1, 0, 0])
        result = ListwiseRerankerLoss(min_group_size=2)({'labels': labels}, {'logits': logits})
        expected = nn.CrossEntropyLoss()(logits[1:4].unsqueeze(0), torch.zeros(1, dtype=torch.long))
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    def test_all_groups_too_small_returns_zero_but_keeps_graph(self):
        """Every group below min_group_size -> num_groups==0 -> zero loss attached to the graph."""
        logits = torch.tensor([1.0, 2.0], requires_grad=True)
        labels = torch.tensor([1, 1])  # two positive-only groups of size 1
        result = ListwiseRerankerLoss(min_group_size=2)({'labels': labels}, {'logits': logits})
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-8)
        result['loss'].backward()
        assert logits.grad is not None, 'zero loss must stay attached to the autograd graph'

    def test_no_positives_returns_zero_but_keeps_graph(self):
        logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        labels = torch.tensor([0, 0, 0])
        result = ListwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-8)
        result['loss'].backward()
        assert logits.grad is not None

    def test_squeezes_trailing_axis_from_2d_scores(self):
        torch.manual_seed(0)
        flat = torch.randn(4)
        labels = torch.tensor([1, 0, 0, 0])
        r_flat = ListwiseRerankerLoss()({'labels': labels}, {'logits': flat})
        r_2d = ListwiseRerankerLoss()({'labels': labels}, {'logits': flat.unsqueeze(-1)})
        assert torch.allclose(r_flat['loss'], r_2d['loss'], atol=1e-6)

    def test_gradient_flows_to_scores(self):
        logits = torch.tensor([1.0, 0.0, -0.5, 0.3, 2.0, -1.0], requires_grad=True)
        labels = torch.tensor([1, 0, 0, 1, 0, 0])
        result = ListwiseRerankerLoss()({'labels': labels}, {'logits': logits})
        result['loss'].backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
