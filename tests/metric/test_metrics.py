# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for metric classes: Accuracy, LossMetric, TrainMetric, CompletionRewardMetric, DPOMetric, GRPOMetric, EmbeddingMetric."""
import time

import pytest
import torch
import torch.nn.functional as F

from twinkle.metric import (
    Accuracy,
    CompletionRewardMetric,
    DPOMetric,
    EmbeddingMetric,
    GRPOMetric,
    GSPOMetric,
    CISPOMetric,
    LossMetric,
    TrainMetric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_dist_metric(cls, **kwargs):
    """Instantiate a metric without distributed groups."""
    return cls(device_mesh=None, process_group=None, **kwargs)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:

    def test_perfect_accuracy(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([1, 2, 3, 4])
        logits = torch.zeros(4, 10)
        logits[0, 1] = 10.0
        logits[1, 2] = 10.0
        logits[2, 3] = 10.0
        logits[3, 4] = 10.0
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        m.accumulate(inputs, outputs)
        result = m.calculate()
        assert result['accuracy'] == '1.00'

    def test_zero_accuracy(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([0, 1, 2])
        logits = torch.zeros(3, 10)
        logits[0, 9] = 10.0  # wrong
        logits[1, 8] = 10.0  # wrong
        logits[2, 7] = 10.0  # wrong
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        m.accumulate(inputs, outputs)
        result = m.calculate()
        assert result['accuracy'] == '0.00'

    def test_partial_accuracy(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([1, 2])
        logits = torch.zeros(2, 10)
        logits[0, 1] = 10.0  # correct
        logits[1, 5] = 10.0  # wrong
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        m.accumulate(inputs, outputs)
        result = m.calculate()
        assert result['accuracy'] == '0.50'

    def test_accuracy_with_completion_mask(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([1, 2, 3, 4])
        logits = torch.zeros(4, 10)
        logits[:, 0] = 10.0  # all predict token 0 → wrong
        # Only count positions where mask is True
        mask = torch.tensor([True, False, True, False])
        inputs = {'labels': labels, 'completion_mask': mask}
        outputs = {'logits': logits}
        m.accumulate(inputs, outputs)
        result = m.calculate()
        # positions 0 and 2 are counted, both wrong → 0%
        assert result['accuracy'] == '0.00'

    def test_accuracy_no_logits_skips(self):
        m = _no_dist_metric(Accuracy)
        inputs = {'labels': torch.tensor([1, 2])}
        outputs = {}
        m.accumulate(inputs, outputs)
        result = m.calculate()
        assert result == {}

    def test_accuracy_reset(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([1])
        logits = torch.zeros(1, 10)
        logits[0, 1] = 10.0
        m.accumulate({'labels': labels}, {'logits': logits})
        m.reset()
        assert m.total_correct == 0
        assert m.total_count == 0

    def test_accuracy_with_ignore_index(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([1, -100, 3])
        logits = torch.zeros(3, 10)
        logits[0, 1] = 10.0  # correct
        logits[2, 5] = 10.0  # wrong
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        m.accumulate(inputs, outputs)
        result = m.calculate()
        # 1 correct out of 2 valid tokens
        assert result['accuracy'] == '0.50'

    def test_accuracy_does_not_accept_list(self):
        m = _no_dist_metric(Accuracy)
        inputs = [{'labels': torch.tensor([1])}]
        with pytest.raises(AssertionError):
            m.accumulate(inputs, {'logits': torch.zeros(1, 10)})


# ---------------------------------------------------------------------------
# LossMetric
# ---------------------------------------------------------------------------

class TestLossMetric:

    def test_basic_accumulate(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(2.5), 'num_tokens': torch.tensor(4.0)})
        result = m.calculate()
        assert 'loss' in result

    def test_loss_mean_reduction(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(3.0)}, loss_reduction='mean', num_tokens=3)
        # With num_tokens, avg_loss = total_loss / num_tokens
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(6.0)}, loss_reduction='sum', num_tokens=6)
        result = m.calculate()
        assert 'loss' in result

    def test_loss_grad_norm(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(1.0)}, grad_norm=0.5)
        result = m.calculate()
        assert 'grad_norm' in result

    def test_loss_no_loss_skips(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {})
        assert m.total_count == 0

    def test_loss_reset(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(1.0)})
        m.reset()
        assert m.total_loss == 0
        assert m.total_count == 0


# ---------------------------------------------------------------------------
# TrainMetric
# ---------------------------------------------------------------------------

class TestTrainMetric:

    def test_basic_train_metric(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=1e-4, step=10, gradient_accumulation_steps=2)
        result = m.calculate()
        assert 'learning rate' in result
        assert 'iters' in result
        assert result['iters'] == 5  # 10 // 2

    def test_train_metric_lr_list(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=[1e-4, 2e-4], step=10, gradient_accumulation_steps=1)
        result = m.calculate()
        assert 'learning rate(param group 1)' in result
        assert 'learning rate(param group 2)' in result

    def test_train_metric_single_lr_in_list(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=[1e-4], step=5, gradient_accumulation_steps=1)
        result = m.calculate()
        assert 'learning rate' in result
        # Single-element list should be flattened to scalar

    def test_train_metric_speed(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=1e-4, step=10)
        # Advance time slightly
        result = m.calculate()
        assert 'speed' in result

    def test_train_metric_reset(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=1e-4, step=10)
        m.reset()
        # After reset, time is updated but step info persists
        assert m.last_step == 10


# ---------------------------------------------------------------------------
# CompletionRewardMetric
# ---------------------------------------------------------------------------

class TestCompletionRewardMetric:

    def test_basic_rewards(self):
        m = _no_dist_metric(CompletionRewardMetric)
        m.accumulate(rewards={'accuracy': [1.0, 0.0, 1.0]}, completion_lengths=[10, 20, 30])
        result = m.calculate()
        assert 'train/accuracy_reward' in result
        assert 'train/completion_length' in result

    def test_reward_std(self):
        m = _no_dist_metric(CompletionRewardMetric)
        m.accumulate(rewards={'format': [1.0, 0.5, 0.0]})
        result = m.calculate()
        assert 'train/format_reward_std' in result

    def test_generate_time(self):
        m = _no_dist_metric(CompletionRewardMetric)
        m.accumulate(rewards={}, generate_time=1.5, weight_sync_time=0.3)
        result = m.calculate()
        assert 'profiling/Time taken: generate' in result
        assert 'profiling/Time taken: move_model_to_sampler' in result

    def test_empty_accumulate(self):
        m = _no_dist_metric(CompletionRewardMetric)
        m.accumulate()
        result = m.calculate()
        assert result == {}

    def test_mean_empty_list(self):
        assert CompletionRewardMetric._mean([]) == -1.0

    def test_std_single_element(self):
        assert CompletionRewardMetric._std([1.0]) == 0.0


# ---------------------------------------------------------------------------
# DPOMetric
# ---------------------------------------------------------------------------

class TestDPOMetric:

    def test_basic_dpo_metric(self):
        m = _no_dist_metric(DPOMetric, beta=0.1)
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        logps = torch.randn(2, 3)
        m.accumulate({'labels': labels}, {'logps': logps})
        result = m.calculate()
        assert 'logps/chosen' in result
        assert 'logps/rejected' in result

    def test_dpo_metric_with_ref(self):
        m = _no_dist_metric(DPOMetric, beta=0.1)
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        logps = torch.randn(2, 3)
        ref_logps = torch.randn(2, 3)
        m.accumulate({'labels': labels}, {'logps': logps}, ref_outputs={'logps': ref_logps})
        result = m.calculate()
        assert 'rewards/chosen' in result
        assert 'rewards/accuracies' in result

    def test_dpo_metric_no_logps_skips(self):
        m = _no_dist_metric(DPOMetric, beta=0.1)
        m.accumulate({'labels': torch.tensor([[1, 2]])}, {})
        result = m.calculate()
        assert result == {}

    def test_dpo_metric_reset(self):
        m = _no_dist_metric(DPOMetric, beta=0.1)
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        m.accumulate({'labels': labels}, {'logps': torch.randn(2, 3)})
        m.reset()
        assert m.total_count == 0

    def test_dpo_metric_odd_batch_raises(self):
        m = _no_dist_metric(DPOMetric, beta=0.1)
        labels = torch.tensor([[1, 2, -100]])
        logps = torch.randn(1, 3)
        with pytest.raises(AssertionError, match='even'):
            m.accumulate({'labels': labels}, {'logps': logps})


# ---------------------------------------------------------------------------
# GRPOMetric / GSPOMetric / CISPOMetric
# ---------------------------------------------------------------------------

class TestGRPOMetric:

    def test_basic_grpo_metric(self):
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps})
        result = m.calculate()
        assert 'train/policy_confidence' in result
        assert 'train/mean_new_logp' in result

    def test_grpo_metric_with_old(self):
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        old_logps = [torch.randn(1, 2)]  # only 2 valid tokens
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old_logps)
        result = m.calculate()
        assert 'train/approx_kl' in result
        assert 'train/logp_diff_mean' in result

    def test_grpo_metric_empty_outputs(self):
        m = _no_dist_metric(GRPOMetric)
        m.accumulate({'labels': torch.tensor([[1, 2]])}, None)
        # Should not crash
        result = m.calculate()
        # Might be empty if nothing accumulated
        assert isinstance(result, dict)

    def test_grpo_metric_entropy(self):
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        entropies = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps, 'entropies': entropies})
        result = m.calculate()
        assert 'train/entropy' in result

    def test_grpo_metric_reset(self):
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps})
        m.reset()
        assert m.n_tokens == 0
        assert m.sum_new == 0.0


class TestGSPOMetric:

    def test_basic_gspo_metric(self):
        m = _no_dist_metric(GSPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps})
        result = m.calculate()
        assert 'train/policy_confidence' in result


class TestCISPOMetric:

    def test_basic_cispo_metric(self):
        m = _no_dist_metric(CISPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps})
        result = m.calculate()
        assert 'train/policy_confidence' in result


# ---------------------------------------------------------------------------
# EmbeddingMetric
# ---------------------------------------------------------------------------

class TestEmbeddingMetric:

    def test_basic_embedding_metric(self):
        m = _no_dist_metric(EmbeddingMetric)
        # 2 samples: anchor (label=1) + positive (label=0), then another pair
        labels = torch.tensor([1, 0, 1, 0])  # Two anchor-positive pairs
        embeddings = torch.randn(4, 8)
        m.accumulate({'labels': labels}, {'embeddings': embeddings})
        result = m.calculate()
        assert 'pos_sim' in result

    def test_embedding_metric_with_loss(self):
        m = _no_dist_metric(EmbeddingMetric)
        labels = torch.tensor([1, 0])
        embeddings = torch.randn(2, 8)
        m.accumulate({'labels': labels}, {'embeddings': embeddings, 'loss': torch.tensor(2.5)})
        result = m.calculate()
        assert 'loss' in result

    def test_embedding_metric_no_embeddings_skips(self):
        m = _no_dist_metric(EmbeddingMetric)
        m.accumulate({'labels': torch.tensor([1, 0])}, {})
        result = m.calculate()
        assert result == {}

    def test_embedding_metric_fallback_to_logits(self):
        m = _no_dist_metric(EmbeddingMetric)
        labels = torch.tensor([1, 0])
        logits_2d = torch.randn(2, 8)
        m.accumulate({'labels': labels}, {'logits': logits_2d})
        result = m.calculate()
        assert 'pos_sim' in result

    def test_embedding_metric_3d_logits(self):
        m = _no_dist_metric(EmbeddingMetric)
        labels = torch.tensor([1, 0])
        logits_3d = torch.randn(2, 5, 8)  # 3D → CLS pooled
        m.accumulate({'labels': labels}, {'logits': logits_3d})
        result = m.calculate()
        assert 'pos_sim' in result

    def test_embedding_metric_no_anchors(self):
        m = _no_dist_metric(EmbeddingMetric)
        labels = torch.tensor([0, 0])  # No anchors
        embeddings = torch.randn(2, 8)
        m.accumulate({'labels': labels}, {'embeddings': embeddings})
        # No anchors → no positive similarity
        assert m.pos_count == 0
