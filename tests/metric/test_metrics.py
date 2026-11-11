# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for metric classes: Accuracy, LossMetric, TrainMetric, CompletionRewardMetric, DPOMetric, GRPOMetric,
EmbeddingMetric, ExactMatch, RougeBleu."""
import time

import pytest
import torch
import torch.nn.functional as F

from twinkle.metric import (
    Accuracy,
    CompletionRewardMetric,
    DPOMetric,
    EmbeddingMetric,
    ExactMatch,
    GRPOMetric,
    GSPOMetric,
    CISPOMetric,
    LossMetric,
    RougeBleu,
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
        logits[0, 0] = 10.0  # position 0 predicted wrong, counted
        logits[1, 2] = 10.0  # position 1 predicted right, masked out
        logits[2, 0] = 10.0  # position 2 predicted wrong, counted
        logits[3, 4] = 10.0  # position 3 predicted right, masked out
        mask = torch.tensor([True, False, True, False])
        m.accumulate({'labels': labels, 'completion_mask': mask}, {'logits': logits})
        result = m.calculate()
        # The two counted positions are wrong; ignoring the mask would score 0.50.
        assert result['accuracy'] == '0.00'
        assert result['correct_tokens'] == 0
        assert result['total_tokens'] == 2

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

    def test_seq_strategy_scores_a_row_wrong_on_one_token(self):
        m = _no_dist_metric(Accuracy, strategy='seq')
        labels = torch.tensor([[1, 2, 3]])
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 9] = 10.0  # last token wrong -> the whole sequence is wrong
        m.accumulate({'labels': labels}, {'logits': logits})
        result = m.calculate()
        assert result['accuracy'] == '0.00'
        assert result['correct_sequences'] == 0
        assert result['total_sequences'] == 1

    def test_seq_strategy_all_tokens_correct(self):
        m = _no_dist_metric(Accuracy, strategy='seq')
        labels = torch.tensor([[1, 2, 3]])
        logits = torch.zeros(1, 3, 10)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        m.accumulate({'labels': labels}, {'logits': logits})
        result = m.calculate()
        assert result['accuracy'] == '1.00'
        assert result['correct_sequences'] == 1
        assert result['total_sequences'] == 1

    def test_seq_strategy_drops_rows_without_trainable_tokens(self):
        m = _no_dist_metric(Accuracy, strategy='seq')
        labels = torch.tensor([[1, 2, 3], [-100, -100, -100]])
        logits = torch.zeros(2, 3, 10)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0
        m.accumulate({'labels': labels}, {'logits': logits})
        result = m.calculate()
        assert result['accuracy'] == '1.00'
        assert result['correct_sequences'] == 1
        assert result['total_sequences'] == 1

    def test_seq_strategy_accepts_one_dimensional_labels(self):
        m = _no_dist_metric(Accuracy, strategy='seq')
        labels = torch.tensor([1, 2])
        logits = torch.zeros(2, 10)
        logits[0, 1] = 10.0
        logits[1, 2] = 10.0
        m.accumulate({'labels': labels}, {'logits': logits})
        result = m.calculate()
        assert result['accuracy'] == '1.00'
        assert result['total_sequences'] == 1

    def test_accuracy_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="token.*seq"):
            Accuracy(device_mesh=None, process_group=None, strategy='sentence')

    def test_float_labels_are_skipped(self):
        # Float labels are regression/value targets, not token ids: no token accuracy to report.
        m = _no_dist_metric(Accuracy)
        logits = torch.zeros(1, 2, 10)
        logits[0, :, 0] = 10.0
        m.accumulate({'labels': torch.tensor([[1.0, 2.0]])}, {'logits': logits})
        assert m.calculate() == {}

    def test_truncates_labels_and_mask_to_logits_width(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([[1, 2, 3, 4]])
        logits = torch.zeros(1, 2, 10)
        logits[0, 0, 3] = 10.0
        logits[0, 1, 4] = 10.0
        mask = torch.tensor([True, True, True, False])
        m.accumulate({'labels': labels, 'completion_mask': mask}, {'logits': logits})
        result = m.calculate()
        # Only the last two columns are scored; the truncated mask keeps one of them.
        assert result['accuracy'] == '1.00'
        assert result['correct_tokens'] == 1
        assert result['total_tokens'] == 1

    def test_truncates_labels_without_completion_mask(self):
        m = _no_dist_metric(Accuracy)
        labels = torch.tensor([[1, 2, 3, 4]])
        logits = torch.zeros(1, 2, 10)
        logits[0, 0, 3] = 10.0
        logits[0, 1, 4] = 10.0
        m.accumulate({'labels': labels}, {'logits': logits})
        result = m.calculate()
        assert result['accuracy'] == '1.00'
        assert result['total_tokens'] == 2


# ---------------------------------------------------------------------------
# LossMetric
# ---------------------------------------------------------------------------

class TestLossMetric:

    def test_basic_accumulate(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(2.5), 'num_tokens': torch.tensor(4.0)})
        result = m.calculate()
        assert 'loss' in result

    def test_sum_reduction_uses_num_tokens_from_outputs(self):
        # The token denominator comes from outputs['num_tokens']: (3 + 6) / 3. A kwarg of
        # the same name is not part of the contract, so it must not be needed here.
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(3.0)})
        m.accumulate(
            {'labels': torch.tensor([1])},
            {'loss': torch.tensor(6.0), 'num_tokens': torch.tensor(3.0)},
            loss_reduction='sum',
        )
        assert m.calculate()['loss'] == '3.00000'

    def test_loss_grad_norm(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(1.0)}, grad_norm=0.5)
        assert m.calculate()['grad_norm'] == '0.500000'

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

    def test_sum_reduction_counts_non_negative_labels_without_num_tokens(self):
        # No outputs['num_tokens'], so the denominator falls back to counting labels >= 0;
        # the 0 token must be counted, proving this is not a > 0 predicate.
        m = _no_dist_metric(LossMetric)
        inputs = {'labels': torch.tensor([[1, 0, -100, 3]])}
        m.accumulate(inputs, {'loss': torch.tensor(6.0)}, loss_reduction='sum')
        m.accumulate(inputs, {'loss': torch.tensor(6.0)}, loss_reduction='sum')
        assert m.num_tokens == 6
        assert m.calculate()['loss'] == '2.00000'

    def test_mean_without_num_tokens_averages_over_steps(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(3.0)})
        m.accumulate({'labels': torch.tensor([1])}, {'loss': torch.tensor(5.0)})
        assert m.calculate()['loss'] == '4.00000'

    def test_channel_loss_is_aggregated_and_zero_token_channels_are_dropped(self):
        m = _no_dist_metric(LossMetric)
        m.accumulate(
            {}, {
                'loss': torch.tensor(4.0),
                'num_tokens': torch.tensor(2),
                'channel_loss': {
                    'math': torch.tensor([3.0, 2.0]),
                    'code': (0.0, 0.0),
                },
            }, loss_reduction='sum')
        result = m.calculate()
        assert result['loss'] == '2.00000'
        assert result['loss_math'] == '1.5000'
        assert 'loss_code' not in result
        assert m.channel_loss == {}

    def test_calculate_without_accumulation_reports_zero_loss(self):
        assert _no_dist_metric(LossMetric).calculate() == {'loss': '0.00000'}

    def test_sum_reduction_accepts_list_inputs_without_num_tokens(self):
        m = _no_dist_metric(LossMetric)
        inputs = [{'labels': torch.tensor([[1, -100]])}, {'labels': torch.tensor([[2, 3]])}]
        m.accumulate(inputs, {'loss': torch.tensor(3.0)}, loss_reduction='sum')
        assert m.num_tokens == 3
        assert m.total_loss == 3.0


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

    def test_calculate_before_accumulate_reports_zero_step(self):
        m = _no_dist_metric(TrainMetric)
        result = m.calculate()
        assert 'learning rate' not in result
        assert result['iters'] == 0
        assert result['speed'].endswith('iters/s')

    def test_calculate_without_step_reports_only_the_learning_rate(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=1e-4, step=None)
        assert set(m.calculate()) == {'learning rate'}

    def test_elapsed_time_over_a_minute_is_reported_in_minutes(self):
        m = _no_dist_metric(TrainMetric)
        m.accumulate({}, {}, lr=1e-4, step=10)
        m.time = time.time() - 61
        assert m.calculate()['total time elapse'].endswith('minutes')


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
        assert result['train/format_reward_std'] == pytest.approx(0.5)

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

    def test_reset_clears_every_accumulator(self):
        m = _no_dist_metric(CompletionRewardMetric)
        m.accumulate(
            rewards={'accuracy': [1.0]}, completion_lengths=[3], generate_time=1.0, weight_sync_time=0.25)
        m.reset()
        assert m.calculate() == {}

    def test_repeated_reward_keys_extend_the_same_list(self):
        m = _no_dist_metric(CompletionRewardMetric)
        m.accumulate(rewards={'accuracy': [1.0]})
        m.accumulate(rewards={'accuracy': [0.0]})
        assert m.calculate()['train/accuracy_reward'] == 0.5


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

    def test_dpo_metric_accepts_json_ref_logps(self):
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        logps = torch.randn(2, 3)
        ref_logps = torch.randn(2, 3)
        tensor_metric = _no_dist_metric(DPOMetric, beta=0.1)
        json_metric = _no_dist_metric(DPOMetric, beta=0.1)

        tensor_metric.accumulate(
            {'labels': labels},
            {'logps': logps},
            ref_outputs={'logps': ref_logps},
        )
        json_metric.accumulate(
            {'labels': labels},
            {'logps': logps},
            ref_outputs={'logps': ref_logps.tolist()},
        )

        assert json_metric.calculate() == pytest.approx(tensor_metric.calculate())

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

    def test_dpo_metric_accepts_ref_logps_kwarg(self):
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        logps = torch.randn(2, 3)
        ref_logps = torch.randn(2, 3)
        via_outputs = _no_dist_metric(DPOMetric, beta=0.1)
        via_kwarg = _no_dist_metric(DPOMetric, beta=0.1)

        via_outputs.accumulate({'labels': labels}, {'logps': logps}, ref_outputs={'logps': ref_logps})
        via_kwarg.accumulate({'labels': labels}, {'logps': logps}, ref_logps=ref_logps)

        assert via_kwarg.calculate() == via_outputs.calculate()

    def test_dpo_metric_accepts_ragged_ref_logps_list(self):
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        logps = torch.randn(2, 3)
        ragged = _no_dist_metric(DPOMetric, beta=0.1)
        stacked = _no_dist_metric(DPOMetric, beta=0.1)

        # Response-only rows are padded to the label width; the ignored suffix stays zero.
        ragged.accumulate({'labels': labels}, {'logps': logps}, ref_logps=[[-0.5, -0.5], [-0.3, -0.3]])
        stacked.accumulate(
            {'labels': labels}, {'logps': logps},
            ref_logps=torch.tensor([[-0.5, -0.5, 0.0], [-0.3, -0.3, 0.0]]))

        assert ragged.calculate() == stacked.calculate()

    def test_dpo_metric_empty_ref_logps_list_skips_rewards(self):
        m = _no_dist_metric(DPOMetric, beta=0.1)
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        m.accumulate({'labels': labels}, {'logps': torch.randn(2, 3)}, ref_outputs={'logps': []})

        result = m.calculate()

        assert set(result) == {'logps/chosen', 'logps/rejected'}

    def test_dpo_metric_accepts_list_logps_and_list_inputs(self):
        label_rows = [torch.tensor([[1, 2, -100]]), torch.tensor([[3, 4, -100]])]
        logps_rows = [torch.tensor([[-1.0, -1.0, -9.0]]), torch.tensor([[-2.0, -2.0, -9.0]])]
        list_metric = _no_dist_metric(DPOMetric, beta=0.1)
        stacked_metric = _no_dist_metric(DPOMetric, beta=0.1)

        list_metric.accumulate(
            [{'labels': label_rows[0]}, {'labels': label_rows[1]}], {'logps': logps_rows})
        stacked_metric.accumulate(
            {'labels': torch.cat(label_rows)}, {'logps': torch.cat(logps_rows)})

        assert list_metric.calculate() == stacked_metric.calculate()
        assert list_metric.calculate() == {}  # calculate() resets, like the other metrics

    def test_dpo_metric_accepts_single_element_list_inputs(self):
        labels = torch.tensor([[1, 2, -100], [3, 4, -100]])
        logps = torch.tensor([[-1.0, -1.0, -9.0], [-2.0, -2.0, -9.0]])
        as_list = _no_dist_metric(DPOMetric, beta=0.1)
        as_dict = _no_dist_metric(DPOMetric, beta=0.1)

        as_list.accumulate([{'labels': labels}], {'logps': logps})
        as_dict.accumulate({'labels': labels}, {'logps': logps})

        assert as_list.calculate() == as_dict.calculate()

    def test_dpo_metric_one_dimensional_labels_do_not_form_pairs(self):
        # A 1-D label vector is promoted to a single row, which can never be an even batch.
        m = _no_dist_metric(DPOMetric, beta=0.1)
        with pytest.raises(AssertionError, match='even'):
            m.accumulate({'labels': torch.tensor([1, 2])}, {'logps': torch.randn(1, 2)})

    def test_dpo_metric_truncates_to_label_width(self):
        # labels are 4 wide but logps only 3: both are truncated to the first 3 columns.
        m = _no_dist_metric(DPOMetric, beta=0.1)
        labels = torch.tensor([[1, 2, 3, -100], [4, 5, 6, -100]])
        logps = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]])
        m.accumulate({'labels': labels}, {'logps': logps})

        result = m.calculate()

        assert result['logps/chosen'] == '-3.00'
        assert result['logps/rejected'] == '-6.00'


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
        assert m.calculate() == {}

    def test_grpo_metric_entropy(self):
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[1, 2, -100, -100]])
        logps = torch.randn(1, 4)
        entropies = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps, 'entropies': entropies})
        result = m.calculate()
        assert 'train/entropy' in result

    def test_grpo_metric_old_logps_wider_than_logps(self):
        """old_logps 来自 forward_only，序列维 pad 到整个 micro batch 的最大长度（dp split 之前），
        而 logps 只 pad 到本 rank 的最大长度 —— old 比 new 宽是常态，不能因此丢掉 ratio/kl。
        取值必须落在 mask 位上（不是行首 N 个），所以 mean_old_logp 是错位的判据。
        """
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11, 12, 13],
                               [-100, -100, -100, -100, 20, 21]])
        logps = torch.zeros(2, 6)
        logps[0, 2:6] = torch.tensor([-1.1, -2.1, -3.1, -4.1])
        logps[1, 4:6] = torch.tensor([-5.1, -6.1])
        old_logps = torch.zeros(2, 9)   # 右 pad 到 9 > 6
        old_logps[0, 2:6] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        old_logps[1, 4:6] = torch.tensor([-5.0, -6.0])
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old_logps)
        result = m.calculate()
        assert 'train/approx_kl' in result
        assert abs(result['train/mean_old_logp'] - (-3.5)) < 1e-6
        assert abs(result['train/logp_diff_mean'] - (-0.1)) < 1e-6

    def test_grpo_metric_old_logps_row_mismatch_skipped(self):
        """行数不匹配是另一类真 bug（凭空 pad 出的假样本行），必须继续被丢弃而不是硬对齐。"""
        m = _no_dist_metric(GRPOMetric)
        labels = torch.tensor([[-100, 10, 11, 12]])
        logps = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=torch.zeros(3, 4))
        result = m.calculate()
        assert 'train/approx_kl' not in result

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

    def test_embedding_metric_reports_similarity_values(self):
        # Two anchor-positive pairs on orthogonal axes: every positive pair scores 1, every
        # cross-pair (in-batch negative) scores 0.
        m = _no_dist_metric(EmbeddingMetric)
        labels = torch.tensor([1, 0, 1, 0])
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        m.accumulate({'labels': labels}, {'embeddings': embeddings})
        result = m.calculate()
        assert result['pos_sim'] == '1.0000'
        assert result['pos_sim_min'] == '1.0000'
        assert result['pos_sim_max'] == '1.0000'
        assert result['neg_sim'] == '0.0000'

    def test_embedding_metric_pools_the_first_token_of_3d_outputs(self):
        # The pair shares only its CLS token; the tail tokens point in opposite directions, so
        # mean pooling would report -0.8 and last-token pooling -1.0 instead of 1.0.
        m = _no_dist_metric(EmbeddingMetric)
        hidden = torch.zeros(2, 4, 2)
        hidden[:, 0] = torch.tensor([1.0, 0.0])
        hidden[0, 1:] = torch.tensor([0.0, 1.0])
        hidden[1, 1:] = torch.tensor([0.0, -1.0])
        m.accumulate({'labels': torch.tensor([1, 0])}, {'logits': hidden})
        assert m.calculate()['pos_sim'] == '1.0000'

    def test_embedding_metric_reports_grad_norm_and_plain_float_loss(self):
        m = _no_dist_metric(EmbeddingMetric)
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        m.accumulate(
            {'labels': torch.tensor([1, 0])}, {'embeddings': embeddings, 'loss': 2.0}, grad_norm=0.125)
        result = m.calculate()
        assert result['loss'] == '2.0000'
        assert result['grad_norm'] == '0.125000'

    def test_embedding_metric_accepts_a_list_of_inputs(self):
        m = _no_dist_metric(EmbeddingMetric)
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        m.accumulate(
            [{'labels': torch.tensor([1])}, {'labels': torch.tensor([0])}], {'embeddings': embeddings})
        assert m.calculate()['pos_sim'] == '1.0000'


# ---------------------------------------------------------------------------
# ExactMatch / RougeBleu (text-level, for evaluation runs)
# ---------------------------------------------------------------------------

class TestExactMatch:

    def test_all_correct(self):
        m = _no_dist_metric(ExactMatch)
        m.accumulate(predictions=['a', 'b'], references=['a', 'b'])
        assert m.calculate() == {'acc': 1.0}

    def test_partial(self):
        m = _no_dist_metric(ExactMatch)
        m.accumulate(predictions=['a', 'b'], references=['a', 'c'])
        assert m.calculate() == {'acc': 0.5}

    def test_is_exact_not_fuzzy(self):
        # Whitespace and case differences count as wrong -- this is string equality, not normalized match.
        m = _no_dist_metric(ExactMatch)
        m.accumulate(predictions=['A', 'b '], references=['a', 'b'])
        assert m.calculate() == {'acc': 0.0}

    def test_accumulate_is_additive(self):
        m = _no_dist_metric(ExactMatch)
        m.accumulate(predictions=['a'], references=['a'])
        m.accumulate(predictions=['b'], references=['c'])
        assert m.calculate() == {'acc': 0.5}

    def test_nothing_accumulated(self):
        assert _no_dist_metric(ExactMatch).calculate() == {}

    def test_accumulate_without_pairs_is_ignored(self):
        m = _no_dist_metric(ExactMatch)
        m.accumulate()
        m.accumulate(predictions=['a'])  # references missing
        assert m.calculate() == {}

    def test_calculate_resets(self):
        m = _no_dist_metric(ExactMatch)
        m.accumulate(predictions=['a'], references=['a'])
        assert m.calculate() == {'acc': 1.0}
        assert m.calculate() == {}

    def test_mismatched_lengths_raise(self):
        m = _no_dist_metric(ExactMatch)
        with pytest.raises(AssertionError, match='parallel'):
            m.accumulate(predictions=['a', 'b'], references=['a'])


class TestRougeBleu:

    def test_identical_text_scores_full_rouge(self):
        m = _no_dist_metric(RougeBleu)
        m.accumulate(predictions=['今天天气很好'], references=['今天天气很好'])
        result = m.calculate()
        assert set(result) == set(RougeBleu.keys)
        assert result['rouge-1'] == 100.0
        assert result['rouge-l'] == 100.0

    def test_disjoint_text_scores_zero(self):
        m = _no_dist_metric(RougeBleu)
        m.accumulate(predictions=['苹果'], references=['火车'])
        result = m.calculate()
        assert result['rouge-1'] == 0.0

    def test_reports_percentages(self):
        # One exact match and one disjoint pair average to 50, reported as a percentage.
        m = _no_dist_metric(RougeBleu)
        m.accumulate(predictions=['苹果', '火车'], references=['苹果', '飞机'])
        assert m.calculate()['rouge-1'] == pytest.approx(50.0)

    def test_empty_tokenization_is_skipped_not_scored_zero(self):
        m = _no_dist_metric(RougeBleu)
        m.accumulate(predictions=['今天天气很好', '  '], references=['今天天气很好', '火车'])
        # The blank prediction is dropped, so the one real pair still scores full marks.
        assert m.calculate()['rouge-1'] == 100.0

    def test_nothing_accumulated(self):
        assert _no_dist_metric(RougeBleu).calculate() == {}


class TestTextMetricBase:

    def _metric(self):
        from twinkle.metric.generation import TextMetric
        return TextMetric()

    def test_accumulate_appends_parallel_pairs(self):
        metric = self._metric()
        metric.accumulate(predictions=['a', 'b'], references=['a', 'c'])
        assert metric.predictions == ['a', 'b']
        assert metric.references == ['a', 'c']

    def test_accumulate_without_pairs_is_a_no_op(self):
        metric = self._metric()
        metric.accumulate(inputs=[1], outputs=[2])
        metric.accumulate(predictions=None, references=None)
        assert metric.predictions == []
        assert metric.references == []

    def test_mismatched_lengths_are_rejected(self):
        import pytest
        metric = self._metric()
        with pytest.raises(AssertionError, match='parallel'):
            metric.accumulate(predictions=['a'], references=['a', 'b'])

    def test_reset_clears_accumulated_pairs(self):
        metric = self._metric()
        metric.accumulate(predictions=['a'], references=['b'])
        metric.reset()
        assert metric.predictions == []
        assert metric.references == []


class TestExactMatchMetric:

    def test_calculate_reports_mean_accuracy(self):
        from twinkle.metric.generation import ExactMatch

        metric = ExactMatch()
        metric.accumulate(predictions=['a', 'b'], references=['a', 'c'])
        result = metric.calculate()
        assert result == {'acc': 0.5}

    def test_calculate_without_pairs_returns_empty(self):
        from twinkle.metric.generation import ExactMatch

        assert ExactMatch().calculate() == {}


class TestReportingHelpers:

    def test_finite_number_accepts_scalars_and_tensor_strings(self):
        from twinkle.metric.reporting import _finite_number

        assert _finite_number(3) == 3
        assert _finite_number(True) == 1
        assert _finite_number(2.5) == 2.5
        assert _finite_number('  tensor(1.5)  ') == 1.5
        assert _finite_number('nan') is None
        assert _finite_number(float('inf')) is None
        assert _finite_number('not a number') is None
        assert _finite_number([1]) is None

    def test_finite_number_reads_tensor_values(self):
        from twinkle.metric.reporting import _finite_number

        assert _finite_number(torch.tensor(2.5)) == 2.5
        assert _finite_number(torch.tensor(float('nan'))) is None

    def test_safe_name_sanitizes_and_never_returns_empty(self):
        from twinkle.metric.reporting import _safe_name

        assert _safe_name('loss/epoch:1') == 'loss_epoch_1'
        assert _safe_name('///') == 'default'
        assert _safe_name('keep-this.v2') == 'keep-this.v2'
