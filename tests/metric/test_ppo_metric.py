# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the two PPO metrics that test_metrics.py leaves uncovered:

* ``PPOValueMetric`` -- critic statistics aggregated over valid response tokens.
* ``PPOMetric`` -- the policy panel; a pure rename subclass of ``GRPOMetric``.

Each case drives real tensors through ``accumulate`` -> ``calculate`` and asserts the
reported numbers against hand-computed values, so the mask alignment (``ignore_index``
prompt positions dropped), the clipping flag, explained variance and the multi
micro-batch cursor are all pinned by behaviour rather than by inspection.
"""
import pytest
import torch

from twinkle.metric import GRPOMetric, PPOMetric, PPOValueMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_dist_metric(cls, **kwargs):
    """Instantiate a metric without distributed groups (gather_results returns local)."""
    return cls(device_mesh=None, process_group=None, **kwargs)


# ---------------------------------------------------------------------------
# PPOValueMetric
# ---------------------------------------------------------------------------

class TestPPOValueMetric:

    def test_perfect_prediction_explained_variance_one(self):
        # values == returns on the two response tokens -> residual variance 0 -> ev 1.
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[0., 0., 1., 3.]])
        returns = torch.tensor([[0., 0., 1., 3.]])
        old = torch.tensor([[0., 0., 1., 3.]])
        m.accumulate({'labels': labels}, {'values': values}, old_values=old, returns=returns)
        result = m.calculate()
        assert result['train/value_mean'] == pytest.approx(2.0)
        assert result['train/return_mean'] == pytest.approx(2.0)
        assert result['train/value_clip_ratio'] == pytest.approx(0.0)
        assert result['train/explained_variance'] == pytest.approx(1.0)
        # No advantages passed -> advantage stats must be absent, not zero-filled.
        assert 'train/advantage_mean' not in result
        assert 'train/advantage_std' not in result

    def test_explained_variance_known_value(self):
        # returns masked [1, 3] -> var 1.0; values masked [1, 1] -> residual [0, 2] var 1.0
        # ev = 1 - 1.0 / 1.0 = 0.0. value_mean 1.0, return_mean 2.0.
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[9., 9., 1., 1.]])
        returns = torch.tensor([[7., 7., 1., 3.]])
        old = torch.tensor([[0., 0., 1., 3.]])
        m.accumulate({'labels': labels}, {'values': values}, old_values=old, returns=returns)
        result = m.calculate()
        assert result['train/value_mean'] == pytest.approx(1.0)
        assert result['train/return_mean'] == pytest.approx(2.0)
        assert result['train/explained_variance'] == pytest.approx(0.0)

    def test_records_keep_only_mask_positions(self):
        # Prompt-position values (9.0 / 7.0) differ wildly from response positions; if the
        # ignore_index mask were ignored they would leak into the stored records.
        m = _no_dist_metric(PPOValueMetric, epsilon=0.2)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[9., 9., 1., 3.]])
        returns = torch.tensor([[7., 7., 1., 3.]])
        old = torch.zeros(1, 4)  # diff at mask = [1, 3], both > epsilon -> all clipped
        m.accumulate({'labels': labels}, {'values': values}, old_values=old, returns=returns)
        assert len(m.records) == 1
        record = m.records[0]
        assert record['values'] == pytest.approx([1., 3.])
        assert record['returns'] == pytest.approx([1., 3.])
        assert record['clipped'] == pytest.approx([1., 1.])
        # clipped = old + (values - old).clamp(-eps, eps) = 0 + [0.2, 0.2]
        assert record['clipped_values'] == pytest.approx([0.2, 0.2])
        assert record['advantages'] == []

    def test_clip_ratio_full_when_all_exceed_epsilon(self):
        m = _no_dist_metric(PPOValueMetric, epsilon=0.2)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[0., 0., 1., 3.]])
        returns = torch.tensor([[0., 0., 1., 3.]])
        old = torch.zeros(1, 4)
        m.accumulate({'labels': labels}, {'values': values}, old_values=old, returns=returns)
        assert m.calculate()['train/value_clip_ratio'] == pytest.approx(1.0)

    def test_advantage_stats(self):
        # advantages masked [2, 4] -> mean 3.0, population std 1.0.
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[0., 0., 1., 3.]])
        returns = torch.tensor([[0., 0., 1., 3.]])
        old = torch.tensor([[0., 0., 1., 3.]])
        advantages = torch.tensor([[0., 0., 2., 4.]])
        m.accumulate(
            {'labels': labels}, {'values': values},
            old_values=old, returns=returns, advantages=advantages)
        result = m.calculate()
        assert result['train/advantage_mean'] == pytest.approx(3.0)
        assert result['train/advantage_std'] == pytest.approx(1.0)

    def test_multiple_micro_batches_advance_cursor(self):
        # Two micro batches, one sample each. old_values / returns are a single [2, seq]
        # tensor sliced by the running cursor; the two value tensors arrive as a list.
        m = _no_dist_metric(PPOValueMetric)
        labels = [{'labels': torch.tensor([[-100, 10, 11]])}, {'labels': torch.tensor([[-100, 12, 13]])}]
        values = [torch.tensor([[0., 1., 3.]]), torch.tensor([[0., 2., 4.]])]
        old = torch.zeros(2, 3)
        returns = torch.tensor([[0., 1., 3.], [0., 2., 4.]])
        m.accumulate(labels, {'values': values}, old_values=old, returns=returns)
        assert len(m.records) == 2
        result = m.calculate()
        # all masked values [1, 3, 2, 4] -> mean 2.5; values == returns -> ev 1.
        assert result['train/value_mean'] == pytest.approx(2.5)
        assert result['train/return_mean'] == pytest.approx(2.5)
        assert result['train/explained_variance'] == pytest.approx(1.0)

    def test_no_valid_tokens_is_skipped(self):
        # All prompt -> mask.any() is False -> no record, empty report.
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, -100, -100]])
        values = torch.zeros(1, 4)
        old = torch.zeros(1, 4)
        returns = torch.zeros(1, 4)
        m.accumulate({'labels': labels}, {'values': values}, old_values=old, returns=returns)
        assert m.records == []
        assert m.calculate() == {}

    @pytest.mark.parametrize('outputs, old, returns', [
        (None, torch.zeros(1, 4), torch.zeros(1, 4)),  # outputs is None
        ({}, torch.zeros(1, 4), torch.zeros(1, 4)),  # no 'values' key
        ({'values': None}, torch.zeros(1, 4), torch.zeros(1, 4)),  # values is None
        ({'values': torch.zeros(1, 4)}, None, torch.zeros(1, 4)),  # old_values None
        ({'values': torch.zeros(1, 4)}, torch.zeros(1, 4), None),  # returns None
    ])
    def test_missing_operand_early_returns(self, outputs, old, returns):
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        m.accumulate({'labels': labels}, outputs, old_values=old, returns=returns)
        assert m.records == []
        assert m.calculate() == {}

    def test_empty_metric_returns_empty(self):
        assert _no_dist_metric(PPOValueMetric).calculate() == {}

    def test_calculate_resets_records(self):
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[0., 0., 1., 3.]])
        m.accumulate({'labels': labels}, {'values': values}, old_values=values, returns=values)
        assert m.calculate() != {}
        assert m.records == []
        assert m.calculate() == {}

    def test_reset_clears_records(self):
        m = _no_dist_metric(PPOValueMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        values = torch.tensor([[0., 0., 1., 3.]])
        m.accumulate({'labels': labels}, {'values': values}, old_values=values, returns=values)
        assert len(m.records) == 1
        m.reset()
        assert m.records == []


# ---------------------------------------------------------------------------
# PPOMetric (policy panel; pure rename subclass of GRPOMetric)
# ---------------------------------------------------------------------------

class TestPPOMetric:

    def test_is_grpo_subclass(self):
        assert issubclass(PPOMetric, GRPOMetric)

    def test_basic_policy_metric(self):
        m = _no_dist_metric(PPOMetric)
        labels = torch.tensor([[10, 11, -100, -100]])
        logps = torch.randn(1, 4)
        m.accumulate({'labels': labels}, {'logps': logps})
        result = m.calculate()
        assert 'train/policy_confidence' in result
        assert 'train/mean_new_logp' in result

    def test_matches_grpo_on_identical_input(self):
        # PPOMetric overrides nothing, so for identical input it must report exactly the
        # same panel as GRPOMetric -- this is the contract that justifies the rename.
        labels = torch.tensor([[-100, -100, 10, 11]])
        torch.manual_seed(0)
        logps = torch.randn(1, 4)
        ppo = _no_dist_metric(PPOMetric)
        grpo = _no_dist_metric(GRPOMetric)
        ppo.accumulate({'labels': labels}, {'logps': logps})
        grpo.accumulate({'labels': labels}, {'logps': logps})
        assert ppo.calculate() == pytest.approx(grpo.calculate())

    def test_with_old_logps_reports_kl(self):
        m = _no_dist_metric(PPOMetric)
        labels = torch.tensor([[-100, -100, 10, 11]])
        logps = torch.tensor([[0., 0., -1.0, -2.0]])
        old_logps = torch.tensor([[0., 0., -1.0, -2.0]])
        m.accumulate({'labels': labels}, {'logps': logps}, old_logps=old_logps)
        result = m.calculate()
        assert 'train/approx_kl' in result
        assert 'train/logp_diff_mean' in result
        # new == old on the response tokens -> zero drift.
        assert result['train/logp_diff_mean'] == pytest.approx(0.0)

    def test_reset(self):
        m = _no_dist_metric(PPOMetric)
        labels = torch.tensor([[10, 11, -100, -100]])
        m.accumulate({'labels': labels}, {'logps': torch.randn(1, 4)})
        m.reset()
        assert m.n_tokens == 0
        assert m.sum_new == 0.0
