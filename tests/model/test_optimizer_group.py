# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for ``twinkle.model.optimizer_group`` (BaseOptimizerGroup / TrainStatus).

``BaseOptimizerGroup`` is the shared per-adapter training-step state that both the
transformers and megatron backends build on. Its own logic had no direct test -- the
existing suites only pass stub objects shaped like it. These cases drive real ``Metric``
and ``Loss`` objects through the real class on CPU (no model, no GPU, no distributed):

* ``do_grad_sync`` gradient-accumulation gating and its state side effect,
* loss assignment / selection and numerical transparency,
* ``accumulate_metrics`` / ``calculate_metrics`` against a real ``LossMetric``.
"""
import pytest
import torch

from twinkle.loss import CrossEntropyLoss
from twinkle.loss.base import Loss
from twinkle.metric import LossMetric
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus


# ---------------------------------------------------------------------------
# TrainStatus
# ---------------------------------------------------------------------------

class TestTrainStatus:

    def test_defaults(self):
        status = TrainStatus()
        assert status.inputs is None
        assert status.outputs is None
        assert status.loss_value is None
        assert status.num_tokens == 0
        assert status.metrics == []
        assert status.forward_kwargs == {}

    def test_mutable_defaults_are_per_instance(self):
        # default_factory must give each status its own list/dict, not a shared one.
        a = TrainStatus()
        b = TrainStatus()
        a.metrics.append('x')
        a.forward_kwargs['k'] = 1
        assert b.metrics == []
        assert b.forward_kwargs == {}


# ---------------------------------------------------------------------------
# do_grad_sync
# ---------------------------------------------------------------------------

class TestDoGradSync:

    def test_single_step_always_syncs(self):
        group = BaseOptimizerGroup(gradient_accumulation_steps=1)
        for step in (1, 2, 3, 7):
            group.cur_step = step
            assert group.do_grad_sync() is True

    def test_first_step_never_syncs_while_accumulating(self):
        # cur_step == 1 satisfies (cur_step-1) % gas == 0 but is excluded by cur_step > 1,
        # so the very first micro step of a run does not trigger a sync.
        group = BaseOptimizerGroup(gradient_accumulation_steps=4, cur_step=1)
        assert group.do_grad_sync() is False

    def test_mid_window_steps_do_not_sync(self):
        group = BaseOptimizerGroup(gradient_accumulation_steps=4)
        for step in (2, 3, 4):
            group.cur_step = step
            assert group.do_grad_sync() is False

    def test_syncs_at_accumulation_boundary(self):
        group = BaseOptimizerGroup(gradient_accumulation_steps=4)
        for step in (5, 9, 13):  # (step-1) % 4 == 0 and step > 1
            group.cur_step = step
            assert group.do_grad_sync() is True

    def test_explicit_argument_updates_state(self):
        group = BaseOptimizerGroup(gradient_accumulation_steps=1)
        group.cur_step = 5
        # Passing gas explicitly both mutates the stored value and gates on it.
        assert group.do_grad_sync(4) is True
        assert group.gradient_accumulation_steps == 4

    def test_none_argument_uses_state_without_mutation(self):
        group = BaseOptimizerGroup(gradient_accumulation_steps=3)
        group.cur_step = 4  # (4-1) % 3 == 0 and > 1 -> sync
        assert group.do_grad_sync(None) is True
        assert group.gradient_accumulation_steps == 3


# ---------------------------------------------------------------------------
# Loss assignment / selection
# ---------------------------------------------------------------------------

class TestLossSelection:

    def test_assigned_loss_is_selected_without_changing_identity(self):
        group = BaseOptimizerGroup()
        raw = CrossEntropyLoss(reduction='sum')
        group.loss_instance = raw
        assert group.get_loss() is raw
        assert isinstance(group.get_loss(), Loss)

    def test_reassignment_is_idempotent(self):
        group = BaseOptimizerGroup()
        raw = CrossEntropyLoss(reduction='sum')
        group.loss_instance = raw
        group.loss_instance = group.get_loss()
        assert group.get_loss() is raw

    def test_assigning_none_clears_default_loss(self):
        group = BaseOptimizerGroup()
        group.loss_instance = CrossEntropyLoss(reduction='sum')
        group.loss_instance = None
        assert group.get_loss() is None

    def test_selected_loss_keeps_numeric_contract_and_propagates_errors(self):
        group = BaseOptimizerGroup()
        group.loss_instance = CrossEntropyLoss(reduction='sum')
        inputs = {'labels': torch.tensor([[1, 2, -100]])}
        outputs = {'logps': torch.tensor([[-1.0, -2.0, -3.0]])}
        selected = group.get_loss()(inputs, outputs)
        raw = CrossEntropyLoss(reduction='sum')(inputs, outputs)
        assert selected['loss'].item() == pytest.approx(raw['loss'].item())
        with pytest.raises(KeyError):
            group.get_loss()(inputs, {})


# ---------------------------------------------------------------------------
# accumulate_metrics / calculate_metrics
# ---------------------------------------------------------------------------

def _loss_status():
    """A TrainStatus holding a real LossMetric plus a minimal loss-bearing step."""
    metric = LossMetric(device_mesh=None, process_group=None)
    status = TrainStatus(
        inputs={'labels': torch.tensor([1])},
        outputs={'loss': torch.tensor(2.5), 'num_tokens': torch.tensor(4.0)},
        metrics=[metric],
    )
    return status, metric


class TestMetricFlow:

    def test_get_lr_default_empty(self):
        assert BaseOptimizerGroup()._get_lr() == []

    def test_accumulate_metrics_drives_train_metric(self):
        group = BaseOptimizerGroup()
        status, metric = _loss_status()
        group.train_status = status
        group.cur_step = 3
        group.accumulate_metrics(is_training=True)
        assert metric.total_count == 1
        assert metric.total_loss == pytest.approx(2.5)

    def test_accumulate_metrics_uses_eval_status_when_not_training(self):
        group = BaseOptimizerGroup()
        train_status, train_metric = _loss_status()
        eval_status, eval_metric = _loss_status()
        group.train_status = train_status
        group.eval_status = eval_status
        group.accumulate_metrics(is_training=False)
        assert eval_metric.total_count == 1
        assert train_metric.total_count == 0

    def test_accumulate_metrics_skips_without_io(self):
        group = BaseOptimizerGroup()
        status, metric = _loss_status()
        status.outputs = None  # forward produced nothing to score yet
        group.train_status = status
        group.accumulate_metrics(is_training=True)
        assert metric.total_count == 0

    def test_accumulate_metrics_no_metrics_is_noop(self):
        group = BaseOptimizerGroup()
        group.train_status = TrainStatus(inputs={'labels': torch.tensor([1])}, outputs={'loss': torch.tensor(1.0)})
        group.accumulate_metrics(is_training=True)  # must not raise on empty metrics

    def test_calculate_metrics_returns_results_and_clears_io(self):
        group = BaseOptimizerGroup()
        status, metric = _loss_status()
        group.train_status = status
        group.cur_step = 2
        result = group.calculate_metrics(is_training=True)
        assert 'loss' in result
        # calculate_metrics consumes the step: inputs/outputs are dropped afterwards.
        assert group.train_status.inputs is None
        assert group.train_status.outputs is None

    def test_calculate_metrics_passes_grad_norm(self):
        group = BaseOptimizerGroup()
        status, metric = _loss_status()
        group.train_status = status
        group._last_grad_norm = 0.75
        result = group.calculate_metrics(is_training=True)
        # LossMetric formats grad_norm to 6 decimals as a string.
        assert result['grad_norm'] == '0.750000'
