"""Batch-dependent objectives must fail before a micro-batched loss is evaluated."""
import sys
from inspect import unwrap
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from twinkle.loss import CrossEntropyLoss, GRPOLoss
from twinkle.loss.clip_cov import ClipCovLoss
from twinkle.loss.grpo import BNPOLoss, CISPOLoss, PPOLoss
from twinkle.loss.infonce import InfonceLoss, OnlineContrastiveLoss
from twinkle.loss.kl_cov import KLCovLoss
from twinkle.model.optimizer_group import BaseOptimizerGroup
from twinkle.processor import InputProcessor
from .test_grpo_micro_batch import TrainingHarness, data

BATCH_DEPENDENT_LOSSES = [
    pytest.param(InfonceLoss(), id='infonce-batch-negatives'),
    pytest.param(OnlineContrastiveLoss(), id='online-contrastive'),
    pytest.param(ClipCovLoss(), id='clip-cov'),
    pytest.param(KLCovLoss(), id='kl-cov'),
    pytest.param(GRPOLoss(top_entropy_quantile=.5), id='grpo-entropy-filter'),
    pytest.param(PPOLoss(top_entropy_quantile=.5), id='ppo-entropy-filter'),
    pytest.param(CISPOLoss(top_entropy_quantile=.5), id='cispo-entropy-filter'),
    pytest.param(BNPOLoss(top_entropy_quantile=.5, token_mean_scope='global'), id='bnpo-entropy-filter'),
]


@pytest.mark.parametrize('loss', BATCH_DEPENDENT_LOSSES)
@pytest.mark.parametrize('algorithm', [None, 'ffd', 'kk'])
def test_transformers_rejects_batch_dependent_split_before_forward(loss, algorithm):
    model = TrainingHarness(loss)
    options = {'micro_batch_size': 2}
    if algorithm is not None:
        options.update(dynamic_batching=True, max_tokens_per_micro_batch=18, packing_algorithm=algorithm)

    with pytest.raises(NotImplementedError, match=f'{type(loss).__name__}.*full local batch') as error:
        model.run(data(), **options)

    assert 'would be split into' in str(error.value)
    assert model.calls == []
    assert model.weight.grad is None
    assert model.optimizer_group['default'].cur_step == 0


@pytest.mark.parametrize('loss', BATCH_DEPENDENT_LOSSES)
@pytest.mark.parametrize('algorithm', [None, 'ffd', 'kk'])
def test_transformers_allows_batch_dependent_loss_when_plan_does_not_split(loss, algorithm):
    model = TrainingHarness(loss)
    model._forward_backward_micro_batch = Mock(return_value={'loss': 0.})
    rows = data()
    options = {'micro_batch_size': len(rows)}
    if algorithm is not None:
        options.update(dynamic_batching=True, max_tokens_per_micro_batch=100, packing_algorithm=algorithm)

    model.run(rows, **options)

    model._forward_backward_micro_batch.assert_called_once()
    assert len(model._forward_backward_micro_batch.call_args.kwargs['inputs']) == len(rows)


@pytest.mark.parametrize('loss', BATCH_DEPENDENT_LOSSES)
@pytest.mark.parametrize('options', [
    pytest.param({}, id='unsplit'),
    pytest.param({'micro_batch_size': 4}, id='fixed-short-batch'),
    pytest.param({'micro_batch_size': 4, 'dynamic_batching': True,
                  'max_tokens_per_micro_batch': 100, 'packing_algorithm': 'ffd'}, id='dynamic-ffd'),
    pytest.param({'micro_batch_size': 4, 'dynamic_batching': True,
                  'max_tokens_per_micro_batch': 100, 'packing_algorithm': 'kk'}, id='dynamic-kk'),
])
def test_transformers_allows_one_actual_micro_batch_despite_configured_ga(loss, options):
    model = TrainingHarness(loss)
    model.optimizer_group['default'].gradient_accumulation_steps = 4
    model._forward_backward_micro_batch = Mock(return_value={'loss': 0.})
    rows = data()[:2]

    model.run(rows, **options)

    model._forward_backward_micro_batch.assert_called_once()
    actual = model._forward_backward_micro_batch.call_args.kwargs['inputs']
    assert sorted(actual, key=lambda row: row['input_ids'][0]) == rows


@pytest.mark.parametrize('loss', [CrossEntropyLoss(), GRPOLoss(), InfonceLoss(use_batch=False)])
def test_losses_without_batch_context_still_allow_splitting(loss):
    loss.validate_micro_batching(4)


@pytest.fixture
def megatron_model(monkeypatch):
    from twinkle.model.megatron.megatron import MegatronModel

    # MCore is optional on CPU; no scheduler/model forward may run before the guard.
    core = ModuleType('megatron.core')
    core.parallel_state = SimpleNamespace()
    pipeline = ModuleType('megatron.core.pipeline_parallel')
    pipeline.get_forward_backward_func = Mock(side_effect=AssertionError('scheduler must not run'))
    monkeypatch.setitem(sys.modules, 'megatron.core', core)
    monkeypatch.setitem(sys.modules, 'megatron.core.pipeline_parallel', pipeline)
    processor = InputProcessor(framework='megatron')
    processor.process_pipeline = [processor.prepare_inputs, processor.collate_fn]
    group = BaseOptimizerGroup(loss_instance=ClipCovLoss())
    group.processor = processor
    chunk = SimpleNamespace(config=SimpleNamespace())
    model = SimpleNamespace(
        _lazy_wrap_model=lambda: None,
        _lazy_finish_param_config=lambda: None,
        _get_default_group=lambda: 'default',
        _not_encoded=lambda _: False,
        optimizer_group={'default': group},
        strategy=SimpleNamespace(unwrap_model=lambda _: [chunk]),
        model=[chunk],
        variable_seq_lengths=False,
        device_mesh=SimpleNamespace(cp_world_size=1),
        _router_replay_enabled=False,
        _maybe_apply_router_replay=lambda: None,
    )
    return model, unwrap(MegatronModel.forward_backward)


@pytest.mark.parametrize('forward_only', [False, True])
def test_megatron_rejects_batch_dependent_training_and_eval_split(megatron_model, forward_only):
    model, forward_backward = megatron_model
    rows = [{'input_ids': [1, 2], 'labels': [1, 2]} for _ in range(4)]

    with pytest.raises(NotImplementedError, match='ClipCovLoss.*full local batch'):
        forward_backward(model, inputs=rows, micro_batch_size=2, forward_only=forward_only)


@pytest.mark.parametrize('forward_only', [False, True])
@pytest.mark.parametrize('micro_batch_size', [None, 2, 4])
def test_megatron_allows_one_actual_micro_batch(megatron_model, monkeypatch, forward_only, micro_batch_size):
    from twinkle.model.megatron import megatron

    class ScheduleSetupReached(Exception):
        pass

    # Stop after the guard, before reaching the optional MCore scheduler or GPU operations.
    setup = Mock(side_effect=ScheduleSetupReached)
    monkeypatch.setattr(megatron, 'RouterReplay', setup)
    model, forward_backward = megatron_model
    model.optimizer_group['default'].gradient_accumulation_steps = 4
    rows = [{'input_ids': [1, 2], 'labels': [1, 2]} for _ in range(2)]

    with pytest.raises(ScheduleSetupReached):
        forward_backward(model, inputs=rows, micro_batch_size=micro_batch_size, forward_only=forward_only)

    setup.assert_called_once()


def test_batch_dependent_error_reports_actual_count_in_english():
    with pytest.raises(NotImplementedError, match='split into 3 micro-batches') as error:
        ClipCovLoss().validate_micro_batching(3)

    assert str(error.value).isascii()
