"""Batch-dependent objectives use the statistics of each physical micro-batch."""
import sys
from inspect import unwrap
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import GRPOLoss
from twinkle.loss.clip_cov import ClipCovLoss
from twinkle.loss.grpo import BNPOLoss, CISPOLoss, PPOLoss
from twinkle.loss.infonce import EmbeddingLoss, InfonceLoss, OnlineContrastiveLoss
from twinkle.loss.kl_cov import KLCovLoss
from twinkle.model.optimizer_group import BaseOptimizerGroup
from twinkle.processor import InputProcessor
from .test_grpo_micro_batch import TrainingHarness, data

BATCH_CONTEXT_LOSS_FACTORIES = [
    pytest.param(InfonceLoss, id='infonce-batch-negatives'),
    pytest.param(OnlineContrastiveLoss, id='online-contrastive'),
    pytest.param(lambda: ClipCovLoss(clip_cov_ratio=.5, clip_cov_lb=.001, clip_cov_ub=.5), id='clip-cov'),
    pytest.param(lambda: KLCovLoss(kl_cov_ratio=.5), id='kl-cov'),
    pytest.param(lambda: GRPOLoss(top_entropy_quantile=.5), id='grpo-entropy-filter'),
    pytest.param(lambda: PPOLoss(top_entropy_quantile=.5), id='ppo-entropy-filter'),
    pytest.param(lambda: CISPOLoss(top_entropy_quantile=.5), id='cispo-entropy-filter'),
    pytest.param(lambda: BNPOLoss(top_entropy_quantile=.5, token_mean_scope='global'), id='bnpo-entropy-filter'),
]


class BatchContextHarness(TrainingHarness):
    def forward(self, *, inputs, **kwargs):
        super().forward(inputs=inputs, **kwargs)
        group = self.optimizer_group['default']
        status = group.train_status
        status.outputs['entropies'] = 1. + status.outputs['logps'].square()
        if isinstance(group.loss_instance, EmbeddingLoss):
            embeddings = []
            infonce = isinstance(group.loss_instance, InfonceLoss)
            for row in inputs:
                index = row['input_ids'][0] + 1
                # Each input row is a complete query/candidate group or labelled pair.
                vectors = torch.tensor([[1., .1 * index, .2], [.1 * index, 1., -.3], [-.4, .3, .1 * index]],
                                       dtype=self.weight.dtype)
                if not infonce:
                    vectors = vectors[:2]
                embeddings.append(F.normalize(vectors + self.weight * vectors.roll(1, dims=-1), dim=-1))
            status.outputs['embeddings'] = torch.cat(embeddings)
            status.inputs['labels'] = torch.tensor(
                [1, 0, 0] * len(inputs) if infonce else [row['input_ids'][0] % 2 for row in inputs])
        return dict(status.outputs)


@pytest.mark.parametrize('factory', BATCH_CONTEXT_LOSS_FACTORIES)
@pytest.mark.parametrize('options', [
    pytest.param({}, id='unsplit'),
    pytest.param({'micro_batch_size': 4}, id='fixed'),
    pytest.param({'micro_batch_size': 6}, id='fixed-short-tail'),
    pytest.param({'micro_batch_size': 4, 'dynamic_batching': True,
                  'max_tokens_per_micro_batch': 18, 'packing_algorithm': 'ffd'}, id='dynamic-ffd'),
    pytest.param({'micro_batch_size': 4, 'dynamic_batching': True,
                  'max_tokens_per_micro_batch': 18, 'packing_algorithm': 'kk'}, id='dynamic-kk'),
])
def test_transformers_batch_context_loss_matches_independent_micro_batch_computation(factory, options):
    rows = data()
    model, oracle = BatchContextHarness(factory()), BatchContextHarness(factory())
    kwargs = {}
    if isinstance(model.optimizer_group['default'].loss_instance, GRPOLoss):
        kwargs = {'advantages': [1., -.5, .7, -1.2, .4, -.3, .9, -.8],
                  'old_logps': [[0.] * len(row['labels']) for row in rows]}

    torch.manual_seed(7)
    actual = model.run(rows, **options, **kwargs)

    if options:
        assert len(model.calls) > 1
    else:
        assert len(model.calls) == 1
    assert sorted(index for indices, _ in model.calls for index in indices) == list(range(len(rows)))
    assert [sync for _, sync in model.calls] == [False] * (len(model.calls) - 1) + [True]
    assert model.optimizer_group['default'].cur_step == 1

    # Replay the same groups directly through the loss: batch statistics must be
    # recomputed for each group, rather than copied from the full update window.
    torch.manual_seed(7)
    terms = []
    loss = oracle.optimizer_group['default'].loss_instance
    for indices, _ in model.calls:
        outputs = oracle.forward(inputs=[rows[index] for index in indices])
        inputs = oracle.optimizer_group['default'].train_status.inputs
        micro_kwargs = {key: [value[index] for index in indices] for key, value in kwargs.items()}
        result = loss(inputs, outputs, **micro_kwargs)
        scale = loss.micro_batch_scale(rows, indices)
        terms.append(result['loss'] * scale)
    expected = torch.stack(terms).sum()
    expected.backward()

    assert actual['loss'] == pytest.approx(expected.item())
    assert torch.isfinite(model.weight.grad)
    torch.testing.assert_close(model.weight.grad, oracle.weight.grad)


@pytest.fixture
def megatron_model(monkeypatch):
    from twinkle.model.megatron.megatron import MegatronModel

    # MCore is optional on CPU; stop after input batching and before scheduler setup.
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


@pytest.mark.parametrize('factory', BATCH_CONTEXT_LOSS_FACTORIES)
@pytest.mark.parametrize('forward_only', [False, True])
@pytest.mark.parametrize('input_count', [2, 4])
def test_megatron_batch_context_losses_reach_schedule_setup(
    megatron_model, monkeypatch, factory, forward_only, input_count,
):
    from twinkle.model.megatron import megatron

    class ScheduleSetupReached(Exception):
        pass

    # Reaching this point proves both one and multiple micro-batches are accepted.
    setup = Mock(side_effect=ScheduleSetupReached)
    monkeypatch.setattr(megatron, 'RouterReplay', setup)
    model, forward_backward = megatron_model
    model.optimizer_group['default'].loss_instance = factory()
    model.optimizer_group['default'].gradient_accumulation_steps = 4
    rows = [{'input_ids': [1, 2], 'labels': [1, 2]} for _ in range(input_count)]

    with pytest.raises(ScheduleSetupReached):
        forward_backward(model, inputs=rows, micro_batch_size=2, forward_only=forward_only)

    setup.assert_called_once()
