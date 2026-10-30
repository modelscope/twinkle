"""Numerical checks for submitting a fixed GRPO update in one call."""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest
import torch

from twinkle.loss import CrossEntropyLoss, GRPOLoss
from twinkle.loss.grpo import BNPOLoss, CISPOLoss
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor


class TrainingHarness:
    _forward_backward_single = TransformersModel._forward_backward_single
    _forward_backward_micro_batch = TransformersModel._forward_backward_micro_batch
    _forward_backward_micro_batches = TransformersModel._forward_backward_micro_batches
    _build_micro_batch_plan = TransformersModel._build_micro_batch_plan
    _global_loss_normalizer = TransformersModel._global_loss_normalizer
    _not_encoded = staticmethod(lambda _: False)

    def __init__(self, loss):
        self.weight = torch.nn.Parameter(torch.tensor(.12, dtype=torch.float64))
        group = BaseOptimizerGroup(loss_instance=loss, train_status=TrainStatus())
        group.processor = InputProcessor(padding_free=False)
        group.scaler = None
        group._ensure_dp_group = lambda: None
        self.optimizer_group = {'default': group}
        self.device_mesh = None
        self.mixed_precision = 'bf16'
        self.sync_enabled = True
        self.calls = []
        self.model = SimpleNamespace(no_sync=self.no_sync, training=True)
        self.calculate_loss = MethodType(TransformersModel.calculate_loss.__wrapped__, self)
        self.backward = MethodType(TransformersModel.backward.__wrapped__, self)

    def _get_default_group(self):
        return 'default'

    @contextmanager
    def no_sync(self):
        previous = self.sync_enabled
        self.sync_enabled = False
        try:
            yield
        finally:
            self.sync_enabled = previous

    def token_logps(self, coefficients):
        return self.weight * coefficients

    def forward(self, *, inputs, **kwargs):
        self.calls.append(([row['input_ids'][0] for row in inputs], self.sync_enabled))
        width = max(len(row['labels']) for row in inputs)
        labels = torch.tensor([row['labels'] + [-100] * (width - len(row['labels'])) for row in inputs])
        coefficients = torch.tensor([row['coefficients'] + [0.] * (width - len(row['labels']))
                                     for row in inputs], dtype=torch.float64)
        status = self.optimizer_group['default'].train_status
        status.inputs = {'labels': labels}
        status.outputs = {'logps': self.token_logps(coefficients)}
        status.loss_value = 0.
        return dict(status.outputs)

    def run(self, inputs, **kwargs):
        return TransformersModel.forward_backward.__wrapped__(self, inputs=inputs, **kwargs)

    def normalized_gradient(self):
        normalizer = self.optimizer_group['default'].train_status.num_tokens
        return self.weight.grad / max(float(normalizer), 1e-30)


def data():
    return [{'input_ids': [index] * length, 'labels': [1] * length,
             'coefficients': [.1 * (index + 1) * (position + 1) for position in range(length)]}
            for index, length in enumerate([1, 2, 5, 7, 3, 9, 4, 6])]


@pytest.mark.parametrize('count', [8, 6])
@pytest.mark.parametrize('loss_factory', [GRPOLoss, CISPOLoss,
                                        lambda: BNPOLoss(token_mean_scope='micro'),
                                        lambda: BNPOLoss(token_mean_scope='global')])
def test_internal_fixed_micro_batches_follow_the_loss_reduction(count, loss_factory):
    rows = data()[:count]
    baseline, internal = TrainingHarness(loss_factory()), TrainingHarness(loss_factory())
    kwargs = {'advantages': [1., -.5, .7, -1.2, .4, -.3, .9, -.8][:count],
              'old_logps': [[0.] * len(row['labels']) for row in rows]}
    # CISPO weights micro objectives by valid tokens, matching the whole-batch
    # token mean. GRPO/BNPO retain their existing accumulation definitions.
    stride = count if isinstance(baseline.optimizer_group['default'].loss_instance, CISPOLoss) else 2
    for lo in range(0, count, stride):
        baseline.run(rows[lo:lo + stride], **{key: value[lo:lo + stride] for key, value in kwargs.items()})
    internal.run(rows, micro_batch_size=2, **kwargs)
    torch.testing.assert_close(internal.normalized_gradient(), baseline.normalized_gradient())
    assert internal.optimizer_group['default'].cur_step == 1
    assert internal.calls == [(list(range(lo, lo + 2)), lo + 2 == count) for lo in range(0, count, 2)]


def test_partitioned_gradients_do_not_enter_the_unsupported_zero_no_sync():
    model = TrainingHarness(GRPOLoss())
    model.model.zero_optimization_partition_gradients = lambda: True

    def unsupported():
        raise AssertionError('ZeRO-2/3 must not enter no_sync')

    model.model.no_sync = unsupported
    rows = data()[:4]
    model.run(rows, micro_batch_size=2, advantages=[1.] * 4,
              old_logps=[[0.] * len(row['labels']) for row in rows])
    assert model.weight.grad is not None


def test_global_bnpo_normalizer_counts_only_completion_tokens():
    inputs = [{'labels': [-100, 1, 2, 3], 'completion_mask': [0, 1, 0, 1]}]
    assert BNPOLoss(token_mean_scope='global').batch_normalizer(inputs) == 2
    assert BNPOLoss(token_mean_scope='micro').batch_normalizer(inputs) is None


@pytest.mark.parametrize('algorithm', ['ffd', 'kk'])
@pytest.mark.parametrize('factory', [GRPOLoss, CISPOLoss, lambda: BNPOLoss(token_mean_scope='global')])
def test_dynamic_update_matches_the_whole_batch_without_extra_return_metadata(algorithm, factory):
    rows = data()
    kwargs = {'advantages': [1., -.5, .7, -1.2, .4, -.3, .9, -.8],
              'old_logps': [[0.] * len(row['labels']) for row in rows]}
    baseline, dynamic = TrainingHarness(factory()), TrainingHarness(factory())
    baseline.run(rows, **kwargs)
    dynamic.run(rows, micro_batch_size=4, dynamic_batching=True,
                max_tokens_per_micro_batch=10, packing_algorithm=algorithm, **kwargs)
    assert len(dynamic.calls) > 2
    assert sorted(index for ids, _ in dynamic.calls for index in ids) == list(range(8))
    assert [sync for _, sync in dynamic.calls] == [False] * (len(dynamic.calls) - 1) + [True]
    assert dynamic.optimizer_group['default'].cur_step == 1
    torch.testing.assert_close(dynamic.normalized_gradient(), baseline.normalized_gradient())


def test_chord_finalization_keeps_the_original_return_format(monkeypatch):
    from twinkle import Platform
    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    model = TrainingHarness(GRPOLoss())
    rows = data()[:4]
    kwargs = {'advantages': [1.] * 4, 'old_logps': [[0.] * len(row['labels']) for row in rows]}
    first = model.run(rows, micro_batch_size=2, finalize_gradients=False, **kwargs)
    last = model.run(rows, micro_batch_size=2, finalize_gradients=True, **kwargs)
    assert set(first) == {'loss', 'logps'}
    assert set(last) == {'loss'}


@pytest.mark.parametrize('loss_factory', [GRPOLoss, CISPOLoss,
                                        lambda: BNPOLoss(token_mean_scope='global')])
def test_chord_normalizes_both_objectives_once(monkeypatch, loss_factory):
    from twinkle import Platform
    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    rows = data()[:4]
    sft_rows = data()[:6]
    model = TrainingHarness(loss_factory())
    group = model.optimizer_group['default']
    group.loss_registry = {'grpo': loss_factory(), 'sft': CrossEntropyLoss(reduction='sum')}
    kwargs = {'advantages': [1., -.5, .7, -1.2],
              'old_logps': [[0.] * len(row['labels']) for row in rows]}
    model.run(rows, micro_batch_size=2,
              loss_key='grpo', loss_weight=.7, finalize_gradients=False, **kwargs)
    assert group.cur_step == 0
    model.run(sft_rows, micro_batch_size=3,
              loss_key='sft', loss_weight=.3, finalize_gradients=True)
    assert group.cur_step == 1
    oracle, terms = TrainingHarness(GRPOLoss()), []
    for lo in (0, 2):
        output = oracle.forward(inputs=rows[lo:lo + 2])
        inputs = oracle.optimizer_group['default'].train_status.inputs
        terms.append(loss_factory()(inputs, output, **{k: v[lo:lo + 2] for k, v in kwargs.items()})['loss'])
    output = oracle.forward(inputs=sft_rows)
    inputs = oracle.optimizer_group['default'].train_status.inputs
    sft = CrossEntropyLoss(reduction='mean')(inputs, output)['loss']
    if isinstance(group.loss_registry['grpo'], CISPOLoss):
        output = oracle.forward(inputs=rows)
        inputs = oracle.optimizer_group['default'].train_status.inputs
        rl_loss = loss_factory()(inputs, output, **kwargs)['loss']
    elif getattr(loss_factory(), 'reduction', 'mean') == 'sum':
        rl_loss = torch.stack(terms).sum() / sum(len(row['labels']) for row in rows)
    else:
        rl_loss = torch.stack(terms).mean()
    (.7 * rl_loss + .3 * sft).backward()
    torch.testing.assert_close(model.normalized_gradient(), oracle.weight.grad)


class ScalarPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.12, dtype=torch.float64))

    def forward(self, coefficients):
        return self.weight * coefficients


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        class DistributedHarness(TrainingHarness):
            def token_logps(self, coefficients):
                return self.model(coefficients)

        for loss_factory in (GRPOLoss, CISPOLoss):
            model = DistributedHarness(loss_factory())
            model.model = DistributedDataParallel(ScalarPolicy())
            model.weight = model.model.module.weight
            model.device_mesh = SimpleNamespace(_get_dp_fsdp_world_size=lambda: 2)
            group = model.optimizer_group['default']
            group._dp_group = dist.group.WORLD
            reductions = {'count': 0}

            def hook(state, bucket):
                state['count'] += 1
                return allreduce_hook(dist.group.WORLD, bucket)

            model.model.register_comm_hook(reductions, hook)
            rows = data() * 2
            advantages = [1., -.5, .7, -1.2, .4, -.3, .9, -.8] * 2
            local = rows[rank * 8:(rank + 1) * 8]
            model.run(local, micro_batch_size=2,
                      advantages=advantages[rank * 8:(rank + 1) * 8],
                      old_logps=[[0.] * len(row['labels']) for row in local])
            normalizer = torch.tensor(group.train_status.num_tokens, dtype=torch.float64)
            dist.all_reduce(normalizer)
            oracle = TrainingHarness(loss_factory())
            stride = len(local) if loss_factory is CISPOLoss else 2
            for start in range(0, len(rows), stride):
                block = rows[start:start + stride]
                oracle.run(block, advantages=advantages[start:start + stride],
                           old_logps=[[0.] * len(row['labels']) for row in block])
            torch.testing.assert_close(model.weight.grad / normalizer, oracle.normalized_gradient())
            assert reductions['count'] == 1
    finally:
        dist.destroy_process_group()


def test_two_dp_ranks_synchronize_once_and_match_the_explicit_gradient(tmp_path):
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path / 'gloo'),), nprocs=2, join=True)
