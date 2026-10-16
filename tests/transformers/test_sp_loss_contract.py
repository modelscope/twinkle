"""Loss-declared output gathering for Sequence Parallel."""
import pytest
import importlib
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
from types import SimpleNamespace

from twinkle.data_format import LossOutput
from twinkle.loss import Loss, PPOValueLoss
from twinkle.model.transformers.strategy.sequence_parallel import DataKeySpec, SequenceParallelStrategy


def _strategy():
    strategy = SequenceParallelStrategy(sp_config={'ulysses_size': 2})
    ctx = strategy._ctx
    ctx.world_size = ctx.sp_world_size = 2
    ctx.rp_world_size = 1
    ctx.extra_kwargs['position_ids'] = torch.arange(5).unsqueeze(0)
    return strategy


def _mock_gather(monkeypatch):
    from twinkle.model.transformers.strategy.sequence_parallel import GatherLoss

    monkeypatch.setattr(GatherLoss, 'apply', lambda ctx, value, labels, dim, positions:
                        (torch.cat((value, value), dim=dim), None))


def test_loss_declared_logits_are_gathered_even_when_not_default(monkeypatch):
    strategy = _strategy()
    _mock_gather(monkeypatch)
    local_values = torch.tensor([[[1.0], [2.0], [3.0]]])

    inputs, outputs = strategy.gather_loss_tensors({}, {'logits': local_values}, sp_gather_keys=('logits',))

    assert inputs == {}
    torch.testing.assert_close(outputs['logits'], torch.tensor([[[1.0], [2.0], [3.0], [1.0], [2.0]]]))


def test_custom_loss_declares_and_consumes_a_registered_output_key(monkeypatch):
    class ToyValueLoss(Loss):
        require_logps = False
        sp_gather_keys = ('values',)

        def __call__(self, inputs, outputs, **kwargs):
            values = outputs['values']
            return LossOutput(loss=values.square().mean(), num_tokens=values.numel())

    strategy = _strategy()
    strategy.register_output_key(
        DataKeySpec('values', split_dim=1, side='output', stage='output', need_split=False, autograd=True))
    _mock_gather(monkeypatch)
    _, outputs = strategy.gather_loss_tensors({}, {'values': torch.tensor([[1.0, 2.0, 3.0]])},
                                             sp_gather_keys=ToyValueLoss.sp_gather_keys)

    assert outputs['values'].shape == (1, 5)
    result = ToyValueLoss()({}, outputs)
    torch.testing.assert_close(result['loss'], torch.tensor((1.0 + 4.0 + 9.0 + 1.0 + 4.0) / 5))


def test_loss_can_declare_custom_sequence_inputs_and_outputs(monkeypatch):
    class TokenAlignmentLoss(Loss):
        require_logps = False
        sp_input_key_specs = (DataKeySpec('token_targets', split_dim=1, pad_value=-1,
                                          need_gather=True, autograd=True, owner='loss'),)
        sp_output_key_specs = (DataKeySpec('token_scores', split_dim=1, side='output', stage='output',
                                           need_split=False, need_gather=True, autograd=True, owner='loss'),)

        def __call__(self, inputs, outputs, **kwargs):
            return LossOutput(loss=(outputs['token_scores'] - inputs['token_targets']).square().mean(),
                              num_tokens=inputs['token_targets'].numel())

    strategy = _strategy()
    strategy.register_loss_key_specs(TokenAlignmentLoss())
    _mock_gather(monkeypatch)
    inputs, outputs = strategy.gather_loss_tensors(
        {'token_targets': torch.tensor([[1.0, 2.0, 3.0]])},
        {'token_scores': torch.tensor([[1.0, 2.0, 3.0]])})

    assert strategy._ctx.input_key_registry.get('token_targets') is not None
    assert strategy._ctx.output_key_registry.get('token_scores') is not None
    torch.testing.assert_close(inputs['token_targets'], torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0]]))
    torch.testing.assert_close(outputs['token_scores'], torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0]]))
    torch.testing.assert_close(TokenAlignmentLoss()(inputs, outputs)['loss'], torch.tensor(0.0))


def test_declared_but_unregistered_or_missing_output_fails_loudly():
    strategy = _strategy()
    with pytest.raises(KeyError, match='unregistered output key'):
        strategy.gather_loss_tensors({}, {'values': torch.ones(1, 3)}, sp_gather_keys=('values',))
    with pytest.raises(KeyError, match='model output is missing'):
        strategy.gather_loss_tensors({}, {}, sp_gather_keys=('logits',))


def test_ppo_value_loss_declares_per_token_value_collection():
    assert PPOValueLoss.sp_gather_keys == ('values',)
    assert PPOValueLoss.sp_output_key_specs[0].name == 'values'


class _ToyValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, _attn_implementation='sdpa', model_type='toy_value')
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, input_ids, position_ids, **kwargs):
        return {'logits': input_ids.to(self.scale.dtype) * self.scale}


def _ppo_value_worker(rank, port):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.set_num_threads(1)
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        from twinkle import DeviceMesh
        from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelStrategy
        from twinkle.processor import InputProcessor

        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        positions = torch.arange(6).unsqueeze(0)
        labels = torch.tensor([[-100, -100, 3, 4, 5, 6]])
        returns = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        old_values = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        loss_fn = PPOValueLoss(cliprange_value=0.2, vf_coef=0.5)

        baseline = _ToyValueModel()
        base_logits = baseline(ids, positions)['logits']
        base_loss = loss_fn({'labels': labels}, {'values': base_logits},
                            returns=returns, old_values=old_values)['loss']
        base_loss.backward()
        base_grad = baseline.scale.grad.clone()

        model = _ToyValueModel()
        strategy = SequenceParallelStrategy(
            DeviceMesh.from_sizes(dp_size=2, ulysses_size=2, device_type='cpu'),
            sp_config={'ulysses_size': 2},
            model=model,
        )
        strategy._get_tokenizer = lambda: SimpleNamespace(pad_token_id=0)
        assert strategy.initialize()
        strategy.register_loss_key_specs(loss_fn)
        prepared = strategy.preprocess_inputs({'input_ids': ids, 'position_ids': positions, 'labels': labels})
        local_labels = prepared.pop('labels')
        local_outputs = model(**{key: prepared[key] for key in ('input_ids', 'position_ids')})
        local_outputs['values'] = local_outputs.pop('logits')
        loss_inputs, loss_outputs = InputProcessor().postprocess_tensor_sp(
            {'labels': local_labels},
            local_outputs,
            sp_strategy=strategy,
            sp_gather_keys=loss_fn.sp_gather_keys,
        )
        parallel_loss = loss_fn(loss_inputs, loss_outputs, returns=returns, old_values=old_values)['loss']
        torch.testing.assert_close(parallel_loss, base_loss, atol=1e-7, rtol=1e-6)
        parallel_loss.backward()
        dist.all_reduce(model.scale.grad)
        model.scale.grad.div_(2)
        torch.testing.assert_close(model.scale.grad, base_grad, atol=1e-7, rtol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_ppo_value_loss_sp_matches_full_sequence_loss_and_gradient():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_ppo_value_worker, args=(port,), nprocs=2, join=True)


class _NoopTrainStrategy:
    """Keep the CPU E2E focused on TransformersModel and SP loss handling."""

    def capture_pre_ep_state_if_needed(self, model, *, enable_ep):
        return None

    def wrap_model(self, model, *args):
        return (model, *args) if args else model


def _ppo_training_pipeline_worker(rank, port):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.set_num_threads(1)
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
        from twinkle import DeviceMesh, Platform
        from twinkle.model.transformers.transformers import TransformersModel
        from twinkle.processor import InputProcessor

        Platform.get_local_device = staticmethod(lambda: torch.device('cpu'))
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                             attention_dropout=0.0, pad_token_id=0, use_cache=False)
        # Exercise a real Transformers attention implementation available in
        # this environment; this E2E does not require flash-attn.
        config._attn_implementation = 'sdpa'
        torch.manual_seed(321)
        baseline = LlamaForCausalLM(config).train()

        mesh = DeviceMesh.from_sizes(dp_size=2, ulysses_size=2, device_type='cpu')
        model_module = importlib.import_module('twinkle.model.transformers.transformers')
        original_template = model_module.Template
        model_module.Template = lambda *_args, **_kwargs: SimpleNamespace(is_mm=False)
        try:
            training_model = TransformersModel(
                model_cls=AutoModelForCausalLM,
                config=config,
                device_mesh=mesh,
                mixed_precision='no',
            )
        finally:
            model_module.Template = original_template
        training_model.strategy = _NoopTrainStrategy()
        training_model.device_mesh = mesh
        training_model._enable_sp = True
        training_model.model.load_state_dict(baseline.state_dict())
        training_model.model.gradient_checkpointing_disable()
        training_model.model.train()

        # The value head's [B, T] values travel through the standard `logits`
        # output slot; retain a differentiable projection of real LM outputs.
        model_forward = training_model.model.forward

        def forward_per_token_values(*args, **kwargs):
            outputs = model_forward(*args, **kwargs)
            outputs['logits'] = outputs.logits[..., 0]
            return outputs

        training_model.model.forward = forward_per_token_values
        training_model.sp_strategy = SequenceParallelStrategy(mesh, model=training_model.model)
        training_model.sp_strategy._get_tokenizer = lambda: SimpleNamespace(pad_token_id=0)
        training_model.set_loss(PPOValueLoss(cliprange_value=0.2, vf_coef=0.5))
        training_model.set_processor(InputProcessor)
        training_model._ensure_sp_strategy()

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]])
        positions = torch.arange(ids.shape[1]).expand_as(ids)
        labels = torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8],
                               [-100, -100, -100, 5, 6, 7, 8, 9]])
        returns = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5], [0.4, 0.8, 1.2, 1.6, 2.0]])
        old_values = torch.tensor([[0.2, 0.4, 0.6, 0.8, 1.0], [0.1, 0.3, 0.5, 0.7, 0.9]])
        batch = [
            {'input_ids': ids[i], 'position_ids': positions[i], 'labels': labels[i]}
            for i in range(ids.shape[0])
        ]

        loss_fn = PPOValueLoss(cliprange_value=0.2, vf_coef=0.5)
        baseline_values = baseline(input_ids=ids, position_ids=positions, use_cache=False).logits[..., 0]
        baseline_loss = loss_fn({'labels': labels}, {'values': baseline_values},
                                returns=returns, old_values=old_values)['loss']
        baseline_loss.backward()
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=1e-3)
        baseline_optimizer.step()

        training_model.forward_backward(inputs=batch, returns=returns, old_values=old_values)
        optimizer_group = training_model.optimizer_group[training_model._get_default_group()]
        assert optimizer_group.train_status.inputs['labels'].shape == labels.shape
        assert optimizer_group.train_status.outputs['values'].shape == labels.shape
        optimizer_group.optimizer = torch.optim.SGD(training_model.model.parameters(), lr=1e-3)
        for parameter in training_model.model.parameters():
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
        training_model.step()

        for (name, actual), (_, expected) in zip(training_model.model.named_parameters(), baseline.named_parameters()):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_ppo_value_loss_through_transformers_sdpa_sp_training_pipeline():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_ppo_training_pipeline_worker, args=(port,), nprocs=2, join=True)
