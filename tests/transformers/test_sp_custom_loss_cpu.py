"""End-to-end contract test for a user-defined sequence loss."""
import importlib
import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from twinkle import DeviceMesh, Platform
from twinkle.data_format import LossOutput
from twinkle.loss import Loss
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.model.transformers.strategy.sequence_parallel import DataKeySpec, SequenceParallelStrategy


class TokenRegressionLoss(Loss):
    """Example plugin loss with custom sequence input and output keys."""

    require_logps = False
    input_key_specs = (DataKeySpec('token_targets', split_dim=1, pad_value=0, owner='loss'),)
    output_key_specs = (DataKeySpec('token_scores', split_dim=1, side='output', stage='output',
                                       need_split=False, need_gather=True, autograd=True, owner='loss'),)

    def __call__(self, inputs, outputs, **kwargs):
        scores = outputs['token_scores']
        targets = inputs['token_targets'].to(dtype=scores.dtype)
        loss = (scores - targets).square().mean()
        return LossOutput(loss=loss, num_tokens=loss.new_tensor(targets.numel()))


class _ToyTokenRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_key_value_heads=2, num_attention_heads=2,
                                      _attn_implementation='sdpa', model_type='toy_token_regressor')
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, input_ids, position_ids, attention_mask=None, **kwargs):
        return {'token_scores': input_ids.to(self.scale.dtype) * self.scale}


class _NoopTrainStrategy:
    """Keep this CPU E2E focused on Twinkle's model/loss/optimizer flow."""

    def capture_pre_ep_state_if_needed(self, model, *, enable_ep):
        return None

    def unwrap_model(self, model):
        return model

    def wrap_model(self, model, *args):
        return (model, *args) if args else model


def _worker(rank, port):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.set_num_threads(1)
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]])
        positions = torch.arange(ids.shape[1]).expand_as(ids)
        targets = ids.float() * 0.4 + 0.25
        loss_fn = TokenRegressionLoss()

        baseline = _ToyTokenRegressor()
        full_outputs = baseline(input_ids=ids, position_ids=positions)
        full_loss = loss_fn({'token_targets': targets}, full_outputs)['loss']
        full_loss.backward()
        expected_grad = baseline.scale.grad.clone()

        model = _ToyTokenRegressor()
        strategy = SequenceParallelStrategy(
            DeviceMesh.from_sizes(dp_size=2, ulysses_size=2, device_type='cpu'), model=model)
        strategy._get_tokenizer = lambda: SimpleNamespace(pad_token_id=0)
        strategy.register_loss_key_specs(loss_fn)
        assert strategy.initialize()

        prepared = strategy.preprocess_inputs({'input_ids': ids, 'position_ids': positions,
                                               'token_targets': targets})
        local_targets = prepared.pop('token_targets')
        local_outputs = model(input_ids=prepared['input_ids'], position_ids=prepared['position_ids'])
        loss_inputs, loss_outputs = strategy.gather_loss_tensors({'token_targets': local_targets}, local_outputs)
        sp_loss = loss_fn(loss_inputs, loss_outputs)['loss']
        torch.testing.assert_close(sp_loss, full_loss, atol=1e-7, rtol=1e-6)
        sp_loss.backward()
        dist.all_reduce(model.scale.grad)
        model.scale.grad.div_(2)
        torch.testing.assert_close(model.scale.grad, expected_grad, atol=1e-7, rtol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_custom_token_regression_loss_matches_full_sequence_loss_and_gradient():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)


def _e2e_worker(rank, port, sp_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = '2'
    torch.set_num_threads(1)
    Platform.get_local_device = staticmethod(lambda: torch.device('cpu'))
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type('cpu')
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                             attention_dropout=0.0, pad_token_id=0, use_cache=False)
        config._attn_implementation = 'sdpa'
        torch.manual_seed(1234)
        baseline = LlamaForCausalLM(config).train()

        mesh = DeviceMesh.from_sizes(dp_size=2, ulysses_size=sp_size, device_type='cpu')
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
        # This local CPU harness supplies worker mesh state explicitly.
        training_model.device_mesh = mesh
        training_model._enable_sp = sp_size > 1
        model_forward = training_model.model.forward

        def forward_with_token_scores(*args, **kwargs):
            outputs = model_forward(*args, **kwargs)
            outputs['token_scores'] = outputs.logits[..., 0]
            return outputs

        training_model.model.forward = forward_with_token_scores
        training_model.model.load_state_dict(baseline.state_dict())
        training_model.model.gradient_checkpointing_disable()
        training_model.model.train()
        strategy = SequenceParallelStrategy(mesh, model=training_model.model) if sp_size > 1 else None
        if strategy is not None:
            strategy._get_tokenizer = lambda: SimpleNamespace(pad_token_id=0)
        training_model.sp_strategy = strategy

        optimizer_group = training_model.optimizer_group[training_model._get_default_group()]
        loss_fn = TokenRegressionLoss()
        optimizer_group.loss_instance = loss_fn
        assert optimizer_group.loss_instance.input_key_specs
        training_model._ensure_sp_strategy()
        if strategy is not None:
            assert 'token_targets' in {spec.name for spec in strategy.input_key_specs}

        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]])
        positions = torch.arange(ids.shape[1]).expand_as(ids)
        targets = ids.float() * 0.4 + 0.25
        labels = torch.full_like(ids, -100)
        batch = [
            {'input_ids': ids[row], 'position_ids': positions[row], 'labels': labels[row],
             'token_targets': targets[row]}
            for row in range(ids.shape[0])
        ]

        # Baseline uses the same model weights and the same optimizer update,
        # but computes the loss on the unsharded full sequence.
        base_optimizer = torch.optim.SGD(baseline.parameters(), lr=1e-3)
        baseline_output = baseline(input_ids=ids, position_ids=positions, use_cache=False)
        baseline_output['token_scores'] = baseline_output.logits[..., 0]
        baseline_loss = loss_fn({'token_targets': targets}, baseline_output)['loss']
        baseline_loss.backward()
        base_optimizer.step()

        # End-to-end Twinkle path: processor -> SP setup/hook -> forward ->
        # loss registration/collection -> calculate_loss -> backward -> step.
        forward_outputs = training_model.forward(inputs=batch)
        assert 'token_targets' in optimizer_group.train_status.inputs, optimizer_group.train_status.inputs.keys()
        sp_loss = training_model.calculate_loss()
        torch.testing.assert_close(torch.tensor(sp_loss), baseline_loss.detach(), atol=1e-6, rtol=1e-5)
        training_model.backward()
        optimizer = torch.optim.SGD(training_model.model.parameters(), lr=1e-3)
        for parameter in training_model.model.parameters():
            assert parameter.grad is not None
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
        optimizer_group.optimizer = optimizer
        training_model.step()
        assert 'token_scores' in forward_outputs
        for (name, actual), (_, expected) in zip(training_model.model.named_parameters(), baseline.named_parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize('sp_size', [1, 2])
def test_custom_loss_through_transformers_training_pipeline_and_optimizer_step(sp_size):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_e2e_worker, args=(port, sp_size), nprocs=2, join=True)
