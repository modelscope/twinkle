"""Tests for model-family SP adapter registration and behavior."""
import pytest
import torch
from types import SimpleNamespace

import twinkle.model.transformers.strategy.sequence_parallel as sp
from twinkle.model.transformers.strategy.sequence_parallel import adapters as sp_adapters


@pytest.mark.parametrize(('model_type', 'expected'), [
    ('qwen3_vl', {'qwen3_vl'}),
    ('qwen3_5', {'qwen3_5_linear_attention'}),
    ('qwen3_5_moe', {'qwen3_5_linear_attention', 'moe_aux'}),
    ('llama', set()),
])
def test_builtin_model_adapters_are_selected_by_model_family(model_type, expected):
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type=model_type)
    assert {adapter.name for adapter in sp_adapters.adapters_for_model(model)} == expected


def test_model_adapter_registration_and_declared_key_specs(monkeypatch):
    class CustomAdapter(sp_adapters.ModelSPAdapter):
        name = 'test_custom_sp_adapter'

        def matches(self, model):
            return getattr(getattr(model, 'config', None), 'model_type', None) == 'custom_sp_test'

        def input_key_specs(self):
            return (sp.DataKeySpec('custom_features', pad_value=-3),)

    monkeypatch.setattr(sp_adapters, '_ADAPTERS', list(sp_adapters._ADAPTERS))
    adapter = sp_adapters.register_model_sp_adapter(CustomAdapter)
    assert adapter is sp_adapters.register_model_sp_adapter(CustomAdapter())
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type='custom_sp_test')
    context = sp.SequenceParallel()
    context._configure_model_adapters(model, model)
    assert context.model_adapters == (adapter,)
    assert context.input_key_registry.get('custom_features') == sp.DataKeySpec('custom_features', pad_value=-3)


def test_model_adapters_declare_mrope_and_keep_qwen35_ring_limit(monkeypatch):
    vl_adapter = sp_adapters.Qwen3VLSPAdapter()
    assert vl_adapter.position_ids_mode == 'mrope'
    ctx = sp.SequenceParallel()
    ctx.position_ids_mode = vl_adapter.position_ids_mode
    real_positions = torch.arange(4).unsqueeze(0)
    assert ctx._extract_real_position_ids(real_positions) is real_positions
    mrope_positions = real_positions.unsqueeze(0).expand(3, -1, -1)
    torch.testing.assert_close(ctx._extract_real_position_ids(mrope_positions), real_positions)
    ctx.rp_world_size = 2
    with pytest.raises(NotImplementedError, match='rp_world_size > 1'):
        sp_adapters.Qwen35LinearAttentionSPAdapter().configure(ctx, torch.nn.Linear(2, 2))


@pytest.mark.parametrize(('adapter', 'patch_name'), [
    (sp_adapters.Qwen3VLSPAdapter(), 'qwen3_vl_deepstack'),
    (sp_adapters.Qwen35LinearAttentionSPAdapter(), 'qwen35_gdn'),
])
def test_builtin_adapters_install_model_specific_patches(monkeypatch, adapter, patch_name):
    calls = []
    monkeypatch.setattr(sp_adapters, 'apply_patch', lambda model, patch, **kwargs: calls.append(
        (model, patch, kwargs['sequence_parallel'])))
    model = torch.nn.Linear(2, 2)
    ctx = sp.SequenceParallel()
    ctx.rp_world_size = 1
    adapter.configure(ctx, model)
    assert len(calls) == 1
    assert calls[0][0] is (None if patch_name == 'qwen35_gdn' else model)
    assert calls[0][2] is ctx


def test_moe_adapter_registers_router_logits_gather_hook(monkeypatch):
    hooks = []

    class MoEConfig:
        model_type = 'example_moe'

    class Model:
        config = MoEConfig()

        def register_forward_hook(self, hook, with_kwargs=False):
            hooks.append((hook, with_kwargs))

    class Output(dict):
        router_logits = (torch.arange(6, dtype=torch.float32).reshape(2, 3),)

    ctx = SimpleNamespace(real_position_ids=torch.tensor([[0, 1, 2]]))
    monkeypatch.setattr(sp_adapters.GatherLoss, 'apply',
                        lambda context, value, labels, dim, positions: (torch.cat((value, value), dim), None))
    model = Model()
    adapter = sp_adapters.MoEAuxSPAdapter()
    assert adapter.matches(model)
    adapter.configure(ctx, model)
    assert hooks and hooks[0][1]
    output = hooks[0][0](model, (), {'attention_mask': torch.ones(1, 3)}, Output())
    assert len(output['router_logits']) == 1
    assert output['router_logits'][0].shape == (3, 3)
