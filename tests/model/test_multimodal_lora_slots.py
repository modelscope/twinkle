# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora import LoraLayer

from twinkle.model.multi_lora import MultiLora
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
from twinkle.model.transformers.transformers import TransformersModel


class _VisionLanguageModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.language = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.visual = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.aligner = torch.nn.Sequential(torch.nn.Linear(4, 4))

    def forward(self, x):
        return self.language(x) + self.aligner(self.visual(x))


@pytest.mark.parametrize('vision_only', [False, True])
def test_multimodal_tenant_updates_only_requested_towers(vision_only):
    manager = MultiLora(max_loras=2, max_r=2)
    module = manager.patch(_VisionLanguageModel(), lora_config=LoraConfig(r=2, target_modules='all-linear'))
    manager.save_initial_weights()
    manager.reset_adapter_status()
    host = SimpleNamespace(
        model=module, model_arch={'language_model': ['language'], 'vision_tower': ['visual'], 'aligner': ['aligner']},
        multi_adapter=manager, strategy=SimpleNamespace(unwrap_model=lambda model: model),
        _copy_lora_config=MultiLoraTransformersModel._copy_lora_config,
        _register_adapter=lambda name, config, **kwargs: manager.acquire_lora(name, config),
    )
    host._expand_multimodal_lora_target = lambda *args, **kwargs: TransformersModel._expand_multimodal_lora_target(
        host, *args, **kwargs)
    config = LoraConfig(r=2, lora_alpha=2, target_modules='all-linear', lora_dropout=0.0)
    original = deepcopy(config.to_dict())
    inspect.unwrap(MultiLoraTransformersModel.add_adapter_to_model)(
        host, 'tenant', config, freeze_llm=vision_only, freeze_vit=not vision_only)
    assert config.to_dict() == original
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    with manager.adapter('tenant'):
        optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=0.1)
        module(torch.ones(2, 4)).square().sum().backward()
        optimizer.step()
    changed = [name for name, value in module.named_parameters() if not torch.equal(value, before[name])]
    assert changed
    expected = '.visual.' if vision_only else '.language.'
    assert all(expected in name and '.lora_' in name for name in changed), changed


def test_megatron_slots_share_each_chunks_original_target_modules(monkeypatch):
    from twinkle.model.megatron.megatron import MegatronModel

    class Chunk(torch.nn.Module):

        def __init__(self, name):
            super().__init__()
            self.add_module(name, torch.nn.Linear(4, 4))
            self.visual = torch.nn.Linear(4, 4)

    chunks = [Chunk('first'), Chunk('second')]
    calls = []

    def targets(model, requested):
        # Expansion must inspect each pristine chunk, before PEFT changes the module tree.
        assert not any(isinstance(layer, LoraLayer) for layer in model.modules())
        assert requested == ['all-linear']
        calls.append(model)
        return ['first' if model is chunks[0] else 'second']

    monkeypatch.setattr(MegatronModel, 'get_target_modules', staticmethod(targets))
    manager = MultiLora(max_loras=2, max_r=2)
    wrapped = manager.patch(chunks, lora_config=LoraConfig(r=2, target_modules='all-linear'))
    assert calls == chunks
    for chunk in wrapped:
        adapted = [(name, layer) for name, layer in chunk.named_modules() if isinstance(layer, LoraLayer)]
        assert len(adapted) == 1 and '.visual' not in adapted[0][0]
        assert set(adapted[0][1].lora_A) == {'lora_0', 'lora_1'}
