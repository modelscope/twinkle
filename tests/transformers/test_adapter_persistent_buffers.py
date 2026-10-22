# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from twinkle.model.transformers.transformers import TransformersModel


def test_adapter_checkpoint_keeps_only_selected_persistent_buffers(tmp_path):
    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    base = LlamaForCausalLM(config)
    base.head = nn.Linear(8, 2)
    base.head.register_buffer('temperature', torch.tensor([2.]))
    base.head.register_buffer('token_ids', torch.tensor([3, 4]))
    base.head.register_buffer('scratch', torch.tensor([99.]), persistent=False)
    base.register_buffer('backbone_buffer', torch.ones(128))
    model = get_peft_model(base, LoraConfig(r=2, target_modules=['q_proj'],
                           modules_to_save=['head']), adapter_name='decision')
    model.add_adapter('other', LoraConfig(r=2, target_modules=['q_proj'], modules_to_save=['head']))
    params = {k: v.detach().cpu() for k, v in model.named_parameters()
              if '.decision.' in k and ('lora_' in k or '.modules_to_save.' in k)}
    strategy = SimpleNamespace(unwrap_model=lambda m: m, get_adapter_state_dict=lambda m, a: dict(params))
    wrapper = SimpleNamespace(strategy=strategy, model=model)
    saved = TransformersModel._get_adapter_state_dict_for_save(wrapper, 'decision')
    assert saved['base_model.model.head.temperature'].item() == 2
    assert saved['base_model.model.head.token_ids'].tolist() == [3, 4]
    assert not any('scratch' in k or 'backbone_buffer' in k or '.other.' in k for k in saved)
    from safetensors.torch import save_file, load_file
    save_file(saved, tmp_path / 'adapter.safetensors')
    saved = load_file(tmp_path / 'adapter.safetensors')
    model.base_model.model.head.modules_to_save['decision'].temperature.zero_()
    set_peft_model_state_dict(model, saved, adapter_name='decision')
    assert model.base_model.model.head.modules_to_save['decision'].temperature.item() == 2
