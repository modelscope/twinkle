# Copyright (c) ModelScope Contributors. All rights reserved.
"""Attached-head checkpoints: the bound tenant's head weights ride task_head.safetensors.

The LoRA base writes through the peft path and never sees the head, so a checkpoint
without the head file silently loses everything the head learned. These tests pin the
full roundtrip: save writes the slot-independent ``task_head.`` keys, a re-bound tenant
(or a cold worker) restores them in place, legacy checkpoints without the file stay
loadable, and a head file that lands on a slot-less host fails loudly.
"""
import copy
from types import SimpleNamespace

import pytest
import torch


def _build(tmp_path, monkeypatch, task):
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from tests.server.integration.cs_tasks.task_models import (encoded_batch, prepare_models,
                                                               tiny_backbone)
    from twinkle import Platform, initialize
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    # paths['embedding'] is the plain causal backbone: the attached-head deployment.
    paths = prepare_models(tiny_backbone(tmp_path / 'source'), tmp_path / 'bases')
    targets = ['q_proj', 'v_proj']
    batch = encoded_batch('seq_cls', AutoTokenizer.from_pretrained(paths['embedding']))
    if task == 'value':
        # Token-level float labels on the response half, plus aligned RL inputs.
        width = max(len(row['input_ids']) for row in batch)
        for row in batch:
            half = max(1, len(row['input_ids']) // 2)
            row['labels'] = [-100.0] * half + [1.0] * (len(row['input_ids']) - half)
        old_values = [[0.0] * width for _ in batch]
        returns = [[1.0] * width for _ in batch]
    else:
        old_values = returns = None

    def build():
        return MultiLoraTransformersModel(
            model_id=paths['embedding'], mixed_precision='no', gradient_checkpointing=False,
            max_loras=2, max_r=8, target_modules=targets,
            task=task, num_labels=3 if task == 'seq_cls' else 1)

    def add(model, name):
        model.add_adapter_to_model(name, LoraConfig(r=4, lora_alpha=8, target_modules=targets))
        model.set_processor('InputProcessor', padding_side='right', adapter_name=name)
        if task == 'seq_cls':
            model.set_loss('SeqClsLoss', problem_type='single_label_classification',
                           num_labels=3, adapter_name=name)
        elif task == 'reranker':
            model.set_loss('PointwiseRerankerLoss', adapter_name=name)
        else:
            model.set_loss('PPOValueLoss', adapter_name=name)
        model.set_optimizer('Adam', lr=1e-3, adapter_name=name)

    def step(model, name):
        model.forward_backward(inputs=copy.deepcopy(batch), task=task, adapter_name=name,
                               **({} if old_values is None else
                                  dict(old_values=old_values, returns=returns)))
        model.clip_grad_and_step(adapter_name=name)

    def read(model, name):
        return model.forward_only(inputs=copy.deepcopy(batch), task=task, return_logits=True,
                                  adapter_name=name)['logits'].detach().clone()

    return build, add, step, read, paths


@pytest.mark.parametrize('task', ['seq_cls', 'reranker', 'value'])
def test_attached_head_checkpoint_roundtrip(tmp_path, monkeypatch, task):
    from safetensors.torch import load_file

    build, add, step, read, _ = _build(tmp_path, monkeypatch, task)
    model = build()
    add(model, 'a')
    initial = read(model, 'a')
    initial_head = _head_params(model, 'a')
    for _ in range(3):
        step(model, 'a')
    trained = read(model, 'a')
    assert not torch.equal(initial, trained)
    # The head itself must train: an optimizer parameter view that missed the head slot
    # would leave every "output moved" signal to the LoRA alone.
    trained_head = _head_params(model, 'a')
    assert any(not torch.equal(initial_head[k], trained_head[k]) for k in initial_head)
    optimizer_names = set()
    for group in model.optimizer_group['a'].optimizer.param_groups:
        optimizer_names.update(group.get('param_names', []))
    assert any(name.startswith('task_head_slot_') for name in optimizer_names), \
        sorted(optimizer_names)[:8]

    saved = model.save('trained', output_dir=str(tmp_path), adapter_name='a')
    head_path = f'{saved}/task_head.safetensors'
    head_state = load_file(head_path)
    assert set(head_state) == {'task_head.classifier.weight', 'task_head.classifier.bias'}
    slot = model.task_head_slots.bindings['a']
    host = model._task_head_host()
    saved_head = dict(getattr(host, f'task_head_slot_{slot}').named_parameters())
    for key, tensor in head_state.items():
        torch.testing.assert_close(tensor, saved_head[key[len('task_head.'):]], rtol=0, atol=0)

    # A fresh tenant on a released slot must restore BOTH the LoRA and the head:
    # the released slot is back at build-time weights, so the fresh tenant reads as initial.
    model.remove_adapter('a')
    add(model, 'c')
    torch.testing.assert_close(read(model, 'c'), initial, rtol=0, atol=0)
    model.load(saved, adapter_name='c')
    torch.testing.assert_close(read(model, 'c'), trained, rtol=0, atol=0)
    step(model, 'c')
    step(model, 'c')
    continued = read(model, 'c')
    model.load(saved, adapter_name='c')
    torch.testing.assert_close(read(model, 'c'), trained, rtol=0, atol=0)
    assert not torch.equal(continued, trained)

    # A cold worker reaches the same trained state from the checkpoint alone.
    cold = build()
    add(cold, 'a')
    cold.load(saved, adapter_name='a')
    torch.testing.assert_close(read(cold, 'a'), trained, rtol=0, atol=0)

    # Legacy checkpoint without the head file stays loadable: the LoRA base restores,
    # the head keeps whatever the slot currently carries (nothing to fail on).
    legacy = f'{tmp_path}/legacy'
    import shutil
    shutil.copytree(saved, legacy)
    import os
    os.remove(f'{legacy}/task_head.safetensors')
    legacy_model = build()
    add(legacy_model, 'a')
    slot = legacy_model.task_head_slots.bindings['a']
    host = legacy_model._task_head_host()
    head_before = {k: p.detach().clone() for k, p in
                   getattr(host, f'task_head_slot_{slot}').named_parameters()}
    legacy_model.load(legacy, adapter_name='a')
    assert not torch.equal(read(legacy_model, 'a'), trained)
    head_after = {k: p.detach().clone() for k, p in
                  getattr(host, f'task_head_slot_{slot}').named_parameters()}
    assert head_before.keys() == head_after.keys()
    for key in head_before:
        torch.testing.assert_close(head_after[key], head_before[key], rtol=0, atol=0)


def test_head_checkpoint_rejected_on_slotless_host(tmp_path, monkeypatch):
    build, add, _, _, paths = _build(tmp_path, monkeypatch, 'seq_cls')
    headed = build()
    add(headed, 'a')
    for _ in range(2):
        headed.forward_backward(
            inputs=_encoded(tmp_path, paths), task='seq_cls', adapter_name='a')
        headed.clip_grad_and_step(adapter_name='a')
    saved = headed.save('trained', output_dir=str(tmp_path), adapter_name='a')
    assert _head_state(saved)

    from peft import LoraConfig
    from tests.server.integration.cs_tasks.task_models import encoded_batch, prepare_models, \
        tiny_backbone
    from twinkle import Platform, initialize
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

    plain = MultiLoraTransformersModel(
        model_id=paths['embedding'], mixed_precision='no', gradient_checkpointing=False,
        max_loras=2, max_r=8, target_modules=['q_proj', 'v_proj'])
    plain.add_adapter_to_model('a', LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj']))
    with pytest.raises(ValueError, match="cannot serve the checkpoint's head task"):
        plain.load(saved, adapter_name='a')
    # The LoRA base itself restored fine; only the head had nowhere to land.
    assert plain.multi_adapter.find_lora_by_tenant('a') is not None


def _head_params(model, adapter_name):
    slot = model.task_head_slots.bindings[adapter_name]
    head = getattr(model._task_head_host(), f'task_head_slot_{slot}')
    return {k: p.detach().clone() for k, p in head.named_parameters()}


def _encoded(tmp_path, paths):
    from transformers import AutoTokenizer
    from tests.server.integration.cs_tasks.task_models import encoded_batch
    return copy.deepcopy(encoded_batch('seq_cls', AutoTokenizer.from_pretrained(paths['embedding'])))


def _head_state(saved):
    from safetensors.torch import load_file
    state = load_file(f'{saved}/task_head.safetensors')
    assert {'task_head.classifier.weight', 'task_head.classifier.bias'} <= set(state)
    return state


def test_megatron_optimizer_context_keeps_the_bound_head(tmp_path):
    """The tenant name reaches the head pattern; the physical slot name must not.

    Regression: optimizer_context used to receive the PHYSICAL LoRA slot name (the value
    multi_adapter.adapter() yields). The LoRA pattern wanted exactly that, but the head
    pattern keys on the tenant -- so the head silently missed every optimizer build and
    trained gradients that never updated."""
    import torch.nn as nn
    from twinkle.model.megatron.multi_lora_megatron import MultiLoraMegatronModel
    from twinkle.model.multi_lora import MultiLoraTaskHeadSlots

    lora_a = nn.Parameter(torch.zeros(2, 2))
    lora_b = nn.Parameter(torch.zeros(2, 2))
    head_w = nn.Parameter(torch.zeros(3, 2))
    head_b = nn.Parameter(torch.zeros(3))
    other_head_w = nn.Parameter(torch.zeros(3, 2))
    base_w = nn.Parameter(torch.zeros(2, 2))
    for parameter in (lora_a, lora_b, head_w, head_b, other_head_w, base_w):
        parameter.requires_grad_(True)

    # named_parameters as the wrapped megatron module reports them: slot-physical LoRA
    # names, slot-named heads, and the frozen base.
    module = SimpleNamespace(named_parameters=lambda prefix='', recurse=True, **kw: iter([
        ('model.0.lora_A.lora_0.weight', lora_a),
        ('model.0.lora_B.lora_0.weight', lora_b),
        ('model.0.task_head_slot_0.classifier.weight', head_w),
        ('model.0.task_head_slot_0.classifier.bias', head_b),
        ('model.0.task_head_slot_1.classifier.weight', other_head_w),
        ('model.0.self_attention.linear_qkv.weight', base_w),
    ]))
    slots = MultiLoraTaskHeadSlots(2)
    slots.bindings['tenant_a'] = 0  # the tenant binds slot 0; LoRA lives in physical lora_0

    class FakeLora:
        adapter_name = 'lora_0'

    wrapper = SimpleNamespace(
        model=[module],
        task_head_slots=slots,
        multi_adapter=SimpleNamespace(find_lora_by_tenant=lambda tenant: FakeLora()),
    )
    with MultiLoraMegatronModel.optimizer_context(wrapper, 'tenant_a'):
        seen = dict(module.named_parameters())
    assert set(seen) == {'model.0.lora_A.lora_0.weight', 'model.0.lora_B.lora_0.weight',
                         'model.0.task_head_slot_0.classifier.weight',
                         'model.0.task_head_slot_0.classifier.bias'}, sorted(seen)


def test_single_model_attached_head_trains(tmp_path, monkeypatch):
    """The single-model (non-multi-LoRA) attached head must reach the optimizer too.

    Regression: heads ride the task_heads dict on the unwrapped model; dict-held modules
    never appear in named_parameters and the LoRA pattern misses them, so the head
    computed gradients the optimizer never updated."""
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from tests.server.integration.cs_tasks.task_models import encoded_batch, prepare_models, \
        tiny_backbone
    from twinkle import Platform, initialize
    from twinkle.model.transformers.transformers import TransformersModel

    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    paths = prepare_models(tiny_backbone(tmp_path / 'source'), tmp_path / 'bases')
    batch = encoded_batch('seq_cls', AutoTokenizer.from_pretrained(paths['embedding']))
    model = TransformersModel(model_id=paths['embedding'], mixed_precision='no',
                              gradient_checkpointing=False)
    model.add_adapter_to_model('a', LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj']),
                               task='seq_cls', num_labels=3)
    model.set_processor('InputProcessor', padding_side='right', adapter_name='a')
    model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3,
                   adapter_name='a')
    model.set_optimizer('Adam', lr=1e-3, adapter_name='a')
    names = []
    for group in model.optimizer_group['a'].optimizer.param_groups:
        names += group.get('param_names', [])
    assert any(name.startswith('task_head.') for name in names), sorted(names)[:8]
    host = model.strategy.unwrap_model(model.model)
    head = host.task_heads['a']
    before = {k: p.detach().clone() for k, p in head.named_parameters()}
    for _ in range(3):
        model.forward_backward(inputs=copy.deepcopy(batch), task='seq_cls', adapter_name='a')
        model.clip_grad_and_step(adapter_name='a')
    after = {k: p.detach().clone() for k, p in head.named_parameters()}
    assert any(not torch.equal(before[k], after[k]) for k in before)
